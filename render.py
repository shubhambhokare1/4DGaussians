#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import json
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
from utils.render_utils import get_state_at_time
def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

# ---------------------------------------------------------------------------
# Temporal stability metric helpers
# ---------------------------------------------------------------------------
_MAX_GAUSSIANS_FOR_METRICS = 50_000  # subsample cap for expensive queries


@torch.no_grad()
def _query_positions_at_time(gaussians, t, idx):
    """Return deformed Gaussian centres [G, 3] for a scalar timestamp t."""
    means3D = gaussians.get_xyz[idx]
    time_t = torch.tensor(t, dtype=torch.float32).cuda().repeat(means3D.shape[0], 1)
    scales = gaussians._scaling[idx]
    rotations = gaussians._rotation[idx]
    opacity = gaussians._opacity[idx]
    shs = gaussians.get_features[idx]
    means3D_def, _, _, _, _ = gaussians._deformation(
        means3D, scales, rotations, opacity, shs, time_t
    )
    return means3D_def  # [G, 3] on CUDA


def _gaussian_subsample_idx(gaussians):
    N = gaussians._xyz.shape[0]
    if N > _MAX_GAUSSIANS_FOR_METRICS:
        return torch.randperm(N, device="cuda")[:_MAX_GAUSSIANS_FOR_METRICS]
    return torch.arange(N, device="cuda")


@torch.no_grad()
def compute_gds(gaussians, views, cam_type):
    """Gaussian Drift Score: mean frame-to-frame Gaussian displacement."""
    # Collect unique timestamps in appearance order
    timestamps = []
    seen = set()
    for v in views:
        t = v.time if cam_type != "PanopticSports" else v["time"]
        if t not in seen:
            seen.add(t)
            timestamps.append(t)

    if len(timestamps) < 2:
        return 0.0

    idx = _gaussian_subsample_idx(gaussians)
    positions = []
    for t in timestamps:
        pos = _query_positions_at_time(gaussians, t, idx)
        positions.append(pos.cpu())

    mu = torch.stack(positions, dim=0)        # [T, G, 3]
    velocity = mu[1:] - mu[:-1]              # [T-1, G, 3]
    drift = torch.norm(velocity, dim=2)       # [T-1, G]
    return drift.mean().item()


def compute_tfs(frames):
    """Temporal Flicker Score: optical-flow-warped frame MSE across consecutive frames.

    *frames* is a list of uint8 numpy arrays in HWC RGB order (0-255).
    """
    if len(frames) < 2:
        return 0.0

    flicker_scores = []
    for i in range(len(frames) - 1):
        f_t = frames[i]    # HWC uint8
        f_t1 = frames[i + 1]

        gray_t = cv2.cvtColor(f_t, cv2.COLOR_RGB2GRAY)
        gray_t1 = cv2.cvtColor(f_t1, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray_t, gray_t1, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        H, W = gray_t.shape
        map_x = (np.arange(W, dtype=np.float32)[None, :] + flow[:, :, 0]).clip(0, W - 1)
        map_y = (np.arange(H, dtype=np.float32)[:, None] + flow[:, :, 1]).clip(0, H - 1)
        warped = cv2.remap(f_t, map_x, map_y, cv2.INTER_LINEAR)

        # MSE in [0, 1] float
        diff = (f_t1.astype(np.float32) - warped.astype(np.float32)) / 255.0
        flicker_scores.append(float((diff ** 2).mean()))

    return float(np.mean(flicker_scores))


@torch.no_grad()
def compute_pois(gaussians, views, cam_type):
    """Per-Object Instability Score.

    Projects Gaussian centres into each masked camera view, looks up the
    object-label mask value, then aggregates mean motion magnitude per label.
    Returns a dict {label_str: float} or {} if no masks are available.
    """
    # Only standard cameras carry masks; skip PanopticSports dict cameras
    if cam_type == "PanopticSports":
        return {}

    masked_views = [v for v in views if getattr(v, "mask", None) is not None]
    if not masked_views:
        return {}

    # Collect unique timestamps (preserving order)
    timestamps = []
    seen = set()
    for v in views:
        t = v.time
        if t not in seen:
            seen.add(t)
            timestamps.append(t)

    if len(timestamps) < 2:
        return {}

    idx = _gaussian_subsample_idx(gaussians)

    # Build position tensor [T, G, 3]
    pos_by_time = {}
    for t in timestamps:
        pos_by_time[t] = _query_positions_at_time(gaussians, t, idx).cpu()

    mu = torch.stack([pos_by_time[t] for t in timestamps], dim=0)  # [T, G, 3]
    velocity = mu[1:] - mu[:-1]                                     # [T-1, G, 3]
    motion_mag = torch.norm(velocity, dim=2).mean(dim=0)            # [G]

    object_motion = {}  # label_int -> list[float]

    for view in masked_views:
        mask = view.mask
        if mask is None:
            continue
        # Normalise mask to a 2-D numpy int array
        if isinstance(mask, torch.Tensor):
            mask_np = mask.squeeze().cpu().numpy()
        else:
            mask_np = np.asarray(mask)
        mask_np = mask_np.astype(np.int32)
        if mask_np.ndim != 2:
            continue
        H, W = mask_np.shape

        t = view.time
        if t not in pos_by_time:
            continue

        pts = pos_by_time[t].cuda()   # [G, 3]
        G = pts.shape[0]
        ones = torch.ones(G, 1, device="cuda")
        pts_h = torch.cat([pts, ones], dim=1)  # [G, 4]

        proj = view.full_proj_transform.cuda()  # [4, 4] column-major
        pts_clip = torch.matmul(pts_h, proj)    # [G, 4]
        w = pts_clip[:, 3:4].abs().clamp(min=1e-6)
        ndc = pts_clip[:, :2] / w              # [G, 2]

        # NDC to pixel (OpenGL: y up, image: y down)
        px = ((ndc[:, 0] + 1.0) * 0.5 * W).long()
        py = ((1.0 - (ndc[:, 1] + 1.0) * 0.5) * H).long()

        in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        valid_idx = torch.where(in_bounds)[0].cpu().numpy()
        px_v = px[in_bounds].cpu().numpy()
        py_v = py[in_bounds].cpu().numpy()

        for k, (vx, vy) in enumerate(zip(px_v, py_v)):
            label = int(mask_np[vy, vx])
            if label == 0:   # treat 0 as background
                continue
            if label not in object_motion:
                object_motion[label] = []
            object_motion[label].append(motion_mag[valid_idx[k]].item())

    return {
        f"object_{lbl}": float(np.mean(mags))
        for lbl, mags in object_motion.items()
    }


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    print("point nums:",gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        
        rendering = render(view, gaussians, pipeline, background,cam_type=cam_type)["render"]
        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)

    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))

    multithread_write(gt_list, gts_path)

    multithread_write(render_list, render_path)

    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30)
    return render_images  # list of HWC uint8 numpy arrays (used for TFS)
def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type)

        video_frames = None
        video_views = None
        if not skip_video:
            video_views = scene.getVideoCameras()
            video_frames = render_set(dataset.model_path,"video",scene.loaded_iter,video_views,gaussians,pipeline,background,cam_type)

        # ---- Temporal stability metrics (GDS, TFS, POIS) ----
        if video_views is not None and video_frames is not None:
            print("\n[Metrics] Computing temporal stability metrics...")

            gds = compute_gds(gaussians, video_views, cam_type)
            print(f"[Metrics] Gaussian Drift Score (GDS): {gds:.6f}")

            tfs = compute_tfs(video_frames)
            print(f"[Metrics] Temporal Flicker Score (TFS): {tfs:.6f}")

            # POIS uses test cameras: they carry real segmentation masks whereas
            # video cameras are synthetic orbit views with mask=None.
            pois_views = scene.getTestCameras()
            pois = compute_pois(gaussians, pois_views, cam_type)
            if pois:
                print("[Metrics] Object Instability Scores (POIS):")
                for obj_name, score in sorted(pois.items()):
                    print(f"  {obj_name}: {score:.6f}")
            else:
                print("[Metrics] POIS: no object masks found in test cameras, skipped.")

            # Persist alongside GPS in temporal_metrics.json
            metrics_path = os.path.join(dataset.model_path, "temporal_metrics.json")
            existing = {}
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    existing = json.load(f)
            existing.update({"GDS": gds, "TFS": tfs, "POIS": pois})
            with open(metrics_path, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"[Metrics] Temporal metrics written to {metrics_path}")
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)