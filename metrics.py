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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
import numpy as np
import cv2
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from pytorch_msssim import ms_ssim


def _compute_tfs_from_dir(renders_dir: Path) -> float:
    """Temporal Flicker Score from a directory of sequentially-named renders."""
    fnames = sorted(
        f for f in os.listdir(renders_dir) if f.lower().endswith((".png", ".jpg"))
    )
    if len(fnames) < 2:
        return 0.0

    flicker_scores = []
    prev_frame = None
    for fname in fnames:
        img = np.array(Image.open(renders_dir / fname).convert("RGB"))
        if prev_frame is not None:
            gray_t = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            gray_t1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                gray_t, gray_t1, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            H, W = gray_t.shape
            map_x = (np.arange(W, dtype=np.float32)[None, :] + flow[:, :, 0]).clip(0, W - 1)
            map_y = (np.arange(H, dtype=np.float32)[:, None] + flow[:, :, 1]).clip(0, H - 1)
            warped = cv2.remap(prev_frame, map_x, map_y, cv2.INTER_LINEAR)
            diff = (img.astype(np.float32) - warped.astype(np.float32)) / 255.0
            flicker_scores.append(float((diff ** 2).mean()))
        prev_frame = img

    return float(np.mean(flicker_scores)) if flicker_scores else 0.0
def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                lpipsa = []
                ms_ssims = []
                Dssims = []
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    ms_ssims.append(ms_ssim(renders[idx], gts[idx],data_range=1, size_average=True ))
                    lpipsa.append(lpips(renders[idx], gts[idx], net_type='alex'))
                    Dssims.append((1-ms_ssims[-1])/2)

                print("Scene: ", scene_dir,  "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("Scene: ", scene_dir,  "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("Scene: ", scene_dir,  "LPIPS-vgg: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("Scene: ", scene_dir,  "LPIPS-alex: {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"))
                print("Scene: ", scene_dir,  "MS-SSIM: {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"))
                print("Scene: ", scene_dir,  "D-SSIM: {:>12.7f}".format(torch.tensor(Dssims).mean(), ".5"))

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS-vgg": torch.tensor(lpipss).mean().item(),
                                                        "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
                                                        "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
                                                        "D-SSIM": torch.tensor(Dssims).mean().item()},

                                                    )
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS-vgg": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                            "LPIPS-alex": {name: lp for lp, name in zip(torch.tensor(lpipsa).tolist(), image_names)},
                                                            "MS-SSIM": {name: lp for lp, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
                                                            "D-SSIM": {name: lp for lp, name in zip(torch.tensor(Dssims).tolist(), image_names)},

                                                            }
                                                        )

            # ---- TFS from saved video renders (if available) ----
            video_dir = Path(scene_dir) / "video"
            if video_dir.exists():
                for vid_method in os.listdir(video_dir):
                    vid_renders = video_dir / vid_method / "renders"
                    if not vid_renders.exists():
                        continue
                    tfs_val = _compute_tfs_from_dir(vid_renders)
                    full_dict[scene_dir][vid_method]["TFS"] = tfs_val
                    print(f"\n[Metrics] Temporal Flicker Score (TFS): {tfs_val:.6f}")
                    # Also log alongside GPS / GDS in temporal_metrics.json
                    tm_path = Path(scene_dir) / "temporal_metrics.json"
                    tm = {}
                    if tm_path.exists():
                        with open(tm_path, "r") as f:
                            tm = json.load(f)
                    tm["TFS"] = tfs_val
                    with open(tm_path, "w") as f:
                        json.dump(tm, f, indent=2)

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:

            print("Unable to compute metrics for model", scene_dir)
            raise e

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
