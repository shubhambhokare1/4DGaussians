"""
Utilities for the trajectory-guided Gaussian drift loss.

Maps scene folder names to trajectory classes, assigns per-Gaussian object
labels at initialisation, and computes the drift penalty at training time.
"""

import os
import sys
import importlib
import numpy as np
import torch

# ── Scene → (module, class) registry ────────────────────────────────────────
_SCENE_TO_MODULE = {
    "scene1_close_proximity":   ("scene1_trajectories",  "Scene1Trajectory"),
    "scene2_identical_objects": ("scene2_trajectories",  "Scene2Trajectory"),
    "scene3_collision":         ("scene3_trajectories",  "Scene3Trajectory"),
    "scene4_occlusion":         ("scene4_trajectories",  "Scene4Trajectory"),
    "scene5_rapid_motion":      ("scene5_trajectories",  "Scene5Trajectory"),
    "scene6_scale_change":      ("scene6_trajectories",  "Scene6Trajectory"),
    "scene7_deformation":       ("scene7_trajectories",  "Scene7Trajectory"),
    "scene8_thin_structure":    ("scene8_trajectories",  "Scene8Trajectory"),
    "scene9_topology":          ("scene9_trajectories",  "Scene9Trajectory"),
    "scene10_texture":          ("scene10_trajectories", "Scene10Trajectory"),
}

_TRAJ_PACKAGE_PARENT = "/root/4dgs-test-dataset"


def load_trajectory(scene_name: str):
    """Instantiate the trajectory class for *scene_name*."""
    if scene_name not in _SCENE_TO_MODULE:
        raise ValueError(f"No trajectory registered for scene '{scene_name}'")
    mod_name, cls_name = _SCENE_TO_MODULE[scene_name]
    if _TRAJ_PACKAGE_PARENT not in sys.path:
        sys.path.insert(0, _TRAJ_PACKAGE_PARENT)
    pkg = importlib.import_module(f"trajectories.{mod_name}")
    return getattr(pkg, cls_name)()


def get_object_centers_at_time(trajectory, time: float) -> dict:
    """Return {object_id: np.array([x,y,z])} for all objects at *time*."""
    return {
        oid: trajectory.get_object_state(time, oid)["position"]
        for oid in trajectory.get_object_ids()
    }


def assign_initial_labels(
    trajectory,
    xyz: np.ndarray,
    drift_radius: float,
    class_mapping: dict = None,
) -> torch.LongTensor:
    """
    Assign each Gaussian a long integer object label based on proximity to
    object centres at t=0.

    Args:
        trajectory:    instantiated TrajectoryBase subclass
        xyz:           float32 ndarray [N, 3] — canonical Gaussian positions
        drift_radius:  Gaussians beyond this from every object → label 0
        class_mapping: optional {object_id_str: int_label} from
                       class_mapping.json.  If None, objects get labels 1, 2, …
                       in get_object_ids() order.

    Returns:
        LongTensor[N], 0 = background, 1…K = objects
    """
    centers = get_object_centers_at_time(trajectory, 0.0)
    object_ids = trajectory.get_object_ids()
    xyz_t = torch.from_numpy(xyz.astype(np.float32))

    labels = torch.zeros(xyz.shape[0], dtype=torch.long)
    for i, oid in enumerate(object_ids):
        label_int = (
            class_mapping[oid]
            if (class_mapping and oid in class_mapping)
            else i + 1
        )
        center = torch.from_numpy(centers[oid].astype(np.float32))
        dists = torch.norm(xyz_t - center.unsqueeze(0), dim=1)
        labels[dists < drift_radius] = label_int

    return labels


def compute_drift_loss(
    gaussians,
    trajectory,
    time: float,
    drift_radius: float,
) -> torch.Tensor:
    """
    Trajectory-guided Gaussian drift penalty.

    For each foreground object k, penalises Gaussians with label k whose
    canonical positions stray beyond *drift_radius* of the object centre
    at *time*:

        loss = Σ_k  mean_{i: label_i==k} [ opacity_i * ReLU(||xyz_i − c_k(t)|| − r) ]

    Opacity is detached so gradients flow only through xyz.
    """
    if gaussians._object_labels is None:
        return torch.tensor(0.0, device="cuda")

    labels = gaussians._object_labels.cuda()   # [N]
    xyz = gaussians.get_xyz                    # [N, 3]  differentiable
    opacity = gaussians.get_opacity.squeeze(-1).detach()  # [N]

    centers = get_object_centers_at_time(trajectory, time)
    object_ids = trajectory.get_object_ids()

    loss = torch.tensor(0.0, device="cuda")
    for i, oid in enumerate(object_ids):
        label_int = i + 1
        center = torch.from_numpy(centers[oid].astype(np.float32)).cuda()
        mask = (labels == label_int)
        if not mask.any():
            continue
        dists = torch.norm(xyz[mask] - center.unsqueeze(0), dim=1)
        penalty = torch.relu(dists - drift_radius)
        loss = loss + (opacity[mask] * penalty).mean()

    return loss
