# Environment Fixes

Commands run to fix dependency issues before training.

## 1. NumPy 2.x incompatibility with PyTorch / torchvision / lpips

**Error:** `_ARRAY_API not found` — torch 2.0.1 and torchvision 0.15.2 were compiled against NumPy 1.x and crash with NumPy 2.x.

```bash
pip install "numpy<2"
```

Installed: `numpy 1.26.4`
Note: opencv-python 4.13 formally requires numpy>=2 but still functions correctly with 1.26.4.

---

## 2. Missing libX11.so.6 (open3d)

**Error:** `OSError: libX11.so.6: cannot open shared object file: No such file or directory`

```bash
apt-get install -y libx11-6
```

---

## 3. Missing libGL.so.1 (open3d)

**Error:** `OSError: libGL.so.1: cannot open shared object file: No such file or directory`

```bash
apt-get install -y libgl1 libglib2.0-0
```

---

## 4. `typing_extensions` too old for open3d / dash

**Error:** `AttributeError: module 'typing_extensions' has no attribute 'Generic'`

```bash
pip install --upgrade typing_extensions
```

Upgraded: `4.5.0 → 4.15.0`

---

## 5. libgomp thread exhaustion → segfault in render.py

**Error:** `libgomp: Thread creation failed: Resource temporarily unavailable` / `free(): corrupted unsorted chunks` / segfault

Two causes:
- `OMP_NUM_THREADS` unset → OpenMP spawns one thread per CPU (64), exhausting resources
- `render.py:multithread_write` uses `ThreadPoolExecutor(max_workers=None)` → another unbounded pool on top

**Fix 1** — cap OMP threads in `run_all_scenes.sh`:
```bash
export OMP_NUM_THREADS=4
```

**Fix 2** — cap the write thread pool in `render.py` line 31:
```python
# before
executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
# after
executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
```

---

## 6. Renders completely white — trivial background solution

**Symptom:** All rendered images are pure white (pixel value 255 everywhere); ground truth images show colored objects correctly; trained model has ~101k Gaussians all with opacity sigmoid(-5.8) ≈ 0.003 (essentially transparent).

**Root cause:** The scenes use a white floor + white skybox (≈90% of each image is white). With `white_background=True` (the 4DGS default), a transparent Gaussian composited against white = white = matches the GT exactly for those pixels. The model converges to the trivial "stay invisible, let the white background explain the image" solution. Densification never fires meaningfully — only ~1,000 new Gaussians were created in 20,000 iterations.

**Fix:** Set `white_background = False` in every per-scene config. The training images are RGBA PNGs; `dataset_readers.py` composites them at load time using this flag. With black background, transparent Gaussians render black where colored objects should be, producing large photometric loss that forces proper densification.

Added to all `4dgs-test-dataset/arguments/scene*.py` files:
```python
ModelParams = dict(
    white_background = False,
)
```

---

## 7. Needle / spike Gaussian artifacts in renders

**Symptom:** Long thin Gaussian spikes radiating from the scene; max Gaussian scale 2678 world units in a scene bounded to ±2 units.

**Root cause — two compounding issues:**

1. `densify_until_iter = 15000` (default) stops all densification *and pruning* at iter 15000, but training runs to 20000. The last 5000 iterations have no scale pruning, so Gaussians grow freely via gradient descent.

2. The `prune()` call in `train.py:274` had a `gaussians.get_xyz.shape[0] > 200000` guard. Starting from 100k Gaussians, this prevented scale-based pruning from firing at all until count crossed 200k — which may never happen in small scenes.

**Fix 1** — `4DGaussians/arguments/dnerf/dnerf_default.py`:
```python
# before
iterations = 20000,
pruning_interval = 8000,
# after
iterations = 20000,
densify_until_iter = 17000,   # extend past original 15000 to cover the post-15k needle growth
                               # gap, but preserve a 3000-iter settling phase before end
pruning_interval = 3000,       # ~5 prune calls vs original 2; avoids the prune-regrow
                               # thrash that interval=1000 caused (needles got worse)
```

**Fix 2** — `train.py:274`, remove the Gaussian count gate:
```python
# before
if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
# after
if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
```

---

## 8. Ablation study: pruning schedule

Three runs to isolate the effect of densification window and pruning frequency. The count gate removal (`gaussians.get_xyz.shape[0] > 200000` dropped) is held constant across all runs as it is a correctness fix with no downside.

| Run | `densify_until_iter` | `pruning_interval` | Hypothesis |
|-----|---------------------|--------------------|------------|
| **A — vanilla** | 15000 (default) | 8000 (default) | Baseline behaviour: needles form in iters 15k–20k unchecked; only 2 prune calls total |
| **B — current** | 17000 | 3000 | ~5 prune calls; covers the 15k–17k needle-growth gap; 3k settling phase preserved |
| **C — aggressive** | 20000 | 1000 | ~19 prune calls; confirmed empirically to make needles **worse** due to prune-regrow thrash |

**Questions each comparison answers:**
- **A vs B**: does extending past 15k + modest frequency increase reduce needles without hurting reconstruction quality (PSNR/SSIM)?
- **B vs C**: does over-aggressive pruning destabilise training on monocular D-NeRF data?
- **A vs C**: is naive "more pruning = better" categorically wrong for this setting?

**To reproduce each run**, change `dnerf_default.py` values and re-run `run_all_scenes.sh` with output directed to a labelled directory (e.g. `--expname scene1_runA`). The `run_all_scenes.sh` `expname` and output path would need per-run suffixes to avoid overwriting.

---

## 9. Training modes and loss flags

Three training modes are implemented, selectable via `run_all_scenes.sh` flags. Each mode appends a suffix to the output directory so all runs coexist.

### Mode 1 — Baseline (no flags)
Standard 4DGS training with the pruning fixes from §7. Output: `output/scene1_close_proximity/`

### Mode 2 — Foreground mask loss (`--fg_mask_loss`)
Output: `output/scene1_close_proximity_fg_mask/`

Replaces the standard L1 loss with a pixel-weighted variant that emphasises foreground objects:

```python
weight = mask_tensor * (1.0 - fg_bg_weight) + fg_bg_weight
Ll1 = (|render - gt| * weight).mean()
```

- `fg_bg_weight=1.0` → identical to standard L1
- `fg_bg_weight=0.05` → default; foreground pixels get 20× more gradient than background
- `fg_bg_weight=0.0` → **broken**: background Gaussians receive no gradient, grow unconstrained into colorful noise

**Important:** zeroing background loss completely (the intuitive interpretation of "foreground masking") is incorrect. Background Gaussians still need some gradient signal to stay transparent and small. The 0.05 weight keeps them constrained while concentrating gradient budget on the objects.

**`--fg_bg_weight F`** (default 0.05) is exposed as a tunable parameter. Values to ablate: 0.01, 0.05, 0.1, 0.2.

Masks are loaded from `{split}/masks/{frame}.png` (uint8, 0=background, 1…K=objects). These were already generated by the dataset and are passed through `dataset_readers.py → camera_utils.py → Camera.mask`.

### Mode 3 — Drift loss + scale regularisation (`--drift_loss --scale_reg`)
Output: `output/scene1_close_proximity_drift_scale_reg/`

**Scale regularisation (`--scale_reg`, `--lambda_scale_reg`, default 0.01):**
Adds `lambda_scale_reg * mean(max_scale_per_gaussian)` to the loss. Directly penalises elongated/needle Gaussians by discouraging large scale values.

**Trajectory-guided drift loss (`--drift_loss`, `--lambda_drift 0.05`):**
Uses ground-truth object trajectories from `4dgs-test-dataset/trajectories/` to anchor Gaussians to their assigned objects. Two methods were considered:

---

#### Method A — Fixed bounding sphere (initial approach, superseded)

Each foreground object is bounded by a single sphere of fixed radius `drift_radius` (default 0.6 m):

```
loss_k = mean_{i: label_i==k} [ opacity_i * ReLU(||xyz_i − c_k(t)|| − drift_radius) ]
```

**Label assignment:** at t=0, any Gaussian within `drift_radius` of an object centre gets that object's label.

**Limitations:**
- A sphere is the wrong shape for non-spherical objects. For scene 8 rods (2 m long, ~0.1 m diameter) a sphere that fits the length is 20× too wide and captures background Gaussians; a tight sphere truncates the ends.
- The 0.6 m default was calibrated for 0.3 m radius spheres — every other scene type required manual tuning.
- Scene 9 droplets change apparent size as they split/merge; a fixed radius is either too tight at rest or too loose when separated.

---

#### Method B — Shape-aware SDF bounding volumes (current implementation)

Each object exposes its true bounding geometry via `trajectory.get_object_bounds(time, object_id)`, returning one of:

```python
{'type': 'sphere',  'radius': float}
{'type': 'capsule', 'half_length': float, 'radius': float}   # axis = local X
{'type': 'obb',     'half_extents': np.array([hx, hy, hz])}  # orientation from quaternion
```

The drift penalty uses the signed-distance function (SDF) for the appropriate shape:

```
loss_k = mean_{i: label_i==k} [ opacity_i * ReLU(SDF_k(xyz_i, t)) ]
```

SDF > 0 means outside the bounding volume; ReLU zeroes the gradient for Gaussians already inside (no unnecessary pull toward centre).

**Per-object geometry:**

| Scene | Object | Bound |
|---|---|---|
| 1–3, 5, 6, 10 | Spheres | `sphere` r=0.3 m |
| 4 | Sphere | `sphere` r=0.3 m |
| 4 | Wall | `obb` half-extents [0.05, 0.4, 1.0] m |
| 7 | Sphere (deformable) | `sphere` r=0.70 m (radius + max compression slack) |
| 7 | Cube | `obb` half-extents [0.6, 0.6, 0.6] m |
| 8 | Both rods | `capsule` half-length=1.0 m, radius=0.06 m |
| 9 | Droplets | `sphere` r=0.40 m (merged) → 0.28 m (fully split), time-varying |

**SDF implementations (`utils/trajectory_utils.py`):**
- `_sphere_sdf`: `||p − c|| − r`
- `_capsule_sdf`: transform to object local frame via quaternion, clamp projection onto X-axis to ±half_length, compute distance to closest point on segment, subtract radius
- `_obb_sdf`: transform to object local frame, apply standard box SDF (`||max(|q| − h, 0)|| + min(max(|q| − h), 0)`)

**Label assignment:** same SDF at t=0 with 5 cm slack (`sdf ≤ 0.05`) replaces the fixed-radius proximity check.

**Files modified:**
1. `4dgs-test-dataset/trajectories/trajectory_base.py` — `get_object_bounds()` default (sphere r=0.35)
2. All 10 scene trajectory files — per-object overrides
3. `utils/trajectory_utils.py` — full SDF rewrite
4. `scene/gaussian_model.py` — `init_object_labels()` signature (no `drift_radius` arg)
5. `train.py` — `--drift_radius` argument removed; `compute_drift_loss` call updated

**Key design choices:**
- Penalty on canonical positions (`_xyz`), not deformed — the deformation network owns temporal motion
- Opacity detached so gradients flow only through xyz (prevents Gaussians going transparent to escape penalty)
- Background Gaussians (label=0) exempt; labels propagate through clone/split/prune

Scenes where drift loss has clearest advantage: **scene2** (identical spheres, no colour cue), **scene3** (three-body collision, momentary overlap), **scene8** (rods require capsule — sphere would be useless).

### Combined mode (`--fg_mask_loss --drift_loss --scale_reg`)
Predicted best overall: fg_mask removes background pressure, scale_reg kills remaining needle Gaussians, drift loss prevents object-boundary scatter.

---

## 10. Ablation study: pruning schedule

Three runs to isolate the effect of densification window and pruning frequency. The count gate removal (`gaussians.get_xyz.shape[0] > 200000` dropped) is held constant across all runs as it is a correctness fix with no downside.

| Run | `densify_until_iter` | `pruning_interval` | Hypothesis |
|-----|---------------------|--------------------|------------|
| **A — vanilla** | 15000 (default) | 8000 (default) | Baseline behaviour: needles form in iters 15k–20k unchecked; only 2 prune calls total |
| **B — current** | 17000 | 3000 | ~5 prune calls; covers the 15k–17k needle-growth gap; 3k settling phase preserved |
| **C — aggressive** | 20000 | 1000 | ~19 prune calls; confirmed empirically to make needles **worse** due to prune-regrow thrash |

**Questions each comparison answers:**
- **A vs B**: does extending past 15k + modest frequency increase reduce needles without hurting reconstruction quality (PSNR/SSIM)?
- **B vs C**: does over-aggressive pruning destabilise training on monocular D-NeRF data?
- **A vs C**: is naive "more pruning = better" categorically wrong for this setting?

---

## Summary (run in order)

```bash
apt-get install -y libx11-6 libgl1 libglib2.0-0
pip install "numpy<2"
pip install --upgrade typing_extensions
```
