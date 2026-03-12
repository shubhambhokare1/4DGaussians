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

## 9. Planned: Trajectory-guided Gaussian drift loss

**Motivation:** Gaussians can drift arbitrarily far from the objects they were initialised to represent, producing floaters and needle artefacts. Standard 4DGS has no constraint anchoring Gaussians to objects.

**Approach:** Use the dataset's ground-truth trajectory data (in `4dgs-test-dataset/trajectories/scene*_trajectories.py`) to penalise Gaussians that drift outside their assigned object's bounding sphere at each training timestep. This replaces the learned feature grouping of Gaussian Grouping (Liang et al.) with exact analytical object bounds — no SAM, no extra GPU memory, fully differentiable.

**Implementation plan:**

1. **`utils/trajectory_utils.py`** (new) — `get_object_bounds(scene_name, time) → dict[str, (center, radius)]`
   Imports the scene's trajectory class and returns per-object 3D bounding spheres at the queried time.

2. **`scene/gaussian_model.py`**
   - Add `self.object_ids: LongTensor[N]` (label per Gaussian: object index or -1 for background/floor)
   - Initialise in `create_from_pcd` by assigning each Gaussian to whichever object's t=0 bounding sphere it falls inside
   - Propagate through `densify_and_clone` / `densify_and_split` (child inherits parent's id)
   - Persist through `save_ply` / `load_ply`

3. **`train.py`**
   - After the render step, query `get_object_bounds(scene, current_time)`
   - For each object `k`, compute `dist = ||means3D_final[object_ids==k] - center_k|| - radius_k`
   - Drift loss: `lambda_drift * sum( opacity[object_ids==k] * ReLU(dist) )`
   - Add to total loss (fine stage only, after coarse stage stabilises)
   - New config param `lambda_drift` (suggested default: `0.01`) in `OptimizationParams`

4. **`4dgs-test-dataset/arguments/dnerf_default.py`** — add `lambda_drift = 0.01`

**Key properties:**
- No penalty for Gaussians inside their object's sphere — only drifters are penalised
- Opacity-weighted so nearly-transparent floaters are suppressed naturally
- Floor/background Gaussians (object_id == -1) are exempt from the drift loss
- Scene 2 (identical spheres) is the hardest case: objects swap visual appearance so Gaussian assignment may not survive the proximity event — a known limitation worth documenting

---

## Summary (run in order)

```bash
apt-get install -y libx11-6 libgl1 libglib2.0-0
pip install "numpy<2"
pip install --upgrade typing_extensions
```
