# 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

## CVPR 2024

### [Project Page](https://guanjunwu.github.io/4dgs/index.html)| [arXiv Paper](https://arxiv.org/abs/2310.08528)

[Guanjun Wu](https://guanjunwu.github.io/) <sup>1*</sup>, [Taoran Yi](https://github.com/taoranyi) <sup>2*</sup>,
[Jiemin Fang](https://jaminfong.cn/) <sup>3‡</sup>, [Lingxi Xie](http://lingxixie.com/) <sup>3 </sup>, </br>[Xiaopeng Zhang](https://scholar.google.com/citations?user=Ud6aBAcAAAAJ&hl=zh-CN) <sup>3 </sup>, [Wei Wei](https://www.eric-weiwei.com/) <sup>1 </sup>,[Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/) <sup>2 </sup>, [Qi Tian](https://www.qitian1987.com/) <sup>3 </sup> , [Xinggang Wang](https://xwcv.github.io) <sup>2‡✉</sup>

<sup>1 </sup>School of CS, HUST &emsp; <sup>2 </sup>School of EIC, HUST &emsp; <sup>3 </sup>Huawei Inc. &emsp;

<sup>\*</sup> Equal Contributions. <sup>$\ddagger$</sup> Project Lead. <sup>✉</sup> Corresponding Author.



![block](assets/teaserfig.jpg)
Our method converges very quickly and achieves real-time rendering speed.

New Colab demo:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wz0D5Y9egAlcxXy8YO9UmpQ9oH51R7OW?usp=sharing) (Thanks [Tasmay-Tibrewal
](https://github.com/Tasmay-Tibrewal))

Old Colab demo:[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hustvl/4DGaussians/blob/master/4DGaussians.ipynb) (Thanks [camenduru](https://github.com/camenduru/4DGaussians-colab).)

Light Gaussian implementation: [This link](https://github.com/pablodawson/4DGaussians) (Thanks [pablodawson](https://github.com/pablodawson))


## News

2024.6.25: we clean the code and add an explanation of the parameters.

2024.3.25: Update guidance for hypernerf and dynerf dataset.

2024.03.04: We change the hyperparameters of the Neu3D dataset, corresponding to our paper.

2024.02.28: Update SIBR viewer guidance.

2024.02.27: Accepted by CVPR 2024. We delete some logging settings for debugging, the corrected training time is only **8 mins** (20 mins before) in D-NeRF datasets and **30 mins** (1 hour before) in HyperNeRF datasets. The rendering quality is not affected.

## Fixes and Improvements (Shubham Bhokare)

This section summarizes the latest bug fixes, training-mode additions, and environment setup adjustments. The newly proposed training modes are described first, followed by additional training/stability fixes, then environment fixes.

### Training modes and loss flags (new; from FIXED.md)

Three training modes are implemented, selectable via `run_all_scenes.sh` flags. Each mode appends a suffix to the output directory so all runs coexist.

- Mode 1 — Baseline (no flags)
  - Standard 4DGS training with the pruning fixes described below.
  - Output: `output/scene1_close_proximity/`

- Mode 2 — Foreground mask loss (`--fg_mask_loss`)
  - Output: `output/scene1_close_proximity_fg_mask/`
  - Replaces the standard L1 loss with a pixel-weighted version that emphasizes foreground:

    ```python
    weight = mask_tensor * (1.0 - fg_bg_weight) + fg_bg_weight
    Ll1 = (|render - gt| * weight).mean()
    ```

  - `fg_bg_weight=1.0` → identical to standard L1
  - `fg_bg_weight=0.05` → default; foreground pixels get 20× more gradient than background
  - `fg_bg_weight=0.0` → broken: background Gaussians receive no gradient, grow unconstrained into colorful noise

  - `--fg_bg_weight F` (default 0.05) is tunable. Values to ablate: 0.01, 0.05, 0.1, 0.2.

- Mode 3 — Drift loss + scale regularisation (`--drift_loss --scale_reg`)
  - Output: `output/scene1_close_proximity_drift_scale_reg/`

  - `--scale_reg` (`--lambda_scale_reg`, default 0.01): penalty on elongated/needle Gaussians via mean(max_scale_per_gaussian)
  - `--drift_loss` (`--lambda_drift 0.05`): anchors Gaussians to ground-truth object trajectories, with shape-aware SDF bounds.

Drift loss currently supports two methods:

- Method B — Shape-aware SDF bounding volumes (current implementation): per-object `sphere` / `capsule` / `obb` bounds and SDF-based ReLU penalty.
- Method A — Fixed bounding sphere (initial, superseded): simpler and less robust to non-spherical objects.

Combined mode: `--fg_mask_loss --drift_loss --scale_reg` (recommended to combine constraints).

### Additional training and stability fixes

#### Needles / spikes and pruning schedule

- Default `densify_until_iter=15000` stopped pruning at 15k, leaving 15–20k as unregulated needle growth.
- Fix: `densify_until_iter=17000`, `pruning_interval=3000`, and remove the `gaussians.get_xyz.shape[0] > 200000` guard in `train.py` to ensure consistent pruning.

  - `dnerf_default.py` change:
    - `densify_until_iter = 17000`
    - `pruning_interval = 3000`

  - `train.py` change:
    - `if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:` (no 200k gate)

- Ablation schedule:
  - A: 15000 / 8000 (baseline, needles form 15k–20k)
  - B: 17000 / 3000 (current)
  - C: 20000 / 1000 (too aggressive, prune-regrow thrash)

#### White-background trivial solution fix

- If training images are RGBA and `white_background=True`, transparent Gaussians can produce trivially correct white pixels; models can collapse to empty scene.
- Fix: set `white_background=False` in per-scene configs and data readers.

#### OMP thread exhaustion in `render.py`

- `OMP_NUM_THREADS` missing can create too many threads and fail with `libgomp: Thread creation failed`.
- `render.py` threadpool was unbounded (`max_workers=None`); change to `max_workers=8`.
- Add `export OMP_NUM_THREADS=4` to `run_all_scenes.sh`.

### Environment fixes

Commands run to fix dependency issues before training.

1. NumPy 2.x incompatibility with PyTorch / torchvision / lpips

   Error: `_ARRAY_API not found`. Fix:
   ```bash
   pip install "numpy<2"
   ```
   Installed: `numpy 1.26.4` (opencv-python 4.13 is okay with 1.26.4).

2. Missing `libX11.so.6` (open3d)

   ```bash
   apt-get install -y libx11-6
   ```

3. Missing `libGL.so.1` (open3d)

   ```bash
   apt-get install -y libgl1 libglib2.0-0
   ```

4. `typing_extensions` too old for open3d / dash

   Error: `AttributeError: module 'typing_extensions' has no attribute 'Generic'`
   ```bash
   pip install --upgrade typing_extensions
   ```
   Upgraded: `4.5.0 → 4.15.0`

5. `libgomp` thread exhaustion → segfault in `render.py` (see thread pool fix above)

6. Renders completely white due to `white_background=True` (see fix above)

---

## Environmental Setups

Please follow the [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) to install the relative packages.

```bash
git clone https://github.com/hustvl/4DGaussians
cd 4DGaussians
git submodule update --init --recursive
conda create -n Gaussians4D python=3.7 
conda activate Gaussians4D

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

In our environment, we use pytorch=1.13.1+cu116.

## Data Preparation

**Target dataset:**
We recommend using the curated test set at `4dgs-test-dataset`: https://github.com/shubhambhokare1/4dgs-test-dataset

**For synthetic scenes:**
The dataset provided in [D-NeRF](https://github.com/albertpumarola/D-NeRF) is used. You can download the dataset from [dropbox](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0).

**For real dynamic scenes:**
The dataset provided in [HyperNeRF](https://github.com/google/hypernerf) is used. You can download scenes from [Hypernerf Dataset](https://github.com/google/hypernerf/releases/tag/v0.1) and organize them as [Nerfies](https://github.com/google/nerfies#datasets). 

Meanwhile, [Plenoptic Dataset](https://github.com/facebookresearch/Neural_3D_Video) could be downloaded from their official websites. To save the memory, you should extract the frames of each video and then organize your dataset as follows.

```
├── data
│   | dnerf 
│     ├── mutant
│     ├── standup 
│     ├── ...
│   | hypernerf
│     ├── interp
│     ├── misc
│     ├── virg
│   | dynerf
│     ├── cook_spinach
│       ├── cam00
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── 0002.png
│               ├── ...
│       ├── cam01
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── ...
│     ├── cut_roasted_beef
|     ├── ...
```

**For multipleviews scenes:**
If you want to train your own dataset of multipleviews scenes, you can orginize your dataset as follows:

```
├── data
|   | multipleview
│     | (your dataset name) 
│   	  | cam01
|     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | cam02
│     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | ...
```
After that, you can use the  `multipleviewprogress.sh` we provided to generate related data of poses and pointcloud.You can use it as follows:
```bash
bash multipleviewprogress.sh (youe dataset name)
```
You need to ensure that the data folder is organized as follows after running multipleviewprogress.sh:
```
├── data
|   | multipleview
│     | (your dataset name) 
│   	  | cam01
|     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | cam02
│     		  ├── frame_00001.jpg
│     		  ├── frame_00002.jpg
│     		  ├── ...
│   	  | ...
│   	  | sparse_
│     		  ├── cameras.bin
│     		  ├── images.bin
│     		  ├── ...
│   	  | points3D_multipleview.ply
│   	  | poses_bounds_multipleview.npy
```


## Training

For training synthetic scenes such as `bouncingballs`, run

```
python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py 
```

For training dynerf scenes such as `cut_roasted_beef`, run
```python
# First, extract the frames of each video.
python scripts/preprocess_dynerf.py --datadir data/dynerf/cut_roasted_beef
# Second, generate point clouds from input data.
bash colmap.sh data/dynerf/cut_roasted_beef llff
# Third, downsample the point clouds generated in the second step.
python scripts/downsample_point.py data/dynerf/cut_roasted_beef/colmap/dense/workspace/fused.ply data/dynerf/cut_roasted_beef/points3D_downsample2.ply
# Finally, train.
python train.py -s data/dynerf/cut_roasted_beef --port 6017 --expname "dynerf/cut_roasted_beef" --configs arguments/dynerf/cut_roasted_beef.py 
```
For training hypernerf scenes such as `virg/broom`: Pregenerated point clouds by COLMAP are provided [here](https://drive.google.com/file/d/1fUHiSgimVjVQZ2OOzTFtz02E9EqCoWr5/view). Just download them and put them in to correspond folder, and you can skip the former two steps. Also, you can run the commands directly.

```python
# First, computing dense point clouds by COLMAP
bash colmap.sh data/hypernerf/virg/broom2 hypernerf
# Second, downsample the point clouds generated in the first step. 
python scripts/downsample_point.py data/hypernerf/virg/broom2/colmap/dense/workspace/fused.ply data/hypernerf/virg/broom2/points3D_downsample2.ply
# Finally, train.
python train.py -s  data/hypernerf/virg/broom2/ --port 6017 --expname "hypernerf/broom2" --configs arguments/hypernerf/broom2.py 
```

For training multipleviews scenes,you are supposed to build a configuration file named (you dataset name).py under "./arguments/mutipleview",after that,run
```python
python train.py -s  data/multipleview/(your dataset name) --port 6017 --expname "multipleview/(your dataset name)" --configs arguments/multipleview/(you dataset name).py 
```


For your custom datasets, install nerfstudio and follow their [COLMAP](https://colmap.github.io/) pipeline. You should install COLMAP at first, then:

```python
pip install nerfstudio
# computing camera poses by colmap pipeline
ns-process-data images --data data/your-data --output-dir data/your-ns-data
cp -r data/your-ns-data/images data/your-ns-data/colmap/images
python train.py -s data/your-ns-data/colmap --port 6017 --expname "custom" --configs arguments/hypernerf/default.py 
```
You can customize your training config through the config files.

## Checkpoint

Also, you can train your model with checkpoint.

```python
python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py --checkpoint_iterations 200 # change it.
```

Then load checkpoint with:

```python
python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py --start_checkpoint "output/dnerf/bouncingballs/chkpnt_coarse_200.pth"
# finestage: --start_checkpoint "output/dnerf/bouncingballs/chkpnt_fine_200.pth"
```

## Rendering

Run the following script to render the images.

```
python render.py --model_path "output/dnerf/bouncingballs/"  --skip_train --configs arguments/dnerf/bouncingballs.py 
```

## Evaluation

You can just run the following script to evaluate the model.

```
python metrics.py --model_path "output/dnerf/bouncingballs/" 
```


## Viewer
[Watch me](./docs/viewer_usage.md)
## Scripts

There are some helpful scripts, please feel free to use them.

`export_perframe_3DGS.py`:
get all 3D Gaussians point clouds at each timestamps.

usage:

```python
python export_perframe_3DGS.py --iteration 14000 --configs arguments/dnerf/lego.py --model_path output/dnerf/lego 
```

You will a set of 3D Gaussians are saved in `output/dnerf/lego/gaussian_pertimestamp`.

`weight_visualization.ipynb`:

visualize the weight of Multi-resolution HexPlane module.

`merge_many_4dgs.py`:
merge your trained 4dgs.
usage:

```python
export exp_name="dynerf"
python merge_many_4dgs.py --model_path output/$exp_name/sear_steak
```

`colmap.sh`:
generate point clouds from input data

```bash
bash colmap.sh data/hypernerf/virg/vrig-chicken hypernerf 
bash colmap.sh data/dynerf/sear_steak llff
```

**Blender** format seems doesn't work. Welcome to raise a pull request to fix it.

`downsample_point.py` :downsample generated point clouds by sfm.

```python
python scripts/downsample_point.py data/dynerf/sear_steak/colmap/dense/workspace/fused.ply data/dynerf/sear_steak/points3D_downsample2.ply
```

In my paper, I always use `colmap.sh` to generate dense point clouds and downsample it to less than 40000 points.

Here are some codes maybe useful but never adopted in my paper, you can also try it.

## Awesome Concurrent/Related Works

Welcome to also check out these awesome concurrent/related works, including but not limited to

[Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction](https://ingra14m.github.io/Deformable-Gaussians/)

[SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes](https://yihua7.github.io/SC-GS-web/)

[MD-Splatting: Learning Metric Deformation from 4D Gaussians in Highly Deformable Scenes](https://md-splatting.github.io/)

[4DGen: Grounded 4D Content Generation with Spatial-temporal Consistency](https://vita-group.github.io/4DGen/)

[Diffusion4D: Fast Spatial-temporal Consistent 4D Generation via Video Diffusion Models](https://github.com/VITA-Group/Diffusion4D)

[DreamGaussian4D: Generative 4D Gaussian Splatting](https://github.com/jiawei-ren/dreamgaussian4d)

[EndoGaussian: Real-time Gaussian Splatting for Dynamic Endoscopic Scene Reconstruction](https://github.com/yifliu3/EndoGaussian)

[EndoGS: Deformable Endoscopic Tissues Reconstruction with Gaussian Splatting](https://github.com/HKU-MedAI/EndoGS)

[Endo-4DGS: Endoscopic Monocular Scene Reconstruction with 4D Gaussian Splatting](https://arxiv.org/abs/2401.16416)



## Contributions

**This project is still under development. Please feel free to raise issues or submit pull requests to contribute to our codebase.**


Some source code of ours is borrowed from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [K-planes](https://github.com/Giodiro/kplanes_nerfstudio), [HexPlane](https://github.com/Caoang327/HexPlane), [TiNeuVox](https://github.com/hustvl/TiNeuVox), [Depth-Rasterization](https://github.com/ingra14m/depth-diff-gaussian-rasterization). We sincerely appreciate the excellent works of these authors.

## Acknowledgement

We would like to express our sincere gratitude to [@zhouzhenghong-gt](https://github.com/zhouzhenghong-gt/) for his revisions to our code and discussions on the content of our paper.

## Citation

Some insights about neural voxel grids and dynamic scenes reconstruction originate from [TiNeuVox](https://github.com/hustvl/TiNeuVox). If you find this repository/work helpful in your research, welcome to cite these papers and give a ⭐.

```
@InProceedings{Wu_2024_CVPR,
    author    = {Wu, Guanjun and Yi, Taoran and Fang, Jiemin and Xie, Lingxi and Zhang, Xiaopeng and Wei, Wei and Liu, Wenyu and Tian, Qi and Wang, Xinggang},
    title     = {4D Gaussian Splatting for Real-Time Dynamic Scene Rendering},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {20310-20320}
}

@inproceedings{TiNeuVox,
  author = {Fang, Jiemin and Yi, Taoran and Wang, Xinggang and Xie, Lingxi and Zhang, Xiaopeng and Liu, Wenyu and Nie\ss{}ner, Matthias and Tian, Qi},
  title = {Fast Dynamic Radiance Fields with Time-Aware Neural Voxels},
  year = {2022},
  booktitle = {SIGGRAPH Asia 2022 Conference Papers}
}
```
