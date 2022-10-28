# M-FUSE: Multi-frame Fusion for Scene Flow Estimation

This repository contains the official code for our paper

[**M-FUSE: Multi-frame Fusion for Scene Flow Estimation**](https://arxiv.org/abs/2207.05704)<br>
L. Mehl, A. Jahedi, J. Schmalfuss, A. Bruhn<br>
*Winter Conference on Applications of Computer Vision (WACV)*, 2023.


```
@inproceedings{Mehl2023,
  title={{M-FUSE}: Multi-frame Fusion for Scene Flow Estimation},
  author={Mehl, Lukas and Jahedi, Azin and Schmalfuss, Jenny and Bruhn, Andr{\'e}s},
  booktitle={Proc. Winter Conference on Applications of Computer Vision (WACV)},
  year={2023}
}
```

Code Overview:
- `data_readers`: code related to data handling
- `mfuse`: model definitions
- `scripts`: scripts for training, evaluation, submission

# Setup
- Install all required python packages:
pytorch, numpy, scipy, opencv, tqdm, scikit-sparse, pypng
- Install the lietorch package. See https://github.com/princeton-vl/lietorch for details.

The code was tested with Python 3.9, PyTorch 1.10.2, CUDA 11.6

## Datasets
Download the KITTI scene flow dataset with the multi-frame extension from
http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php
and make sure that it is located in the directory `$DATASETS/kitti15` where $DATASETS is an environment variable.

Create disparity files for KITTI using the LEAStereo code
https://github.com/XuelianCheng/LEAStereo
and put them into `$DATASETS/kitti15/training/disp_lea` and `$DATASETS/kitti15/testing/disp_lea` respectively.
You can also download precomputed results of LEAStereo for the [train](https://bwsyncandshare.kit.edu/s/z3ZqKp5g3ZpiYJr) and [testing](https://bwsyncandshare.kit.edu/s/5tN3CQtgfN9QTFc) split.

# Usage
After training M-FUSE on the KITTI dataset for 50K steps, results can be evaluated using
```
python scripts/evaluation_fusion.py --model=<path-to-checkpoint>.pth
```

A submission for the KITTI benchmark can be created using
```
python scripts/kitti_submission_fusion.py --model=<path-to-checkpoint>.pth
```

Our resulting checkpoint can be downloaded [here](https://bwsyncandshare.kit.edu/s/RTZ4H9pCNfcgAM3), which yields an SF-all error of 4.83 on the KITTI benchmark.

# Training
1. Retrain the RAFT-3D model `raft3d_bilaplacian` on the FlyingThings3D dataset for 200K steps with their provided code https://github.com/princeton-vl/RAFT-3D or use their checkpoint.

2. Train our fusion model:
```
python scripts/train_fusion.py --ckpt_r3d=<path-to-pretrained-r3d>
```
