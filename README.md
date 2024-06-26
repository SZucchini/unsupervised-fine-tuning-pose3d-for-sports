# [Pseudo-label based unsupervised fine-tuning of a monocular 3D pose estimation model for sports motions](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Suzuki_Pseudo-label_Based_Unsupervised_Fine-tuning_of_a_Monocular_3D_Pose_Estimation_CVPRW_2024_paper.html)
This is an official implementation of "[Pseudo-label based unsupervised fine-tuning of a monocular 3D pose estimation model for sports motions](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/html/Suzuki_Pseudo-label_Based_Unsupervised_Fine-tuning_of_a_Monocular_3D_Pose_Estimation_CVPRW_2024_paper.html)" at CVsports 2024.

<p align="center"><img src="fig/demo.gif" width="55%" alt="" /></p>

## Overview
We are preparing the dataset and model weights now. Please wait for a while.

This repository contains:
- unsupervised fine-tuning scripts for our original runner dataset and ASPset-510
- our original runner dataset and pre-processed ASPset-510

## Getting Started
> [!NOTE]
> For the ASPset-510, due to the type and number of GPUs and random numbers, it is difficult to perfectly reproduce the results in the paper, but we have already confirmed that the results show the same trend as in the paper after multiple training runs.

### Installation
0. Prerequisites
Before running this project, make sure you have the following requirements
- Python 3.8 and CUDA 11.3: The author has tested the experiments using Python 3.8.18 and CUDA 11.3.
- Poetry: This project uses Poetry to manage dependencies. Install Poetry before using this repository.

1. Clone this repository:
```
$ git clone https://github.com/SZucchini/unsupervised-fine-tuning-pose3d-for-sports.git
```

2. Install dependencies:
```
$ cd unsupervised-fine-tuning-pose3d-for-sports
$ poetry install
```

### Unsupervised fine-tuning

#### Prepare datasets and checkpoints

##### Runner Dataset
1. Download the runner dataset from the [Google Drive](https://drive.google.com/drive/folders/11OmaW8vQ7rz4mMKDgbkPP7fGDWSSJQaW?usp=sharing).
2. Place the dataset in the `./data` directory.
3. Download the pre-trained MotionAGFormer weights from [this repository](https://github.com/TaatiTeam/MotionAGFormer?tab=readme-ov-file#evaluation). We used MotionAGFormer-xs for the runner dataset.
4. Place the pre-trained weights in the `./common/MotionAGFormer/checkpoint` directory.

##### ASPset-510
> [!NOTE]
> Original ASPset-510 is available at [here](https://archive.org/details/aspset510). We provide the pre-processed ASPset-510 (ASP-27) for the convenience of the user. The pre-processing code will be released soon.

1. Download the pre-processed ASPset-510 (ASP-27) from the [Google Drive](https://drive.google.com/drive/folders/11OmaW8vQ7rz4mMKDgbkPP7fGDWSSJQaW?usp=sharing).
2. Place the dataset in the `./data` directory.
3. Download the our original scale augmented MotionAGFormer weights (`scale_augment_best.pth.tr`) from the [Google Drive](https://drive.google.com/drive/folders/11OmaW8vQ7rz4mMKDgbkPP7fGDWSSJQaW?usp=sharing).
4. Place the pre-trained weights in the `./common/MotionAGFormer/checkpoint` directory.

#### Run the unsupervised fine-tuning
Run the following command to start the unsupervised fine-tuning.
- Runner Dataset
```
$ poetry run python ./run.py \
    --config ./config/runner/MotionAGFormer-xs.yaml \
    --workspace ./workspace/your_work_space
```
- ASPset-510
```
$ poetry run python ./run_asp.py \
    --config ./config/asp/MotionAGFormer-pre.yaml \
    --workspace ./workspace/your_work_space
```

### Fine-tuned checkpoints
You can download the fine-tuned model weights with each dataset from the link.

- Runner Dataset: [Google Drive](https://drive.google.com/drive/folders/1Bj5V-V_tk3IWaEWZIPSS92OmsgT3E9Tc?usp=sharing)
- ASPset-510: [Google Drive](https://drive.google.com/drive/folders/1mcTQt519KEGRvBGW4BT-WfkSpc7P7Yf-?usp=sharing)


## Citation
If our work is useful for your project, please consider citing the paper:
```
@InProceedings{Suzuki_2024_CVPR,
    author    = {Suzuki, Tomohiro and Tanaka, Ryota and Takeda, Kazuya and Fujii, Keisuke},
    title     = {Pseudo-label Based Unsupervised Fine-tuning of a Monocular 3D Pose Estimation Model for Sports Motions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {3315-3324}
}
```

## Acknowledgements
We appreciate the following repositories:
- [kyotovision-public/extrinsic-camera-calibration-from-a-moving-person](https://github.com/kyotovision-public/extrinsic-camera-calibration-from-a-moving-person)
- [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose)
- [TaatiTeam/MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)
- [ViTAE-Transformer/ViTPose](ViTAE-Transformer/ViTPose)
- [anibali/aspset-510](https://github.com/anibali/aspset-510)

## License
- This software is created under multiple licenses.
- We write the license of each script in the header of the script.
- Please refer to the each script for the details.

## Contact
If you have any questions, please contact author:
- Tomohiro Suzuki (suzuki.tomohiro[at]g.sp.m.is.nagoya-u.ac.jp)
