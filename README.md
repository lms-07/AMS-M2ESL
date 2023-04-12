# Adaptive Mask Sampling and Manifold to Euclidean Subspace Learning with Distance Covariance Representation for Hyperspectral Image Classification

[Mingsong Li](https://orcid.org/0000-0001-6133-3923), [Wei Li](https://fdss.bit.edu.cn/yjdw/js/b153191.htm), Yikun Liu, [Yuwen Huang](https://jsj.hezeu.edu.cn/info/1302/6525.htm), and [Gongping Yang](https://faculty.sdu.edu.cn/gpyang)

[Time Lab](https://time.sdu.edu.cn/), [SDU](https://www.sdu.edu.cn/) ; [BIT](https://www.bit.edu.cn/)

-----------
This repository is the official implementation of our paper:
[Adaptive Mask Sampling and Manifold to Euclidean Subspace Learning with Distance Covariance Representation for Hyperspectral Image Classification](https://doi.org/10.1109/TGRS.2023.3265388), IEEE TGRS 2023.

## Contents
1. [Brief Introduction](#Brief-Introduction)
1. [Environment](#Environment)
1. [Data Sets](#Data-Sets)
1. [Citation](#Citation)
1. [License and Acknowledgement](License-and-Acknowledgement)

## Brief Introduction
> <p align="left">For the abundant spectral and spatial information recorded in hyperspectral images (HSIs), fully exploring spectral-spatial relationships has attracted widespread attention in hyperspectral image classification (HSIC) community. However, there are still some intractable obstructs. For one thing, in the patch-based processing pattern, some spatial neighbor pixels are often inconsistent with the central pixel in land-cover class. For another thing, linear and nonlinear correlations between different spectral bands are vital yet tough for representing and excavating. To overcome these mentioned issues, an adaptive mask sampling and manifold to Euclidean subspace learning (AMS-M2ESL) framework is proposed for HSIC. Specifically, an adaptive mask based intra-patch sampling (AMIPS) module is firstly formulated for intra-patch sampling in an adaptive mask manner based on central spectral vector oriented spatial relationships. Subsequently, based on distance covariance  descriptor, a dual channel distance covariance representation (DC-DCR) module is proposed for modeling unified spectral-spatial feature representations and exploring spectral-spatial relationships, especially linear and nonlinear interdependence in spectral domain. Furthermore, considering that distance covariance matrix lies on the symmetric positive definite (SPD) manifold, we implement a manifold to Euclidean subspace learning (M2ESL) module respecting Riemannian geometry of SPD manifold for high-level spectral-spatial feature learning. Additionally, we introduce an approximate matrix square-root (ASQRT) layer for efficient Euclidean subspace projection. Extensive experimental results on three popular HSI data sets with limited training samples demonstrate the superior performance of the proposed method compared with other state-of-the-art methods. The source code is available at https://github.com/lms-07/AMS-M2ESL.</p>

|                   AMS-M2ESL Framwork
| :-----------------------------------------: |
| <img src="./src/framework.pdf"  >  |

## Environment
- The software environment is Ubuntu 18.04.5 LTS 64 bit.
- This project is running on a single Nvidia GeForce RTX 3090 GPU based on Cuda 11.0.
- We adopt Python 3.8.5, PyTorch 1.10.0+cu111.
- The py+torch combination may not limietd by our adopted one.


## Data Sets

Three representative HSI data sets are adopted in our experiments, i.e., Indian Pines (IP), University of Pavia (UP), and University of Houston 13 (UH).
The first two data sets could be access through [link1](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes##anomaly_detection),
and the UH data set through [link2](https://hyperspectral.ee.uh.edu/?page_id=459).
Our project is organized as follows:

```text
CVSSN
|-- process_cls_xxx         // main files 
|-- data                    
|   |-- IP
|   |   |-- Indian_pines_corrected.mat
|   |   |-- Indian_pines_gt.mat
|   |-- UP
|   |   |-- PaviaU.mat
|   |   |-- PaviaU_gt.mat
|   |-- HU13_tif
|   |   |--Houston13_data.mat
|   |   |--Houston13_gt_train.mat
|   |   |--Houston13_gt_test.mat
|-- model                   // the compared methodes and our proposed method
|-- output
|   |-- cls_maps            // classification map visualizations 
|   |-- results             // classification result files
|-- src                     // source files
|-- utils                   // data loading, processing, and evaluating
|-- visual                  // cls maps visual
```

## Citation

Please kindly cite our work if this work is helpful for your research.

[1] M. Li, W. Li, Y. Liu, Y. Huang and G. Yang, "Adaptive Mask Sampling and Manifold to Euclidean Subspace Learning with Distance Covariance Representation for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2023.3265388.

BibTex entry:
```text
@article{li2023adaptive,
  title={Adaptive Mask Sampling and Manifold to Euclidean Subspace Learning with Distance Covariance Representation for Hyperspectral Image Classification},
  author={Li, Mingsong and Li, Wei and Liu, Yikun and Huang, Yuwen and Yang, Gongping},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2023},
  doi={10.1109/TGRS.2023.3265388},
  publisher={IEEE},
}
```

## Contact information

If you have any problem, please do not hesitate to contact us `msli@mail.sdu.edu.cn`.

## License and Acknowledgement
This project is released under [GPLv3](http://www.gnu.org/licenses/) license.

- We would like to thank the Hyperspectral Image Analysis group and the NSF Funded Center for
  Airborne Laser Mapping (NCALM) at the University of Houston for providing the UH data set used in this work.
- Part of our HSIC framework is referred to [HybridSN](https://github.com/gokriznastic/HybridSN), [A2S2K-ResNet](https://github.com/suvojit-0x55aa/A2S2K-ResNet), and [CNN_Enhanced_GCN](https://github.com/qichaoliu/CNN_Enhanced_GCN). Please also follow their licenses. Thanks for their awesome works.
- Among the adopted compared methods, we also would like to thank Assistant Professor [Xiangtao Zheng](https://xiangtaozheng.github.io/) and
  Dr. Xuming Zhang for providing the source tensorflow code of [SSAN](https://ieeexplore.ieee.org/document/8909379) and
  the part of source keras code of [SSSAN](https://ieeexplore.ieee.org/document/9508777?arnumber=9508777), respectively.
  
