# PH-Mamba
Implementation of PH-Mamba.
- ph_mamba.py: main network structure.
- mamba_sys.py: VMamba-based component.
- css_module.py: CSS(Coarse Segmentation aSsistance, CSA in our paper) related function.

## Envs
- CUDA 12.2
- Pytorch 2.1.0
- causal-conv1d 1.0.0
- mamba-ssm from mzusman/mamba (2.2.2)

## Details:
[mzusman/mamba](https://github.com/mzusman/mamba/tree/chunked_mamba) provides the CUDA code which can prefill state to Mamba SSM. Thank you so much!
The official says this feature is on the way, and we hope that day comes soon.


## Citation
Refer to the following, thanks!

```bibtex
@article{wang2024mamba,
  title={Mamba-unet: Unet-like pure visual mamba for medical image segmentation},
  author={Wang, Ziyang and Zheng, Jian-Qing and Zhang, Yichi and Cui, Ge and Li, Lei},
  journal={arXiv preprint arXiv:2402.05079},
  year={2024}
},
@article{liu2024vmamba,
  title={VMamba: Visual State Space Model},
  author={Liu, Yue and Tian, Yunjie and Zhao, Yuzhong and Yu, Hongtian and Xie, Lingxi and Wang, Yaowei and Ye, Qixiang and Liu, Yunfan},
  journal={arXiv preprint arXiv:2401.10166},
  year={2024}
},
@InProceedings{swinunet,
author = {Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
title = {Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation},
booktitle = {Proceedings of the European Conference on Computer Vision Workshops(ECCVW)},
year = {2022}
},
```
