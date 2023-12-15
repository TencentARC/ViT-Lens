# 🧩 Model Zoo

We are keeping updating models for ViT-Lens, please stay tuned!

## 3D Point Cloud
|                            Model                             |        Training Data        | MN40 (Top1/Top3/Top5) | Objaverse-LVIS (Top1/Top3/Top5) | ScanObjectNN (Top1/Top3/Top5) |
| :----------------------------------------------------------: | :-------------------------: | :-------------------: | :-----------------------------: | :---------------------------: |
|                           vitlensB                           |   ULIP-ShapeNet Triplets    |      65.4/-/92.7      |                -                |               -               |
|                           vitlensB                           |  ULIP2-Objaverse Triplets   |      74.8/-/93.8      |                -                |               -               |
| [vitlensL](https://huggingface.co/TencentARC/ViT-Lens/blob/main/Datacomp_L14_ShapeNet.pt) |   ULIP-ShapeNet Triplets    |      70.6/-/94.4      |                -                |               -               |
| [vitlensL](https://huggingface.co/TencentARC/ViT-Lens/blob/main/Datacomp_L14_objaverse.pt) |  ULIP2-Objaverse Triplets   |      80.6/-/95.8      |                -                |               -               |
| [vitlensG](https://huggingface.co/TencentARC/ViT-Lens/blob/main/bigG14_sk16_openshape_all.pt) |     OpenShape-Triplets      |    87.6/96.6/98.4     |         52.0/73.3/79.9          |        60.1/81.0/90.3         |
| [vitlensG](https://huggingface.co/TencentARC/ViT-Lens/blob/main/bigG14_openshape_nolvis.pt) | OpenShape-Triplets(No LVIS) |    86.8/96.8/97.8     |         50.1/71.3/78.1          |        59.8/79.3/87.7         |

## Depth
|                            Model                             | Training Data  | SUN.D (Top1) | NYU.D (Top1) |
| :----------------------------------------------------------: | :------------: | :----------: | :----------: |
|                           vitlensB                           | SUN RGBD (I+T) |     51.4     |     65.0     |
| [vitlensL](https://huggingface.co/TencentARC/ViT-Lens/blob/main/vitlensL_depth.pt) | SUN RGBD (I+T) |     52.2     |     68.5     |
|                           vitlensG                           | SUN RGBD (I+T) |     54.6     |     69.0     |


## Audio

|                            Model                             |                       Training Data                       | Audioset (mAP) | VGGSound (Top1) | ESC50 (Top1) | Clotho (R@1/R@10) | AudioCaps (R@1/R@10) |
| :----------------------------------------------------------: | :-------------------------------------------------------: | :------------: | :-------------: | :----------: | :---------------: | :------------------: |
|                           vitlensB                           |            Audioset `train`, 5-sec clips (V+T)            |      26.3      |      29.9       |     72.9     |     7.5/29.5      |      13.5/54.1       |
| [vitlensL](https://huggingface.co/TencentARC/ViT-Lens/blob/main/vitlensL_audio.pt) |            Audioset `train`, 5-sec clips (V+T)            |      26.7      |      31.7       |     75.9     |     8.1/31.2      |      14.4/54.9       |
|                           vitlensL                           |            Audioset `train`, 2-sec clips (V+T)            |      29.0      |      32.5       |     75.1     |     7.9/31.6      |      14.8/53.3       |
|                           vitlensL                           | Audioset `train` and VGGSound `train` , 5-sec clips (V+T) |      27.2      |      51.7       |     80.9     |     7.9/31.5      |      14.9/55.2       |



## Tactile
|                            Model                             | Training Data | Material (Top1) | Hard/Soft (Top1) | Rough/Smooth (Top1) |
| :----------------------------------------------------------: | :-----------: | :-------------: | :--------------: | :-----------------: |
|           vitlensB(aligned to Image) - LinearProbe           | Touch-and-Go  |      63.0       |       92.0       |        85.1         |
| [vitlensL](https://huggingface.co/TencentARC/ViT-Lens/blob/main/vitlensL_tactile.pt) | Touch-and-Go  |      65.8       |       74.7       |        63.8         |


## EEG
|                            Model                             | Training Data | INEEG-Val (Top1) | INEEG-Test (Top1) |
| :----------------------------------------------------------: | :-----------: | :-------: | :--------: |
|                           vitlensB                           | ImageNet EEG  |   37.3    |    35.9    |
| [vitlensL](https://huggingface.co/TencentARC/ViT-Lens/blob/main/vitlensL_eeg.pt) | ImageNet EEG  |   41.8    |    42.7    |