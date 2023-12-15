# 📦 Datasets

## 3D Point Cloud
### Training Data
- ULIP-ShapeNet Triplets: follow [ULIP](https://github.com/salesforce/ULIP) to download and prepare the data.
- ULIP2-Objaverse Triplets: follow [ULIP](https://github.com/salesforce/ULIP) to download and prepare the data.
- OpenShape Triplets: follow [OpenShape](https://github.com/Colin97/OpenShape_code) to download and prepare the data.

### Downstream Data
- ModelNet40: we prepare two versions following [ULIP](https://github.com/salesforce/ULIP) (when training models on ULIP-ShapeNet Triplets and ULIP2-Objaverse Triplets) and [OpenShape](https://github.com/Colin97/OpenShape_code) (when training models on OpenShape Triplets).
- ScanObjectNN: follow [OpenShape](https://github.com/Colin97/OpenShape_code) to download and prepare the data.
- Objaverse LVIS: follow [OpenShape](https://github.com/Colin97/OpenShape_code) to download and prepare the data.

We provide meta data for 3D point cloud datasets in [vitlens/src/open_clip/modal_3d/data](vitlens/src/open_clip/modal_3d/data).


## Depth
### Training Data
- SUN-RGBD: We use the SUN-RGBD (`train` split) for training. Download the data from this [website](https://rgbd.cs.princeton.edu/) through this [link](https://rgbd.cs.princeton.edu/data/SUNRGBD.zip).

### Downstream Data
- SUN-RGBD: We use the SUN-RGBD (`test` split) for testing. Download the data from this [website](https://rgbd.cs.princeton.edu/) through this [link](https://rgbd.cs.princeton.edu/data/SUNRGBD.zip).
- NYUv2: We use the NYUv2 (`test` split) for testing. Download the data from this [website](https://rgbd.cs.princeton.edu/) through this [link](https://rgbd.cs.princeton.edu/data/SUNRGBD.zip). Use the NYU data in the downloaded dataset.

We provide meta data for RGBD/Depth datasets in [vitlens/src/open_clip/modal_depth/data](vitlens/src/open_clip/modal_depth/data).


## Audio
### Training data
- Audioset: We use the training splits of the Audioset for training. We download the data according to the meta data provided in the [official website](https://research.google.com/audioset/download.html). Since some videos are no longer available, we do not obtain all the videos listed. We list the videos used in our experiments in [vitlens/src/open_clip/modal_audio/data/audioset_*.json](vitlens/src/open_clip/modal_audio/data).
- VGGSound: In our later experiments, we combind VGGSound(`train` split) and Audioset for training. We download the data according to the meta data provided in [this page](https://github.com/hche11/VGGSound/tree/master/data). Similar to Audioset, some videos are no longer available, we do not obtain all the videos listed. We list the videos used in our experiments in [vitlens/src/open_clip/modal_audio/data/vggsound_*.json](vitlens/src/open_clip/modal_audio/data).

### Downstream data
- Audioset: We use the `val` split for testing. We list the videos used in our experiments in [audioset_val.json](vitlens/src/open_clip/modal_audio/data/audioset_val.json).
- VGGSound: We use the `val` split for testing. We list the videos used in our experiments in [vggsound_audio-only_val.json](vitlens/src/open_clip/modal_audio/data/vggsound_audio-only_val.json). The audio data could be obtained from this [link](https://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/vggsound.zip) provided by [ONE-PEACE](https://github.com/OFA-Sys/ONE-PEACE/blob/main/datasets.md).
- ESC: We use all the ESC data for testing. We download the data through this [link](https://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/esc50.zip) provided by [ONE-PEACE](https://github.com/OFA-Sys/ONE-PEACE/blob/main/datasets.md).
- Clotho: We use the `eval/val` split for testing. We download the data through this [link](https://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/clotho.zip) provided by [ONE-PEACE](https://github.com/OFA-Sys/ONE-PEACE/blob/main/datasets.md).
- AudioCaps. We use the `test` split provided in this [link](https://www.robots.ox.ac.uk/~vgg/research/audio-retrieval/resources/benchmark-files/AudioCaps_retrieval_dataset.tar.gz). We download the data through this [link](https://one-peace-shanghai.oss-accelerate.aliyuncs.com/one_peace_datasets/audiocaps.zip) provided by [ONE-PEACE](https://github.com/OFA-Sys/ONE-PEACE/blob/main/datasets.md), and use the data listed in the downloaded split.

We provide meta data for audio datasets in [vitlens/src/open_clip/modal_audio/data](vitlens/src/open_clip/modal_audio/data).


## Tactile
### Training data
- Touch-and-Go: We use the `train` split of the Touch-and-Go dataset for training. We download the data following the [official website](https://touch-and-go.github.io/).

### Downstream data
- Touch-and-Go: We use `test-material`, `test-hard/soft`, `test-rough/smooth` splits for testing. We download the data following the [official website](https://touch-and-go.github.io/).

We provide meta data for audio datasets in [vitlens/src/open_clip/modal_tactile/data](vitlens/src/open_clip/modal_tactile/data).


## EEG
### Training data
- ImageNet EEG: We use the `train` split in ImageNet EEG dataset for training. We follow this [website](https://github.com/perceivelab/eeg_visual_classification?tab=readme-ov-file) to download the EEG data. We download the corresponding images following this [page](https://github.com/bbaaii/DreamDiffusion/tree/main?tab=readme-ov-file).

### Downstream data
- ImageNet EEG: We use `val/test` splits in ImageNet EEG for testing. We follow the training split to obtain the data.

For spliting used in experiments, please refer to [datasets.py](vitlens/src/open_clip/modal_eeg/datasets.py).