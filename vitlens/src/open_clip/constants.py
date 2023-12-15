import os
from types import SimpleNamespace

ModalityType = SimpleNamespace(
    IMAGE="image",
    VIDEO="video",
    TEXT="text",
    AUDIO="audio",
    DEPTH="depth",
    EEG="eeg",
    TACTILE="tactile",
    PC="pc",
)

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

# project dir
PROJECT_DIR = "/PATH/TO/ViT-Lens"                   

# checkpoint cache dir, used to store CLIP ckpt or other models
CKPT_CACHE_DIR = "/PATH_TO/CACHE/DIR"


# data related dir, `*_DATA_DIR` for training/testing data, `*_META_DATA_DIR` for meta data
OBJAVERSE_DATA_DIR = "/PATH_TO/3d/ulip_batches"
PC_DATA_DIR = "/PATH_TO/3d"
PC_META_DATA_DIR = "/PATH_TO/vitlens/src/open_clip/modal_3d/data"

AUDIO_DATA_DIR = "/PATH_TO/audio_datasets"
AUDIO_META_DATA_DIR = (
    "/PATH_TO/vitlens/src/open_clip/modal_audio/data"
)

DEPTH_DATA_DIR = "/PATH_TO/SUNRGBD"
DEPTH_META_DATA_DIR = (
    "/PATH_TO/vitlens/src/open_clip/modal_depth/data"
)

TACTILE_DATA_DIR = "/PATH_TO/touch_and_go/dataset"
TACTILE_META_DATA_DIR = (
    "/PATH_TO/vitlens/src/open_clip/modal_tactile/data"
)

EEG_DATA_DIR = "/PATH_TO/EEG"
EEG_META_DATA_DIR = "/PATH_TO/vitlens/src/open_clip/modal_eeg/data"
