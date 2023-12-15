import os
import random
import json
import logging

import torch
import torch.nn as nn

from typing import Iterable
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from omegaconf import OmegaConf
from easydict import EasyDict as edict

from open_clip.util.Sample import Sample, SampleCollator
from open_clip.modal_eeg.processors.eeg_processor import (
    Image_Processor_Train,
    Image_Processor_Eval,
    BlipCaptionProcessor,
    EEG_Processor,
)
from open_clip.constants import EEG_DATA_DIR, EEG_META_DATA_DIR


class BaseDataset(Dataset):
    def __init__(
        self,
        vis_processor=None,
        eeg_processor=None,
        text_processor=None,
        data_root=None,
        anno_path={},
        args=None,
        split=None,
        tokenizer=None,
        **kwargs,
    ):
        self.data_root = data_root
        self.anno_path = anno_path
        self.annotation = None

        self.vis_processor = vis_processor
        self.eeg_processor = eeg_processor
        self.text_processor = text_processor

        self.args = args
        self.split = split

        self.tokenizer = tokenizer
        self.data_type = None

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return SampleCollator(self, samples)

    def set_processors(self, vis_processor, eeg_processor, text_processor):
        self.vis_processor = vis_processor
        self.eeg_processor = eeg_processor
        self.text_processor = text_processor


class EEGDataset(BaseDataset):
    def __init__(
        self,
        vis_processor=None,
        eeg_processor=None,
        text_processor=None,
        data_root=f"{EEG_DATA_DIR}/imageNet_images",
        anno_path={
            "data": f"{EEG_DATA_DIR}/eeg_5_95_std.pth",
            "split_info": f"{EEG_DATA_DIR}/block_splits_by_image_all.pth",
        },
        args=None,
        split=None,
        tokenizer=None,
        split_num=0,
        n_repeat_train=50,
        **kwargs,
    ):
        super().__init__(
            vis_processor,
            eeg_processor,
            text_processor,
            data_root,
            anno_path,
            args,
            split,
            tokenizer,
            **kwargs,
        )

        data = torch.load(anno_path["data"])
        self.dataset = data["dataset"]
        self.synset_labels = data["labels"]
        self.image_list = data["images"]

        self.split_info = torch.load(anno_path["split_info"])
        self.imagenet_mapping = json.load(
            open(f"{EEG_META_DATA_DIR}/imagenet_cls_mapping.json", "r")
        )

        self.split_indices = self.split_info["splits"][split_num][self.split]

        # filter data
        self.split_indices = [
            i
            for i in self.split_indices
            if 450 <= self.dataset[i]["eeg"].size(1) <= 600
        ]
        if self.split in ["train", "pretrain"]:
            self.split_indices = self.split_indices * n_repeat_train

        self.init_labels()

    def init_labels(self):
        self.idx2label = [self.imagenet_mapping[i][0] for i in self.synset_labels]
        self.label2idx = {self.idx2label[i]: i for i in range(len(self.idx2label))}

    def __len__(self):
        return len(self.split_indices)

    def __getitem__(self, index):
        fetch_idx = self.split_indices[index]
        datum = self.dataset[fetch_idx]

        eeg_signal = datum["eeg"]
        eeg = self.eeg_processor(eeg_signal)

        image_name = self.image_list[datum["image"]]
        image_path = os.path.join(
            self.data_root, image_name.split("_")[0], image_name + ".JPEG"
        )
        image = self.vis_processor(image_path)

        label = datum["label"]
        synset_label = self.synset_labels[label]
        cls_name = None
        if self.split in ["train", "pretrain"]:
            cls_name = random.choice(self.imagenet_mapping[synset_label])
        else:
            cls_name = self.imagenet_mapping[synset_label][0]

        caption = "an image of {}.".format(cls_name)
        caption = self.text_processor(caption)
        tokenized_caption = self.tokenizer([caption])[0]

        return Sample(
            {
                "image": image,
                "label": torch.tensor(label).long(),
                "synset_label": synset_label,
                "eeg": eeg,
                "caption": tokenized_caption,
            }
        )


name2dataset = {
    "eeg": EEGDataset,
}


def create_eeg_datasets(
    args, is_train, tokenizer, mean=None, std=None, image_transform=None
):
    dataset_names = args.train_data if is_train else args.val_data
    dataset_names = dataset_names.split("::")
    # format: "eeg@train::eeg@test"

    datasets = dict()
    for dset_ns in dataset_names:
        # e.g. "audioset@unbalanced-train"
        ns = dset_ns.split("@")
        assert len(ns) == 2
        name, specific_split = ns[0], ns[1]
        dataset_cls = name2dataset[name]
        conf = OmegaConf.create({})

        vis_proc_cls = Image_Processor_Train if is_train else Image_Processor_Eval
        eeg_proc_cls = EEG_Processor

        vis_processor = vis_proc_cls.from_config(conf)
        text_processor = BlipCaptionProcessor.from_config()
        eeg_processor = eeg_proc_cls.from_config(conf)

        dataset = dataset_cls(
            vis_processor=vis_processor,
            eeg_processor=eeg_processor,
            text_processor=text_processor,
            args=args,
            split=specific_split,
            tokenizer=tokenizer,
            image_transform=image_transform,
        )

        datasets[dset_ns] = dataset

    datasets_list = [v for _, v in datasets.items()]
    if len(datasets) > 1 and is_train:
        return concat_datasets(datasets_list)
    elif len(datasets) == 1 and is_train:
        return datasets_list[0]
    elif len(datasets) == 1 and not is_train:
        return datasets_list[0]
    else:
        return datasets
