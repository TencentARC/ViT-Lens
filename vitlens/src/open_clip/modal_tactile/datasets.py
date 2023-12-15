import os
import random
import json
import logging
import torch

from typing import Iterable
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from omegaconf import OmegaConf
from easydict import EasyDict as edict

from open_clip.util.Sample import Sample, SampleCollator
from open_clip.modal_tactile.processors.tact_processor import (
    Image_Processor_Train,
    Image_Processor_Eval,
    Tactile_Processor_Train,
    Tactile_Processor_Eval,
    BlipCaptionProcessor,
)
from open_clip.constants import TACTILE_DATA_DIR, TACTILE_META_DATA_DIR


class BaseDataset(Dataset):
    def __init__(
        self,
        vis_processor=None,
        tactile_processor=None,
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
        self.tactile_processor = tactile_processor
        self.text_processor = text_processor

        self.args = args
        self.split = split

        self.tokenizer = tokenizer
        self.data_type = None

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return SampleCollator(self, samples)

    def set_processors(self, vis_processor, tactile_processor, text_processor):
        self.vis_processor = vis_processor
        self.tactile_processor = tactile_processor
        self.text_processor = text_processor


class TactileBaseDataset(Dataset):
    def __init__(
        self,
        vis_processor=None,
        tactile_processor=None,
        text_processor=None,
        data_root=None,
        anno_path={},
        args=None,
        split=None,
        tokenizer=None,
        image_transform=None,
        **kwargs,
    ):
        self.data_root = data_root
        self.anno_path = anno_path
        self.annotation = None

        self.vis_processor = vis_processor
        self.tactile_processor = tactile_processor
        self.text_processor = text_processor

        self.args = args
        self.split = split
        self.tokenizer = tokenizer
        self.image_transform = image_transform

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return SampleCollator(self, samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __getitem__(self, index):
        rtn = dict()
        return Sample(rtn)


class TAGDataset(TactileBaseDataset):
    def __init__(
        self,
        vis_processor=None,
        tactile_processor=None,
        text_processor=None,
        data_root=TACTILE_DATA_DIR,
        anno_path={
            "pretrain": f"{TACTILE_META_DATA_DIR}/pretrain.json",
            "train_material": f"{TACTILE_META_DATA_DIR}/train.json",
            "test_material": f"{TACTILE_META_DATA_DIR}/test.json",
            "train_hard": f"{TACTILE_META_DATA_DIR}/train.json",
            "test_hard": f"{TACTILE_META_DATA_DIR}/test.json",
            "train_rough": f"{TACTILE_META_DATA_DIR}/train_rough.json",
            "test_rough": f"{TACTILE_META_DATA_DIR}/test_rough.json",
            "pretrain_exclude_others": f"{TACTILE_META_DATA_DIR}/pretrain_exclude_others.json",
            "test_material_exclude_others": f"{TACTILE_META_DATA_DIR}/test_exclude_others.json",
        },
        args=None,
        split=None,
        tokenizer=None,
        image_transform=None,
        n_repeat_train=40,
        **kwargs,
    ):
        super().__init__(
            vis_processor,
            tactile_processor,
            text_processor,
            data_root,
            anno_path,
            args,
            split,
            tokenizer,
            image_transform,
            **kwargs,
        )

        self.annotation = json.load(open(anno_path[split], "r"))
        if split == "pretrain" and n_repeat_train > 1:
            print(f" ***  Repeat dataset {n_repeat_train} times.  ***")
            self.annotation = self.annotation * n_repeat_train
        elif "train" in split:  # linear probe train
            self.annotation = self.annotation * args.lp_train_n_repeat

        self.init_labels()

    def init_labels(self):
        self.idx2label = None
        self.label2idx = None
        if "material" in self.split:
            self.idx2label = [
                "concrete",
                "plastic",
                "glass",
                "wood",
                "metal",
                "brick",
                "tile",
                "leather",
                "synthetic fabric",
                "natural fabric",
                "ruber",
                "paper",
                "tree",
                "grass",
                "soil",
                "rock",
                "gravel",
                "sand",
                "plants",
                "others",
            ]
            if "exclude_others" in self.split:
                self.idx2label = self.idx2label[:-1]  # exclude "others" class
            self.label2idx = {self.idx2label[i]: i for i in range(len(self.idx2label))}

        elif "hard" in self.split:
            self.idx2label = ["hard", "soft"]
            self.label2idx = {"hard": 0, "soft": 1}

        elif "rough" in self.split:
            self.idx2label = ["smooth", "rough"]
            self.label2idx = {"smooth": 0, "rough": 1}

        else:
            pass

    def __getitem__(self, index):
        rtn = dict()
        """
          {
            "gel_path": "20220601_182052/gelsight_frame/0000033833.jpg",
            "image_path": "20220601_182052/video_frame/0000033833.jpg",
            "material_label": 3,
            "material_name": "wood",
            "sr_label": null,
            "sr_name": null,
            "hs_label": 0,
            "hs_name": "hard"
        }
        """
        ann = self.annotation[index]
        img_path = os.path.join(self.data_root, ann["image_path"])
        gel_path = os.path.join(self.data_root, ann["gel_path"])
        rgb = self.vis_processor(img_path)
        gel = self.tactile_processor(gel_path)
        if self.args.use_openclip_transform:
            rgb = Image.open(img_path).convert("RGB")
            rgb = self.image_transform(rgb)

        # material
        m_label = ann["material_label"]
        if m_label is not None:
            # assert m_label >= 0
            pass

        m_name = ann["material_name"]
        tokenized_caption = None
        if m_name is not None:
            caption = "an image of {}.".format(m_name)
            caption = self.text_processor(caption)
            tokenized_caption = self.tokenizer([caption])[0]
        else:
            caption = "an image showing a material."  # dummy
            tokenized_caption = self.tokenizer([caption])[0]

        label = None
        if "rough" in self.split:
            label = ann["sr_label"]
        elif "hard" in self.split:
            label = ann["hs_label"]
        else:
            label = ann["material_label"]
        label = torch.tensor(label).long()

        rtn = {
            "image": rgb,
            "tactile": gel,
            "id": f"{index}_{img_path}",
            "label": label,
            "caption": tokenized_caption,
        }
        return Sample(rtn)


name2dataset = {
    "tag": TAGDataset,
}


def create_tactile_datasets(
    args, is_train, tokenizer, mean=None, std=None, image_transform=None
):
    dataset_names = args.train_data if is_train else args.val_data
    dataset_names = dataset_names.split("::")
    # format: "tag@pretrain::tag@test"

    datasets = dict()
    for dset_ns in dataset_names:
        # e.g. "audioset@unbalanced-train"
        ns = dset_ns.split("@")
        assert len(ns) == 2
        name, specific_split = ns[0], ns[1]
        dataset_cls = name2dataset[name]
        conf = OmegaConf.create({})

        vis_proc_cls = Image_Processor_Train if is_train else Image_Processor_Eval
        tactile_proc_cls = (
            Tactile_Processor_Train if is_train else Tactile_Processor_Eval
        )

        vis_processor = vis_proc_cls.from_config(conf)
        text_processor = BlipCaptionProcessor.from_config()
        tactile_processor = tactile_proc_cls.from_config(conf)

        dataset = dataset_cls(
            vis_processor=vis_processor,
            tactile_processor=tactile_processor,
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
