import os
import csv
import random
import json
import logging
import pandas as pd
import numpy as np
import einops

from typing import Iterable
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset
from omegaconf import OmegaConf
from easydict import EasyDict as edict

from open_clip.util.Sample import Sample, SampleCollator
from open_clip.modal_audio.processors.at_processor import (
    PVProcessorTrain,
    PVProcessorEval,
    ASTProcessorTrain,
    ASTProcessorEval,
    CLAPProcessorTrain,
    CLAPProcessprEval,
    BlipCaptionProcessor,
)
from open_clip.modal_audio.data.sound_cls_template import SOUND_AS_IMAGE_TEMPLATE
from open_clip.constants import AUDIO_DATA_DIR, AUDIO_META_DATA_DIR


def extract_sound_description(input_string):
    prefixes = [
        "the sound of",
        "the sounds of",
        "sound of",
        "sounds of",
        "a sound of",
        "a sounds of",
        "the noise of",
        "the hum of",
        "the roar of",
        "the chirping of",
        "the rustle of",
        "the howl of",
        "the buzz of",
        "the patter of",
        "the crash of",
        "the whir of",
    ]

    input_string = input_string.lower()

    for prefix in prefixes:
        if input_string.startswith(prefix):
            return input_string[len(prefix) :].strip()

    return input_string


def wrap_list(x):
    if isinstance(x, list):
        return x
    return [
        x,
    ]


def load_annotation(filename, sep="\t", header=0):
    if filename.endswith(".json"):
        anno = json.load(open(filename, "r"))
    elif filename.endswith(".tsv"):
        anno = pd.read_csv(filename, sep=sep, header=header)
    else:
        raise NotImplementedError
    return anno


def concat_datasets(datasets):
    if isinstance(datasets, dict):
        dataset_list = [d for d in datasets.values()]
    elif isinstance(datasets, list):
        dataset_list = datasets
    else:
        NotImplemented

    concat_dataset = ConcatDataset(dataset_list)
    return concat_dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        vis_processor=None,
        audio_processor=None,
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
        self.audio_processor = audio_processor
        self.text_processor = text_processor

        self.args = args
        self.split = split

        self.tokenizer = tokenizer
        self.data_type = None

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return SampleCollator(self, samples)

    def set_processors(self, vis_processor, audio_processor, text_processor):
        self.vis_processor = vis_processor
        self.audio_processor = audio_processor
        self.text_processor = text_processor


class AudioBaseDataset(Dataset):
    def __init__(
        self,
        vis_processor=None,
        audio_processor=None,
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
        self.audio_processor = audio_processor
        self.text_processor = text_processor

        self.args = args
        self.split = split
        self.tokenizer = tokenizer
        self.image_transform = image_transform

        # whether load vision for training
        self.load_vision = args.audio_load_vision

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


class AudioSetDataset(AudioBaseDataset):
    def __init__(
        self,
        vis_processor=None,
        audio_processor=None,
        text_processor=None,
        data_root=AUDIO_DATA_DIR,
        anno_path={
            "balanced_train": f"{AUDIO_META_DATA_DIR}/audioset_balanced_train.json",
            "unbalanced_train": f"{AUDIO_META_DATA_DIR}/audioset_unbalanced_train.json",
            "audioset_train_all": [
                f"{AUDIO_META_DATA_DIR}/audioset_balanced_train.json",
                f"{AUDIO_META_DATA_DIR}/audioset_unbalanced_train.json",
            ],
            "audioset_vgg": [
                f"{AUDIO_META_DATA_DIR}/audioset_balanced_train.json",
                f"{AUDIO_META_DATA_DIR}/audioset_unbalanced_train.json",
                f"{AUDIO_META_DATA_DIR}/vggsound_train.json",
            ],
            "val": f"{AUDIO_META_DATA_DIR}/audioset_val.json",
            "test": f"{AUDIO_META_DATA_DIR}/audioset_val.json",
        },
        args=None,
        split=None,
        tokenizer=None,
        image_transform=None,
        **kwargs,
    ):
        super().__init__(
            vis_processor,
            audio_processor,
            text_processor,
            data_root,
            anno_path,
            args,
            split,
            tokenizer,
            image_transform,
            **kwargs,
        )

        # init annotation
        if isinstance(anno_path[split], list):
            self.annotation = []
            for path in anno_path[split]:
                self.annotation = self.annotation + load_annotation(path)
        else:
            self.annotation = load_annotation(anno_path[split])

        if (
            split
            in [
                "balanced_train",
                "unbalanced_train",
                "audioset_vgg",
                "audioset_train_all",
            ]
            and self.load_vision
        ):
            logging.info("[AudiosetDataset] : Load good videos.")
            self.annotation = [
                item for item in self.annotation if item["is_good_video"] == True  # some videos are corrupted
            ]

        self.init_class_labels()

        # Pretrain specific
        self.is_train = self.split in [
            "train",
            "balanced_train",
            "unbalanced_train",
            "audioset_vgg",
            "audioset_train_all",
        ]
        self.mix_up = self.is_train and self.args.audio_mix_up

        # Evaluation specific
        self.eval_metric = "mAP"

    def init_class_labels(self):
        self.num_classes = 527
        self.idx2label = []
        self.label2idx = {}

        class_f = pd.read_csv(
            f"{AUDIO_META_DATA_DIR}/audioset_class_labels_indices.csv", header=0
        )
        for i in range(len(class_f)):
            item = class_f.iloc[i]
            assert item["index"] == i
            cls_name = item["display_name"].lower()  # use lower case
            self.idx2label.append(cls_name)
            self.label2idx[cls_name] = i
        assert len(self.idx2label) == self.num_classes

    def __getitem__(self, index):
        rtn = dict()
        n_retry = 0
        while len(rtn) == 0:
            try:
                ann = self.annotation[index]
                ann["video_path"] = os.path.join(self.data_root, ann["video_path"])
                ann["audio_path"] = os.path.join(self.data_root, ann["audio_path"])
                second_ann = None  # for mix up
                mix_lambda = None
                if self.mix_up and random.random() < self.args.audio_mix_up_p:
                    second_ann = self.annotation[
                        random.randint(0, len(self.annotation) - 1)
                    ]
                    second_ann["video_path"] = os.path.join(self.data_root, second_ann["video_path"])
                    second_ann["audio_path"] = os.path.join(self.data_root, second_ann["audio_path"])
                    mix_lambda = np.random.beta(10, 10)

                if self.is_train and self.load_vision:  # load video data for training
                    if second_ann is not None:  # mixup
                        vis_data, (start, end) = self.vis_processor(ann["video_path"])
                        sec_vis_data, (sec_start, sec_end) = self.vis_processor(
                            second_ann["video_path"]
                        )
                        vis_data = (
                            mix_lambda * vis_data + (1 - mix_lambda) * sec_vis_data
                        )

                        wf = self.audio_processor.load_audio_clip(
                            ann["audio_path"], start=start, end=end
                        )  # already sub mean
                        sec_wf = self.audio_processor.load_audio_clip(
                            second_ann["audio_path"], start=sec_start, end=sec_end
                        )  # already sub mean
                        mix_wf = mix_lambda * wf + (1 - mix_lambda) * sec_wf
                        mix_wf = mix_wf - mix_wf.mean()
                        audio_data = self.audio_processor(mix_wf)

                    else:  # no mixup
                        vis_data, (start, end) = self.vis_processor(ann["video_path"])
                        audio_data = self.audio_processor(
                            ann["audio_path"], se=(start, end)
                        )

                else:
                    vis_data = None
                    if second_ann is not None:  # mixup
                        wf = self.audio_processor.load_audio_clip(
                            ann["audio_path"]
                        )  # already sub mean
                        sec_wf = self.audio_processor.load_audio_clip(
                            second_ann["audio_path"]
                        )  # already sub mean
                        mix_wf = mix_lambda * wf + (1 - mix_lambda) * sec_wf
                        mix_wf = mix_wf - mix_wf.mean()
                        audio_data = self.audio_processor(mix_wf)
                    else:  # no mixup
                        audio_data = self.audio_processor(ann["audio_path"])

                if vis_data is not None:
                    if (
                        vis_data.ndim == 4 and vis_data.size(1) == self.args.n_frames
                    ):  # [ C x T x H x W ]
                        vis_data = einops.rearrange(vis_data, "c t h w -> t c h w")
                        assert vis_data.size(1) == 3  # rgb channel
                    elif vis_data.ndim == 3:
                        assert vis_data.size(0) == 3  # rgb channel

                rtn["image"] = vis_data
                rtn["audio"] = audio_data

                # text; TODO: make caption and template configurable later
                caption = None
                if len(ann["captions"]) > 1 and random.random() < 0.5:
                    caption = random.choice(
                        ann["captions"][1:]
                    )  # choose from additional captions
                else:
                    caption = ann["captions"][0]  # ann["captions"][0] is class names
                    caption = random.choice(SOUND_AS_IMAGE_TEMPLATE)(
                        caption
                    )  # add templates
                caption = self.text_processor(caption)

                if second_ann is not None:
                    # mixup case
                    sec_caption = random.choice(second_ann["captions"])
                    if caption.endswith("."):
                        caption = caption[:-1]
                    caption += f" and {sec_caption.lower()}"

                tokenized_caption = self.tokenizer([caption])[0]
                rtn["caption"] = tokenized_caption

                rtn["class_name"] = wrap_list(ann["class_name"])
                rtn["class_labels"] = wrap_list(ann["class_labels"])
                if second_ann:
                    # not use in training, in case for check
                    rtn["class_name"] = (
                        rtn["class_name"]
                        + [
                            "###",
                        ]
                        + wrap_list(second_ann["class_name"])
                    )
                    rtn["class_labels"] = (
                        rtn["class_labels"]
                        + [
                            "###",
                        ]
                        + wrap_list(second_ann["class_labels"])
                    )

                rtn["mixup_lambda"] = mix_lambda

                # for map
                label_item = torch.zeros(self.num_classes, dtype=torch.float)
                for lbl_id in rtn["class_labels"]:
                    if isinstance(lbl_id, str):
                        continue
                    label_item[lbl_id] = 1
                label_item = label_item.unsqueeze(0)
                rtn["target"] = label_item
                rtn["id"] = index

            except Exception as e:
                print(f"[AudiosetDataset] {e} -- {ann['audio_path']}")
                rtn = dict()
                index = random.randint(0, len(self.annotation) - 1)
                n_retry += 1
                if n_retry > 10:
                    raise ValueError("Exceed max retry.")

        return Sample(rtn)


class AudioCapsDataset(AudioBaseDataset):
    def __init__(
        self,
        vis_processor=None,
        audio_processor=None,
        text_processor=None,
        data_root=AUDIO_DATA_DIR,
        anno_path={
            "val": dict(
                audio=f"{AUDIO_META_DATA_DIR}/audiocaps_val_new.tsv",
                text=f"{AUDIO_META_DATA_DIR}/audiocaps_val_texts.json",
            ),
            "test": dict(
                audio=f"{AUDIO_META_DATA_DIR}/audiocaps_test_new.tsv",
                text=f"{AUDIO_META_DATA_DIR}/audiocaps_test_texts.json",
            ),
            "test_ib": dict(
                audio=f"{AUDIO_META_DATA_DIR}/audiocaps_test_ib.tsv",
                text=f"{AUDIO_META_DATA_DIR}/audiocaps_test_ib_texts.json",
            ),
        },
        args=None,
        split=None,
        tokenizer=None,
        image_transform=None,
        **kwargs,
    ):
        super().__init__(
            vis_processor,
            audio_processor,
            text_processor,
            data_root,
            anno_path,
            args,
            split,
            tokenizer,
            image_transform,
            **kwargs,
        )

        self.annotation = load_annotation(anno_path[split]["audio"], header=0, sep="\t")

        self.text_ids = None
        self.texts = None
        self.init_ret_texts()

        # Evaluation specific
        self.eval_metric = "recall"

    def init_ret_texts(self):
        self.text_ids = []
        self.texts = []
        fn = self.anno_path[self.split]["text"]
        text_infos = load_annotation(fn)
        for text_id, text_list in text_infos.items():
            for text in text_list:
                self.text_ids.append(int(text_id))
                self.texts.append(text)
        # TODO: move `text_ids` to cuda, do when evaluating this task

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]
        uniq_id = int(ann["uniq_id"])

        apath = os.path.join(self.data_root, ann["audio"])
        audio_data = self.audio_processor(apath)

        caption = ann["text"]
        tokenized_caption = self.tokenizer([caption])[0]

        return Sample(
            {
                "audio": audio_data,
                "caption": tokenized_caption,
                "uniq_id": uniq_id,
            }
        )


class ClothoDataset(AudioBaseDataset):
    def __init__(
        self,
        vis_processor=None,
        audio_processor=None,
        text_processor=None,
        data_root=AUDIO_DATA_DIR,
        anno_path={
            "val": dict(
                audio=f"{AUDIO_META_DATA_DIR}/clotho_validation_new.tsv",
                text=f"{AUDIO_META_DATA_DIR}/clotho_validation_texts.json",
            ),
            "test": dict(
                audio=f"{AUDIO_META_DATA_DIR}/clotho_evaluation_new.tsv",
                text=f"{AUDIO_META_DATA_DIR}/clotho_evaluation_texts.json",
            ),
        },
        args=None,
        split=None,
        tokenizer=None,
        image_transform=None,
        **kwargs,
    ):
        super().__init__(
            vis_processor,
            audio_processor,
            text_processor,
            data_root,
            anno_path,
            args,
            split,
            tokenizer,
            image_transform,
            **kwargs,
        )

        self.annotation = load_annotation(anno_path[split]["audio"], header=0, sep="\t")

        self.text_ids = None
        self.texts = None
        self.init_ret_texts()

        # Evaluation specific
        self.eval_metric = "recall"

    def init_ret_texts(self):
        self.text_ids = []
        self.texts = []
        fn = self.anno_path[self.split]["text"]
        text_infos = load_annotation(fn)
        for text_id, text_list in text_infos.items():
            for text in text_list:
                self.text_ids.append(int(text_id))
                self.texts.append(text)
        # TODO: move `text_ids` to cuda, do when evaluating this task

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]
        uniq_id = int(ann["uniq_id"])

        apath = os.path.join(self.data_root, ann["audio"])
        audio_data = self.audio_processor(apath)

        caption = ann["text"]
        tokenized_caption = self.tokenizer([caption])[0]

        return Sample(
            {
                "audio": audio_data,
                "caption": tokenized_caption,
                "uniq_id": uniq_id,
            }
        )


class ESC50Dataset(AudioBaseDataset):
    def __init__(
        self,
        vis_processor=None,
        audio_processor=None,
        text_processor=None,
        data_root=AUDIO_DATA_DIR,
        anno_path={
            "val-all": f"{AUDIO_META_DATA_DIR}/esc50_fold-all.json",
            "val-fold-1": f"{AUDIO_META_DATA_DIR}/esc50_fold-1.json",
            "val-fold-2": f"{AUDIO_META_DATA_DIR}/esc50_fold-2.json",
            "val-fold-3": f"{AUDIO_META_DATA_DIR}/esc50_fold-3.json",
            "val-fold-4": f"{AUDIO_META_DATA_DIR}/esc50_fold-4.json",
            "val-fold-5": f"{AUDIO_META_DATA_DIR}/esc50_fold-5.json",
        },
        args=None,
        split=None,
        tokenizer=None,
        image_transform=None,
        **kwargs,
    ):
        super().__init__(
            vis_processor,
            audio_processor,
            text_processor,
            data_root,
            anno_path,
            args,
            split,
            tokenizer,
            image_transform,
            **kwargs,
        )

        self.annotation = load_annotation(anno_path[split])
        self.init_class_labels()

        # Evaluation specific
        self.eval_metric = "acc"

    def init_class_labels(self):
        self.num_classes = 50

        self.idx2label = []
        self.label2idx = {}
        tmp_idx2label = {}

        class_f = load_annotation(f"{AUDIO_META_DATA_DIR}/esc50_label.json")
        for stri, names in class_f.items():
            assert len(names) == 1
            cls_name = names[0].lower()
            self.label2idx[cls_name] = int(stri)
            tmp_idx2label[stri] = cls_name

        assert len(self.label2idx) == self.num_classes

        for i in range(self.num_classes):
            self.idx2label.append(tmp_idx2label[str(i)])
        assert len(self.idx2label) == self.num_classes

    def __getitem__(self, index):
        ann = self.annotation[index]
        uniq_id = int(ann["uniq_id"])

        apath = os.path.join(self.data_root, ann["audio_path"])
        audio_data = self.audio_processor(apath)

        caption = ann["text"]
        tokenized_caption = self.tokenizer([caption])[0]

        label = ann["class_label"]

        return Sample(
            {
                "id": index,
                "audio": audio_data,
                "caption": tokenized_caption,
                "uniq_id": uniq_id,
                "label": label,
            }
        )


class VGGSoundCLSDataset(AudioBaseDataset):
    def __init__(
        self,
        vis_processor=None,
        audio_processor=None,
        text_processor=None,
        data_root=AUDIO_DATA_DIR,
        anno_path={"val": f"{AUDIO_META_DATA_DIR}/vggsound_audio-only_val.json"},
        args=None,
        split=None,
        tokenizer=None,
        image_transform=None,
        **kwargs,
    ):
        super().__init__(
            vis_processor,
            audio_processor,
            text_processor,
            data_root,
            anno_path,
            args,
            split,
            tokenizer,
            image_transform,
            **kwargs,
        )

        self.annotation = load_annotation(anno_path[split])
        self.init_class_labels()

        # Evaluation specific
        self.eval_metric = "acc"

    def init_class_labels(self):
        self.num_classes = 309

        df = pd.read_csv(f"{AUDIO_META_DATA_DIR}/vggsound_stat.csv", header=None)
        self.idx2label = []
        self.label2idx = {}
        for i in range(len(df)):
            item = df.iloc[i]
            cls_name = item[0].strip()
            self.idx2label.append(cls_name)
            self.label2idx[cls_name] = i

        assert len(self.idx2label) == self.num_classes

    def __getitem__(self, index):
        rtn = None
        while rtn is None:
            try:
                ann = self.annotation[index]
                vid = ann["vid"]

                apath = os.path.join(self.data_root, ann["audio_path"])
                audio_data = self.audio_processor(apath)

                caption = random.choice(ann["captions"])
                tokenized_caption = self.tokenizer([caption])[0]

                label = ann["class_labels"]

                rtn = {
                    "id": index,
                    "audio": audio_data,
                    "caption": tokenized_caption,
                    "uniq_id": vid,
                    "label": label,
                }
            except Exception as e:
                print(f"[VGGSound CLS]: {vid} -- {e}")
                rtn = None
                index = random.randint(0, len(self.annotation) - 1)

        return Sample(rtn)


name2dataset = {
    "audioset": AudioSetDataset,
    "esc50": ESC50Dataset,
    "clotho": ClothoDataset,
    "vggsound": VGGSoundCLSDataset,
    "audiocaps": AudioCapsDataset,
}
nclip_cfg_2 = {
    "audioset": 6,
    "esc50": 3,
    "clotho": 10,
    "vggsound": 6,
    "audiocaps": 6,
}
nclip_cfg_5 = {
    "audioset": 3,
    "esc50": 2,
    "clotho": 6,
    "vggsound": 3,
    "audiocaps": 3,
}
nclip_cfg_8 = {
    "audioset": 2,
    "esc50": 2,
    "clotho": 6,
    "vggsound": 2,
    "audiocaps": 2,
}
nclip_cfg_10 = {
    "audioset": 2,
    "esc50": 2,
    "clotho": 5,
    "vggsound": 2,
    "audiocaps": 2,
}

DURATION2CLIP = {
    "2": nclip_cfg_2,
    "5": nclip_cfg_5,
    "8": nclip_cfg_8,
    "10": nclip_cfg_10,
}


def create_audio_datasets(
    args, is_train, tokenizer, mean=None, std=None, image_transform=None
):
    dataset_names = args.train_data if is_train else args.val_data
    dataset_names = dataset_names.split("::")
    # format: "audioset@unbalanced-train::esc50@fold-1::vggsound@eval"

    datasets = dict()
    for dset_ns in dataset_names:
        # e.g. "audioset@unbalanced-train"
        ns = dset_ns.split("@")
        assert len(ns) == 2
        name, specific_split = ns[0], ns[1]
        dataset_cls = name2dataset[name]
        conf = OmegaConf.create(
            {
                "params": {
                    "sampling_rate": args.audio_sampling_rate,
                    "clip_duration": args.audio_clip_duration,
                    "n_clip": DURATION2CLIP[str(int(args.audio_clip_duration))][name],
                    "target_length": args.audio_target_length,
                    "mel_bins": args.audio_mel_bins,
                    "nframes": args.n_frames,
                    "freqm": args.audio_freqm,
                    "timem": args.audio_timem,
                    "noise_aug": args.audio_noise_aug,
                    "agg_eval": True,
                },
            }
        )

        vis_proc_cls = None
        if args.audio_load_vision:
            vis_proc_cls = PVProcessorTrain if is_train else None
        audio_proc_cls = ASTProcessorTrain if is_train else ASTProcessorEval

        vis_processor = (
            vis_proc_cls.from_config(cfg=conf) if vis_proc_cls is not None else None
        )
        text_processor = BlipCaptionProcessor.from_config()
        audio_processor = audio_proc_cls.from_config(cfg=conf)

        dataset = dataset_cls(
            vis_processor=vis_processor,
            audio_processor=audio_processor,
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
