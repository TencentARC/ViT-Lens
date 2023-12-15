import sys
import os
import logging
import shutil
import data
import random
import models

try:
    import MinkowskiEngine as ME
except:
    logging.info("MinkowskiEngine not installed.")
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from collections import OrderedDict
from datetime import datetime
from param import parse_args
from utils.misc import load_config, dump_config
from utils.logger import setup_logging
from utils.distributed import init_distributed_device
from models.clip_bind import CLIPBindWrap
from models.LogitScaleNetwork import LogitScaleNetwork

import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import suppress


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12358"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def isCLIPBindModel(module):
    if isinstance(module, CLIPBindWrap):
        return True
    if hasattr(module, "module"):
        if isinstance(module.module, CLIPBindWrap):
            return True
    return False


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ("bf16", "pure_bf16"):
        input_dtype = torch.bfloat16
    elif precision in ("fp16", "pure_fp16"):
        input_dtype = torch.float16
    return input_dtype


class Infer(object):
    def __init__(
        self,
        rank,
        config,
        model,
        logit_scale,
        image_proj,
        text_proj,
        modelnet40_loader,
        objaverse_lvis_loader=None,
        scanobjectnn_loader=None,
    ):
        self.rank = rank
        self.config = config
        self.model = model
        self.logit_scale = logit_scale
        self.image_proj = image_proj
        self.text_proj = text_proj
        self.modelnet40_loader = modelnet40_loader
        self.objaverse_lvis_loader = objaverse_lvis_loader
        self.scanobjectnn_loader = scanobjectnn_loader

        self.input_dtype = get_input_dtype(self.config.precision)
        self.autocast = get_autocast(self.config.precision)

    def forward_model(self, data):
        pred_feat = None
        if not self.config.model.get("use_dense", False):
            xyz = data["xyz"].to(
                self.config.device, dtype=self.input_dtype, non_blocking=True
            )
            features = data["features"].to(
                self.config.device, dtype=self.input_dtype, non_blocking=True
            )
            pred_feat = self.model(
                xyz,
                features,
                device=self.config.device,
                quantization_size=self.config.model.voxel_size,
            )
        else:
            features = data["features_dense"].to(
                self.config.device, dtype=self.input_dtype, non_blocking=True
            )
            xyz = data["xyz_dense"].to(
                self.config.device, dtype=self.input_dtype, non_blocking=True
            )
            if isCLIPBindModel(self.model):
                pred_feat = self.model(features, xyz=xyz)
            else:
                pred_feat = self.model(xyz, features)
        return pred_feat

    def dist_barrier(self):
        if self.config.distributed:
            torch.distributed.barrier()

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.config.device)
        # import pdb; pdb.set_trace()
        self.model.load_state_dict(self.convert_state_dict(checkpoint["state_dict"]))
        if self.config.training.use_text_proj:
            self.text_proj.load_state_dict(
                self.convert_state_dict(checkpoint["text_proj"])
            )
        if self.config.training.use_image_proj:
            self.image_proj.load_state_dict(
                self.convert_state_dict(checkpoint["image_proj"])
            )

    def convert_state_dict(self, state_dict):
        is_dist = list(state_dict.keys())[0].startswith("module.")
        if is_dist and not self.config.distributed:
            sd = OrderedDict()
            for k, v in state_dict.items():
                sd.update({k[7:]: v})
            return sd
        if not is_dist and self.config.distributed:
            sd = OrderedDict
            for k, v in state_dict.items():
                sd.update({"module.{}".format(k): v})
            return sd
        return state_dict

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res, correct

    def inference(self):
        self.test_modelnet40()
        self.test_objaverse_lvis()
        self.test_scanobjectnn()

    def test_modelnet40(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(
            self.modelnet40_loader.dataset.clip_cat_feat
        ).to(self.config.device, dtype=self.input_dtype)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)

        category2idx = self.modelnet40_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        topk_acc, correct, overall_acc, per_cat_acc = self.calculate_metrics(
            self.modelnet40_loader, clip_text_feat, 40
        )
        if self.config.rank == 0:
            logging.info(
                "Test ModelNet40: overall acc: {0} class_acc: {1}".format(
                    overall_acc, per_cat_acc.mean()
                )
            )
            logging.info(
                "Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}".format(
                    topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()
                )
            )
            for i in range(40):
                logging.info("[Cate: {}]: {}".format(idx2category[i], per_cat_acc[i]))
        self.dist_barrier()

    def test_objaverse_lvis(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(
            self.objaverse_lvis_loader.dataset.clip_cat_feat
        ).to(self.config.device, dtype=self.input_dtype)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        category2idx = self.objaverse_lvis_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        topk_acc, correct, overall_acc, per_cat_acc = self.calculate_metrics(
            self.objaverse_lvis_loader, clip_text_feat, 1156
        )

        if self.config.rank == 0:
            logging.info(
                "Test ObjaverseLVIS: overall acc: {0} class_acc: {1}".format(
                    overall_acc, per_cat_acc.mean()
                )
            )
            logging.info(
                "Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}".format(
                    topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()
                )
            )
        self.dist_barrier()

    def test_scanobjectnn(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(
            self.scanobjectnn_loader.dataset.clip_cat_feat
        ).to(self.config.device, dtype=self.input_dtype)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)

        category2idx = self.scanobjectnn_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        topk_acc, correct, overall_acc, per_cat_acc = self.calculate_metrics(
            self.scanobjectnn_loader, clip_text_feat, 15
        )

        if self.config.rank == 0:
            logging.info(
                "Test ScanObjectNN: overall acc: {0} class_acc: {1}".format(
                    overall_acc, per_cat_acc.mean()
                )
            )
            logging.info(
                "Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}".format(
                    topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()
                )
            )

    def calculate_metrics(self, dataloader, clip_text_feat, num_class):
        logits_all = []
        labels_all = []
        per_cat_correct = torch.zeros(num_class).to(
            self.config.device, dtype=self.input_dtype
        )
        per_cat_count = torch.zeros(num_class).to(
            self.config.device, dtype=self.input_dtype
        )

        with torch.no_grad():
            for data in tqdm(
                dataloader, total=len(dataloader), desc=f"Rank:{self.config.rank}"
            ):
                with self.autocast():
                    pred_feat = self.forward_model(data)
                    logits = (
                        F.normalize(pred_feat, dim=1)
                        @ F.normalize(clip_text_feat, dim=1).T
                    )
                    logits_all.append(logits.detach())
                labels = data["category"].to(self.config.device)
                labels_all.append(labels)
                # calculate per class accuracy
                for i in range(num_class):
                    idx = labels == i
                    if idx.sum() > 0:
                        per_cat_correct[i] += (
                            (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                        )
                        per_cat_count[i] += idx.sum()

        logits_all = torch.cat(logits_all)
        labels_all = torch.cat(labels_all)
        if self.config.distributed:
            dist_logits_all = [
                torch.zeros_like(logits_all) for _ in range(self.config.world_size)
            ]
            dist_labels_all = [
                torch.zeros_like(labels_all) for _ in range(self.config.world_size)
            ]
            torch.distributed.all_gather(dist_logits_all, logits_all)
            torch.distributed.all_gather(dist_labels_all, labels_all)
            logits_all = torch.cat(dist_logits_all)
            labels_all = torch.cat(dist_labels_all)

            torch.distributed.all_reduce(
                per_cat_correct, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                per_cat_count, op=torch.distributed.ReduceOp.SUM
            )

        topk_acc, correct = self.accuracy(
            logits_all,
            labels_all,
            topk=(
                1,
                3,
                5,
            ),
        )

        overall_acc = per_cat_correct.sum() / per_cat_count.sum()
        per_cat_acc = per_cat_correct / per_cat_count

        return topk_acc, correct, overall_acc, per_cat_acc


def main(args):
    # (rank, world_size, cli_args, extras):
    cli_args, extras = parse_args(sys.argv[1:])

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config = load_config(cli_args.config, cli_args=vars(cli_args), extra_args=extras)
    device = init_distributed_device(args=config)

    config.trial_name = config.get("trial_name") + datetime.now().strftime(
        "@%Y%m%d-%H%M%S"
    )
    if config.rank == 0:
        os.makedirs(
            os.path.join(config.exp_dir, config.trial_name), exist_ok=config.autoresume
        )

    if config.rank == 0:
        config.log_path = config.get("log_path") or os.path.join(
            config.exp_dir, config.trial_name, "log.txt"
        )
        config.log_level = logging.DEBUG if config.debug else logging.INFO
        setup_logging(config.log_path, config.log_level)
        dump_config(
            os.path.join(config.exp_dir, config.trial_name, "config.yaml"), config
        )

    if config.distributed:
        torch.distributed.barrier()

    model = models.make(config).to(config.device)
    torch.cuda.set_device(config.device)
    model.to(config.device)

    if config.distributed:
        model = DDP(
            model,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=False,
        )
    if config.model.name.startswith("Mink"):
        raise NotImplemented
    else:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info("Using SyncBatchNorm")

    logit_scale = LogitScaleNetwork(config.training.logit_scale_init).to(config.device)
    image_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(
        config.device
    )
    text_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(
        config.device
    )

    if config.distributed:
        logit_scale = DDP(
            logit_scale,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=False,
        )
        image_proj = DDP(
            image_proj,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=False,
        )
        text_proj = DDP(
            text_proj,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=False,
        )

    modelnet40_loader = data.make_modelnet40test(config)
    objaverse_lvis_loader = data.make_objaverse_lvis(config)
    scanobjectnn_loader = data.make_scanobjectnntest(config)

    runner = Infer(
        config.rank,
        config,
        model,
        logit_scale,
        image_proj,
        text_proj,
        modelnet40_loader,
        objaverse_lvis_loader,
        scanobjectnn_loader,
    )

    assert (
        config.resume is not None
    ), "Please provide the checkpoint path for inference."
    runner.load_from_checkpoint(config.resume)

    runner.inference()
    exit(0)


if __name__ == "__main__":
    main(sys.argv[:1])
