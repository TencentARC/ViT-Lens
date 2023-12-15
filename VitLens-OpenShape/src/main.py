import sys
import os
import logging
import shutil
import data
import random
import models
import torch

from omegaconf import OmegaConf
from datetime import datetime
from param import parse_args
from utils.misc import load_config, dump_config
from utils.logger import setup_logging
from utils.scheduler import cosine_lr
from utils.distributed import init_distributed_device
from train import Trainer
from models.LogitScaleNetwork import LogitScaleNetwork

import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
import torch.multiprocessing as mp

try:
    import MinkowskiEngine as ME
except:
    logging.info("MinkowskiEngine not installed.")

try:
    import wandb
except Exception as e:
    print(e)
    wandb = None


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

    if config.autoresume:
        config.trial_name = config.get("trial_name") + "@autoresume"
    else:
        config.trial_name = config.get("trial_name") + datetime.now().strftime(
            "@%Y%m%d-%H%M%S"
        )
    config.ckpt_dir = config.get("ckpt_dir") or os.path.join(
        config.exp_dir, config.trial_name, "ckpt"
    )
    config.code_dir = config.get("code_dir") or os.path.join(
        config.exp_dir, config.trial_name, "code"
    )

    if config.rank == 0:
        os.makedirs(
            os.path.join(config.exp_dir, config.trial_name), exist_ok=config.autoresume
        )
        os.makedirs(config.ckpt_dir, exist_ok=True)
        if os.path.exists(config.code_dir):
            shutil.rmtree(config.code_dir)
        shutil.copytree("./src", config.code_dir)

    # config.device = 'cuda:{0}'.format(rank)

    if config.rank == 0:
        config.log_path = config.get("log_path") or os.path.join(
            config.exp_dir, config.trial_name, "log.txt"
        )
        config.log_level = logging.DEBUG if config.debug else logging.INFO
        setup_logging(config.log_path, config.log_level)
        dump_config(
            os.path.join(config.exp_dir, config.trial_name, "config.yaml"), config
        )
        # logging.info("Using {} GPU(s).".format(config.ngpu))
        # wandb.init(project=config.project_name, name=config.trial_name, config=OmegaConf.to_object(config))
    if config.distributed:
        torch.distributed.barrier()

    if config.train:
        model = models.make(config).to(config.device)
        if config.lock_visual:
            model.lock(
                config.lock_visual_unlocked_groups,
                config.lock_visual_freeze_bn_stats,
                config.unlock_cls,
                config.unlock_trans_first_n_layers,
            )
        if config.rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            logging.info(model)
            logging.info(
                "Network:{}, Number of parameters: {}".format(
                    config.model.name, total_params
                )
            )

            for n, p in model.named_parameters():
                if p.requires_grad:
                    logging.info(n)

            n_trainable_param = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            logging.info(
                "Network:{}, Number of trainable parameters: {}".format(
                    config.model.name, n_trainable_param
                )
            )

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
            # model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model) # minkowski only
            # logging.info("Using MinkowskiSyncBatchNorm")
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logging.info("Using SyncBatchNorm")

        logit_scale = LogitScaleNetwork(config.training.logit_scale_init).to(
            config.device
        )
        image_proj = torch.nn.Linear(
            config.model.out_channel, config.model.out_channel
        ).to(config.device)
        text_proj = torch.nn.Linear(
            config.model.out_channel, config.model.out_channel
        ).to(config.device)

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

        train_loader = data.make(config, "train", config.rank, config.world_size)

        modelnet40_loader = data.make_modelnet40test(config)
        objaverse_lvis_loader = data.make_objaverse_lvis(config)
        scanobjectnn_loader = data.make_scanobjectnntest(config)

        if config.rank == 0:
            if train_loader is not None:
                logging.info("Train iterations: {}".format(len(train_loader)))

        exclude_wd = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )
        include_wd = lambda n, p: not exclude_wd(n, p)
        backbone_trans = lambda n: "backbone.transformer" in n

        named_parameters = (
            list(model.named_parameters())
            + list(image_proj.named_parameters())
            + list(text_proj.named_parameters())
            + list(logit_scale.named_parameters())
        )

        trans_exclude_wd = [
            p
            for n, p in named_parameters
            if backbone_trans(n) and exclude_wd(n, p) and p.requires_grad
        ]
        trans_include_wd = [
            p
            for n, p in named_parameters
            if backbone_trans(n) and include_wd(n, p) and p.requires_grad
        ]
        base_exclude_wd = [
            p
            for n, p in named_parameters
            if (not backbone_trans(n)) and exclude_wd(n, p) and p.requires_grad
        ]
        base_include_wd = [
            p
            for n, p in named_parameters
            if (not backbone_trans(n)) and include_wd(n, p) and p.requires_grad
        ]

        param_groups = (
            [
                {
                    "params": trans_exclude_wd,
                    "weight_decay": 0.0,
                    "lr": 0.1 * config.training.lr,
                },
                {
                    "params": trans_include_wd,
                    "weight_decay": config.training.weight_decay,
                    "lr": 0.1 * config.training.lr,
                },
                {
                    "params": base_exclude_wd,
                    "weight_decay": 0.0,
                },
                {
                    "params": base_include_wd,
                    "weight_decay": config.training.weight_decay,
                },
            ]
            if config.model.name == "clipbind"
            else list(model.parameters())
        )

        if config.training.use_openclip_optimizer_scheduler:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.training.lr,
                betas=(config.training.beta1, config.training.beta2),
                eps=config.training.eps,
            )
            scheduler = cosine_lr(
                optimizer,
                config.training.lr,
                config.training.warmup,
                len(train_loader)
                // config.accum_freq
                * 160,  # FIXME: this number is the max epoch
            )
        else:
            optimizer = torch.optim.AdamW(param_groups, lr=config.training.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.training.lr_decay_step,
                gamma=config.training.lr_decay_rate,
            )

        scaler = GradScaler() if config.precision == "amp" else None

        trainer = Trainer(
            config.rank,
            config,
            model,
            logit_scale,
            image_proj,
            text_proj,
            optimizer,
            scaler,
            scheduler,
            train_loader,
            modelnet40_loader,
            objaverse_lvis_loader,
            scanobjectnn_loader,
        )

        if config.resume is not None:
            trainer.load_from_checkpoint(config.resume)
        elif config.autoresume:
            if os.path.exists(os.path.join(config.ckpt_dir, "{}.pt".format("latest"))):
                trainer.load_from_checkpoint(
                    os.path.join(config.ckpt_dir, "{}.pt".format("latest"))
                )

        trainer.train()

    cleanup()


if __name__ == "__main__":
    main(sys.argv[:1])
