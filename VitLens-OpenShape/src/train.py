import torch
import numpy as np
import logging
import os
import torch.distributed.nn
import torch.nn.functional as F
from tqdm import tqdm
from contextlib import suppress

from models.clip_bind import CLIPBindWrap
from loss import TriClipLoss

try:
    import wandb
except Exception as e:
    print(e)
    wandb = None


def isCLIPBindModel(module):
    if isinstance(module, CLIPBindWrap):
        return True
    if hasattr(module, "module"):
        if isinstance(module.module, CLIPBindWrap):
            return True
    return False


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


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


class Trainer(object):
    def __init__(
        self,
        rank,
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
        objaverse_lvis_loader=None,
        scanobjectnn_loader=None,
    ):
        self.rank = rank
        self.config = config
        self.model = model
        self.logit_scale = logit_scale
        self.image_proj = image_proj
        self.text_proj = text_proj
        self.scaler = scaler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.modelnet40_loader = modelnet40_loader
        self.objaverse_lvis_loader = objaverse_lvis_loader
        self.scanobjectnn_loader = scanobjectnn_loader
        self.epoch = 0
        self.step = 0
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc = 0
        self.best_lvis_acc = 0

        self.input_dtype = get_input_dtype(self.config.precision)
        self.autocast = get_autocast(self.config.precision)

        self.openclip_loss = None
        if self.config.training.use_openclip_loss:
            self.openclip_loss = TriClipLoss(
                local_loss=self.config.training.local_loss,
                gather_with_grad=self.config.training.gather_with_grad,
                cache_labels=True,
                rank=self.config.rank,
                world_size=self.config.world_size,
                use_horovod=False,
            )

        self.base_perf = [
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
        ]

    def beat_perf(self, perf, modelnet_acc, lvis_acc, scanobjnn_acc):
        # compare top1, top3, top5
        model_perf = [
            [modelnet_acc[i].item() for i in range(3)],
            [lvis_acc[i].item() for i in range(3)],
            [scanobjnn_acc[i].item() for i in range(3)],
        ]
        for i in range(3):
            for j in range(3):
                if model_perf[i][j] < perf[i][j]:
                    return False, model_perf

        return True, model_perf

    def dist_barrier(self):
        if self.config.distributed:
            torch.distributed.barrier()

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        if self.config.training.use_text_proj:
            self.text_proj.load_state_dict(checkpoint["text_proj"])
        if self.config.training.use_image_proj:
            self.image_proj.load_state_dict(checkpoint["image_proj"])
        self.logit_scale.load_state_dict(
            checkpoint["logit_scale"]
        )  # module.logit_scale = checkpoint['logit_scale']
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.config.training.use_openclip_optimizer_scheduler == False:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.scaler is not None and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]

        self.best_modelnet40_overall_acc = checkpoint["best_modelnet40_overall_acc"]
        self.best_modelnet40_class_acc = checkpoint["best_modelnet40_class_acc"]
        self.best_lvis_acc = checkpoint["best_lvis_acc"]

        logging.info("Loaded checkpoint from {}".format(path))
        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))
        logging.info(
            "----Best modelnet40 overall acc: {}".format(
                self.best_modelnet40_overall_acc
            )
        )
        logging.info(
            "----Best modelnet40 class acc: {}".format(self.best_modelnet40_class_acc)
        )
        logging.info("----Best lvis acc: {}".format(self.best_lvis_acc))

    def contras_loss(self, feat1, feat2, logit_scale=1, mask=None):
        if self.config.ngpu > 1:
            feat1 = F.normalize(feat1, dim=1)
            feat2 = F.normalize(feat2, dim=1)
            all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
            all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
            logits = logit_scale * all_feat1 @ all_feat2.T
        else:
            logits = (
                logit_scale * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
            )
        if mask is not None:
            logits = logits * mask
        labels = torch.arange(logits.shape[0]).to(self.config.device)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss, accuracy

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
                pred_feat = self.model(data["features_dense"], xyz=data["xyz_dense"])
            else:
                pred_feat = self.model(data["xyz_dense"], data["features_dense"])
        return pred_feat

    def train_one_epoch(self):
        self.model.train()
        if self.config.training.use_text_proj:
            self.text_proj.train()
        if self.config.training.use_image_proj:
            self.image_proj.train()

        text_contras_acc_list = []
        img_contras_acc_list = []

        num_batches_per_epoch = len(self.train_loader) // self.config.accum_freq

        if self.config.accum_freq > 1:
            accum_images, accum_texts, accum_visuals, accum_features = (
                [],
                [],
                [],
                dict(image=[], text=[], visual_i=[], visual_t=[]),
            )

        if self.config.training.use_mask:
            k = self.config.dataset.negative_sample_num
            s = self.config.dataset.train_batch_size
            mask1 = np.eye(k * s).astype(np.bool)  # (k s) x (k s)
            mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool)  # (k s) x (k s)
            mask_other = (
                torch.from_numpy(np.logical_or(mask1, 1 - mask2))
                .bool()
                .to(self.config.device)
            )  # (k s) x (k s)

        for i, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            i_accum = i // self.config.accum_freq
            self.step = num_batches_per_epoch * self.epoch + i_accum

            self.optimizer.zero_grad()

            ###########   forward for loss, modified for accum_freq
            if self.config.accum_freq == 1:
                loss = 0
                with self.autocast():
                    pred_feat = self.forward_model(data)
                    logit_scale = self.logit_scale(None)
                    idx = data["has_text_idx"]

                    text_feat = torch.vstack(data["text_feat"]).to(
                        device=self.config.device,
                        dtype=self.input_dtype,
                        non_blocking=True,
                    )
                    img_feat = torch.vstack(data["img_feat"]).to(
                        self.config.device, dtype=self.input_dtype, non_blocking=True
                    )

                    if self.config.training.use_mask:
                        img_text_sim = (
                            F.normalize(img_feat, dim=-1)
                            @ F.normalize(text_feat, dim=-1).T
                        )
                        mask = (
                            torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim
                            > self.config.training.mask_threshold
                        )
                        mask = torch.logical_or(mask, mask_other).detach()
                    else:
                        mask = None
                    if len(idx) > 0:
                        if self.config.training.use_text_proj:
                            text_feat = self.text_proj(text_feat)
                        text_contras_loss, text_contras_acc = self.contras_loss(
                            pred_feat[idx],
                            text_feat,
                            logit_scale=logit_scale,
                            mask=mask,
                        )
                        loss += (
                            text_contras_loss * self.config.training.lambda_text_contras
                        )
                        text_contras_acc_list.append(text_contras_acc.item())

                    if self.config.training.use_image_proj:
                        img_feat = self.image_proj(img_feat)
                    img_contras_loss, img_contras_acc = self.contras_loss(
                        pred_feat, img_feat, logit_scale=logit_scale, mask=mask
                    )

                    loss += img_contras_loss * self.config.training.lambda_img_contras
                    img_contras_acc_list.append(img_contras_acc.item())

                backward(loss, self.scaler)

            else:  # accum_freq > 1
                assert not self.config.training.use_mask
                mask = None
                loss = None
                load_text_feat = torch.vstack(data["text_feat"]).to(
                    device=self.config.device, dtype=self.input_dtype, non_blocking=True
                )
                load_img_feat = torch.vstack(data["img_feat"]).to(
                    device=self.config.device, dtype=self.input_dtype, non_blocking=True
                )
                idx = data["has_text_idx"]
                with torch.no_grad():
                    with self.autocast():
                        pred_feat = self.forward_model(data)
                        text_feat = (
                            self.text_proj(load_text_feat)
                            if self.config.training.use_text_proj
                            else load_text_feat
                        )
                        img_feat = (
                            self.image_proj(load_img_feat)
                            if self.config.training.use_image_proj
                            else load_img_feat
                        )

                        accum_features["image"].append(img_feat)
                        accum_features["text"].append(text_feat)
                        accum_features["visual_i"].append(pred_feat)
                        accum_features["visual_t"].append(pred_feat[idx])

                    accum_images.append(load_img_feat)
                    accum_texts.append(load_text_feat)
                    accum_visuals.append(data)

                if ((i + 1) % self.config.accum_freq) > 0:
                    continue

                self.optimizer.zero_grad()
                for j in range(self.config.accum_freq):
                    loss = 0.0  # important to set 0 here
                    l_data = accum_visuals[j]
                    l_img_feat = accum_images[j]
                    l_text_feat = accum_texts[j]
                    l_idx = l_data["has_text_idx"]
                    with self.autocast():
                        pred_feat = self.forward_model(l_data)
                        img_feat = (
                            self.image_proj(l_img_feat)
                            if self.config.training.use_image_proj
                            else l_img_feat
                        )
                        text_feat = (
                            self.text_proj(l_text_feat)
                            if self.config.training.use_text_proj
                            else l_text_feat
                        )
                        logit_scale = self.logit_scale(None)

                        accumulated_img_feat = torch.cat(
                            accum_features["image"][:j]
                            + [img_feat]
                            + accum_features["image"][j + 1 :]
                        )
                        accumulated_text_feat = torch.cat(
                            accum_features["text"][:j]
                            + [text_feat]
                            + accum_features["text"][j + 1 :]
                        )
                        accumulated_pred_feat_i = torch.cat(
                            accum_features["visual_i"][:j]
                            + [pred_feat]
                            + accum_features["visual_i"][j + 1 :]
                        )
                        accumulated_pred_feat_t = torch.cat(
                            accum_features["visual_t"][:j]
                            + [pred_feat[l_idx]]
                            + accum_features["visual_t"][j + 1 :]
                        )
                        # torch.cat(
                        #     [accum_features["visual"][r][accum_visuals[r]['has_text_idx']] for r in range(j)] + \
                        #     [pred_feat[l_data["has_text_idx"]]] + \
                        #     [accum_features["visual"][r][accum_visuals[r]['has_text_idx']] for r in range(j+1, self.config.accum_freq)]
                        # )

                        img_contras_loss, img_contras_acc = self.contras_loss(
                            accumulated_pred_feat_i,
                            accumulated_img_feat,
                            logit_scale=logit_scale,
                            mask=mask,
                        )
                        text_contras_loss, text_contras_acc = self.contras_loss(
                            accumulated_pred_feat_t,
                            accumulated_text_feat,
                            logit_scale=logit_scale,
                            mask=mask,
                        )
                        text_contras_acc_list.append(text_contras_acc.item())
                        img_contras_acc_list.append(img_contras_acc.item())
                        loss += (
                            img_contras_loss * self.config.training.lambda_img_contras
                        )
                        loss += (
                            text_contras_loss * self.config.training.lambda_text_contras
                        )

                    backward(loss, self.scaler)

            # ======================= end of loss_backward for any accum_freq ===================
            # optim step
            if self.scaler is not None:
                if self.config.training.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip_norm,
                        norm_type=2.0,
                    )
                    if self.config.training.use_image_proj:
                        torch.nn.utils.clip_grad_norm_(
                            self.image_proj.parameters(),
                            self.config.training.grad_clip_norm,
                            norm_type=2.0,
                        )
                    if self.config.training.use_text_proj:
                        torch.nn.utils.clip_grad_norm_(
                            self.text_proj.parameters(),
                            self.config.training.grad_clip_norm,
                            norm_type=2.0,
                        )
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                if self.config.training.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip_norm,
                        norm_type=2.0,
                    )
                    if self.config.training.use_image_proj:
                        torch.nn.utils.clip_grad_norm_(
                            self.image_proj.parameters(),
                            self.config.training.grad_clip_norm,
                            norm_type=2.0,
                        )
                    if self.config.training.use_text_proj:
                        torch.nn.utils.clip_grad_norm_(
                            self.text_proj.parameters(),
                            self.config.training.grad_clip_norm,
                            norm_type=2.0,
                        )
                self.optimizer.step()

            # scheduler
            if self.config.training.use_openclip_optimizer_scheduler:
                self.scheduler(self.step)
            else:
                self.scheduler.step()

            # reset gradient accum, if enabled
            if self.config.accum_freq > 1:
                accum_images, accum_texts, accum_visuals, accum_features = (
                    [],
                    [],
                    [],
                    dict(image=[], text=[], visual_i=[], visual_t=[]),
                )

            if self.rank == 0 and i_accum % self.config.training.log_freq == 0:
                try:
                    logging.info(
                        "[ Epoch {} | step {}]"
                        " lr:{:.6f} loss: {:.4f}, text_con_loss: {:.4f}, img_con_loss: {:.4f},"
                        " text_con_acc: {:.3f}, img_con_acc: {:.3f}, logit_scale: {:.3f}".format(
                            self.epoch,
                            self.step,
                            self.optimizer.param_groups[0]["lr"],
                            loss.item(),
                            text_contras_loss.item(),
                            img_contras_loss.item(),
                            text_contras_acc.item(),
                            img_contras_acc.item(),
                            logit_scale,
                        )
                    )
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/text_contras_loss": text_contras_loss.item()
                            if len(idx) > 0
                            else 0,
                            "train/img_contras_loss": img_contras_loss.item(),
                            "train/text_contras_acc": text_contras_acc.item()
                            if len(idx) > 0
                            else 0,
                            "train/img_contras_acc": img_contras_acc.item(),
                            "train/lr": self.optimizer.param_groups[0]["lr"],
                            "train/epoch": self.epoch,
                            "train/step": self.step,
                            "train/logit_scale": logit_scale,
                            "train/has_text": len(idx),
                            "train/filtered_pair": (mask == False).sum()
                            if mask is not None
                            else 0,
                        }
                    )
                except:
                    # print("wandb log error", flush=True)
                    pass
        if self.rank == 0:
            logging.info(
                "Train: text_cotras_acc: {0} image_contras_acc: {1}".format(
                    np.mean(text_contras_acc_list)
                    if len(text_contras_acc_list) > 0
                    else 0,
                    np.mean(img_contras_acc_list),
                )
            )

    def save_model(self, name):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "logit_scale": self.logit_scale.state_dict(),  # module.logit_scale,
                "text_proj": self.text_proj.state_dict()
                if self.config.training.use_text_proj
                else None,
                "image_proj": self.image_proj.state_dict()
                if self.config.training.use_image_proj
                else None,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()
                if self.config.training.use_openclip_optimizer_scheduler == False
                else None,
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
                "epoch": self.epoch,
                "step": self.step,
                "best_modelnet40_overall_acc": self.best_modelnet40_overall_acc,
                "best_modelnet40_class_acc": self.best_modelnet40_class_acc,
                "best_lvis_acc": self.best_lvis_acc,
            },
            os.path.join(self.config.ckpt_dir, "{}.pt".format(name)),
        )

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

    def train(self):
        for epoch in range(self.epoch, self.config.training.max_epoch):
            self.epoch = epoch
            if self.rank == 0:
                logging.info("Epoch: {}".format(self.epoch))

            if not self.config.training.use_openclip_loss:
                self.train_one_epoch()
            else:
                self.train_one_epoch_openclip()

            if self.rank == 0:
                self.save_model("latest")
            self.dist_barrier()

            modelnet_acc = self.test_modelnet40()
            objaverse_acc = self.test_objaverse_lvis()
            scanobjnn_acc = self.test_scanobjectnn()

            beat_prev_all, _ = self.beat_perf(
                self.base_perf, modelnet_acc, objaverse_acc, scanobjnn_acc
            )

            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                self.save_model("epoch_{}".format(self.epoch))

            if self.rank == 0:
                if beat_prev_all:
                    self.save_model("modelbest_all")
                    self.base_perf = model_perf
                    logging.info(
                        "Epoch {} become best. Check perf in log.".format(epoch)
                    )

            self.dist_barrier()

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
        # for i in range(40):
        #    print(idx2category[i], per_cat_acc[i])

        if self.config.rank == 0:
            if overall_acc > self.best_modelnet40_overall_acc:
                self.best_modelnet40_overall_acc = overall_acc
                self.save_model("best_modelnet40_overall")
            if per_cat_acc.mean() > self.best_modelnet40_class_acc:
                self.best_modelnet40_class_acc = per_cat_acc.mean()
                self.save_model("best_modelnet40_class")

            logging.info(
                "Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})".format(
                    overall_acc,
                    self.best_modelnet40_overall_acc,
                    per_cat_acc.mean(),
                    self.best_modelnet40_class_acc,
                )
            )
            logging.info(
                "Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}".format(
                    topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()
                )
            )
        self.dist_barrier()
        return topk_acc

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

        if overall_acc > self.best_lvis_acc and self.config.rank == 0:
            self.best_lvis_acc = overall_acc
            self.save_model("best_lvis")

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
        return topk_acc

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
        self.dist_barrier()
        return topk_acc

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

    def train_one_epoch_openclip(self):
        assert not self.config.training.use_mask
        self.model.train()
        if self.config.training.use_text_proj:
            self.text_proj.train()
        if self.config.training.use_image_proj:
            self.image_proj.train()

        i2v_acc_list, v2i_acc_list = [], []
        t2v_acc_list, v2t_acc_list = [], []

        num_batches_per_epoch = len(self.train_loader) // self.config.accum_freq

        if self.config.accum_freq > 1:
            accum_images, accum_texts, accum_visuals, accum_features = (
                [],
                [],
                [],
                dict(image=[], text=[], visual=[]),
            )

        for i, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            i_accum = i // self.config.accum_freq
            self.step = num_batches_per_epoch * self.epoch + i_accum

            self.optimizer.zero_grad()

            ###########   forward for loss, modified for accum_freq according to open_clip
            if self.config.accum_freq == 1:
                loss = None
                text_feat = torch.vstack(data["text_feat"]).to(
                    device=self.config.device, dtype=self.input_dtype, non_blocking=True
                )
                img_feat = torch.vstack(data["img_feat"]).to(
                    self.config.device, dtype=self.input_dtype, non_blocking=True
                )

                with self.autocast():
                    pred_feat = self.forward_model(data)
                    logit_scale = self.logit_scale(None)

                    if self.config.training.use_text_proj:
                        text_feat = self.text_proj(text_feat)
                    if self.config.training.use_image_proj:
                        img_feat = self.image_proj(img_feat)

                    text_feat = F.normalize(text_feat, dim=1)
                    img_feat = F.normalize(img_feat, dim=1)
                    pred_feat = F.normalize(pred_feat, dim=1)

                    loss_dict = self.openclip_loss(
                        img_feat, text_feat, pred_feat, logit_scale, output_dict=True
                    )
                    loss = loss_dict["contrastive_loss"]
                    t_contra_loss = loss_dict["t_contra_loss"]
                    i_contra_loss = loss_dict["i_contra_loss"]

                    t2v_acc_list.append(loss_dict["t2v_acc"].item())
                    v2t_acc_list.append(loss_dict["v2t_acc"].item())
                    i2v_acc_list.append(loss_dict["i2v_acc"].item())
                    v2i_acc_list.append(loss_dict["v2i_acc"].item())

                backward(loss, self.scaler)

            else:  # accum_freq > 1
                loss = None
                load_text_feat = torch.vstack(data["text_feat"]).to(
                    device=self.config.device, dtype=self.input_dtype, non_blocking=True
                )
                load_img_feat = torch.vstack(data["img_feat"]).to(
                    device=self.config.device, dtype=self.input_dtype, non_blocking=True
                )

                with torch.no_grad():
                    with self.autocast():
                        pred_feat = self.forward_model(data)
                        text_feat = (
                            self.text_proj(load_text_feat)
                            if self.config.training.use_text_proj
                            else load_text_feat
                        )
                        img_feat = (
                            self.image_proj(load_img_feat)
                            if self.config.training.use_image_proj
                            else load_img_feat
                        )

                        pred_feat = F.normalize(pred_feat, dim=1)
                        text_feat = F.normalize(text_feat, dim=1)
                        img_feat = F.normalize(img_feat, dim=1)

                        accum_features["image"].append(img_feat)
                        accum_features["text"].append(text_feat)
                        accum_features["visual"].append(pred_feat)

                    accum_images.append(load_img_feat)
                    accum_texts.append(load_text_feat)
                    accum_visuals.append(data)

                if ((i + 1) % self.config.accum_freq) > 0:
                    continue

                self.optimizer.zero_grad()
                for j in range(self.config.accum_freq):
                    l_data = accum_visuals[j]
                    l_img_feat = accum_images[j]
                    l_text_feat = accum_texts[j]

                    with self.autocast():
                        pred_feat = self.forward_model(l_data)
                        img_feat = (
                            self.image_proj(l_img_feat)
                            if self.config.training.use_image_proj
                            else l_img_feat
                        )
                        text_feat = (
                            self.text_proj(l_text_feat)
                            if self.config.training.use_text_proj
                            else l_text_feat
                        )
                        logit_scale = self.logit_scale(None)

                        pred_feat = F.normalize(pred_feat, dim=1)
                        img_feat = F.normalize(img_feat, dim=1)
                        text_feat = F.normalize(text_feat, dim=1)

                        accumulated_img_feat = torch.cat(
                            accum_features["image"][:j]
                            + [img_feat]
                            + accum_features["image"][j + 1 :]
                        )
                        accumulated_text_feat = torch.cat(
                            accum_features["text"][:j]
                            + [text_feat]
                            + accum_features["text"][j + 1 :]
                        )
                        accumulated_pred_feat = torch.cat(
                            accum_features["visual"][:j]
                            + [pred_feat]
                            + accum_features["visual"][j + 1 :]
                        )

                        loss_dict = self.openclip_loss(
                            accumulated_img_feat,
                            accumulated_text_feat,
                            accumulated_pred_feat,
                            logit_scale,
                            output_dict=True,
                        )
                        loss = loss_dict["contrastive_loss"]
                        t_contra_loss = loss_dict["t_contra_loss"]
                        i_contra_loss = loss_dict["i_contra_loss"]

                        t2v_acc_list.append(loss_dict["t2v_acc"].item())
                        v2t_acc_list.append(loss_dict["v2t_acc"].item())
                        i2v_acc_list.append(loss_dict["i2v_acc"].item())
                        v2i_acc_list.append(loss_dict["v2i_acc"].item())

                    backward(loss, self.scaler)

            # ======================= end of loss_backward for any accum_freq ===================
            # optim step
            if self.scaler is not None:
                if self.config.training.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip_norm,
                        norm_type=2.0,
                    )
                    if self.config.training.use_image_proj:
                        torch.nn.utils.clip_grad_norm_(
                            self.image_proj.parameters(),
                            self.config.training.grad_clip_norm,
                            norm_type=2.0,
                        )
                    if self.config.training.use_text_proj:
                        torch.nn.utils.clip_grad_norm_(
                            self.text_proj.parameters(),
                            self.config.training.grad_clip_norm,
                            norm_type=2.0,
                        )
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                if self.config.training.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip_norm,
                        norm_type=2.0,
                    )
                    if self.config.training.use_image_proj:
                        torch.nn.utils.clip_grad_norm_(
                            self.image_proj.parameters(),
                            self.config.training.grad_clip_norm,
                            norm_type=2.0,
                        )
                    if self.config.training.use_text_proj:
                        torch.nn.utils.clip_grad_norm_(
                            self.text_proj.parameters(),
                            self.config.training.grad_clip_norm,
                            norm_type=2.0,
                        )

                self.optimizer.step()

            # scheduler
            if self.config.training.use_openclip_optimizer_scheduler:
                self.scheduler(self.step)
            else:
                self.scheduler.step()

            # reset gradient accum, if enabled
            if self.config.accum_freq > 1:
                accum_images, accum_texts, accum_visuals, accum_features = (
                    [],
                    [],
                    [],
                    dict(image=[], text=[], visual=[]),
                )

            if self.rank == 0 and i_accum % self.config.training.log_freq == 0:
                logging.info(
                    "[ Epoch {} | step {}]"
                    " lr:{:.6f} loss: {:.4f}, text_con_loss: {:.4f}, img_con_loss: {:.4f},"
                    " t2v_acc: {:.3f}, v2t_acc: {:.3f}, i2v_acc: {:.3f}, v2i_acc: {:.3f}, logit_scale: {:.3f}".format(
                        self.epoch,
                        self.step,
                        self.optimizer.param_groups[0]["lr"],
                        loss.item(),
                        t_contra_loss.item(),
                        i_contra_loss.item(),
                        loss_dict["t2v_acc"].item(),
                        loss_dict["v2t_acc"].item(),
                        loss_dict["i2v_acc"].item(),
                        loss_dict["v2i_acc"].item(),
                        logit_scale,
                    )
                )

        ## ======================= end of for loop over one epoch of training (train loader) ===========================
        if self.rank == 0:
            logging.info(
                "Train: t2v_acc: {0} v2t_acc: {1}  i2v_acc: {2}  v2i_acc: {3}".format(
                    np.mean(t2v_acc_list),
                    np.mean(v2t_acc_list),
                    np.mean(i2v_acc_list),
                    np.mean(v2i_acc_list),
                )
            )
