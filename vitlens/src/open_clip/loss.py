import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True

except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
):
    assert (
        has_distributed
    ), "torch.distributed did not import correctly, please use a PyTorch version with support."
    if use_horovod:
        assert hvd is not None, "Please install horovod"
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(
                    all_image_features.chunk(world_size, dim=0)
                )
                gathered_text_features = list(
                    all_text_features.chunk(world_size, dim=0)
                )
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0
            )
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0
            )
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class TriClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, features_x, features_y, logit_scale):
        if self.world_size > 1:
            all_features_x, all_features_y = gather_features(
                features_x,
                features_y,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_x = logit_scale * features_x @ all_features_y.T
                logits_per_y = logit_scale * features_y @ all_features_x.T
            else:
                logits_per_x = logit_scale * all_features_x @ all_features_y.T
                logits_per_y = logits_per_x.T
        else:
            logits_per_x = logit_scale * features_x @ features_y.T
            logits_per_y = logit_scale * features_y @ features_x.T

        return logits_per_x, logits_per_y

    def forward(
        self,
        image_features,
        text_features,
        visual_features,
        logit_scale,
        output_dict=False,
    ):
        device = image_features.device
        IV_logits_per_image, IV_logits_per_visual = self.get_logits(
            image_features, visual_features, logit_scale
        )
        TV_logits_per_text, TV_logits_per_visual = self.get_logits(
            text_features, visual_features, logit_scale
        )

        labels = self.get_ground_truth(device, IV_logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(IV_logits_per_image, labels)
            + F.cross_entropy(IV_logits_per_visual, labels)
            + F.cross_entropy(TV_logits_per_text, labels)
            + F.cross_entropy(TV_logits_per_visual, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class TriClipDistillTokenLoss(TriClipLoss):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        triclip_loss_weight=1.0,
        distill_loss_weight=1.0,
        loss_type="mse",
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
        )
        self.triclip_loss_weight = triclip_loss_weight
        self.distill_loss_weight = distill_loss_weight
        self.distill_loss_type = loss_type

    def distill_token_loss(self, inp, target):
        # may provide more options later
        if self.distill_loss_type == "mse":
            return F.mse_loss(inp, target)
        elif self.distill_loss_type == "cos":
            return -torch.nn.CosineSimilarity(dim=-1)(inp, target).mean()

    def forward(
        self,
        image_features,
        text_features,
        visual_features,
        image_tokens,
        visual_tokens,
        logit_scale,
        output_dict=False,
    ):
        triclip_loss = 0.0
        if self.triclip_loss_weight:
            triclip_loss = super().forward(
                image_features,
                text_features,
                visual_features,
                logit_scale,
                output_dict=False,
            )
            triclip_loss = self.triclip_loss_weight * triclip_loss

        dst_token_loss = 0
        if self.distill_loss_weight:
            dst_token_loss = self.distill_token_loss(visual_tokens, image_tokens)
            dst_token_loss = self.distill_loss_weight * dst_token_loss

        if output_dict:
            return {
                "triclip_loss": triclip_loss,
                "dst_token_loss": dst_token_loss,
            }
        return triclip_loss, dst_token_loss


class ClipLossGeneral(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, x_features, y_features, logit_scale):
        if self.world_size > 1:
            all_x_features, all_y_features = gather_features(
                x_features,
                y_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_x = logit_scale * x_features @ all_y_features.T
                logits_per_y = logit_scale * y_features @ all_x_features.T
            else:
                logits_per_x = logit_scale * all_x_features @ all_y_features.T
                logits_per_y = logits_per_x.T
        else:
            logits_per_x = logit_scale * x_features @ y_features.T
            logits_per_y = logit_scale * y_features @ x_features.T

        return logits_per_x, logits_per_y

    def forward(
        self, x_features, y_features, logit_scale, output_dict=False, key="image-text"
    ):
        device = x_features.device
        logits_per_x, logits_per_y = self.get_logits(
            x_features, y_features, logit_scale
        )

        labels = self.get_ground_truth(device, logits_per_x.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_x, labels)
            + F.cross_entropy(logits_per_y, labels)
        ) / 2

        return {key: total_loss} if output_dict else total_loss


class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features, text_features, logit_scale
        )

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
        self,
        caption_loss_weight,
        clip_loss_weight,
        pad_id=0,  # pad_token for open_clip custom tokenizer
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(
        self,
        image_features,
        text_features,
        logits,
        labels,
        logit_scale,
        output_dict=False,
    ):
        clip_loss = 0

        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):
    def dist_loss(self, teacher_logits, student_logits):
        return (
            -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1))
            .sum(dim=1)
            .mean(dim=0)
        )

    def forward(
        self,
        image_features,
        text_features,
        logit_scale,
        dist_image_features,
        dist_text_features,
        dist_logit_scale,
        output_dict=False,
    ):
        logits_per_image, logits_per_text = self.get_logits(
            image_features, text_features, logit_scale
        )

        dist_logits_per_image, dist_logits_per_text = self.get_logits(
            dist_image_features, dist_text_features, dist_logit_scale
        )

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image)
            + self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


class ClipLossSimMask(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        sim_thres=0.9,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.sim_thres = sim_thres
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, x_features, y_features, logit_scale):
        # Note: x_features are the features extracted by `teacher`, use x_features to design mask
        if self.world_size > 1:
            all_x_features, all_y_features = gather_features(
                x_features,
                y_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            sim_mat = all_x_features @ all_x_features.T
            mask = torch.logical_or(
                torch.logical_not(sim_mat >= self.sim_thres),
                torch.eye(
                    all_x_features.shape[0], device=x_features.device, dtype=torch.bool
                ),
            )

            if self.local_loss:
                logits_per_x = logit_scale * x_features @ all_y_features.T
                logits_per_y = logit_scale * y_features @ all_x_features.T

                iterval = x_features.shape[0]
                local_mask_x = mask[
                    self.rank * iterval : self.rank * iterval + iterval, :
                ]
                logits_per_x = logits_per_x * local_mask_x
                local_mask_y = mask.T[
                    self.rank * iterval : self.rank * iterval + iterval, :
                ]
                logits_per_y = logits_per_y * local_mask_y

            else:
                logits_per_x = logit_scale * all_x_features @ all_y_features.T
                logits_per_x = logits_per_x * mask
                logits_per_y = logits_per_x.T

        else:  # world_size == 1
            sim_mat = x_features @ x_features.T
            mask = torch.logical_or(
                torch.logical_not(sim_mat >= self.sim_thres),  # n_x x n_y
                torch.eye(
                    x_features.shape[0], device=x_features.device, dtype=torch.bool
                ),
            )

            logits_per_x = logit_scale * x_features @ y_features.T
            logits_per_y = logit_scale * y_features @ x_features.T
            logits_per_x = logits_per_x * mask
            logits_per_y = logits_per_y * mask.T

        return logits_per_x, logits_per_y

    def forward(
        self,
        x_features,
        y_features,
        logit_scale,
        output_dict=False,
        key="contrastive loss[with sim mask]",
    ):
        device = x_features.device
        logits_per_x, logits_per_y = self.get_logits(
            x_features, y_features, logit_scale
        )

        labels = self.get_ground_truth(device, logits_per_x.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_x, labels)
            + F.cross_entropy(logits_per_y, labels)
        ) / 2

        return {key: total_loss} if output_dict else total_loss


class ClipLossLabelMask(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_mask=False,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_mask = use_mask
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(
        self, x_features, y_features, logit_scale, x_labels=None, y_labels=None
    ):
        if self.world_size > 1:
            all_x_features, all_y_features = gather_features(
                x_features,
                y_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            all_x_labels, all_y_labels = None, None
            mask = None
            if (
                x_labels is not None and y_labels is not None and self.use_mask
            ):  # mask flag here
                assert x_labels.shape == y_labels.shape
                all_x_labels, all_y_labels = gather_features(
                    x_labels,
                    y_labels,
                    self.local_loss,
                    False,
                    self.rank,
                    self.world_size,
                    self.use_horovod,
                )
                if all_x_labels.ndim == 1:
                    all_x_labels = all_x_labels.unsqueeze(0)
                    all_y_labels = all_y_labels.unsqueeze(0)
                mask = torch.logical_or(
                    torch.logical_not(all_x_labels.T == all_y_labels),  # n_x x n_y
                    torch.eye(
                        all_x_features.shape[0],
                        device=x_features.device,
                        dtype=torch.bool,
                    ),
                )

            if self.local_loss:
                logits_per_x = logit_scale * x_features @ all_y_features.T
                logits_per_y = logit_scale * y_features @ all_x_features.T
                if mask is not None:
                    iterval = x_features.shape[0]
                    local_mask_x = mask[
                        self.rank * iterval : self.rank * iterval + iterval, :
                    ]
                    logits_per_x = logits_per_x * local_mask_x
                    local_mask_y = mask.T[
                        self.rank * iterval : self.rank * iterval + iterval, :
                    ]
                    logits_per_y = logits_per_y * local_mask_y

            else:
                logits_per_x = logit_scale * all_x_features @ all_y_features.T
                if mask is not None:
                    logits_per_x = logits_per_x * mask
                logits_per_y = logits_per_x.T

        else:  # world_size == 1
            mask = None
            if (
                x_labels is not None and y_labels is not None and self.use_mask
            ):  # mask flag here
                assert x_labels.shape == y_labels.shape
                if x_labels.ndim == 1:
                    x_labels = x_labels.unsqueeze(0)
                    y_labels = y_labels.unsqueeze(0)
                mask = torch.logical_or(
                    torch.logical_not(x_labels.T == y_labels),  # n_x x n_y
                    torch.eye(
                        x_features.shape[0], device=x_features.device, dtype=torch.bool
                    ),
                )

            logits_per_x = logit_scale * x_features @ y_features.T
            logits_per_y = logit_scale * y_features @ x_features.T

            if mask is not None:
                logits_per_x = logits_per_x * mask
                logits_per_y = logits_per_y * mask.T

        return logits_per_x, logits_per_y

    def forward(
        self,
        x_features,
        y_features,
        logit_scale,
        x_labels=None,
        y_labels=None,
        output_dict=False,
        key="image-text",
    ):
        device = x_features.device
        logits_per_x, logits_per_y = self.get_logits(
            x_features, y_features, logit_scale, x_labels, y_labels
        )

        labels = self.get_ground_truth(device, logits_per_x.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_x, labels)
            + F.cross_entropy(logits_per_y, labels)
        ) / 2

        return {key: total_loss} if output_dict else total_loss


class TriClipLossLabelMask(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_mask=False,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_mask = use_mask
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(
        self, x_features, y_features, logit_scale, x_labels=None, y_labels=None
    ):
        if self.world_size > 1:
            all_x_features, all_y_features = gather_features(
                x_features,
                y_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            all_x_labels, all_y_labels = None, None
            mask = None
            if (
                x_labels is not None and y_labels is not None and self.use_mask
            ):  # mask flag here
                assert x_labels.shape == y_labels.shape
                # if x_labels.ndim == 1:
                #     x_labels = x_labels.unsqueeze(0)
                #     y_labels = y_labels.unsqueeze(0)
                all_x_labels, all_y_labels = gather_features(
                    x_labels,
                    y_labels,
                    self.local_loss,
                    False,
                    self.rank,
                    self.world_size,
                    self.use_horovod,
                )
                if all_x_labels.ndim == 1:
                    all_x_labels = all_x_labels.unsqueeze(0)
                    all_y_labels = all_y_labels.unsqueeze(0)
                mask = torch.logical_or(
                    torch.logical_not(all_x_labels.T == all_y_labels),  # n_x x n_y
                    torch.eye(
                        all_x_features.shape[0],
                        device=x_features.device,
                        dtype=torch.bool,
                    ),
                )

            if self.local_loss:
                logits_per_x = logit_scale * x_features @ all_y_features.T
                logits_per_y = logit_scale * y_features @ all_x_features.T
                if mask is not None:
                    iterval = x_features.shape[0]
                    local_mask_x = mask[
                        self.rank * iterval : self.rank * iterval + iterval, :
                    ]
                    logits_per_x = logits_per_x * local_mask_x
                    local_mask_y = mask.T[
                        self.rank * iterval : self.rank * iterval + iterval, :
                    ]
                    logits_per_y = logits_per_y * local_mask_y

            else:
                logits_per_x = logit_scale * all_x_features @ all_y_features.T
                if mask is not None:
                    logits_per_x = logits_per_x * mask
                logits_per_y = logits_per_x.T

        else:  # world_size == 1
            mask = None
            if (
                x_labels is not None and y_labels is not None and self.use_mask
            ):  # mask flag here
                assert x_labels.shape == y_labels.shape
                if x_labels.ndim == 1:
                    x_labels = x_labels.unsqueeze(0)
                    y_labels = y_labels.unsqueeze(0)
                mask = torch.logical_or(
                    torch.logical_not(x_labels.T == y_labels),  # n_x x n_y
                    torch.eye(
                        x_features.shape[0], device=x_features.device, dtype=torch.bool
                    ),
                )

            logits_per_x = logit_scale * x_features @ y_features.T
            logits_per_y = logit_scale * y_features @ x_features.T

            if mask is not None:
                logits_per_x = logits_per_x * mask
                logits_per_y = logits_per_y * mask.T

        return logits_per_x, logits_per_y

    def forward(
        self,
        image_features,
        text_features,
        visual_features,
        logit_scale,
        image_labels=None,
        text_labels=None,
        visual_labels=None,
        output_dict=False,
    ):
        device = image_features.device
        IV_logits_per_image, IV_logits_per_visual = self.get_logits(
            image_features, visual_features, logit_scale, image_labels, visual_labels
        )
        TV_logits_per_text, TV_logits_per_visual = self.get_logits(
            text_features, visual_features, logit_scale, text_labels, visual_labels
        )

        labels = self.get_ground_truth(device, IV_logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(IV_logits_per_image, labels)
            + F.cross_entropy(IV_logits_per_visual, labels)
            + F.cross_entropy(TV_logits_per_text, labels)
            + F.cross_entropy(TV_logits_per_visual, labels)
        ) / 2

        return {"tri_contrastive_loss": total_loss} if output_dict else total_loss
