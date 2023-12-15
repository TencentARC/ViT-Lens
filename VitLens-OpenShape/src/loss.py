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

        i2v_acc = (IV_logits_per_image.argmax(dim=1) == labels).float().mean()
        v2i_acc = (IV_logits_per_visual.argmax(dim=1) == labels).float().mean()
        t2v_acc = (TV_logits_per_text.argmax(dim=1) == labels).float().mean()
        v2t_acc = (TV_logits_per_text.argmax(dim=1) == labels).float().mean()

        i_constra_loss = (
            F.cross_entropy(IV_logits_per_image, labels)
            + F.cross_entropy(IV_logits_per_visual, labels)
        ) / 2
        t_constra_loss = (
            F.cross_entropy(TV_logits_per_text, labels)
            + F.cross_entropy(TV_logits_per_visual, labels)
        ) / 2

        total_loss = i_constra_loss + t_constra_loss

        return (
            {
                "contrastive_loss": total_loss,
                "i_contra_loss": i_constra_loss,
                "t_contra_loss": t_constra_loss,
                "i2v_acc": i2v_acc,
                "v2i_acc": v2i_acc,
                "t2v_acc": t2v_acc,
                "v2t_acc": v2t_acc,
            }
            if output_dict
            else total_loss
        )
