import time
import json
import collections
import logging
import einops
import torch
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm

from open_clip import (
    get_input_dtype,
    get_tokenizer,
    build_zero_shot_classifier,
    IMAGENET_CLASSNAMES,
    OPENAI_IMAGENET_TEMPLATES,
)
from .precision import get_autocast
from open_clip.utils import (
    AverageMeter,
    ProgressMeter,
    scaled_all_reduce,
    get_model,
    is_dist_avail_and_initialized,
    concat_all_gather,
    new_islice,
    all_gather,
)  # these two are borrowed from ONE-PEACE for audio-related tasks
from open_clip.constants import PC_META_DATA_DIR
from open_clip.modal_depth.data.scene_cls_template import SCENE_CLS_TEMPLATE
from open_clip.modal_audio.data.sound_cls_template import SOUND_AS_IMAGE_TEMPLATE
from open_clip.metrics import Accuracy, MAP, Recall


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def acc(output, target, topk=(1,)):
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


def cond_acc(output, target, idx_mapping, merge_idx=100, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        for idx in idx_mapping:
            target[target == idx] = merge_idx
            pred[pred == idx] = merge_idx

        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().max(dim=0)[0].sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        top1, top5, n = 0.0, 0.0, 0.0
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = (
                    output["image_features"] if isinstance(output, dict) else output[0]
                )
                logits = 100.0 * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = top1 / n
    top5 = top5 / n
    return top1, top5


def zero_shot_eval(model, data, epoch, args):
    if "imagenet-val" not in data and "imagenet-v2" not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info("Starting zero-shot imagenet.")

    logging.info("Building zero-shot classifier")
    autocast = get_autocast(args.precision)
    with autocast():
        tokenizer = get_tokenizer(args.model)
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

    logging.info("Using classifier")
    results = {}
    if "imagenet-val" in data:
        top1, top5 = run(model, classifier, data["imagenet-val"].dataloader, args)
        results["imagenet-zeroshot-val-top1"] = top1
        results["imagenet-zeroshot-val-top5"] = top5
    if "imagenet-v2" in data:
        top1, top5 = run(model, classifier, data["imagenet-v2"].dataloader, args)
        results["imagenetv2-zeroshot-val-top1"] = top1
        results["imagenetv2-zeroshot-val-top5"] = top5

    logging.info("Finished zero-shot imagenet.")

    return results


def test_zeroshot_3d_core(test_loader, model, tokenizer, args=None):
    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(test_loader), [batch_time, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model = get_model(model)
    model.eval()

    print("=> encoding captions")
    with open(f"{PC_META_DATA_DIR}/templates.json") as f:
        templates = json.load(f)[args.val_data_prompt]

    with open(f"{PC_META_DATA_DIR}/labels.json") as f:
        labels = json.load(f)[args.val_data]

    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).cuda(args.device, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            class_embeddings = model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        end = time.time()
        per_class_stats = collections.defaultdict(int)
        per_class_correct_top1 = collections.defaultdict(int)
        per_class_correct_top5 = collections.defaultdict(int)

        for i, batch in enumerate(test_loader):
            pc, target, target_name = batch["pc"], batch["label"], batch["class_name"]
            for name in target_name:
                per_class_stats[name] += 1

            pc = pc.cuda(args.device, non_blocking=True)
            if isinstance(target, list):
                target = torch.LongTensor(target)
            target = target.cuda(args.device, non_blocking=True)

            # encode pc
            if hasattr(model, "encode_visual"):
                pc_features = model.encode_visual(pc)
            else:
                pc_features = model.encode_image(pc)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_pc = pc_features @ text_features.t()

            # measure accuracy and record loss
            (acc1, acc5), correct = acc(logits_per_pc, target, topk=(1, 5))
            # TODO: fix the all reduce for the correct variable, assuming only one process for evaluation!
            acc1, acc5 = scaled_all_reduce([acc1, acc5])
            top1.update(acc1.item(), pc.size(0))
            top5.update(acc5.item(), pc.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            top1_accurate = correct[:1].squeeze()
            top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()
            for idx, name in enumerate(target_name):
                if top1_accurate[idx].item():
                    per_class_correct_top1[name] += 1
                if top5_accurate[idx].item():
                    per_class_correct_top5[name] += 1

            if i % args.log_every_n_steps == 0:
                progress.display(i)

        top1_accuracy_per_class = {}
        top5_accuracy_per_class = {}
        for name in per_class_stats.keys():
            top1_accuracy_per_class[name] = (
                per_class_correct_top1[name] / per_class_stats[name]
            )
            top5_accuracy_per_class[name] = (
                per_class_correct_top5[name] / per_class_stats[name]
            )

        top1_accuracy_per_class = collections.OrderedDict(top1_accuracy_per_class)
        top5_accuracy_per_class = collections.OrderedDict(top5_accuracy_per_class)
        print(",".join(top1_accuracy_per_class.keys()))
        print(",".join([str(value) for value in top1_accuracy_per_class.values()]))
        print(",".join([str(value) for value in top5_accuracy_per_class.values()]))

    progress.synchronize()
    logging.info(f"0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}")
    return dict(modelnet40={"acc1": top1.avg, "acc5": top5.avg})


def test_rgbd_cls_single(test_loader, model, tokenizer, dataset_name, args=None):
    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(test_loader), [batch_time, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model = get_model(model)
    model.eval()

    print("=> encoding captions w/ templates")
    templates = SCENE_CLS_TEMPLATE
    test_dataset = test_loader.dataset
    labels = test_dataset.idx2label

    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t(l) for t in templates]
            texts = tokenizer(texts).cuda(args.device, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            class_embeddings = model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        end = time.time()

        for i, batch in enumerate(test_loader):
            depth, target, target_name = (
                batch["depth"],
                batch["label"],
                batch["cleaned_label"],
            )

            depth = depth.cuda(args.device, non_blocking=True)
            if isinstance(target, list):
                target = torch.LongTensor(target)
            target = target.cuda(args.device, non_blocking=True)

            # encode visual
            if hasattr(model, "encode_visual"):
                depth_features = model.encode_visual(depth)
            else:
                depth_features = model.encode_image(depth)
            depth_features = depth_features / depth_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_depth = depth_features @ text_features.t()

            # measure accuracy and record loss
            if hasattr(test_dataset, "other_idx"):
                merge_idx = test_dataset.other_idx
                mapping_indices = test_dataset.map_to_others_idx
                (acc1, acc5), correct = cond_acc(
                    logits_per_depth, target, mapping_indices, merge_idx, topk=(1, 5)
                )
            else:
                (acc1, acc5), correct = acc(logits_per_depth, target, topk=(1, 5))

            # TODO: fix the all reduce for the correct variable, assuming only one process for evaluation!
            acc1, acc5 = scaled_all_reduce([acc1, acc5])
            top1.update(acc1.item(), depth.size(0))
            top5.update(acc5.item(), depth.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            top1_accurate = correct[:1].squeeze()
            top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()

    if args.distributed:
        torch.distributed.barrier()

    progress.synchronize()
    logging.info(
        f"[{dataset_name}] : 0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}"
    )
    return {"acc1": top1.avg, "acc5": top5.avg}


def test_rgbd_cls_core(testloaders, model, tokenizer, args=None):
    metrics = dict()
    if isinstance(testloaders, dict):
        for dname, dloader in testloaders.items():
            m = test_rgbd_cls_single(dloader, model, tokenizer, dname, args)
            metrics.update({dname: m})
    else:
        m = test_rgbd_cls_single(
            testloaders, model, tokenizer, "Eval RGBD-CLS Dataset", args
        )
        metrics.update({"Single Eval": m})
    return metrics


def test_imgret_single(testloader, model, dataset_name="Eval set", args=None):
    model = get_model(model)
    model.eval()

    img_feats = []
    text_feats = []
    image_ids = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(testloader), total=len(testloader)):
            image, text, vid = batch["image"], batch["caption"], batch["image_id"]

            vfeat = model.encode_image(image)
            vfeat = vfeat / vfeat.norm(dim=-1, keepdim=True)

            tfeat = model.encode_text(text)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

            vid = torch.LongTensor(vid).to(vfeat.device)

            img_feats.append(vfeat)
            text_feats.append(tfeat)
            image_ids.append(vid)

    # collect visual features
    visual_feats = {}
    for feats, ids in zip(img_feats, image_ids):
        for i, _idx in enumerate(ids):
            idx = _idx.item()
            if idx not in visual_feats:
                visual_feats[idx] = feats[i]

    tiids = torch.cat(image_ids, dim=0)
    iids = []
    sorted_tensors = []
    for key in sorted(visual_feats.keys()):
        sorted_tensors.append(visual_feats[key].view(1, -1))
        iids.append(key)

    img_feats = torch.cat(sorted_tensors, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    iids = torch.LongTensor(iids).to(img_feats.device)

    if is_dist_avail_and_initialized():  # in get data, use distributed sampler
        torch.distributed.barrier()
        iids = concat_all_gather(iids)
        tiids = concat_all_gather(tiids)
        img_feats = concat_all_gather(img_feats)
        text_feats = concat_all_gather(text_feats)

    scores = img_feats @ text_feats.t()

    print("scores: {}".format(scores.size()))
    print("iids: {}".format(iids.size()))
    print("tiids: {}".format(tiids.size()))

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)

    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    eval_result = {
        "tr_r10": tr_r10.item() * 100.0,
        "tr_r5": tr_r5.item() * 100.0,
        "tr_r1": tr_r1.item() * 100.0,
        "ir_r10": ir_r10.item() * 100.0,
        "ir_r5": ir_r5.item() * 100.0,
        "ir_r1": ir_r1.item() * 100.0,
        "average_score": 100.0
        * (tr_r1 + tr_r5 + tr_r10 + ir_r1 + ir_r5 + ir_r10).item()
        / 6.0,
    }

    logging.info("**  %s ** Eval result = %s" % (dataset_name, json.dumps(eval_result)))
    return eval_result


def test_vidret_single(testloader, model, dataset_name="Eval set", args=None):
    model = get_model(model)
    model.eval()
    visual_encode_fn = (
        model.encode_visual if hasattr(model, "encode_visual") else model.encode_image
    )

    vis_feats = []
    text_feats = []
    image_ids = []

    zs_mean_pool = args.vid_dire_mean_pool
    n_frames = args.n_frames

    with torch.no_grad():
        for i, batch in tqdm(enumerate(testloader), total=len(testloader)):
            video, text, vid = batch["video"], batch["caption"], batch["image_id"]

            vfeat = visual_encode_fn(video)
            if zs_mean_pool:
                vfeat = einops.rearrange(vfeat, "(b t) ... -> b t ...", t=n_frames)
                vfeat = torch.mean(vfeat, dim=1)
            vfeat = vfeat / vfeat.norm(dim=-1, keepdim=True)

            tfeat = model.encode_text(text)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

            vid = torch.LongTensor(vid).to(vfeat.device)

            vis_feats.append(vfeat)
            text_feats.append(tfeat)
            image_ids.append(vid)

    # ###
    visual_feats = {}
    for feats, ids in zip(vis_feats, image_ids):
        for i, _idx in enumerate(ids):
            idx = _idx.item()
            if idx not in visual_feats:
                visual_feats[idx] = feats[i]

    tiids = torch.cat(image_ids, dim=0)
    iids = []
    sorted_tensors = []
    for key in sorted(visual_feats.keys()):
        sorted_tensors.append(visual_feats[key].view(1, -1))
        iids.append(key)

    video_feats = torch.cat(sorted_tensors, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    iids = torch.LongTensor(iids).to(video_feats.device)

    if is_dist_avail_and_initialized():  # in get data, use distributed sampler
        torch.distributed.barrier()
        iids = concat_all_gather(iids)
        tiids = concat_all_gather(tiids)
        video_feats = concat_all_gather(video_feats)
        text_feats = concat_all_gather(text_feats)

    scores = video_feats @ text_feats.t()

    print("scores: {}".format(scores.size()))
    print("iids: {}".format(iids.size()))
    print("tiids: {}".format(tiids.size()))

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)

    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    eval_result = {
        "tr_r10": tr_r10.item() * 100.0,
        "tr_r5": tr_r5.item() * 100.0,
        "tr_r1": tr_r1.item() * 100.0,
        "ir_r10": ir_r10.item() * 100.0,
        "ir_r5": ir_r5.item() * 100.0,
        "ir_r1": ir_r1.item() * 100.0,
        "average_score": 100.0
        * (tr_r1 + tr_r5 + tr_r10 + ir_r1 + ir_r5 + ir_r10).item()
        / 6.0,
    }

    logging.info("**  %s ** Eval result = %s" % (dataset_name, json.dumps(eval_result)))
    return eval_result


def test_vidret_core(testloaders, model, tokenizer, args=None):
    if isinstance(testloaders, dict):
        for dname, dloader in testloaders.items():
            test_vidret_single(dloader, model, dname, args)
    else:
        test_vidret_single(testloaders, model, "Eval VidRet Dataset", args)


def test_audio_single_map(
    testloader, model, tokenizer, dataset_name="Eval Audio mAP", args=None
):
    model = get_model(model)
    model.eval()

    metric = MAP()
    metric.initialize()

    labels = testloader.dataset.idx2label

    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t(l) for t in SOUND_AS_IMAGE_TEMPLATE]
            texts = tokenizer(texts).cuda(args.device, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            class_embeddings = model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        # audio forward
        for i, batch in tqdm(
            enumerate(testloader),
            total=len(testloader),
            desc=f"{dataset_name}@mAP feature calc",
        ):
            ids, audio, targets = batch["id"], batch["audio"], batch["target"]

            audio = audio.cuda(args.device, non_blocking=True)
            targets = targets.cuda(args.device, non_blocking=True)
            ids = torch.tensor(ids).to(args.device)

            # encode visual
            afeat = None
            if audio.ndim == 4:
                # bsz x n_clip x tdim x fdim
                n_clip = audio.size(1)
                audio = einops.rearrange(audio, "b n ... -> (b n) ...")
                afeat = model.encode_visual(audio)
                afeat = einops.rearrange(afeat, "(b n) ... -> b n ...", n=n_clip)
                afeat = torch.mean(afeat, dim=1)
            elif audio.ndim == 3:
                afeat = model.encode_visual(audio)
            audio_features = afeat / afeat.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_audio = audio_features @ text_features.t()
            metric.compute(ids, logits_per_audio, targets)

    if is_dist_avail_and_initialized():
        dist.barrier()

    stats = metric.merge_results()
    # hack, `acc1` field for saving best checkpoint
    stats["acc1"] = stats["map"]
    logging.info(f'[{dataset_name}] : 0-shot * mAP {stats["map"]}')

    return stats


def test_audio_single_cls(
    testloader, model, tokenizer, dataset_name="Eval Audio Cls", args=None
):
    model = get_model(model)
    model.eval()

    metric = Accuracy()
    metric.initialize()

    print("=> encoding captions w/ templates")
    test_dataset = testloader.dataset
    labels = test_dataset.idx2label

    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t(l) for t in SOUND_AS_IMAGE_TEMPLATE]
            texts = tokenizer(texts).cuda(args.device, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            class_embeddings = model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        # audio forward
        for i, batch in tqdm(
            enumerate(testloader), total=len(testloader), desc=f"{dataset_name} @ CLS"
        ):
            ids, audio, targets = batch["id"], batch["audio"], batch["label"]

            audio = audio.cuda(args.device, non_blocking=True)
            if isinstance(targets, list):
                targets = torch.LongTensor(targets)
            targets = targets.cuda(args.device, non_blocking=True)
            ids = torch.tensor(ids).cuda(args.device, non_blocking=True)

            # encode visual
            afeat = None
            if audio.ndim == 4:
                # bsz x n_clip x tdim x fdim
                n_clip = audio.size(1)
                audio = einops.rearrange(audio, "b n ... -> (b n) ...")
                afeat = model.encode_visual(audio)
                afeat = einops.rearrange(afeat, "(b n) ... -> b n ...", n=n_clip)
                afeat = torch.mean(afeat, dim=1)
            elif audio.ndim == 3:
                afeat = model.encode_visual(audio)
            audio_features = afeat / afeat.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_audio = audio_features @ text_features.t()
            metric.compute(ids, logits_per_audio, targets)

    stats = metric.merge_results()
    # hack: `acc1` field for saving best checkpoint
    stats["acc1"] = stats["accuracy"]

    logging.info(f'[{dataset_name}] : Classification * Acc@1 {stats["accuracy"]}')
    return stats


def test_audio_single_ret(
    testloader, model, tokenizer, dataset_name="Eval Audio Ret", args=None
):
    model = get_model(model)
    model.eval()
    metric = Recall()

    dataset = testloader.dataset
    text_ids = dataset.text_ids
    texts = dataset.texts
    stats = {}

    with torch.no_grad():
        text_ids = torch.tensor(text_ids).cuda()
        text_cnt = len(text_ids)

        if is_dist_avail_and_initialized():
            slice_id = dist.get_rank()
            slice_count = dist.get_world_size()
        else:
            slice_id = 0
            slice_count = 1
        batch_sampler = new_islice(range(text_cnt), slice_id, text_cnt, slice_count)
        start_idx = batch_sampler[0]
        end_idx = batch_sampler[-1] + 1

        text_logits_list = []
        for i in range(start_idx, end_idx, 50):
            samples_list = []
            for text in texts[i : min(i + 50, end_idx)]:
                # text = text --> seems no need for template for retrieval
                samples_list.append(text)
            tokenized_captions = tokenizer(samples_list).cuda(
                args.device, non_blocking=True
            )
            text_logits = model.encode_text(tokenized_captions)
            text_logits_list.append(text_logits)

        text_logits = torch.cat(text_logits_list, dim=0)
        text_logits = (
            all_gather(text_logits) if is_dist_avail_and_initialized() else text_logits
        )
        metric.initialize(text_ids=text_ids, text_logits=text_logits)

        # forward audio
        for i, batch in tqdm(enumerate(testloader), total=len(testloader)):
            audio, audio_ids = batch["audio"], batch["uniq_id"]
            if isinstance(audio_ids, list):
                audio_ids = torch.tensor(audio_ids).to(args.device)
            afeat = None
            if audio.ndim == 4:
                # bsz x n_clip x tdim x fdim
                n_clip = audio.size(1)
                audio = einops.rearrange(audio, "b n ... -> (b n) ...")
                afeat = model.encode_visual(audio)
                afeat = einops.rearrange(afeat, "(b n) ... -> b n ...", n=n_clip)
                afeat = torch.mean(afeat, dim=1)
            elif audio.ndim == 3:
                afeat = model.encode_visual(audio)
            audio_logits = afeat / afeat.norm(dim=-1, keepdim=True)
            metric.compute(audio_ids, audio_logits)

        stats = metric.merge_results()

    for key in list(stats.keys()):
        if key.startswith("img"):
            stats[key.replace("img", "audio")] = stats[key]
            del stats[key]
    # hack: use `acc1` field to save best checkpoint, retrieval scale up 100 for printing result
    stats["acc1"] = (
        stats["txt_r1"]
        + stats["txt_r5"]
        + stats["txt_r10"]
        + stats["audio_r1"]
        + stats["audio_r5"]
        + stats["audio_r10"]
    ) / (6 * 100.0)
    logging.info("**  %s ** Eval result = %s" % (dataset_name, json.dumps(stats)))

    return stats


def test_audiotasks_core(testloaders, model, tokenizer, args=None):
    metrics = dict()
    test_fn_mapping = {
        "map": test_audio_single_map,
        "acc": test_audio_single_cls,
        "recall": test_audio_single_ret,
    }

    if isinstance(testloaders, dict):
        for dname, dloader in testloaders.items():
            eval_metric_key = dloader.dataset.eval_metric.lower()
            m = test_fn_mapping[eval_metric_key](dloader, model, tokenizer, dname, args)
            metrics.update({dname: m})
    else:
        eval_metric_key = testloaders.dataset.eval_metric.lower()
        m = test_fn_mapping[eval_metric_key](
            testloaders, model, tokenizer, "Eval Audio Dataset", args
        )
        metrics.update({"Single Eval": m})
    return metrics


def test_tactle_cls_single(
    test_loader, model, tokenizer, dataset_name="Eval tag cls", args=None
):
    test_dataset = test_loader.dataset
    split = test_dataset.split
    labels = test_dataset.idx2label
    do_test_acc5 = len(labels) >= 5
    print(test_dataset.label2idx)

    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = (
        ProgressMeter(len(test_loader), [batch_time, top1, top5], prefix="Test: ")
        if do_test_acc5
        else ProgressMeter(len(test_loader), [batch_time, top1], prefix="Test: ")
    )

    # switch to evaluate mode
    model = get_model(model)
    model.eval()

    print("=> encoding captions w/ templates")
    templates = (
        [lambda c: f"the meterial is {c}.", lambda c: f"this type of material is {c}."]
        if "rough" in split or "hard" in split
        else [lambda c: f"an image of {c}.", lambda c: f"a tactile image of {c}."]
    )

    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t(l) for t in templates]
            texts = tokenizer(texts).cuda(args.device, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            class_embeddings = model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        end = time.time()

        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            tactile, target = batch["tactile"], batch["label"]

            tactile = tactile.cuda(args.device, non_blocking=True)
            if isinstance(target, list):
                target = torch.LongTensor(target)
            target = target.cuda(args.device, non_blocking=True)

            # encode visual
            if hasattr(model, "encode_visual"):
                tactile_features = model.encode_visual(tactile)
            else:
                tactile_features = model.encode_image(tactile)
            tactile_features = tactile_features / tactile_features.norm(
                dim=-1, keepdim=True
            )

            # cosine similarity as logits
            logits_per_tactile = tactile_features @ text_features.t()

            # measure accuracy and record loss
            if do_test_acc5:
                (acc1, acc5), correct = acc(logits_per_tactile, target, topk=(1, 5))
                # TODO: fix the all reduce for the correct variable, assuming only one process for evaluation!
                acc1, acc5 = scaled_all_reduce([acc1, acc5])
                top1.update(acc1.item(), tactile.size(0))
                top5.update(acc5.item(), tactile.size(0))
            else:
                (acc1, dummy), correct = acc(logits_per_tactile, target, topk=(1, 2))
                acc1, _ = scaled_all_reduce([acc1, dummy])
                top1.update(acc1.item(), tactile.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            top1_accurate = correct[:1].squeeze()
            if do_test_acc5:
                top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()

    if args.distributed:
        torch.distributed.barrier()

    progress.synchronize()
    logging.info(
        f"[{dataset_name}] : 0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}"
    ) if do_test_acc5 else logging.info(
        f"[{dataset_name}] : 0-shot * Acc@1 {top1.avg:.3f}"
    )

    return {"acc1": top1.avg, "acc5": top5.avg} if do_test_acc5 else {"acc1": top1.avg}


def test_tactiletasks_core(testloaders, model, tokenizer, args=None):
    metrics = dict()
    if isinstance(testloaders, dict):
        for dname, dloader in testloaders.items():
            m = test_tactle_cls_single(dloader, model, tokenizer, dname, args)
            metrics.update({dname: m})
    else:
        m = test_tactle_cls_single(testloaders, model, tokenizer, "Eval Tac-CLS", args)
        metrics.update({"Single Eval": m})
    return metrics


def test_eeg_cls_single(
    test_loader, model, tokenizer, dataset_name="Eval EEG cls", args=None
):
    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(test_loader), [batch_time, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model = get_model(model)
    model.eval()

    print("=> encoding captions w/ templates")
    templates = (
        lambda c: f"a photo of {c}.",
        lambda c: f"an image of {c}.",
        lambda c: f"a picture of {c}.",
    )
    test_dataset = test_loader.dataset
    labels = test_dataset.idx2label

    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t(l) for t in templates]
            texts = tokenizer(texts).cuda(args.device, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            class_embeddings = model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        end = time.time()

        for i, batch in enumerate(test_loader):
            eeg, target = batch["eeg"], batch["label"]

            eeg = eeg.cuda(args.device, non_blocking=True)
            if isinstance(target, list):
                target = torch.LongTensor(target)
            target = target.cuda(args.device, non_blocking=True)

            # encode visual
            if hasattr(model, "encode_visual"):
                eeg_features = model.encode_visual(eeg)
            else:
                eeg_features = model.encode_image(eeg)
            eeg_features = eeg_features / eeg_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_eeg = eeg_features @ text_features.t()

            # measure accuracy and record loss
            (acc1, acc5), correct = acc(logits_per_eeg, target, topk=(1, 5))

            # TODO: fix the all reduce for the correct variable, assuming only one process for evaluation!
            acc1, acc5 = scaled_all_reduce([acc1, acc5])
            top1.update(acc1.item(), eeg.size(0))
            top5.update(acc5.item(), eeg.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            top1_accurate = correct[:1].squeeze()
            top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()

    if args.distributed:
        torch.distributed.barrier()

    progress.synchronize()
    logging.info(
        f"[{dataset_name}] : 0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}"
    )
    return {"acc1": top1.avg, "acc5": top5.avg}


def test_eegtasks_core(testloaders, model, tokenizer, args=None):
    metrics = dict()
    if isinstance(testloaders, dict):
        for dname, dloader in testloaders.items():
            m = test_eeg_cls_single(dloader, model, tokenizer, dname, args)
            metrics.update({dname: m})
    else:
        m = test_eeg_cls_single(testloaders, model, tokenizer, "Eval EEG-CLS", args)
        metrics.update({"Single Eval": m})
    return metrics


def test_linprob_single(
    test_loader, model, tokenizer, dataset_name="Linear Probe CLS", args=None
):
    test_dataset = test_loader.dataset
    split = test_dataset.split
    labels = test_dataset.idx2label
    do_test_acc5 = len(labels) >= 5
    print(test_dataset.label2idx)

    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = (
        ProgressMeter(len(test_loader), [batch_time, top1, top5], prefix="Test: ")
        if do_test_acc5
        else ProgressMeter(len(test_loader), [batch_time, top1], prefix="Test: ")
    )

    # switch to evaluate mode
    model = get_model(model)
    model.eval()

    with torch.no_grad():
        end = time.time()

        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            x, target = batch[args.v_key], batch["label"]

            x = x.cuda(args.device, non_blocking=True)
            if isinstance(target, list):
                target = torch.LongTensor(target)
            target = target.cuda(args.device, non_blocking=True)

            # calculate logits
            logits = model(x)

            # measure accuracy and record loss
            if do_test_acc5:
                (acc1, acc5), correct = acc(logits, target, topk=(1, 5))
                # TODO: fix the all reduce for the correct variable, assuming only one process for evaluation!
                acc1, acc5 = scaled_all_reduce([acc1, acc5])
                top1.update(acc1.item(), x.size(0))
                top5.update(acc5.item(), x.size(0))
            else:
                (acc1, dummy), correct = acc(logits, target, topk=(1, 2))
                acc1, _ = scaled_all_reduce([acc1, dummy])
                top1.update(acc1.item(), x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            top1_accurate = correct[:1].squeeze()
            if do_test_acc5:
                top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()

    if args.distributed:
        torch.distributed.barrier()

    progress.synchronize()
    logging.info(
        f"[{dataset_name}] : Linear Probe * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}"
    ) if do_test_acc5 else logging.info(
        f"[{dataset_name}] : Linear Probe * Acc@1 {top1.avg:.3f}"
    )

    return {"acc1": top1.avg, "acc5": top5.avg} if do_test_acc5 else {"acc1": top1.avg}
