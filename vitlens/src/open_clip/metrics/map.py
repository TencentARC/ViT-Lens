import torch
import torch.distributed as dist

import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from .base_metric import BaseMetric
from ..utils import all_gather, concat_all_gather


class MAP(BaseMetric):

    def __init__(self):
        super().__init__()

    def initialize(self):
        self.logits = torch.FloatTensor([]).cuda()
        self.targets = torch.FloatTensor([]).cuda()
        self.ids = torch.LongTensor([]).cuda()

    def compute(self, ids, logits, targets):
        self.ids = torch.cat([self.ids, ids], dim=0)
        self.logits = torch.cat([self.logits, logits], dim=0)
        self.targets = torch.cat([self.targets, targets], dim=0)

    def merge_results(self, output_predict=False):
        if dist.is_initialized():
            ids = all_gather(self.ids)
            preds = all_gather(self.logits)
            targets = all_gather(self.targets)
        else:
            ids = self.ids
            preds = self.logits
            targets = self.targets
        preds = torch.sigmoid(preds).cpu().numpy()
        
        if targets.ndim != preds.ndim:
            if targets.size(0) == 1:
                targets.squeeze_(0)
            if targets.size(1) == 1:
                targets.squeeze_(1)
        targets = targets.cpu().numpy()

        predict_results = {}
        if output_predict:
            for idx, pred in tqdm(zip(ids.cpu().tolist(), preds.tolist()), total=len(ids), desc="mAP iter"):
                predict_results[idx] = pred

        return {
            'map': np.mean(average_precision_score(targets, preds, average=None)),
            'map_cnt': len(targets),
            'predict_results': predict_results
        }