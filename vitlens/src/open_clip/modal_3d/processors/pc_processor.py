import os
import random
import torch
import numpy as np
from omegaconf import OmegaConf


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def pc_norm(pc):
    """pc: NxC, return NxC"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def random_sample(pc, num):
    permutation = np.arange(pc.shape[0])
    np.random.shuffle(permutation)
    pc = pc[permutation[:num]]
    return pc


class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    def build(self, **kwargs):
        cfg = OmegaConf.create(kwargs)

        return self.from_config(cfg)


class PCProcessorEval(BaseProcessor):
    def __init__(self, npoint, uniform, idendity=False):
        self.npoint = npoint
        self.uniform = uniform
        self.idendity = idendity

    def set_attr(**kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def __call__(self, pc):
        if self.idendity:
            pc = torch.from_numpy(pc)
            return pc
        else:
            if self.uniform and self.npoint < pc.shape[0]:
                pc = farthest_point_sample(pc, self.npoint)
            else:
                pc = random_sample(pc, self.npoint)
            pc = pc_norm(pc)
            pc = torch.from_numpy(pc)

            return pc

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        npoint = cfg.get("npoint", 8192)
        uniform = cfg.get("uniform", True)
        return cls(npoint=npoint, uniform=uniform)
