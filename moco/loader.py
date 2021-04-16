# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import torch
import molgrid

class TwoMolDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        self.moldataset = molgrid.MolDataset(*args, **kwargs)

    def __len__(self):
        return len(self.moldataset)
    
    def __getitem__(self, idx):
        q, labels1 = self.moldataset[idx]
        k, labels2 = self.moldataset[idx]
        assert labels1 == labels2, f"Somehow the data has different labels"
        return [q, k], labels

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
