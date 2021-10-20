import os
import shutil
import time
import pprint
import numpy as np
import torchvision
import torch.nn.functional as F

import torch
from samplers import CategoriesSampler
from torch.utils.data import DataLoader, Subset
import torch.nn as nn

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def get_collate(args, batch_transform=None):
    def mycollate(batch):
        collated = torch.utils.data.dataloader.default_collate(batch)
        if batch_transform is not None:
            collated = batch_transform(collated, args)
        return collated
    return mycollate

query_transform = torchvision.transforms.Compose([
        Cutout(21)
])

def query_augment(batch, args):
    batch[args.way * args.shot:] = query_transform(batch[args.way * args.shot:])

def get_dataloader(args):
    if args.dataset == 'MiniImageNet':
        from mini_imagenet import MiniImageNet as Dataset
    else:
        raise ValueError('Non-supported Dataset.')

    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(testset.label,
                                2000, # args.num_eval_episodes,
                                args.way, args.shot + args.test_query)
    test_loader = DataLoader(dataset=testset,
                                batch_sampler=test_sampler,
                                num_workers=args.num_workers,
                                pin_memory=True)

    num_device = torch.cuda.device_count()
    num_episodes = args.episodes_per_epoch*num_device if args.multi_gpu else args.episodes_per_epoch
    num_workers=args.num_workers*num_device if args.multi_gpu else args.num_workers
    trainset = Dataset('train', args, augment=args.augment)
    args.num_class = trainset.num_class
    train_sampler = CategoriesSampler(trainset.label,
                                    num_episodes,
                                    args.way,
                                    args.shot + args.query)
    train_loader = DataLoader(dataset=trainset,
                                num_workers=num_workers,
                                batch_sampler=train_sampler,
                                pin_memory=True)

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label,
                            args.num_eval_episodes,
                            args.way, args.shot + args.test_query)
    val_loader = DataLoader(dataset=valset,
                            batch_sampler=val_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)


    return train_loader, val_loader, test_loader

def hse_loss(prototypes):
    # prototypes = (num_batch, num_proto, num_emb)
    num_proto = prototypes.shape[1]
    loss = 0
    for i in range(num_proto):
        for j in range(num_proto):
            if i != j:
                loss += torch.log((F.normalize(prototypes[0][i], dim=0) - F.normalize(prototypes[0][j], dim=0)).pow(2).sum().sqrt().pow(-1))

    return loss