import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from models.convnet import Convnet
from models.resnet import ResNet
from models.protonet import ProtoNet
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric, get_dataloader
import torch.nn.functional as F
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load', default='./save/proto-1/max-acc.pth')
    parser.add_argument('--batch', type=int, default=2000)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=30)
    parser.add_argument('--ind-block', type=int, default=0)
    parser.add_argument('--cycles', type=int, default=0)
    parser.add_argument('--model', type=str, choices=['Conv64', 'ResNet12', 'WRN28'])
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)

    dataset = MiniImageNet('test')
    sampler = CategoriesSampler(dataset.label,
                                args.batch, args.way, args.shot + args.query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=8, pin_memory=False)
    model = ProtoNet(args).cuda()
    model.load_state_dict(torch.load(args.load))
    model.eval()

    ave_acc = Averager()
    label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)

    label = label.type(torch.LongTensor)

    if torch.cuda.is_available():
        label = label.cuda()

    for i, batch in enumerate(loader, 1):
        data, _ = [_.cuda() for _ in batch]

        logits = model(data)
        loss = F.cross_entropy(logits, label)

        acc = count_acc(logits, label)
        ave_acc.add(acc)
        print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

        x = None; p = None; logits = None
