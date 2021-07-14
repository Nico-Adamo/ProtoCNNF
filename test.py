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
    parser.add_argument('--temperature', type=int, default=1)
    parser.add_argument('--model', type=str, choices=['Conv64', 'ResNet12', 'WRN28'])
    parser.add_argument('--dataset', choices=['MiniImageNet'], default='MiniImageNet')
    parser.add_argument('--num-workers', type=int, default=8)

    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)

    test_loader = get_dataloader(args, test = True)
    model = ProtoNet(args).cuda()
    model.load_state_dict(torch.load(args.load))
    model.eval()

    ave_acc = Averager()
    label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)

    label = label.type(torch.LongTensor)

    if torch.cuda.is_available():
        label = label.cuda()

    for i, batch in enumerate(test_loader, 1):
        data, _ = [_.cuda() for _ in batch]

        logits = model(data)
        loss = F.cross_entropy(logits, label)

        acc = count_acc(logits, label)
        ave_acc.add(acc)
        print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

        x = None; p = None; logits = None
