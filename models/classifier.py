import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import torch.optim as optim
import numpy as np
import math
import torchvision
from torchvision import datasets, transforms
from models.layers import DropBlock
import matplotlib.pyplot as plt
import pdb
import shutil
from utils import euclidean_metric

class Classifier(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        self.args = args

        self.num_classes = 64
        if(args.model == 'ResNet12' or args.model == 'WRN28'):
            self.hdim = 640
        self.encoder = encoder
        self.fc = nn.Linear(self.hdim, self.num_classes)

    def forward(self, x, inter_cycle = False, **kwargs):
        if inter_cycle:
            out = []
            cycle_proto = torch.stack(self.encoder(x, inter_cycle = True, **kwargs))
            for cycle in range(self.args.cycles + 1):
                out.append(self.fc(cycle_proto[cycle]))
            out = torch.stack(out)
        else:
            out = self.encoder(x, **kwargs)
            out = self.fc(out)
        return out

    def forward_proto(self, data_shot, data_query, way = None, **kwargs):
        proto = self.encoder(data_shot,  **kwargs)
        proto = proto.reshape(self.args.shot, way, -1).mean(dim=0)
        query = self.encoder(data_query,  **kwargs)

        logits_dist = euclidean_metric(query, proto)
        logits_sim = torch.mm(query, F.normalize(proto, p=2, dim=-1).t())
        return logits_dist, logits_sim
