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

class Classifier(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        self.args = args

        self.num_classes = 100
        if(args.model == 'ResNet12'):
            self.hdim = 640
        self.encoder = encoder
        self.fc = nn.Linear(self.hdim, self.num_classes)

    def forward(self, x, **kwargs):
        out = self.encoder(x, **kwargs)
        out = self.fc(out)
        return out

    def forward_proto(self, x, **kwargs):
        out = self.encoder(x, **kwargs)
        return out
