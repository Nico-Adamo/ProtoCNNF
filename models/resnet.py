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
import models.layers_relax as layers
import matplotlib.pyplot as plt
import pdb
import shutil
import sys

def res_conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return layers.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    """Basic ResNet block."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1, res_param = 0.1):
        super(BasicBlock, self).__init__()
        self.ln1 = layers.GroupNorm(8, inplanes)
        self.relu1 = layers.resReLU(res_param)
        self.conv1 = res_conv3x3(inplanes, planes)
        self.ln2 = layers.GroupNorm(8, planes)
        self.relu2 = layers.resReLU(res_param)
        self.conv2 = res_conv3x3(planes, planes)
        self.conv3 = res_conv3x3(planes, planes)
        self.relu3 = layers.resReLU(res_param)
        self.ln3 = layers.GroupNorm(8, planes)
        self.maxpool = layers.MaxPool2d(stride)
        self.is_in_equal_out = (inplanes == planes)
        self.downsample = (not self.is_in_equal_out) and layers.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False) or None
        self.drop_rate = drop_rate

    def forward(self, x, step='forward'):
        if ('forward' in step):
            residual = x
            out = self.relu1(self.ln1(x))
            out = self.relu2(self.ln2(self.conv1(out)))
            out = self.conv2(out)
            out = self.ln2(out)
            out = self.conv3(out)
            out = self.ln3(out)
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu3(out)
            out = self.maxpool(out)

        elif ('backward' in step):
            out = self.maxpool(out, step='backward')
            out = self.relu3(out, step='backward')
            if self.downsample is not None:
                residual = self.downsample(x, step='backward')
            out = self.ln3(self.conv3(x, step='backward'))
            out = self.ln2(self.conv2(x, step='backward'))
            out = self.relu2(out, step='backward')
            out = self.ln1(self.conv1(out, step='backward'), step='backward')
            out = self.relu1(out, step='backward')
            out = torch.add(x, out)
            return out

class NetworkBlock(nn.Module):
    """Layer container for blocks."""
    def __init__(self,
               in_planes,
               out_planes,
               block,
               stride,
               drop_rate=0.0,
               drop_block = False,
               block_size = 1,
               ind=0,
               res_param=0.1):
        super(NetworkBlock, self).__init__()
        self.nb_layers = nb_layers
        self.res_param = res_param
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers,
                                      stride, drop_rate, drop_block = drop_block, block_size = block_size)
        # index of basic block to reconstruct to.
        self.ind = ind

    def forward(self, x, step='forward', first=True, inter=False):
        # first: the first forward pass is the same as conventional CNN.
        # inter: if True, return intemediate features.

        # reconstruct to pixel level
        if (self.ind==0):
            if ('forward' in step):
                for block in self.layer:
                    x = block(x)
            elif ('backward' in step):
                for block in self.layer[::-1]:
                    x = block(x, step='backward')

        # reconstruct to intermediate layers
        elif (self.ind>0):
            if ('forward' in step):
                if (first==True):
                    if(inter==False):
                        for block in self.layer:
                            x = block(x)
                    elif(inter==True):
                        for idx, block in enumerate(self.layer):
                            x = block(x)
                            if ((idx+1)==self.ind):
                                orig_feature = x
                elif (first==False):
                    for idx, block in enumerate(self.layer):
                        if (idx+1 > self.ind):
                            x = block(x)
            elif ('backward' in step):
                ind_back = self.nb_layers-self.ind
                for idx, block in enumerate(self.layer[::-1]):
                    if (idx < ind_back):
                        x = block(x, step='backward')
        if (inter==False):
            return x
        elif (inter==True):
            return x, orig_feature


    def _make_layer(self, block, in_planes, planes, stride,
                  drop_rate, drop_block=False, block_size = 1):
        layers = []
        layers.append(
              block(in_planes, planes, stride, drop_rate, self.res_param))
        return nn.Sequential(*layers)

class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, keep_prob=1.0, avg_pool=True, drop_rate=0.1, dropblock_size=5, ind = 0, res_param = 0.1):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = NetworkBlock(block, self.inplanes, 64, stride=2, drop_rate=drop_rate, res_param=0.1)
        self.layer2 = NetworkBlock(block, 64, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = NetworkBlock(block, 160, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = NetworkBlock(block, 320, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = layers.AvgPool2d(5, scale_factor=1, stride=1)
        self.flatten = layers.Flatten()
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = layers.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.ind = ind
        self.res_param = res_param

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride, drop_rate, drop_block, block_size, self.res_param))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, step='forward', first=True):
        if ('forward' in step):
            if first == True:
                x, orig_feature = self.layer1(x, step='forward', first=True)
            else:
                x = self.layer1(x, step='forward')
            x = self.layer2(x, step='forward')
            x = self.layer3(x, step='forward')
            x = self.layer4(x, step='forward')
            if self.keep_avg_pool:
                x = self.avgpool(x, step='forward')
            x = self.flatten(x, step='forward')
            return x
        else:
            x = self.flatten(x, step='backward')
            if self.keep_avg_pool:
                x = self.avgpool(x, step='backward')
            x = self.layer4(x, step='backward')
            x = self.layer3(x, step='backward')
            x = self.layer2(x, step='backward')
            x = self.layer1(x, step='backward')
            return x

    def reset(self):
        """
        Resets the pooling and activation states
        """

        for BasicBlock in self.layer1.layer:
            BasicBlock.relu1.reset()
            BasicBlock.relu2.reset()
            BasicBlock.relu3.reset()

        for BasicBlock in self.layer2.layer:
            BasicBlock.relu1.reset()
            BasicBlock.relu2.reset()
            BasicBlock.relu3.reset()

        for BasicBlock in self.layer3.layer:
            BasicBlock.relu1.reset()
            BasicBlock.relu2.reset()
            BasicBlock.relu3.reset()

        for BasicBlock in self.layer3.layer:
            BasicBlock.relu1.reset()
            BasicBlock.relu2.reset()
            BasicBlock.relu3.reset()

if __name__ == "__main__":
    print(sys.path)
    model = ResNet()
    model.reset()
    rand_img_batch = torch.randn(1,3,84,84)
    out = model(rand_img_batch)
    print(out.size())
