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
        self.ln1 = layers.GroupNorm(8, planes)
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
        self.downsample = (stride != 1 or not self.is_in_equal_out) and layers.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False) or None
        self.drop_rate = drop_rate
        self.dropout = layers.Dropout(p = drop_rate)
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x, step='forward'):
        if ('forward' in step):
            residual = x
            out = self.conv1(x)
            out = self.ln1(out)
            out = self.relu1(out)

            out = self.conv2(out)
            out = self.ln2(out)
            out = self.relu2(out)

            out = self.conv3(out)
            out = self.ln3(out)

            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu3(out)
            out = self.maxpool(out)

            if self.drop_rate > 0:
                if self.drop_block == True:
                    feat_size = out.size()[2]
                    keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                    gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                    out = self.DropBlock(out, gamma=gamma)
                else:
                    out = self.dropout(out, training=self.training)
            return out

        elif ('backward' in step):
            out = x
            residual = x

            if self.drop_rate > 0:
                if self.drop_block == True:
                    feat_size = out.size()[2]
                    keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                    gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                    out = self.DropBlock(out, gamma=gamma, step='backward')
                else:
                    out = self.dropout(out, training=self.training, step='backward')

            out = self.maxpool(out, step='backward')
            out = self.relu3(out, step='backward')
            if self.downsample is not None:
                residual = self.downsample(out, step='backward')
            out = self.ln3(out, step='backward')
            out = self.conv3(out, step='backward')
            out = self.relu2(out, step='backward')
            out = self.ln2(out, step='backward')
            out = self.conv2(out, step='backward')
            out = self.relu1(out, step='backward')
            out = self.ln1(out, step='backward')
            out = self.conv1(out, step='backward')

            out += residual

            return out

class NetworkBlock(nn.Module):
    """Layer container for blocks."""
    def __init__(self, block,
               in_planes,
               out_planes,
               stride,
               drop_rate=0.0,
               drop_block = False,
               block_size = 1,
               ind=0,
               res_param=0.1):
        super(NetworkBlock, self).__init__()
        self.res_param = res_param
        self.layer = self._make_layer(block, in_planes, out_planes,
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

    def __init__(self, block=BasicBlock, keep_prob=1.0, avg_pool=True, drop_rate=0.1, dropblock_size=5, ind_block = 0, ind_layer = 0, cycles = 0, res_param = 0.1):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = NetworkBlock(block, self.inplanes, 64, stride=2, drop_rate=drop_rate, res_param=0.1, ind = ind_layer)
        self.layer2 = NetworkBlock(block, 64, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = NetworkBlock(block, 160, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = NetworkBlock(block, 320, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer = [self.layer1, self.layer2, self.layer3, self.layer4]
        if avg_pool:
            self.avgpool = layers.AvgPool2d(5, scale_factor=5, stride=1)
        self.flatten = layers.Flatten()
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = layers.Dropout(p=1 - self.keep_prob)
        self.drop_rate = drop_rate
        self.ind_layer = ind_layer
        self.ind_block = ind_block
        self.res_param = res_param
        self.cycles = cycles

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride, drop_rate, drop_block, block_size, self.res_param))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, step='forward', first=True, inter=False):
        if ('backward' in step):
            x = self.flatten(x, step='backward')
            if self.keep_avg_pool:
                x = self.avgpool(x, step='backward')

        if (self.ind_block==0):
            if ('forward' in step):
                orig_feature = x
                for block in self.layer:
                    x = block(x)
            elif ('backward' in step):
                for block in self.layer[::-1]:
                    x = block(x, step='backward')

        # reconstruct to intermediate layers
        elif (self.ind_block>0):

            if ('forward' in step):
                if (first==True):
                    if(inter==False):
                        for block in self.layer:
                            x = block(x)
                    elif(inter==True):
                        for idx, block in enumerate(self.layer):
                            x = block(x)
                            if ((idx+1)==self.ind_block):
                                orig_feature = x
                elif (first==False):
                    for idx, block in enumerate(self.layer):
                        if (idx+1 > self.ind_block):
                            x = block(x)
            elif ('backward' in step):
                ind_back = len(self.layer)-self.ind_block
                for idx, block in enumerate(self.layer[::-1]):
                    if (idx < ind_back):
                        x = block(x, step='backward')

        if ('forward' in step):
            if self.keep_avg_pool:
                x = self.avgpool(x, step='forward')
            x = self.flatten(x, step='forward')

        if (inter==False):
            return x
        elif (inter==True):
            return x, orig_feature

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

    def forward_cycles(self, x):
        proto, orig_feature = self.forward(x, first=True, inter=True)
        ff_prev = orig_feature

        for i_cycle in range(self.cycles):
            # feedback
            recon = model(proto, step='backward')
            # feedforward
            ff_current = ff_prev + self.res_parameter * (recon - ff_prev)
            proto = model(ff_current, first=False)
            ff_prev = ff_current

        return proto

if __name__ == "__main__":
    print(sys.path)
    model = ResNet(ind_block = 2, cycles = 2)
    model.reset()
    rand_img_batch = torch.randn(3,3,84,84)
    proto = model.forward_cycles(rand_img_batch)

    print(proto.size())
