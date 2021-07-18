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
import matplotlib.pyplot as p
import pdb
import shutil
import sys

def res_conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return layers.Conv2d(in_planes, out_planes, 3, bias=False, stride=stride,
                     padding=1)

def mean(tensor):
    print("Mean: " + str(tensor.view(tensor.size(0),-1).mean()))

class BasicBlock(nn.Module):
    """Basic ResNet block."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1, res_param = 0.1, cycles = 0):
        super(BasicBlock, self).__init__()
        self.ln1 = layers.BatchNorm(planes, cycles)
        # self.relu1 = layers.resReLU(res_param)
        # self.ln1 = nn.BatchNorm2d(planes)
        self.relu1 = layers.leakyReLU(0.1)
        self.conv1 = res_conv3x3(inplanes, planes)
        self.ln2 = layers.BatchNorm(planes, cycles)
        # self.relu2 = layers.resReLU(res_param)
        # self.ln2 = nn.BatchNorm2d(planes)
        self.relu2 = layers.leakyReLU(0.1)

        self.conv2 = res_conv3x3(planes, planes)
        self.conv3 = res_conv3x3(planes, planes)
        # self.relu3 = layers.resReLU(res_param)
        self.ln3 = layers.BatchNorm(planes, cycles)
        self.ln4 = layers.BatchNorm(planes, cycles)
        # self.ln3 = nn.BatchNorm2d(planes)
        self.relu3 = layers.leakyReLU(0.1)

        self.maxpool = layers.MaxPool2d(stride)
        self.is_in_equal_out = (inplanes == planes)
        self.downsample = (stride != 1 or not self.is_in_equal_out) and layers.Conv2d(inplanes, planes, 1, bias=False, stride=1, padding=0) or None
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
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
                residual = self.ln4(residual)
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
                residual = self.ln4(residual, step='backward')
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

class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, keep_prob=1.0, avg_pool=True, drop_rate=0.1, dropblock_size=5, ind_block = 0, cycles = 0, res_param = 0.1):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = block(self.inplanes, 64, stride=2, drop_rate=drop_rate, res_param=0.1, cycles = cycles)
        self.layer2 = block(64, 160, stride=2, drop_rate=drop_rate, cycles = cycles)
        self.layer3 = block(160, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size, cycles = cycles)
        self.layer4 = block(320, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size, cycles = cycles)
        self.layer = [self.layer1, self.layer2, self.layer3, self.layer4]
        if avg_pool:
            self.avgpool = layers.AvgPool2d(5, scale_factor=5, stride=1)
        self.flatten = layers.Flatten()
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = layers.Dropout(p=1 - self.keep_prob)
        self.drop_rate = drop_rate
        self.ind_block = ind_block
        self.res_param = res_param
        self.cycles = cycles

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride, drop_rate, drop_block, block_size, self.res_param))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward_cycle(self, x, step='forward', first=True, inter=False):
        if ('backward' in step):
            x = self.flatten(x, step='backward')
            if self.keep_avg_pool:
                x = self.avgpool(x, step='backward')

        if (self.ind_block==0):
            blocks = [x]
            if ('forward' in step):
                orig_feature = x
                for idx in range(4):
                    # Rather than having a list of layers, we have to do this stupid hack for pytorch to recognize
                    # the layer as a module, see https://github.com/pytorch/pytorch/issues/8637
                    if (idx==0):
                        x = self.layer1(x, step='forward')
                    elif (idx==1):
                        x = self.layer2(x, step='forward')
                    elif (idx==2):
                        x = self.layer3(x, step='forward')
                    elif (idx==3):
                        x = self.layer4(x, step='forward')
                    blocks.append(x.view(x.size(0), -1))
            elif ('backward' in step):
                for idx in range(3, -1, -1):
                    if (idx == 0):
                        x = self.layer1(x, step='backward')
                    elif (idx == 1):
                        x = self.layer2(x, step='backward')
                    elif (idx == 2):
                        x = self.layer3(x, step='backward')
                    elif (idx == 3):
                        x = self.layer4(x, step='backward')

        # reconstruct to intermediate layers
        elif (self.ind_block>0):
            if ('forward' in step):
                blocks = []
                if (first==True):
                    if(inter==False):
                        for block in self.layer:
                            x = block(x)
                    elif(inter==True):
                        for idx in range(4):
                            if (idx==0):
                                x = self.layer1(x, step='forward')
                            elif (idx==1):
                                x = self.layer2(x, step='forward')
                            elif (idx==2):
                                x = self.layer3(x, step='forward')
                            elif (idx==3):
                                x = self.layer4(x, step='forward')
                            if ((idx+1)==self.ind_block):
                                orig_feature = x
                            if ((idx+1) >= self.ind_block):
                                blocks.append(x.view(x.size(0), -1))
                elif (first==False):
                    for idx in range(self.ind_block, 4):
                        if (idx == 0):
                            x = self.layer1(x, step='forward')
                        elif (idx == 1):
                            x = self.layer2(x, step='forward')
                        elif (idx == 2):
                            x = self.layer3(x, step='forward')
                        elif (idx == 3):
                            x = self.layer4(x, step='forward')
                        blocks.append(x.view(x.size(0), -1))
            elif ('backward' in step):
                for idx in range(3, self.ind_block - 1, -1):
                    if (idx == 0):
                        x = self.layer1(x, step='backward')
                    elif (idx == 1):
                        x = self.layer2(x, step='backward')
                    elif (idx == 2):
                        x = self.layer3(x, step='backward')
                    elif (idx == 3):
                        x = self.layer4(x, step='backward')

        if ('forward' in step):
            if self.keep_avg_pool:
                x = self.avgpool(x, step='forward')
            x = self.flatten(x, step='forward')
            blocks.append(x)


        if (inter==False):
            return x
        elif (inter==True and first==True):
            return x, orig_feature, blocks
        elif (inter==True and first==False):
            return x, blocks

    def reset(self):
        """
        Resets the pooling and activation states
        """
        self.layer1.relu1.reset()
        self.layer1.relu2.reset()
        self.layer1.relu3.reset()
        self.layer1.DropBlock.reset()
        self.layer1.dropout.reset()
        self.layer1.maxpool.reset()

        self.layer2.relu1.reset()
        self.layer2.relu2.reset()
        self.layer2.relu3.reset()
        self.layer2.DropBlock.reset()
        self.layer2.dropout.reset()
        self.layer2.maxpool.reset()

        self.layer3.relu1.reset()
        self.layer3.relu2.reset()
        self.layer3.relu3.reset()
        self.layer3.DropBlock.reset()
        self.layer3.dropout.reset()
        self.layer3.maxpool.reset()

        self.layer4.relu1.reset()
        self.layer4.relu2.reset()
        self.layer4.relu3.reset()
        self.layer4.DropBlock.reset()
        self.layer4.dropout.reset()
        self.layer4.maxpool.reset()

    def forward(self, x, inter_cycle = False, inter_layer = False):
        self.reset()
        proto, orig_feature, blocks = self.forward_cycle(x, first=True, inter=True)
        if inter_layer:
            cycle_proto = [blocks]
        else:
            cycle_proto = [proto]

        ff_prev = orig_feature

        self.layer1.num_batches_tracked += 1
        self.layer2.num_batches_tracked += 1
        self.layer3.num_batches_tracked += 1
        self.layer4.num_batches_tracked += 1

        for i_cycle in range(self.cycles):
            # feedback
            recon = self.forward_cycle(proto, step='backward')
            # feedforward
            ff_current = ff_prev + self.res_param * (recon - ff_prev)
            proto, blocks = self.forward_cycle(ff_current, first=False, inter=True)
            if inter_layer:
                cycle_proto.append(blocks)
            else:
                cycle_proto.append(proto)
            ff_prev = ff_current

        if inter_cycle:
            return cycle_proto
        else:
            return cycle_proto[-1]

if __name__ == "__main__":
    model = ResNet(ind_block = 2, cycles = 1).cuda()
    rand_img_batch = torch.randn(3,3,84,84).cuda()
    proto = model(rand_img_batch, inter_cycle=True, inter_layer=True)

    label = torch.arange(1).repeat(3)
    label = label.type(torch.cuda.LongTensor)


    print(proto[0][0].shape)

