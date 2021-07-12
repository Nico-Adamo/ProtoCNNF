import torch
import torch.nn as nn
import torch.nn.functional as F
import models.layers_relax as layers
import logging
import os
import torch.optim as optim
import numpy as np
import math
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pdb
import shutil

class BasicBlock(nn.Module):
    """Basic ResNet block."""

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, res_param=0.1):
        super(BasicBlock, self).__init__()
        self.ln1 = layers.GroupNorm(8, in_planes)
        self.relu1 = layers.resReLU(res_param)
        self.conv1 = layers.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.ln2 = layers.GroupNorm(8, out_planes)
        self.relu2 = layers.resReLU(res_param)
        self.conv2 = layers.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.is_in_equal_out = (in_planes == out_planes)
        self.conv_shortcut = (not self.is_in_equal_out) and layers.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False) or None

        self.dropout = layers.Dropout(p = drop_rate)

    def forward(self, x, step='forward'):
        if ('forward' in step):
            if not self.is_in_equal_out:
                x = self.relu1(self.ln1(x))
            else:
                out = self.relu1(self.ln1(x))
            if self.is_in_equal_out:
                out = self.relu2(self.ln2(self.conv1(out)))
            else:
                out = self.relu2(self.ln2(self.conv1(x)))
            if self.drop_rate > 0:
                out = self.dropout(out, training=self.training)
            out = self.conv2(out)
            if not self.is_in_equal_out:
                return torch.add(self.conv_shortcut(x), out)
            else:
                return torch.add(x, out)

        elif ('backward' in step):
            out = self.ln2(self.conv2(x, step='backward'))
            if self.drop_rate > 0:
                out = self.dropout(out, training=self.training, step='backward')
            out = self.relu2(out, step='backward')
            out = self.ln1(self.conv1(out, step='backward'), step='backward')
            if not self.is_in_equal_out:
                out = torch.add(self.conv_shortcut(x, step='backward'), out)
            out = self.relu1(out, step='backward')
            if self.is_in_equal_out:
                out = torch.add(x, out)
            return out

class NetworkBlock(nn.Module):
    """Layer container for blocks."""
    def __init__(self,
               nb_layers,
               in_planes,
               out_planes,
               block,
               stride,
               drop_rate=0.0,
               ind=0,
               res_param=0.1):
        super(NetworkBlock, self).__init__()
        self.nb_layers = nb_layers
        self.res_param = res_param
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers,
                                      stride, drop_rate)
        # index of basic block to reconstruct to.
        self.ind = ind

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,
                  drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(
              block(i == 0 and in_planes or out_planes, out_planes,
                    i == 0 and stride or 1, drop_rate, self.res_param))
        return nn.ModuleList(layers)

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


class WideResNet(nn.Module):
    """ CNNF on Wide ResNet Architecture. """

    def __init__(self, depth = 28, num_classes = 64, widen_factor = 10, drop_rate=0.5, ind_block=0, ind_layer=0, cycles=2, res_param=0.1):
        super(WideResNet, self).__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlock
        self.ind_block = ind_block
        self.ind = ind_layer
        self.res_param = res_param
        self.cycles = cycles
        # 1st conv before any network block
        self.conv1 = layers.Conv2d(
            3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], block, 1,
                                   drop_rate, self.ind, res_param=self.res_param)
        # 2nd block
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], block, 2,
                                   drop_rate, res_param=self.res_param)
        # 3rd block
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], block, 2,
                                   drop_rate, res_param=self.res_param)
        # global average pooling and classifier
        self.layer = [self.conv1, self.block1, self.block2, self.block3]
        self.ln1 = layers.GroupNorm(8, n_channels[3])
        self.relu = layers.resReLU(res_param)
        self.flatten = layers.Flatten()
        self.n_channels = n_channels[3]

    def forward_cycle(self, x, step='forward', first=True, inter=False, inter_recon=False):
        if ('backward' in step):
            x = self.flatten(x, step='backward')
            x = F.interpolate(x, scale_factor=21)
            x = self.relu(self.ln1(x, step='backward'), step='backward')

        if (self.ind_block==0):
            if ('forward' in step):
                orig_feature = x
                for block in self.layer:
                    print(x.size())
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
            x = self.relu(self.ln1(x))
            x = F.avg_pool2d(x, 21)
            x = self.flatten(x, step='forward')

        if (inter==False):
            return x
        elif (inter==True):
            return x, orig_feature


    def reset(self):
        """
        Resets the pooling and activation states
        """
        self.relu.reset()

        for BasicBlock in self.block1.layer:
            BasicBlock.relu1.reset()
            BasicBlock.relu2.reset()
            BasicBlock.dropout.reset()

        for BasicBlock in self.block2.layer:
            BasicBlock.relu1.reset()
            BasicBlock.relu2.reset()
            BasicBlock.dropout.reset()

        for BasicBlock in self.block3.layer:
            BasicBlock.relu1.reset()
            BasicBlock.relu2.reset()
            BasicBlock.dropout.reset()

    def forward(self, x):
        self.reset()
        proto, orig_feature = self.forward_cycle(x, first=True, inter=True)
        ff_prev = orig_feature

        for i_cycle in range(self.cycles):
            # feedback
            recon = self.forward_cycle(proto, step='backward')
            # feedforward
            ff_current = ff_prev + self.res_param * (recon - ff_prev)
            proto = self.forward_cycle(ff_current, first=False)
            ff_prev = ff_current

        return proto

if __name__ == "__main__":
    model = WideResNet(ind_block = 2, cycles = 2, ind_layer = 2).cuda()
    rand_img_batch = torch.randn(3,3,84,84).cuda()
    proto = model(rand_img_batch)
    label = torch.arange(1).repeat(3)
    label = label.type(torch.cuda.LongTensor)

    loss = F.cross_entropy(proto, label)

    print(loss)
    print(proto.size())
