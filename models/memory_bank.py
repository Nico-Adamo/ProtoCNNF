import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
from torchvision.utils import make_grid, save_image


class MemoryBank(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        # Possible self.bias to add to cosine matrix?
        self.embedding_memory = {"train": torch.tensor([]).cuda(), "val": torch.tensor([]).cuda(), "eval": torch.tensor([]).cuda(), "debug": torch.tensor([]).cuda()}
        self.image_memory = {"train": torch.tensor([]).cuda()}

        self._debug_count = 0

        # self.layer1_rn = nn.Sequential(
        #         nn.UpsamplingNearest2d(scale_factor=4),
        #         nn.Conv2d(640,320,kernel_size=2,padding=0),
        #         nn.BatchNorm2d(320, momentum=1, affine=True),
        #         nn.ReLU(),
        #         nn.MaxPool2d(2))
        # self.fc1_rn = nn.Sequential(
        #         nn.Linear(320 * 1 * 1, 160),
        #         nn.BatchNorm1d(160, momentum=1, affine=True),
        #         nn.ReLU())
        # self.fc2_rn = nn.Linear(160, 1)
        # nn.init.xavier_uniform_(self.fc2_rn.weight)
        # self.alpha = nn.Parameter(torch.Tensor(1))
        # nn.init.constant_(self.alpha, 0)
        # self.beta = nn.Parameter(torch.Tensor(1))
        # nn.init.constant_(self.beta, 0)

    def instance_scale(self, x):
        out = x.view(x.size(0), 640, 1, 1)
        out = self.layer1_rn(out)
        out = out.view(out.size(0), -1)
        out = self.fc1_rn(out)
        out = self.fc2_rn(out)
        out = torch.sigmoid(out)
        out = torch.exp(self.alpha) * out + torch.exp(self.beta)
        return out

    def add_embedding_memory(self, data, mode = "train"):
        memory = self.embedding_memory[mode]
        # Add memory to the end of the memory bank
        # data: [batch_size, emb_size]
        if memory.size(0) < self.size:
            self.embedding_memory[mode] = torch.cat((memory, data))
        else:
            self.embedding_memory[mode] = torch.cat((memory[data.shape[0]:], data))

    def add_image_memory(self, data, mode = "train"):
        memory = self.image_memory[mode]
        # Add memory to the end of the memory bank
        # data: [batch_size, emb_size]
        if memory.size(0) < self.size:
            self.image_memory[mode] = torch.cat((memory, data))
        else:
            self.image_memory[mode] = torch.cat((memory[data.shape[0]:], data))

    def get_embedding_memory(self, mode = "train"):
        return self.embedding_memory[mode]

    def get_image_memory(self, mode = "train"):
        return self.image_memory[mode]

    def get_length(self, mode = "train"):
        return self.embedding_memory[mode].size(0)

