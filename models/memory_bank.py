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
        self.embedding_memory = {"train": torch.tensor([]).cuda(), "val": torch.tensor([]).cuda(), "eval": torch.tensor([]).cuda()}
        self.image_memory = {"train": torch.tensor([]).cuda(), "val": torch.tensor([]).cuda(), "eval": torch.tensor([]).cuda()}
        self.debug_memory = {"train": torch.tensor([]).cuda(), "val": torch.tensor([]).cuda(), "eval": torch.tensor([]).cuda()}

        self._debug_count = 0

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

    def add_debug_memory(self, data, mode = "train"):
        memory = self.debug_memory[mode]
        # Add memory to the end of the memory bank
        # data: [batch_size, emb_size]
        if memory.size(0) < self.size:
            self.debug_memory[mode] = torch.cat((memory, data))
        else:
            self.debug_memory[mode] = torch.cat((memory[data.shape[0]:], data))

    def get_embedding_memory(self, mode = "train"):
        return self.embedding_memory[mode]

    def get_debug_memory(self, mode = "train"):
        return self.debug_memory[mode]

    def get_image_memory(self, mode = "train"):
        return self.image_memory[mode]

    def get_length(self, mode = "train"):
        return self.embedding_memory[mode].size(0)

    def reset(self, mode = "train"):
        self.embedding_memory[mode] = torch.tensor([]).cuda()
        self.image_memory[mode] = torch.tensor([]).cuda()
        self.debug_memory[mode] = torch.tensor([]).cuda()

