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
        self.memory = None
        self.augment_size = 8 # "Make everything n-shot"
        # Possible self.bias to add to cosine matrix?

        self._debug_memory = None

    def add_memory(self, emb):
        # Add memory to the end of the memory bank
        # emb: [batch_size, emb_size]
        if self.memory is None:
            self.memory = emb
        elif self.memory.size(0) < self.size:
            self.memory = torch.cat([self.memory, emb], dim=0)
        else:
            self.memory = torch.cat([self.memory[emb.shape[0]:], self.memory], dim=0)

    def get_memory(self):
        return self.memory

    def reset(self):
        self.memory = None

    def __len__(self):
        if self.memory is None:
            return 0
        return self.memory.size(0)

    def get_similarity_scores(self, support, memory):
        """
        Compute the average cosine similarity matrix between support and memory
        Inputs:
            Support: [batch_size, n_shot, n_way, n_dim]
            Memory: [n_way + n_memory, n_dim]
        Output: [batch_size, n_way, n_shot + n_memory]
        """
        memory_t = memory.permute(0,2,1,3)  # [batch_size, n_way, n_shot + n_memory, n_dim]
        support_t = support.permute(0,2,1,3) # [batch_size, n_way, n_shot, n_dim]
        memory_t = F.normalize(memory_t, dim=-1)
        support_t = F.normalize(support_t, dim=-1)
        # [batch_size, n_way, n_shot, n_dim] x [batch_size, n_way, n_dim, n_shot + n_memory] -> # [batch_size, n_way, n_shot, n_shot + n_memory]
        cos_matrix = torch.matmul(support_t, memory_t.permute(0,1,3,2))
        return cos_matrix.mean(dim=2) # [batch_size, n_way, n_shot + n_memory]

    def compute_prototypes(self, support, debug_support = None):
        """
        Augment the support examples with the memory bank, and compute the prototypes
           self.memory: [n_memory, n_dim]
           support: [batch_size, n_shot, n_way, n_dim]
           return: [batch_size, n_way, n_dim]
        """
        memory = self.memory.clone()
        n_memory, _ = memory.shape
        batch_size, n_shot, n_way, n_dim = support.shape
        memory_x = memory.view(batch_size, n_memory, 1, n_dim).expand(-1, -1, n_way, -1)
        shot_memory = torch.cat([support, memory_x], dim=1) # [batch_size, n_shot + n_memory, n_way, n_dim]
        sim = self.get_similarity_scores(support, shot_memory) # [batch_size, n_way, n_shot + n_memory]
        # Take average along support examples, i.e. compute similarity between each memory/support example and each support example

        topk, ind = torch.topk(sim, 8, dim=-1) # 8 = num of support examples per class
        print(ind.shape)
        if debug_support is not None:
            print(debug_support.shape)
            print(self._debug_memory.shape)
            memory_support = self._debug_memory.view(batch_size, n_memory, 1, 3, 84, 84).expand(-1, -1, n_way, -1, -1 ,-1)
            support_memory_imgs = torch.cat([debug_support, memory_support], dim=1) # [batch_size, n_shot + n_memory, n_way, 3,84,84]
            print(support_memory_imgs.shape)
            topk_support = support_memory_imgs.permute(0,2,1,3)[ind] # [batch_size, n_way, n_shot, 8, 3,84,84]
            print(topk_support.shape)
            sample = topk_support[0][random.randrange(5)]
            rand_shot = sample.view(n_shot * 8, *(sample.size()[2:])) # [n_shot * 8, 3,84,84]
            print(rand_shot.shape)
            grid = make_grid(rand_shot, nrow=8)
            save_image(grid, "memory_images_" + str(random.randrange(5))+".png")
        res = Variable(torch.zeros(batch_size, n_way, n_shot + n_memory).cuda())
        sim = res.scatter(2, ind, topk) # Make all weights but top-k 0

        # mask_thresh = (sim > 0.75).float()
        # sim = sim * mask_thresh + 1e-8
        # mask_weight = torch.cat([torch.tensor([1]).expand(batch_size, n_way, n_shot), torch.tensor([0.01]).expand(batch_size, n_way, n_memory)], dim=-1).cuda()
        # sim = sim * mask_weight

        sim = sim.permute(0,2,1).unsqueeze(-1) # [batch_size, n_shot + n_memory, n_way, 1]
        proto = (sim * shot_memory).sum(dim=1) / sim.sum(dim=1) # [batch_size, n_way, n_dim]

        return proto

    def _debug_add_memory(self, data):
        if self._debug_memory is None:
            self._debug_memory = data
        elif self._debug_memory.size(0) < self.size:
            self._debug_memory = torch.cat((self._debug_memory, data))
        else:
            self._debug_memory = torch.cat((self._debug_memory[data.shape[0]:], self._debug_memory))
