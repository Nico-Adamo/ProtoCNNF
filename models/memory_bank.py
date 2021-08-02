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
        self.memory = torch.tensor([]).cuda()
        self.augment_size = 8 # "Make everything n-shot"
        # Possible self.bias to add to cosine matrix?

        self._debug_memory = torch.tensor([]).cuda()
        self._debug_count = 0

    def add_memory(self, data):
        # Add memory to the end of the memory bank
        # data: [batch_size, emb_size]
        if self.memory.size(0) < self.size:
            self.memory = torch.cat((self.memory, data))
        else:
            self.memory = torch.cat((self.memory[data.shape[0]:], data))

    def get_memory(self):
        return self.memory

    def reset(self):
        self.memory = torch.tensor([])

    def __len__(self):
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

    def compute_prototypes(self, support, memory_encoded, debug_support = None):
        """
        Augment the support examples with the memory bank, and compute the prototypes
           self.memory: [n_memory, n_dim]
           support: [batch_size, n_shot, n_way, n_dim]
           return: [batch_size, n_way, n_dim]
        """
        memory = memory_encoded

        n_memory, _ = memory.shape
        batch_size, n_shot, n_way, n_dim = support.shape
        memory_x = memory.view(batch_size, n_memory, 1, n_dim).expand(-1, -1, n_way, -1)
        shot_memory = torch.cat([support, memory_x], dim=1) # [batch_size, n_shot + n_memory, n_way, n_dim]
        sim = self.get_similarity_scores(support, shot_memory) # [batch_size, n_way, n_shot + n_memory]

        # mask_weight = torch.cat([torch.tensor([1]).expand(batch_size, n_way, n_shot), torch.tensor([0.2]).expand(batch_size, n_way, n_memory)], dim=-1).cuda()
        # sim = sim * mask_weight

        # Take average along support examples, i.e. compute similarity between each memory/support example and each support example

        topk, ind = torch.topk(sim, self.augment_size, dim=-1) # 8 = num of support examples per class
        if debug_support is not None and random.randrange(10) == 4:
            memory_support = self._debug_memory.view(batch_size, n_memory, 1, 3, 84, 84).expand(-1, -1, n_way, -1, -1 ,-1)
            support_memory_imgs = torch.cat([debug_support, memory_support], dim=1) # [batch_size, n_shot + n_memory, n_way, 3,84,84]
            support_t = support_memory_imgs.permute(0,2,1,3,4,5) # [batch_size, n_way, n_shot + n_memory, 3,84,84]
            topk_support = torch.zeros(n_way, 16, 3, 84, 84)
            for i in range(n_way):
                topk_support[i] = support_t[0][i][ind[0][i]]
            rand_shot = topk_support.view(n_way * 16, *(topk_support.size()[2:])) # [n_way * 8, 3,84,84]
            grid = make_grid(rand_shot, nrow=16)
            save_image(grid, "memory_images_1500_" + str(self._debug_count)+".png")
            self._debug_count += 1
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
        if self._debug_memory.size(0) < self.size:
            self._debug_memory = torch.cat((self._debug_memory, data))
        else:
            self._debug_memory = torch.cat((self._debug_memory[data.shape[0]:], data))
