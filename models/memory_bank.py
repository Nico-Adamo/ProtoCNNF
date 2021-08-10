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

        self.memory = {"train": torch.tensor([]).cuda(), "val": torch.tensor([]).cuda(), "eval": torch.tensor([]).cuda(), "debug": torch.tensor([]).cuda()}

        self.augment_size = 16 # "Make everything n-shot"
        # Possible self.bias to add to cosine matrix?

        self._debug_count = 0

        self.layer1_rn = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=4),
                nn.Conv2d(640,320,kernel_size=2,padding=0),
                nn.BatchNorm2d(320, momentum=1, affine=True),
                nn.ReLU(),
                nn.MaxPool2d(2))
        self.fc1_rn = nn.Sequential(
                nn.Linear(320 * 1 * 1, 160),
                nn.BatchNorm1d(160, momentum=1, affine=True),
                nn.ReLU())
        self.fc2_rn = nn.Linear(160, 1)
        nn.init.xavier_uniform_(self.fc2_rn.weight)
        self.alpha = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.alpha, 0)
        self.beta = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.beta, 0)

    def instance_scale(self, x):
        out = x.view(x.size(0), 640, 1, 1)
        out = self.layer1_rn(out)
        out = out.view(out.size(0), -1)
        out = self.fc1_rn(out)
        out = self.fc2_rn(out)
        out = torch.sigmoid(out)
        out = torch.exp(self.alpha) * out + torch.exp(self.beta)
        return out

    def add_memory(self, data, mode = "train"):
        memory = self.memory[mode]
        # Add memory to the end of the memory bank
        # data: [batch_size, emb_size]
        if memory.size(0) < self.size:
            self.memory[mode] = torch.cat((memory, data))
        else:
            self.memory[mode] = torch.cat((memory[data.shape[0]:], data))

    def get_memory(self, mode = "train"):
        return self.memory[mode]

    def reset(self, mode = "train"):
        self.memory[mode] = torch.tensor([])

    def get_length(self, mode = "train"):
        return self.memory[mode].size(0)

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

        # Per-instance temperature
        memory_weights = self.instance_scale(memory)
        support_weights = self.instance_scale(support.view(batch_size * n_shot * n_way, n_dim)).view(batch_size, n_shot, n_way)
        memory_weights_x = memory_weights.view(batch_size, n_memory, 1).expand(-1, -1, n_way) # [batch_size, n_shot + n_memory, n_way]
        shot_memory_weights = torch.cat([support_weights, memory_weights_x], dim=1).permute(0,2,1) # [batch_size, n_way, n_shot + n_memory]
        sim = sim / shot_memory_weights

        # mask_weight = torch.cat([torch.tensor([1]).expand(batch_size, n_way, n_shot), torch.tensor([0.5]).expand(batch_size, n_way, n_memory)], dim=-1).cuda()
        # sim = sim * mask_weight

        # Take average along support examples, i.e. compute similarity between each memory/support example and each support example
        topk, ind = torch.topk(sim, self.augment_size, dim=-1) # [batch_size, n_way, augment_size]
        res = Variable(torch.zeros(batch_size, n_way, n_shot + n_memory).cuda())
        sim = res.scatter(2, ind, topk) # Make all weights but top-k 0

        if debug_support is not None:
            memory_support = self._debug_memory.view(batch_size, n_memory, 1).expand(-1, -1, n_way)
            debug_support = debug_support.view(batch_size, n_shot, n_way)
            support_memory_imgs = torch.cat([debug_support, memory_support], dim=1) # [batch_size, n_shot + n_memory, n_way, 3,84,84]
            support_t = support_memory_imgs.permute(0,2,1) # [batch_size, n_way, n_shot + n_memory, 3,84,84]
            topk_support = torch.zeros(n_way, self.augment_size)
            for i in range(n_way):
                topk_support[i] = support_t[0][i][ind[0][i]]
            # print(topk_support)
            # rand_shot = topk_support.view(n_way * 16, *(topk_support.size()[2:])) # [n_way * 8, 3,84,84]
            # grid = make_grid(rand_shot, nrow=16)
            # save_image(grid, "memory_images_1500_" + str(self._debug_count)+".png")
            self._debug_count += 1

            # Make all weights not in the class 0
            for i in range(n_way):
                class_num = topk_support[i][0]
                for j in range(self.augment_size):
                    if topk_support[i][j] != class_num:
                        sim[0][i][ind[0][i][j]] = 0

        # mask_thresh = (sim > 0.75).float()
        # sim = sim * mask_thresh + 1e-8
        # mask_weight = torch.cat([torch.tensor([1]).expand(batch_size, n_way, n_shot), torch.tensor([0.01]).expand(batch_size, n_way, n_memory)], dim=-1).cuda()
        # sim = sim * mask_weight

        sim = sim.permute(0,2,1).unsqueeze(-1) # [batch_size, n_shot + n_memory, n_way, 1]
        proto = (sim * shot_memory).sum(dim=1) / sim.sum(dim=1) # [batch_size, n_way, n_dim]

        return proto
