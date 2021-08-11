import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from argparse import Namespace
from torch.autograd import Variable
from models.memory_bank import MemoryBank

class ProtoNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.model == 'ConvNet':
            from models.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.model == 'ResNet12':
            from models.resnet import ResNet
            self.encoder = ResNet(ind_block = args.ind_block, cycles=args.cycles)
        elif args.model == 'WRN28':
            from models.wrn import WideResNet
            self.encoder = WideResNet(ind_block = args.ind_block, cycles=args.cycles, ind_layer = args.ind_layer)
        else:
            raise ValueError('')

        self.memory_bank = MemoryBank(args.memory_size)
        self.augment_size = 16 # "Make everything n-shot"

        # self.global_w = nn.Conv2d(in_channels=640, out_channels=64, kernel_size=1, stride=1)
        # nn.init.xavier_uniform_(self.global_w.weight)

    def split_instances(self, data):
        args = self.args
        return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way),
                    torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))

    def forward(self, x, memory_bank = False, get_feature = False, mode = "train", debug_labels = None):
        if get_feature:
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs = self.encoder(x) # If inter cycle: [cycles + 1, 6 - ind_block, n_batch, n_emb]
                                                                                      # 6: [Pixel space, block 1, 2, 3, 4, pool/flatten][ind_block::]
            memory_bank = True if self.memory_bank.get_length(mode=mode) > 100 and memory_bank else False

            support_idx, query_idx = self.split_instances(x)
            debug_support = x[support_idx.flatten()].view(1, self.args.shot, self.args.way, 3, 84, 84)
            support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
            query = instance_embs[query_idx.flatten()].view(*(query_idx.shape   + (-1,)))

            logits = self._forward(support, query, memory_bank = memory_bank, mode = mode)

            # Update memory bank:
            self.memory_bank.add_embedding_memory(support.view(self.args.way * self.args.shot, 640).detach(), mode = mode)
            if mode == "train":
                self.memory_bank.add_image_memory(debug_support.view(self.args.way * self.args.shot,3,84,84), mode = mode)

            if self.training:
                #class_embs = self.global_w(instance_embs.unsqueeze(-1).unsqueeze(-1)).view(-1, 64)
                return logits#, class_embs
            else:
                return logits

    def _forward(self, support, query, memory_bank = False, mode = "train"):
        emb_dim = support.size(-1)
        # organize support/query data

        batch_size, n_shot, n_way, n_dim = support.shape
        proto = self.compute_prototypes(support, memory_bank = memory_bank, mode = mode)

        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = query.shape[1] * n_way

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        # Diminish cross-class bias by adding the difference in mean embedding between support and query to each query embedding
        if self.args.bias_shift:
            shift_embedding = (proto.mean(dim=1) - query.view(num_batch, num_query, -1).mean(dim=1) ).unsqueeze(1) # (num_batch, 1, num_emb)
            query = query + shift_embedding
        if not self.args.use_cosine_similarity: # Use euclidean distance:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
            proto = proto.contiguous().view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else: # cosine similarity: more memory efficient
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

            # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
            # (Nb, Nq*Np, d) * (Nb, d, Np) -> (Nb, Nq*Nw, Np)
            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)
            # (Nb, Np, d), (Nb, d, Np) -> (Nb, Np, Np)
            # logits_support = torch.bmm(proto, proto.permute([0,2,1])) / self.args.temperature
            # logits_support = logits.view(-1, num_proto)

        return logits

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

    def compute_prototypes(self, support, memory_bank = False, mode="train"):
        if memory_bank:
            memory = self.memory_bank.get_embedding_memory(mode=mode)
            if mode == "train":
                image_memory = self.memory_bank.get_image_memory(mode="train")

            n_memory, _ = memory.shape
            batch_size, n_shot, n_way, n_dim = support.shape
            memory_x = memory.view(batch_size, n_memory, 1, n_dim).expand(-1, -1, n_way, -1)
            shot_memory = torch.cat([support, memory_x], dim=1) # [batch_size, n_shot + n_memory, n_way, n_dim]
            sim = self.get_similarity_scores(support, shot_memory) # [batch_size, n_way, n_shot + n_memory]

            mask_weight = torch.cat([torch.tensor([1]).expand(batch_size, n_way, n_shot), torch.tensor([0.2]).expand(batch_size, n_way, n_memory)], dim=-1).cuda()
            sim = sim * mask_weight

            topk, ind = torch.topk(sim, self.augment_size, dim=-1) # [batch_size, n_way, augment_size]
            shot_memory_p = shot_memory.permute(0,2,1,3) # [batch_size, n_way, n_shot + n_memory, n_dim]
            shot_memory_topk = Variable(torch.zeros(batch_size, n_way, self.augment_size, n_dim).cuda())
            sim_topk = Variable(torch.zeros(batch_size, n_way, self.augment_size).cuda())
            for way in range(n_way):
                for shot in range(self.augment_size):
                    if ind[0][way][shot] < 5 or mode == "eval" or mode == "val": # Support embedding, no need to update
                        shot_memory_topk[0][way][shot] = shot_memory_p[0][way][ind[0][way][shot]]
                    else: # Updated embedding
                        memory_ind = ind[0][way][shot-5]
                        shot_memory_topk[0][way][shot] = self.encoder(image_memory[memory_ind].unsqueeze(0)).squeeze()

                    sim_topk[0][way][shot] = sim[0][way][ind[0][way][shot]]

            sim = sim_topk.permute(0,2,1).unsqueeze(-1) # [batch_size, augment_size, n_way, 1]
            shot_memory = shot_memory_topk.permute(0,2,1,3)  # [batch_size, augment_size, n_way, n_dim]
            proto = (sim * shot_memory).sum(dim=1) / sim.sum(dim=1) # [batch_size, n_way, n_dim]
            return proto
        else:
            return support.mean(dim=1)
