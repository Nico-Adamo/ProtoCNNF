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

        # self.global_w = nn.Conv2d(in_channels=640, out_channels=64, kernel_size=1, stride=1)
        # nn.init.xavier_uniform_(self.global_w.weight)
        if args.adaptive_distance:
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

    def split_instances(self, data, mode="train"):
        query = self.args.query if mode == "train" else self.args.test_query
        args = self.args
        return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way),
                    torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + query))).long().view(1, query, args.way))

    def forward(self, x, memory_bank = False, get_feature = False, mode = "train", debug_labels = None):
        if get_feature:
            return self.encoder(x)
        else:
            # feature extraction
            n_query = self.args.query if mode == "train" else self.args.test_query
            x = x.squeeze(0)
            instance_embs = self.encoder(x) # (n_batch, way * (shot+query), n_dim)

            # Power transformation:
            instance_embs = F.normalize(torch.pow((instance_embs + 1e-6),0.5), p=2, dim=-1) + 1e-6

            memory_bank = True if self.memory_bank.get_length(mode=mode) > 100 and memory_bank else False

            support_idx, query_idx = self.split_instances(x, mode=mode)
            support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
            query = instance_embs[query_idx.flatten()].view(*(query_idx.shape   + (-1,)))

            self.memory_bank.add_embedding_memory(query.view(self.args.way * n_query, 640).detach(), mode = mode)
            if debug_labels is not None:
                self.memory_bank.add_debug_memory(debug_labels[self.args.way*self.args.shot:self.args.way * (self.args.shot + n_query)], mode = mode)

            logits = self._forward(support, query, memory_bank = memory_bank, mode = mode, debug_labels = debug_labels)

            # Update memory bank:
            self.memory_bank.add_embedding_memory(support.view(self.args.way * self.args.shot, 640).detach(), mode = mode)

            if debug_labels is not None:
                self.memory_bank.add_debug_memory(debug_labels[:self.args.way*self.args.shot], mode = mode)

            if self.training:
                #class_embs = self.global_w(instance_embs.unsqueeze(-1).unsqueeze(-1)).view(-1, 64)
                return logits#, class_embs
            else:
                return logits

    def _forward(self, support, query, memory_bank = False, mode = "train", debug_labels = None):
        emb_dim = support.size(-1)
        # organize support/query data

        batch_size, n_shot, n_way, n_dim = support.shape
        if memory_bank:
            proto = support.mean(dim=1)
            for i in range(self.args.test_transduction_steps if mode == "eval" else 1):
                proto = self.compute_memory_prototypes(support, mode = mode, prototype_compare = proto, debug_labels = debug_labels)
        else:
            proto = support.mean(dim=1)

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

    def get_similarity_scores(self, support, memory, prototype_compare = None, alpha = 1):
        """
        Compute the average cosine similarity matrix between support and memory
        Inputs:
            Support: [batch_size, n_shot, n_way, n_dim]
            Memory: [batch_size, n_shot + n_memory, n_way, n_dim]
        Output: [batch_size, n_shot + n_memory, n_way]
        """
        if prototype_compare is None:
            basic_proto = support.mean(dim=1) # [batch_size, n_way, n_dim]
        else:
            basic_proto = prototype_compare
        # [batch_size, n_shot + n_memory, n_way, n_dim] x [batch_size, n_way, n_dim] -> # [batch_size, n_shot + n_memory, n_way]
        sim = (F.cosine_similarity(memory, basic_proto, dim=-1) + 1) / 2 # Normalized similarity
        return torch.exp(sim / alpha)

    def compute_memory_prototypes(self, support, mode="train", prototype_compare = None, debug_labels = None):
        memory = self.memory_bank.get_embedding_memory(mode=mode)
        label_memory = self.memory_bank.get_debug_memory(mode=mode)

        n_memory, _ = memory.shape
        batch_size, n_shot, n_way, n_dim = support.shape
        memory_x = memory.view(batch_size, n_memory, 1, n_dim).expand(-1, -1, n_way, -1)
        shot_memory = torch.cat([support, memory_x], dim=1) # [batch_size, n_shot + n_memory, n_way, n_dim]
        sim = self.get_similarity_scores(support, shot_memory, prototype_compare = prototype_compare) # [batch_size, n_shot + n_memory, n_way]

        mask_weight = torch.cat([torch.tensor([1]).expand(batch_size, n_shot, n_way), torch.tensor([self.args.memory_weight]).expand(batch_size, n_memory, n_way)], dim=1).cuda()
        sim = sim * mask_weight

        # Per-instance temperature
        if self.args.adaptive_distance:
            memory_weights = self.instance_scale(memory)
            support_weights = self.instance_scale(support.view(batch_size * n_shot * n_way, n_dim)).view(batch_size, n_shot, n_way)
            memory_weights_x = memory_weights.view(batch_size, n_memory, 1).expand(-1, -1, n_way) # [batch_size, n_shot + n_memory, n_way]
            shot_memory_weights = torch.cat([support_weights, memory_weights_x], dim=1) # [batch_size, n_shot + n_memory, n_way]
            sim = sim / shot_memory_weights

        if self.training and self.args.use_training_labels:
            mask_class = torch.ones_like(sim).cuda()
            for way in range(n_way):
                for shot in range(sim.size(1)):
                    memory_ind = shot - self.args.shot
                    if label_memory[memory_ind] != debug_labels[way]:
                        mask_class[0][shot][way] = 0

            sim = sim * mask_class

        topk, ind = torch.topk(sim, self.args.augment_size, dim=1)
        topk_mask = torch.zeros_like(sim)
        for way in range(n_way):
            for shot in range(self.args.augment_size):
                topk_mask[0][ind[0][shot][way]][way] = sim[0][ind[0][shot][way]][way]

        sim = sim * topk_mask

        sim = sim.unsqueeze(-1)
        proto = (sim * shot_memory).sum(dim=1) / sim.sum(dim=1) # [batch_size, n_way, n_dim]
        return proto
