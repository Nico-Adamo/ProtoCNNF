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
        self.val_memory_bank = MemoryBank(args.memory_size)
        self.eval_memory_bank = MemoryBank(args.memory_size)


    def split_instances(self, data):
        args = self.args
        return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way),
                    torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))

    def forward(self, x, memory_bank = False, get_feature = False, validation = False, debug_labels = None):
        if get_feature:
            return self.encoder(x)
        else:
            if self.training:
                cur_memory_bank = self.memory_bank
            elif validation:
                cur_memory_bank = self.val_memory_bank
            else:
                cur_memory_bank = self.eval_memory_bank
            # feature extraction
            x = x.squeeze(0)
            cycle_instance_embs = self.encoder(x, inter_cycle=True) # [cycles + 1, 6 - ind_block, n_batch, n_emb]
                                                                                      # 6: [Pixel space, block 1, 2, 3, 4, pool/flatten][ind_block::]
            cycle_logits = []
            cur_memory_bank = cur_memory_bank if len(cur_memory_bank) > 100 and memory_bank else None
            # encode memory bank
            memory_encoded = self.encoder(cur_memory_bank.memory) if memory_bank else None
            for cycle in range(self.args.cycles + 1):
                instance_embs = cycle_instance_embs[cycle]

                # split support query set for few-shot data
                support_idx, query_idx = self.split_instances(x)
                debug_support = x[support_idx.flatten()].view(1, self.args.shot, self.args.way, 3, 84, 84)
                support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
                query = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))

                logits = self._forward(support, query, memory_bank = cur_memory_bank, memory_encoded = memory_encoded, debug_support = debug_labels) # add debug_support = debug_support to visualize memory bank
                cycle_logits.append(logits)

            # Update memory bank:
            cur_memory_bank._debug_add_memory(debug_labels)
            cur_memory_bank.add_memory(debug_support.view(self.args.shot*self.args.way,3,84,84))

            return logits

    def _forward(self, support, query, memory_bank = None, memory_encoded = None, debug_support = None):
        emb_dim = support.size(-1)
        # organize support/query data

        batch_size, n_shot, n_way, n_dim = support.shape
        if memory_bank != None:
            proto = memory_bank.compute_prototypes(support, memory_encoded, debug_support = debug_support)
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
