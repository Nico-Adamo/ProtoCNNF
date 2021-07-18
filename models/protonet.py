import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

    def split_instances(self, data):
        args = self.args
        return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way),
                    torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))

    def forward(self, x, get_feature=False, inter_cycle=False, inter_layer=False):
        if get_feature:
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0)
            cycle_instance_embs = self.encoder(x, inter_cycle=True, inter_layer=True) # [cycles + 1, 6 - ind_block, n_batch, n_emb]
                                                                                      # 6: [Pixel space, block 1, 2, 3, 4, pool/flatten][ind_block::]
            cycle_logits = []
            for cycle in range(self.args.cycles + 1):
                instance_embs = cycle_instance_embs[cycle][-1]
                num_inst = instance_embs.shape[0]
                # split support query set for few-shot data
                support_idx, query_idx = self.split_instances(x)

                if inter_layer:
                    cycle_logits = []
                    # Get prototypical logits for each layer
                    for i in range(6 - self.args.ind_block):
                        cycle_logits.append(self._forward(cycle_instance_embs[cycle][i], support_idx, query_idx))
                else:
                    logits = self._forward(instance_embs, support_idx, query_idx)
                cycle_logits.append(logits)

            if inter_cycle:
                return cycle_logits
            else:
                return logits

    def _forward(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))

        # get mean of the support
        proto = support.mean(dim=1) # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])

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
            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)

        return logits
