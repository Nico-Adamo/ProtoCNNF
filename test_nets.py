from mini_imagenet import MiniImageNet
from models.protonet import ProtoNet
import torch
from argparse import Namespace
import torch.nn.functional as F
from torch.utils.data import DataLoader
from samplers import CategoriesSampler

# FOR DEBUG
if __name__ == '__main__':
    torch.manual_seed(0)
    args = Namespace(
        use_cosine_similarity = True,
        ind_block = 0,
        cycles = 1,
        ind_layer = 2,
        temperature = 1.0,
        query = 5,
        shot = 1,
        way = 2,
        model = "ResNet12",
        bias_shift = True
    )
    trainset = MiniImageNet('train')
    train_sampler = CategoriesSampler(trainset.label,
                                      100,
                                      args.way,
                                      args.shot + args.query)

    train_loader = DataLoader(dataset=trainset,
                                  num_workers=8,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)

    for i, batch in enumerate(train_loader, 1):
        img_batch = batch
        if i == 1:
            break
    data, _ = [_.cuda() for _ in img_batch]

    model = ProtoNet(args).cuda()
    cycle_logits = model(data, inter_cycle = True, inter_layer = True)
    loss = 0

    label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
    label = label.type(torch.LongTensor).cuda()
    print(label)

    for j in range(args.cycles + 1):
        for k in range(6 - args.ind_block):
            print(cycle_logits[j][k])
            loss_add = F.cross_entropy(cycle_logits[j][k], label) / ((args.cycles + 1) * (6 - args.ind_block))
            loss += loss_add
            print(loss_add)
            print(loss)
