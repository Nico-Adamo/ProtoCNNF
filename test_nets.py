from mini_imagenet import MiniImageNet
from models.protonet import ProtoNet
import torch
# FOR DEBUG
if __name__ == '__main__':
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
    train_loader = MiniImageNet('train')
    for i, batch in enumerate(train_loader, 1):
        img_batch = batch
        if i == 1:
            break

    model = ProtoNet(args).cuda()
    cycle_logits = model(img_batch, inter_cycle = True, inter_layer = True)
    loss = 0

    label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
    label = label.type(torch.LongTensor).cuda()
    print(label)

    for j in range(args.cycles + 1):
        for k in range(6 - args.ind_block):
            print(cycle_logits[j][k])
            loss += F.cross_entropy(cycle_logits[j][k], label) / ((args.cycles + 1) * (6 - args.ind_block))
            print(loss)
