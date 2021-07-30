from mini_imagenet import MiniImageNet
from models.protonet import ProtoNet
import torch
from argparse import Namespace
import torch.nn.functional as F
from torch.utils.data import DataLoader
from samplers import CategoriesSampler
from utils import get_dataloader

# FOR DEBUG
if __name__ == '__main__':
    restore_from = "models/resnet-feedback-21-bias/max-acc.pth"

    torch.manual_seed(0)
    args = Namespace(
        use_cosine_similarity = True,
        ind_block = 2,
        cycles = 1,
        ind_layer = 0,
        temperature = 1.0,
        query = 1,
        shot = 5,
        way = 5,
        episodes_per_epoch = 100,
        num_workers = 8,
        multi_gpu = False,
        num_eval_episodes = 1000,
        augment = False,
        model = "ResNet12",
        bias_shift = True,
        dataset = "MiniImageNet"
    )

    train_loader, val_loader, test_loader = get_dataloader(args)
    global memory_bank_train
    memory_bank_train = None
    model = ProtoNet(args).cuda()

    if restore_from != "":
        print("Restoring from {}".format(restore_from))
        checkpoint = torch.load(restore_from)
        checkpoint = {k: v for k, v in checkpoint.items() if k in model.state_dict()}
        model.load_state_dict(checkpoint)
    label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
    label = label.type(torch.LongTensor).cuda()

    model.eval()
    for i, batch in enumerate(train_loader, 1):
        data, _ = [_.cuda() for _ in batch]
        cycle_logits = model.encoder(data, inter_cycle=True)
        cycle0_support = cycle_logits[0][0:25]
        cycle0_query = cycle_logits[0][25:]
        cycle1_support = cycle_logits[1][0:25]
        cycle1_query = cycle_logits[1][25:]

        print(cycle0_support.shape)
        print(cycle0_query.shape)
        U, S, V = torch.pca_lowrank(cycle0_support)
        cycle0_viz = torch.matmul(cycle0_support, V[:,:2])
        print(cycle0_viz)
        print(cycle0_viz.shape)

        if i == 1:
            break
