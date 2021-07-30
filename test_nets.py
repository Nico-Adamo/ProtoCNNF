from mini_imagenet import MiniImageNet
from models.protonet import ProtoNet
import torch
from argparse import Namespace
import torch.nn.functional as F
from torch.utils.data import DataLoader
from samplers import CategoriesSampler
from utils import get_dataloader
from utils import pprint, set_gpu, count_acc
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
        query = 15,
        shot = 20,
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

    set_gpu("3")
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
    count = 0
    count_up = 0
    count_down = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            cycle_logits = model.encoder(data, inter_cycle=True)

            support_idx, query_idx = model.split_instances(data)

            logits, _ = model._forward(cycle_logits[0], support_idx, query_idx, memory_bank = None)
            loss = F.cross_entropy(logits, label)
            acc_0 = count_acc(logits, label)

            logits, _ = model._forward(cycle_logits[1], support_idx, query_idx, memory_bank = None)
            loss = F.cross_entropy(logits, label)
            acc_1 = count_acc(logits, label)

            if (acc_0 < 1 and acc_1 < acc_0 and count_down < 1) or (acc_0 < 1 and acc_1 > acc_0 and count_up < 3):
                if (acc_0 < 1 and acc_1 < acc_0 and count_down < 1):
                    count_down += 1
                if (acc_0 < 1 and acc_1 > acc_0 and count_up < 3):
                    count_up += 1
                count += 1
                print("Accuracy cycle 0: " + str(acc_0))

                print("Accuracy cycle 1: " + str(acc_1))

                cycle0_support = cycle_logits[0][0:100]
                cycle0_query = cycle_logits[0][100:]
                cycle1_support = cycle_logits[1][0:100]
                cycle1_query = cycle_logits[1][100:]

                U, S, V = torch.pca_lowrank(cycle0_support)
                cycle0_viz = torch.matmul(cycle0_support, V[:,:2])
                U, S, V = torch.pca_lowrank(cycle1_support)
                cycle1_viz = torch.matmul(cycle1_support, V[:,:2])
                U, S, V = torch.pca_lowrank(cycle0_query)
                cycle0_viz_q = torch.matmul(cycle0_query, V[:,:2])
                U, S, V = torch.pca_lowrank(cycle1_query)
                cycle1_viz_q = torch.matmul(cycle1_query, V[:,:2])

                if count==1:
                    f = open("cycle.txt", "w")
                else:
                    f = open("cycle.txt", "a")
                f.write(str(cycle0_viz.detach()))
                f.write("\n")
                f.write(str(cycle1_viz.detach()))
                f.write("\n--\n")
                f.close()


            if count == 4:
                break
