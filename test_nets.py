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
    restore_from = "models/resnet-feedback-21-bias/epoch-20.pth"

    torch.manual_seed(0)
    args = Namespace(
        use_cosine_similarity = True,
        ind_block = 2,
        cycles = 1,
        ind_layer = 0,
        temperature = 1.0,
        query = 15,
        shot = 5,
        way = 5,
        episodes_per_epoch = 100,
        num_workers = 8,
        multi_gpu = False,
        num_eval_episodes = 1000,
        augment = False,
        model = "ResNet12",
        bias_shift = True,
        dataset = "MiniImageNet",
        memory_size = 1500,
        memory_start = 0
    )

    set_gpu("3")
    train_loader, val_loader, test_loader = get_dataloader(args)

    model = ProtoNet(args).cuda()

    if restore_from != "":
        print("Restoring from {}".format(restore_from))
        checkpoint = torch.load(restore_from)
        checkpoint = {k: v for k, v in checkpoint.items() if k in model.state_dict()}
        model.load_state_dict(checkpoint)
    label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
    label = label.type(torch.LongTensor).cuda()

    model.train()
    with torch.no_grad():
        for i, batch in enumerate(test_loader, 1):
            data, target = [_.cuda() for _ in batch]
            support_label = target[:args.shot * args.way]
            logits = model(data, memory_bank = True, debug_labels = support_label)

            if i == 150:
                break
