import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from models.classifier import Classifier
from models.convnet import Convnet
from models.resnet import ResNet
from models.wrn import WideResNet
from models.protonet import ProtoNet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, get_dataloader
from tqdm import tqdm
import torch.nn as nn

def prepare_label(args):

    # prepare one-hot label
    label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)

    label = label.type(torch.LongTensor)

    if torch.cuda.is_available():
        label = label.cuda()

    return label

def save_model(name):
    torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))

if __name__ == '__main__':
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--save-path', default='./models/proto-1')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--model', type=str, choices=['Conv64', 'ResNet12', 'WRN28'])
    parser.add_argument('--schedule', type=str, choices=['step'], default='step')
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--drop-rate', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--project', type=str, default='CNNF-Prototype')
    parser.add_argument('--restore-from', type=str, default="")

    parser.add_argument('--episodes-per-epoch', type=int, default=100)
    parser.add_argument('--num-eval-episodes', type=int, default=200)

    parser.add_argument('--multi-gpu', action='store_true', default=False)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--dataset', choices=['MiniImageNet'], default='MiniImageNet')
    parser.add_argument('--temperature', type=int, default=1)

    parser.add_argument('--inter-cycle-loss', action='store_true', default=False)
    parser.add_argument('--inter-layer-loss', action='store_true', default=False)

    parser.add_argument('--bias-shift', action='store_true', default=False)
    parser.add_argument('--ind-layer', type=int, default=0)
    parser.add_argument('--ind-block', type=int, default=1)
    parser.add_argument('--use-cosine-similarity', action='store_true', default=False)
    parser.add_argument('--memory-size', type=int, default=256)

    parser.add_argument('--cycles', type=int, default = 2)

    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    ensure_path(args.save_path)

    wandb.init(project=args.project, config=args)

    train_loader, val_loader, test_loader = get_dataloader(args)

    global memory_bank_train
    global memory_bank_val
    global memory_bank_test
    memory_bank_train = None
    memory_bank_val = None
    memory_bank_test = None
    model = ProtoNet(args).cuda()

    if args.restore_from != "":
        print("Restoring from {}".format(args.restore_from))
        checkpoint = torch.load(args.restore_from)
        checkpoint = {k: v for k, v in checkpoint.items() if k in model.state_dict()}
        model.load_state_dict(checkpoint)

    if args.multi_gpu:
        model.encoder = nn.DataParallel(model.encoder, dim=0)

    optimizer = torch.optim.SGD(
          model.parameters(),
          args.lr,
          momentum=0.9,
          nesterov=True,
          weight_decay=5e-4)

    if args.schedule == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    wandb.watch(model, log_freq=10)
    label = prepare_label(args)

    for epoch in range(1, args.max_epoch + 1):
        print("Epoch " + str(epoch))
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()
        with tqdm(train_loader, total=args.episodes_per_epoch) as pbar:
            for i, batch in enumerate(pbar, 1):
                data, _ = [_.cuda() for _ in batch]

                if args.inter_layer_loss:
                    cycle_logits, cycle_embs, recon_embs = model(data, inter_cycle = True, inter_layer = True)
                    loss = 0
                    for j in range(args.cycles):
                        for k in range(6 - args.ind_block):
                            loss += F.mse_loss(cycle_embs[j][k], recon_embs[j][k]) / ((args.cycles) * (6 - args.ind_block)) * 0.1
                        loss += F.cross_entropy(cycle_logits[j], label) / (args.cycles + 1)
                    loss += F.cross_entropy(cycle_logits[args.cycles], label) / (args.cycles + 1)
                    logits = cycle_logits[-1]
                elif args.inter_cycle_loss:
                    cycle_logits = model(data, inter_cycle=True)
                    loss = 0
                    for j in range(args.cycles + 1):
                        loss += F.cross_entropy(cycle_logits[j], label) / (args.cycles + 1)
                    logits = cycle_logits[-1]
                else:
                    logits, memory_addition = model(data, memory_bank_train)
                    if memory_bank_train == None:
                        memory_bank_train = memory_addition
                    elif memory_bank_train.shape[0] > args.memory_size:
                        memory_bank_train = torch.cat([memory_bank_train[memory_addition.shape[0]:], memory_addition], dim=0)
                    else:
                        memory_bank_train = torch.cat([memory_bank_train, memory_addition.shape[0]], dim=0)

                    loss = F.cross_entropy(logits, label)

                acc = count_acc(logits, label)
                pbar.set_postfix(accuracy='{0:.4f}'.format(100*acc),loss='{0:.4f}'.format(loss.item()))

                tl.add(loss.item())
                ta.add(acc)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                proto = None; logits = None; loss = None

            tl = tl.item()
            ta = ta.item()

            model.eval()

            vl = Averager()
            va = Averager()

            for i, batch in enumerate(val_loader, 1):
                data, _ = [_.cuda() for _ in batch]
                logits, memory_addition = model(data, memory_bank_val)
                if memory_bank_val == None:
                    memory_bank_val = memory_addition
                elif memory_bank_val.shape[0] > args.memory_size:
                    memory_bank_val = torch.cat([memory_bank_val[memory_addition.shape[0]:], memory_addition], dim=0)
                else:
                    memory_bank_val = torch.cat([memory_bank_val, memory_addition.shape[0]], dim=0)
                loss = F.cross_entropy(logits, label)

                acc = count_acc(logits, label)

                vl.add(loss.item())
                va.add(acc)

                proto = None; logits = None; loss = None

            vl = vl.item()
            va = va.item()

            if (epoch - 1) % 20 == 0:
                ave_acc = Averager()

                with torch.no_grad():
                    for i, batch in enumerate(test_loader, 1):
                        data, _ = [_.cuda() for _ in batch]

                        logits, memory_addition = model(data, memory_bank_test)
                        if memory_bank_test == None:
                            memory_bank_test = memory_addition
                        elif memory_bank_test.shape[0] > args.memory_size:
                            memory_bank_test = torch.cat([memory_bank_test[memory_addition.shape[0]:], memory_addition], dim=0)
                        else:
                            memory_bank_test = torch.cat([memory_bank_test, memory_addition.shape[0]], dim=0)
                        loss = F.cross_entropy(logits, label)

                        acc = count_acc(logits, label)
                        ave_acc.add(acc)
                wandb.log({"eval_acc": ave_acc.item()}, step=epoch - 1)

            print('Epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
            wandb.log({"train_loss": tl, "train_acc": ta, "test_loss": vl, "test_acc": va})

            if va > trlog['max_acc']:
                trlog['max_acc'] = va
                save_model('max-acc')

            trlog['train_loss'].append(tl)
            trlog['train_acc'].append(ta)
            trlog['val_loss'].append(vl)
            trlog['val_acc'].append(va)

            torch.save(trlog, osp.join(args.save_path, 'trlog'))

            save_model('epoch-last')

            if epoch % args.save_epoch == 0:
                save_model('epoch-{}'.format(epoch))

            print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

