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
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=30)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./models/proto-1')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--model', type=str, choices=['Conv64', 'ResNet12'])
    parser.add_argument('--schedule', type=str, choices=['step'], default='step')
    parser.add_argument('--step-size', type=int, default=30)
    parser.add_argument('--drop-rate', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--project', type=str, default='CNNF-Prototype')
    parser.add_argument('--restore-from', type=str, default="")

    parser.add_argument('--ind-block', type=int, default=1)
    parser.add_argument('--cycles', type=int, default = 2)

    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    ensure_path(args.save_path)

    wandb.init(project=args.project, config=args)

    trainset = MiniImageNet('train')
    train_sampler = CategoriesSampler(trainset.label, 500,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=False)

    valset = MiniImageNet('val')
    val_sampler = CategoriesSampler(valset.label, 200,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=False)
    if args.restore_from != "":
        print("Restoring from {}".format(args.restore_from))
        checkpoint = torch.load(args.restore_from)
        classifier = Classifier(ResNet(ind_block = args.ind_block, cycles=args.cycles), args)
        classifier.load_state_dict(checkpoint)
        model = classifier.encoder.cuda()
    else:
        if args.model == "Conv64":
            model = Convnet().cuda()
        else:
            model = ResNet(ind_block = args.ind_block, cycles=args.cycles).cuda()

    optimizer = torch.optim.SGD(
          model.parameters(),
          args.lr,
          momentum=0.9,
          nesterov=True,
          weight_decay=5e-4)

    if args.schedule == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    wandb.watch(model, log_freq=10)

    for epoch in range(1, args.max_epoch + 1):
        print("Epoch " + str(epoch))
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()
        with tqdm(train_loader, total=500) as pbar:
            for i, batch in enumerate(pbar, 1):
                data, _ = [_.cuda() for _ in batch]
                p = args.shot * args.train_way
                data_shot, data_query = data[:p], data[p:]

                proto = model.forward_cycles(data_shot)
                proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

                label = torch.arange(args.train_way).repeat(args.query)
                label = label.type(torch.cuda.LongTensor)

                logits = euclidean_metric(model(data_query), proto)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                pbar.set_postfix(accuracy='{0:.4f}'.format(100*acc),loss='{0:.4f}'.format(loss.item()))

                tl.add(loss.item())
                ta.add(acc)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                proto = None; logits = None; loss = None

            tl = tl.item()
            ta = ta.item()

            model.eval()

            vl = Averager()
            va = Averager()

            for i, batch in enumerate(val_loader, 1):
                data, _ = [_.cuda() for _ in batch]
                p = args.shot * args.test_way
                data_shot, data_query = data[:p], data[p:]

                proto = model.forward_cycles(data_shot)
                proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

                label = torch.arange(args.test_way).repeat(args.query)
                label = label.type(torch.cuda.LongTensor)

                logits = euclidean_metric(model(data_query), proto)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)

                vl.add(loss.item())
                va.add(acc)

                proto = None; logits = None; loss = None

            vl = vl.item()
            va = va.item()
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

