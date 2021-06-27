import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from models.convnet import Convnet
from models.resnet import ResNet_baseline
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric
from tqdm import tqdm
from models.classifier import Classifier

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=30)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./models-backbone/net-1')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--model', type=str, choices=['Conv64', 'ResNet12'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--project', type=str, default='CNNF-Prototype-Pretrain')
    parser.add_argument('--schedule', type=int, nargs='+', default=[15, 30, 60])
    parser.add_argument('--gamma', type=float, default=0.1)

    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    ensure_path(args.save_path)

    wandb.init(project=args.project, config=args)

    trainset = MiniImageNet('train', augment=True)
    train_loader = DataLoader(dataset=trainset, batch_size = 1, shuffle=True,
                              num_workers=8, pin_memory=False)

    valset = MiniImageNet('val')
    val_sampler = CategoriesSampler(valset.label, 200,
                                    valset.num_class, 1 + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=False)

    if args.model == "Conv64":
        model = Classifier(Convnet(), args).cuda()
    else:
        model = Classifier(ResNet_baseline(), args).cuda()
    initial_lr = args.lr
    optimizer = torch.optim.SGD(
          model.parameters(),
          args.lr,
          momentum=0.9,
          nesterov=True,
          weight_decay=5e-4)

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
        if epoch in args.schedule:
            initial_lr *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr

        model.train()

        tl = Averager()
        ta = Averager()
        with tqdm(train_loader, total=500) as pbar:
            for i, batch in enumerate(pbar, 1):
                data, label = [_.cuda() for _ in batch]
                label = label.type(torch.cuda.LongTensor)

                logits = model(data)

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


            if epoch > 20 or (epoch-1) % 5 == 0:
                model.eval()

                vl = Averager()
                va = Averager()
                with torch.no_grad():
                    # Test few shot performance
                    for i, batch in enumerate(val_loader, 1):
                        data, _ = [_.cuda() for _ in batch]
                        p = args.shot * args.test_way
                        data_shot, data_query = data[:p], data[p:]

                        proto = model(data_shot)
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
                print('Epoch {}, few-shot val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
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

