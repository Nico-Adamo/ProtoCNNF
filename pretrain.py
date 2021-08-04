import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from models.convnet import Convnet
from models.resnet import ResNet
from models.wrn import WideResNet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric
from tqdm import tqdm
from models.classifier import Classifier

if __name__ == '__main__':
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--save-path', default='./models-backbone/net-1')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--model', type=str, choices=['Conv64', 'ResNet12', 'WRN28'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--project', type=str, default='CNNF-Prototype-Pretrain')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--schedule', type=int, nargs='+', default=[75, 150, 300], help='Decrease learning rate at these epochs.')
    parser.add_argument('--ind-block', type=int, default=1)
    parser.add_argument('--ind-layer', type=int, default=0)
    parser.add_argument('--cycles', type=int, default = 2)
    parser.add_argument('--ngpu', type=int, default = 1)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--inter-cycle-loss', action='store_true', default=False)
    parser.add_argument('--bias-shift', action='store_true', default=False)

    args = parser.parse_args()
    pprint(vars(args))
    if args.ngpu == 1:
        set_gpu(args.gpu)
    else:
        set_gpu(", ".join([str(i) for i in range(args.ngpu)]))

    ensure_path(args.save_path)
    if args.wandb:
        wandb.init(project=args.project, config=args)
    else:
        wandb.init(project=args.project, config=args, mode="disabled")

    trainset = MiniImageNet('train', args, augment=True)
    train_loader = DataLoader(dataset=trainset, batch_size = 16, shuffle=True,
                              num_workers=8, pin_memory=False)

    valset = MiniImageNet('val', args)
    val_sampler = CategoriesSampler(valset.label, 200,
                                    valset.num_class, 1 + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=False)

    if args.model == "Conv64":
        model = Classifier(Convnet(), args).cuda()
    elif args.model == "WRN28":
        model = Classifier(WideResNet(ind_block = args.ind_block, ind_layer = args.ind_layer, cycles = args.cycles), args).cuda()
    else:
        model = Classifier(ResNet(ind_block = args.ind_block, cycles = args.cycles), args).cuda()

    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if args.ngpu  > 1:
            model.encoder = torch.nn.DataParallel(model.encoder, device_ids=list(range(args.ngpu)))

        model = model.cuda()
        criterion = criterion.cuda()
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
    trlog['val_loss_dist'] = []
    trlog['val_loss_sim'] = []
    trlog['train_acc'] = []
    trlog['val_acc_dist'] = []
    trlog['max_acc_dist'] = 0.0
    trlog['val_acc_sim'] = []
    trlog['max_acc_sim'] = 0.0

    timer = Timer()

    wandb.watch(model, log_freq=10)

    # For validation
    args.way = valset.num_class
    args.shot = 1

    for epoch in range(1, args.max_epoch + 1):
        if epoch in args.schedule:
            initial_lr *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr
        print("Epoch " + str(epoch))

        model.train()

        tl = Averager()
        ta = Averager()
        with tqdm(train_loader, total=2400) as pbar:
            for i, batch in enumerate(pbar, 1):
                data, label = [_.cuda() for _ in batch]
                label = label.type(torch.cuda.LongTensor)

                if args.inter_cycle_loss:
                    logits_cycle = model(data, inter_cycle=True)
                    loss = 0
                    for cycle in range(args.cycles + 1):
                        loss += criterion(logits_cycle[cycle], label) / (args.cycles + 1)
                    logits = logits_cycle[-1]
                else:
                    logits = model(data)
                    loss = criterion(logits, label)

                acc = count_acc(logits, label)
                pbar.set_postfix(accuracy='{0:.4f}'.format(100*acc),loss='{0:.4f}'.format(loss.item()))

                tl.add(loss.item())
                ta.add(acc)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            tl = tl.item()
            ta = ta.item()


            if epoch > 100 or (epoch-1) % 5 == 0:
                model.eval()

                vl_dist = Averager()
                va_dist = Averager()
                vl_sim = Averager()
                va_sim = Averager()

                with torch.no_grad():
                    # Test few shot performance
                    for i, batch in enumerate(val_loader, 1):
                        data, _ = [_.cuda() for _ in batch]
                        data_shot, data_query = data[:valset.num_class], data[valset.num_class:]

                        logits_dist, logits_sim = model.forward_proto(data_shot, data_query, valset.num_class)

                        label = torch.arange(valset.num_class).repeat(args.query)
                        label = label.type(torch.cuda.LongTensor)

                        loss_dist = F.cross_entropy(logits_dist, label)
                        loss_sim = F.cross_entropy(logits_sim, label)

                        acc_dist = count_acc(logits_dist, label)
                        acc_sim = count_acc(logits_sim, label)

                        vl_dist.add(loss_dist.item())
                        va_dist.add(acc_dist)
                        vl_sim.add(loss_sim.item())
                        va_sim.add(acc_sim)

                    vl_dist = vl_dist.item()
                    vl_sim = vl_sim.item()
                    va_dist = va_dist.item()
                    va_sim = va_sim.item()

                print('Epoch {}, few-shot val, loss_dist={:.4f} acc_dist={:.4f}, acc_sim={:.4f}'.format(epoch, vl_dist, va_dist, va_sim))
                wandb.log({"test_loss_dist": vl_dist, "test_acc_dist": va_dist, "test_loss_sim": vl_sim, "test_acc_sim": va_sim}, step=epoch)
                if va_dist > trlog['max_acc_dist']:
                    trlog['max_acc_dist'] = va_dist
                    save_model('max-acc-dist')
                if va_sim > trlog['max_acc_sim']:
                    trlog['max_acc'] = va_sim
                    save_model('max-acc-sim')

                trlog['train_loss'].append(tl)
                trlog['train_acc'].append(ta)
                trlog['val_loss_dist'].append(vl_dist)
                trlog['val_acc_dist'].append(va_dist)
                trlog['val_loss_sim'].append(vl_dist)
                trlog['val_acc_sim'].append(va_dist)

                torch.save(trlog, osp.join(args.save_path, 'trlog'))

                save_model('epoch-last')

                if epoch % args.save_epoch == 0:
                    save_model('epoch-{}'.format(epoch))

            wandb.log({"train_loss": tl, "train_acc": ta}, step=epoch)
            print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

