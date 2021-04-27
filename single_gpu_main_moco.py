#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import molgrid

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models
# should look into getting some sort of ResNet working with this

import moco.loader
import moco.builder_single

import wandb

parser = argparse.ArgumentParser(description='PyTorch Molgrid Training')
parser.add_argument('data', metavar='TYPES',
                    help='path to types file')
parser.add_argument('--ligmolcache', metavar='LIGCACHE',
                    required=True, help='path to ligmolcache')
parser.add_argument('--recmolcache', metavar='RECCACHE',
                    required=True, help='path to recmolcache')
parser.add_argument('--dataroot', metavar='DATAROOT',
                    default="", help='path to dataroot')
parser.add_argument('-a', '--arch', metavar='ARCH', default='default2018',
                    help='model architecture (default: default2018)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# moco specific configs:
parser.add_argument('--moco-dim', default=1024, type=int,
                    help='feature dimension (default: 1024)')
parser.add_argument('--moco-k', default=32768, type=int,
                    help='queue size; number of negative keys (default: 32768)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp',default=True, action='store_false',
                    help='use mlp head')
# parser.add_argument('--aug-plus', action='store_true',
#                     help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

def main():
    args = parser.parse_args()
    tgs = ['MoCo_SingleGPU']
    wandb.init(entity='andmcnutt', project='DDG_model_Regression',config=args, tags=tgs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.gpu = device

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder_single.MoCo(
        args.arch,
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    print(model)

    torch.cuda.set_device(device)
    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    train_dataset = molgrid.torch_bindings.MolDataset(
        args.data,
        ligmolcache=args.ligmolcache, recmolcache=args.recmolcache, data_root=args.dataroot)
    gmaker = molgrid.GridMaker()
    shape = gmaker.grid_dimensions(28)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, sampler=train_sampler, drop_last=True,collate_fn=moco.loader.collateMolDataset)

    wandb.watch(model, log='all')
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, gmaker, shape, epoch, args)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


def train(train_loader, model, criterion, optimizer, gmaker, tensorshape, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    total_loss = 0
    for i, (lengths, center, coords, types, radii, _) in enumerate(train_loader):
        types = types.cuda(args.gpu, non_blocking=True)
        radii = radii.squeeze().cuda(args.gpu, non_blocking=True)
        coords = coords.cuda(args.gpu, non_blocking=True)
        coords_q = torch.empty(*coords.shape,device=coords.device,dtype=coords.dtype)
        batch_size = coords.shape[0]
        if i == 0:
            print(batch_size)
        if batch_size != types.shape[0] or batch_size != radii.shape[0]:
            raise RuntimeError("Inconsistent batch sizes in dataset outputs")
        output1 = torch.empty(batch_size,*tensorshape,dtype=coords.dtype,device=coords.device)
        output2 = torch.empty(batch_size,*tensorshape,dtype=coords.dtype,device=coords.device)
        for idx in range(batch_size):
            t = molgrid.Transform(molgrid.float3(*(center[idx].numpy().tolist())),random_translate=2,random_rotation=True)
            t.forward(coords[idx][:lengths[idx]],coords_q[idx][:lengths[idx]])
            gmaker.forward(t.get_rotation_center(), coords_q[idx][:lengths[idx]], types[idx][:lengths[idx]], radii[idx][:lengths[idx]], molgrid.tensor_as_grid(output1[idx]))
            t.forward(coords[idx][:lengths[idx]],coords[idx][:lengths[idx]])
            gmaker.forward(t.get_rotation_center(), coords[idx][:lengths[idx]], types[idx][:lengths[idx]], radii[idx][:lengths[idx]], molgrid.tensor_as_grid(output2[idx]))

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output, target = model(im_q=output1, im_k=output2)
        loss = criterion(output, target)
        total_loss += loss

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), output1.size(0))
        # top1.update(acc1[0], images[0].size(0))
        # top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    wandb.log({"Total Loss": total_loss/len(train_loader.dataset)})


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
