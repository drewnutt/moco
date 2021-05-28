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
import wandb

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn import functional as F
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
from moco.default2018_single_model import Net as Default2018
from moco.dense import Dense
import moco.resnet
# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Molgrid Training')
parser.add_argument('data', metavar='TYPES',
                    help='path to types file')
parser.add_argument('--ligmolcache', default='', metavar='LIGCACHE',
                    required=False, help='path to ligmolcache')
parser.add_argument('--recmolcache', default='', metavar='RECCACHE',
                    required=False, help='path to recmolcache')
parser.add_argument('--dataroot', default='', metavar='DATAROOT',
                    required=False, help='path to dataroot')
parser.add_argument('-a', '--arch', metavar='ARCH', default='default2018',
        choices=['default2018','dense','resnet10','resnet18','resnet34'], help='model architecture (default: default2018)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='learning_rate')
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
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed',default=True, action='store_false',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--rep-size', default=1024, type=int,
                    help='feature dimension (default: %(default)d)')
parser.add_argument('--proj-size', default=4096, type=int,
        help='dimension to project the representation to when calculating loss (default: %(default)d)')
parser.add_argument('--cc-lambda', type=float,
        help='lambda value to use for the Cross Correlation loss (default: 1/(proj_size) )')
parser.add_argument('--tags', nargs='+', help='tags to use in the wandb run')

# option for adding labels
parser.add_argument('--semi-super',action='store_true',
                    help='Use semi-supervised training for available labels')
parser.add_argument('--sf', '--scaling-factor', default=10, type=float,
                    metavar='SF', help='scaling factor to increase importance of Supervised loss', dest='sf')
# Shouldn't be necessary when using LARS
# parser.add_argument('--clip',default=1.0,type=float,
#                     help='Value to use to clip gradients (to prevent exploding gradients)')

def main():
    args = parser.parse_args()

    if args.arch == 'dense':
        args.rep_size = 244
    elif args.arch.startswith('resnet'):
        args.rep_size = 512

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print(f"GPUs per node:{ngpus_per_node}\nWorld Size:{args.world_size}")
        # mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        main_worker(args.local_rank, ngpus_per_node, args)
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    else:
        tgs = ['BarlowTwins'] + args.tags
        wandb.init(entity='andmcnutt', project='DDG_model_Regression',config=args, tags=tgs)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.local_rank == -1:
            args.local_rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            # args.rank = args.rank * ngpus_per_node + gpu
            print(f"rank:{args.local_rank}")
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.local_rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnet'):
        resnet_num = int(args.arch.split('t')[-1])
        model = moco.resnet.generate_model(resnet_num)
        model.fc = nn.Identity()
    elif args.arch == 'default2018':
        model = Default2018((28,48,48,48), args.rep_size)
    elif args.arch == 'dense':
        model = Dense((28,48,48,48))
    projector = Projector(args.rep_size,args.proj_size)
    predictor = None
    if args.semi_super:
        predictor = Predictor(args.rep_size)
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            projector.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            if args.arch.startswith('resnet'):
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
            projector = nn.SyncBatchNorm.convert_sync_batchnorm(projector)
            projector = torch.nn.parallel.DistributedDataParallel(projector, device_ids=[args.gpu], output_device=args.gpu)
            if args.semi_super:
                predictor.cuda(args.gpu)
                predictor = torch.nn.parallel.DistributedDataParallel(predictor, device_ids=[args.gpu], output_device=args.gpu)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    if args.cc_lambda is None:
        args.cc_lambda = 1.0/args.proj_size
        print(f'updated cc_lambda: {args.cc_lambda}')
    criterion = CrossCorrLoss(args.proj_size,args.cc_lambda,args.batch_size,device=args.gpu).cuda(args.gpu)

    parameters = [p for p in model.parameters()] + [p for p in projector.parameters()]
    if args.semi_super:
        parameters += [p for p in predictor.parameters()]
    optimizer = LARS(parameters, lr=0,
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
    train_dataset = molgrid.MolDataset(
        args.data, data_root=args.dataroot,
        ligmolcache=args.ligmolcache, recmolcache=args.recmolcache)
    #Need to use random trans/rot when actually running
    gmaker = molgrid.GridMaker()
    shape = gmaker.grid_dimensions(28)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=moco.loader.collateMolDataset)

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        # train for one epoch
        loss, lr = train(train_loader, model, projector, predictor, criterion, optimizer, gmaker, shape, epoch, args)
        print(f'Epoch: {epoch}, Loss:{loss}')
        if args.local_rank == 0:
            if args.semi_super:
                wandb.log({'Loss':loss[0],'Supervised Loss':loss[1], 'Representation Loss':loss[2], "Learning Rate": lr})
            else: 
                wandb.log({'Loss':loss})

        if (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.local_rank % ngpus_per_node == 0)) and (epoch % 50 == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'projector': projector.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


def train(train_loader, model, projector, predictor, criterion, optimizer, gmaker, tensorshape, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    slosses = AverageMeter('SuperLoss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, slosses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    total_loss = 0
    super_loss = 0
    rep_loss = 0
    end = time.time()
    for i, (lengths, center, coords, types, radii, afflabel) in enumerate(train_loader):
        types = types.cuda(args.gpu, non_blocking=True)
        radii = radii.squeeze().cuda(args.gpu, non_blocking=True)
        coords = coords.cuda(args.gpu, non_blocking=True)
        coords_q = torch.empty(*coords.shape,device=coords.device,dtype=coords.dtype)
        batch_size = coords.shape[0]
        if batch_size != types.shape[0] or batch_size != radii.shape[0]:
            raise RuntimeError("Inconsistent batch sizes in dataset outputs")
        input1 = torch.empty(batch_size,*tensorshape,dtype=coords.dtype,device=coords.device)
        input2 = torch.empty(batch_size,*tensorshape,dtype=coords.dtype,device=coords.device)
        for idx in range(batch_size):
            t = molgrid.Transform(molgrid.float3(*(center[idx].numpy().tolist())),random_translate=2,random_rotation=True)
            t.forward(coords[idx][:lengths[idx]],coords_q[idx][:lengths[idx]])
            gmaker.forward(t.get_rotation_center(), coords_q[idx][:lengths[idx]], types[idx][:lengths[idx]], radii[idx][:lengths[idx]], molgrid.tensor_as_grid(input1[idx]))
            t.forward(coords[idx][:lengths[idx]],coords[idx][:lengths[idx]])
            gmaker.forward(t.get_rotation_center(), coords[idx][:lengths[idx]], types[idx][:lengths[idx]], radii[idx][:lengths[idx]], molgrid.tensor_as_grid(input2[idx]))

        del lengths, center, coords, types, radii
        torch.cuda.empty_cache()

        # measure data loading time
        data_time.update(time.time() - end)

        lr = adjust_learning_rate(args, optimizer, train_loader, i)

        # compute output
        rep1 = model(input1)
        proj1 = projector(rep1)
        rep2 = model(input2)
        proj2 = projector(rep2)
        loss = criterion(proj1, proj2)
        rep_loss += loss.item()
        if args.semi_super:
            pred = predictor(rep1)
            deltaG = afflabel.cuda(args.gpu,non_blocking=True)
            lossmask = deltaG.gt(0)
            sloss = args.sf*torch.sum(lossmask * nn.functional.l1_loss(pred, deltaG, reduction='none'))/lossmask.sum()
            super_loss += sloss.item()
            loss += sloss
            slosses.update(sloss.item(), lossmask.sum())
        total_loss += float(loss.item())

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input1.size(0))
        # top1.update(acc1[0], images[0].size(0))
        # top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # if args.semi_super:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    if args.semi_super:
        losses = (total_loss/len(train_loader),super_loss/len(train_loader),rep_loss/len(train_loader))
    else:
        losses = total_loss/len(train_loader)
    return losses, lr

class CrossCorrLoss(nn.Module):    
    def __init__(self, rep_size, lambd, batch_size, device='cpu'):    
        super(CrossCorrLoss,self).__init__()    
        self.bn = nn.BatchNorm1d(rep_size, affine=False).to(device)    
        self.device = device
        self.lambd = lambd    
        self.batch_size = batch_size
        
    def forward(self, z1, z2):    
        z1 = z1.to(self.device)
        z2 = z2.to(self.device)
        c = self.bn(z1).T @ self.bn(z2)    
        c.div_(self.batch_size)
        torch.distributed.all_reduce(c)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()    
            
        n, m = c.shape    
        assert n == m    
        off_diagonals = c.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()    
        off_diag = off_diagonals.pow_(2).sum()    
           
        loss = on_diag + self.lambd * off_diag    
        return loss


class Predictor(nn.Module):
    def __init__(self, rep_size):
        super(Predictor, self).__init__()

        self.pred_layer = nn.Linear(rep_size,1)

    def forward(self, x):
        x = self.pred_layer(F.relu(x)) # Need ReLU first to disconnect from last layer
        return x

class Projector(nn.Module):
    def __init__(self, rep_size, final_dim):
        super(Projector, self).__init__()

        self.first_layer = nn.Linear(rep_size,final_dim)
        self.batchnorm1 = nn.BatchNorm1d(final_dim)
        self.next_layer = nn.Linear(final_dim,final_dim)
        self.batchnorm2 = nn.BatchNorm1d(final_dim)
        self.last_layer = nn.Linear(final_dim,final_dim)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.first_layer(x)))
        x = F.relu(self.batchnorm2(self.next_layer(x)))
        return self.last_layer(x)

class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

def exclude_bias_and_norm(p):
    return p.ndim == 1

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


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.learning_rate * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

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
