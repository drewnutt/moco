#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
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
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from moco.default2018_single_model_pred import Net as default2018
from scipy.stats import pearsonr
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='TRAINTYPES',
                    help='path to types file')
parser.add_argument('--ligmolcache', metavar='LIGCACHE',
                    required=True, help='path to ligmolcache')
parser.add_argument('--recmolcache', metavar='RECCACHE',
                    required=True, help='path to recmolcache')
parser.add_argument('--dataroot', metavar='DATAROOT',
                    required=False, help='path to dataroot')
parser.add_argument('--val_ligcache', default=None, help='location of testing ligand cache file input, if different from the training cache')
parser.add_argument('--val_reccache', default=None, help='location of testing receptor cache file input, if different from the training cache')
parser.add_argument('--val_types', default=None, help='location of testing information, this must have a group indicator')
parser.add_argument('-a', '--arch', metavar='ARCH', default='default2018',
                    help='model architecture (default: default2018)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
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

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')

best_acc1 = 0


def main():
    args = parser.parse_args()
    
    if args.val_types is not None:
        if args.val_ligcache is None:
            args.val_ligcache = args.ligmolcache
        if args.val_reccache is None:
            args.val_reccache = args.recmolcache


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
        main_worker(args.local_rank, ngpus_per_node, args)
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    else:
        tgs = ['Supervised training']
        # wandb.init(entity='andmcnutt', project='DDG_model_Regression',config=args, tags=tgs)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            # args.rank = args.rank * ngpus_per_node + gpu
            print(f"rank:{args.local_rank}")
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.local_rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = default2018((28,48,48,48),num_classes=1024)


    init_whole_model = False
    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                del_end = None
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q'):
                    if k.startswith('module.encoder_q.fc'):
                        # dealing with the mlp option
                        check_num = k.split('.')[-2]
                        if check_num in ['0','2']:
                            if check_num == '2':
                                del state_dict[k]
                            else:
                                del_end = -2
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):del_end]] = state_dict[k]

                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))

            # freeze all layers but the last fc
            for name, param in model.named_parameters():
                if name not in ['ll.weight', 'll.bias']:
                    param.requires_grad = False
            # init the fc layer
            model.ll.weight.data.normal_(mean=0.0, std=0.01)
            model.ll.bias.data.zero_()
        else:
            print("=> no checkpoint found at '{}', training whole model".format(args.pretrained))
            init_whole_model = True
    else:
        init_whole_model = True
    if init_whole_model:
        model.apply(weights_init)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # # DataParallel will divide and allocate batch_size to all available GPUs
        # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        #     model.features = torch.nn.DataParallel(model.features)
        #     model.cuda()
        # else:
        #     model = torch.nn.DataParallel(model).cuda()
        raise NotImplementedError("Use a GPU for supervised training")

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, args.lr,
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
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
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

    #Setup grid_maker (probably should allow different AtomTypers)
    gmaker = molgrid.GridMaker()
    shape = gmaker.grid_dimensions(28)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        drop_last=True, collate_fn=moco.loader.collateMolDataset)

    if args.val_types is not None:
        val_dataset = molgrid.MolDataset(
            args.val_types, data_root=args.dataroot,
            ligmolcache=args.val_ligcache, recmolcache=args.val_reccache)

        val_loader = torch.utils.data.dataloader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            drop_last=True, collate_fn=moco.loader.collateMolDataset)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        if args.val_types is not None:
            pearsonr = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = pearsonr > best_r
        best_r = max(pearsonr, best_r)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_r,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
            if epoch == args.start_epoch:
                sanity_check(model.state_dict(), args.pretrained)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    r = AverageMeter('Pearson R', ':6.2f')
    rmse = AverageMeter('RMSE', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    targets = []
    predictions = []
    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    total_loss = 0
    end = time.time()
    for i, (lengths, center, coords, types, radii, labels) in enumerate(train_loader):
        types = types.cuda(args.gpu, non_blocking=True)
        radii = radii.squeeze().cuda(args.gpu, non_blocking=True)
        coords = coords.cuda(args.gpu, non_blocking=True)
        batch_size = coords.shape[0]
        if batch_size != types.shape[0] or batch_size != radii.shape[0]:
            raise RuntimeError("Inconsistent batch sizes in dataset outputs")
        output1 = torch.empty(batch_size,*tensorshape,dtype=coords.dtype,device=coords.device)
        for idx in range(batch_size):
            t = molgrid.Transform(molgrid.float3(*(center[idx].numpy().tolist())),random_translate=2,random_rotation=True)
            t.forward(coords[idx][:lengths[idx]],coords_q[idx][:lengths[idx]])
            gmaker.forward(t.get_rotation_center(), coords_q[idx][:lengths[idx]], types[idx][:lengths[idx]], radii[idx][:lengths[idx]], molgrid.tensor_as_grid(output1[idx]))

        del lengths, center, coords, types, radii
        torch.cuda.empty_cache()
        target = labels.cuda(args.gpu, non_blocking=True)

        # compute output
        prediction = model(output1)
        loss = criterion(prediction, target)

        # measure accuracy and record loss
        r_val, rmse_val = accuracy(prediction, target)
        losses.update(loss.item(), output1.size(0))
        r.update(r_val, output1.size(0))
        rmse.update(rmse_val, output1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions += prediction.detach().flatten().tolist()
        targets += target.detach().flatten().tolist()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    r_avg, rmse_avg = accuracy(predictions,targets)
    return r_avg, rmse_avg 

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    r = AverageMeter('Pearson R', ':6.2f')
    rmse = AverageMeter('RMSE', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, r, rmse],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    predictions = []
    targets = []
    with torch.no_grad():
        end = time.time()
        for i, (lengths, center, coords, types, radii, labels) in enumerate(train_loader):
            types = types.cuda(args.gpu, non_blocking=True)
            radii = radii.squeeze().cuda(args.gpu, non_blocking=True)
            coords = coords.cuda(args.gpu, non_blocking=True)
            batch_size = coords.shape[0]
            if batch_size != types.shape[0] or batch_size != radii.shape[0]:
                raise RuntimeError("Inconsistent batch sizes in dataset outputs")
            output1 = torch.empty(batch_size,*tensorshape,dtype=coords.dtype,device=coords.device)
            for idx in range(batch_size):
                t = molgrid.Transform(molgrid.float3(*(center[idx].numpy().tolist())),random_translate=2,random_rotation=True)
                t.forward(coords[idx][:lengths[idx]],coords_q[idx][:lengths[idx]])
                gmaker.forward(t.get_rotation_center(), coords_q[idx][:lengths[idx]], types[idx][:lengths[idx]], radii[idx][:lengths[idx]], molgrid.tensor_as_grid(output1[idx]))

            del lengths, center, coords, types, radii
            torch.cuda.empty_cache()
            target = labels.cuda(args.gpu, non_blocking=True)

            # compute output
            prediction = model(output1)
            loss = criterion(prediction, target)

            # measure accuracy and record loss
            r_val, rmse_val = accuracy(prediction, target)
            losses.update(loss.item(), output1.size(0))
            r.update(r_val, output1.size(0))
            rmse.update(rmse_val, output1.size(0))

            predictions += prediction.detach().flatten().tolist()
            targets += target.detach().flatten().tolist()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        r_avg, rmse_avg = accuracy(predictions,target)

    return r_avg, rmse_avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


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

    def average(self):
        fmtstr = '{name} ({avg' + self.fmt + '})'
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

    def disp_avg(self):
        entries = [self.prefix]
        entries += [average(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        try:
            r, _=pearsonr(np.array(output),np.array(target))
        except ValueError as e:
            print('{}:{}'.format(epoch,e))
            r=np.nan
        
        rmse = np.sqrt(((np.array(output_dist)-np.array(actual)) ** 2).mean())

        return r,rmse

def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)

if __name__ == '__main__':
    main()
