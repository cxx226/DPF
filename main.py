#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import math
import os
from os.path import exists, join, split
import threading
from datetime import datetime

import time

import numpy as np
import shutil
import cv2

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp
import utils.data_transforms as transforms
from model.models import DPF



FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT, filename='./'+ datetime.now().strftime("%Y%m%d_%H%M%S") + '.txt')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TASK = 'SEGMENTATION'  
TRANSFER_FROM_TASK = 'SEGMENTATION' 


NYU40_PALETTE = np.asarray([
    [0, 0, 0], 
    [0, 0, 80], 
    [0, 0, 160], 
    [0, 0, 240], 
    [0, 80, 0], 
    [0, 80, 80], 
    [0, 80, 160], 
    [0, 80, 240], 
    [0, 160, 0], 
    [0, 160, 80], 
    [0, 160, 160], 
    [0, 160, 240], 
    [0, 240, 0], 
    [0, 240, 80], 
    [0, 240, 160], 
    [0, 240, 240], 
    [80, 0, 0], 
    [80, 0, 80], 
    [80, 0, 160], 
    [80, 0, 240], 
    [80, 80, 0], 
    [80, 80, 80], 
    [80, 80, 160], 
    [80, 80, 240], 
    [80, 160, 0], 
    [80, 160, 80], 
    [80, 160, 160], 
    [80, 160, 240], [80, 240, 0], [80, 240, 80], [80, 240, 160], [80, 240, 240], 
    [160, 0, 0], [160, 0, 80], [160, 0, 160], [160, 0, 240], [160, 80, 0], 
    [160, 80, 80], [160, 80, 160], [160, 80, 240]], dtype=np.uint8)


task_list = None
middle_task_list = None

if TASK =='SEGMENTATION':
    task_list = ['Segmentation']
    FILE_DESCRIPTION = ''
    PALETTE = NYU40_PALETTE
    EVAL_METHOD = 'mIoUAll'
else:
    task_list = None
    FILE_DESCRIPTION = ''
    PALETTE = None
    EVAL_METHOD = None

if TRANSFER_FROM_TASK =='SEGMENTATION':
    middle_task_list = ['Segmentation']
elif TRANSFER_FROM_TASK is None:
    pass



def downsampling(x, size=None, scale=None, mode='nearest'):
    if size is None:
        size = (int(scale * x.size(2)) , int(scale * x.size(3)))
    h = torch.arange(0,size[0]) / (size[0] - 1) * 2 - 1
    w = torch.arange(0,size[1]) / (size[1] - 1) * 2 - 1
    grid = torch.zeros(size[0] , size[1] , 2)
    grid[: , : , 0] = w.unsqueeze(0).repeat(size[0] , 1)
    grid[: , : , 1] = h.unsqueeze(0).repeat(size[1] , 1).transpose(0 , 1)
    grid = grid.unsqueeze(0).repeat(x.size(0),1,1,1)
    if x.is_cuda:
        grid = grid.cuda()
    return torch.nn.functional.grid_sample(x , grid , mode = mode)

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class SegMultiHeadList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, normalize, list_dir=None, guide_size=512):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_path = None
        self.label_path = None
        self.image_list = None
        self.label_list = None
        self.size = 0
        self.scale = 2
        self.normalize = normalize
        self.read_lists()
        self.guide_size = guide_size
    def __getitem__(self, index):
        data = [Image.open(join(self.image_path, self.image_list[index]))]
        data = np.array(data[0])
        height = data.shape[0]
        width = data.shape[1]
        if len(data.shape) == 2:
            data = np.stack([data , data , data] , axis = 2)
        data = [Image.fromarray(data)]
        
        label_data = list()
        label = np.load(join(self.label_path, self.label_list[index]))
        labels = cv2.resize(label, [width, height],interpolation=cv2.INTER_NEAREST)
        labels = Image.fromarray(labels)
        label_data.append(labels)
        data.append(label_data)
        data = list(self.transforms(*data))

        hr_guide = data[0].detach().cpu().numpy().transpose(2,1,0)*255
        hr_guide = Image.fromarray(hr_guide.astype(np.uint8))
        hr_guide = hr_guide.resize((self.guide_size, self.guide_size), Image.BICUBIC)
        hr_guide = torch.from_numpy(np.array(hr_guide)).permute(2,1,0).contiguous().float().div(255)
        data.append(self.normalize(hr_guide))            
        data[0] = self.normalize(data[0])
        return tuple(data)

    def __len__(self):
        return self.size

    def read_lists(self):
        if self.phase == 'val':
            self.image_path = join(self.list_dir, 'val')
            self.label_path = join(self.list_dir, 'valnpy')
        elif self.phase == 'test':
            self.image_path = join(self.list_dir, 'val')
            self.label_path = join(self.list_dir, 'valnpy')
        elif self.phase == 'train':
            self.image_path = join(self.list_dir, 'train')
            self.label_path = join(self.list_dir, 'Point_Annotation')
        self.image_list = sorted(os.listdir(self.image_path))
        self.label_list = sorted(os.listdir(self.label_path))
        self.size = len(self.image_list)


class SegListMSMultiHead(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, scales, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.scales = scales
        self.read_lists()
        
    def __getitem__(self, index):
        w, h = 640, 480
        data = [Image.open(join(self.image_path, self.image_list[index]))]
        data = np.array(data[0])
        height = data.shape[0]
        width = data.shape[1]
        if len(data.shape) == 2:
            data = np.stack([data , data , data] , axis = 2)
        data = [Image.fromarray(data)]
        
        label_data = list()
        label = np.load(join(self.label_path, self.label_list[index]))
        labels = cv2.resize(label, [width, height],interpolation=cv2.INTER_NEAREST)
        labels = Image.fromarray(labels)
        label_data.append(labels)
        data.append(label_data)
        out_data = list(self.transforms(*data))
        ms_images = list()
        lr_images = list()
        for s in self.scales:
            ww = round(int(w * s)/32) * 32
            hh = round(int(h * s)/32) * 32
            ms_image = data[0].resize((ww, hh), Image.BICUBIC)
            lr_image = ms_image.resize((ww//2, hh//2), Image.BICUBIC) 
            ms_image = self.transforms(ms_image)[0]
            lr_image = self.transforms(lr_image)[0]
            ms_images.append(ms_image)
            lr_images.append(lr_image)
            
        out_data.append(self.image_list[index])
        out_data.extend(ms_images)
        out_data.extend(lr_images)
        return tuple(out_data)

    def __len__(self):
        return self.size

    def read_lists(self):
        if self.phase == 'val':
            self.image_path = join(self.list_dir, 'val')
            self.label_path = join(self.list_dir, 'valnpy')
        elif self.phase == 'test':
            self.image_path = join(self.list_dir, 'val')
            self.label_path = join(self.list_dir, 'valnpy')
        elif self.phase == 'train':
            self.image_path = join(self.list_dir, 'train')
            self.label_path = join(self.list_dir, 'Point_Annotation')
        self.image_list = sorted(os.listdir(self.image_path))
        self.label_list = sorted(os.listdir(self.label_path))
        self.size = len(self.image_list)


def validate(val_loader, model, criterion, train_writer, eval_score=None, print_freq=1, transfer_model=None, epoch=None, num_classes=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    output_loss = AverageMeter()
    guide_loss = AverageMeter()
    end = time.time()
    hist_array_acc = list()
    hist_array = list()
    hist_array_acc.append(np.zeros((num_classes, num_classes)))
    hist_array.append(np.zeros((num_classes, num_classes)))
    iou_compute_cmd = 'per_class_iu(hist_array[idx])'
    iou_compute_cmd_acc = 'per_class_iu(hist_array_acc[idx])'

    num_scales = 1

    # switch to evaluate mode
    model.eval()
    if transfer_model is not None:
        transfer_model.eval()

    end = time.time()
    for itera, input_data in enumerate(val_loader):
        data_time.update(time.time() - end)
        label = input_data[1]

        h, w = input_data[0].size()[2:4]
        images = input_data[-2*num_scales:-num_scales]
        lr_images = input_data[-num_scales:]
        outputs = []
        guides = []

        with torch.no_grad():
            for idx in range(len(images)):
                image_var = Variable(images[idx], requires_grad=False)
                image_var = image_var.cuda()
                output, guide = model(image_var, image_var, continous=False)
                guide_array = list()
                output_array = list()
                guide_array.append(guide.data)
                output_array.append(output.data)
                guides.append(guide_array)
                outputs.append(output_array)

            final_guide = list()
            final_output = list()
            for label_idx in range(len(guides[0])):
                guide_tensor_list = list()
                output_tensor_list = list()
                for guide in guides:
                    guide_tensor_list.append(resize_4d_tensor(guide[label_idx], w, h))
                for output in outputs:
                    output_tensor_list.append(resize_4d_tensor(output[label_idx], w, h))
                
                final_guide.append(sum(guide_tensor_list))
                final_output.append(sum(output_tensor_list))
            pred = list()
            for label_entity in final_guide:
                pred.append(label_entity.argmax(axis=1))

            
            output = torch.from_numpy(final_output[0]).cuda()
            guide = torch.from_numpy(final_guide[0]).cuda()
            target = label[0].cuda()
            target_var = torch.autograd.Variable(target, requires_grad=False)

            # compute loss
            softmaxf = nn.LogSoftmax()
            output_ = softmaxf(output)
            guide_ = softmaxf(guide)

            output_loss.update(criterion(output_,target_var).item(), input_data[0].size(0))
            guide_loss.update(criterion(guide_,target_var).item(), input_data[0].size(0)) 
            loss = criterion(output_,target_var) + criterion(guide_,target_var)
            losses.update(loss.item(), input_data[0].size(0))

        
        batch_time.update(time.time() - end)

        map_score_array = list()
        for idx in range(len(label)):
            label[idx] = label[idx].numpy()
            hist_array[idx] = fast_hist(pred[idx].flatten(), label[idx].flatten(), num_classes)
            hist_array_acc[idx] += hist_array[idx]

            map_score_array.append(round(np.nanmean(eval(iou_compute_cmd)) * 100, 2))

            logger.info('===> task${}$ mAP {mAP:.3f}'.format(
                task_list[idx],
                mAP= map_score_array[idx]))
            
        if len(map_score_array) > 1:
            logger.info('===> task${}$ mAP {mAP:.3f}'.format(
                TASK,
                mAP= round(np.nanmean(map_score_array),2)))

        end = time.time()
        logger.info('Val: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Guide_Loss {guide_loss.val:.4f} ({guide_loss.avg:.4f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    .format(itera, len(val_loader), batch_time=batch_time, data_time=data_time,
                        guide_loss=guide_loss, loss=losses,))
    ious = list()
    for idx in range(len(hist_array_acc)):
        tmp_result = [i * 100.0 for i in eval(iou_compute_cmd_acc)]
        ious.append(tmp_result)
    for idx, i in enumerate(ious):
        logger.info('task %s', task_list[idx])
        logger.info(' '.join('{:.3f}'.format(ii) for ii in i))
    top = round(np.nanmean(ious), 2)
    logger.info(' * Score {top1:.3f}'.format(top1=top))
    
    train_writer.add_scalar('test_output_loss_average', output_loss.avg, global_step=epoch)
    train_writer.add_scalar('test_guide_loss_average', guide_loss.avg, global_step=epoch)
    train_writer.add_scalar('test_guide_score_average', top, global_step=epoch)
    
    return top



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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



def mIoUAll(output, target):
    """Computes the iou for the specified values of k"""
    num_classes = output.shape[1]
    hist = np.zeros((num_classes, num_classes))
    _, pred = output.max(1)
    pred = pred.cpu().data.numpy()
    target = target.cpu().data.numpy()
    hist += fast_hist(pred.flatten(), target.flatten(), num_classes)
    ious = per_class_iu(hist) * 100
    # cv2.imwrite('IMG.jpg', pred)
    return round(np.nanmean(ious), 2)
    

def train(train_loader, model, criterion, optimizer, epoch, local_rank, train_writer,
          print_freq=1, transfer_model=None, transfer_optim=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    output_loss = AverageMeter()
    guide_loss = AverageMeter()

    # switch to train mode
    model.train()

    if transfer_model is not None:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        transfer_model.train()

    end = time.time()

    for i, (input, target, guide_input) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if isinstance(input, list):
            input = input[0]
        if isinstance(target, list):
            target = target[0]
        if isinstance(guide_input, list):
            guide_input = guide_input[0]

        input = input.clone().cuda(local_rank, non_blocking=True)
        input = torch.autograd.Variable(input)
        guide_input = guide_input.clone().cuda(local_rank, non_blocking=True)
        guide_input = torch.autograd.Variable(input)
        target = target.clone().cuda(local_rank, non_blocking=True)
        target = torch.autograd.Variable(target)

        # compute output
        if transfer_model is None:
            output, guide = model(input, guide_input, continous=False)
        elif transfer_model is not None:
            _, features = model(input, guide_input)
            output = transfer_model(features)

        softmaxf = nn.LogSoftmax()
        output = softmaxf(output)
        guide = softmaxf(guide)

        output_loss.update(criterion(output,target).item(), input.size(0))
        guide_loss.update(criterion(guide,target).item(), input.size(0)) 
        loss = criterion(output,target) + criterion(guide,target)
        losses.update(loss.item(), input.size(0))
        train_loss = {'output': output_loss, 'guide': guide_loss}

        # compute gradient and do SGD step
        if transfer_optim is not None:
            transfer_optim.zero_grad()
        elif transfer_optim is None:
            optimizer.zero_grad()

        loss.backward()

        if transfer_optim is not None:
            transfer_optim.step()
        elif transfer_optim is None:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and local_rank == 0:
            losses_info = ''
            for loss_name, single_loss in train_loss.items():
                losses_info += 'Loss_{0} {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss_name, loss=single_loss)
                train_writer.add_scalar(loss_name + '_loss_val', single_loss.val, 
                    global_step= epoch * len(train_loader) + i)
                train_writer.add_scalar(loss_name + '_loss_average', single_loss.avg,
                    global_step= epoch * len(train_loader) + i)

            
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        '{loss_info}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,loss_info=losses_info))

            train_writer.add_scalar('train_output_loss_val', output_loss.val, global_step= epoch * len(train_loader) + i)
            train_writer.add_scalar('train_output_loss_average', output_loss.avg, global_step= epoch * len(train_loader) + i)
            train_writer.add_scalar('train_guide_loss_val', guide_loss.val, global_step= epoch * len(train_loader) + i)
            train_writer.add_scalar('train_guide_loss_average', guide_loss.avg, global_step= epoch * len(train_loader) + i)            

    if local_rank == 0:
        train_writer.add_scalar('train_epoch_loss_average', losses.avg, global_step= epoch)



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_bs8.pth.tar')


def train_seg(args):
    
    # add dist: init #
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    # add dist: dist batch #
    assert args.local_rank == dist.get_rank()
    batch_size = args.batch_size

    if args.local_rank == 0:
        if TRANSFER_FROM_TASK is not None:
            train_writer = SummaryWriter(comment='JIIF_NEW_right')
        elif TASK is not None:
            train_writer = SummaryWriter(comment=TASK)
        else:
            train_writer = SummaryWriter(comment='Nontype')
    if args.local_rank == 1:
        train_writer = SummaryWriter(comment='rank1(ade)_useless')
    if args.local_rank == 2:
        train_writer = SummaryWriter(comment='rank2(ade)_useless')
    if args.local_rank == 3:
        train_writer = SummaryWriter(comment='rank3(ade)_useless')
        
    num_workers = args.workers
    crop_size = args.crop_size
    if(args.train_data == 'PASCAL'):
        num_classes = 60
        train_data = 'PASCALContext'
    if(args.train_data == 'ade20k'):
        num_classes = 150
        train_data = 'ade20k'

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)
    
    single_model = DPF(num_classes)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(single_model).cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True, broadcast_buffers=True)


        
    criterion = nn.NLLLoss2d(ignore_index=255)
    criterion.cuda(args.local_rank)

    # Data loading code
    data_dir = join(args.data_dir, train_data)
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    t = []
    scales = [1]

    if args.random_rotate > 0:
        t.append(transforms.RandomRotateMultiHead(args.random_rotate))
    if args.random_scale > 0:
        t.append(transforms.RandomScaleMultiHead(args.random_scale))
    t.extend([transforms.RandomCropMultiHead(crop_size),
                transforms.RandomHorizontalFlipMultiHead(),
                transforms.ToTensorMultiHead()])

    # dist: add train-sampler #
    if dist.get_rank() == 0:
        logger.info(f"rank = {args.local_rank}, batch_size == {batch_size}")

    train_set = SegMultiHeadList(data_dir, 'train', transforms.Compose(t), normalize=normalize, guide_size=args.guide_size)    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

    train_loader = torch.utils.data.DataLoader(
        train_set,batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, drop_last=True, sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
            SegListMSMultiHead(data_dir, 'val', transforms.Compose([
            transforms.ToTensorMultiHead(),
            normalize
        ]), scales), 
        batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )


    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion, eval_score=eval(EVAL_METHOD), epoch=0)
        return

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        if dist.get_rank() == 0:
            logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.local_rank, train_writer)

        # evaluate on validation set
        if args.local_rank == 0:

            prec1 = validate(val_loader, model, criterion, train_writer, eval_score=eval(EVAL_METHOD), epoch=epoch, num_classes=num_classes)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            checkpoint_path = 'checkpoint_latest_bs8.pth.tar'
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=checkpoint_path)
            if (epoch + 1) % 10 == 0:
                history_path = 'checkpoint_bs8_{:03d}.pth.tar'.format(epoch + 1)
                shutil.copyfile(checkpoint_path, history_path)


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    #adjust the learning rate of sigma
    optimizer.param_groups[-1]['lr'] = lr * 0.01
    
    return lr


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def save_colorful_images(predictions, filenames, output_dir, palettes):
   """
   Saves a given (B x C x H x W) into an image file.
   If given a mini-batch tensor, will save the tensor as a grid of images.
   """
   for ind in range(len(filenames)):
       im = Image.fromarray(palettes[predictions[ind].squeeze()])
       fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
       out_dir = split(fn)[0]
       if not exists(out_dir):
           os.makedirs(out_dir)
       im.save(fn)

def resize_4d_tensor(tensor, width, height):
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_one(i, j):
        out[i, j] = np.array(
            Image.fromarray(tensor_cpu[i, j]).resize(
                (width, height), Image.BILINEAR))

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILINEAR))

    workers = [threading.Thread(target=resize_channel, args=(j,))
               for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return out


def test_ms(eval_data_loader, model, num_classes, scales,
            output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist_array_acc = list()
    hist_array = list()
    iou_compute_cmd = 'per_class_iu(hist_array[idx])'
    if num_classes == 2:
        iou_compute_cmd = '[' + iou_compute_cmd + '[1]]'

    iou_compute_cmd_acc = 'per_class_iu(hist_array_acc[idx])'
    if num_classes == 2:
        iou_compute_cmd_acc = '[' + iou_compute_cmd_acc + '[1]]'

    for i in range(len(task_list)):
        hist_array_acc.append(np.zeros((num_classes, num_classes)))
        hist_array.append(np.zeros((num_classes, num_classes)))

    num_scales = len(scales)
    for itera, input_data in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        
        if has_gt:
            name = input_data[2]
            label = input_data[1]
        else:
            name = input_data[1]

        logger.info('file name is %s', name)
        
        h, w = input_data[0].size()[2:4]
        images = input_data[-2*num_scales:-num_scales]

        outputs = []

        with torch.no_grad():
            for idx in range(len(images)):
                image_var = Variable(images[idx], requires_grad=False)
                image_var = image_var.cuda()
                output, guide = model(image_var, image_var, continous=False)
                final_array = list()
                final_array.append(guide.data)
                outputs.append(final_array)


            final = list()
            for label_idx in range(len(outputs[0])):
                tmp_tensor_list = list()
                for out in outputs:
                    tmp_tensor_list.append(resize_4d_tensor(out[label_idx], w, h))
                
                final.append(sum(tmp_tensor_list))
            pred = list()
            for label_entity in final:
                pred.append(label_entity.argmax(axis=1))

        batch_time.update(time.time() - end)
        if save_vis:
            for idx in range(len(label)):
                assert len(name) == 1
                file_name = (name[0][:-4] + task_list[idx] + '.png',)
                # save_output_images(pred[idx], file_name, output_dir)
                # save_colorful_images(pred[idx], file_name, output_dir + '_color',
                #                     PALETTE)
        if has_gt:
            map_score_array = list()
            for idx in range(len(label)):
                label[idx] = label[idx].numpy()
                hist_array[idx] = fast_hist(pred[idx].flatten(), label[idx].flatten(), num_classes)
                hist_array_acc[idx] += hist_array[idx]

                map_score_array.append(round(np.nanmean(eval(iou_compute_cmd)) * 100, 2))

                logger.info('===> task${}$ mAP {mAP:.3f}'.format(
                    task_list[idx],
                    mAP= map_score_array[idx]))
            
            if len(map_score_array) > 1:
                assert len(map_score_array) == len(label)
                logger.info('===> task${}$ mAP {mAP:.3f}'.format(
                    TASK,
                    mAP= round(np.nanmean(map_score_array),2)))

        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(itera, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    if has_gt: #val
        ious = list()
        for idx in range(len(hist_array_acc)):
            tmp_result = [i * 100.0 for i in eval(iou_compute_cmd_acc)]
            ious.append(tmp_result)
        for idx, i in enumerate(ious):
            logger.info('task %s', task_list[idx])
            logger.info(' '.join('{:.3f}'.format(ii) for ii in i))
        return round(np.nanmean(ious), 2)


def test_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase
    if(args.train_data == 'PASCAL'):
        num_classes = 60
        train_data = 'datasets'
    if(args.train_data == 'ade20k'):
        num_classes = 150
        train_data = 'ade20k'

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = DPF(num_classes)
   
    checkpoint = torch.load(args.resume)
    
    for name, param in checkpoint['state_dict'].items():
        # name = name[7:]
        single_model.state_dict()[name].copy_(param)
    
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = single_model.cuda()

    data_dir = join(args.data_dir, train_data)
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'], std=info['std'])
    scales = [1]#[0.9, 1, 1.25]

    dataset = SegListMSMultiHead(data_dir, phase, transforms.Compose([
        transforms.ToTensorMultiHead(),
        normalize,
    ]), scales)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            for name, param in checkpoint['state_dict'].items():
                # name = name[7:]
                model.state_dict()[name].copy_(param)
            # model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    out_dir = '{}_{:03d}_{}'.format(args.arch, start_epoch, phase)
    if len(args.test_suffix) > 0:
        out_dir += '_' + args.test_suffix
    if args.ms:
        out_dir += '_ms'

    if args.ms:
        mAP = test_ms(test_loader, model, num_classes, save_vis=True,
                      has_gt=phase != 'test' or args.with_gt,
                      output_dir=out_dir,
                      scales=scales)

    logger.info('%s mAP: %f', TASK, mAP)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('-d', '--data-dir', default='../dataset/nyud2')
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    parser.add_argument('-g', '--guide-size', default=512, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--trans-resume', default='', type=str, metavar='PATH',
                        help='path to latest trans checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('--pretrained-model', dest='pretrained_model',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--load-release', dest='load_rel', default=None)
    parser.add_argument('--phase', default='val')
    parser.add_argument('--train_data', type=str, default='ade20k')
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--random-rotate', default=0, type=int)
    parser.add_argument('--bn-sync', action='store_true')
    parser.add_argument('--ms', action='store_true',
                        help='Turn on multi-scale testing')
    parser.add_argument('--trans', action='store_true',
                        help='Turn on transfer learning')
    parser.add_argument('--with-gt', default=True, action='store_true')
    parser.add_argument('--test-suffix', default='', type=str)
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    assert args.data_dir is not None
    

    print(' '.join(sys.argv))
    print(args)

    return args


def main():
    args = parse_args()
    print(os.environ['MASTER_PORT'])
    if args.cmd == 'train':
        train_seg(args)
    elif args.cmd == 'test':
        test_seg(args)


if __name__ == '__main__':
    main()
