#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
from datetime import datetime
from tkinter import N
from tkinter.messagebox import NO
import time

import numpy as np
import shutil
import sys
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import utils.data_transforms_iiw as transforms
from model.models import DPF
from iiw_dataset.iiw_dataset import IIWDataset

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT, filename='./'+ datetime.now().strftime("%Y%m%d_%H%M%S") + '.txt')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


task_list = ['reflectance']

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


def compute_whdr_score(reflectance, point_pairs, pair_labels, name=None,draw = False,delta=0.10):
    assert len(point_pairs) == len(pair_labels)

    rows, cols = reflectance.shape[0:2]
    error_sum = 0.0
    weight_sum = 0.0
    for i, point_pair in enumerate(point_pairs):
        # "darker" is "J_i"
        darker = pair_labels[i][0]
        if darker not in ('1', '2', 'E'):
            continue

        if point_pair.max() > 1 or point_pair.min() < 0: # not in the image
            continue
            
        # "darker_score" is "w_i"
        weight = pair_labels[i][1]
        if weight <= 0 or weight is None:
            continue

        point1 = point_pair[0]
        point2 = point_pair[1]
    
        # convert to grayscale and threshold
        r1 = max(1e-10, reflectance[int(point1[1] * rows), int(point1[0] * cols)])
        r2 = max(1e-10, reflectance[int(point2[1] * rows), int(point2[0] * cols)])
        # convert algorithm value to the same units as human judgements
        if r2 / r1 > 1.0 + delta:
            alg_darker = '1'
        elif r1 / r2 > 1.0 + delta:
            alg_darker = '2'
        else:
            alg_darker = 'E'

        if darker != alg_darker:
            error_sum += weight

        weight_sum += weight

    return error_sum / (weight_sum+1e-10)
    

def compute_whdr_loss(reflectance, point_pairs, pair_labels, delta=0.12, epsilon=0.08):

    assert len(point_pairs) == len(pair_labels)

    rows, cols = reflectance.shape[0:2]

    whdr_loss = torch.tensor(0.0).cuda()
    
    point_num = 0.0
    
    for i, point_pair in enumerate(point_pairs):
        # "darker" is "J_i"
        
        darker = pair_labels[i][0]
        if darker not in ('1', '2', 'E'):
            continue

        if point_pair.max() > 1 or point_pair.min() < 0: # not in the image
            continue
            
        # "darker_score" is "w_i"
        weight = pair_labels[i][1]
        if weight <= 0 or weight is None:
            continue

        point1 = point_pair[0]
        point2 = point_pair[1]
        # convert to grayscale and threshold
        r1 = max(1e-10, reflectance[int(point1[1] * rows), int(point1[0] * cols)])
        r2 = max(1e-10, reflectance[int(point2[1] * rows), int(point2[0] * cols)])
        
        J = r1/r2
        
        whdr_loss_ = 0.0
        if darker == '1':
            whdr_loss_ =  max(0,J-1/(1+delta+epsilon))
        elif darker == '2':
            whdr_loss_ = max(0,1+delta+epsilon-J)
        else: #darker == 'E'
            l1 = 1/(1+delta-epsilon) - J
            l2 = J-(1+delta-epsilon)
            whdr_loss_ = max(0,l1,l2)
        whdr_loss = whdr_loss + weight * whdr_loss_
        point_num = point_num + 1
            
    return 10 * whdr_loss / point_num
    

def train(train_loader, model, optimizer, epoch, print_freq=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()


    # switch to train mode
    model.train()
    
    end = time.time()
    for i, (input, point_pair, pair_label, name, guide) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        guide = guide.cuda()
        input_guide = torch.autograd.Variable(guide)
        # compute output
        
        output, guide_output = model(input_var,input_guide,continous=True)
        loss_array = list()
        bs = output.shape[0]
        for j in range(bs):
            reflectance_map = output[j].permute(1,2,0)
            whdr_loss = compute_whdr_loss(reflectance_map,point_pair[j],pair_label[j])
            loss_array.append(whdr_loss)   
        
        if guide_output is not None:
            for j in range(bs):
                reflectance_map = guide_output[j].permute(1,2,0)
                whdr_loss = compute_whdr_loss(reflectance_map,point_pair[j],pair_label[j])
                loss_array.append(whdr_loss)   
                
        loss = sum(loss_array)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # scores_array = list()
        # for i in range(bs):
        #     reflectance_map = output[i].permute(1,2,0)
        #     whdr_score = compute_whdr_score(reflectance_map,point_pair[i],pair_label[i])
        #     scores_array.append(whdr_score)
        # scores.update(np.nanmean(scores_array), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if loss.requires_grad:
            loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        #end = time.time()

        if i % print_freq == 0:
            losses_info = ''
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        '{loss_info}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,loss_info=losses_info))


def validate(val_loader, model, print_freq=10, epoch=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_array = list()
    for it in task_list:
        losses_array.append(AverageMeter())
    score = AverageMeter()

    # switch to evaluate mode
    model.eval()

    attr_scores = list()
    for it in range(11):
        attr_scores.append(list())
    
    end = time.time()
    for i, (input, point_pair, pair_label, name,guide) in enumerate(val_loader):
        with torch.no_grad():
            input = input.cuda()
            input_var = torch.autograd.Variable(input, volatile=True)
            guide = guide.cuda()
            input_guide = torch.autograd.Variable(guide)    
            
            output, guide_output = model(input_var,input_guide, continous=True)
                                    
            loss_array = list()
            bs = output.shape[0]
            for j in range(bs):
                reflectance_map = output[j].permute(1,2,0)
                whdr_loss = compute_whdr_loss(reflectance_map,point_pair[j],pair_label[j])
                loss_array.append(whdr_loss)   
                
            if guide_output is not None:
                for j in range(bs):
                    reflectance_map = guide_output[j].permute(1,2,0)
                    whdr_loss = compute_whdr_loss(reflectance_map,point_pair[j],pair_label[j])
                    loss_array.append(whdr_loss)                   

            loss = sum(loss_array)

            # measure accuracy and record loss

            losses.update(loss.item(), input.size(0))

            for idx, it in enumerate(task_list):
                (losses_array[idx]).update((loss_array[idx]).item(), input.size(0))

            scores_array = list()
            for j in range(bs):
                if guide_output is not None:
                    reflectance_map = guide_output[j].permute(1,2,0)
                else:
                    reflectance_map = output[j].permute(1,2,0)
                    
                whdr_score = compute_whdr_score(reflectance_map,point_pair[j],pair_label[j],name=name[j],draw=False)
                scores_array.append(whdr_score)
            score.update(np.nanmean(scores_array), input.size(0))


 
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {score.val:.3f} ({score.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                score=score))

    logger.info(' * Score {top1.avg:.3f}'.format(top1=score))

    return score.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def my_collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    
    data = [item[0] for item in batch]
    data = torch.stack(data,0)
    
    point_pairs = [item[1] for item in batch]
    pair_labels = [item[2] for item in batch]
    out_names = [item[3] for item in batch]
    
    #guidance
    guide_data = [item[4] for item in batch]
    guide_data = torch.stack(guide_data,0)
    
    return [data,point_pairs,pair_labels,out_names,guide_data]


def train_iiw(args):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size
    guide_size = args.guide_size

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)
    
    single_model = DPF(args.classes,guide=True)
    model = single_model.cuda()
        
    # Data loading code
    data_dir = args.data_dir
    info = json.load(open(data_dir+'/info.json'))#, 'r')
    normalize = transforms.Normalize(mean=info['mean'],
                                     std=info['std'])
    naive_t = [transforms.Resize(crop_size),
            transforms.ToTensorMultiHead(),
            normalize]
            
    train_loader = torch.utils.data.DataLoader(
        IIWDataset(data_dir= data_dir, split = 'train',transforms = transforms.Compose(naive_t),guide_size=guide_size), #t
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,collate_fn = my_collate_fn
    )
    
    
    val_loader = torch.utils.data.DataLoader(
            IIWDataset(data_dir=data_dir,split= 'test',transforms= transforms.Compose(naive_t),guide_size=guide_size),
        batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=True,collate_fn = my_collate_fn
    )

    # define loss function (criterion) and pptimizer
    optimizer = torch.optim.SGD(single_model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    best_prec1 = 100 #lower, better
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
        validate(val_loader, model, epoch=0)
        return

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch

        train(train_loader, model, optimizer, epoch)
        prec1 = validate(val_loader, model, epoch=epoch)

        # evaluate on validation set
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        checkpoint_path = 'checkpoint_latest.pth.tar'
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        
        
        if (epoch + 1) % 10 == 0:
            history_path = 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
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


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data-dir', default='../dataset/nyud2')
    parser.add_argument('-c', '--classes', default=1, type=int)
    parser.add_argument('-s', '--crop-size', default=512, type=int)
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
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--random-rotate', default=0, type=int)
    parser.add_argument('--bn-sync', action='store_true')
    parser.add_argument('--ms', action='store_true',
                        help='Turn on multi-scale testing')
    parser.add_argument('--trans', action='store_true',
                        help='Turn on transfer learning')
    parser.add_argument('--with-gt', action='store_true')
    parser.add_argument('--test-suffix', default='', type=str)
    args = parser.parse_args()

    assert args.data_dir is not None
    assert args.classes > 0

    print(' '.join(sys.argv))
    print(args)


    return args

def main():
    args = parse_args()
    train_iiw(args)

if __name__ == '__main__':
    main()
