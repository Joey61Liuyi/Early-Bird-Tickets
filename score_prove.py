# -*- coding: utf-8 -*-
# @Time    : 2021/12/26 22:22
# @Author  : LIU YI

import argparse
import random

import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import wandb
# from models import *
import models
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=16,
                    help='depth of the vgg')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./baseline/vgg16-cifar100', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--save_1', default='./baseline/vgg16-cifar100', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='manual start epoch number')
parser.add_argument('--end_epoch', default=160, type=int, metavar='N', help='manual end epoch number')

# quantized parameters
parser.add_argument('--bits_A', default=8, type=int, help='input quantization bits')
parser.add_argument('--bits_W', default=8, type=int, help='weight quantization bits')
parser.add_argument('--bits_G', default=8, type=int, help='gradient quantization bits')
parser.add_argument('--bits_E', default=8, type=int, help='error quantization bits')
parser.add_argument('--bits_R', default=16, type=int, help='rand number quantization bits')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')

# multi-gpus
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
seed = 1
if not os.path.exists(args.save):
    os.makedirs(args.save)

gpu = args.gpu_ids
gpu_ids = args.gpu_ids.split(',')
args.gpu_ids = []
for gpu_id in gpu_ids:
   id = int(gpu_id)
   if id > 0:
       args.gpu_ids.append(id)
if len(args.gpu_ids) > 0:
   torch.cuda.set_device(args.gpu_ids[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_batch_jacobian(net, x, target, device):
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach()



if args.arch.endswith('lp'):
    # model = models.__dict__[args.arch](bits_A=args.bits_A, bits_E=args.bits_E, bits_W=args.bits_W, dataset=args.dataset, depth=args.depth)
    model = models.__dict__[args.arch](8, 8, 32, dataset=args.dataset, depth=args.depth)
elif args.dataset == 'imagenet':
    model = models.__dict__[args.arch](pretrained=False)
    if len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
else:
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True)

def check_score(model, cfg, cfg_mask):

    if args.arch.endswith('lp'):
        # model = models.__dict__[args.arch](bits_A=args.bits_A, bits_E=args.bits_E, bits_W=args.bits_W, dataset=args.dataset, depth=args.depth)
        newmodel = models.__dict__[args.arch](8, 8, 32, dataset=args.dataset, depth=args.depth)
    elif args.dataset == 'imagenet':
        newmodel = models.__dict__[args.arch](pretrained=False)
        if len(args.gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    else:
        newmodel = models.__dict__[args.arch](dataset=args.dataset, cfg = cfg)

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            if torch.sum(end_mask) == 0:
                continue
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if torch.sum(end_mask) == 0:
                continue
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # random set for test
            # new_end_mask = np.asarray(end_mask.cpu().numpy())
            # new_end_mask = np.append(new_end_mask[int(len(new_end_mask)/2):], new_end_mask[:int(len(new_end_mask)/2)])
            # idx1 = np.squeeze(np.argwhere(new_end_mask))

            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()
    newmodel.K = np.zeros((args.test_batch_size, args.test_batch_size))

    def counting_forward_hook(module, inp, out):
        try:
            if not module.visited_backwards:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            K2 = (1. - x) @ (1. - x.t())
            newmodel.K = newmodel.K + K.cpu().numpy() + K2.cpu().numpy()
        except:
            pass

    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True

    for name, module in newmodel.named_modules():
        if 'ReLU' in str(type(module)):
            # hooks[name] = module.register_forward_hook(counting_hook)
            module.register_forward_hook(counting_forward_hook)
            module.register_backward_hook(counting_backward_hook)

    newmodel = newmodel.to(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    s = []

    for j in range(1):
        data_iterator = iter(train_loader)
        x, target = next(data_iterator)
        x2 = torch.clone(x)
        x2 = x2.to(device)
        x, target = x.to(device), target.to(device)
        jacobs, labels, y = get_batch_jacobian(newmodel, x, target, device)
        newmodel(x2.to(device))
        s_, ld = np.linalg.slogdet(newmodel.K)
        s.append(ld)
    score = np.mean(s)
    return score

if args.cuda:
    model.cuda()

def pruning(model):
    total = 0
    cfg = []
    cfg_mask = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    thre_index = int(total * args.percent)
    thre = y[thre_index]
    # print('Pruning threshold: {}'.format(thre))
    mask = torch.zeros(total)
    index = 0
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.numel()
            weight_copy = m.weight.data.abs().clone()
            _mask = weight_copy.gt(thre.cuda()).float().cuda()
            cfg_mask.append(_mask.clone())
            if int(torch.sum(_mask)) > 0:
                cfg.append(int(torch.sum(_mask)))
            mask[index:(index+size)] = _mask.view(-1)
            # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, _mask.shape[0], int(torch.sum(_mask))))
            index += size
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    # print('Pre-processing Successful!')
    return mask, cfg, cfg_mask

resume = args.save + '/model_best.pth.tar'
print('==> resumeing from model_best ...')
checkpoint = torch.load(resume)
best_epoch = checkpoint['epoch']
print('best epoch: ', best_epoch)
model.load_state_dict(checkpoint['state_dict'])
best_mask, best_cfg, best_mask_cfg = pruning(model)
size = best_mask.size(0)

# resume = args.save_1 + '/model_best.pth.tar'
# resume = args.save_1 + '/ckpt159.pth.tar'
# print('==> resumeing from model_best ...')
# checkpoint = torch.load(resume)
# best_epoch = checkpoint['epoch']
# print('best epoch: ', best_epoch)
# model.load_state_dict(checkpoint['state_dict'])
# best_mask_1 = pruning(model)

# print('overlap rate of two best model: ', float(torch.sum(best_mask==best_mask_1)) / size)


epochs = args.end_epoch - args.start_epoch + 1
overlap = np.zeros((epochs, epochs))
save_dir = os.path.join(args.save, 'overlap_'+str(args.percent))
masks = []

for i in range(args.start_epoch, args.end_epoch+1):
    resume = args.save + '/ckpt' + str(i-1) + '.pth.tar'
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    masks.append(pruning(model))

# for i in range(args.start_epoch, args.end_epoch+1):
#     for j in range(args.start_epoch, args.end_epoch+1):
#         overlap[i-1, j-1] = float(torch.sum(masks[i-1] == masks[j-1])) / size
#         print('overlap[{}, {}] = {}'.format(i-1, j-1, overlap[i-1, j-1]))
#
# np.save(save_dir, overlap)
wandb_project = 'pruning_score'
name = 'trail'
wandb.init(project=wandb_project, name=name)
best_info = {}
best_score = 0

bird = [15, 25, 40, 159]

for i in range(args.start_epoch, args.end_epoch):
    score = check_score(model, masks[i][1], masks[i][2])
    info_dict = {
        'epoch': i,
        'score': score,
        'cfg': masks[i][1],
        'cfg_mask': masks[i][2]
    }
    wandb.log(info_dict)
    print(score)
    if score > best_score:
        best_score = score
        best_info = info_dict
    if i in bird:
        np.save('{}-{:.2f}.npy'.format(i, best_score), info_dict)