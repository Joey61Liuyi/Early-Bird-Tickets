import argparse
import copy

import numpy as np
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from torch.autograd import Variable
from torchvision import datasets, transforms

# from models import *
import models
from model_complexity import get_model_infos
from score_based_pruning import create_cfg
from score_based_pruning import create_model
from score_based_pruning import count_channel


# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
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


if args.arch.endswith('lp'):
    # model = models.__dict__[args.arch](bits_A=args.bits_A, bits_E=args.bits_E, bits_W=args.bits_W, dataset=args.dataset, depth=args.depth)
    model = models.__dict__[args.arch](8, 8, 32, dataset=args.dataset, depth=args.depth)
elif args.dataset == 'imagenet':
    model = models.__dict__[args.arch](pretrained=False)
    if len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
else:
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.cuda:
    model.cuda()

import wandb
wandb.init(project="pruning_score", name="bn_score")

def pruning(model):
    total = 0
    xshape = (1, 3, 32, 32)
    flops_original, param_original = get_model_infos(model, xshape)
    total, form = count_channel(model)
    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    # tep = copy.deepcopy(y.numpy())
    # plt.hist(tep, bins=20)
    # plt.show()

    for thre_index in range(total):
        thre = y[thre_index]
        mask = torch.zeros(total)
        index = 0
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.numel()
                weight_copy = m.weight.data.abs().clone()
                _mask = weight_copy.gt(thre.cuda()).float().cuda()
                mask[index:(index + size)] = _mask.view(-1)
                # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, _mask.shape[0], int(torch.sum(_mask))))
                index += size

        cfg, cfg_mask = create_cfg(form, mask.numpy())
        model_new = create_model(model, cfg, cfg_mask)
        flops, param = get_model_infos(model_new, xshape)

        info_dict = {
            "channels": np.sum(mask.numpy()),
            "f_rate": flops/flops_original,
            "param": param/param_original,
            "cfg": cfg
        }
        wandb.log(info_dict)
        if flops/flops_original <= 0.8296:
            break
    torch.save(model_new, "bn_pruning.pth")

    # thre_index = int(total * args.percent)
    # thre = y[thre_index]
    # # print('Pruning threshold: {}'.format(thre))




    # print('Pre-processing Successful!')
    return mask

resume = args.save + '/model_best.pth.tar'
print('==> resumeing from model_best ...')
checkpoint = torch.load(resume)
best_epoch = checkpoint['epoch']
print('best epoch: ', best_epoch)
model.load_state_dict(checkpoint['state_dict'])
best_mask = pruning(model)
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
# masks = []

# for i in range(args.start_epoch, args.end_epoch+1):
resume = args.save + '/ckpt' + str(args.end_epoch) + '.pth.tar'
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint['state_dict'])
mask = pruning(model)

# for i in range(args.start_epoch, args.end_epoch+1):
#     for j in range(args.start_epoch, args.end_epoch+1):
#         overlap[i-1, j-1] = float(torch.sum(masks[i-1] == masks[j-1])) / size
#         print('overlap[{}, {}] = {}'.format(i-1, j-1, overlap[i-1, j-1]))

# np.save(save_dir, overlap)
