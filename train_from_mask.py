# -*- coding: utf-8 -*-
# @Time    : 2021/12/28 14:22
# @Author  : LIU YI
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from model_complexity import get_model_infos
import models
import copy
import wandb
import os
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(0.0001*torch.sign(m.weight.data))  # L1

def train(epoch, model, train_loader, optimizer):
    model.train()
    global history_score
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        # pred = output.data.max(1, keepdim=True)[1]
        # train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        train_acc += prec1.item()
        loss.backward()
        updateBN()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    history_score[epoch][0] = avg_loss / len(train_loader)
    history_score[epoch][1] = np.round(train_acc / len(train_loader), 2)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# def save_checkpoint(state, is_best, filepath):
#     torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
#     if is_best:
#         shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


def test(model, test_loader):
    model.eval()
    test_loss = 0
    test_acc = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        test_acc += prec1.item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, test_acc, len(test_loader), test_acc / len(test_loader)))
    return np.round(test_acc / len(test_loader), 2)

if __name__ == '__main__':
    # data = np.load('1633.66.npy', allow_pickle=True)
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--arch', default='vgg', type=str,
                        help='architecture to use')
    parser.add_argument('--depth', type=int, default=16,
                        help='depth of the vgg')

    args = parser.parse_args()

    mask_list = ['shuffle_channel_remove_n_f0.6100_p0.2032_1651.32.pth']
    wandb_project = 'pruning_score'
    for mask in mask_list:
        # wandb.init(project=wandb_project, name='train_greedy')
        seed = 1
        np.random.seed(seed)
        torch.manual_seed(seed)
        # mask_name = '25-1632.01.npy'
        # data = np.load(mask_name, allow_pickle=True)
        model = torch.load(mask)
        # data = checkpoint
        # # data = data.item()
        # cfg = data['cfg']
        # cfg_mask = data['cfg_mask']
        # state_dict = data['state_dict']
        # for i in range(len(cfg_mask)):
        #     cfg_mask[i] = np.asarray(cfg_mask[i].cpu().numpy())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 128
        test_batch_size = 128
        # arch = 'vgg'
        dataset = 'cifar100'
        # model = models.__dict__[arch](dataset=dataset, cfg=cfg)
        # model = (state_dict)
        # model = create_model(model, cfg, cfg_mask)
        # model.load(checkpoint)
        model.to('cuda')

        if dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('./data.cifar10', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.Pad(4),
                                     transforms.RandomCrop(32),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                 ])),
                batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])),
                batch_size=test_batch_size, shuffle=True)
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
                batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])),
                batch_size=test_batch_size, shuffle=True)

        schedule = [80, 120]
        epochs = 160
        history_score = np.zeros((epochs, 3))
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

        best_prec1 = 0
        for epoch in range(epochs):
            if epoch in schedule:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
            train(epoch, model, train_loader, optimizer)
            prec1 = test(model, test_loader)
            history_score[epoch][2] = prec1
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            # save_checkpoint({
            #     'epoch': epoch + 1,s
            #     'state_dict': model.state_dict(),
            #     'best_prec1': best_prec1,
            #     'optimizer': optimizer.state_dict(),
            # }, is_best, filepath=args.save)

        print("Best accuracy: " + str(best_prec1))
        history_score[-1][0] = best_prec1
        xshape = (1, 3, 32, 32)
        flops, param = get_model_infos(model, xshape)
        print(flops, param)
        print(mask)
        # wandb.finish()



    # print(data)
