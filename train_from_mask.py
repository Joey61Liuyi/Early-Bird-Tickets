# -*- coding: utf-8 -*-
# @Time    : 2021/12/28 14:22
# @Author  : LIU YI

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import models
import copy
import os

def create_model(model, cfg, cfg_mask):
    arch = 'vgg'
    dataset = 'cifar100'
    newmodel = models.__dict__[arch](dataset=dataset, cfg =cfg)

    layer_id_in_cfg = 0
    start_mask = np.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            if np.sum(end_mask) == 0:
                continue
            idx1 = np.squeeze(np.argwhere(end_mask))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = copy.copy(end_mask)
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if np.sum(end_mask) == 0:
                continue
            idx0 = np.squeeze(np.argwhere(start_mask))
            idx1 = np.squeeze(np.argwhere(end_mask))
            # random set for test
            # new_end_mask = np.asarray(end_mask.cpu().numpy())
            # new_end_mask = np.append(new_end_mask[int(len(new_end_mask)/2):], new_end_mask[:int(len(new_end_mask)/2)])
            # idx1 = np.squeeze(np.argwhere(new_end_mask))

            # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(start_mask))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()
    return newmodel

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
    data = np.load('1610.43.npy', allow_pickle=True)
    data = data.item()
    cfg = data['cfg']
    cfg_mask = data['cfg_mask']
    # for i in range(len(cfg_mask)):
    #     cfg_mask[i] = np.asarray(cfg_mask[i].cpu().numpy())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    test_batch_size = 128
    arch = 'vgg'
    dataset = 'cifar100'
    model = models.__dict__[arch](dataset=dataset, depth = 16)
    model = create_model(model, cfg, cfg_mask)
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
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        #     'optimizer': optimizer.state_dict(),
        # }, is_best, filepath=args.save)

    print("Best accuracy: " + str(best_prec1))
    history_score[-1][0] = best_prec1




    print(data)
