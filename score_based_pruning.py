# -*- coding: utf-8 -*-
# @Time    : 2021/12/27 16:34
# @Author  : LIU YI

import argparse
import copy
import random
import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import wandb
# from models import *
import models
import wandb
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from model_complexity import get_model_infos


def create_model(model, cfg, cfg_mask):

    while 0 in cfg:
        cfg.remove(0)
    if args.arch.endswith('lp'):
        # model = models.__dict__[args.arch](bits_A=args.bits_A, bits_E=args.bits_E, bits_W=args.bits_W, dataset=args.dataset, depth=args.depth)
        newmodel = models.__dict__[args.arch](8, 8, 32, dataset=args.dataset, depth=args.depth)
    elif args.dataset == 'imagenet':
        newmodel = models.__dict__[args.arch](pretrained=False)
        if len(args.gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    else:
        newmodel = models.__dict__[args.arch](dataset=args.dataset, cfg = cfg)

    # for [m0, m1] in zip(model.modules(), newmodel.modules()):
    #     if isinstance(m0, nn.BatchNorm2d):
    #         if np.sum(end_mask) == 0:
    #             continue
    #         idx1 = np.squeeze(np.argwhere(end_mask))
    #         if idx1.size == 1:
    #             idx1 = np.resize(idx1, (1,))
    #         m1.weight.data = m0.weight.data[idx1.tolist()].clone()
    #         m1.bias.data = m0.bias.data[idx1.tolist()].clone()
    #         m1.running_mean = m0.running_mean[idx1.tolist()].clone()
    #         m1.running_var = m0.running_var[idx1.tolist()].clone()
    #         layer_id_in_cfg += 1
    #         start_mask = copy.copy(end_mask)
    #         if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
    #             end_mask = cfg_mask[layer_id_in_cfg]
    #     elif isinstance(m0, nn.Conv2d):
    #         if np.sum(end_mask) == 0:
    #             continue
    #         idx0 = np.squeeze(np.argwhere(start_mask))
    #         idx1 = np.squeeze(np.argwhere(end_mask))
    #         # random set for test
    #         # new_end_mask = np.asarray(end_mask.cpu().numpy())
    #         # new_end_mask = np.append(new_end_mask[int(len(new_end_mask)/2):], new_end_mask[:int(len(new_end_mask)/2)])
    #         # idx1 = np.squeeze(np.argwhere(new_end_mask))
    #
    #         # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
    #         if idx0.size == 1:
    #             idx0 = np.resize(idx0, (1,))
    #         if idx1.size == 1:
    #             idx1 = np.resize(idx1, (1,))
    #         w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
    #         w1 = w1[idx1.tolist(), :, :, :].clone()
    #         m1.weight.data = w1.clone()
    #     elif isinstance(m0, nn.Linear):
    #         idx0 = np.squeeze(np.argwhere(start_mask))
    #         if idx0.size == 1:
    #             idx0 = np.resize(idx0, (1,))
    #         m1.weight.data = m0.weight.data[:, idx0].clone()
    #         m1.bias.data = m0.bias.data.clone()

    layer_id_in_cfg = 0
    start_mask = np.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    parameter_buffer = {}

    for m0 in model.modules():
        if isinstance(m0, nn.BatchNorm2d):
            key = str(layer_id_in_cfg) + 'BatchNorm'
            value = []
            if np.sum(end_mask) == 0:
                pass
            else:
                idx1 = np.squeeze(np.argwhere(end_mask))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                value.append(m0.weight.data[idx1.tolist()].clone())
                value.append(m0.bias.data[idx1.tolist()].clone())
                value.append(m0.running_mean[idx1.tolist()].clone())
                value.append(m0.running_var[idx1.tolist()].clone())
                start_mask = copy.copy(end_mask)
            parameter_buffer[key] = value
            layer_id_in_cfg += 1
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]

        elif isinstance(m0, nn.Conv2d):
            key = str(layer_id_in_cfg) + 'Conv'
            value = []
            if np.sum(end_mask) == 0:
                pass
            else:
                idx0 = np.squeeze(np.argwhere(start_mask))
                idx1 = np.squeeze(np.argwhere(end_mask))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))

                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                value.append(w1.clone())
            parameter_buffer[key] = value
        elif isinstance(m0, nn.Linear):
            key = str(layer_id_in_cfg) + 'Linear'
            value = []
            idx0 = np.squeeze(np.argwhere(start_mask))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            value.append(m0.weight.data[:, idx0].clone())
            value.append(m0.bias.data.clone())
            parameter_buffer[key] = value

    layer_id_in_cfg = 0
    for m1 in newmodel.modules():
        if isinstance(m1, nn.BatchNorm2d):
            key = str(layer_id_in_cfg) + 'BatchNorm'
            while len(parameter_buffer[key]) == 0:
                layer_id_in_cfg += 1
                key = str(layer_id_in_cfg) + 'BatchNorm'
            m1.weight.data = parameter_buffer[key][0]
            m1.bias.data = parameter_buffer[key][1]
            m1.running_mean = parameter_buffer[key][2]
            m1.running_var = parameter_buffer[key][3]
            layer_id_in_cfg += 1

        elif isinstance(m1, nn.Conv2d):
            key = str(layer_id_in_cfg) + 'Conv'
            while len(parameter_buffer[key]) == 0:
                layer_id_in_cfg += 1
                key = str(layer_id_in_cfg) + 'Conv'
            m1.weight.data = parameter_buffer[key][0]
        elif isinstance(m1, nn.Linear):
            key = str(layer_id_in_cfg) + 'Linear'
            m1.weight.data = parameter_buffer[key][0]
            m1.bias.data = parameter_buffer[key][1]
            pass
    return newmodel



def get_batch_jacobian(net, x, target, device):
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach()


def check_score(model, train_loader):
    newmodel = copy.deepcopy(model)
    reset_seed()
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
    s = []

    for j in range(5):
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


def check_channel_score(model, train_loader):
    newmodel = copy.deepcopy(model)
    reset_seed()
    def counting_forward_hook(module, inp, out):
        try:
            # if not module.visited_backwards:
            #     return
            if isinstance(inp, tuple):
                inp = inp[0]
            K_layer = np.zeros((args.test_batch_size, args.test_batch_size))
            inp = inp.permute(1, 0, 2, 3)
            inp = inp.view(inp.size(0), inp.size(1), -1)
            inp = (inp > 0).float()
            score_list = []
            for i in range(inp.size(0)):
                x = inp[i]
                K1 = x @ x.t()
                K2 = (1. - x) @ (1. - x.t())
                K = K1.cpu().numpy() + K2.cpu().numpy()
                K_layer += K
                s_, ld = np.linalg.slogdet(K)
                score_list.append(ld)
            s_, ld = np.linalg.slogdet(K_layer)
            newmodel.layer_score.append(ld)
            newmodel.channel_score.append(score_list)
        except Exception as e:
            print(e)

    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True

    def counting_backward_hook_ini(module, inp, out):
        newmodel.layer_score = []
        newmodel.channel_score = []

    for name, module in newmodel.named_modules():
        if 'ReLU' in str(type(module)):
            # hooks[name] = module.register_forward_hook(counting_hook)
            module.register_forward_hook(counting_forward_hook)

        if name == 'feature.0':
            module.register_forward_hook(counting_backward_hook_ini)

    newmodel = newmodel.to(device)
    s = []
    layer_s = []
    for j in range(5):
        data_iterator = iter(train_loader)
        x, target = next(data_iterator)
        x2 = torch.clone(x)
        x2 = x2.to(device)
        x, target = x.to(device), target.to(device)
        # jacobs, labels, y = get_batch_jacobian(newmodel, x, target, device)
        newmodel(x2.to(device))
        s.append(copy.deepcopy(newmodel.channel_score))
        layer_s.append(copy.deepcopy(newmodel.layer_score))


    layer_s = np.array(layer_s)
    layer_s = np.mean(layer_s, axis=0)
    channel_score = []

    for channel in range(len(s[0])):
        tep = []
        for j in range(len(s)):
            tep.append(s[j][channel])
        tep = np.array(tep)
        tep = np.mean(tep, axis=0)
        channel_score.append(tep)
    # s = np.array(s).astype(float)
    # # s = np.mean(s, axis=0)
    # s = s.transpose()
    # tep = np.array(s[0])
    return layer_s, channel_score

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

def create_cfg(cfg_mask_all, indicator):
    form = copy.deepcopy(cfg_mask_all)
    while 'M' in form:
        form.remove('M')
    # np.random.shuffle(mask_all)
    cfg_mask = []
    end = 0
    for i in form:
        cfg_mask.append(indicator[end:end + i])
        end += i
    cfg = []
    index = 0
    for i in range(len(cfg_mask_all)):
        if cfg_mask_all[i] != 'M':
            if np.sum(cfg_mask[index]) != 0:
                cfg.append(int(np.sum(cfg_mask[index])))
            index += 1
        else:
            cfg.append('M')
    return cfg, cfg_mask

def random_search(cfg_mask_all, percent):

    form = copy.deepcopy(cfg_mask_all)
    while 'M' in form:
        form.remove('M')
    total = np.sum(form)
    choose_num = int(total * percent)
    mask_all = np.append(np.ones(choose_num), np.zeros(total - choose_num))
    record_dict = {}
    for i in range(len(mask_all)):
        record_dict[i] = []
    score_test = 0
    trail_index = 0
    while score_test < 1450:
        for i in range(100):
            np.random.shuffle(mask_all)
            cfg, cfg_mask = create_cfg(cfg_mask_all, mask_all)
            model_new = create_model(model, cfg, cfg_mask)
            score = check_score(model_new, train_loader)
            for i in range(len(mask_all)):
                if not mask_all[i]:
                    record_dict[i].append(score)

        average_score = pd.DataFrame([], columns=["position", "score"])

        for i in range(len(mask_all)):
            info_dict = {
                'position':i,
                'score':np.max(record_dict[i])
            }
            average_score = average_score.append(info_dict, ignore_index=True)

        average_score = average_score.sort_values(by=['score'], ascending=False)
        indexes = average_score['position'][0: int(len(average_score) * percent)]
        indexes = indexes.astype(int)
        indicator = np.ones(total)
        indicator[indexes] = 0
        cfg, cfg_mask = create_cfg(cfg_mask_all, indicator)
        model_new = create_model(model, cfg, cfg_mask)
        score = check_score(model_new, train_loader)
        info_dict = {
            'index': trail_index,
            'cfg': cfg,
            'cfg_mask': cfg_mask,
            'score': score
        }
        wandb.log(info_dict)
        print('The trial of {}, the score is {:.2f}'.format(trail_index, score))
        trail_index += 1
        if score > score_test:
            score_test = score
            np.save('{:.2f}.npy'.format(score_test), info_dict)


def reset_seed(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def count_channel(model):
    cfg_mask_all = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            cfg_mask_all.append(m.weight.data.shape[0])
        elif isinstance(m, nn.MaxPool2d):
            cfg_mask_all.append('M')

    form = copy.deepcopy(cfg_mask_all)
    while 'M' in cfg_mask_all:
        cfg_mask_all.remove('M')
    total = np.sum(cfg_mask_all)
    return total, form

def greedy_search_new(model, percent, train_loader):
    buffer = 33
    total, form = count_channel(model)
    channel_num = total
    progress_index = 0
    indicator = np.ones(total)
    while channel_num > total * percent:
        # indicator = np.ones(total)
        score_dict = pd.DataFrame([], columns=['index', 'score'])
        for position in range(total):
            if indicator[position]:
                indicator_tep = copy.deepcopy(indicator)
                indicator_tep[position] = 0
                cfg, cfg_mask = create_cfg(form, indicator_tep)
                model_new = create_model(model, cfg, cfg_mask)
                score = check_score(model_new, train_loader)
                info_dict = {
                    'index': position,
                    'score': score
                }
                score_dict = score_dict.append(info_dict, ignore_index=True)
                print('{}----{}/{}: score {:.2f}'.format(channel_num, position, total, score))
            else:
                score = -1
                info_dict = {
                    'index': position,
                    'score': -1,
                }
                score_dict = score_dict.append(info_dict, ignore_index=True)
                print('{}----{}/{}: score {:.2f}'.format(channel_num, position, total, score))

        score_dict = score_dict.sort_values(by=['score'], ascending=False)
        indexes = score_dict['index'][0:buffer]
        indexes = indexes.astype(int)
        indicator[indexes] = 0
        cfg, cfg_mask = create_cfg(form, indicator)
        channel_num = count_cfg_channel(cfg)
        newmodel = create_model(model, cfg, cfg_mask)
        score = check_score(newmodel, train_loader)
        info_dict = {
            'index': progress_index*buffer,
            'score': score
        }
        wandb.log(info_dict)
        progress_index += 1

        # for i in range(len(cfg_mask)):
        #     cfg = copy.copy(cfg_mask)
        #     cfg[i] -= 1
        #     newmodel = models.__dict__[args.arch](dataset=args.dataset, cfg=cfg)
        #     score = check_score(newmodel, train_loader)
        #     score_dict[i] = score
        # print(score_dict)

    save_dict = {
        'state_dict': newmodel.state_dict(),
        'cfg': cfg,
        'cfg_mask': cfg_mask,
        'score': score
    }
    torch.save(save_dict, '{:.2f}.pth'.format(score))
    # np.save('{:.2f}.npy'.format(score), save_dict)


def channel_score_search(model, percent, train_loader):
    total, form = count_channel(model)
    indicator = np.ones(total)
    cfg, cfg_mask = create_cfg(form, indicator)
    new_model = copy.copy(model)
    for i in range(0, len(cfg_mask)):
        # score = check_score(new_model, train_loader)
        channel_score = check_channel_score(new_model, train_loader)
        channel_score = channel_score[i]
        channel_score_rank = copy.deepcopy(channel_score)
        channel_score_rank.sort()
        thre_index = int(len(channel_score)*percent)
        thre_score = channel_score_rank[thre_index-1]
        mask = [0 if j <= thre_score else 1 for j in channel_score]
        cfg_mask[i] = mask
        indicator_tep = []
        for j in cfg_mask:
            indicator_tep += list(j)
        cfg_new, cfg_mask_new = create_cfg(form, indicator_tep)
        new_model = create_model(model, cfg_new, cfg_mask_new)
        score = check_score(new_model, train_loader)
        print(score)

def rate_check(model, percent, train_loader):
    xshape = (1, 3, 32, 32)
    flops_original, param_original = get_model_infos(model, xshape)
    total, form = count_channel(model)
    indicator = np.ones(total)
    cfg, cfg_mask = create_cfg(form, indicator)
    f_list = []
    p_list = []

    for i in range(len(cfg_mask)):
        cfg_mask_new = copy.deepcopy(cfg_mask)
        cfg_mask_new[i][0:30] = 0
        indicator_new = []
        for one in cfg_mask_new:
            indicator_new += list(one)
        cfg_new, cfg_mask_new = create_cfg(form, indicator_new)
        model_new = create_model(model, cfg_new, cfg_mask_new)
        flops, param = get_model_infos(model_new, xshape)
        flops_rate = (flops_original-flops)/flops_original
        param_rate = (param_original-param)/param_original
        f_list.append(flops_rate)
        p_list.append(param_rate)
    print("")

def channel_remove_check(model, train_loader):
    baseline_f_rate = 260292527 / 313772032
    baseline_p_rate = 8300726 / 15299748
    xshape = (1, 3, 32, 32)
    flops_original, param_original = get_model_infos(model, xshape)
    score = check_score(model, train_loader)
    score_layer_original, score_channel_original = check_channel_score(model, train_loader)
    total, form = count_channel(model)
    indicator = np.ones(total)
    cfg_original, cfg_mask_original = create_cfg(form, indicator)
    cfg_mask = copy.deepcopy(cfg_mask_original)
    model_new = copy.deepcopy(model)
    for i in range(len(cfg_mask_original)-1):
        score_layer, score_channel = check_channel_score(model_new, train_loader)
        cfg_mask[i] = score_channel[i] != -np.inf
        indicator = []
        for one in cfg_mask:
            indicator += list(one)
        cfg, cfg_mask = create_cfg(form, indicator)
        model_new = create_model(model, cfg, cfg_mask)

    score_layer, score_channel = check_channel_score(model_new, train_loader)
    flops, param = get_model_infos(model_new, xshape)
    f_rate = flops/flops_original
    p_rate = param/param_original
    score_prune = check_score(model_new, train_loader)

    for i in range(len(score_channel)):
        print(np.sum(score_channel[i] == -np.inf), len(score_channel[i]))

    torch.save(model_new, 'channel_remove_rough_{:.2f}.pth'.format(score_prune))


def layer_remove_check(model, train_loader):

    baseline_f_rate = 260292527/313772032
    baseline_p_rate = 8300726/15299748

    xshape = (1, 3, 32, 32)
    flops_original, param_original = get_model_infos(model, xshape)
    score = check_score(model, train_loader)
    score_layer_original, score_channel_original = check_channel_score(model, train_loader)
    total, form = count_channel(model)
    indicator = np.ones(total)
    cfg, cfg_mask = create_cfg(form, indicator)

    # cut_list = [11,10,8,7,4,5,2,0]
    cut_list = [11, 10]
    for one in cut_list:
        cfg_mask[one] = np.zeros(len(cfg_mask[one]))

    # cfg_mask[4] = np.zeros(len(cfg_mask[4]))
    indicator_new = []
    for one in cfg_mask:
        indicator_new += list(one)
    cfg_new, cfg_mask_new = create_cfg(form, indicator_new)
    model_new = create_model(model, cfg_new, cfg_mask_new)
    flops, param = get_model_infos(model_new, xshape)
    score_prune = check_score(model_new, train_loader)
    f_rate = flops/flops_original
    p_rate = param/param_original

    score_layer, score_channel = check_channel_score(model_new, train_loader)
    # for one in score_channel:
    #     print(np.sum(one == -np.inf), len(one))


    #
    # save_dict = {
    #     'state_dict': model_new.state_dict(),
    #     'cfg': cfg_new,
    #     'cfg_mask': cfg_mask_new,
    #     'score': score_prune
    # }
    # torch.save(model_new, '{:.2f}.pth'.format(score_prune))
    # model_new = models.__dict__[args.arch](dataset=args.dataset, cfg = cfg_new)
    # score_prune = check_score(model_new, train_loader)
    # score_layer, score_channel = check_channel_score(model_new, train_loader)
    torch.save(model_new, '{:.2f}.pth'.format(score_prune))
    print("")


    # f_list = []
    # p_list = []
    #
    # for i in range(len(cfg_mask)):
    #     cfg_mask_new = copy.deepcopy(cfg_mask)
    #     cfg_mask_new[i][0:30] = 0
    #     indicator_new = []
    #     for one in cfg_mask_new:
    #         indicator_new += list(one)
    #     cfg_new, cfg_mask_new = create_cfg(form, indicator_new)
    #     model_new = create_model(model, cfg_new, cfg_mask_new)
    #     flops, param = get_model_infos(model_new, xshape)
    #     flops_rate = (flops_original-flops)/flops_original
    #     param_rate = (param_original-param)/param_original
    #     f_list.append(flops_rate)
    #     p_list.append(param_rate)


def count_cfg_channel(cfg):
    form = copy.deepcopy(cfg)
    while 'M' in form:
        form.remove('M')
    channel = np.sum(form)
    return channel


def create_base(model, train_loader):
    total, form = count_channel(model)
    indicator = []
    for one in form:
        if one != 'M':
            tep = np.zeros(one)
            tep[0] = 1
            indicator.append(tep)

    for i in range(len(indicator)):
        record = pd.DataFrame([], columns=["index", "score"])
        for j in range(len(indicator[i])):
            tep = np.zeros(len(indicator[i]))
            tep[j] = 1
            indicator_tep = copy.deepcopy(indicator)
            indicator_tep[i] = tep
            indicator_list = []
            for k in indicator_tep:
                indicator_list += list(k)
            indicator_tep = np.array(indicator_list).astype(int)
            cfg, cfg_mask = create_cfg(form, indicator_tep)
            new_model = create_model(model, cfg, cfg_mask)
            score = check_score(new_model, train_loader)
            info_dict = {
                "index": j,
                "score": score
            }
            record = record.append(info_dict, ignore_index=True)
            # print("for the {}-th module, tried {}/{}, the score is {:.2f}".format(i, j, len(indicator[i]), score))
        record = record.sort_values(by=['score'], ascending=False)
        indexes = record['index'][0]
        tep = np.zeros(len(indicator[i]))
        tep[int(indexes)] = 1
        indicator[i] = tep
        print("for the {}-th module, tried {}/{}, the score is {:.2f}".format(i, j, len(indicator[i]), record['score'][0]))
    indicator_list = []
    for i in indicator:
        indicator_list += (list(i))
    indicator = np.array(indicator_list)
    return indicator


def greedy_search_increase(model, percent, train_loader):

    buffer = 11
    total, form = count_channel(model)
    indicator = create_base(model, train_loader)
    left_channels = int(total*percent - np.sum(indicator))
    while left_channels > 0:
        record = pd.DataFrame([], columns=["index", "score"])
        for i in range(len(indicator)):
            if indicator[i]:
                print("already chosen")
            else:
                indicator_tep = copy.deepcopy(indicator)
                cfg, cfg_mask = create_cfg(form, indicator_tep)
                new_model = create_model(model, cfg, cfg_mask)
                score = check_score(new_model, train_loader)
                info_dict = {
                    "index": i,
                    "score": score
                }
                record = record.append(info_dict, ignore_index=True)
        record = record.sort_values(by=['score'], ascending=False)
        if left_channels < buffer:
            buffer = left_channels
        indexes = record['index'][0:buffer]
        indexes = indexes.astype(int)
        indicator[indexes] = 1

        cfg, cfg_mask = create_cfg(form, indicator)
        new_model = create_model(model, cfg, cfg_mask)
        score = check_score(new_model, train_loader)
        left_channels = int(total * percent - np.sum(indicator))
        print("Still have {} channels to prune, now the highest score is {:.2f}".format(left_channels, score))
        info_dict = {
            "channels": np.sum(indicator),
            "score": score
        }
        wandb.log(info_dict)

    save_dict = {
        'state_dict': new_model.state_dict(),
        'cfg': cfg,
        'cfg_mask': cfg_mask,
        'score': score
    }
    torch.save(save_dict, '{:.2f}.pth'.format(score))

def greedy_search(model, percent, train_loader):
    total, form = count_channel(model)
    index_trial = 0
    while count_cfg_channel(form) > total*percent:
        score_list = []
        for i in range(len(form)):
            if form[i] != 'M':
                cfg = copy.deepcopy(form)
                cfg[i] -= 1
                newmodel = models.__dict__[args.arch](dataset=args.dataset, cfg = cfg)
                score = check_score(newmodel, train_loader)
                score_list.append(score)
            else:
                score_list.append(-1)
        max_score = max(score_list)
        index = score_list.index(max_score)

        info_dict = {
            'index': index_trial,
            'cfg': form,
            'score': max_score
        }
        index_trial +=1
        wandb.log(info_dict)
        form[index] -= 1
        print(max_score)

    info_dict = {
        'cfg': form,
        'score': max_score
    }
    np.save('{:.2f}.npy'.format(score), info_dict)


    # cfg_mask_all = []
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         cfg_mask_all.append(m.weight.data.shape[0])
    #     elif isinstance(m, nn.MaxPool2d):
    #         cfg_mask_all.append('M')
    #
    # form = copy.deepcopy(cfg_mask_all)
    # while 'M' in form:
    #     form.remove('M')
    # total = np.sum(form)
    # indicator = np.ones(total)

    # cfg_mask = []
    # end = 0
    # for i in form:
    #     cfg_mask.append(indicator[end:end + i])
    #     end += i
    # bn = torch.zeros(total)
    # cfg = []
    # index = 0
    # for i in range(len(cfg_mask_all)):
    #     if cfg_mask_all[i] != 'M':
    #         cfg.append(int(np.sum(cfg_mask[index])))
    #         index += 1
    #     else:
    #         cfg.append('M')
    # model_new = create_model(model, cfg, cfg_mask)
    # score = check_score(model_new, train_loader)
    # score_dict = pd.DataFrame([], columns=['index', 'score'])
    # score_dict.to_csv('score_indicator.csv')
    # score_dict = score_dict.sort_values(by=['score'], ascending=False)
    # indexes = score_dict['index'][0: int(len(score_dict)*percent)]
    # indexes = indexes.astype(int)
    # indicator_tep = copy.deepcopy(indicator)
    # indicator_tep[indexes] = 0
    # cfg_mask = []
    # end = 0
    # for i in form:
    #     cfg_mask.append(indicator_tep[end:end + i])
    #     end += i
    # cfg = []
    # index = 0
    # for i in range(len(cfg_mask_all)):
    #     if cfg_mask_all[i] != 'M':
    #         cfg.append(int(np.sum(cfg_mask[index])))
    #         index += 1
    #     else:
    #         cfg.append('M')
    # model_new = create_model(model, cfg, cfg_mask)
    # score = check_score(model_new, train_loader)

    # info_dict = {
    #     'cfg': cfg,
    #     'cfg_mask': cfg_mask,
    #     'score': score
    # }
    # np.save('{:.2f}.npy'.format(score), info_dict)


if __name__ == '__main__':
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    if args.cuda:
        model.cuda()


    wandb_project = 'pruning_score'
    name = '128_greedy'
    # wandb.init(project=wandb_project, name=name)
    #
    # random_search(cfg_mask_all, args.percent)
    # channel_score_search(model, args.percent, train_loader)
    # greedy_search(model, args.percent, train_loader)
    # layer_remove_check(model, train_loader)
    # rate_check(model, args.percent, train_loader)
    channel_remove_check(model, train_loader)
    #
    # data = np.load('1633.66.npy', allow_pickle=True)
    # data = data.item()
    # cfg = data['cfg']
    # cfg_mask = data['mask_cfg']
    # for i in range(len(cfg_mask)):
    #     cfg_mask[i] = np.asarray(cfg_mask[i].cpu().numpy())
    # score = check_score(model, cfg, cfg_mask)
    # print(score)

