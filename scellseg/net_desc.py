import os, copy, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import datetime
import cv2

from . import transforms, io, dynamics, utils, metrics, dataset, core
from .net_utils import *
from torch.utils.data import DataLoader


class Extractor(nn.Module):
    def __init__(self, nbase, sz, residual_on=True, style_on=True,
                 concatenation=False, attn_on=False, dense_on=False, style_scale_on=True):
        super(Extractor, self).__init__()
        self.choose_layers = [3, 3, 3, 3]
        self.style_channels = [256, 256, 256, 256]  # only use the last feature map of conv unit, shape

        self.downsample = downsample(nbase, sz, residual_on=residual_on)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])  # [32, 64, 128, 256, 256]

        if style_scale_on:
            self.choose_layers = [0, 1, 2, 3]
            self.style_channels = [256+128+64+32, 256+128+64, 256+128, 256]  # hierarchical style concat shape

        self.make_style = makeStyle()
        self.upsample = upsample(nbaseup, sz, residual_on=residual_on, concatenation=concatenation,
                                 attn_on=attn_on, dense_on=dense_on, style_channels=self.style_channels)
        self.style_on = style_on

    def forward(self, data, use_upsample=True, mkldnn=False):
        if mkldnn:
            data = data.to_mkldnn()
        T0 = self.downsample(data)

        styles = []
        for n in self.choose_layers:
            if mkldnn:
                style = self.make_style(T0[n].to_dense())
            else:
                style = self.make_style(T0[n])
            styles.append(style)

        style0 = styles

        if self.style_channels[0] > 257 and self.style_channels[1] < 449:
            for n in [-2, -3, -4]:
                styles[n] = torch.cat((styles[n + 1], styles[n]), dim=1)
        if self.style_channels[1] > 449:
            style_concat_all = torch.cat((styles[0], styles[1], styles[2], styles[3]), dim=1)
            for n in range(4):
                styles[n] = style_concat_all

        if not self.style_on:
            styles = [style * 0 for style in styles]
        if use_upsample:
            T1 = self.upsample(styles, T0, mkldnn=mkldnn)
        else:
            T1 = None

        return T1, style0[-1]


class Tasker(nn.Module):
    def __init__(self, nin, nout, task_mode='cellpose'):
        super(Tasker, self).__init__()
        # self.mkldnn = mkldnn if mkldnn is not None else False

        self.base_bn = nn.BatchNorm2d(nin, eps=1e-5)
        self.base_relu = nn.ReLU(inplace=True)
        self.base_conv = nn.Conv2d(nin, nout, 1, padding=1 // 2)

    def forward(self, T0, the_vars=None, mkldnn=False):
        if the_vars is not None:
            self.base_conv.weight.data = the_vars[0]
            self.base_conv.bias.data = the_vars[1]
        T0 = self.base_conv(self.base_relu(self.base_bn(T0)))

        if mkldnn:
            T0 = T0.to_dense()
        return T0

class LossFn(nn.Module):
    def __init__(self, task_mode):
        super(LossFn, self).__init__()
        self.task_mode = task_mode
        if task_mode == 'cellpose' or self.task_mode == 'hover':
            self.criterion = nn.MSELoss(reduction='mean', size_average=True)
            self.criterion2 = nn.BCEWithLogitsLoss(reduction='mean')
            self.criterion3 = nn.CrossEntropyLoss()

            self.alpha = torch.tensor(0.2, requires_grad=True)
        elif 'unet' in task_mode:
            self.alpha = torch.tensor(0.2, requires_grad=True)
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, lbl, y, device, styles=None):
        if self.task_mode == 'cellpose' or self.task_mode == 'hover':
            veci = (5. * lbl[:, 1:3]).float().to(device)
            cellprob = (lbl[:, 0] > .5).float().to(device)
            loss_map = self.criterion(y[:, :2], veci)
            loss_map /= 2.
            loss_cellprob = self.criterion2(y[:, 2], cellprob)

            loss_contrast = 0
            if styles is not None:
                style, style_pos, style_neg = styles
                # target_pos=torch.ones(style.shape[0]).to(device)
                # target_neg=(torch.ones(style.shape[0])).to(device)
                # loss_pos = nn.functional.cosine_embedding_loss(style, style_pos, target=target_pos)
                # loss_neg = nn.functional.cosine_embedding_loss(style, style_neg, target=target_neg)
                if isinstance(style, list):
                    for i in range(len(style)):
                        loss_posi = mse_pairs(style[i], style_pos[i], self.criterion)
                        loss_negi = mse_pairs(style[i], style_neg[i], self.criterion)
                        print('>>>', loss_posi, loss_negi)
                        loss_contrast += torch.pow((loss_posi), 2) + torch.pow((1 - loss_negi), 2)
                else:
                    loss_pos = mse_pairs(style, style_pos, self.criterion)
                    loss_neg = mse_pairs(style, style_neg, self.criterion)
                    loss_contrast = torch.pow(loss_pos, 1) / (torch.pow(loss_neg, 1) + torch.tensor(1e-5))

            loss = loss_map * 1. + loss_cellprob * 1. + loss_contrast * nn.functional.sigmoid(self.alpha) # 系数影响的是对应loss更新的速度
        elif 'unet' in self.task_mode:
            nclasses = int(self.task_mode.split('-')[-1])
            if lbl.shape[1] > 1 and nclasses > 2:
                boundary = lbl[:, 1] <= 4
                lbl = lbl[:, 0]
                lbl[boundary] *= 2
            else:
                lbl = lbl[:, 0]
            lbl = lbl.float().to(device)
            loss = 8 * 1. / nclasses * self.criterion(y, lbl.long())
        return loss


class sCSnet(nn.Module):
    def __init__(self, nbase, nout, sz, diam_mean=30.0,
                 residual_on=True, style_on=True, concatenation=False, mkldnn=False, update_step=1,
                 attn_on=False, dense_on=False, style_scale_on=True, phase='eval',
                 device=None, net_avg=False, task_mode='cellpose'):

        super(sCSnet, self).__init__()
        self.nbase = nbase  # [2, 32, 64, 128, 256]
        self.nout = nout  # 3
        self.sz = sz
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.mkldnn = mkldnn
        self.net_avg= net_avg
        self.device = device
        self.phase = phase
        self.task_mode = task_mode
        self.update_step = update_step

        if 'unet' in task_mode:
            self.update_step = 1

        self.last_result = -np.inf

        # TODO: 未适配CPU模式下mkldnn加速
        self.extractor = Extractor(nbase, sz, residual_on=residual_on, style_on=style_on, style_scale_on=style_scale_on,
             concatenation=concatenation, attn_on=attn_on, dense_on=dense_on)
        self.tasker = Tasker(nbase[1], nout, task_mode=task_mode)
        self.loss_fn = LossFn(task_mode=task_mode)

    def forward(self, inp):
        """
            The function to forward the model.
        """
        if self.phase == 'shot_train':
            data_shot, label_shot = inp
            return self.shot_train_forward(data_shot, label_shot)
        else:
            return self.eval_forward(inp)

    def shot_train_forward(self, data_shot, label_shot):
        for _ in range(0, self.update_step):
            embedding_shot, style = self.extractor(data_shot, mkldnn=self.mkldnn)
            logits = self.tasker(embedding_shot, mkldnn=self.mkldnn)

            styles = None
            if self.contrast_on:
                pos_imgs, neg_imgs, neg_lbls = self.pair_gen.get_pair(n_sample1class=len(data_shot))
                embedding_pos, style_pos = self.extractor(pos_imgs, mkldnn=self.mkldnn)
                embedding_neg, style_neg = self.extractor(neg_imgs, mkldnn=self.mkldnn)

                styles = (style, style_pos, style_neg)

                if self.pair_gen.use_negative_masks:
                    label_shot = torch.cat((label_shot, neg_lbls), dim=0)
                    logits_neg = self.tasker(embedding_neg, mkldnn=self.mkldnn)
                    logits = torch.cat((logits, logits_neg), dim=0)

            loss = self.loss_fn(label_shot, logits, device=self.device, styles=styles)
            # if _ == 0:
            #     print("loss_now", loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # model_dict = self.state_dict()

        # save_path = r'G:\Python\9-Project\1-flurSeg\sCellSeg\output\models\preeval'
        # transfer_path = os.path.join(save_path, 'transfer_cellpose')
        # now_result = -loss.item()  # loss_last模式( loss.item() ) or loss_first模式( loss_first )
        # if self.last_result == 0:
        #     print('\033[1;34m>>>>get first weights\033[0m')
        #     self.last_result = now_result
        #     self.save_model(transfer_path)
        # elif now_result > self.last_result:
        #     print('\033[1;34m>>>>get better weights\033[0m')
        #     self.last_result = now_result
        #     self.save_model(transfer_path)

        return loss.item()

    def eval_forward(self, data):
        embedding, style = self.extractor(data, mkldnn=self.mkldnn)
        logits_q = self.tasker(embedding, mkldnn=self.mkldnn)
        return logits_q, style

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, cpu=False, last_conv_on=True):
        """
        Because we re-divided the Cellpose architecture,
            1. besides provide a way to load the model file pre-trianed by our Scellseg model
            2. we also design the way to load model file pre-trained by Cellpose model
        last_conv_on: bool (optional, default True)
            if Flase, the weights for last conv in pre-trained model will not be loaded
        """
        if not cpu:
            premodel_dict_0 = torch.load(filename)
        else:
            # self.__init__(self.nbase,self.nout,self.sz,self.residual_on,self.style_on,self.concatenation,self.mkldnn)
            premodel_dict_0 = torch.load(filename, map_location=torch.device('cpu'))

        # parameter adaptation
        premodel_dict_1 = copy.deepcopy(premodel_dict_0)
        is_cellpose_provide = True
        for k, v in premodel_dict_1.items():
            if ('extractor' in k):
                is_cellpose_provide = False
                break

        if is_cellpose_provide:
            premodel_dict = {'extractor.' + k: v for k, v in premodel_dict_1.items() if 'output' not in k}  # 改名

            for k, v in premodel_dict_1.items():  # for model file pre-trained by Cellpose model
                k_new = None
                if 'output.0' in k:
                    k_new = k.replace('output.0', 'tasker.base_bn')
                if last_conv_on and ('output.2' in k):
                    k_new = k.replace('output.2', 'tasker.base_conv')

                if k_new is not None:
                    premodel_dict[k_new] = premodel_dict_0.pop(k)
        else:
            premodel_dict = copy.deepcopy(premodel_dict_0)

        if (not last_conv_on):
            for k, v in premodel_dict_1.items():
                if 'base_conv' in k:
                    premodel_dict.pop(k)

        model_dict = self.state_dict()
        model_dict.update(premodel_dict)
        self.load_state_dict(model_dict, strict=False)


def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    fast_weights = []
    for (p, g), (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * g.data
        s[:] = beta2 * s + (1 - beta2) * g.data ** 2
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        fast_weights.append(p.data - hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + eps))
    hyperparams['t'] += 1
    for fast_weight in fast_weights:
        fast_weight.requires_grad = True
    return fast_weights

def mse_pairs(styles, styles_, loss_fn=None):
    loss = 0
    for style in styles:
        for style_ in styles_:
            loss += loss_fn(style, style_)
    return loss

