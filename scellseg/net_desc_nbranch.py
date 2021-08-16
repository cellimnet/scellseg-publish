"""
这里定义了计算图
最后一层Conv要分离做Tasker
"""

import os, copy, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import datetime
import cv2

from . import transforms, io, dynamics, utils, metrics
from .net_utils3 import *
from .loss_utils import *


class Extractor(nn.Module):
    def __init__(self, nbase, sz, ndecoder=1, residual_on=True, mkldnn=False, style_on=True,
                 concatenation=False, attn_on=False, dense_on=False):
        super(Extractor, self).__init__()
        self.choose_layers = [3, 3, 3, 3]
        self.style_channels = [256, 256, 256, 256]  # 只使用高维的

        self.downsample = downsample(nbase, sz, residual_on=residual_on)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])  # [32, 64, 128, 256, 256]

        # self.choose_layers = [0, 1, 2, 3]
        # # self.style_channels = [256+128+64+32, 256+128+64, 256+128, 256]  # 多层次concat模式
        # self.style_channels = [32, 64, 128, 256]  # 多层次

        self.upsample = nn.Sequential()
        for i in range(ndecoder):
            self.upsample.add_module('upsample_branch_%d' % (i),
                                     upsample(nbaseup, sz, residual_on=residual_on, concatenation=concatenation,
                                              attn_on=attn_on, dense_on=dense_on, style_channels=self.style_channels))
        self.make_style = makeStyle()
        self.mkldnn = mkldnn
        self.style_on = style_on

    def forward(self, data):
        if self.mkldnn:
            data = data.to_mkldnn()
        T0 = self.downsample(data)

        styles = []
        for n in self.choose_layers:
            if self.mkldnn:
                style = self.make_style(T0[n].to_dense())
            else:
                style = self.make_style(T0[n])
            styles.append(style)

        if self.style_channels[0] > 257:
            for n in [-2, -3, -4]:
                styles[n] = torch.cat((styles[n + 1], styles[n]), dim=1)

        style0 = styles

        if not self.style_on:
            styles = [style * 0 for style in styles]

        Ts = []
        for i in range(len(self.upsample)):
            t = self.upsample[i](styles, T0, mkldnn=self.mkldnn)
            Ts.append(t)

        return Ts, style0[-1]


class Tasker(nn.Module):
    def __init__(self, nin, nout, mkldnn=False, task_mode='cellpose'):
        super(Tasker, self).__init__()
        # self.mkldnn = mkldnn if mkldnn is not None else False
        self.mkldnn = mkldnn
        self.base_bn = nn.BatchNorm2d(nin, eps=1e-5)
        self.base_relu = nn.ReLU(inplace=True)
        self.base_conv = nn.Conv2d(nin, nout, 1, padding=1 // 2)  # 这里的参数再看一下kernel_size=3，padding=1??

    def forward(self, T0, the_vars=None):
        if the_vars is not None:
            print('the_vars', the_vars)
            print('self.base_conv.weight', self.base_conv.weight)
            self.base_conv.weight.data = the_vars[0]
            self.base_conv.bias.data = the_vars[1]
        T0 = self.base_conv(self.base_relu(self.base_bn(T0)))

        if self.mkldnn:
            T0 = T0.to_dense()
        return T0

    def parameters(self):
        return nn.ParameterList([self.base_conv.weight, self.base_conv.bias])  # 加上更新BN层会降低性能


class LossFn(nn.Module):
    def __init__(self, task_mode, multi_class=False):
        super(LossFn, self).__init__()
        self.task_mode = task_mode
        self.multi_class = multi_class
        if task_mode == 'cellpose' or self.task_mode == 'hover':
            self.criterion = nn.MSELoss(reduction='mean', size_average=True)  # 能把背景去除的很干净
            # self.criterion = nn.SmoothL1Loss(reduction='mean', size_average=True)  # 理论上学的更准一点
            self.criterion2 = nn.BCEWithLogitsLoss(reduction='mean')
            self.criterion3 = nn.CrossEntropyLoss()

            # self.alpha = torch.tensor(1., requires_grad=True)
        elif 'classic' in task_mode:
            self.criterion = nn.CrossEntropyLoss()  #

    def forward(self, lbl, y, device, styles=None):
        if self.task_mode == 'cellpose' or self.task_mode == 'hover':
            # norm_size = torch.max(y[:, :2])
            # if norm_size < 5:
            #     norm_size = 5
            veci = (5. * lbl[:, 1:3]).float().to(device)
            cellprob = (lbl[:, 0] > .5).float().to(device)  # lbl第一个通道是概率
            # loss = self.criterion(y[:, :2], veci)  # 梯度计算上的loss
            # y_expand = normalization(torch.unsqueeze(y[:, -1], dim=1))  # 改
            loss_map = self.criterion(y[:, :2], veci)
            loss_map /= 2.
            loss_cellprob = self.criterion2(y[:, 2], cellprob)  # 细胞概率上的loss, y[:, -1] or y[:, 2:]

            loss_class = 0
            if self.multi_class:
                # loss_class = self.criterion3(y[:, 3:], lbl[:, 3].float().to(device).long())

                true_class = F.one_hot(lbl[:, 3].float().to(device).long())  # 使用自定义的loss，这里onehot肯定没有问题,
                pred_class = torch.transpose(y[:, 3:], 1, 3)
                loss_class = dice_loss(pred_class, true_class) * 0.1
                # loss_class = xentropy_loss(pred_class, true_class)

            loss_contrast = 0
            if styles is not None:
                style, style_pos, style_neg = styles
                # target_pos=torch.ones(style.shape[0]).to(device)  # 余弦相似性效果确实不好
                # target_neg=(torch.ones(style.shape[0])).to(device)
                # loss_pos = nn.functional.cosine_embedding_loss(style, style_pos, target=target_pos)
                # loss_neg = nn.functional.cosine_embedding_loss(style, style_neg, target=target_neg)
                loss_pos = self.criterion(style, style_pos)
                loss_neg = self.criterion(style, style_neg)
                # print('>>>', loss_pos, loss_neg, self.alpha)

                loss_contrast = torch.pow((loss_pos), 2) + torch.pow((loss_neg - 1), 2)

            print('>>>loss', loss_map, loss_cellprob, loss_contrast, loss_class)

                # loss3 = torch.pow((loss_pos), 2)/torch.pow((loss_neg), 2)
                # loss3 = torch.pow((loss_pos-loss_neg), 2)
                # loss3 = torch.abs(loss_pos) - torch.abs(loss_neg - 1)
            loss = loss_map * 1. + loss_cellprob * 1. + loss_class * 1. + loss_contrast * 1.  # 系数影响的是对应loss更新的速度

        elif 'classic' in self.task_mode :
            nclasses = int(self.task_mode.split('-')[-1])
            if lbl.shape[1] > 1 and nclasses > 2:
                boundary = lbl[:, 1] <= 4
                lbl = lbl[:, 0]  # lbl是一张图上的像素级分类
                lbl[boundary] *= 2
            else:
                lbl = lbl[:, 0]
            lbl = lbl.float().to(device)
            loss = 8 * 1. / nclasses * self.criterion(y, lbl.long())
        return loss


class sCSnet(nn.Module):
    def __init__(self, nbase, sz, multi_class=False,
                 ndecoder=False, ntasker_seg=False,
                 residual_on=True, style_on=True, concatenation=False, mkldnn=False, update_step=1,
                 attn_on=False, dense_on=False, phase='eval',
                 device=None, net_avg=False, task_mode='cellpose'):
        """
        默认是训练模式，此时网络架构和正常的Cellpose无差异
        Args:
            nbase:
            nout:
            sz:
            residual_on:
            style_on:
            concatenation:
            mkldnn:
            mode:
        """
        super(sCSnet, self).__init__()
        self.nbase = nbase  # [2, 32, 64, 128, 256] 2是指图像有两个通道，舍弃第三维通道，减少计算量
        self.sz = sz
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.mkldnn = False
        self.net_avg= net_avg
        self.device = device
        self.phase = phase
        self.task_mode = task_mode
        self.update_step = update_step

        if 'classic' in task_mode:  # classic模式下loss计算只能一次，不然会报错
            self.update_step = 1

        self.last_result = -np.inf

        # TODO: 未适配CPU模式下mkldnn加速
        self.extractor = Extractor(nbase, sz, ndecoder=ndecoder, residual_on=residual_on, mkldnn=self.mkldnn, style_on=style_on,
             concatenation=concatenation, attn_on=attn_on, dense_on=dense_on)
        if ntasker_seg ==2 or ndecoder >= 2:
            self.tasker0 = Tasker(nbase, 2, mkldnn=self.mkldnn, task_mode=task_mode)  # map
            self.tasker1 = Tasker(nbase, 1, mkldnn=self.mkldnn, task_mode=task_mode)  # cellprob
        else:
            self.tasker_seg = Tasker(nbase, 3, mkldnn=self.mkldnn, task_mode=task_mode)  # map+cellprob

        if multi_class:
            self.tasker_class = Tasker(nbase, multi_class, mkldnn=self.mkldnn, task_mode=task_mode)

        self.loss_fn = LossFn(task_mode=task_mode, multi_class=multi_class)


    def forward(self, inp):
        """The function to forward the model.
        Args:
          inp: input images.
        Returns:
          the outputs of transfer model.
        """
        if self.phase == 'shot_train':
            data_shot, label_shot = inp
            return self.shot_train_forward(data_shot, label_shot)
        else:
            return self.eval_forward(inp)


    def shot_train_forward(self, data_shot, label_shot):
        save_path = r'G:\Python\9-Project\1-flurSeg\sCellSeg\output\models\preeval'

        transfer_path = os.path.join(save_path, 'transfer_cellpose')
        for _ in range(0, self.update_step):
            embedding_shot, style = self.extractor(data_shot)
            if self.ndecoder>=2:
                logits = torch.cat((self.tasker0(embedding_shot[0]), self.tasker1(embedding_shot[1])), dim=1)
            else:
                if self.ntasker_seg==2:
                    logits = torch.cat((self.tasker0(embedding_shot[0]), self.tasker1(embedding_shot[0])), dim=1)
                else:
                    logits= self.tasker_seg(embedding_shot[0])  # 预测类别
            if self.multi_class:
                logits_class = self.tasker_class(embedding_shot[-1])  # 预测map
                logits = torch.cat((logits, logits_class), dim=1)

            styles = None
            if self.contrast_on:
                pos_imgs, neg_imgs = self.pair_gen.get_pair()

                embedding_pos, style_pos = self.extractor(pos_imgs)
                if self.ndecoder >= 2:
                    logits_pos = torch.cat((self.tasker0(embedding_pos[0]), self.tasker1(embedding_pos[1])), dim=1)
                else:
                    if self.ntasker_seg == 2:
                        logits_pos = torch.cat((self.tasker0(embedding_pos[0]), self.tasker1(embedding_pos[0])), dim=1)
                    else:
                        logits_pos = self.tasker_seg(embedding_pos[0])  # 预测类别
                if self.multi_class:
                    logits_class_pos = self.tasker_class(embedding_pos[-1])  # 预测map

                # embedding_pos = embedding_pos.detach()
                # style_pos = style_pos.detach()

                embedding_neg, style_neg = self.extractor(neg_imgs)
                if self.ndecoder >= 2:
                    logits_neg = torch.cat((self.tasker0(embedding_neg[0]), self.tasker1(embedding_neg[1])), dim=1)
                else:
                    if self.ntasker_seg == 2:
                        logits_neg = torch.cat((self.tasker0(embedding_neg[0]), self.tasker1(embedding_neg[0])), dim=1)
                    else:
                        logits_neg = self.tasker_seg(embedding_neg[0])  # 预测类别
                if self.multi_class:
                    logits_class_neg = self.tasker_class(embedding_neg[-1])  # 预测map

                styles = (style, style_pos, style_neg)

                # style_ = style/((style ** 2).sum() ** 0.5)
                # style_pos_ = style_pos / ((style_pos ** 2).sum() ** 0.5)
                # style_neg_ = style_neg / ((style_neg ** 2).sum() ** 0.5)
                # styles = (style_, style_pos_, style_neg_)

            loss = self.loss_fn(label_shot, logits, device=self.device, styles=styles)
            if _ == 0:
                print("loss_first", loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        model_dict = self.state_dict()

        now_result = -loss.item()  # loss_last模式( loss.item() ) or loss_first模式( loss_first )
        if self.last_result == 0:
            print('\033[1;34m>>>>get first weights\033[0m')
            self.last_result = now_result
            self.save_model(transfer_path)
        elif now_result > self.last_result:
            print('\033[1;34m>>>>get better weights\033[0m')
            self.last_result = now_result
            self.save_model(transfer_path)


    def eval_forward(self, data):
        # transfer_path = r'G:\Python\9-Project\1-flurSeg\MTL-cellpose\output\models\mtl_cellpose'
        # self.load_model(transfer_path)
        # model_dict = self.state_dict()

        embedding, style = self.extractor(data)
        if self.ndecoder >= 2:
            logits = torch.cat((self.tasker0(embedding[0]), self.tasker1(embedding[1])), dim=1)
        else:
            if self.ntasker_seg == 2:
                logits = torch.cat((self.tasker0(embedding[0]), self.tasker1(embedding[0])), dim=1)
            else:
                logits = self.tasker_seg(embedding[0])  # 预测类别
        if self.multi_class:
            logits_class = self.tasker_class(embedding[-1])  # 预测map
            logits_class = torch.argmax(logits_class, dim=1, keepdim=True)
            logits = torch.cat((logits, logits_class), dim=1)

        return logits, style


    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, cpu=False, last_conv_on=True):
        if not cpu:
            premodel_dict_0 = torch.load(filename)
        else:
            # self.__init__(self.nbase,self.nout,self.sz,self.residual_on,self.style_on,self.concatenation,self.mkldnn)
            premodel_dict_0 = torch.load(filename, map_location=torch.device('cpu'))

        # 参数适配
        premodel_dict_1 = copy.deepcopy(premodel_dict_0)  # 拷贝一份用于迭代
        is_cellpose_provide = False
        for k, v in premodel_dict_1.items():
            if 'output' in k:
                is_cellpose_provide = True
                break

        if is_cellpose_provide:
            premodel_dict = {'extractor.' + k: v for k, v in premodel_dict_1.items() if 'downsample' in k}  # 改名
            for k, v in premodel_dict_1.items():  # 这里针对的是：读取的cellpose模型，mtl模式肯定能够读进去，也确实能开关cellpose最后一层
                k_new = None

                if 'upsample' in k:
                    k_new = k.replace('upsample', 'extractor.upsample.upsample_branch_0')

                if 'output.0' in k:
                    k_new = k.replace('output.0', 'tasker_seg.base_bn')
                if last_conv_on and ('output.2' in k):
                        k_new = k.replace('output.2', 'tasker_seg.base_conv')

                if k_new is not None:
                    premodel_dict[k_new] = premodel_dict_0.pop(k)
                    for i in range(1, self.ndecoder):
                        k_newi = k.replace('upsample', 'extractor.upsample.upsample_branch_%d' % (i))
                        premodel_dict[k_newi] = premodel_dict[k_new]

        else:
            premodel_dict = copy.deepcopy(premodel_dict_0)

        if (not last_conv_on):  # 对mtl模式进行后处理,控制最后conv的读取
            for k, v in premodel_dict_1.items():
                if 'base_conv' in k:
                    premodel_dict.pop(k)

        model_dict = self.state_dict()
        model_dict.update(premodel_dict)  # 更新premodel_dict中有的参数
        self.load_state_dict(model_dict, strict=False)  # 加载模型参数，此时会进行形状验证
