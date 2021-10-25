import os
import numpy as np

from scellseg import models, io, metrics
from scellseg.contrast_learning.dataset import DatasetPairEval
from scellseg.dataset import DatasetShot, DatasetQuery
from torch.utils.data import DataLoader
from scellseg.utils import set_manual_seed
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

t0 = time.time()
use_GPU = True
num_batch = 8
channel = [2, 1]
flow_threshold = 0.4
cellprob_threshold = 0.5
min_size = ((30. // 2) ** 2) * np.pi * 0.05
# min_size = 15
dataset_dir = r'G:\Python\9-Project\1-flurSeg\scellseg\input\meta_eval\BBBC010_elegans'
# task_mode： cellpose, hover, classic-3, classic-2
task_mode = 'classic-3' # 用1000
task_mode = 'cellpose' # 用400
contrast_on = 1

set_manual_seed(5)

active_ind = [9, 5, 2, 4]
train_epoch = 100
shotset = DatasetShot(eval_dir=dataset_dir, class_name=None, image_filter='_img', mask_filter='_masks', channels=channel,
                      train_num= train_epoch * num_batch, task_mode=task_mode, active_ind=active_ind, rescale=True)
shot_gen = DataLoader(dataset=shotset, batch_size=num_batch, num_workers=0, pin_memory=True)

queryset = DatasetQuery(dataset_dir, class_name=None, image_filter='_img', mask_filter='_masks')
query_image_names = queryset.query_image_names
query_label_names = queryset.query_label_names

diameter = shotset.md
print('>>>> mean diameter of this style,', round(diameter, 3))
pretrained_model = r'C:\Users\admin\.scellseg\models\cytotorch_0'
# pretrained_model = r'G:\Python\9-Project\1-flurSeg\scellseg\output\models\preeval\transfer_cellpose'  # 用于增量学习测试

# model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021071309_5618'  # attn+多层次style
# model_name = r'499_cellpose_residual_on_style_off_concatenation_off_cellpose_2021071313_3307'  # style_off

model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021070913_3905'  # cellpose自训练的
# model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021071722_3201'  # cellpose自训练的-2
model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021100718_1426'

# model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021071216_1451'  # 多层次style
# # model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021071720_3731'# 多层次style-2

# model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021072123_5236'  # all,要开attn+dense
# model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021072120_1647'  # all2

# model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021072323_4252'  # all+多层次style
# model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021070920_3743'  # attn自训练, 要开attn
# model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021072223_5440'  # dense
# model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021080720_3401'  # all+多层次style-3

pretrained_model = os.path.join(r'G:\Python\9-Project\1-flurSeg\scellseg\output\models\scellstyle_train', model_name)

# model_name = r'399_cellpose_residual_on_style_on_concatenation_off_retrain_2021072516_3607'
# pretrained_model = os.path.join(r'G:\Python\9-Project\1-flurSeg\scellseg\output\models\retrain\models', model_name)

lr = {'downsample': 0.001, 'upsample': 0.001, 'tasker': 0.001, 'alpha': 0.005,
      'tasker0': 0.0005, 'tasker1': 0.0005, 'tasker_seg': 0.0005}
lr_schedule_gamma = {'downsample': 0.5, 'upsample': 0.5, 'tasker': 0.5, 'alpha': 1.,
                      'tasker0': 1., 'tasker1': 1., 'tasker_seg': 1.}
step_size = int(train_epoch * 0.25)
model = models.sCellSeg(pretrained_model=pretrained_model, gpu=use_GPU, update_step=1, nclasses=3,
                        task_mode=task_mode, net_avg=False,
                        attn_on=False, dense_on=False, style_on=True,
                        use_branch=False, ntasker_seg=1,
                        last_conv_on=True)
model_dict = model.net.state_dict()

model.net.contrast_on = contrast_on
if contrast_on:
    model.net.pair_gen = DatasetPairEval(positive_dir=dataset_dir, gpu=True, rescale=True)

# count = 0
# for name, param in model.net.named_parameters():
#     if param.requires_grad:
#         print(name, ':', param.size())
#         count += 1
# print(count)

shot_pairs = (shotset.shot_img_names, shotset.shot_mask_names, True)  # 第三个参数为是否根据shot重新计算
masks, flows, styles = model.query_eval(shot_gen=None, use_transfer_model=False,
                               query_image_names=query_image_names, channel=channel, diameter=diameter,
                               resample=False, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                               min_size=min_size, tile_overlap=0.5, eval_batch_size=16, tile=True,
                               lr=lr, lr_schedule_gamma=lr_schedule_gamma, step_size=step_size,
                               postproc_mode='cellpose',
                               shot_pairs=shot_pairs)

t1 = time.time()
print('\033[1;32m>>>> Total Time:\033[0m', t1-t0, 's')

show_single = False
query_labels = [np.array(io.imread(query_label_name)) for query_label_name in query_label_names]
image_names = [query_image_name.split('query\\')[-1] for query_image_name in query_image_names]

query_labels = metrics.refine_masks(query_labels)  # 防止标记mask中的索引有跳值
# 计算AP
thresholds = np.arange(0.5, 1.05, 0.05)
ap = metrics.average_precision(query_labels, masks, threshold=thresholds)[0]
if show_single:
    ap_dict = dict(zip(image_names, ap))
    print('\033[1;34m>>>> scellseg - AP\033[0m')
    for k,v in ap_dict.items():
        print(k, v)
print('\033[1;34m>>>> AP Mean:\033[0m', [round(ap_i, 3) for ap_i in ap.mean(axis=0)])
# 计算AJI
aji = metrics.aggregated_jaccard_index(query_labels, masks)
if show_single:
    aji_dict = dict(zip(image_names, aji))
    print('\033[1;34m>>>> scellseg - AJI\033[0m')
    for k,v in aji_dict.items():
        print(k, v)
print('\033[1;34m>>>> AJI Mean:\033[0m', aji.mean())
# 计算BP
# scales = np.arange(0.025, 0.275, 0.025)
# bp = metrics.boundary_scores(query_labels, masks, scales)[0]
# if show_single:
#     bp_dict = dict(zip(image_names, np.transpose(bp, (1,0))))
#     print('\033[1;34m>>>> scellseg - BP\033[0m')
#     for k,v in bp_dict.items():
#         print(k, v)
# print('\033[1;34m>>>> BP Mean:\033[0m', [round(bp_i, 3) for bp_i in bp.mean(axis=1)])
# 计算PQ
dq, sq, pq = metrics.panoptic_quality(query_labels, masks, thresholds)
if show_single:
    print('\033[1;34m>>>> scellseg - DQ@0.5, SQ@0.5, PQ@0.5\033[0m')
    for i, img_name in enumerate(image_names):
        print(img_name, dq[i][0], sq[i][0], pq[i][0])
print('\033[1;34m>>>> DQ Mean:\033[0m', [round(dq_i, 3) for dq_i in dq.mean(axis=0)])
print('\033[1;34m>>>> SQ Mean:\033[0m', [round(sq_i, 3) for sq_i in sq.mean(axis=0)])
print('\033[1;34m>>>> PQ Mean:\033[0m', [round(pq_i, 3) for pq_i in pq.mean(axis=0)])

# 输出output图片和预测的mask
diams = np.ones(len(query_image_names))*diameter
imgs = [io.imread(query_image_name) for query_image_name in query_image_names]
io.masks_flows_to_seg(imgs, masks, flows, diams, query_image_names, [channel for i in range(len(query_image_names))])
io.save_to_png(imgs, masks, flows, query_image_names, labels=query_labels, aps=None, task_mode=task_mode)
