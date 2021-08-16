import os

import cv2
import numpy as np

from scellseg import models, io, metrics
from scellseg.contrast_learning.dataset import DatasetPairEval
from scellseg.dataset import DatasetShot, DatasetQuery
from torch.utils.data import DataLoader
from scellseg.utils import set_manual_seed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

use_GPU = True
num_batch = 16
channel = [0, 0]
flow_threshold = 0.4
cellprob_threshold = 0.5
min_size = ((30. // 2) ** 2) * np.pi * 0.05
# min_size = 15
dataset_dir = r'G:\Python\9-Project\1-flurSeg\scellseg\input\meta_eval\CoNSeP'
# task_mode： cellpose, hover, classic-3, classic-2
task_mode = 'classic-3' # 用1000
task_mode = 'cellpose' # 用400
postproc_mode = 'cellpose'  # cellpose or watershed

set_manual_seed(5)

shotset = DatasetShot(eval_dir=dataset_dir, class_name=None, channels=channel,
                      image_filter='_img', mask_filter='_masks', class_filter='_classes',
                      train_num= 400 * num_batch, task_mode=task_mode, multi_class=True)
shot_gen = DataLoader(dataset=shotset, batch_size=num_batch, num_workers=0, pin_memory=True)

queryset = DatasetQuery(dataset_dir, class_name=None, image_filter='_img', mask_filter='_masks')
query_image_names = queryset.query_image_names
query_label_names = queryset.query_label_names
query_classes_names = queryset.query_classes_names

diameter = shotset.md
print('>>>> mean diameter of this style,', round(diameter, 3))
print('>>>> total classes,', round(shotset.classes_num))
pretrained_model = r'C:\Users\admin\.scellseg\models\cytotorch_0'
# pretrained_model = r'G:\Python\9-Project\1-flurSeg\scellseg\output\models\preeval\transfer_cellpose'  # 用于增量学习测试

# model_name = r'499_cellpose_residual_on_style_off_concatenation_off_cellpose_2021060422_2737'  # attn
# model_name = r'499_cellpose_residual_on_style_off_concatenation_off_cellpose_2021060509_1111'  # dense
# model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021060516_5429'  # style
# model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021060518_4208'  # attn+style
# model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021060520_4337'  # dense+style
# model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021061122_5647'  # attn+style+2branch
# pretrained_model = os.path.join(r'G:\Python\9-Project\1-flurSeg\scellseg\output\models\scellseg_train', model_name)\
lr = {'downsample': 0.0005, 'upsample': 0.0005,
      'tasker': 0.005, 'tasker0': 0.0005, 'tasker1': 0.0005,
      'tasker_seg': 0.0005, 'tasker_class': 0.05}
lr_schedule_gamma = {'downsample': 1., 'upsample': 1.,
                     'tasker': 0.1, 'tasker0': 0.1, 'tasker1': 0.1,
                     'tasker_seg': 1., 'tasker_class': 0.5}
step_size = 400
model = models.sCellSeg(pretrained_model=pretrained_model, gpu=use_GPU, update_step=1, nclasses=3,
                        task_mode=task_mode, net_avg=False,
                        attn_on=False, dense_on=False, style_on=True,
                        multi_class=shotset.classes_num,
                        use_branch=False, ndecoder=1, ntasker_seg=1,
                        last_conv_on=True)
model_dict = model.net.state_dict()

contrast_on = 0
model.net.contrast_on = contrast_on
if contrast_on:
    model.net.pair_gen = DatasetPairEval(positive_dir=dataset_dir, gpu=True, n_sample1class=num_batch)

shot_pairs = (shotset.shot_img_names, shotset.shot_mask_names, True)  # 第三个参数为是否根据shot重新计算
maskclass, flows, styles = model.query_eval(shot_gen=shot_gen,
                               query_image_names=query_image_names, channel=channel, diameter=diameter,
                               resample=False, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                               min_size=min_size, tile_overlap=0.5, eval_batch_size=16, tile=True,
                               lr=lr, lr_schedule_gamma=lr_schedule_gamma, step_size=step_size,
                               postproc_mode=postproc_mode,
                               shot_pairs=shot_pairs)
if len(maskclass) == 2:
    masks, classes = maskclass
else:
    masks = maskclass

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
# 计算Type
query_classes = [np.array(io.imread(query_classes_name)) for query_classes_name in query_classes_names]
results_list = metrics.type_stat(query_classes, classes, query_labels, masks)
for i, query_classes_name in enumerate(query_classes_names):
    cp_classes_name = query_classes_name.replace('_classes', '_cp_classes')
    cv2.imwrite(cp_classes_name, classes[i].astype(np.uint16))
print('\033[1;34m>>>> Type Stat:\033[0m', results_list)


diams = np.ones(len(query_image_names))*diameter
imgs = [io.imread(query_image_name) for query_image_name in query_image_names]
io.masks_flows_to_seg(imgs, masks, flows, diams, query_image_names, [channel for i in range(len(query_image_names))])
io.save_to_png(imgs, masks, flows, query_image_names, labels=query_labels, aps=ap, task_mode=task_mode)

# 绘制ap曲线
# mpl.rcParams['figure.dpi'] = 300
# rc('font', **{'size': 6})
# fig = plt.figure(figsize=(3, 1.5), facecolor='w', frameon=True, dpi=300)
# colors = [c for c in plt.get_cmap('Dark2').colors]
# ax = fig.add_axes([0.1+.22, 0.1, 0.17, 0.25])  # ax1 = fig.add_axes([left, bottom, width, height]) 绘制的距离边框的百分比
# ax.plot(thresholds, np.mean(ap, axis=0), color=colors[0])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_ylim([0, 1])
# ax.set_xlim([0.5, 1])
# plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.], size=4)
# # ax.set_xlabel('IoU matching threshold')
# # titles = ['specialist model / \n specialized data']
# # ax.text(0, 1.05, titles[0])
# # ax.text(-.25, 1.15, string.ascii_lowercase[0], fontsize = 10, transform=ax.transAxes)

# save_root = r"G:\Python\9-Project\1-flurSeg\scellseg\output"
# os.makedirs(os.path.join(save_root, 'figs'), exist_ok=True)
# fig.savefig(os.path.join(save_root, 'figs/test.pdf'), bbox_inches='tight')
