import os
import numpy as np
import matplotlib as mpl
from scellseg import io, metrics, models
from scellseg.io import get_image_files, get_label_files, imread
from scellseg.utils import diameters
from scellseg.dataset import DatasetQuery
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pretrained_model = r'C:\Users\admin\.scellseg\models\cytotorch_0'

# model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021070913_3905'  # cellpose自训练的
# model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021071700_1949'  # cellpose自训练的-2
# model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021071216_1451'  # 多层次style
# model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021071622_5232'# 多层次style-2

# pretrained_model = os.path.join(r'G:\Python\9-Project\1-flurSeg\scellseg\output\models\scellstyle_train', model_name)

model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021100718_1426'

# model_name= r'499_cellpose_residual_on_style_on_concatenation_off_retrain_2021073121_4151'
# pretrained_model = os.path.join(r'G:\Python\9-Project\1-flurSeg\scellseg\output\models\retrain\models', model_name)


channel = [2, 1]
mpl.rcParams['figure.dpi'] = 300
dataset_name = 'nc_nuclei'
dataset_dir = r'G:\Python\9-Project\1-flurSeg\scellseg\input\meta_eval\{}'.format(dataset_name)
query_dir = os.path.join(dataset_dir, 'query')

# 根据label计算diam
shot_dir = os.path.join(dataset_dir, 'shot')
shot_names = get_image_files(shot_dir, '_masks', '_img')
shot_names = get_image_files(shot_dir, '_masks', '_img')
shot_label_names, _ = get_label_files(shot_names, "_masks", imf="_img")
shot_labels = []
mds = []
for shot_label_name in shot_label_names:
    shot_label = imread(shot_label_name)
    md = diameters(shot_label)[0]
    if md < 5.0:
        md = 5.0
    mds.append(md)
    shot_labels.append(shot_label)
diam = np.array(mds).mean()
print('>>>> mean diam of this style,', round(diam, 3))

use_GPU = True

queryset = DatasetQuery(dataset_dir, class_name=None, image_filter='_img', mask_filter='_masks')
query_image_names = queryset.query_image_names
query_label_names = queryset.query_label_names

imgs = [io.imread(image_name) for image_name in query_image_names]
nimg = len(imgs)
channels = [channel for i in range(len(query_image_names))]
min_size = ((30. // 2) ** 2) * np.pi * 0.05

# model = models.Cellpose(gpu=use_GPU, model_type='cyto', net_avg=True)  # nuclei/cyto
model = models.CellposeModel(gpu=use_GPU, style_on=True, pretrained_model=pretrained_model)  # nuclei/cyto
masks, flows, styles = model.eval(imgs, channels=channels, net_avg=False, tile=True, diameter=diam,
                                     flow_threshold=0.4, cellprob_threshold=0.5, tile_overlap=0.5, min_size=min_size)

# model = models_transfer.CellposeModel(gpu=use_GPU, pretrained_model=pretrained_model, last_conv_on=True, attn=False)
# model_dict = model.net.state_dict()  # 对原模型的eval没有影响，参数可以正常读取
# masks, flows, styles = model.eval(imgs, channels=channels, net_avg=False, tile=True, diameter=diam,
#                                   flow_threshold=0.4, cellprob_threshold=0.5, tile_overlap=0.5,
#                                  min_size=min_size)

show_single = False
query_labels = [np.array(io.imread(query_label_name)) for query_label_name in query_label_names]
image_names = [query_image_name.split('query\\')[-1] for query_image_name in query_image_names]

query_labels = metrics.refine_masks(query_labels)  # 防止标记mask中的索引有跳值

# 计算AP
thresholds = np.arange(0.5, 1.05, 0.05)
ap, _, _, _, pred_ious = metrics.average_precision(query_labels, masks, threshold=thresholds, return_pred_iou=True)
# io.mask_to_cocojson(r'G:\Python\9-Project\1-flurSeg\scellseg\input\meta_eval', dataset_name, pred_masks=masks, pred_ious=pred_ious)

if show_single:
    ap_dict = dict(zip(image_names, ap))
    print('\033[1;34m>>>> cellpose - AP\033[0m')
    for k,v in ap_dict.items():
        print(k, v)
print('\033[1;34m>>>> AP@0.5 Mean:\033[0m', [round(ap_i, 3) for ap_i in ap.mean(axis=0)])
# 计算AJI
aji = metrics.aggregated_jaccard_index(query_labels, masks)
if show_single:
    aji_dict = dict(zip(image_names, aji))
    print('\033[1;34m>>>> cellpose - AJI\033[0m')
    for k,v in aji_dict.items():
        print(k, v)
print('\033[1;34m>>>> AJI Mean:\033[0m', aji.mean())
# 计算BP
# scales = np.arange(0.025, 0.275, 0.025)
# bp = metrics.boundary_scores(query_labels, masks, scales)[0]
# if show_single:
#     bp_dict = dict(zip(image_names, np.transpose(bp, (1,0))))
#     print('\033[1;34m>>>> cellpose - BP\033[0m')
#     for k,v in bp_dict.items():
#         print(k, v)
# print('\033[1;34m>>>> BP@0.025 Mean:\033[0m', [round(bp_i, 3) for bp_i in bp.mean(axis=1)])
# 计算PQ
dq, sq, pq = metrics.panoptic_quality(query_labels, masks, thresholds)
if show_single:
    print('\033[1;34m>>>> cellpose - DQ@0.5, SQ@0.5, PQ@0.5\033[0m')
    for i, img_name in enumerate(image_names):
        print(img_name, dq[i][0], sq[i][0], pq[i][0])
print('\033[1;34m>>>> DQ@0.5 Mean:\033[0m', [round(dq_i, 3) for dq_i in dq.mean(axis=0)])
print('\033[1;34m>>>> SQ@0.5 Mean:\033[0m', [round(sq_i, 3) for sq_i in sq.mean(axis=0)])
print('\033[1;34m>>>> PQ@0.5 Mean:\033[0m', [round(pq_i, 3) for pq_i in pq.mean(axis=0)])


# diams = [diam]
# io.masks_flows_to_seg(imgs, masks, flows, diams, query_image_names, channels)
# io.save_to_png(imgs, masks, flows, query_image_names, labels=query_labels, aps=ap)
