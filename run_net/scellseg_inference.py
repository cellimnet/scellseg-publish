import os, time, json
import numpy as np

from scellseg import models, io, metrics
from scellseg.contrast_learning.dataset import DatasetPairEval
from scellseg.dataset import DatasetShot, DatasetQuery
from torch.utils.data import DataLoader
from scellseg.utils import set_manual_seed, make_folder, process_different_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

use_GPU = True
num_batch = 8
channel = [2, 1]
flow_threshold = 0.4
cellprob_threshold = 0.5
min_size = ((30. // 2) ** 2) * np.pi * 0.05

project_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")
output_path = os.path.join(project_path, 'output')
make_folder(output_path)
output_excel_path =  os.path.join(output_path, 'excels')
make_folder(output_excel_path)

dataset_dir_root = r'G:\Python\9-Project\1-cellseg\scellseg\input\eval'
dataset_names = ['BBBC010_elegans']  # 'BBBC010_elegans', 'mito', 'bv2'

model_name = 'scellseg'  # unet2, unet3, hover, cellpose, scellseg, scellseg_sneuro, scellseg_sfluor, scellseg_scell, scellseg_smicro
net_avg = False
finetune_model = r'G:\Python\9-Project\1-cellseg\scellseg-gui\output\fine-tune'  # TODO: you can provide the model file or folder_name of model files

pretrained_model = os.path.join(project_path, 'assets', 'pretrained_models', model_name)
task_mode, postproc_mode, attn_on, dense_on, style_scale_on = process_different_model(model_name)  # task_mode mean different instance representation

for dataset_name in dataset_names:
    dataset_dir = os.path.join(dataset_dir_root, dataset_name)

    shot_data_dir = os.path.join(dataset_dir, 'shot')
    shot_img_names = io.get_image_files(shot_data_dir, '_masks', '_img')

    index_label = ['AP', 'AJI', '']
    output = pd.DataFrame()
    save_name = model_name+'_'+dataset_name
    t0 = time.time()
    set_manual_seed(5)
    shotset = DatasetShot(eval_dir=dataset_dir, class_name=None, image_filter='_img', mask_filter='_masks',
                          channels=channel, task_mode=task_mode, active_ind=None, rescale=True)

    queryset = DatasetQuery(dataset_dir, class_name=None, image_filter='_img', mask_filter='_masks')
    query_image_names = queryset.query_image_names
    query_label_names = queryset.query_label_names

    diameter = shotset.md
    print('>>>> mean diameter of this style,', round(diameter, 3))

    model = models.sCellSeg(pretrained_model=pretrained_model, gpu=use_GPU, nclasses=3,
                            task_mode=task_mode, net_avg=net_avg,
                            attn_on=attn_on, dense_on=dense_on, style_scale_on=style_scale_on,
                            last_conv_on=True, model=None)

    model_dict = model.net.state_dict()
    model.net.save_name = save_name

    shot_pairs = (np.array(shotset.shot_img_names), np.array(shotset.shot_mask_names), True)  # 第三个参数为是否根据shot重新计算
    masks, flows, styles = model.inference(finetune_model=finetune_model, net_avg=net_avg,
                                   query_image_names=query_image_names, channel=channel, diameter=diameter,
                                   resample=False, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                                   min_size=min_size, tile_overlap=0.5, eval_batch_size=16, tile=True,
                                   postproc_mode=postproc_mode, shot_pairs=shot_pairs)
    t1 = time.time()
    print('\033[1;32m>>>> Total Time:\033[0m', t1 - t0, 's')

    show_single = False
    query_labels = [np.array(io.imread(query_label_name)) for query_label_name in query_label_names]
    image_names = [query_image_name.split('query\\')[-1] for query_image_name in query_image_names]

    query_labels = metrics.refine_masks(query_labels)  # prevent

    # compute AP
    thresholds = np.arange(0.5, 1.05, 0.05)
    ap, _, _, _, pred_ious = metrics.average_precision(query_labels, masks, threshold=thresholds, return_pred_iou=True)

    if show_single:
        ap_dict = dict(zip(image_names, ap))
        print('\033[1;34m>>>> scellseg - AP\033[0m')
        for k,v in ap_dict.items():
            print(k, v)
    print('\033[1;34m>>>> AP:\033[0m', [round(ap_i, 3) for ap_i in ap.mean(axis=0)])


    # save AJI
    aji = metrics.aggregated_jaccard_index(query_labels, masks)
    if show_single:
        aji_dict = dict(zip(image_names, aji))
        print('\033[1;34m>>>> scellseg - AJI\033[0m')
        for k,v in aji_dict.items():
            print(k, v)
    print('\033[1;34m>>>> AJI:\033[0m', aji.mean())

    # make dataframe
    output1 = pd.DataFrame([round(ap_i, 3) for ap_i in ap.mean(axis=0)]).T
    output2 = pd.DataFrame(aji).T
    output_blank = pd.DataFrame([' ']).T
    output = pd.concat([output, output1, output2, output_blank], ignore_index=True)

    # save output images
    diams = np.ones(len(query_image_names)) * diameter
    imgs = [io.imread(query_image_name) for query_image_name in query_image_names]
    io.masks_flows_to_seg(imgs, masks, flows, diams, query_image_names, [channel for i in range(len(query_image_names))])
    io.save_to_png(imgs, masks, flows, query_image_names, labels=query_labels, aps=None, task_mode=task_mode)

output.index = index_label * 1
output.to_excel(os.path.join(output_excel_path, save_name+'.xlsx'))
