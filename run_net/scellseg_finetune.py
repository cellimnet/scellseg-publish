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

project_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")
output_path = os.path.join(project_path, 'output')
make_folder(output_path)
output_excel_path =  os.path.join(output_path, 'excels')
make_folder(output_excel_path)


train_epoch = 100
dataset_dir_root = r'G:\Python\9-Project\1-flurSeg\scellseg\input\meta_eval'
dataset_names = ['BBBC010_elegans']  # 'BBBC010_elegans', 'mito', 'bv2'
contrast_on = 1
model_name = 'scellseg'  # scellseg, cellpose, hover, unet3, unet2, scellseg_sneuro, scellseg_sfluor, scellseg_scell, scellseg_smicro


sample_times = 1
pretrained_model = os.path.join(project_path, 'assets', 'pretrained_models', model_name)
task_mode, postproc_mode, attn_on, dense_on, style_scale_on = process_different_model(model_name)  # task_mode mean different instance representation

for dataset_name in dataset_names:
    dataset_dir = os.path.join(dataset_dir_root, dataset_name)

    save_name = model_name+'_'+dataset_name
    print(save_name)
    if contrast_on:
        save_name += '-cft'
    t0 = time.time()

    set_manual_seed(5)
    shotset = DatasetShot(eval_dir=dataset_dir, class_name=None, image_filter='_img', mask_filter='_masks', channels=channel,
                          train_num= train_epoch * num_batch, task_mode=task_mode, rescale=True)
    shot_gen = DataLoader(dataset=shotset, batch_size=num_batch, num_workers=0, pin_memory=True)

    diameter = shotset.md
    print('>>>> mean diameter of this style,', round(diameter, 3))

    lr = {'downsample': 0.001, 'upsample': 0.001, 'tasker': 0.001, 'alpha': 0.1}
    lr_schedule_gamma = {'downsample': 0.5, 'upsample': 0.5, 'tasker': 0.5, 'alpha': 0.5}
    step_size = int(train_epoch * 0.25)
    model = models.sCellSeg(pretrained_model=pretrained_model, gpu=use_GPU, update_step=1, nclasses=3,
                            task_mode=task_mode, net_avg=False,
                            attn_on=attn_on, dense_on=dense_on, style_scale_on=style_scale_on,
                            last_conv_on=True, model=None)

    model_dict = model.net.state_dict()
    model.net.pretrained_model = pretrained_model
    model.net.save_name = save_name

    model.net.contrast_on = contrast_on
    if contrast_on:
        model.net.pair_gen = DatasetPairEval(positive_dir=dataset_dir, use_negative_masks=False, gpu=use_GPU, rescale=True)

    model.finetune(shot_gen=shot_gen, lr=lr, lr_schedule_gamma=lr_schedule_gamma, step_size=step_size)
