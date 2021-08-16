import os, cv2
import numpy as np
import matplotlib as mpl
from scellseg import io, metrics, models
from scellseg.io import get_image_files, get_label_files, imread
from scellseg.utils import diameters
from scellseg.dataset import DatasetQuery
from skimage.measure import regionprops
from math import floor, ceil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pretrained_model = r'C:\Users\admin\.scellseg\models\cytotorch_0'

channel = [2, 1]
mpl.rcParams['figure.dpi'] = 300
dataset_dir = r'G:\Python\9-Project\1-flurSeg\scellseg\input\meta_eval\cxc'
query_dir = os.path.join(dataset_dir, 'query')

# 根据label计算diam
shot_dir = os.path.join(dataset_dir, 'shot')
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

model = models.Cellpose(gpu=use_GPU, model_type='cyto', net_avg=True)  # nuclei/cyto
masks, flows, styles = model.eval(imgs, channels=channels, net_avg=False, tile=True, diameter=diam,
                                     flow_threshold=0.4, cellprob_threshold=0.5, tile_overlap=0.5, min_size=min_size)

sta = 256
save_dir = r'G:\Python\9-Project\1-flurSeg\scellseg\output\single'

for n in range(len(masks)):
    maskn = masks[n]
    props = regionprops(maskn)
    i_max = maskn.max() + 1
    for i in range(1, i_max):
        maskn_ = np.zeros_like(maskn)
        maskn_[maskn == i] = 1
        bbox = props[i - 1]['bbox']
        if imgs[n].ndim == 3:
            imgn_single = imgs[n][bbox[0]:bbox[2], bbox[1]:bbox[3]] * maskn_[bbox[0]:bbox[2], bbox[1]:bbox[3], np.newaxis]
        else:
            imgn_single = imgs[n][bbox[0]:bbox[2], bbox[1]:bbox[3]] * maskn_[bbox[0]:bbox[2], bbox[1]:bbox[3]]

        shape = imgn_single.shape
        shape_x = shape[0]
        shape_y = shape[1]
        add_x = sta - shape_x
        add_y = sta - shape_y
        add_x_l = int(floor(add_x / 2))
        add_x_r = int(ceil(add_x / 2))
        add_y_l = int(floor(add_y / 2))
        add_y_r = int(ceil(add_y / 2))
        if add_x > 0 and add_y > 0:
            if imgn_single.ndim == 3:
                imgn_single = np.pad(imgn_single, ((add_x_l, add_x_r), (add_y_l, add_y_r), (0, 0)), 'constant',
                                        constant_values=(0, 0))
            else:
                imgn_single = np.pad(imgn_single, ((add_x_l, add_x_r), (add_y_l, add_y_r)), 'constant',
                                        constant_values=(0, 0))

            save_name = os.path.join(save_dir, query_image_names[n].split('query')[-1].split('.')[0][1:]+'_'+str(i)+'.tif')
            cv2.imwrite(save_name, imgn_single)
