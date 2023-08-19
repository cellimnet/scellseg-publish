import os, cv2
import numpy as np
from scellseg import io
from scellseg.io import get_image_files, get_label_files
from skimage.measure import regionprops
from math import floor, ceil
from tqdm import trange

data_path = r'G:\Python\9-Project\1-flurSeg\scellseg\input\meta_eval\BBBC010_elegans\query'
sta = 256
save_dir = r'G:\Python\9-Project\1-flurSeg\scellseg\output\single'


image_names = get_image_files(data_path, '_masks', imf='_img')
mask_names, _ = get_label_files(image_names, '_img_cp_masks', imf='_img')
imgs = [io.imread(os.path.join(data_path, image_name)) for image_name in image_names]
masks = [io.imread(os.path.join(data_path, mask_name)) for mask_name in mask_names]

for n in trange(len(masks)):
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

            save_name = os.path.join(save_dir, image_names[n].split('query')[-1].split('.')[0][1:]+'_'+str(i)+'.tif')
            cv2.imwrite(save_name, imgn_single)
