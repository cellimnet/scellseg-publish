import os
import random

from torch.utils.data import Dataset
from scellseg.io import get_image_files, get_label_files, imread, imsave
from scellseg.transforms import reshape_and_normalize_data, random_rotate_and_resize, reshape_train_test
from scellseg.utils import diameters
import numpy as np
from scellseg.dataset import random_clip, resize_image
import torch


"""
20210331：该版本重构了shot数据的读取
"""

class DatasetPairEval():
    """The class to load the shot data of mdataset"""

    def __init__(self, positive_dir=None, positive_md_mean=None, negative_dir=None, image_filter='_img', mask_filter='_masks',
                 rescale=True, diam_mean=30.0, channels=[2,1], gpu=False):
        """
        negative_dir可以是所有类，或者是只用神经细胞那一类
        """
        self.rescale = rescale
        self.diam_mean = diam_mean
        self.channels = channels
        self.gpu = gpu

        # 获取文件夹名字，读取图片，label用于放缩
        # 正样本图片，选用代迁移数据的未标注数据
        positive_path = os.path.join(positive_dir, 'query')
        positive_img_names = get_image_files(positive_path, mask_filter, image_filter)
        positive_imgs = []
        for positive_img_name in positive_img_names:
            positive_img = imread(positive_img_name)
            positive_imgs.append(positive_img)
        if positive_md_mean is None:
            shot_path = os.path.join(positive_dir, 'shot')
            shot_img_names = get_image_files(shot_path, mask_filter, image_filter)
            shot_mask_names, _ = get_label_files(shot_img_names, mask_filter, imf=image_filter)
            mds = []
            for shot_mask_name in shot_mask_names:
                shot_mask = imread(shot_mask_name)
                md = diameters(shot_mask)[0]
                if md < 5.0:
                    md = 5.0
                mds.append(md)
            positive_md_mean = np.array(mds).mean()  # 对多张图片的中位数进行平均
        rescale = (self.diam_mean / positive_md_mean) * np.ones(len(positive_imgs))
        positive_imgs, _, _ = resize_image(positive_imgs, rsc=rescale)  # 这一步之后image的shape变为(nchan, Ly, Lx)
        self.positive_md_mean = positive_md_mean

        # 负样本图片，选用cellpose提供的数据
        if negative_dir is None:
            negative_dirs = r'G:\Python\9-Project\1-flurSeg\scellseg\input\meta_train\5-train'
            negative_dir = [os.path.join(negative_dirs, classi) for classi in os.listdir(negative_dirs)]
        elif not isinstance(negative_dir, (list, np.ndarray)):
            negative_dir = [negative_dir]
        negative_img_names=[]
        negative_mask_names=[]
        for i in range(len(negative_dir)):
            negative_img_namesi = get_image_files(negative_dir[i], mask_filter, image_filter)
            negative_img_names.append(negative_img_namesi)
            negative_mask_names.append(get_label_files(negative_img_namesi, mask_filter, imf=image_filter)[0])
        negative_imgs = []
        negative_md_means = []
        for nclass in range(len(negative_dir)):
            negative_imgn = []
            for negative_img_name in negative_img_names[nclass]:
                negative_img = imread(negative_img_name)
                negative_imgn.append(negative_img)
            mdsn = []
            for negative_mask_name in negative_mask_names[nclass]:
                negative_mask = imread(negative_mask_name)
                md = diameters(negative_mask)[0]
                if md < 5.0:
                    md = 5.0
                mdsn.append(md)
            negative_md_mean = np.array(mdsn).mean()  # 对多张图片的中位数进行平均
            negative_md_means.append(negative_md_mean)
            rescale = (self.diam_mean / negative_md_mean) * np.ones(len(negative_imgn))
            negative_imgn, _, _ = resize_image(negative_imgn, rsc=rescale)  # 这一步之后image的shape变为(nchan, Ly, Lx)        #
            negative_imgs.append(negative_imgn)

        self.negative_md_means = negative_md_means
        self.positive_imgs = positive_imgs
        self.negative_imgs = negative_imgs

    def get_pair(self, n_sample1class):
        # 1. 随机选择n_sample1class张shot图片
        positive_imgs = []
        negative_imgs = []
        negclass = len(self.negative_imgs)
        negclass_ind = np.random.randint(0, negclass)  # Todo

        for j in range(n_sample1class):
            npos = len(self.positive_imgs)
            pos_ind = np.random.randint(0, npos)
            positive_imgs.append(self.positive_imgs[pos_ind])
            nneg = len(self.negative_imgs[negclass_ind])
            neg_ind = np.random.randint(0, nneg)
            negative_imgs.append(self.negative_imgs[negclass_ind][neg_ind])
        negative_md_mean = self.negative_md_means[negclass_ind]

        # 1.5 在前面reshape
        positive_imgs, _, _ = reshape_and_normalize_data(positive_imgs, channels=self.channels, normalize=True)  # 返回的是list，这里的channels要分析一下
        negative_imgs, _, _ = reshape_and_normalize_data(negative_imgs, channels=self.channels, normalize=True)  # 返回的是list，这里的channels要分析一下

        # 随机切出224的区域, 随机旋转
        scale_range = 0.5 if self.rescale else 1.
        crop_size = random.randrange(156, 291)  # 0.7*224, 1.3*224

        positive_imgs = random_clip(X=positive_imgs, xy=(crop_size, crop_size))[0]
        rsc = np.array([1.]*n_sample1class, np.float32) if self.rescale else np.ones(n_sample1class, np.float32)
        positive_imgs = random_rotate_and_resize(positive_imgs, rescale=rsc, scale_range=scale_range)[0]

        negative_imgs = random_clip(X=negative_imgs, xy=(crop_size, crop_size))[0]
        rsc = np.array([1.]*n_sample1class, np.float32) if self.rescale else np.ones(n_sample1class, np.float32)
        negative_imgs = random_rotate_and_resize(negative_imgs, rescale=rsc, scale_range=scale_range)[0]

        positive_imgs = torch.from_numpy(positive_imgs)
        negative_imgs = torch.from_numpy(negative_imgs)
        if self.gpu:
            positive_imgs = positive_imgs.cuda()
            negative_imgs = negative_imgs.cuda()
        return positive_imgs, negative_imgs

if __name__ == '__main__':
    dataset = DatasetPairEval(positive_dir=r'G:\Python\9-Project\1-flurSeg\scellseg\input\meta_eval\2018DB_nuclei_long',
                              gpu=True)
    a, b = dataset.get_pair()
    print(a.shape, b.shape)
