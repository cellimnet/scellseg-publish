import os
import random

from torch.utils.data import Dataset
from scellseg.io import get_image_files, get_label_files, imread, imsave
from scellseg.transforms import reshape_and_normalize_data, random_rotate_and_resize, reshape_train_test
from scellseg.utils import diameters
import numpy as np
from scellseg.dataset import random_clip, resize_image
from scellseg.dynamics import labels_to_flows
import torch


class DatasetPairEval():
    """The class to load the shot data of mdataset"""

    def __init__(self, positive_dir=None, positive_md_mean=None, negative_dir=None, image_filter='_img', mask_filter='_masks',
                 rescale=True, diam_mean=30.0, channels=[2,1], gpu=False, use_negative_masks=False):
        """
        negative_dir可以是所有类，或者是只用神经细胞那一类
        """
        self.rescale = rescale
        self.diam_mean = diam_mean
        self.channels = channels
        self.gpu = gpu
        self.use_negative_masks = use_negative_masks

        # postive data, unlabelled images from a new experiment
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
            positive_md_mean = np.array(mds).mean()
        rescale = (self.diam_mean / positive_md_mean) * np.ones(len(positive_imgs))
        positive_imgs, _, _ = resize_image(positive_imgs, rsc=rescale)  # (nchan, Ly, Lx)
        self.positive_md_mean = positive_md_mean

        # negtive data, subset of cellpose train set
        if negative_dir is None:
            project_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.path.sep + ".")
            negative_dirs = os.path.join(project_path, 'assets', 'contrast_data')
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
        if self.use_negative_masks: negative_masks = []
        negative_md_means = []
        for nclass in range(len(negative_dir)):
            negative_imgn = []
            if self.use_negative_masks: negative_maskn = []
            for negative_img_name in negative_img_names[nclass]:
                negative_img = imread(negative_img_name)
                negative_imgn.append(negative_img)
            mdsn = []
            for negative_mask_name in negative_mask_names[nclass]:
                negative_mask = imread(negative_mask_name)
                if self.use_negative_masks: negative_maskn.append(negative_mask)
                md = diameters(negative_mask)[0]
                if md < 5.0:
                    md = 5.0
                mdsn.append(md)
            negative_md_mean = np.array(mdsn).mean()
            negative_md_means.append(negative_md_mean)
            rescale = (self.diam_mean / negative_md_mean) * np.ones(len(negative_imgn))
            if self.use_negative_masks:
                negative_imgn, negative_maskn, _ = resize_image(negative_imgn, M=negative_maskn, rsc=rescale)  # (nchan, Ly, Lx)
            else:
                negative_imgn, _, _ = resize_image(negative_imgn, rsc=rescale)  # (nchan, Ly, Lx)

            negative_imgs.append(negative_imgn)
            if self.use_negative_masks: negative_masks.append(negative_maskn)

        self.negative_md_means = negative_md_means
        self.positive_imgs = positive_imgs
        self.negative_imgs = negative_imgs
        if self.use_negative_masks: self.negative_masks = negative_masks  #

    def get_pair(self, n_sample1class):
        """
        n_sample1class: int
            the number of positive_imgs and negative_imgs, here we set it equal to the batch of shot data
        """
        # random choose n_sample1class data
        positive_imgs = []
        negative_imgs = []
        if self.use_negative_masks: negative_masks = []

        negclass = len(self.negative_imgs)
        npos = len(self.positive_imgs)
        negclass_ind = np.random.randint(0, negclass)  # Todo

        for j in range(n_sample1class):
            # negclass_ind = np.random.randint(0, negclass)  # Todo
            pos_ind = np.random.randint(0, npos)
            # print(pos_ind)
            positive_imgs.append(self.positive_imgs[pos_ind])
            nneg = len(self.negative_imgs[negclass_ind])
            neg_ind = np.random.randint(0, nneg)
            negative_imgs.append(self.negative_imgs[negclass_ind][neg_ind])
            if self.use_negative_masks: negative_masks.append(self.negative_masks[negclass_ind][neg_ind])
        negative_md_mean = self.negative_md_means[negclass_ind]

        # reshape
        positive_imgs, _, _ = reshape_and_normalize_data(positive_imgs, channels=self.channels, normalize=True)
        negative_imgs, _, _ = reshape_and_normalize_data(negative_imgs, channels=self.channels, normalize=True)

        # random crop, rotate
        scale_range = 0.5 if self.rescale else 1.
        crop_size = random.randrange(156, 291)  # 0.7*224, 1.3*224
        positive_imgs = random_clip(X=positive_imgs, xy=(crop_size, crop_size))[0]
        rsc = np.array([1.]*n_sample1class, np.float32) if self.rescale else np.ones(n_sample1class, np.float32)
        positive_imgs = random_rotate_and_resize(positive_imgs, rescale=rsc, scale_range=scale_range)[0]

        if self.use_negative_masks:
            negative_imgs, negative_masks = random_clip(X=negative_imgs, M=negative_masks, xy=(crop_size, crop_size))[0:2]
            negative_lbls = labels_to_flows(negative_masks)
            negative_lbls = [negative_lbl[1:] for negative_lbl in negative_lbls]  # (3, ...)
            negative_imgs, negative_lbls = random_rotate_and_resize(negative_imgs, negative_lbls, rescale=rsc, scale_range=scale_range)[0:2]
        else:
            negative_imgs = random_clip(X=negative_imgs, xy=(crop_size, crop_size))[0]
            rsc = np.array([1.]*n_sample1class, np.float32) if self.rescale else np.ones(n_sample1class, np.float32)
            negative_imgs = random_rotate_and_resize(negative_imgs, rescale=rsc, scale_range=scale_range)[0]

        positive_imgs = torch.from_numpy(positive_imgs)
        negative_imgs = torch.from_numpy(negative_imgs)
        if self.use_negative_masks: negative_lbls = torch.from_numpy(negative_lbls)
        if self.gpu:
            positive_imgs = positive_imgs.cuda()
            negative_imgs = negative_imgs.cuda()
            if self.use_negative_masks: negative_lbls = negative_lbls.cuda()
        if self.use_negative_masks:
            return positive_imgs, negative_imgs, negative_lbls
        else:
            return positive_imgs, negative_imgs, None
