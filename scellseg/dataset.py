""" Dataloader for scellseg. """
import os.path as osp
import os
import cv2
from torch.utils.data import Dataset
from scellseg.io import get_image_files, get_label_files, imread, imsave
from scellseg.transforms import reshape_and_normalize_data, random_rotate_and_resize, reshape_train_test
from scellseg.utils import diameters, distance_to_boundary
import scellseg.dynamics as dynamics
import numpy as np
import random
import copy as cp
import math
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


class DatasetShot(Dataset):
    """The class to load the shot data of mdataset"""

    def __init__(self, eval_dir=None, class_name=None, shot_folder='shot',
                 image_filter=None, mask_filter='_masks', class_filter='_classes',
                 active_ind=None,
                 shot_names=None,
                 shot_datas=None,
                 train_num=400,
                 rescale=True, diam_mean=30.0, channels=[2,1], task_mode='cellpose',
                 multi_class=False):
        self.rescale = rescale
        self.diam_mean = diam_mean
        self.train_num = train_num
        self.channels = channels
        self.task_mode = task_mode
        self.multi_class = multi_class
        self.active_ind = active_ind

        self.mdataset_dir = eval_dir

        if shot_datas is None:
            if shot_names is None:
                assert eval_dir is not None, print('please input the right path of your task')
                if class_name is not None:
                    eval_dir = osp.join(eval_dir, class_name)
                shot_path = osp.join(eval_dir, shot_folder)
                shot_img_names = get_image_files(shot_path, mask_filter, image_filter)
                shot_mask_names, shot_flow_names = get_label_files(shot_img_names, mask_filter, imf=image_filter)
            else:
                shot_img_names, shot_mask_names = shot_names
            self.shot_img_names = shot_img_names
            self.shot_mask_names = shot_mask_names

            # read images and lbl of shot data
            shot_images = []
            for shot_img_name in shot_img_names:
                shot_image = imread(shot_img_name)
                shot_images.append(shot_image)
            shot_masks = []
            mds = []
            for maski, shot_mask_name in enumerate(shot_mask_names):
                shot_mask = imread(shot_mask_name)
                if active_ind is not None:
                    if isinstance(active_ind, int): active_ind = [active_ind] # for condition: type(active_ind) = int
                    if maski in active_ind:
                        md = diameters(shot_mask)[0]
                        if md < 5.0:
                            md = 5.0
                        mds.append(md)
                else:
                    md = diameters(shot_mask)[0]
                    if md < 5.0:
                        md = 5.0
                    mds.append(md)
                shot_masks.append(shot_mask)

            if self.multi_class:
                classes_num = 0
                shot_classes = []
                shot_class_names = [shot_class_name.replace(mask_filter, class_filter) for shot_class_name in shot_mask_names]
                if not os.path.exists(shot_class_names[0]):
                    raise ValueError('Cannot find class img: must be --png format or not provided correct --class_filter ')
                for shot_class_name in shot_class_names:
                    shot_class = imread(shot_class_name)
                    classes_numi = np.max(shot_class)
                    if classes_num < classes_numi:
                        classes_num = classes_numi
                    shot_classes.append(shot_class)
                self.classes_num=classes_num+1
            else:
                self.classes_num=False

        else:
            mds = []
            shot_images, shot_masks = shot_datas
            for maski, shot_mask in enumerate(shot_masks):
                if active_ind is not None:
                    if isinstance(active_ind, int): active_ind = [active_ind]
                    if maski in active_ind:
                        md = diameters(shot_mask)[0]
                        if md < 5.0:
                            md = 5.0
                        mds.append(md)
                else:
                    md = diameters(shot_mask)[0]
                    if md < 5.0:
                        md = 5.0
                    mds.append(md)

        md_mean = np.array(mds).mean()
        self.md = md_mean
        rescale = (self.diam_mean / md_mean) * np.ones(len(shot_images))

        if self.multi_class:
            shot_images, shot_masks, _, shot_classes = resize_image(shot_images , M=shot_masks, C=shot_classes, rsc=rescale)  # (nchan, Ly, Lx)
            self.shot_classes = shot_classes
        else:
            shot_images, shot_masks, _ = resize_image(shot_images , M=shot_masks, rsc=rescale)  # (nchan, Ly, Lx)

        shot_images, shot_masks, _, _, _ = reshape_train_test(shot_images, shot_masks, test_data=None, test_labels=None,
                            channels=self.channels, normalize=True)

        shot_lbls = None
        if self.task_mode == 'cellpose':
            shot_flows = dynamics.labels_to_flows(shot_masks)
            shot_lbls = [shot_flow[1:] for shot_flow in shot_flows]  # (3, ...)
            self.unet = False
        elif self.task_mode == 'unet3':
            shot_lbl0 = [np.stack((label, label > 0, distance_to_boundary(label)), axis=0).astype(np.float32) for label in shot_masks]
            shot_lbls = [shot_lbl[1:] for shot_lbl in shot_lbl0]  # (2, ...)
            self.unet = True
        elif self.task_mode == 'unet2':
            shot_lbl0 = [np.stack((label, label > 0), axis=0).astype(np.float32) for label in shot_masks]
            shot_lbls = [shot_lbl[1:] for shot_lbl in shot_lbl0]  # (1, ...)
            self.unet = True
        elif self.task_mode == 'hover':
            shot_flows = dynamics.labels_to_hovers(shot_masks)
            shot_lbls = [shot_flow[1:] for shot_flow in shot_flows]  # (3, ...)
            self.unet = False
        self.shot_lbls = shot_lbls

        self.shot_images = shot_images
        self.shot_masks = shot_masks

    def __len__(self):
        return self.train_num

    def __getitem__(self, i):
        # print('testi', i)
        unet=self.unet

        # random select a shot data
        if self.active_ind is not None:
            if isinstance(self.active_ind, list) or isinstance(self.active_ind, np.ndarray):
                choose_img_ind = np.random.choice(np.array(self.active_ind))
            else:
                choose_img_ind = self.active_ind
        else:
            nimg = len(self.shot_img_names)
            choose_img_ind = np.random.randint(0, nimg)
        shot_image=[self.shot_images[choose_img_ind]]
        shot_mask=[self.shot_masks[choose_img_ind]]
        if self.multi_class:
            shot_class=[self.shot_classes[choose_img_ind]]

        shot_lbl=[self.shot_lbls[choose_img_ind]]

        # random rotatet
        if i>=0:
            md = diameters(shot_mask[0])[0]
            scale_range = 0.5 if self.rescale else 1.
            rsc = np.array([md / self.diam_mean], np.float32) if self.rescale else np.ones(1, np.float32)
            if self.multi_class:
                shot_image, shot_lbl, scale, shot_class = random_rotate_and_resize(
                    shot_image, Y=shot_lbl, C=shot_class,
                    rescale=rsc, scale_range=scale_range,
                    unet=unet)
                if unet and shot_lbl.shape[1] > 1 and self.rescale:
                    shot_lbl[:, 1] *= scale[0] ** 2
                shot_lbl = [np.concatenate((shot_lbl[0], shot_class[0]), axis=0)]
            else:
                shot_image, shot_lbl, scale = random_rotate_and_resize(
                    shot_image, Y=shot_lbl,
                    rescale=rsc, scale_range=scale_range,
                    unet=unet)
                if unet and shot_lbl.shape[1] > 1 and self.rescale:
                    shot_lbl[:, 1] *= scale[0] ** 2

        return shot_image[0], shot_lbl[0]


class DatasetQuery(Dataset):
    """The class to load the mdataset"""

    def __init__(self, eval_dir, class_name=None, image_filter='_img', mask_filter='_masks', class_filter='_classes'):
        if class_name is not None:
            eval_dir = osp.join(eval_dir, class_name)

        query_path = osp.join(eval_dir, 'query')
        query_image_names = get_image_files(query_path, mask_filter, image_filter)
        self.query_image_names = query_image_names
        try:
            query_label_names, _ = get_label_files(query_image_names, mask_filter, imf=image_filter)
            self.query_label_names = query_label_names
            query_classes_names = [shot_class_name.replace(mask_filter, class_filter) for shot_class_name in query_label_names]
            self.query_classes_names = query_classes_names
        except:
            print('query images do not have masks')

    def __len__(self):
        return len(self.query_image_names)  # return number of query data

    def __getitem__(self, i):
        inppath = self.query_image_names[i]
        return inppath


def resize_image(X, M=None, F=None, rsc=np.ones(1, np.float32), xy=None, interpolation=cv2.INTER_LINEAR, C=None):
    """
        X: imgs
        M: masks
        F: flows
    """
    nimg = len(X)
    imgs = []
    masks = []
    flows = []
    if C is not None:
        classes = []
    for i in range(nimg):
        if xy is None:
            Ly = int(rsc[i] * X[i].shape[0])
            Lx = int(rsc[i] * X[i].shape[1])
        else:
            Ly, Lx = xy

        if X[i].ndim == 3:
            nchan = X[i].shape[-1]
            imgi = np.zeros((nchan, Ly, Lx), np.float32)
            for m in range(nchan):
                imgi[m] = cv2.resize(X[i][..., m], (Lx, Ly), interpolation=interpolation)
        elif X[i].ndim == 2:
            imgi = cv2.resize(X[i], (Lx, Ly), interpolation=interpolation)
        imgs.append(imgi)

        if F is not None:
            flowi = np.zeros((3, Ly, Lx), np.float32)
            for n in range(3):
                flowi[n] = cv2.resize(F[i][n], (Lx, Ly), interpolation=interpolation)
            flows.append(flowi)

        if M is not None:
            maski = cv2.resize(M[i], (Lx, Ly), interpolation=cv2.INTER_NEAREST)
            masks.append(maski)

        if C is not None:
            classi = cv2.resize(C[i], (Lx, Ly), interpolation=cv2.INTER_NEAREST)
            classes.append(classi)

    if C is not None:
        return imgs, masks, flows, classes
    return imgs, masks, flows


def random_clip(X, M=None, xy = (224,224), C=None):
    """
        # 先clip再pad
        X: imgs
        M: masks
    """
    nimg = len(X)
    imgs = []
    masks = []
    if C is not None:
        classes = []
    for i in range(nimg):
        Ly, Lx = X[i].shape[-2:]
        # print('Ly-Lx', Ly, Lx, X[i].shape)
        dy = Ly - xy[0]
        dx = Lx - xy[1]
        if dy>0:
            yrange = np.arange(0, dy, 2)
            ycenter = (Ly // 2) - (xy[0] // 2)
            yrange = np.append(yrange, ycenter)
            yst = np.random.choice(yrange)
            yed = yst + xy[0]
            ypad1 = 0
            ypad2 = 0
        else:
            yst = 0
            yed = Ly
            ypad1 = int(abs(dy)/2)
            ypad2 = abs(dy) - ypad1
        if dx>0:
            xrange = np.arange(0, dx, 2)
            xcenter = (Lx // 2) - (xy[1] // 2)
            xrange = np.append(xrange, xcenter)
            xst = np.random.choice(xrange)
            xed = xst + xy[1]
            xpad1 = 0
            xpad2 = 0
        else:
            xst = 0
            xed = Lx
            xpad1 = int(abs(dx)/2)
            xpad2 = abs(dx) - xpad1

        if X[i].ndim == 3:
            nchan = X[0].shape[0]
            imgi = np.zeros((nchan, xy[0], xy[1]), np.float32)
            for m in range(nchan):
                imgi[m] = np.pad(X[i][m, yst:yed, xst:xed], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
        elif X[i].ndim == 2:
            imgi = np.pad(X[i][yst:yed, xst:xed], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
        imgs.append(imgi)

        if M is not None:
            maski = np.pad(M[i][yst:yed, xst:xed], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
            masks.append(maski)

        if C is not None:
            classi = np.pad(C[i][yst:yed, xst:xed], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
            classes.append(classi)

    if C is not None:
        return imgs, masks, classes
    return imgs, masks


def clip_center(X, M=None, xy = (224,224), C=None):
    """
        # 先clip再pad
        X: imgs
        M: masks
    """
    nimg = len(X)
    imgs = []
    masks = []
    if C is not None:
        classes = []
    for i in range(nimg):
        Ly, Lx = X[i].shape[-2:]
        # print('Ly-Lx', Ly, Lx)
        dy = Ly - xy[0]
        dx = Lx - xy[1]
        if dy>0:
            yst = (Ly // 2) - (xy[0] // 2)
            yed = yst + xy[0]
            ypad1 = 0
            ypad2 = 0
        else:
            yst = 0
            yed = Ly
            ypad1 = int(abs(dy)/2)
            ypad2 = abs(dy) - ypad1
        if dx>0:
            xst = (Lx // 2) - (xy[1] // 2)
            xed = xst + xy[1]
            xpad1 = 0
            xpad2 = 0
        else:
            xst = 0
            xed = Lx
            xpad1 = int(abs(dx)/2)
            xpad2 = abs(dx) - xpad1

        if X[i].ndim == 3:
            nchan = X[0].shape[0]
            imgi = np.zeros((nchan, xy[0], xy[1]), np.float32)
            for m in range(nchan):
                imgi[m] = np.pad(X[i][m, yst:yed, xst:xed], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
        elif X[i].ndim == 2:
            imgi = np.pad(X[i][yst:yed, xst:xed], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
        imgs.append(imgi)

        maski = np.pad(M[i][yst:yed, xst:xed], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
        masks.append(maski)

        if C is not None:
            classi = np.pad(C[i][yst:yed, xst:xed], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
            classes.append(classi)

    if C is not None:
        return imgs, masks, classes

    return imgs, masks


def pad_image(X, M, C=None):
    nimg = len(X)
    Ly, Lx = X[0].shape[-2:]
    imgs = []
    masks = []
    if C is not None:
        classes = []
    for i in range(nimg):
        dxy = Ly - Lx
        if dxy < 0:
            n = math.ceil(Lx/16) * 16
            dxy = n - Ly
            ypad1 = int(dxy//2)
            ypad2 = dxy-ypad1
            xpad1 = int((n-Lx)//2)
            xpad2 = n-Lx-xpad1
        elif dxy > 0:
            n = math.ceil(Ly/16) * 16
            dxy = n - Lx
            xpad1 = int(abs(dxy//2))
            xpad2 = abs(dxy)-xpad1
            ypad1 = int((n-Ly)//2)
            ypad2 = n-Ly-ypad1
        if X[i].ndim == 3:
            nchan = X[0].shape[0]
            imgi = np.zeros((nchan, n, n), np.float32)
            for m in range(nchan):
                imgi[m] = np.pad(X[i][m], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
        elif X[i].ndim == 2:
            imgi = np.pad(X[i], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
        imgs.append(imgi)

        maski = np.pad(M[i], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
        masks.append(maski)

        if C is not None:
            classi = np.pad(C[i], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
            classes.append(classi)

    if C is not None:
        return imgs, masks, classes

    return imgs, masks


def random_sample_cell(X, M, percent=0.1, number=None):
    """
        X: imgs
        M: masks
    """
    nimg = len(X)
    imgs = []
    masks = []

    for i in range(nimg):
        total_num = int(M[i].max())
        if number is None:
            number = percent*total_num
        if number == 0:
            return X, M
        a = random.sample(range(1, total_num + 1), int(number))

        assert M[i].ndim == 2, 'Mask should be gray image'

        Ly, Lx = np.array(X[i]).shape[-2:]
        maski = np.zeros((Ly, Lx), np.uint16)
        for k in a:
            maski[M[i] == k] = k

        maski_copy = cp.deepcopy(maski)
        maski_copy[maski_copy != 0] = 1
        if X[i].ndim == 3:
            nchan = np.array(X[i]).shape[0]
            imgi = np.zeros((nchan, Ly, Lx), np.float32)
            for n in range(nchan):
                imgi[n, :, :] = X[i][n, :, :] * maski_copy
        elif X[i].ndim == 2:
            imgi = X[i][:, :] * maski_copy

        imgs.append(imgi)
        masks.append(maski)

        return imgs, masks

def mask_quality_contorl(M, M_raw, threshold=0.1, C=None):
    nimg = len(M)
    for i in range(nimg):
        total_num = int(M[i].max())
        for j in range(1, total_num):
            new_size=np.sum(M[i] == j)
            old_size=np.sum(M_raw[i] == j)
            # print(new_size, old_size)
            if (new_size/old_size) < threshold:
                M[i][M[i] == j] = 0
                if C is not None:
                    C[i][M[i] == j] = 0
    if C is not None:
        return M, C
    return M
