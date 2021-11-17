import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import tempfile

from scipy.ndimage import median_filter
import cv2
import torch
import torch.nn as nn
from scellseg.io import imread
from scellseg.utils import diameters

from . import transforms, dynamics, utils, plot, metrics, core, io, dataset
from .contrast_learning.dataset import DatasetPairEval
from .core import UnetModel, assign_device, check_mkl, use_gpu, convert_images, parse_model_string, TORCH_ENABLED
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


urls = ['https://www.cellpose.org/models/cyto_0',
        'https://www.cellpose.org/models/cyto_1',
        'https://www.cellpose.org/models/cyto_2',
        'https://www.cellpose.org/models/cyto_3',
        'https://www.cellpose.org/models/size_cyto_0.npy',
        'https://www.cellpose.org/models/cytotorch_0',
        'https://www.cellpose.org/models/cytotorch_1',
        'https://www.cellpose.org/models/cytotorch_2',
        'https://www.cellpose.org/models/cytotorch_3',
        'https://www.cellpose.org/models/size_cytotorch_0.npy',
        'https://www.cellpose.org/models/nuclei_0',
        'https://www.cellpose.org/models/nuclei_1',
        'https://www.cellpose.org/models/nuclei_2',
        'https://www.cellpose.org/models/nuclei_3',
        'https://www.cellpose.org/models/size_nuclei_0.npy',
        'https://www.cellpose.org/models/nucleitorch_0',
        'https://www.cellpose.org/models/nucleitorch_1',
        'https://www.cellpose.org/models/nucleitorch_2',
        'https://www.cellpose.org/models/nucleitorch_3',
        'https://www.cellpose.org/models/size_nucleitorch_0.npy']

def download_model_weights(urls=urls):
    # scellseg directory
    cp_dir = pathlib.Path.home().joinpath('.scellseg')
    cp_dir.mkdir(exist_ok=True)
    model_dir = cp_dir.joinpath('models')
    model_dir.mkdir(exist_ok=True)

    for url in urls:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(model_dir, filename)
        if not os.path.exists(cached_file):
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            utils.download_url_to_file(url, cached_file, progress=True)


download_model_weights()
project_path = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())) + os.path.sep + ".")
model_dir = os.path.join(project_path, 'assets', 'pretrained_models')

def dx_to_circ(dP):
    """ dP is 2 x Y x X => 'optic' flow representation """
    sc = max(np.percentile(dP[0], 99), np.percentile(dP[0], 1))
    Y = np.clip(dP[0] / sc, -1, 1)
    sc = max(np.percentile(dP[1], 99), np.percentile(dP[1], 1))
    X = np.clip(dP[1] / sc, -1, 1)
    H = (np.arctan2(Y, X) + np.pi) / (2 * np.pi)
    S = utils.normalize99(dP[0] ** 2 + dP[1] ** 2)
    V = np.ones_like(S)
    HSV = np.concatenate((H[:, :, np.newaxis], S[:, :, np.newaxis], S[:, :, np.newaxis]), axis=-1)
    HSV = np.clip(HSV, 0.0, 1.0)
    flow = (utils.hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return flow


class sCellSeg(UnetModel):
    """
    Parameters:

    pretrained_model: string (default, None)
        path to model file, if False, no model loaded
        this pretrained model parameter is used in pre-train or fine-tuning process
    net_avg: bool (optional, default True)
        loads the model file in the pretrained model folder and averages them if True, loads one network if False
        (here the pretrained model parameter is provided in inference process, we set two pretrained model parameter to adapt to more different conditions)
    diam_mean: float (optional, default 30.)
        mean 'diameter', 30. is built in value for 'cyto' model
    device: torch device (optional, default None)
    attn_on: bool (optional, default True)
        choose to use attention gates
    dense_on: bool (optional, default True)
        choose to use dense unit
    style_scale_on: bool (optional, default True)
        choose to use hierarchical style information, for Scellseg, attn_on, dense_on, style_scale_on are set to Ture as a default
    task_mode: string (default, None)
        different instance representation, "cellpose", "hover", "unet3", "unet2" can be chosen
    model: nn.Module (default, None)
        you can input your designed model architecture like class sCSnet(nn.Module) in net_desc.py

    """
    def __init__(self, gpu=False, pretrained_model=False, model_type='scellseg',
                 diam_mean=30., net_avg=True, device=None, nclasses=3,
                 residual_on=True, style_on=True, concatenation=False, update_step=1,
                 last_conv_on=True, attn_on=False, dense_on=False, style_scale_on=True,
                 task_mode='cellpose', model=None):

        model_type = utils.process_model_type(model_type)
        model_types = ['scellseg', 'cellpose', 'hover', 'unet3', 'unet2']
        if model_type not in model_types:
            print(model_type, 'not in pre-designed model, you can design your own model in this mode')
        else:
            task_mode, postproc_mode, attn_on, dense_on, style_scale_on = utils.process_different_model(model_type)  # task_mode mean different instance representation
            pretrained_model = os.path.join(model_dir, model_type)

        self.net_avg = net_avg
        self.postproc_mode = postproc_mode

        self.task_mode = task_mode
        if task_mode=='unet2':
            nclasses = 2
        self.nclasses = nclasses
        if 'unet' not in self.task_mode:
            self.unet = False

        # initialize network
        super().__init__(gpu=gpu, pretrained_model=False,
                         diam_mean=diam_mean, net_avg=net_avg, device=device,
                         residual_on=residual_on, style_on=style_on, concatenation=concatenation, update_step=update_step,
                         attn_on=attn_on, dense_on=dense_on, style_scale_on=style_scale_on,
                         nclasses=nclasses,
                         last_conv_on=last_conv_on, task_mode=task_mode, model=model)

        if isinstance(pretrained_model, np.ndarray):
            pretrained_model = list(pretrained_model)
        self.pretrained_model = pretrained_model
        if self.pretrained_model is not None and isinstance(self.pretrained_model, str):
            self.net.load_model(self.pretrained_model, cpu=(not self.gpu), last_conv_on=last_conv_on)
            print('>>>> load pretrained model', self.pretrained_model)
        ostr = ['off', 'on']
        self.net_type = 'residual_{}_style_{}_concatenation_{}'.format(ostr[residual_on],
                                                                                ostr[style_on],
                                                                                ostr[concatenation])


    def loss_fn(self, lbl, y):
        """ loss function between true labels lbl and prediction y """
        if self.task_mode == 'cellpose' or self.task_mode == 'hover':
            veci = (5. * lbl[:, 1:3]).float().to(self.device)
            cellprob = (lbl[:, 0] > .5).float().to(self.device)  # lbl第一个通道是概率
            mse_criterion = nn.MSELoss(reduction='mean', size_average=True)
            bce_criterion = nn.BCEWithLogitsLoss(reduction='mean')

            loss_map = mse_criterion(y[:, :2], veci)
            loss_cellprob = bce_criterion(y[:, 2], cellprob)
            # true_cellprob = F.one_hot(cellprob.float().to(self.device).long())  # 使用自定义的loss，这里onehot肯定没有问题,
            # pred_cellprob = torch.transpose(y[:, 2:3], 1, 3)
            # loss_cellprob = dice_loss(pred_cellprob, true_cellprob)  # 细胞概率上的loss, y[:, -1] or y[:, 2:]

            loss = loss_map * 0.5 + loss_cellprob * 1.
        elif 'unet' in self.task_mode:
            ce_criterion = nn.CrossEntropyLoss()
            nclasses = int(self.task_mode[-1])
            if lbl.shape[1] > 1 and nclasses > 2:
                boundary = lbl[:, 1] <= 4
                lbl = lbl[:, 0]  # lbl是一张图上的像素级分类
                lbl[boundary] *= 2
            else:
                lbl = lbl[:, 0]
            lbl = lbl.float().to(self.device)
            loss = 8 * 1. / nclasses * ce_criterion(y, lbl.long())

        return loss

    def set_optimizer(self, lr):
        optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, self.net.extractor.downsample.parameters()), 'key_name': 'downsample'},
             {'params': self.net.extractor.upsample.parameters(), 'lr': lr['upsample'], 'key_name': 'upsample'},
             {'params': self.net.tasker.parameters(), 'lr': lr['tasker'], 'key_name': 'tasker'},
             {'params': self.net.loss_fn.alpha, 'lr': lr['alpha'], 'key_name': 'alpha'}], lr=lr['downsample'])  # 原来的单分支，class也并到一起了，tasker要调大
        return optimizer

    def lr_step(self, now_step, step_size):
        if now_step > 0:
            if (now_step % step_size)==0:
                for p in self.net.optimizer.param_groups:
                    key_name = p['key_name']
                    p['lr'] = np.float32((p['lr'] * self.net.lr_schedule_gamma[key_name]))


    def pretrain(self, train_data, train_labels, train_files=None,
              test_data=None, test_labels=None, test_files=None,
              channels=None, normalize=True, pretrained_model=None,
              save_path=None, save_every=100, use_adam=False,
              learning_rate=0.2, n_epochs=500, momentum=0.9, weight_decay=0.00001, batch_size=8, rescale=True):

        """ train network with images train_data

            Parameters
            ------------------

            train_data: list of arrays (2D or 3D)
                images for training

            train_labels: list of arrays (2D or 3D)
                labels for train_data, where 0=no masks; 1,2,...=mask labels
                can include flows as additional images

            train_files: list of strings
                file names for images in train_data (to save flows for future runs)

            test_data: list of arrays (2D or 3D)
                images for testing

            test_labels: list of arrays (2D or 3D)
                labels for test_data, where 0=no masks; 1,2,...=mask labels;
                can include flows as additional images

            test_files: list of strings
                file names for images in test_data (to save flows for future runs)

            channels: list of ints (default, None)
                channels to use for training

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            pretrained_model: string (default, None)
                path to pretrained_model to start from, if None it is trained from scratch

            save_path: string (default, None)
                where to save trained model, if None it is not saved

            save_every: int (default, 100)
                save network every [save_every] epochs

            learning_rate: float (default, 0.2)
                learning rate for training

            n_epochs: int (default, 500)
                how many times to go through whole training set during training

            weight_decay: float (default, 0.00001)

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            rescale: bool (default, True)
                whether or not to rescale images to diam_mean during training,
                if True it assumes you will fit a size model after training or resize your images accordingly,
                if False it will try to train the model to be scale-invariant (works worse)

        """

        train_data, train_labels, test_data, test_labels, run_test = transforms.reshape_train_test(train_data,
                                                                                                   train_labels,
                                                                                                   test_data,
                                                                                                   test_labels,
                                                                                                   channels, normalize)

        # check if train_labels have flows
        if self.task_mode=='cellpose':
            train_lbls = dynamics.labels_to_flows(train_labels, files=train_files)
            if run_test:
                test_lbls = dynamics.labels_to_flows(test_labels, files=test_files)
            else:
                test_lbls = None
        elif self.task_mode=='hover':
            train_lbls = dynamics.labels_to_hovers(train_labels, files=train_files)
            if run_test:
                test_lbls = dynamics.labels_to_hovers(test_labels, files=test_files)
            else:
                test_lbls = None
        elif self.task_mode=='unet3':
            print('computing boundary pixels')
            train_lbls = [np.stack((train_label, train_label>0, utils.distance_to_boundary(train_label)), axis=0).astype(np.float32)
                                for train_label in tqdm(train_labels)]
            if run_test:
                test_lbls = [np.stack((test_label, test_label>0, utils.distance_to_boundary(test_label)), axis=0).astype(np.float32)
                                    for test_label in tqdm(test_labels)]
            else:
                test_lbls = None
        elif self.task_mode=='unet2':
            train_lbls = [np.stack((train_label, train_label>0), axis=0).astype(np.float32)
                                for train_label in tqdm(train_labels)]
            if run_test:
                test_lbls = [np.stack((test_label, test_label>0), axis=0).astype(np.float32)
                                    for test_label in tqdm(test_labels)]
            else:
                test_lbls = None

        d = datetime.datetime.now()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.net.alpha = torch.tensor(1., requires_grad=True)
        self._set_optimizer(self.learning_rate, momentum, weight_decay, use_adam=use_adam)

        # set learning rate schedule
        LR = np.linspace(0, self.learning_rate, 10)
        if self.n_epochs > 250:
            LR = np.append(LR, self.learning_rate * np.ones(self.n_epochs - 100))
            for i in range(10):
                LR = np.append(LR, LR[-1] / 2 * np.ones(10))
        else:
            LR = np.append(LR, self.learning_rate * np.ones(max(0, self.n_epochs - 10)))

        nimg = len(train_data)

        # compute average cell diameter
        if rescale:
            diam_train = np.array(
                [utils.diameters(train_labels[k][0])[0] for k in range(len(train_labels))])  # 求出每个训练图像的平均直径
            diam_train[diam_train < 5] = 5.
            if test_data is not None:
                diam_test = np.array([utils.diameters(test_labels[k][0])[0] for k in range(len(test_labels))])
                diam_test[diam_test < 5] = 5.
            scale_range = 0.5
        else:
            scale_range = 1.0

        nchan = train_data[0].shape[0]
        print('>>>> training network with %d channel input <<<<' % nchan)
        print('>>>> saving every %d epochs' % save_every)
        print('>>>> median diameter = %d' % self.diam_mean)
        print(
            '>>>> LR: %0.5f, batch_size: %d, weight_decay: %0.5f' % (self.learning_rate, self.batch_size, weight_decay))
        print('>>>> ntrain = %d' % nimg)
        if test_data is not None:
            print('>>>> ntest = %d' % len(test_data))
        # print(train_data[0].shape)

        tic = time.time()
        lavg, nsum = 0, 0

        project_path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
        output_path = os.path.join(project_path, 'output')
        utils.make_folder(output_path)
        output_log_path = os.path.join(output_path, 'train_logs')
        utils.make_folder(output_log_path)
        log_dir = os.path.join(output_log_path, d.strftime("%Y%m%d-%H%M"))
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        log_writer = SummaryWriter(log_dir)

        if save_path is not None:
            _, file_label = os.path.split(save_path)
            file_path = os.path.join(save_path, 'models/')

            if not os.path.exists(file_path):
                os.makedirs(file_path)
        else:
            print('WARNING: no save_path given, model not saving')

        ksave = 0
        rsc = 1.0

        # cannot train with mkldnn
        self.net.mkldnn = False

        for iepoch in range(self.n_epochs):
            np.random.seed(iepoch)
            rperm = np.random.permutation(nimg)  # 随机打乱图像
            self._set_learning_rate(LR[iepoch])
            self.net.phase = 'train'
            self.net.train()

            for ibatch in range(0, nimg, batch_size):
                inds = rperm[ibatch:ibatch + batch_size]  # 取batch个图像
                rsc = diam_train[inds] / self.diam_mean if rescale else np.ones(len(inds),
                                                                                np.float32)  # 每个图像的平均直径与平均直径的比值
                # print('rsc', rsc)
                # print('scale_range', scale_range)
                imgi, lbl, scale = transforms.random_rotate_and_resize(
                    [train_data[i] for i in inds], Y=[train_lbls[i][1:] for i in inds],
                    rescale=rsc, scale_range=scale_range, unet=self.unet)
                # if self.unet and lbl.shape[1]>1 and rescale:
                # lbl[:,1] /= diam_batch[:,np.newaxis,np.newaxis]**2

                imgi = torch.from_numpy(imgi).float().to(self.device)
                lbl = torch.from_numpy(lbl).float().to(self.device)

                logits_train = self.net(imgi)[0]
                train_loss = self.loss_fn(lbl, logits_train)

                self.optimizer.zero_grad()
                train_loss.backward()
                train_loss_ = train_loss.item()
                self.optimizer.step()

                n_train_img = len(imgi)
                train_loss_ *= n_train_img  # 等于query的数量，因为这里外部只用query的计算了loss，对meta-learner进行更新
                lavg += train_loss_
                nsum += n_train_img  # 也是为query的数量

            if iepoch % 10 == 0 or iepoch < 10:  # 测试前10批和10的整数倍的
                lavg = lavg / nsum  # 平均的loss
                if test_data is not None:
                    lavgt, nsum = 0., 0
                    np.random.seed(42)
                    rperm = np.arange(0, len(test_data), 1, int)
                    self.net.phase = 'val'
                    self.net.eval()
                    for ibatch in range(0, len(test_data), batch_size):
                        inds = rperm[ibatch:ibatch + batch_size]
                        rsc = diam_test[inds] / self.diam_mean if rescale else np.ones(len(inds), np.float32)
                        imgi, lbl, scale = transforms.random_rotate_and_resize(
                            [test_data[i] for i in inds],
                            Y=[test_lbls[i][1:] for i in inds],
                            scale_range=0., rescale=rsc, unet=self.unet)  # TODO:评估时scale_range不一样了
                        if self.unet and lbl.shape[1] > 1 and rescale:
                            lbl[:, 1] *= scale[0] ** 2

                        imgi = torch.from_numpy(imgi).float().to(self.device)
                        lbl = torch.from_numpy(lbl).float().to(self.device)

                        logits_val = self.net(imgi)[0]
                        val_loss = self.loss_fn(lbl, logits_val)
                        val_loss_ = val_loss.item()
                        n_val_img = len(imgi)
                        val_loss_ *= n_val_img  # 等于query的数量，因为这里外部只用query的计算了loss，对meta-learner进行更新
                        lavgt += val_loss_
                        nsum += n_val_img

                    log_writer.add_scalars('train_scellseg', {'train_loss': lavg, 'val_loss': lavgt / nsum}, iepoch)
                    print('Epoch %d, Time %4.1fs, Loss %2.4f, Loss Test %2.4f, LR %2.4f' %
                          (iepoch, time.time() - tic, lavg, lavgt / nsum, LR[iepoch]))
                else:
                    log_writer.add_scalars('train_scellseg', {'train_loss': lavg}, iepoch)
                    print('Epoch %d, Time %4.1fs, Loss %2.4f, LR %2.4f' %
                          (iepoch, time.time() - tic, lavg, LR[iepoch]))
                lavg, nsum = 0, 0

            if save_path is not None:
                if iepoch == self.n_epochs - 1 or iepoch % save_every == 1:
                    # save model at the end
                    file = '{}_{}_{}_{}'.format(str(iepoch), self.net_type, file_label, d.strftime("%Y%m%d%H_%M%S"))
                    ksave += 1
                    print('saving network parameters')
                    self.net.save_model(os.path.join(file_path, file))

        # reset to mkldnn if available
        self.net.mkldnn = self.mkldnn
        model_path = os.path.join(file_path, file)

        # find threshold using validation set
        if 'unet' in self.task_mode:
            print('>>>> finding best thresholds using validation set')
            cell_threshold, boundary_threshold = self.threshold_validation(test_data, test_lbls)
            np.save(model_path+'_cell_boundary_threshold.npy', np.array([cell_threshold, boundary_threshold]))

        self.pretrained_model = model_path
        return model_path


    def finetune(self, shot_gen=None, lr=None, lr_schedule_gamma=None, step_size=20):
        """The function for the fine-tune phase."""
        assert shot_gen is not None, print('please provide shot images')
        self.net.train()
        self.net.phase = 'shot_train'

        # cannot train with mkldnn
        self.net.mkldnn = False

        if lr is None:
            self.net.lr = {'downsample': 0.001, 'upsample': 0.001, 'tasker': 0.001, 'alpha': 0.1}
        else:
            self.net.lr = lr
        if lr_schedule_gamma is None:
            self.net.lr_schedule_gamma = {'downsample': 0.5, 'upsample': 0.5, 'tasker': 0.5, 'alpha': 0.5}
        else:
            self.net.lr_schedule_gamma = lr_schedule_gamma

        self.net.optimizer = self.set_optimizer(lr)

        project_path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
        output_path = os.path.join(project_path, 'output')
        utils.make_folder(output_path)
        output_model_path = os.path.join(output_path, 'fine-tune')
        utils.make_folder(output_model_path)
        finetune_model_path = os.path.join(output_model_path, 'finetune_' + self.net.save_name)

        for m, batch in enumerate(shot_gen):
            self.net.m = m

            shot_images = batch[0]
            shot_lbls = batch[1]
            if self.gpu:
                shot_images = shot_images.cuda()
                shot_lbls = shot_lbls.cuda()
            self.net((shot_images, shot_lbls))
            self.lr_step(m, step_size=step_size)

        self.net.save_model(finetune_model_path)
        self.net.mkldnn = self.mkldnn
        print('\033[1;32m>>>> model fine-tuned and saved to: \033[0m', finetune_model_path)


    def inference(self, finetune_model=None, query_image_names=None, query_images=None, eval_batch_size=8,
            channel=None, diameter=None, normalize=True, invert=False,
            augment=False, tile=True, tile_overlap=0.5, net_avg=False,
            resample=False, interp=True, do_3D=False, anisotropy=None, stitch_threshold=0,
            postproc_mode='cellpose', flow_threshold=0.4, cellprob_threshold=0.5, compute_masks=True, min_size=15,
            progress=None, shot_pairs=None):
        """The function for the query-evaluate phase."""
        self.postproc_mode = postproc_mode

        self.net.eval()
        self.net.phase = 'eval'

        if finetune_model is not None:
            if os.path.isdir(finetune_model):
                finetune_model = [os.path.join(finetune_model, finetune_modeli) for finetune_modeli in os.listdir(finetune_model)]
            elif os.path.isfile(finetune_model):
                finetune_model = [finetune_model]
            self.pretrained_model = finetune_model

        print(">>>> load inference model", self.pretrained_model)

        if (query_images is None):
            query_images = self.load_eval_imgs(query_image_names, do_3D=do_3D)

        if np.array(channel).ndim==1:
            channels = [channel for n in range(len(query_images))]
        if self.task_mode=='cellpose' or self.task_mode == 'hover':
            masks, flows, styles = self.cellpose_eval(query_images, batch_size=eval_batch_size, net_avg=net_avg,
                     channels=channels, normalize=normalize, invert=invert, rescale=None, diameter=diameter,
                     do_3D=do_3D, anisotropy=anisotropy, stitch_threshold=stitch_threshold,
                     augment=augment, tile=tile, tile_overlap=tile_overlap,
                     resample=resample, interp=interp, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, compute_masks=compute_masks, min_size=min_size,
                     progress=progress)

        elif 'unet' in self.task_mode:
            masks, flows, styles = self.classic_eval(query_images, batch_size=eval_batch_size, net_avg=net_avg,
                     channels=channels, invert=invert, normalize=normalize, rescale=None, diameter=diameter,
                     do_3D=do_3D, anisotropy=anisotropy,
                     augment=augment, tile=tile,
                     min_size=min_size, shot_pairs=shot_pairs,
                     progress=progress)

        return masks, flows, styles


    def cellpose_eval(self, imgs, batch_size=8, net_avg=True,
             channels=None, normalize=True, invert=False, rescale=None, diameter=None,
             do_3D=False, anisotropy=None, stitch_threshold=0.0,
             augment=False, tile=True, tile_overlap=0.5,
             resample=False, interp=True, flow_threshold=0.4, cellprob_threshold=0.5, compute_masks=True, min_size=15,
             progress=None):
        """
            segment list of images imgs, or 4D array - Z x nchan x Y x X

            Parameters
            ----------
            imgs: list or array of images
                can be list of 2D/3D/4D images, or array of 2D/3D images

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            channels: list (optional, default None)
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            invert: bool (optional, default False)
                invert image pixel intensity before running network

            rescale: float (optional, default None)
                resize factor for each image, if None, set to 1.0

            diameter: float (optional, default None)
                diameter for each image (only used if rescale is None),
                if diameter is None, set to diam_mean

            do_3D: bool (optional, default False)
                set to True to run 3D segmentation on 4D image input

            anisotropy: float (optional, default None)
                for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

            net_avg: bool (optional, default True)
                runs the 4 built-in networks and averages them if True, runs one network if False

            augment: bool (optional, default False)
                tiles image with overlapping tiles and flips overlapped regions to augment

            tile: bool (optional, default True)
                tiles image to ensure GPU/CPU memory usage limited (recommended)

            tile_overlap: float (optional, default 0.1)
                fraction of overlap of tiles when computing flows

            resample: bool (optional, default False)
                run dynamics at original image size (will be slower but create more accurate boundaries)

            interp: bool (optional, default True)
                interpolate during 2D dynamics (not available in 3D)
                (in previous versions it was False)

            flow_threshold: float (optional, default 0.4)
                flow error threshold (all cells with errors below threshold are kept) (not used for 3D)

            cellprob_threshold: float (optional, default 0.0)
                cell probability threshold (all pixels with prob above threshold kept for masks)

            compute_masks: bool (optional, default True)
                Whether or not to compute dynamics and return masks.
                This is set to False when retrieving the styles for the size model.

            min_size: int (optional, default 15)
                minimum number of pixels per mask, can turn off with -1

            stitch_threshold: float (optional, default 0.0)
                if stitch_threshold>0.0 and not do_3D, masks are stitched in 3D to return volume segmentation

            progress: pyqt progress bar (optional, default None)
                to return progress bar status to GUI

            Returns
            -------
            masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels

            flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
                flows[k][0] = XY flow in HSV 0-255
                flows[k][1] = flows at each pixel
                flows[k][2] = the cell probability centered at 0.0

            styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
                style vector summarizing each image, also used to estimate size of objects in image

        """
        x, nolist = convert_images(imgs.copy(), channels, do_3D, normalize, invert)

        nimg = len(x)
        self.batch_size = batch_size

        styles = []
        flows = []
        masks = []

        if rescale is None:
            if diameter is not None:
                if not isinstance(diameter, (list, np.ndarray)):
                    diameter = diameter * np.ones(nimg)
                rescale = self.diam_mean / diameter
            else:
                rescale = np.ones(nimg)
        elif isinstance(rescale, float):
            rescale = rescale * np.ones(nimg)

        iterator = trange(nimg) if nimg > 1 else range(nimg)

        if isinstance(self.pretrained_model, list) and not net_avg:
            self.net.load_model(self.pretrained_model[0], cpu=(not self.gpu), last_conv_on=True)
        # model_dict = self.net.state_dict()

        if not do_3D:
            flow_time = 0
            net_time = 0
            for i in iterator:
                img = x[i].copy()
                Ly, Lx = img.shape[:2]

                tic = time.time()
                shape = img.shape
                # rescale image for flow computation
                img = transforms.resize_image(img, rsz=rescale[i])
                y, style = self._run_nets(img, net_avg=net_avg,
                                          augment=augment, tile=tile,
                                          tile_overlap=tile_overlap)
                net_time += time.time() - tic
                if progress is not None:
                    progress.setValue(55)
                styles.append(style)
                if compute_masks:
                    tic = time.time()
                    if resample:  # run dynamics at original image size (will be slower but create more accurate boundaries)
                        y[:, :, :3] = transforms.resize_image(y[:, :, :3], shape[-3], shape[-2])

                    if self.postproc_mode == 'cellpose':
                        cellprob = y[:, :, 2]
                        # cellprob = y[:, :, -1] - y[:, :, -2]
                        dP = y[:, :, :2].transpose((2, 0, 1))
                        niter = 1 / rescale[i] * 200
                        p = dynamics.follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5.,
                                                  niter=niter, interp=interp, use_gpu=self.gpu)
                        if progress is not None:
                            progress.setValue(65)
                        maski = dynamics.get_masks(p, iscell=(cellprob > cellprob_threshold),
                                                   flows=dP, threshold=flow_threshold)
                        maski = utils.fill_holes_and_remove_small_masks(maski, min_size=min_size)
                        flows.append([dx_to_circ(dP), dP, cellprob, p])
                    elif self.postproc_mode == 'watershed':
                        maski = dynamics.get_masks_watershed(y[:, :, :3], cellprob_threshold=cellprob_threshold, min_size=min_size)
                        cellprob = y[:, :, 2]
                        dP = y[:, :, :2].transpose((2, 0, 1))
                        niter = 1 / rescale[i] * 200
                        p = dynamics.follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5.,
                                                  niter=niter, interp=interp, use_gpu=self.gpu)
                        flows.append([dx_to_circ(dP), dP, cellprob, p])

                    maski = transforms.resize_image(maski, shape[-3], shape[-2],
                                                    interpolation=cv2.INTER_NEAREST)

                    if progress is not None:
                        progress.setValue(75)
                    # dP = np.concatenate((dP, np.zeros((1,dP.shape[1],dP.shape[2]), np.uint8)), axis=0)
                    masks.append(maski)
                    flow_time += time.time() - tic
                else:
                    flows.append([None] * 3)
                    masks.append([])
            if compute_masks:
                print('time spent: running network %0.2fs; flow+mask computation %0.2f' % (net_time, flow_time))

            if stitch_threshold > 0.0 and nimg > 1 and all([m.shape == masks[0].shape for m in masks]):
                print('stitching %d masks using stitch_threshold=%0.3f to make 3D masks' % (nimg, stitch_threshold))
                masks = utils.stitch3D(np.array(masks), stitch_threshold=stitch_threshold)
        else:
            for i in iterator:
                tic = time.time()
                shape = x[i].shape
                yf, style = self._run_3D(x[i], rsz=rescale[i], anisotropy=anisotropy,
                                         net_avg=net_avg, augment=augment, tile=tile,
                                         tile_overlap=tile_overlap, progress=progress)
                cellprob = yf[0][-1] + yf[1][-1] + yf[2][-1]
                dP = np.stack((yf[1][0] + yf[2][0], yf[0][0] + yf[2][1], yf[0][1] + yf[1][1]),
                              axis=0)  # (dZ, dY, dX)
                print('flows computed %2.2fs' % (time.time() - tic))
                # ** mask out values using cellprob to increase speed and reduce memory requirements **
                yout = dynamics.follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5.)
                print('dynamics computed %2.2fs' % (time.time() - tic))
                maski = dynamics.get_masks(yout, iscell=(cellprob > cellprob_threshold))
                maski = utils.fill_holes_and_remove_small_masks(maski, min_size=min_size)
                print('masks computed %2.2fs' % (time.time() - tic))
                flow = np.array([dx_to_circ(dP[1:, i]) for i in range(dP.shape[1])])
                flows.append([flow, dP, cellprob, yout])
                masks.append(maski)
                styles.append(style)
        if nolist:
            masks, flows, styles = masks[0], flows[0], styles[0]

        return masks, flows, styles

    def classic_eval(self, x, batch_size=8, channels=None, invert=False, normalize=True,
             rescale=None, diameter=None, do_3D=False, anisotropy=None, net_avg=True, augment=False,
             tile=True, cell_threshold=None, boundary_threshold=None, min_size=15, shot_pairs=None, progress=None):
        """ segment list of images x

            Parameters
            ----------
            x: list or array of images
                can be list of 2D/3D images, or array of 2D/3D images, or 4D image array

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            channels: list (optional, default None)
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=blue, 3=green).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=blue, 3=green).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].

            invert: bool (optional, default False)
                invert image pixel intensity before running network

            normalize: bool (optional, default True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            rescale: float (optional, default None)
                resize factor for each image, if None, set to 1.0

            do_3D: bool (optional, default False)
                set to True to run 3D segmentation on 4D image input

            anisotropy: float (optional, default None)
                for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

            net_avg: bool (optional, default True)
                runs the 4 built-in networks and averages them if True, runs one network if False

            augment: bool (optional, default False)
                tiles image with overlapping tiles and flips overlapped regions to augment

            tile: bool (optional, default True)
                tiles image to ensure GPU/CPU memory usage limited (recommended)

            cell_threshold: float (optional, default 0.0)
                cell probability threshold (all pixels with prob above threshold kept for masks)

            boundary_threshold: float (optional, default 0.0)
                cell probability threshold (all pixels with prob above threshold kept for masks)

            min_size: int (optional, default 15)
                minimum number of pixels per mask, can turn off with -1

            Returns
            -------
            masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels

            flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
                flows[k][0] = XY flow in HSV 0-255
                flows[k][1] = flows at each pixel
                flows[k][2] = the cell probability centered at 0.0

            styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
                style vector summarizing each image, also used to estimate size of objects in image

        """
        self.batch_size = batch_size
        x, nolist = convert_images(x, channels, do_3D, normalize, invert)
        nimg = len(x)

        styles = []
        flows = []
        masks = []

        if rescale is None:
            if diameter is not None:
                if not isinstance(diameter, (list, np.ndarray)):
                    diameter = diameter * np.ones(nimg)
                rescale = self.diam_mean / diameter
            else:
                rescale = np.ones(nimg)
        elif isinstance(rescale, float):
            rescale = rescale * np.ones(nimg)

        if isinstance(self.pretrained_model, list):
            model_path = self.pretrained_model[0].split('preeval')[0]
            if not net_avg:
                self.net.load_model(self.pretrained_model[0], cpu=(not self.gpu), last_conv_on=True)
        else:
            model_path = self.pretrained_model.split('preeval')[0]

        # 首先根据shot_imgs计算两个阈值
        if shot_pairs is not None:
            if shot_pairs[-1]:
                print('>>>> computing cell and boundary threshold ...')
                shot_imgs = self.load_eval_imgs(shot_pairs[0])
                shot_imgs, _ = convert_images(shot_imgs, [2, 1], do_3D, normalize, invert)  # 取10张加快速度
                shot_masks = [np.array(imread(shot_mask_name)) for shot_mask_name in shot_pairs[1]]

                shot_imgs, shot_masks, _ = dataset.resize_image(shot_imgs, M=shot_masks, xy=[224, 224])
                shot_imgs = [np.transpose(shot_img, (1, 2, 0)) for shot_img in shot_imgs]  # 调整轴的顺序

                cell_threshold, boundary_threshold = self.threshold_validation(shot_imgs, shot_masks)
                np.save(model_path + 'classic_cell_boundary_threshold.npy', np.array([cell_threshold, boundary_threshold]))
        if cell_threshold is None or boundary_threshold is None:
            try:
                thresholds = np.load(model_path + 'classic_cell_boundary_threshold.npy')
                cell_threshold, boundary_threshold = thresholds
                print('>>>> found saved thresholds')
            except:
                print('WARNING: no thresholds found, using default / user input')

        cell_threshold = 2.0 if cell_threshold is None else cell_threshold
        boundary_threshold = 0.5 if boundary_threshold is None else boundary_threshold

        if nimg > 1:
            iterator = trange(nimg)
        else:
            iterator = range(nimg)
        if not do_3D:  # 2D情况下
            for i in iterator:
                img = x[i].copy()
                shape = img.shape
                # rescale image for flow computation
                imgs = transforms.resize_image(img, rsz=rescale[i])
                y, style = self._run_nets(img, net_avg=net_avg, augment=augment, tile=tile)
                if progress is not None:
                    progress.setValue(85)
                maski = utils.get_masks_unet(y, cell_threshold, boundary_threshold)
                maski = utils.fill_holes_and_remove_small_masks(maski, min_size=min_size)
                maski = transforms.resize_image(maski, shape[-3], shape[-2],
                                                interpolation=cv2.INTER_NEAREST)
                masks.append(maski)
                styles.append(style)
        else:
            for i in iterator:
                tic = time.time()
                yf, style = self._run_3D(x[i], rsz=rescale[i], anisotropy=anisotropy,
                                         net_avg=net_avg, augment=augment, tile=tile)
                if progress is not None:
                    progress.setValue(85)
                yf = yf.mean(axis=0)
                print('probabilities computed %2.2fs' % (time.time() - tic))
                maski = utils.get_masks_unet(yf.transpose((1, 2, 3, 0)), cell_threshold, boundary_threshold)
                maski = utils.fill_holes_and_remove_small_masks(maski, min_size=min_size)
                masks.append(maski)
                styles.append(style)
                print('masks computed %2.2fs' % (time.time() - tic))
                flows.append(yf)

        if nolist:
            masks, flows, styles = masks[0], flows[0], styles[0]

        return masks, flows, styles


    def load_eval_imgs(self, names, do_3D=False):
        imgs = [imread(image_name) for image_name in names]
        if not isinstance(imgs,list):
            if imgs.ndim < 2 or imgs.ndim > 5:
                raise ValueError('%dD images not supported'%imgs.ndim)
            if imgs.ndim==4:
                if do_3D:
                    imgs = [imgs]
                else:
                    imgs = list(imgs)
            elif imgs.ndim==5:
                if do_3D:
                    imgs = list(imgs)
                else:
                    raise ValueError('4D images must be processed using 3D')
            else:
                imgs = [imgs]
        else:
            for imgi in imgs:
                if imgi.ndim < 2 or imgi.ndim > 5:
                    raise ValueError('%dD images not supported'%imgi.ndim)
        return imgs

    def threshold_validation(self, val_data, val_labels):
        cell_thresholds = np.arange(-4.0, 4.25, 0.5)
        nclasses = int(self.task_mode[-1])
        if nclasses == 3:
            boundary_thresholds = np.arange(-2, 2.25, 1.0)
        else:
            boundary_thresholds = np.zeros(1)
        aps = np.zeros((cell_thresholds.size, boundary_thresholds.size, 3))
        for j, cell_threshold in tqdm(enumerate(cell_thresholds)):
            for k, boundary_threshold in enumerate(boundary_thresholds):
                masks = []
                for i in range(len(val_data)):
                    output, style = self._run_nets(val_data[i], augment=False)
                    masks.append(utils.get_masks_unet(output, cell_threshold, boundary_threshold))
                ap = metrics.average_precision(val_labels, masks)[0]
                ap0 = ap.mean(axis=0)
                aps[j, k] = ap0
            if nclasses == 3:
                kbest = aps[j].mean(axis=-1).argmax()
            else:
                kbest = 0
            # if j % 4 == 0:
            #     print('best threshold at cell_threshold = {} => boundary_threshold = {}, ap @ 0.5 = {}'.format(
            #         cell_threshold, boundary_thresholds[kbest],
            #         aps[j, kbest, 0]))
        if nclasses == 3:
            jbest, kbest = np.unravel_index(aps.mean(axis=-1).argmax(), aps.shape[:2])
        else:
            jbest = aps.squeeze().mean(axis=-1).argmax()
            kbest = 0
        cell_threshold, boundary_threshold = cell_thresholds[jbest], boundary_thresholds[kbest]
        print('>>>> best overall thresholds: (cell_threshold = {}, boundary_threshold = {}); ap @ 0.5 = {}'.format(
            cell_threshold, boundary_threshold,
            aps[jbest, kbest, 0]))
        return cell_threshold, boundary_threshold

class Cellpose():
    """ main model which combines SizeModel and CellposeModel

    Parameters
    ----------

    gpu: bool (optional, default False)
        whether or not to use GPU, will check if GPU available

    model_type: str (optional, default 'cyto')
        'cyto'=cytoplasm model; 'nuclei'=nucleus model

    net_avg: bool (optional, default True)
        loads the 4 built-in networks and averages them if True, loads one network if False

    device: gpu device (optional, default None)


    torch: bool (optional, default False)
        run model using torch if available

    """

    def __init__(self, gpu=False, model_type='cyto', net_avg=True, device=None):
        super(Cellpose, self).__init__()
        torch_str = ['', 'torch'][1]

        # assign device (GPU or CPU)
        sdevice, gpu = assign_device(gpu)
        self.device = device if device is not None else sdevice
        self.gpu = gpu
        model_type = 'cyto' if model_type is None else model_type
        self.pretrained_model = [os.fspath(model_dir.joinpath('%s%s_%d' % (model_type, torch_str, j))) for j in
                                 range(4)]
        self.pretrained_size = os.fspath(model_dir.joinpath('size_%s%s_0.npy' % (model_type, torch_str)))
        self.diam_mean = 30. if model_type == 'cyto' else 17.

        if not net_avg:
            self.pretrained_model = self.pretrained_model[0]

        self.cp = CellposeModel(device=self.device, gpu=self.gpu,
                                pretrained_model=self.pretrained_model,
                                diam_mean=self.diam_mean)
        self.cp.model_type = model_type

        self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size,
                            cp_model=self.cp)
        self.sz.model_type = model_type

    def eval(self, x, batch_size=8, channels=None, invert=False, normalize=True, diameter=30., do_3D=False,
             anisotropy=None,
             net_avg=True, augment=False, tile=True, tile_overlap=0.1, resample=False, interp=True,
             flow_threshold=0.4, cellprob_threshold=0.0, min_size=15,
             stitch_threshold=0.0, rescale=None, progress=None):
        """ run scellseg and get masks

        Parameters
        ----------
        x: list or array of images
            can be list of 2D/3D images, or array of 2D/3D images, or 4D image array

        batch_size: int (optional, default 8)
            number of 224x224 patches to run simultaneously on the GPU
            (can make smaller or bigger depending on GPU memory usage)

        channels: list (optional, default None)
            list of channels, either of length 2 or of length number of images by 2.
            First element of list is the channel to segment (0=grayscale, 1=red, 2=blue, 3=green).
            Second element of list is the optional nuclear channel (0=none, 1=red, 2=blue, 3=green).
            For instance, to segment grayscale images, input [0,0]. To segment images with cells
            in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
            image with cells in green and nuclei in blue, input [[0,0], [2,3]].

        invert: bool (optional, default False)
            invert image pixel intensity before running network (if True, image is also normalized)

        normalize: bool (optional, default True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

        diameter: float (optional, default 30.)
            if set to None, then diameter is automatically estimated if size model is loaded

        do_3D: bool (optional, default False)
            set to True to run 3D segmentation on 4D image input

        anisotropy: float (optional, default None)
            for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

        net_avg: bool (optional, default True)
            runs the 4 built-in networks and averages them if True, runs one network if False

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU/CPU memory usage limited (recommended)

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        resample: bool (optional, default False)
            run dynamics at original image size (will be slower but create more accurate boundaries)

        interp: bool (optional, default True)
                interpolate during 2D dynamics (not available in 3D)
                (in previous versions it was False)

        flow_threshold: float (optional, default 0.4)
            flow error threshold (all cells with errors below threshold are kept) (not used for 3D)

        cellprob_threshold: float (optional, default 0.0)
            cell probability threshold (all pixels with prob above threshold kept for masks)

        min_size: int (optional, default 15)
                minimum number of pixels per mask, can turn off with -1

        stitch_threshold: float (optional, default 0.0)
            if stitch_threshold>0.0 and not do_3D and equal image sizes, masks are stitched in 3D to return volume segmentation

        rescale: float (optional, default None)
            if diameter is set to None, and rescale is not None, then rescale is used instead of diameter for resizing image

        progress: pyqt progress bar (optional, default None)
            to return progress bar status to GUI

        Returns
        -------
        masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels

        flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
            flows[k][0] = XY flow in HSV 0-255
            flows[k][1] = flows at each pixel
            flows[k][2] = the cell probability centered at 0.0

        styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
            style vector summarizing each image, also used to estimate size of objects in image

        diams: list of diameters, or float (if do_3D=True)

        """

        if not isinstance(x, list):
            nolist = True
            if x.ndim < 2 or x.ndim > 5:
                raise ValueError('%dD images not supported' % x.ndim)
            if x.ndim == 4:
                if do_3D:
                    x = [x]
                else:
                    x = list(x)
                    nolist = False
            elif x.ndim == 5:
                if do_3D:
                    x = list(x)
                    nolist = False
                else:
                    raise ValueError('4D images must be processed using 3D')
            else:
                x = [x]
        else:
            nolist = False
            for xi in x:
                if xi.ndim < 2 or xi.ndim > 5:
                    raise ValueError('%dD images not supported' % xi.ndim)

        tic0 = time.time()

        nimg = len(x)
        print('processing %d image(s)' % nimg)
        # make rescale into length of x
        if diameter is not None and not (not isinstance(diameter, (list, np.ndarray)) and
                                         (diameter == 0 or (diameter == 30. and rescale is not None))):
            if not isinstance(diameter, (list, np.ndarray)) or len(diameter) == 1 or len(diameter) < nimg:
                diams = diameter * np.ones(nimg, np.float32)
            else:
                diams = diameter
            rescale = self.diam_mean / diams
        else:
            if rescale is not None and (not isinstance(rescale, (list, np.ndarray)) or len(rescale) == 1):
                rescale = rescale * np.ones(nimg, np.float32)
            if self.pretrained_size is not None and rescale is None and not do_3D:
                tic = time.time()
                diams, _ = self.sz.eval(x, channels=channels, invert=invert, batch_size=batch_size,
                                        augment=augment, tile=tile)
                rescale = self.diam_mean / diams
                print('estimated cell diameters for %d image(s) in %0.2f sec' % (nimg, time.time() - tic))
                print('>>> diameter(s) = ', diams)
            else:
                if rescale is None:
                    if do_3D:
                        rescale = np.ones(1)
                    else:
                        rescale = np.ones(nimg, np.float32)
                diams = self.diam_mean / rescale
        tic = time.time()
        masks, flows, styles = self.cp.eval(x,
                                           batch_size=batch_size,
                                           invert=invert,
                                           rescale=rescale,
                                           anisotropy=anisotropy,
                                           channels=channels,
                                           augment=augment,
                                           tile=tile,
                                           do_3D=do_3D,
                                           net_avg=net_avg, progress=progress,
                                           tile_overlap=tile_overlap,
                                           resample=resample,
                                           interp=interp,
                                           flow_threshold=flow_threshold,
                                           cellprob_threshold=cellprob_threshold,
                                           min_size=min_size,
                                           stitch_threshold=stitch_threshold)
        print('estimated masks for %d image(s) in %0.2f sec' % (nimg, time.time() - tic))
        print('>>>> TOTAL TIME %0.2f sec' % (time.time() - tic0))

        if nolist:
            masks, flows, styles, diams = masks[0], flows[0], styles[0], diams[0]

        return masks, flows, styles


class CellposeModel(UnetModel):
    """

    Parameters
    -------------------

    gpu: bool (optional, default False)
        whether or not to save model to GPU, will check if GPU available

    pretrained_model: str or list of strings (optional, default False)
        path to pretrained scellseg model(s), if False, no model loaded;
        if None, built-in 'cyto' model loaded

    net_avg: bool (optional, default True)
        loads the 4 built-in networks and averages them if True, loads one network if False

    diam_mean: float (optional, default 27.)
        mean 'diameter', 27. is built in value for 'cyto' model

    device: torch device (optional, default None)

    """

    def __init__(self, gpu=False, pretrained_model=False,
                 diam_mean=30., net_avg=True, device=None,
                 residual_on=True, style_on=True, concatenation=False):

        if isinstance(pretrained_model, np.ndarray):
            pretrained_model = list(pretrained_model)
        nclasses = 3  # 3 prediction maps (dY, dX and cellprob)
        self.nclasses = nclasses
        # if pretrained_model:
        #     params = parse_model_string(pretrained_model)
        #     if params is not None:
        #         nclasses, residual_on, style_on, concatenation = params
        # # load default cyto model if pretrained_model is None
        # elif pretrained_model is None:
        #     torch_str = ['', 'torch'][1]
        #     pretrained_model = [os.fspath(model_dir.joinpath('cyto%s_%d' % (torch_str, j))) for j in
        #                         range(4)] if net_avg else os.fspath(model_dir.joinpath('cyto_0'))
        #     self.diam_mean = 30.
        #     residual_on, style_on, concatenation = True, True, False

        # initialize network
        super().__init__(gpu=gpu, pretrained_model=False,
                         diam_mean=diam_mean, net_avg=net_avg, device=device,
                         residual_on=residual_on, style_on=style_on, concatenation=concatenation,
                         nclasses=nclasses)  # 这里的pretrained_model为false，在super中不加载模型
        self.unet = False  # 覆盖super中的self.unet
        self.pretrained_model = pretrained_model
        if self.pretrained_model is not None and isinstance(self.pretrained_model, str):
            self.net.load_model(self.pretrained_model, cpu=(not self.gpu))  # 加载模型
            print("Load Pretrained Model", pretrained_model)
        ostr = ['off', 'on']
        self.net_type = 'residual_{}_style_{}_concatenation_{}'.format(ostr[residual_on],
                                                                                ostr[style_on],
                                                                                ostr[concatenation])

    def eval(self, imgs, batch_size=8, channels=None, normalize=True, invert=False,
             rescale=None, diameter=None, do_3D=False, anisotropy=None,
             augment=False, tile=True, tile_overlap=0.1, net_avg=True,
             resample=False, interp=True, flow_threshold=0.4, cellprob_threshold=0.0,
             compute_masks=True,
             min_size=15, stitch_threshold=0.0, progress=None):
        """
            segment list of images imgs, or 4D array - Z x nchan x Y x X

            Parameters
            ----------
            imgs: list or array of images
                can be list of 2D/3D/4D images, or array of 2D/3D images

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            channels: list (optional, default None)
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            invert: bool (optional, default False)
                invert image pixel intensity before running network

            rescale: float (optional, default None)
                resize factor for each image, if None, set to 1.0

            diameter: float (optional, default None)
                diameter for each image (only used if rescale is None),
                if diameter is None, set to diam_mean

            do_3D: bool (optional, default False)
                set to True to run 3D segmentation on 4D image input

            anisotropy: float (optional, default None)
                for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

            net_avg: bool (optional, default True)
                runs the 4 built-in networks and averages them if True, runs one network if False

            augment: bool (optional, default False)
                tiles image with overlapping tiles and flips overlapped regions to augment

            tile: bool (optional, default True)
                tiles image to ensure GPU/CPU memory usage limited (recommended)

            tile_overlap: float (optional, default 0.1)
                fraction of overlap of tiles when computing flows

            resample: bool (optional, default False)
                run dynamics at original image size (will be slower but create more accurate boundaries)

            interp: bool (optional, default True)
                interpolate during 2D dynamics (not available in 3D)
                (in previous versions it was False)

            flow_threshold: float (optional, default 0.4)
                flow error threshold (all cells with errors below threshold are kept) (not used for 3D)

            cellprob_threshold: float (optional, default 0.0)
                cell probability threshold (all pixels with prob above threshold kept for masks)

            compute_masks: bool (optional, default True)
                Whether or not to compute dynamics and return masks.
                This is set to False when retrieving the styles for the size model.

            min_size: int (optional, default 15)
                minimum number of pixels per mask, can turn off with -1

            stitch_threshold: float (optional, default 0.0)
                if stitch_threshold>0.0 and not do_3D, masks are stitched in 3D to return volume segmentation

            progress: pyqt progress bar (optional, default None)
                to return progress bar status to GUI

            Returns
            -------
            masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels

            flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
                flows[k][0] = XY flow in HSV 0-255
                flows[k][1] = flows at each pixel
                flows[k][2] = the cell probability centered at 0.0

            styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
                style vector summarizing each image, also used to estimate size of objects in image

        """
        x, nolist = convert_images(imgs.copy(), channels, do_3D, normalize, invert)

        nimg = len(x)
        self.batch_size = batch_size

        styles = []
        flows = []
        masks = []

        if rescale is None:
            if diameter is not None:
                if not isinstance(diameter, (list, np.ndarray)):
                    diameter = diameter * np.ones(nimg)
                rescale = self.diam_mean / diameter
            else:
                rescale = np.ones(nimg)
        elif isinstance(rescale, float):
            rescale = rescale * np.ones(nimg)

        iterator = trange(nimg) if nimg > 1 else range(nimg)

        if isinstance(self.pretrained_model, list) and not net_avg:
            self.net.load_model(self.pretrained_model[0], cpu=(not self.gpu))

        if not do_3D:
            flow_time = 0
            net_time = 0
            for i in iterator:
                img = x[i].copy()
                Ly, Lx = img.shape[:2]

                tic = time.time()
                shape = img.shape
                # rescale image for flow computation
                img = transforms.resize_image(img, rsz=rescale[i])
                y, style = self._run_nets(img, net_avg=net_avg,
                                          augment=augment, tile=tile,
                                          tile_overlap=tile_overlap)
                net_time += time.time() - tic
                if progress is not None:
                    progress.setValue(55)
                styles.append(style)
                if compute_masks:
                    tic = time.time()
                    if resample:  # run dynamics at original image size (will be slower but create more accurate boundaries)
                        y = transforms.resize_image(y, shape[-3], shape[-2])

                    cellprob = y[:, :, -1]
                    dP = y[:, :, :2].transpose((2, 0, 1))
                    niter = 1 / rescale[i] * 200
                    p = dynamics.follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5.,
                                              niter=niter, interp=interp, use_gpu=self.gpu)
                    if progress is not None:
                        progress.setValue(65)
                    maski = dynamics.get_masks(p, iscell=(cellprob > cellprob_threshold),
                                               flows=dP, threshold=flow_threshold)
                    maski = utils.fill_holes_and_remove_small_masks(maski)
                    maski = transforms.resize_image(maski, shape[-3], shape[-2],
                                                    interpolation=cv2.INTER_NEAREST)
                    if progress is not None:
                        progress.setValue(75)
                    # dP = np.concatenate((dP, np.zeros((1,dP.shape[1],dP.shape[2]), np.uint8)), axis=0)
                    flows.append([dx_to_circ(dP), dP, cellprob, p])
                    masks.append(maski)
                    flow_time += time.time() - tic
                else:
                    flows.append([None] * 3)
                    masks.append([])
            if compute_masks:
                print('time spent: running network %0.2fs; flow+mask computation %0.2f' % (net_time, flow_time))

            if stitch_threshold > 0.0 and nimg > 1 and all([m.shape == masks[0].shape for m in masks]):
                print('stitching %d masks using stitch_threshold=%0.3f to make 3D masks' % (nimg, stitch_threshold))
                masks = utils.stitch3D(np.array(masks), stitch_threshold=stitch_threshold)
        else:
            for i in iterator:
                tic = time.time()
                shape = x[i].shape
                yf, style = self._run_3D(x[i], rsz=rescale[i], anisotropy=anisotropy,
                                         net_avg=net_avg, augment=augment, tile=tile,
                                         tile_overlap=tile_overlap, progress=progress)
                cellprob = yf[0][-1] + yf[1][-1] + yf[2][-1]
                dP = np.stack((yf[1][0] + yf[2][0], yf[0][0] + yf[2][1], yf[0][1] + yf[1][1]),
                              axis=0)  # (dZ, dY, dX)
                print('flows computed %2.2fs' % (time.time() - tic))
                # ** mask out values using cellprob to increase speed and reduce memory requirements **
                yout = dynamics.follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5.)
                print('dynamics computed %2.2fs' % (time.time() - tic))
                maski = dynamics.get_masks(yout, iscell=(cellprob > cellprob_threshold))
                maski = utils.fill_holes_and_remove_small_masks(maski, min_size=min_size)
                print('masks computed %2.2fs' % (time.time() - tic))
                flow = np.array([dx_to_circ(dP[1:, i]) for i in range(dP.shape[1])])
                flows.append([flow, dP, cellprob, yout])
                masks.append(maski)
                styles.append(style)
        if nolist:
            masks, flows, styles = masks[0], flows[0], styles[0]
        return masks, flows, styles

    def loss_fn(self, lbl, y):
        """ loss function between true labels lbl and prediction y """
        veci = 5. * self._to_device(lbl[:, 1:])
        lbl = self._to_device(lbl[:, 0] > .5)  # 第一个通道时概率，这里的lbl是怎么生成的
        loss = self.criterion(y[:, :2], veci)  #
        loss /= 2.
        loss2 = self.criterion2(y[:, -1], lbl)
        loss = loss + loss2
        return loss

    def train(self, train_data, train_labels, train_files=None,
              test_data=None, test_labels=None, test_files=None,
              channels=None, normalize=True, pretrained_model=None,
              save_path=None, save_every=100,
              learning_rate=0.2, n_epochs=500, momentum=0.9, weight_decay=0.00001, batch_size=8, rescale=True):

        """ train network with images train_data

            Parameters
            ------------------

            train_data: list of arrays (2D or 3D)
                images for training

            train_labels: list of arrays (2D or 3D)
                labels for train_data, where 0=no masks; 1,2,...=mask labels
                can include flows as additional images

            train_files: list of strings
                file names for images in train_data (to save flows for future runs)

            test_data: list of arrays (2D or 3D)
                images for testing

            test_labels: list of arrays (2D or 3D)
                labels for test_data, where 0=no masks; 1,2,...=mask labels;
                can include flows as additional images

            test_files: list of strings
                file names for images in test_data (to save flows for future runs)

            channels: list of ints (default, None)
                channels to use for training

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            pretrained_model: string (default, None)
                path to pretrained_model to start from, if None it is trained from scratch

            save_path: string (default, None)
                where to save trained model, if None it is not saved

            save_every: int (default, 100)
                save network every [save_every] epochs

            learning_rate: float (default, 0.2)
                learning rate for training

            n_epochs: int (default, 500)
                how many times to go through whole training set during training

            weight_decay: float (default, 0.00001)

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            rescale: bool (default, True)
                whether or not to rescale images to diam_mean during training,
                if True it assumes you will fit a size model after training or resize your images accordingly,
                if False it will try to train the model to be scale-invariant (works worse)

        """

        train_data, train_labels, test_data, test_labels, run_test = transforms.reshape_train_test(train_data,
                                                                                                   train_labels,
                                                                                                   test_data,
                                                                                                   test_labels,
                                                                                                   channels, normalize)

        # check if train_labels have flows
        train_flows = dynamics.labels_to_flows(train_labels, files=train_files)
        if run_test:
            test_flows = dynamics.labels_to_flows(test_labels, files=test_files)
        else:
            test_flows = None

        model_path = self._train_net(train_data, train_flows,
                                     test_data, test_flows,
                                     pretrained_model, save_path, save_every,
                                     learning_rate, n_epochs, momentum, weight_decay, batch_size, rescale)
        self.pretrained_model = model_path
        return model_path


class SizeModel():
    """ linear regression model for determining the size of objects in image
        used to rescale before input to cp_model
        uses styles from cp_model

        Parameters
        -------------------

        cp_model: UnetModel or CellposeModel
            model from which to get styles

        device: gpu device (optional, default torch.cpu())

        pretrained_size: str
            path to pretrained size model

    """

    def __init__(self, cp_model, device=None, pretrained_size=None, **kwargs):
        super(SizeModel, self).__init__(**kwargs)

        self.pretrained_size = pretrained_size
        self.cp = cp_model  # SizeModel是在CellposeModel创建初始化之后
        self.device = self.cp.device
        self.diam_mean = self.cp.diam_mean
        if pretrained_size is not None:
            self.params = np.load(self.pretrained_size, allow_pickle=True).item()
            self.diam_mean = self.params['diam_mean']
        if not hasattr(self.cp, 'pretrained_model'):
            raise ValueError('provided model does not have a pretrained_model')

    def eval(self, imgs=None, styles=None, channels=None, normalize=True, invert=False, augment=False, tile=True,
             batch_size=8, progress=None):
        """ use images imgs to produce style or use style input to predict size of objects in image

            Object size estimation is done in two steps:
            1. use a linear regression model to predict size from style in image
            2. resize image to predicted size and run CellposeModel to get output masks.
                Take the median object size of the predicted masks as the final predicted size.

            Parameters
            -------------------

            imgs: list or array of images (optional, default None)
                can be list of 2D/3D images, or array of 2D/3D images

            styles: list or array of styles (optional, default None)
                styles for images x - if x is None then styles must not be None

            channels: list (optional, default None)
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=blue, 3=green).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=blue, 3=green).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            invert: bool (optional, default False)
                invert image pixel intensity before running network

            augment: bool (optional, default False)
                tiles image with overlapping tiles and flips overlapped regions to augment

            tile: bool (optional, default True)
                tiles image to ensure GPU/CPU memory usage limited (recommended)

            progress: pyqt progress bar (optional, default None)
                to return progress bar status to GUI

            Returns
            -------
            diam: array, float
                final estimated diameters from images x or styles style after running both steps

            diam_style: array, float
                estimated diameters from style alone

        """
        if styles is None and imgs is None:
            raise ValueError('no image or features given')

        if progress is not None:
            progress.setValue(10)

        if imgs is not None:
            x, nolist = convert_images(imgs.copy(), channels, False, normalize, invert)
            nimg = len(x)

        if styles is None:
            print('computing styles from images')
            styles = self.cp.eval(x, net_avg=False, augment=augment, tile=tile, compute_masks=False)[-1]
            if progress is not None:
                progress.setValue(30)
            diam_style = self._size_estimation(np.array(styles))
            if progress is not None:
                progress.setValue(50)
        else:
            styles = np.array(styles) if isinstance(styles, list) else styles
            diam_style = self._size_estimation(styles)
        diam_style[np.isnan(diam_style)] = self.diam_mean

        if imgs is not None:
            masks = self.cp.eval(x, rescale=self.diam_mean / diam_style, net_avg=False,
                                 augment=augment, tile=tile, interp=False)[0]
            diam = np.array([utils.diameters(masks[i])[0] for i in range(nimg)])
            if progress is not None:
                progress.setValue(100)

            diam[diam == 0] = self.diam_mean
            diam[np.isnan(diam)] = self.diam_mean
        else:
            diam = diam_style
            print('no images provided, using diameters estimated from styles alone')
        if nolist:
            return diam[0], diam_style[0]
        else:
            return diam, diam_style

    def _size_estimation(self, style):
        """ linear regression from style to size

            sizes were estimated using "diameters" from square estimates not circles;
            therefore a conversion factor is included (to be removed)

        """
        szest = np.exp(self.params['A'] @ (style - self.params['smean']).T +
                       np.log(self.diam_mean) + self.params['ymean'])
        szest = np.maximum(5., szest)
        return szest

    def train(self, train_data, train_labels,
              test_data=None, test_labels=None,
              channels=None, normalize=True,
              learning_rate=0.2, n_epochs=10,
              l2_regularization=1.0, batch_size=8):
        """ train size model with images train_data to estimate linear model from styles to diameters

            Parameters
            ------------------

            train_data: list of arrays (2D or 3D)
                images for training

            train_labels: list of arrays (2D or 3D)
                labels for train_data, where 0=no masks; 1,2,...=mask labels
                can include flows as additional images

            channels: list of ints (default, None)
                channels to use for training

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            n_epochs: int (default, 10)
                how many times to go through whole training set (taking random patches) for styles for diameter estimation

            l2_regularization: float (default, 1.0)
                regularize linear model from styles to diameters

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)
        """
        batch_size /= 2  # reduce batch_size by factor of 2 to use larger tiles
        batch_size = int(max(1, batch_size))
        self.cp.batch_size = batch_size
        train_data, train_labels, test_data, test_labels, run_test = transforms.reshape_train_test(train_data,
                                                                                                   train_labels,
                                                                                                   test_data,
                                                                                                   test_labels,
                                                                                                   channels, normalize)
        if isinstance(self.cp.pretrained_model, list) and len(self.cp.pretrained_model) > 1:
            cp_model_path = self.cp.pretrained_model[0]
            self.cp.net.load_model(cp_model_path, cpu=(not self.gpu))
        else:
            cp_model_path = self.cp.pretrained_model

        diam_train = np.array([utils.diameters(lbl)[0] for lbl in train_labels])
        if run_test:
            diam_test = np.array([utils.diameters(lbl)[0] for lbl in test_labels])

        nimg = len(train_data)
        styles = np.zeros((n_epochs * nimg, 256), np.float32)
        diams = np.zeros((n_epochs * nimg,), np.float32)
        tic = time.time()
        for iepoch in range(n_epochs):
            iall = np.arange(0, nimg, 1, int)
            for ibatch in range(0, nimg, batch_size):
                inds = iall[ibatch:ibatch + batch_size]
                imgi, lbl, scale = transforms.random_rotate_and_resize(
                    [train_data[i] for i in inds],
                    Y=[train_labels[i].astype(np.int16) for i in inds], scale_range=1, xy=(512, 512))
                feat = self.cp.network(imgi)[-1]
                styles[inds + nimg * iepoch] = feat
                diams[inds + nimg * iepoch] = np.log(diam_train[inds]) - np.log(self.diam_mean) + np.log(scale)
            del feat
            if (iepoch + 1) % 2 == 0:
                print('ran %d epochs in %0.3f sec' % (iepoch + 1, time.time() - tic))

        # create model
        smean = styles.mean(axis=0)
        X = ((styles - smean).T).copy()
        ymean = diams.mean()
        y = diams - ymean

        A = np.linalg.solve(X @ X.T + l2_regularization * np.eye(X.shape[0]), X @ y)
        ypred = A @ X
        print('train correlation: %0.4f' % np.corrcoef(y, ypred)[0, 1])

        if run_test:
            nimg_test = len(test_data)
            styles_test = np.zeros((nimg_test, 256), np.float32)
            for i in range(nimg_test):
                styles_test[i] = self.cp._run_net(test_data[i].transpose((1, 2, 0)))[1]
            diam_test_pred = np.exp(A @ (styles_test - smean).T + np.log(self.diam_mean) + ymean)
            diam_test_pred = np.maximum(5., diam_test_pred)
            print('test correlation: %0.4f' % np.corrcoef(diam_test, diam_test_pred)[0, 1])

        self.pretrained_size = cp_model_path + '_size.npy'
        self.params = {'A': A, 'smean': smean, 'diam_mean': self.diam_mean, 'ymean': ymean}
        np.save(self.pretrained_size, self.params)
        return self.params

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()