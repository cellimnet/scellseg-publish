import matplotlib as mpl
from scellseg import models, io
import os, time
from scellseg.core import use_gpu, UnetModel
from scellseg import models
from scellseg.contrast_learning.dataset import DatasetPairEval

mpl.rcParams['figure.dpi'] = 300

t0 = time.time()
# use_GPU = use_gpu()
use_GPU = True
channel = [2, 1]

# train_dir = r'E:\3-dataset\cellpose\train2'
# test_dir = r'E:\3-dataset\cellpose\val2'
# model_dir = r'E:\3-dataset\cellpose'
# pretrained_model = r'C:\Users\admin\.scellseg\models\cytotorch_0'

# 测试某个类别用一张训练
pretrained_model = False
train_dir = r'G:\Python\9-Project\1-flurSeg\scellseg\input\meta_eval\BBBC010_elegans\shot'  # 这里是用原cellpose函数进行微调的结果
model_dir = r'G:\Python\9-Project\1-flurSeg\scellseg\output\models\retrain'
# pretrained_model = r'C:\Users\admin\.scellseg\models\cytotorch_0'

model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021070913_3905'  # cellpose自训练的
# model_name = r'499_cellpose_residual_on_style_on_concatenation_off_cellpose_2021072123_5236'  # all,要开attn+dense
pretrained_model = os.path.join(r'G:\Python\9-Project\1-flurSeg\scellseg\output\models\scellstyle_train', model_name)

# task_mode： cellpose, hover, classic-3 (nbranch=1), classic-2 (nbranch=1)
task_mode = 'cellpose'
contrast_on =1
batch_size = 8

images, labels, files, test_images, test_labels, test_files = \
    io.load_train_test_data(train_dir, test_dir=None, image_filter='_img', mask_filter='_masks', task_mode=task_mode)

# 模型实例化, 3种特征处理方式
style_on = True
attn_on, dense_on=False, False
# if attn_on: style_on=False
# if dense_on: style_on=False

print(attn_on, dense_on, style_on)
model = models.sCellSeg(diam_mean=30, gpu=use_GPU, pretrained_model=pretrained_model, nclasses=3,
                        attn_on=attn_on, dense_on=dense_on, style_on=style_on,
                        use_branch=False, ndecoder=1,
                        task_mode=task_mode)
# model = UnetModel(diam_mean=30, gpu=use_GPU, attn=False, residual_on=True, style_on=True, concatenation=True)

model_dict = model.net.state_dict()

dataset_dir = r'G:\Python\9-Project\1-flurSeg\scellseg\input\meta_eval\BBBC010_elegans'
model.net.contrast_on = contrast_on
if contrast_on:
    model.net.pair_gen = DatasetPairEval(positive_dir=dataset_dir, gpu=True, rescale=True)

# 训练，给定数据
# cpmodel_path = model.train(train_data=images, train_labels=labels, train_files=files,
#                            test_data=test_images, test_labels=test_labels, test_files=test_files,
#                            channels=channel, save_path=model_dir, n_epochs=500)

cpmodel_path = model.train3(train_data=images, train_labels=labels, train_files=files,
                           test_data=test_images, test_labels=test_labels, test_files=test_files,
                           channels=channel, save_path=model_dir, learning_rate=0.1, save_every=500, rescale=True,
                           n_epochs=500, batch_size=batch_size)

print('>>>> model trained and saved to %s'%cpmodel_path)

t1 = time.time()
print('\033[1;32m>>>> Total Time:\033[0m', t1-t0, 's')