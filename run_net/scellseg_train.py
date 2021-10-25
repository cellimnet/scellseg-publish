import matplotlib as mpl
from scellseg import models, io
from scellseg.core import use_gpu, UnetModel
from scellseg import models

mpl.rcParams['figure.dpi'] = 300

# use_GPU = use_gpu()
use_GPU = True
channel = [2, 1]

train_dir = r'E:\3-dataset\cellpose\train-cellpose'
test_dir = r'E:\3-dataset\cellpose\val-cellpose'
model_dir = r'E:\3-dataset\cellpose'
# pretrained_model = r'C:\Users\admin\.scellseg\models\cytotorch_0'

# 测试某个类别用一张训练
pretrained_model = False
# train_dir = r'G:\Python\9-Project\1-flurSeg\scellseg\input\meta_eval\2018DB_nuclei_2\shot'  # 这里是用原cellpose函数进行微调的结果
# model_dir = r'G:\Python\9-Project\1-flurSeg\scellseg\output\models\train'
# pretrained_model = r'C:\Users\admin\.scellseg\models\cytotorch_0'

# task_mode： cellpose, hover, classic-3 (nbranch=1), classic-2 (nbranch=1)
task_mode = 'cellpose'

images, labels, files, test_images, test_labels, test_files = \
    io.load_train_test_data(train_dir, test_dir=test_dir, image_filter='_img', mask_filter='_masks', task_mode=task_mode)

# 模型实例化, 3种特征处理方式
style_on = False
attn_on, dense_on=True, True
# if attn_on: style_on=False
# if dense_on: style_on=False

print(attn_on, dense_on, style_on)
model = models.sCellSeg(diam_mean=30, gpu=use_GPU, pretrained_model=pretrained_model, nclasses=3,
                        attn_on=attn_on, dense_on=dense_on, style_on=style_on,
                        use_branch=False, ndecoder=1,
                        task_mode=task_mode)
# model = UnetModel(diam_mean=30, gpu=use_GPU, attn=False, residual_on=True, style_on=True, concatenation=True)

model_dict = model.net.state_dict()

# 训练，给定数据
# cpmodel_path = model.train(train_data=images, train_labels=labels, train_files=files,
#                            test_data=test_images, test_labels=test_labels, test_files=test_files,
#                            channels=channel, save_path=model_dir, n_epochs=500)

cpmodel_path = model.train2(train_data=images, train_labels=labels, train_files=files,
                           test_data=test_images, test_labels=test_labels, test_files=test_files,
                           channels=channel, save_path=model_dir, n_epochs=500, learning_rate=0.2)

print('>>>> model trained and saved to %s'%cpmodel_path)
