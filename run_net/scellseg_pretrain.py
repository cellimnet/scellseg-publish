from scellseg import models, io
from scellseg.utils import make_folder
from scellseg.utils import process_different_model
import os

use_GPU = True

train_dir = r'E:\3-dataset\cellpose\train-single-micro'  # Todo: you should change to your own dataset path
test_dir = r'E:\3-dataset\cellpose\val-single-micro'
project_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.path.sep + ".")
output_path = os.path.join(project_path, 'output')

model_name = 'scellseg'  # model_name： scellseg, cellpose, hover, unet3, unet2
task_mode, _, attn_on, dense_on, style_scale_on = process_different_model(model_name)

images, labels, files, test_images, test_labels, test_files = \
    io.load_train_test_data(train_dir, test_dir=test_dir, image_filter='_img', mask_filter='_masks', task_mode=task_mode)

model = models.sCellSeg(diam_mean=30, gpu=use_GPU, pretrained_model=False, nclasses=3,
                        attn_on=attn_on, dense_on=dense_on, style_on=True, style_scale_on=style_scale_on,
                        task_mode=task_mode)
cpmodel_path = model.pretrain(train_data=images, train_labels=labels, train_files=files,
                           test_data=test_images, test_labels=test_labels, test_files=test_files,
                           channels=[2, 1], save_path=output_path, n_epochs=100, learning_rate=0.2)

print('>>>> model trained and saved to %s'%cpmodel_path)
