# Scellseg 

A style-aware cell instance segmentation tool with pre-training and contrastive fine-tuning<img src="https://github.com/cellimnet/scellseg-publish/raw/main/logo.svg" width="200" title="scellseg" alt="scellseg" align="right" vspace = "50">

### **Description**

We proposed a "pre-trained + fine-tuning" pipeline for cell instance segmentation. To make Scellseg easy to use, we also developed a graphical user interface integrated with functions of annotation, fine-tuning and inference. Biologists can specialize their own cell segmentation model to conduct single-cell image analysis.

### Install

Operating system: It has been tested on Windows 10. Theoretically, it can work on any system that can run Python.

Programing language: Python.

Hardware: >= 8G memory, equipped with a CPU with Core i5 or above.

Our Environment: Python --3.7.4，CUDA --10.1.243， GPU：Nvidia 2080Ti

This project uses Numpy, Opencv, skimage, tqdm, pytorch, pyqt. Go check them out if you don't have them, you can install them with conda or pip.

### How to use GUI

#### **1. Annotation**

​	Besides the  basic function of Cellpose,

​	a) You can modify the mask of instance directly in pixel level without deleting it and drawing it from scratch. You can check "Edit mask" or [E],  in this mode, you need firstly select a mask you wanted to edit, the selected mask will be highlighted, use right-click to add pixels and Shift+right-click to delete pixels

​	b) You can also take an overall look at of masks you have labelled with a list for each image, each index corresponds to a instance, you can pitch on and add notes to it, besides, the list can be saved and loaded next time

​	c) Drag a image/mask or a folder is supported, for a image, we autoload its parent directory, for a mask, we autoload its corresponding image and its parent directory. You can use [ctrl+←/→]  to cycle through images in current directory

​	d) You can save the masks in ".png" format

#### 2. Fine-tuning

​	a) You should prepare your data in one folder with your experiment name like "mito-20211116", here we call it <b>parent folder</b>. Into this folder, it should contain a **shot subfolder** and a **query subfolder**. The shot subfolder contains images and the corresponding labelled masks, the query subfolder contains the images you want to segment. Into the shot subfolder, images should be named with **"\_img" suffix** and masks should be named as **"\_masks" suffix**. Except the suffix, the name of image and mask should be identical, for example, 001_img and 001_masks, notably, 001_masks should not be named as 001_cp_masks or 001_img_cp_masks (You should rename your masks name after annotation). Into the query subfolder, images should be named with **"\_img" suffix**

​	b) Click "Dataset path" to choose the parent folder of your dataset, such as "mito-20211116"

​	c) Set the channel you want to segment, you can also provide a chan2 like nuclei channel for better learning

​	d) Set the epoch you want conduct, the default value is 100, which was used in our paper. You can increase the number for adequate training

​	e) You can select different pre-trained model ("Scellseg", "Cellpose", or "Hover") and fine-tuning strategy ("contrastive fine-tuning" or "classic fine-tuning")

​	f) Click "Start fine-tuning" button to start fine-tuning. After fine-tuning, it will show the saved path of the model file in the bottom of display window

#### 3. Inference

​	a) There are two modes for inference,  (1) run segmentation for image in window (2) batch segmentation

​	b) If you want to conduct batch segmentation, click "Data path" to choose the parent folder of your dataset, such as "mito-20211116" 

​	c) You can choose your own model file for inference, the default is the pre-trained Scellseg model file

​	d) The default "model match threshold" is set to 0.4 and "cellprob threshold" is set to 0.5, which was used in our paper, you can change it for better performance

​	e) Set the channel you want to segment, you can also provide a chan2 like nuclei channel for better learning, you should set the same setting as fine-tuning process

​	f) You can get each instance image after inference, click "Data path" to choose the query folder of your dataset, such as "mito-20211116/query", the output files will be saved at a subfolder in parent folder named "single", mito-20211116/single"

### **Declaration**

Our pipeline is heavily based on [Cellpose](https://github.com/MouseLand/cellpose) and we also referred to the following projects:
Hover-Net: https://github.com/vqdang/hover_net
Attention-Unet: https://github.com/ozan-oktay/Attention-Gated-Networks

