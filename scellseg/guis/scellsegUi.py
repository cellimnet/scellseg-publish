# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cellPoseUI.ui'

import numpy as np
import sys, os, pathlib, warnings, datetime, tempfile, glob, time, threading

from natsort import natsorted
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
import pyqtgraph as pg

import cv2

from scellseg.guis import guiparts, iopart, menus, plot
from scellseg import models, utils, transforms, dynamics, dataset, io
from scellseg.dataset import DatasetShot, DatasetQuery
from scellseg.contrast_learning.dataset import DatasetPairEval
from skimage.measure import regionprops
from tqdm import trange
from math import floor, ceil

from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB = True
except:
    MATPLOTLIB = False


class Ui_MainWindow(QtGui.QMainWindow):
    """UI Widget Initialize and UI Layout Initialize,
       With any bug or problem, please do connact us from Github Issue"""
    def __init__(self, image=None):
        super(Ui_MainWindow, self).__init__()
        if image is not None:
            self.filename = image
            iopart._load_image(self, self.filename)
        self.now_pyfile_path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')

    def setupUi(self, MainWindow, image=None):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1420, 800)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(self.now_pyfile_path + "/assets/logo.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)

        menus.mainmenu(self)
        menus.editmenu(self)
        menus.helpmenu(self)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setContentsMargins(6, 6, 6, 6)
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")

        self.splitter2 = QtWidgets.QSplitter()
        self.splitter2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter2.setObjectName("splitter2")

        self.scrollArea = QtWidgets.QScrollArea(self.splitter)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        # self.scrollAreaWidgetContents.setFixedWidth(500)
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1500, 848))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")

        # self.TableModel = QtGui.QStandardItemModel(self.tableRow, self.tableCol)
        # self.TableModel.setHorizontalHeaderLabels(["INDEX", "NAME"])
        # self.TableView = QtGui.QTableView()
        # self.TableView.setModel(self.TableModel)

        self.mainLayout = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(0)
        self.mainLayout.setObjectName("mainLayout")

        self.previous_button = QtWidgets.QPushButton("previous image [Ctrl + ←]")
        self.load_folder = QtWidgets.QPushButton("load image folder ")
        self.next_button = QtWidgets.QPushButton("next image [Ctrl + →]")
        self.previous_button.setShortcut(Qt.QKeySequence.MoveToPreviousWord)
        self.next_button.setShortcut(Qt.QKeySequence.MoveToNextWord)
        self.mainLayout.addWidget(self.previous_button, 1, 1, 1, 1)
        self.mainLayout.addWidget(self.load_folder, 1, 2, 1, 1)
        self.mainLayout.addWidget(self.next_button, 1, 3, 1, 1)

        self.previous_button.clicked.connect(self.PreImBntClicked)
        self.next_button.clicked.connect(self.NextImBntClicked)
        self.load_folder.clicked.connect(self.OpenDirBntClicked)

        # leftside cell list widget
        self.listView = QtWidgets.QTableView()
        self.myCellList = []
        self.listmodel = Qt.QStandardItemModel(0,1)
        self.listmodel.setHorizontalHeaderLabels(["Annotation"])
        # self.listmodel.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem())
        self.listView.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignLeft)
        # self.listView.horizontalHeader().setStyle("background-color: #F0F0F0")
        # self.listView.horizontalHeader().setVisible(False)
        self.listView.verticalHeader().setVisible(False)
        for i in range(len(self.myCellList)):
            self.listmodel.setItem(i,Qt.QStandardItem(self.myCellList[i]))

        self.listView.horizontalHeader().setDefaultSectionSize(140)
        self.listView.setMaximumWidth(120)
        self.listView.setModel(self.listmodel)
        self.listView.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.listView.AdjustToContents
        self.listView.customContextMenuRequested.connect(self.show_menu)
        # self.listView.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.listView.clicked.connect(self.showChoosen)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.toolBox = QtWidgets.QToolBox(self.splitter)
        self.toolBox.setObjectName("toolBox")
        self.toolBox.setMaximumWidth(340)

        self.page = QtWidgets.QWidget()
        self.page.setFixedWidth(340)
        self.page.setObjectName("page")

        self.gridLayout = QtWidgets.QGridLayout(self.page)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")

        # cross-hair/Draw area
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.layer_off = False
        self.masksOn = True
        self.win = pg.GraphicsLayoutWidget()
        self.state_label = pg.LabelItem("Scellseg has been initialized!")
        self.win.addItem(self.state_label, 3, 0)

        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.win.scene().sigMouseMoved.connect(self.mouse_moved)
        self.make_viewbox()
        bwrmap = make_bwr()
        self.bwr = bwrmap.getLookupTable(start=0.0, stop=255.0, alpha=False)
        self.cmap = []
        # spectral colormap
        self.cmap.append(make_spectral().getLookupTable(start=0.0, stop=255.0, alpha=False))
        # single channel colormaps
        for i in range(3):
            self.cmap.append(make_cmap(i).getLookupTable(start=0.0, stop=255.0, alpha=False))

        if MATPLOTLIB:
            self.colormap = (plt.get_cmap('gist_ncar')(np.linspace(0.0, .9, 1000)) * 255).astype(np.uint8)
        else:
            self.colormap = ((np.random.rand(1000, 3) * 0.8 + 0.1) * 255).astype(np.uint8)

        self.is_stack = True  # always loading images of same FOV
        # if called with image, load it
        # if image is not None:
        #     self.filename = image
        #     iopart._load_image(self, self.filename)

        self.setAcceptDrops(True)
        self.win.show()
        self.show()

        self.splitter2.addWidget(self.listView)
        self.splitter2.addWidget(self.win)
        self.mainLayout.addWidget(self.splitter2,0,1,1,3)

        self.label_2 = QtWidgets.QLabel(self.page)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 7, 0, 1, 1)

        self.brush_size = 3
        self.BrushChoose = QtWidgets.QComboBox()
        self.BrushChoose.addItems(["1", "3", "5", "7", "9", "11", "13", "15", "17", "19"])
        self.BrushChoose.currentIndexChanged.connect(self.brush_choose)
        self.gridLayout.addWidget(self.BrushChoose, 7, 1, 1, 1)

        # turn on single stroke mode
        self.sstroke_On = True
        self.SSCheckBox = QtWidgets.QCheckBox(self.page)
        self.SSCheckBox.setObjectName("SSCheckBox")
        self.SSCheckBox.setChecked(True)
        self.SSCheckBox.toggled.connect(self.toggle_sstroke)
        self.gridLayout.addWidget(self.SSCheckBox, 8, 0, 1, 1)

        self.eraser_button = QtWidgets.QCheckBox(self.page)
        self.eraser_button.setObjectName("Edit mask")
        self.eraser_button.setChecked(False)
        self.eraser_button.toggled.connect(self.eraser_model_change)
        self.eraser_button.setToolTip("Right-click to add pixels\nShift+Right-click to delete pixels")
        self.gridLayout.addWidget(self.eraser_button, 9, 0, 1, 1)

        self.CHCheckBox = QtWidgets.QCheckBox(self.page)
        self.CHCheckBox.setObjectName("CHCheckBox")
        self.CHCheckBox.toggled.connect(self.cross_hairs)
        self.gridLayout.addWidget(self.CHCheckBox, 10, 0, 1, 1)

        self.MCheckBox = QtWidgets.QCheckBox(self.page)
        self.MCheckBox.setChecked(True)
        self.MCheckBox.setObjectName("MCheckBox")
        self.MCheckBox.setChecked(True)
        self.MCheckBox.toggled.connect(self.toggle_masks)
        self.gridLayout.addWidget(self.MCheckBox, 11, 0, 1, 1)

        self.OCheckBox = QtWidgets.QCheckBox(self.page)
        self.outlinesOn = True
        self.OCheckBox.setChecked(True)
        self.OCheckBox.setObjectName("OCheckBox")
        self.OCheckBox.toggled.connect(self.toggle_masks)
        self.gridLayout.addWidget(self.OCheckBox, 12, 0, 1, 1)

        self.scale_on = True
        self.SCheckBox = QtWidgets.QCheckBox(self.page)
        self.SCheckBox.setObjectName("SCheckBox")
        self.SCheckBox.setChecked(True)
        self.SCheckBox.toggled.connect(self.toggle_scale)
        self.gridLayout.addWidget(self.SCheckBox, 13, 0, 1, 1)

        self.autosaveOn = True
        self.ASCheckBox = QtWidgets.QCheckBox(self.page)
        self.ASCheckBox.setObjectName("ASCheckBox")
        self.ASCheckBox.setChecked(True)
        self.ASCheckBox.toggled.connect(self.toggle_autosave)
        self.ASCheckBox.setToolTip("If ON, masks/npy/list will be autosaved")
        self.gridLayout.addWidget(self.ASCheckBox, 14, 0, 1, 1)

        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 15, 0, 1, 2)
        # self.eraser_combobox = QtWidgets.QComboBox()
        # self.eraser_combobox.addItems(["Pixal delete", "Pixal add"])
        # self.gridLayout.addWidget(self.eraser_combobox, 8, 1, 1, 1)

        self.RGBChoose = guiparts.RGBRadioButtons(self, 3, 1)
        self.RGBDropDown = QtGui.QComboBox()
        self.RGBDropDown.addItems(["rgb", "gray", "spectral", "red", "green", "blue"])
        self.RGBDropDown.currentIndexChanged.connect(self.color_choose)
        self.gridLayout.addWidget(self.RGBDropDown, 3, 0, 1, 1)

        self.saturation_label = QtWidgets.QLabel("Saturation")
        self.gridLayout.addWidget(self.saturation_label, 0, 0, 1, 1)

        self.autobtn = QtGui.QCheckBox('Auto-adjust')
        self.autobtn.setChecked(True)
        self.autobtn.toggled.connect(self.toggle_autosaturation)
        self.gridLayout.addWidget(self.autobtn, 0, 1, 1, 1)

        self.currentZ = 0
        self.zpos = QtGui.QLineEdit()
        self.zpos.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.zpos.setText(str(self.currentZ))
        self.zpos.returnPressed.connect(self.compute_scale)
        self.zpos.setFixedWidth(20)
        # self.gridLayout.addWidget(self.zpos, 0, 2, 1, 1)

        self.slider = guiparts.RangeSlider(self)
        self.slider.setMaximum(255)
        self.slider.setMinimum(0)
        self.slider.setHigh(255)
        self.slider.setLow(0)
        self.gridLayout.addWidget(self.slider, 2, 0, 1, 4)
        self.slider.setObjectName("rangeslider")

        self.page_2 = QtWidgets.QWidget()
        self.page_2.setFixedWidth(340)
        self.page_2.setObjectName("page_2")

        self.gridLayout_2 = QtWidgets.QGridLayout(self.page_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")

        page2_l = 0
        self.useGPU = QtWidgets.QCheckBox(self.page_2)
        self.useGPU.setObjectName("useGPU")
        self.gridLayout_2.addWidget(self.useGPU, page2_l, 0, 1, 1)
        self.check_gpu()

        page2_l += 1
        self.label_4 = QtWidgets.QLabel(self.page_2)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, page2_l, 0, 1, 1)
        self.ModelChoose = QtWidgets.QComboBox(self.page_2)
        self.ModelChoose.setObjectName("ModelChoose")
        self.project_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + os.path.sep + ".")
        self.model_dir = os.path.join(self.project_path, 'assets', 'pretrained_models')
        print('self.model_dir', self.model_dir)

        self.ModelChoose.addItem("")
        self.ModelChoose.addItem("")
        self.ModelChoose.addItem("")
        self.gridLayout_2.addWidget(self.ModelChoose, page2_l, 1, 1, 1)

        page2_l += 1
        self.label_5 = QtWidgets.QLabel(self.page_2)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, page2_l, 0, 1, 1)
        self.jCBChanToSegment = QtWidgets.QComboBox(self.page_2)
        self.jCBChanToSegment.setObjectName("jCBChanToSegment")
        self.jCBChanToSegment.addItems(["gray", "red", "green", "blue"])
        self.jCBChanToSegment.setCurrentIndex(0)
        self.gridLayout_2.addWidget(self.jCBChanToSegment, page2_l, 1, 1, 1)

        page2_l += 1
        self.label_6 = QtWidgets.QLabel(self.page_2)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, page2_l, 0, 1, 1)
        self.jCBChan2 = QtWidgets.QComboBox(self.page_2)
        self.jCBChan2.setObjectName("jCBChan2")
        self.jCBChan2.addItems(["none", "red", "green", "blue"])
        self.jCBChan2.setCurrentIndex(0)
        self.gridLayout_2.addWidget(self.jCBChan2, page2_l, 1, 1, 1)

        page2_l += 1
        self.model_choose_btn = QtWidgets.QPushButton("Model file")
        self.model_choose_btn.clicked.connect(self.model_file_dir_choose)
        self.gridLayout_2.addWidget(self.model_choose_btn, page2_l, 0, 1, 1)
        self.model_choose_btn = QtWidgets.QPushButton("Reset pre-trained")
        self.model_choose_btn.clicked.connect(self.reset_pretrain_model)
        self.gridLayout_2.addWidget(self.model_choose_btn, page2_l, 1, 1, 1)

        page2_l += 1
        self.label_null = QtWidgets.QLabel("")
        self.gridLayout_2.addWidget(self.label_null, page2_l, 0, 1, 1)
        slider_image_path = self.now_pyfile_path + '/assets/slider_handle.png'
        self.sliderSheet = [
        'QSlider::groove:vertical {',
        'background-color: #D3D3D3;',
        'position: absolute;',
        'left: 4px; right: 4px;',
        '}',
        '',
        'QSlider::groove:horizontal{',
        'background-color:#D3D3D3;',
        'position: absolute;',
        'top: 4px; bottom: 4px;',
        '}',
        '',
        'QSlider::handle:vertical {',
        'height: 10px;',
        'background-color: {0:s};'.format('#A9A9A9'),
        'margin: 0 -4px;',
        '}',

        '',
        'QSlider::handle:horizontal{',
        'width: 10px;',
        'border-image: url({0:s});'.format(slider_image_path),
        'margin: -4px 0px -4px 0px;',
        '}',
        'QSlider::sub-page:horizontal',
        '{',
        'background-color: {0:s};'.format('#A9A9A9'),
        '}',

        '',
        'QSlider::add-page {',
        'background-color: {0:s};'.format('#D3D3D3'),
        '}',
        '',
        'QSlider::sub-page {',
        'background-color: {0:s};'.format('#D3D3D3'),
        '}',

]
        page2_l += 1
        self.label_seg = QtWidgets.QLabel("Run seg for image in window")
        self.gridLayout_2.addWidget(self.label_seg, page2_l, 0, 1, 4)
        self.label_seg.setObjectName('label_seg')

        page2_l += 1
        self.label_3 = QtWidgets.QLabel(self.page_2)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, page2_l, 0, 1, 4)

        page2_l += 1
        self.prev_selected = 0
        self.diameter = 30
        # self.Diameter = QtWidgets.QSpinBox(self.page_2)
        self.Diameter = QtWidgets.QLineEdit(self.page_2)
        self.Diameter.setObjectName("Diameter")
        self.Diameter.setText(str(self.diameter))
        self.Diameter.setFixedWidth(100)
        self.Diameter.editingFinished.connect(self.compute_scale)
        self.gridLayout_2.addWidget(self.Diameter, page2_l, 0, 1, 2)
        self.SizeButton = QtWidgets.QPushButton(self.page_2)
        self.SizeButton.setObjectName("SizeButton")
        self.gridLayout_2.addWidget(self.SizeButton, page2_l, 1, 1, 1)
        self.SizeButton.clicked.connect(self.calibrate_size)
        self.SizeButton.setEnabled(False)

        page2_l += 1
        self.label_mode = QtWidgets.QLabel("Inference mode")
        self.gridLayout_2.addWidget(self.label_mode, page2_l, 0, 1, 1)
        self.NetAvg = QtWidgets.QComboBox(self.page_2)
        self.NetAvg.setObjectName("NetAvg")
        self.NetAvg.addItems(["run 1 net (fast)", "+ resample (slow)"])
        self.gridLayout_2.addWidget(self.NetAvg, page2_l, 1, 1, 1)
        page2_l += 1
        self.invert = QtWidgets.QCheckBox(self.page_2)
        self.invert.setObjectName("invert")
        self.gridLayout_2.addWidget(self.invert, page2_l, 0, 1, 1)

        page2_l += 1
        self.ModelButton = QtWidgets.QPushButton(' Run segmentation ')
        self.ModelButton.setObjectName("runsegbtn")
        self.ModelButton.clicked.connect(self.compute_model)
        self.gridLayout_2.addWidget(self.ModelButton, page2_l, 0, 1, 2)
        self.ModelButton.setEnabled(False)

        page2_l += 1
        self.label_7 = QtWidgets.QLabel(self.page_2)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, page2_l, 0, 1, 1)
        self.threshold = 0.4
        self.threshslider = QtWidgets.QSlider(self.page_2)
        self.threshslider.setOrientation(QtCore.Qt.Horizontal)
        self.threshslider.setObjectName("threshslider")
        self.threshslider.setMinimum(1.0)
        self.threshslider.setMaximum(30.0)
        self.threshslider.setValue(31 - 4)
        self.threshslider.valueChanged.connect(self.compute_cprob)
        self.threshslider.setEnabled(False)
        self.threshslider.setStyleSheet('\n'.join(self.sliderSheet))
        self.gridLayout_2.addWidget(self.threshslider, page2_l, 1, 1, 1)
        self.threshslider.setToolTip("Value: " + str(self.threshold))

        page2_l += 1
        self.label_8 = QtWidgets.QLabel(self.page_2)
        self.label_8.setObjectName("label_8")
        self.gridLayout_2.addWidget(self.label_8, page2_l, 0, 1, 1)
        self.probslider = QtWidgets.QSlider(self.page_2)
        self.probslider.setOrientation(QtCore.Qt.Horizontal)
        self.probslider.setObjectName("probslider")
        self.probslider.setStyleSheet('\n'.join(self.sliderSheet))
        self.gridLayout_2.addWidget(self.probslider, page2_l, 1, 1, 1)
        self.probslider.setMinimum(-6.0)
        self.probslider.setMaximum(6.0)
        self.probslider.setValue(0.0)
        self.cellprob = 0.5
        self.probslider.valueChanged.connect(self.compute_cprob)
        self.probslider.setEnabled(False)
        self.probslider.setToolTip("Value: " + str(self.cellprob))

        page2_l += 1
        self.label_batchseg = QtWidgets.QLabel("Batch segmentation")
        self.label_batchseg.setObjectName('label_batchseg')
        self.gridLayout_2.addWidget(self.label_batchseg, page2_l, 0, 1, 4)
        page2_l += 1
        self.label_bz = QtWidgets.QLabel("Batch size")
        self.gridLayout_2.addWidget(self.label_bz, page2_l, 0, 1, 1)
        self.bz_line = QtWidgets.QLineEdit()
        self.bz_line.setPlaceholderText('Default: 8')
        self.bz_line.setFixedWidth(120)
        self.gridLayout_2.addWidget(self.bz_line, page2_l, 1, 1, 1)
        page2_l += 1
        self.dataset_inference_bnt = QtWidgets.QPushButton("Data path")
        self.gridLayout_2.addWidget(self.dataset_inference_bnt, page2_l, 0, 1, 1)
        self.dataset_inference_bnt.clicked.connect(self.batch_inference_dir_choose)
        self.batch_inference_bnt = QtWidgets.QPushButton("Run batch")
        self.batch_inference_bnt.setObjectName("binferbnt")
        self.batch_inference_bnt.clicked.connect(self.batch_inference)
        self.gridLayout_2.addWidget(self.batch_inference_bnt, page2_l, 1, 1, 1)
        self.batch_inference_bnt.setEnabled(False)

        page2_l += 1
        self.label_getsingle = QtWidgets.QLabel("Get single instance")
        self.label_getsingle.setObjectName('label_getsingle')
        self.gridLayout_2.addWidget(self.label_getsingle, page2_l,0,1,2)
        page2_l += 1
        self.single_dir_bnt = QtWidgets.QPushButton("Data path")
        self.single_dir_bnt.clicked.connect(self.single_dir_choose)
        self.gridLayout_2.addWidget(self.single_dir_bnt, page2_l,0,1,1)
        self.single_cell_btn = QtWidgets.QPushButton("Run batch")
        self.single_cell_btn.setObjectName('single_cell_btn')
        self.single_cell_btn.clicked.connect(self.get_single_cell)
        self.gridLayout_2.addWidget(self.single_cell_btn, page2_l,1,1,1)
        self.single_cell_btn.setEnabled(False)

        page2_l += 1
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem2, page2_l, 0, 1, 2)

        self.page_3 = QtWidgets.QWidget()
        self.page_3.setFixedWidth(340)
        self.page_3.setObjectName("page_3")

        self.progress = QtWidgets.QProgressBar()
        self.progress.setProperty("value", 0)
        self.progress.setAlignment(QtCore.Qt.AlignCenter)
        self.progress.setObjectName("progress")

        self.gridLayout_3 = QtWidgets.QGridLayout(self.page_3)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")

        self.ftuseGPU = QtWidgets.QCheckBox("Use GPU")
        self.ftuseGPU.setObjectName("ftuseGPU")
        self.gridLayout_3.addWidget(self.ftuseGPU, 0, 0, 1, 2)
        self.check_ftgpu()
        self.ftdirbtn = QtWidgets.QPushButton("Dataset path")
        self.ftdirbtn.clicked.connect(self.fine_tune_dir_choose)
        self.gridLayout_3.addWidget(self.ftdirbtn, 0, 2, 1, 2)

        self.label_10 = QtWidgets.QLabel("Model architecture")
        self.gridLayout_3.addWidget(self.label_10, 1, 0, 1, 2)

        self.ftmodelchooseBnt = QtWidgets.QComboBox()
        self.ftmodelchooseBnt.addItems(["scellseg", "cellpose", "hover"])
        self.gridLayout_3.addWidget(self.ftmodelchooseBnt, 1, 2, 1, 2)

        self.label_11 = QtWidgets.QLabel("Chan to segment")
        self.gridLayout_3.addWidget(self.label_11, 2, 0, 1, 2)

        self.chan1chooseBnt = QtWidgets.QComboBox()
        self.chan1chooseBnt.addItems(["gray", "red", "green", "blue"])
        self.chan1chooseBnt.setCurrentIndex(0)
        self.gridLayout_3.addWidget(self.chan1chooseBnt, 2, 2, 1, 2)

        self.label_12 = QtWidgets.QLabel("Chan2 (optional)")
        self.gridLayout_3.addWidget(self.label_12, 3, 0, 1, 2)

        self.chan2chooseBnt = QtWidgets.QComboBox()
        self.chan2chooseBnt.addItems(["none", "red", "green", "blue"])
        self.chan2chooseBnt.setCurrentIndex(0)
        self.gridLayout_3.addWidget(self.chan2chooseBnt, 3, 2, 1, 2)

        self.label_13 = QtWidgets.QLabel("Fine-tune strategy")
        self.gridLayout_3.addWidget(self.label_13, 4, 0, 1, 2)

        self.stmodelchooseBnt = QtWidgets.QComboBox()
        self.stmodelchooseBnt.addItems(["contrastive", "classic"])
        self.gridLayout_3.addWidget(self.stmodelchooseBnt, 4, 2, 1, 2)

        self.label_14 = QtWidgets.QLabel("Epoch")
        self.gridLayout_3.addWidget(self.label_14, 5, 0, 1, 2)
        self.epoch_line = QtWidgets.QLineEdit()
        self.epoch_line.setPlaceholderText('Default: 100')
        self.gridLayout_3.addWidget(self.epoch_line, 5, 2, 1, 2)

        self.label_ftbz = QtWidgets.QLabel("Batch size")
        self.gridLayout_3.addWidget(self.label_ftbz, 6, 0, 1, 2)
        self.ftbz_line = QtWidgets.QLineEdit()
        self.ftbz_line.setPlaceholderText('Default: 8')
        self.gridLayout_3.addWidget(self.ftbz_line, 6, 2, 1, 2)

        self.ftbnt = QtWidgets.QPushButton("Start fine-tuning")
        self.ftbnt.setObjectName('ftbnt')
        self.ftbnt.clicked.connect(self.fine_tune)
        self.gridLayout_3.addWidget(self.ftbnt, 7, 0, 1, 4)
        self.ftbnt.setEnabled(False)

        spacerItem3 = QtWidgets.QSpacerItem(20, 320, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem3, 8, 0, 1, 1)

        #initialize scroll size
        self.scroll = QtGui.QScrollBar(QtCore.Qt.Horizontal)
        # self.scroll.setMaximum(10)
        # self.scroll.valueChanged.connect(self.move_in_Z)
        # self.gridLayout_3.addWidget(self.scroll)

        spacerItem2 = QtWidgets.QSpacerItem(20, 320, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem2)
        self.toolBox.addItem(self.page, "")
        self.toolBox.addItem(self.page_3, "")
        self.toolBox.addItem(self.page_2, "")
        self.verticalLayout_2.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.toolBox.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.centralwidget.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.reset()

    def show_menu(self, point):
        # print(point.x())
        # item = self.listView.itemAt(point)
        # print(item)
        temp_cell_idx = self.listView.rowAt(point.y())

        self.list_select_cell(temp_cell_idx+1)

        # print(self.myCellList[temp_cell_idx])
        if self.listView.rowAt(point.y()) >= 0:
            self.contextMenu = QtWidgets.QMenu()
            self.actionA = QtGui.QAction("Delete this cell", self)
            self.actionB = QtGui.QAction("Edit this cell", self)

            self.contextMenu.addAction(self.actionA)
            self.contextMenu.addAction(self.actionB)
            self.contextMenu.popup(QtGui.QCursor.pos())

            self.actionA.triggered.connect(lambda: self.remove_cell(temp_cell_idx + 1))
            self.actionB.triggered.connect(lambda: self.edit_cell(temp_cell_idx + 1))

            self.contextMenu.show()


    def edit_cell(self, index):
        self.select_cell(index)
        self.eraser_button.setChecked(True)
        self.toolBox.setCurrentIndex(0)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Scellseg"))
        self.CHCheckBox.setText(_translate("MainWindow", "Crosshair on [C]"))
        self.MCheckBox.setText(_translate("MainWindow", "Masks on [X]"))
        self.label_2.setText(_translate("MainWindow", "Brush size"))
        self.OCheckBox.setText(_translate("MainWindow", "Outlines on [Z]"))
        # self.ServerButton.setText(_translate("MainWindow", "send manual seg. to server"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page), _translate("MainWindow", "View and Draw"))
        self.SizeButton.setText(_translate("MainWindow", "Calibrate diam"))
        self.label_3.setText(_translate("MainWindow", "Cell diameter (pixels):"))

        self.useGPU.setText(_translate("MainWindow", "Use GPU"))
        self.SCheckBox.setText(_translate("MainWindow", "Scale disk on [S]"))
        self.ASCheckBox.setText(_translate("MainWindow", "Autosave [P]"))
        self.SSCheckBox.setText(_translate("MainWindow", "Single stroke"))
        self.eraser_button.setText(_translate("MainWindow", "Edit mask [E]"))
        self.ModelChoose.setItemText(0, _translate("MainWindow", "scellseg"))
        self.ModelChoose.setItemText(1, _translate("MainWindow", "cellpose"))
        self.ModelChoose.setItemText(2, _translate("MainWindow", "hover"))

        self.invert.setText(_translate("MainWindow", "Invert grayscale"))
        self.label_4.setText(_translate("MainWindow", "Model architecture"))
        self.label_5.setText(_translate("MainWindow", "Chan to segment"))
        self.label_6.setText(_translate("MainWindow", "Chan2 (optional)"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2), _translate("MainWindow", "Inference"))
        self.label_7.setText(_translate("MainWindow", "Model match TH"))
        self.label_8.setText(_translate("MainWindow", "Cell prob TH"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_3), _translate("MainWindow", "Fine-tune"))
        # self.menuFile.setTitle(_translate("MainWindow", "File"))
        # self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        # self.menuHelp.setTitle(_translate("MainWindow", "Help"))

        self.ImFolder = ''
        self.ImNameSet = []
        self.CurImId = 0

        self.CurFolder = os.getcwd()
        self.DefaultImFolder = self.CurFolder

    def setWinTop(self):
        print('get')

    def OpenDirDropped(self, curFile=None):
        # dir dropped callback func
        if self.ImFolder != '':
            self.ImNameSet = []
            self.ImNameRowSet = os.listdir(self.ImFolder)
            # print(self.ImNameRowSet)
            for tmp in self.ImNameRowSet:
                ext = os.path.splitext(tmp)[-1]
                if ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.jfif'] and '_mask' not in tmp:
                    self.ImNameSet.append(tmp)
            self.ImNameSet.sort()
            self.ImPath = self.ImFolder + r'/' + self.ImNameSet[0]
            ImNameSetNosuffix = [os.path.splitext(imNameSeti)[0] for imNameSeti in self.ImNameSet]
            # pix = QtGui.QPixmap(self.ImPath)
            # self.ImShowLabel.setPixmap(pix)
            if curFile is not None:
                curFile = os.path.splitext(curFile)[0]
                try:
                    self.CurImId = ImNameSetNosuffix.index(curFile)
                    print(self.CurImId)
                except:
                    curFile = curFile.replace('_cp_masks', '')
                    curFile = curFile.replace('_masks', '')
                    self.CurImId = ImNameSetNosuffix.index(curFile)
                    print(self.CurImId)
                    return
                    # self.state_label.setText("", color='#FF6A56')
            else:
                self.CurImId = 0
                iopart._load_image(self, filename=self.ImPath)
                self.initialize_listView()

        else:
            print('Please Find Another File Folder')

    def OpenDirBntClicked(self):
        # dir choosing callback function
        self.ImFolder = QtWidgets.QFileDialog.getExistingDirectory(None, "select folder", self.DefaultImFolder)
        if self.ImFolder != '':
            self.ImNameSet = []
            self.ImNameRowSet = os.listdir(self.ImFolder)
            # print(self.ImNameRowSet)
            for tmp in self.ImNameRowSet:
                ext = os.path.splitext(tmp)[-1]
                if ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.jfif'] and '_mask' not in tmp:
                    self.ImNameSet.append(tmp)
            self.ImNameSet.sort()
            print(self.ImNameSet)
            self.ImPath = self.ImFolder + r'/' + self.ImNameSet[0]
            # pix = QtGui.QPixmap(self.ImPath)
            # self.ImShowLabel.setPixmap(pix)
            self.CurImId = 0
            iopart._load_image(self, filename=self.ImPath)
            self.initialize_listView()

        else:
            print('Please Find Another File Folder')

    def PreImBntClicked(self):
        self.auto_save()
        # show previous image
        self.ImFolder = self.ImFolder
        self.ImNameSet = self.ImNameSet
        self.CurImId = self.CurImId
        self.ImNum = len(self.ImNameSet)
        print(self.ImFolder, self.ImNameSet)
        self.CurImId = self.CurImId - 1

        if self.CurImId >= 0:  # 第一张图片没有前一张
            self.ImPath = self.ImFolder + r'/' + self.ImNameSet[self.CurImId]
            iopart._load_image(self, filename=self.ImPath)
            self.initialize_listView()

        if self.CurImId < 0:
            self.CurImId = 0
            self.state_label.setText("This is the first image", color='#FF6A56')

    def NextImBntClicked(self):
        self.auto_save()

        # show next image
        self.ImFolder = self.ImFolder
        self.ImNameSet = self.ImNameSet
        self.CurImId = self.CurImId
        self.ImNum = len(self.ImNameSet)
        if self.CurImId < self.ImNum - 1:
            self.ImPath = self.ImFolder + r'/' + self.ImNameSet[self.CurImId + 1]
            iopart._load_image(self, filename=self.ImPath)
            self.initialize_listView()
            self.CurImId = self.CurImId + 1
        else:
            self.state_label.setText("This is the last image", color='#FF6A56')

    def eraser_model_change(self):
        if self.eraser_button.isChecked() == True:
            self.outlinesOn = False
            self.OCheckBox.setChecked(False)
            # self.OCheckBox.setEnabled(False)
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)
            # self.cur_size = self.brush_size * 6
            # cursor = Qt.QPixmap("./assets/eraser.png")
            # cursor_scaled = cursor.scaled(self.cur_size, self.cur_size)
            # cursor_set = Qt.QCursor(cursor_scaled, self.cur_size/2, self.cur_size/2)
            # QtWidgets.QApplication.setOverrideCursor(cursor_set)
            self.update_plot()

        else:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)

    def showChoosen(self, item):
        temp_cell_idx = int(item.row())
        self.list_select_cell(int(temp_cell_idx) + 1)

    def save_cell_list(self):
        self.listView.selectAll()
        self.myCellList = []
        for item in self.listView.selectedIndexes():
            data = item.data()
            self.myCellList.append(data)
        self.cell_list_name = os.path.splitext(self.filename)[0] + "_instance_list.txt"
        np.savetxt(self.cell_list_name, np.array(self.myCellList), fmt="%s")
        self.listView.clearSelection()

    def save_cell_list_menu(self):
        self.listView.selectAll()
        self.myCellList = []
        for item in self.listView.selectedIndexes():
            data = item.data()
            self.myCellList.append(data)
        self.cell_list_name = os.path.splitext(self.filename)[0] + "_instance_list.txt"
        np.savetxt(self.cell_list_name, np.array(self.myCellList), fmt="%s")
        self.state_label.setText("Saved outlines", color='#39B54A')
        self.listView.clearSelection()

    def help_window(self):
        HW = guiparts.HelpWindow(self)
        HW.show()

    def gui_window(self):
        EG = guiparts.ExampleGUI(self)
        EG.show()

    def toggle_autosave(self):
        if self.ASCheckBox.isChecked():
            self.autosaveOn = True
        else:
            self.autosaveOn = False
        print('self.autosaveOn', self.autosaveOn)

    def toggle_sstroke(self):
        if self.SSCheckBox.isChecked():
            self.sstroke_On = True
        else:
            self.sstroke_On = False
        print('self.sstroke_On', self.sstroke_On)

    def toggle_autosaturation(self):
        if self.autobtn.isChecked():
            self.compute_saturation()
            self.update_plot()

    def cross_hairs(self):
        if self.CHCheckBox.isChecked():
            self.p0.addItem(self.vLine, ignoreBounds=True)
            self.p0.addItem(self.hLine, ignoreBounds=True)
        else:
            self.p0.removeItem(self.vLine)
            self.p0.removeItem(self.hLine)

    def plot_clicked(self, event):
        if event.double():
            if event.button() == QtCore.Qt.LeftButton:
                print("will initialize the range")
                if (event.modifiers() != QtCore.Qt.ShiftModifier and
                    event.modifiers() != QtCore.Qt.AltModifier):
                    try:
                        self.p0.setYRange(0,self.Ly+self.pr)
                    except:
                        self.p0.setYRange(0,self.Ly)
                    self.p0.setXRange(0,self.Lx)

    def mouse_moved(self, pos):
        # print('moved')
        items = self.win.scene().items(pos)
        for x in items:
            if x == self.p0:
                mousePoint = self.p0.mapSceneToView(pos)
                if self.CHCheckBox.isChecked():
                    self.vLine.setPos(mousePoint.x())
                    self.hLine.setPos(mousePoint.y())
            # else:
            #    QtWidgets.QApplication.restoreOverrideCursor()
            # QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.DefaultCursor)

    def color_choose(self):
        self.color = self.RGBDropDown.currentIndex()
        self.view = 0
        self.RGBChoose.button(self.view).setChecked(True)
        self.update_plot()

    def update_ztext(self):
        zpos = self.currentZ
        try:
            zpos = int(self.zpos.text())
        except:
            print('ERROR: zposition is not a number')
        self.currentZ = max(0, min(self.NZ - 1, zpos))
        self.zpos.setText(str(self.currentZ))
        self.scroll.setValue(self.currentZ)

    def calibrate_size(self):
        model_type = self.ModelChoose.currentText()
        pretrained_model = os.path.join(self.model_dir, model_type)
        self.initialize_model(pretrained_model=pretrained_model, gpu=self.useGPU.isChecked(),
                              model_type=model_type)
        diams, _ = self.model.sz.eval(self.stack[self.currentZ].copy(), invert=self.invert.isChecked(),
                                      channels=self.get_channels(), progress=self.progress)
        diams = np.maximum(5.0, diams)
        print('estimated diameter of cells using %s model = %0.1f pixels' %
              (self.current_model, diams))
        self.state_label.setText('Estimated diameter of cells using %s model = %0.1f pixels' %
              (self.current_model, diams), color='#969696')
        self.Diameter.setText('%0.1f'%diams)
        self.diameter = diams
        self.compute_scale()
        self.progress.setValue(100)

    def enable_buttons(self):
        # self.X2Up.setEnabled(True)
        # self.X2Down.setEnabled(True)
        self.ModelButton.setEnabled(True)
        self.SizeButton.setEnabled(True)

        self.saveSet.setEnabled(True)
        self.savePNG.setEnabled(True)
        self.saveOutlines.setEnabled(True)
        self.saveCellList.setEnabled(True)
        self.saveAll.setEnabled(True)

        self.loadMasks.setEnabled(True)
        self.loadManual.setEnabled(True)
        self.loadCellList.setEnabled(True)

        self.toggle_mask_ops()

        self.update_plot()
        self.setWindowTitle('Scellseg @ ' + self.filename)

    def add_set(self):
        if len(self.current_point_set) > 0:
            # print(self.current_point_set)
            # print(np.array(self.current_point_set).shape)
            self.current_point_set = np.array(self.current_point_set)
            while len(self.strokes) > 0:
                self.remove_stroke(delete_points=False)
            if len(self.current_point_set) > 8:
                col_rand = np.random.randint(1000)
                color = self.colormap[col_rand, :3]
                median = self.add_mask(points=self.current_point_set, color=color)
                if median is not None:
                    self.removed_cell = []
                    self.toggle_mask_ops()
                    self.cellcolors.append(color)
                    self.ncells += 1
                    self.add_list_item()
                    self.ismanual = np.append(self.ismanual, True)
                    # if self.NZ == 1:
                    #     # only save after each cell if single image
                    #     iopart._save_sets(self)

            self.current_stroke = []
            self.strokes = []
            self.current_point_set = []
            self.update_plot()

    def add_mask(self, points=None, color=None):
        # loop over z values
        median = []
        if points.shape[1] < 3:
            points = np.concatenate((np.zeros((points.shape[0], 1), np.int32), points), axis=1)

        zdraw = np.unique(points[:, 0])
        zrange = np.arange(zdraw.min(), zdraw.max() + 1, 1, int)
        zmin = zdraw.min()
        pix = np.zeros((2, 0), np.uint16)
        mall = np.zeros((len(zrange), self.Ly, self.Lx), np.bool)
        k = 0
        for z in zdraw:
            iz = points[:, 0] == z
            vr = points[iz, 1]
            vc = points[iz, 2]
            # get points inside drawn points
            mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), np.uint8)
            pts = np.stack((vc - vc.min() + 2, vr - vr.min() + 2), axis=-1)[:, np.newaxis, :]
            mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
            ar, ac = np.nonzero(mask)
            ar, ac = ar + vr.min() - 2, ac + vc.min() - 2
            # get dense outline
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = contours[-2][0].squeeze().T
            vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
            # concatenate all points
            ar, ac = np.hstack((np.vstack((vr, vc)), np.vstack((ar, ac))))
            # if these pixels are overlapping with another cell, reassign them
            ioverlap = self.cellpix[z][ar, ac] > 0
            if (~ioverlap).sum() < 8:
                print('ERROR: cell too small without overlaps, not drawn')
                return None
            elif ioverlap.sum() > 0:
                ar, ac = ar[~ioverlap], ac[~ioverlap]
                # compute outline of new mask
                mask = np.zeros((np.ptp(ar) + 4, np.ptp(ac) + 4), np.uint8)
                mask[ar - ar.min() + 2, ac - ac.min() + 2] = 1
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pvc, pvr = contours[-2][0].squeeze().T
                vr, vc = pvr + ar.min() - 2, pvc + ac.min() - 2
            self.draw_mask(z, ar, ac, vr, vc, color)

            median.append(np.array([np.median(ar), np.median(ac)]))
            mall[z - zmin, ar, ac] = True
            pix = np.append(pix, np.vstack((ar, ac)), axis=-1)

        mall = mall[:, pix[0].min():pix[0].max() + 1, pix[1].min():pix[1].max() + 1].astype(np.float32)
        ymin, xmin = pix[0].min(), pix[1].min()
        if len(zdraw) > 1:
            mall, zfill = interpZ(mall, zdraw - zmin)
            for z in zfill:
                mask = mall[z].copy()
                ar, ac = np.nonzero(mask)
                ioverlap = self.cellpix[z + zmin][ar + ymin, ac + xmin] > 0
                if (~ioverlap).sum() < 5:
                    print('WARNING: stroke on plane %d not included due to overlaps' % z)
                elif ioverlap.sum() > 0:
                    mask[ar[ioverlap], ac[ioverlap]] = 0
                    ar, ac = ar[~ioverlap], ac[~ioverlap]
                # compute outline of mask
                outlines = utils.masks_to_outlines(mask)
                vr, vc = np.nonzero(outlines)
                vr, vc = vr + ymin, vc + xmin
                ar, ac = ar + ymin, ac + xmin
                self.draw_mask(z + zmin, ar, ac, vr, vc, color)
        self.zdraw.append(zdraw)

        return median

    def move_in_Z(self):
        if self.loaded:
            self.currentZ = min(self.NZ, max(0, int(self.scroll.value())))
            self.zpos.setText(str(self.currentZ))
            self.update_plot()

    def make_viewbox(self):
        # intialize the main viewport widget
        # print("making viewbox")
        self.p0 = guiparts.ViewBoxNoRightDrag(
            parent=self,
            lockAspect=True,
            name="plot1",
            border=[100, 100, 100],
            invertY=True
        )
        # self.p0.setBackgroundColor(color='#292929')

        self.brush_size = 3
        self.win.addItem(self.p0, 0, 0)

        self.p0.setMenuEnabled(False)
        self.p0.setMouseEnabled(x=True, y=True)

        self.img = pg.ImageItem(viewbox=self.p0, parent=self, axisOrder='row-major')
        self.img.autoDownsample = False
        # self.null_image = np.ones((200,200))
        # self.img.setImage(self.null_image)

        self.layer = guiparts.ImageDraw(viewbox=self.p0, parent=self)
        self.layer.setLevels([0, 255])

        self.scale = pg.ImageItem(viewbox=self.p0, parent=self)
        self.scale.setLevels([0, 255])
        self.p0.scene().contextMenuItem = self.p0
        # self.p0.setMouseEnabled(x=False,y=False)
        self.Ly, self.Lx = 512, 512
        self.p0.addItem(self.img)
        self.p0.addItem(self.layer)
        self.p0.addItem(self.scale)

        # guiparts.make_quadrants(self)

    def get_channels(self):
        channels = [self.jCBChanToSegment.currentIndex(), self.jCBChan2.currentIndex()]
        return channels

    def compute_saturation(self):
        # compute percentiles from stack
        self.saturation = []
        self.slider._low = np.percentile(self.stack[0].astype(np.float32), 1)
        self.slider._high = np.percentile(self.stack[0].astype(np.float32), 99)

        for n in range(len(self.stack)):
            print('n,', n)
            self.saturation.append([np.percentile(self.stack[n].astype(np.float32), 1),
                                    np.percentile(self.stack[n].astype(np.float32), 99)])


    def keyReleaseEvent(self, event):
        # print('self.loaded', self.loaded)
        if self.loaded:
            # self.p0.setMouseEnabled(x=True, y=True)
            if (event.modifiers() != QtCore.Qt.ControlModifier and
                event.modifiers() != QtCore.Qt.ShiftModifier and
                event.modifiers() != QtCore.Qt.AltModifier) and not self.in_stroke:
                updated = False
                if len(self.current_point_set) > 0:
                    if event.key() == QtCore.Qt.Key_Return:
                        self.add_set()
                    if self.NZ > 1:
                        if event.key() == QtCore.Qt.Key_Left:
                            self.currentZ = max(0, self.currentZ - 1)
                            self.zpos.setText(str(self.currentZ))
                        elif event.key() == QtCore.Qt.Key_Right:
                            self.currentZ = min(self.NZ - 1, self.currentZ + 1)
                            self.zpos.setText(str(self.currentZ))
                else:
                    if event.key() == QtCore.Qt.Key_M:
                        self.MCheckBox.toggle()
                    if event.key() == QtCore.Qt.Key_O:
                        self.OCheckBox.toggle()
                    if event.key() == QtCore.Qt.Key_C:
                        self.CHCheckBox.toggle()
                    if event.key() == QtCore.Qt.Key_S:
                        self.SCheckBox.toggle()
                    if event.key() == QtCore.Qt.Key_E:
                        self.eraser_button.toggle()
                        self.toolBox.setCurrentIndex(0)
                    if event.key() == QtCore.Qt.Key_P:
                        self.ASCheckBox.toggle()

                if event.key() == QtCore.Qt.Key_PageDown:
                    self.view = (self.view + 1) % (len(self.RGBChoose.bstr))
                    print('self.view ', self.view)
                    self.RGBChoose.button(self.view).setChecked(True)
                elif event.key() == QtCore.Qt.Key_PageUp:
                    self.view = (self.view - 1) % (len(self.RGBChoose.bstr))
                    print('self.view ', self.view)

                    self.RGBChoose.button(self.view).setChecked(True)

                # can change background or stroke size if cell not finished
                if event.key() == QtCore.Qt.Key_Up:
                    self.color = (self.color - 1) % (6)
                    print('self.color', self.color)
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_Down:
                    self.color = (self.color + 1) % (6)
                    print('self.color', self.color)
                    self.RGBDropDown.setCurrentIndex(self.color)

                if (event.key() == QtCore.Qt.Key_BracketLeft or
                      event.key() == QtCore.Qt.Key_BracketRight):
                    count = self.BrushChoose.count()
                    gci = self.BrushChoose.currentIndex()
                    if event.key() == QtCore.Qt.Key_BracketLeft:
                        gci = max(0, gci - 1)
                    else:
                        gci = min(count - 1, gci + 1)
                    self.BrushChoose.setCurrentIndex(gci)
                    self.brush_choose()
                    self.state_label.setText("Brush size: %s"%(2*gci+1), color='#969696')
                if not updated:
                    self.update_plot()
                elif event.modifiers() == QtCore.Qt.ControlModifier:
                    if event.key() == QtCore.Qt.Key_Z:
                        self.undo_action()
                    if event.key() == QtCore.Qt.Key_0:
                        self.clear_all()

    def keyPressEvent(self, event):
        if event.modifiers() == QtCore.Qt.ControlModifier:
            if event.key() == QtCore.Qt.Key_1:
                self.toolBox.setCurrentIndex(0)
            if event.key() == QtCore.Qt.Key_2:
                self.toolBox.setCurrentIndex(1)
            if event.key() == QtCore.Qt.Key_3:
                self.toolBox.setCurrentIndex(2)
        if event.key() == QtCore.Qt.Key_Minus or event.key() == QtCore.Qt.Key_Equal:
            self.p0.keyPressEvent(event)

    def chanchoose(self, image):
        if image.ndim > 2:
            if self.jCBChanToSegment.currentIndex() == 0:
                image = image.astype(np.float32).mean(axis=-1)[..., np.newaxis]
            else:
                chanid = [self.jCBChanToSegment.currentIndex() - 1]
                if self.jCBChan2.currentIndex() > 0:
                    chanid.append(self.jCBChan2.currentIndex() - 1)
                image = image[:, :, chanid].astype(np.float32)
        return image

    def initialize_model(self, gpu=False, pretrained_model=False, model_type='scellseg',
                 diam_mean=30., net_avg=False, device=None, nclasses=3,
                 residual_on=True, style_on=True, concatenation=False, update_step=1,
                 last_conv_on=True, attn_on=False, dense_on=False, style_scale_on=True,
                 task_mode='cellpose', model=None):
        self.current_model = model_type
        self.model = models.sCellSeg(gpu=gpu, pretrained_model=pretrained_model, model_type=model_type,
                 diam_mean=diam_mean, net_avg=net_avg, device=device, nclasses=nclasses,
                 residual_on=residual_on, style_on=style_on, concatenation=concatenation, update_step=update_step,
                 last_conv_on=last_conv_on, attn_on=attn_on, dense_on=dense_on, style_scale_on=style_scale_on,
                 task_mode=task_mode, model=model)


    def set_compute_thread(self):
        self.seg_thread = threading.Thread(target = self.compute_model)
        self.seg_thread.setDeamon(True)
        self.seg_thread.start()

    def compute_model(self):
        self.progress.setValue(0)
        self.update_plot()
        self.state_label.setText("Running...", color='#969696')
        QtWidgets.qApp.processEvents()  # force update gui

        if True:
            tic = time.time()
            self.clear_all()
            self.flows = [[], [], []]
            pretrained_model = os.path.join(self.model_dir, self.ModelChoose.currentText())
            self.initialize_model(pretrained_model=pretrained_model, gpu=self.useGPU.isChecked(),
                            model_type=self.ModelChoose.currentText())

            print('using model %s' % self.current_model)
            self.progress.setValue(10)
            do_3D = False
            if self.NZ > 1:
                do_3D = True
                data = self.stack.copy()
            else:
                data = self.stack[0].copy()
            channels = self.get_channels()

            # print(channels)
            self.diameter = float(self.Diameter.text())
            self.update_plot()
            try:
                # net_avg = self.NetAvg.currentIndex() == 0
                resample = self.NetAvg.currentIndex() == 1  # we need modify from here

                min_size = ((30. // 2) ** 2) * np.pi * 0.05

                try:
                    finetune_model = self.model_file_path[0]
                    print('ft_model', finetune_model)
                except:
                    finetune_model = None

                # inference
                masks, flows, _ = self.model.inference(finetune_model=finetune_model, net_avg=False,
                                                       query_images=data, channel=channels,
                                                       diameter=self.diameter,
                                                       resample=resample, flow_threshold=self.threshold,
                                                       cellprob_threshold=self.cellprob,
                                                       min_size=min_size, eval_batch_size=8,
                                                       postproc_mode=self.model.postproc_mode,
                                                       progress=self.progress)

                self.state_label.setText(
                    '%d cells found with scellseg net in %0.3fs' % (
                    len(np.unique(masks)[1:]), time.time() - tic),
                    color='#39B54A')
                # self.state_label.setStyleSheet("color:green;")
                self.update_plot()
                self.progress.setValue(75)

                self.flows[0] = flows[0].copy()
                self.flows[1] = (np.clip(utils.normalize99(flows[2].copy()), 0, 1) * 255).astype(np.uint8)
                if not do_3D:
                    masks = masks[np.newaxis, ...]
                    self.flows[0] = transforms.resize_image(self.flows[0], masks.shape[-2], masks.shape[-1],
                                                            interpolation=cv2.INTER_NEAREST)
                    self.flows[1] = transforms.resize_image(self.flows[1], masks.shape[-2], masks.shape[-1])
                if not do_3D:
                    self.flows[2] = np.zeros(masks.shape[1:], dtype=np.uint8)
                    self.flows = [self.flows[n][np.newaxis, ...] for n in range(len(self.flows))]
                else:
                    self.flows[2] = (flows[1][0] / 10 * 127 + 127).astype(np.uint8)

                if len(flows) > 2:
                    self.flows.append(flows[3])
                    self.flows.append(np.concatenate((flows[1], flows[2][np.newaxis, ...]), axis=0))

                print()
                self.progress.setValue(80)
                z = 0
                self.masksOn = True
                self.outlinesOn = True
                self.MCheckBox.setChecked(True)
                self.OCheckBox.setChecked(True)

                iopart._masks_to_gui(self, masks, outlines=None)
                self.progress.setValue(100)
                self.first_load_listView()

                # self.toggle_server(off=True)
                if not do_3D:
                    self.threshslider.setEnabled(True)
                    self.probslider.setEnabled(True)

                self.masks_for_save = masks

            except Exception as e:
                print('NET ERROR: %s' % e)
                self.progress.setValue(0)
                return

        else:  # except Exception as e:
            print('ERROR: %s' % e)

        print('Finished inference')


    def batch_inference(self):
        self.progress.setValue(0)
        # print('threshold', self.threshold, self.cellprob)
        # self.update_plot()

        if True:
            tic = time.time()
            self.clear_all()

            model_type =self.ModelChoose.currentText()
            pretrained_model = os.path.join(self.model_dir, model_type)
            self.initialize_model(pretrained_model=pretrained_model, gpu=self.useGPU.isChecked(),
                                  model_type=model_type)
            print('using model %s' % self.current_model)
            self.progress.setValue(10)

            channels = self.get_channels()
            self.diameter = float(self.Diameter.text())

            try:
                # net_avg = self.NetAvg.currentIndex() < 2
                # resample = self.NetAvg.currentIndex() == 1

                min_size = ((30. // 2) ** 2) * np.pi * 0.05

                try:
                    finetune_model = self.model_file_path[0]
                    print('ft_model', finetune_model)
                except:
                    finetune_model = None
                try:
                    dataset_path = self.batch_inference_dir
                except:
                    dataset_path = None

                # batch inference
                bz = 8 if self.bz_line.text() == '' else int(self.bz_line.text())
                save_name = self.current_model + '_' + dataset_path.split('\\')[-1]
                utils.set_manual_seed(5)
                try:
                    shotset = dataset.DatasetShot(eval_dir=dataset_path, class_name=None, image_filter='_img',
                                                  mask_filter='_masks',
                                                  channels=channels, task_mode=self.model.task_mode, active_ind=None,
                                                  rescale=True)
                    iopart._initialize_image_portable(self, iopart.imread(self.now_pyfile_path + '/assets/Loading1.png'),
                                                      resize=self.resize, X2=0)
                    self.state_label.setText("Running...", color='#969696')
                    QtWidgets.qApp.processEvents()  # force update gui
                except:
                    iopart._initialize_image_portable(self, iopart.imread(self.now_pyfile_path + '/assets/Loading4.png'),
                                                      resize=self.resize, X2=0)
                    self.state_label.setText("Please choose right data path",
                                             color='#FF6A56')
                    print("Please choose right data path")
                    self.batch_inference_bnt.setEnabled(False)
                    return

                queryset = dataset.DatasetQuery(dataset_path, class_name=None, image_filter='_img',
                                                mask_filter='_masks')
                query_image_names = queryset.query_image_names
                diameter = shotset.md
                print('>>>> mean diameter of this style,', round(diameter, 3))
                self.model.net.save_name = save_name

                self.img.setImage(iopart.imread(self.now_pyfile_path + '/assets/Loading2.png'), autoLevels=False, lut=None)
                self.state_label.setText("Running...", color='#969696')
                QtWidgets.qApp.processEvents()  # force update gui

                # flow_threshold was set to 0.4, and cellprob_threshold was set to 0.5
                try:
                    masks, flows, _ = self.model.inference(finetune_model=finetune_model, net_avg=False,
                                                           query_image_names=query_image_names, channel=channels,
                                                           diameter=diameter,
                                                           resample=False, flow_threshold=0.4,
                                                           cellprob_threshold=0.5,
                                                           min_size=min_size, eval_batch_size=bz,
                                                           postproc_mode=self.model.postproc_mode,
                                                           progress=self.progress)
                except RuntimeError:
                    iopart._initialize_image_portable(self,
                                                      iopart.imread(self.now_pyfile_path + '/assets/Loading4.png'),
                                                      resize=self.resize, X2=0)
                    self.state_label.setText("Batch size is too big, please set smaller",
                                             color='#FF6A56')
                    print("Batch size is too big, please set smaller")
                    return

                # save output images
                diams = np.ones(len(query_image_names)) * diameter
                imgs = [io.imread(query_image_name) for query_image_name in query_image_names]
                io.masks_flows_to_seg(imgs, masks, flows, diams, query_image_names,
                                      [channels for i in range(len(query_image_names))])
                io.save_to_png(imgs, masks, flows, query_image_names, labels=None, aps=None,
                               task_mode=self.model.task_mode)

                self.masks_for_save = masks

            except:
                iopart._initialize_image_portable(self, iopart.imread(self.now_pyfile_path + '/assets/Loading4.png'), resize=self.resize,
                                                  X2=0)
                self.state_label.setText("Please choose right data path",
                                         color='#FF6A56')
                return

        else:  # except Exception as e:
            print('ERROR: %s' % e)

        self.img.setImage(iopart.imread(self.now_pyfile_path + '/assets/Loading3.png'), autoLevels=False, lut=None)
        self.state_label.setText('Finished inference in %0.3fs!'%(time.time() - tic), color='#39B54A')
        self.batch_inference_bnt.setEnabled(False)

    def compute_cprob(self):
        rerun = False
        if self.cellprob != self.probslider.value():
            rerun = True
            self.cellprob = self.probslider.value()
        if self.threshold != (31 - self.threshslider.value()) / 10.:
            rerun = True
            self.threshold = (31 - self.threshslider.value()) / 10.
        if not rerun:
            return

        if self.threshold == 3.0 or self.NZ > 1:
            thresh = None
            print('computing masks with cell prob=%0.3f, no flow error threshold' %
                  (self.cellprob))
        else:
            thresh = self.threshold
            print('computing masks with cell prob=%0.3f, flow error threshold=%0.3f' %
                  (self.cellprob, thresh))

        maski = dynamics.get_masks(self.flows[3].copy(), iscell=(self.flows[4][-1] > self.cellprob),
                                   flows=self.flows[4][:-1], threshold=thresh)
        if self.NZ == 1:
            maski = utils.fill_holes_and_remove_small_masks(maski)
        maski = transforms.resize_image(maski, self.cellpix.shape[-2], self.cellpix.shape[-1],
                                        interpolation=cv2.INTER_NEAREST)
        self.masksOn = True
        self.outlinesOn = True
        self.MCheckBox.setChecked(True)
        self.OCheckBox.setChecked(True)
        if maski.ndim < 3:
            maski = maski[np.newaxis, ...]
        print('%d cells found' % (len(np.unique(maski)[1:])))
        iopart._masks_to_gui(self, maski, outlines=None)
        self.threshslider.setToolTip("Value: " + str(self.threshold))
        self.probslider.setToolTip("Value: " + str(self.cellprob))

        self.first_load_listView()
        self.show()


    def reset(self):
        # ---- start sets of points ---- #
        self.selected = 0
        self.X2 = 0
        self.resize = -1
        self.onechan = False
        self.loaded = False
        self.channel = [0, 1]
        self.current_point_set = []
        self.in_stroke = False
        self.strokes = []
        self.stroke_appended = True
        self.ncells = 0
        self.zdraw = []
        self.removed_cell = []
        self.cellcolors = [np.array([255, 255, 255])]
        # -- set menus to default -- #
        self.color = 0
        self.RGBDropDown.setCurrentIndex(self.color)
        self.view = 0
        self.RGBChoose.button(self.view).setChecked(True)
        self.BrushChoose.setCurrentIndex(1)
        self.CHCheckBox.setChecked(False)
        self.OCheckBox.setEnabled(True)
        self.SSCheckBox.setChecked(True)

        # -- zero out image stack -- #
        self.opacity = 128  # how opaque masks should be
        self.outcolor = [200, 200, 255, 200]
        self.NZ, self.Ly, self.Lx = 1, 512, 512
        if self.autobtn.isChecked():
            self.saturation = [[0, 255] for n in range(self.NZ)]
        self.currentZ = 0
        self.flows = [[], [], [], [], [[]]]
        self.stack = np.zeros((1, self.Ly, self.Lx, 3))
        # masks matrix
        self.layers = 0 * np.ones((1, self.Ly, self.Lx, 4), np.uint8)
        # image matrix with a scale disk
        self.radii = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
        self.cellpix = np.zeros((1, self.Ly, self.Lx), np.uint16)
        self.outpix = np.zeros((1, self.Ly, self.Lx), np.uint16)
        self.ismanual = np.zeros(0, np.bool)
        self.update_plot()
        self.filename = []
        self.loaded = False

    def first_load_listView(self):
        self.listmodel = Qt.QStandardItemModel(self.ncells,1)
        # self.listmodel = Qt.QStringListModel()
        self.listmodel.setHorizontalHeaderLabels(["Annotation"])
        self.myCellList = ['instance_' + str(i) for i in range(1, self.ncells + 1)]
        for i in range(len(self.myCellList)):
            self.listmodel.setItem(i,Qt.QStandardItem(self.myCellList[i]))
        self.listView.setModel(self.listmodel)

    def initialize_listView(self):
        if self.filename != []:
            if os.path.isfile(os.path.splitext(self.filename)[0] + '_instance_list.txt'):
                self.list_file_name = str(os.path.splitext(self.filename)[0] + '_instance_list.txt')
                self.myCellList_array = np.loadtxt(self.list_file_name, dtype=str)
                self.myCellList = self.myCellList_array.tolist()
                if len(self.myCellList) == self.ncells:
                    self.listmodel = Qt.QStandardItemModel(self.ncells, 1)
                    self.listmodel.setHorizontalHeaderLabels(["Annotation"])
                    for i in range(len(self.myCellList)):
                        self.listmodel.setItem(i,Qt.QStandardItem(self.myCellList[i]))
                    self.listView.setModel(self.listmodel)
                else:
                    self.listmodel = Qt.QStandardItemModel(self.ncells, 1)
                    # self.listmodel = Qt.QStringListModel()
                    self.listmodel.setHorizontalHeaderLabels(["Annotation"])
                    self.myCellList = ['instance_' + str(i) for i in range(1, self.ncells + 1)]
                    for i in range(len(self.myCellList)):
                        self.listmodel.setItem(i,Qt.QStandardItem(self.myCellList[i]))
                    self.listView.setModel(self.listmodel)
            else:
                self.myCellList = ['instance_' + str(i) for i in range(1, self.ncells + 1)]
                self.listmodel = Qt.QStandardItemModel(self.ncells, 1)
                # self.listmodel = Qt.QStringListModel()
                self.listmodel.setHorizontalHeaderLabels(["Annotation"])

                for i in range(len(self.myCellList)):
                    self.listmodel.setItem(i,Qt.QStandardItem(self.myCellList[i]))
                self.listView.setModel(self.listmodel)


    def initinal_p0(self):
        # self.p0.removeItem(self.img)
        self.p0.removeItem(self.layer)
        self.p0.removeItem(self.scale)

        # self.img.deleteLater()
        self.layer.deleteLater()
        self.scale.deleteLater()
        # self.img = pg.ImageItem(viewbox=self.p0, parent=self, axisOrder='row-major')
        # self.img.autoDownsample = False
        self.layer = guiparts.ImageDraw(viewbox=self.p0, parent=self)
        self.layer.setLevels([0, 255])
        self.scale = pg.ImageItem(viewbox=self.p0, parent=self)
        self.scale.setLevels([0, 255])
        self.p0.scene().contextMenuItem = self.p0

        # self.p0.addItem(self.img)
        self.p0.addItem(self.layer)
        self.p0.addItem(self.scale)

    def add_list_item(self):
        # print(self.ncells)
        # self.myCellList = self.listmodel.data()
        self.listView.selectAll()
        self.myCellList = []
        for item in self.listView.selectedIndexes():
            data = item.data()
            self.myCellList.append(data)

        temp_nums = []
        for celli in self.myCellList:
            if 'instance_' in celli:
                temp_nums.append(int(celli.split('instance_')[-1]))
        if len(temp_nums) == 0:
            now_cellIdx = 0
        else:
            now_cellIdx = np.max(np.array(temp_nums))

        self.myCellList.append('instance_' + str(now_cellIdx+1))
        # self.myCellList.append('instance_' + str(self.ncells))
        self.listmodel = Qt.QStandardItemModel(self.ncells, 1)
        # self.listmodel = Qt.QStringListModel()
        self.listmodel.setHorizontalHeaderLabels(["Annotation"])
        for i in range(len(self.myCellList)):
            self.listmodel.setItem(i, Qt.QStandardItem(self.myCellList[i]))
        self.listView.setModel(self.listmodel)

    def delete_list_item(self, index):

        # self.myCellList = self.listmodel.data()
        self.listView.selectAll()
        self.myCellList = []
        for item in self.listView.selectedIndexes():
            data = item.data()
            self.myCellList.append(data)
        self.last_remove_index = index
        self.last_remove_item = self.myCellList.pop(index - 1)
        self.listmodel = Qt.QStandardItemModel(self.ncells, 1)
        # self.listmodel = Qt.QStringListModel()
        self.listmodel.setHorizontalHeaderLabels(["Annotation"])
        for i in range(len(self.myCellList)):
            self.listmodel.setItem(i, Qt.QStandardItem(self.myCellList[i]))
        self.listView.setModel(self.listmodel)

    def check_gpu(self, torch=True):
        # also decide whether or not to use torch
        self.useGPU.setChecked(False)
        self.useGPU.setEnabled(False)
        if models.use_gpu():
            self.useGPU.setEnabled(True)
            self.useGPU.setChecked(True)

    def check_ftgpu(self, torch=True):
        # also decide whether or not to use torch
        self.ftuseGPU.setChecked(False)
        self.ftuseGPU.setEnabled(False)
        if models.use_gpu():
            self.ftuseGPU.setEnabled(True)
            self.ftuseGPU.setChecked(True)

    def clear_all(self):
        self.prev_selected = 0
        self.selected = 0
        # self.layers_undo, self.cellpix_undo, self.outpix_undo = [],[],[]
        self.layers = 0 * np.ones((self.NZ, self.Ly, self.Lx, 4), np.uint8)
        self.cellpix = np.zeros((self.NZ, self.Ly, self.Lx), np.uint16)
        self.outpix = np.zeros((self.NZ, self.Ly, self.Lx), np.uint16)
        self.cellcolors = [np.array([255, 255, 255])]
        self.ncells = 0
        self.initialize_listView()
        print('removed all cells')
        self.toggle_removals()
        self.update_plot()

    def list_select_cell(self, idx):
        self.prev_selected = self.selected
        self.selected = idx
        # print(idx)
        # print(self.prev_selected)

        if self.selected > 0:
            self.layers[self.cellpix == idx] = np.array([255, 255, 255, 255])
            if idx < self.ncells + 1 and self.prev_selected > 0 and self.prev_selected != idx:
                self.layers[self.cellpix == self.prev_selected] = np.append(self.cellcolors[self.prev_selected],
                                                                            self.opacity)
                # if self.outlinesOn:
                #     self.layers[self.outpix == idx] = np.array(self.outcolor).astype(np.uint8)
            self.update_plot()

    def select_cell(self, idx):
        self.prev_selected = self.selected
        self.selected = idx
        self.listView.selectRow(idx - 1)

        # print('the prev-selected is ', self.prev_selected)
        if self.selected > 0:
            self.layers[self.cellpix == idx] = np.array([255, 255, 255, self.opacity])

            print('idx', self.prev_selected, idx)
            if idx < self.ncells + 1 and self.prev_selected > 0 and self.prev_selected != idx:
                self.layers[self.cellpix == self.prev_selected] = np.append(self.cellcolors[self.prev_selected],
                                                                            self.opacity)
            # if self.outlinesOn:
            #    self.layers[self.outpix==idx] = np.array(self.outcolor)
            self.update_plot()

    def unselect_cell(self):
        if self.selected > 0:
            idx = self.selected
            if idx < self.ncells + 1:
                self.layers[self.cellpix == idx] = np.append(self.cellcolors[idx], self.opacity)
                if self.outlinesOn:
                    self.layers[self.outpix == idx] = np.array(self.outcolor).astype(np.uint8)
                    # [0,0,0,self.opacity])
                self.update_plot()
        self.selected = 0

    def remove_cell(self, idx):
        # remove from manual array
        # self.selected = 0
        for z in range(self.NZ):
            cp = self.cellpix[z] == idx
            op = self.outpix[z] == idx
            # remove from mask layer
            self.layers[z, cp] = np.array([0, 0, 0, 0])
            # remove from self.cellpix and self.outpix
            self.cellpix[z, cp] = 0
            self.outpix[z, op] = 0
            # reduce other pixels by -1
            self.cellpix[z, self.cellpix[z] > idx] -= 1
            self.outpix[z, self.outpix[z] > idx] -= 1
        self.update_plot()
        if self.NZ == 1:
            self.removed_cell = [self.ismanual[idx - 1], self.cellcolors[idx], np.nonzero(cp), np.nonzero(op)]
            self.redo.setEnabled(True)
        # remove cell from lists
        self.ismanual = np.delete(self.ismanual, idx - 1)
        del self.cellcolors[idx]
        del self.zdraw[idx - 1]
        self.ncells -= 1
        print('removed cell %d' % (idx - 1))
        self.delete_list_item(index=idx)

        if self.ncells == 0:
            self.ClearButton.setEnabled(False)
        if self.NZ == 1:
            iopart._save_sets(self)
        # self.select_cell(0)

    def merge_cells(self, idx):
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected != self.prev_selected:
            for z in range(self.NZ):
                ar0, ac0 = np.nonzero(self.cellpix[z] == self.prev_selected)
                ar1, ac1 = np.nonzero(self.cellpix[z] == self.selected)
                touching = np.logical_and((ar0[:, np.newaxis] - ar1) == 1,
                                          (ac0[:, np.newaxis] - ac1) == 1).sum()
                print(touching)
                ar = np.hstack((ar0, ar1))
                ac = np.hstack((ac0, ac1))
                if touching:
                    mask = np.zeros((np.ptp(ar) + 4, np.ptp(ac) + 4), np.uint8)
                    mask[ar - ar.min() + 2, ac - ac.min() + 2] = 1
                    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    pvc, pvr = contours[-2][0].squeeze().T
                    vr, vc = pvr + ar.min() - 2, pvc + ac.min() - 2
                else:
                    vr0, vc0 = np.nonzero(self.outpix[z] == self.prev_selected)
                    vr1, vc1 = np.nonzero(self.outpix[z] == self.selected)
                    vr = np.hstack((vr0, vr1))
                    vc = np.hstack((vc0, vc1))
                color = self.cellcolors[self.prev_selected]
                self.draw_mask(z, ar, ac, vr, vc, color, idx=self.prev_selected)
            self.remove_cell(self.selected)
            print('merged two cells')
            self.update_plot()
            iopart._save_sets(self)

            self.undo.setEnabled(False)
            self.redo.setEnabled(False)

    def undo_remove_cell(self):
        if len(self.removed_cell) > 0:
            z = 0
            ar, ac = self.removed_cell[2]
            vr, vc = self.removed_cell[3]
            color = self.removed_cell[1]
            self.draw_mask(z, ar, ac, vr, vc, color)
            self.toggle_mask_ops()
            self.cellcolors.append(color)
            self.ncells += 1
            self.add_list_item()
            self.ismanual = np.append(self.ismanual, self.removed_cell[0])
            self.zdraw.append([])
            print('added back removed cell')
            self.update_plot()
            iopart._save_sets(self)
            self.removed_cell = []
            self.redo.setEnabled(False)

    def fine_tune(self):
        tic = time.time()

        dataset_dir = self.fine_tune_dir
        self.state_label.setText("%s"%(dataset_dir), color='#969696')

        if not isinstance(dataset_dir, str):  # TODO: 改成警告
            print('dataset_dir is not provided')

        train_epoch = 100 if self.epoch_line.text() == '' else int(self.epoch_line.text())
        ft_bz = 8 if self.ftbz_line.text() == '' else int(self.ftbz_line.text())
        contrast_on = 1 if self.stmodelchooseBnt.currentText() == 'contrastive' else 0
        model_type = self.ftmodelchooseBnt.currentText()
        task_mode, postproc_mode, attn_on, dense_on, style_scale_on = utils.process_different_model(model_type)  # task_mode mean different instance representation
        pretrained_model = os.path.join(self.model_dir, model_type)
        channels = [self.chan1chooseBnt.currentIndex(), self.chan2chooseBnt.currentIndex()]
        print(dataset_dir, train_epoch, channels)
        utils.set_manual_seed(5)
        try:
            print('ft_bz', ft_bz)
            shotset = DatasetShot(eval_dir=dataset_dir, class_name=None, image_filter='_img', mask_filter='_masks',
                                  channels=channels,
                                  train_num=train_epoch * ft_bz, task_mode=task_mode, rescale=True)
            iopart._initialize_image_portable(self, iopart.imread(self.now_pyfile_path + '/assets/Loading1.png'), resize=self.resize, X2=0)
            self.state_label.setText("Running...", color='#969696')
            QtWidgets.qApp.processEvents()  # force update gui
        except ValueError:
            iopart._initialize_image_portable(self, iopart.imread(self.now_pyfile_path + '/assets/Loading4.png'), resize=self.resize, X2=0)
            self.state_label.setText("Please choose right data path",
                                     color='#FF6A56')
            print("Please choose right data path")
            self.ftbnt.setEnabled(False)
            return


        shot_gen = DataLoader(dataset=shotset, batch_size=ft_bz, num_workers=0, pin_memory=True)

        diameter = shotset.md
        print('>>>> mean diameter of this style,', round(diameter, 3))

        lr = {'downsample': 0.001, 'upsample': 0.001, 'tasker': 0.001, 'alpha': 0.1}
        lr_schedule_gamma = {'downsample': 0.5, 'upsample': 0.5, 'tasker': 0.5, 'alpha': 0.5}

        step_size = int(train_epoch * 0.25)
        print('step_size', step_size)
        self.initialize_model(pretrained_model=pretrained_model, gpu=self.ftuseGPU.isChecked(), model_type=model_type)
        self.model.net.pretrained_model = pretrained_model

        save_name = model_type + '_' + os.path.basename(dataset_dir)

        self.model.net.contrast_on = contrast_on
        if contrast_on:
            self.model.net.pair_gen = DatasetPairEval(positive_dir=dataset_dir, use_negative_masks=False, gpu=self.ftuseGPU.isChecked(),
                                                 rescale=True)
            self.model.net.save_name = save_name + '-cft'
        else:
            self.model.net.save_name = save_name + '-ft'

        try:
            print('Now is fine-tuning...Please Wait')
            self.img.setImage(iopart.imread(self.now_pyfile_path + '/assets/Loading2.png'), autoLevels=False, lut=None)
            self.state_label.setText("Running...", color='#969696')
            QtWidgets.qApp.processEvents()  # force update gui
            self.model.finetune(shot_gen=shot_gen, lr=lr, lr_schedule_gamma=lr_schedule_gamma, step_size=step_size, savepath=dataset_dir)
        except RuntimeError:
            iopart._initialize_image_portable(self, iopart.imread(self.now_pyfile_path + '/assets/Loading4.png'), resize=self.resize, X2=0)
            self.state_label.setText("Batch size is too big, please set smaller",
                                     color='#FF6A56')
            print("Batch size is too big, please set smaller")
            return

        print('Finished fine-tuning')
        self.img.setImage(iopart.imread(self.now_pyfile_path + '/assets/Loading3.png'), autoLevels=False, lut=None)
        self.state_label.setText("Finished in %0.3fs, model saved at %s/fine-tune/%s" %(time.time()-tic, dataset_dir, self.model.net.save_name), color='#39B54A')
        self.ftbnt.setEnabled(False)
        self.fine_tune_dir = ''


    def get_single_cell(self):
        tic = time.time()

        try:
            data_path = self.single_cell_dir
        except:
            iopart._initialize_image_portable(self, iopart.imread(self.now_pyfile_path + '/assets/Loading4.png'), resize=self.resize, X2=0)
            self.state_label.setText("Please choose right data path",
                                     color='#FF6A56')

        print(data_path)
        try:
            image_names = io.get_image_files(data_path, '_masks', imf='_img')
            mask_names, _ = io.get_label_files(image_names, '_img_cp_masks', imf='_img')
            iopart._initialize_image_portable(self, iopart.imread(self.now_pyfile_path + '/assets/Loading1.png'), resize=self.resize, X2=0)
            self.state_label.setText("Running...", color='#969696')
            QtWidgets.qApp.processEvents()  # force update gui
        except:
            iopart._initialize_image_portable(self, iopart.imread(self.now_pyfile_path + '/assets/Loading4.png'), resize=self.resize, X2=0)
            self.state_label.setText("Please choose right data path",
                                     color='#FF6A56')
            print("Please choose right data path")
            self.single_cell_btn.setEnabled(False)
            return

        sta = 256
        save_dir = os.path.join(os.path.dirname(data_path), 'single')
        utils.make_folder(save_dir)
        imgs = [io.imread(os.path.join(data_path, image_name)) for image_name in image_names]
        masks = [io.imread(os.path.join(data_path, mask_name)) for mask_name in mask_names]

        self.img.setImage(iopart.imread(self.now_pyfile_path + '/assets/Loading2.png'), autoLevels=False, lut=None)
        self.state_label.setText("Running...", color='#969696')
        QtWidgets.qApp.processEvents()  # force update gui

        for n in trange(len(masks)):
            maskn = masks[n]
            props = regionprops(maskn)
            i_max = maskn.max() + 1
            for i in range(1, i_max):
                maskn_ = np.zeros_like(maskn)
                maskn_[maskn == i] = 1
                bbox = props[i - 1]['bbox']
                if imgs[n].ndim == 3:
                    imgn_single = imgs[n][bbox[0]:bbox[2], bbox[1]:bbox[3]] * maskn_[bbox[0]:bbox[2], bbox[1]:bbox[3],
                                                                              np.newaxis]
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

                    save_name = os.path.join(save_dir, image_names[n].split('query')[-1].split('.')[0][1:] + '_' + str(
                        i) + '.tif')
                    cv2.imwrite(save_name, imgn_single)

        print('Finish getting single instance')
        self.img.setImage(iopart.imread(self.now_pyfile_path + '/assets/Loading3.png'), autoLevels=False, lut=None)
        self.state_label.setText("Finished in %0.3fs, saved at %s"%(time.time()-tic, os.path.dirname(data_path)+'/single') , color='#39B54A')
        self.single_cell_btn.setEnabled(False)

    def fine_tune_dir_choose(self):
        self.fine_tune_dir = QtWidgets.QFileDialog.getExistingDirectory(None,"choose fine-tune data",self.DefaultImFolder)
        if self.fine_tune_dir =='':
            self.state_label.setText("Choose nothing", color='#FF6A56')
        else:
            self.state_label.setText("Choose data at %s"%str(self.fine_tune_dir), color='#969696')
            self.ftbnt.setEnabled(True)
            iopart._initialize_image_portable(self, iopart.imread(self.now_pyfile_path + '/assets/black.png'), resize=self.resize, X2=0)

    def batch_inference_dir_choose(self):
        self.batch_inference_dir = QtWidgets.QFileDialog.getExistingDirectory(None, "choose batch segmentation data", self.DefaultImFolder)
        if self.batch_inference_dir =='':
            self.state_label.setText("Choose nothing", color='#FF6A56')
        else:
            self.state_label.setText("Choose data at %s"%str(self.batch_inference_dir), color='#969696')
            self.batch_inference_bnt.setEnabled(True)
            iopart._initialize_image_portable(self, iopart.imread(self.now_pyfile_path + '/assets/black.png'), resize=self.resize, X2=0)


    def single_dir_choose(self):
        self.single_cell_dir = QtWidgets.QFileDialog.getExistingDirectory(None, "choose get single instance data", self.DefaultImFolder)
        if self.single_cell_dir =='':
            self.state_label.setText("Choose nothing", color='#FF6A56')
        else:
            self.state_label.setText("Choose data at %s"%str(self.single_cell_dir), color='#969696')
            self.single_cell_btn.setEnabled(True)
            iopart._initialize_image_portable(self, iopart.imread(self.now_pyfile_path + '/assets/black.png'), resize=self.resize, X2=0)


    def model_file_dir_choose(self):
        """ model after fine-tuning"""
        self.model_file_path = QtWidgets.QFileDialog.getOpenFileName(None, "choose model file", self.DefaultImFolder)
        if self.model_file_path[0] =='':
            self.state_label.setText("Choose nothing", color='#FF6A56')
        else:
            self.state_label.setText("Choose model at %s"%str(self.model_file_path[0]), color='#969696')


    def reset_pretrain_model(self):
        self.model_file_path = None
        self.state_label.setText("Reset to pre-trained model", color='#969696')

    def remove_stroke(self, delete_points=True):
        # self.current_stroke = get_unique_points(self.current_stroke)
        stroke = np.array(self.strokes[-1])
        cZ = stroke[0, 0]
        outpix = self.outpix[cZ][stroke[:, 1], stroke[:, 2]] > 0
        self.layers[cZ][stroke[~outpix, 1], stroke[~outpix, 2]] = np.array([0, 0, 0, 0])
        if self.masksOn:
            cellpix = self.cellpix[cZ][stroke[:, 1], stroke[:, 2]]
            ccol = np.array(self.cellcolors.copy())
            if self.selected > 0:
                ccol[self.selected] = np.array([255, 255, 255])
            col2mask = ccol[cellpix]
            col2mask = np.concatenate((col2mask, self.opacity * (cellpix[:, np.newaxis] > 0)), axis=-1)
            self.layers[cZ][stroke[:, 1], stroke[:, 2], :] = col2mask
        if self.outlinesOn:
            self.layers[cZ][stroke[outpix, 1], stroke[outpix, 2]] = np.array(self.outcolor)
        if delete_points:
            self.current_point_set = self.current_point_set[:-1 * (stroke[:, -1] == 1).sum()]
        del self.strokes[-1]
        self.update_plot()


    def brush_choose(self):
        self.brush_size = self.BrushChoose.currentIndex() * 2 + 1
        if self.loaded:
            self.layer.setDrawKernel(kernel_size=self.brush_size)
            self.update_plot()
        # if self.eraser_button.isChecked():
            # print("will change")
            # self.cur_size = self.brush_size * 6
            # cursor = Qt.QPixmap("./assets/eraser.png")
            # cursor_scaled = cursor.scaled(self.cur_size, self.cur_size)
            # cursor_set = Qt.QCursor(cursor_scaled, self.cur_size/2, self.cur_size/2)
            # QtWidgets.QApplication.setOverrideCursor(cursor_set)
            # self.update_plot()

    #
    # def toggle_server(self, off=False):
    #     if SERVER_UPLOAD:
    #         if self.ncells>0 and not off:
    #             self.saveServer.setEnabled(True)
    #             self.ServerButton.setEnabled(True)
    #
    #         else:
    #             self.saveServer.setEnabled(False)
    #             self.ServerButton.setEnabled(False)

    def toggle_mask_ops(self):
        self.toggle_removals()
        # self.toggle_server()

    def load_cell_list(self):
        self.list_file_name = QtWidgets.QFileDialog.getOpenFileName(None, "Load instance list", self.DefaultImFolder)
        self.myCellList_array = np.loadtxt(self.list_file_name[0], dtype=str)
        self.myCellList = self.myCellList_array.tolist()
        if len(self.myCellList) == self.ncells:
            self.listmodel = Qt.QStandardItemModel(self.ncells, 1)
            self.listmodel.setHorizontalHeaderLabels(["Annotation"])
            for i in range(len(self.myCellList)):
                self.listmodel.setItem(i, Qt.QStandardItem(self.myCellList[i]))
            self.listView.setModel(self.listmodel)

    def toggle_scale(self):
        if self.scale_on:
            self.p0.removeItem(self.scale)
            self.scale_on = False
        else:
            self.p0.addItem(self.scale)
            self.scale_on = True

    def toggle_removals(self):

        if self.ncells > 0:
            # self.ClearButton.setEnabled(True)
            self.remcell.setEnabled(True)
            self.undo.setEnabled(True)
        else:
            # self.ClearButton.setEnabled(False)
            self.remcell.setEnabled(False)
            self.undo.setEnabled(False)

    def remove_action(self):
        if self.selected > 0:
            self.remove_cell(self.selected)

    def undo_action(self):
        if (len(self.strokes) > 0 and
                self.strokes[-1][0][0] == self.currentZ):
            self.remove_stroke()
        else:
            # remove previous cell
            if self.ncells > 0:
                self.remove_cell(self.ncells)

    def undo_remove_action(self):
        self.undo_remove_cell()

    def get_files(self):
        images = []
        images.extend(glob.glob(os.path.dirname(self.filename) + '/*.png'))
        images.extend(glob.glob(os.path.dirname(self.filename) + '/*.jpg'))
        images.extend(glob.glob(os.path.dirname(self.filename) + '/*.jpeg'))
        images.extend(glob.glob(os.path.dirname(self.filename) + '/*.tif'))
        images.extend(glob.glob(os.path.dirname(self.filename) + '/*.tiff'))
        images.extend(glob.glob(os.path.dirname(self.filename) + '/*.jfif'))
        images = natsorted(images)
        fnames = [os.path.split(images[k])[-1] for k in range(len(images))]
        f0 = os.path.split(self.filename)[-1]
        idx = np.nonzero(np.array(fnames) == f0)[0][0]
        return images, idx

    def get_prev_image(self):
        images, idx = self.get_files()
        idx = (idx - 1) % len(images)
        iopart._load_image(self, filename=images[idx])

    def get_next_image(self):
        images, idx = self.get_files()
        idx = (idx + 1) % len(images)
        iopart._load_image(self, filename=images[idx])

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        # print(files)

        if os.path.splitext(files[0])[-1] == '.npy':
            iopart._load_seg(self, filename=files[0])
            self.initialize_listView()
        if os.path.isdir(files[0]):
            print("loading a folder")
            self.ImFolder = files[0]
            try:
                self.OpenDirDropped()
            except:
                iopart._initialize_image_portable(self, iopart.imread(self.now_pyfile_path + '/assets/black.png'), resize=self.resize, X2=0)
                self.state_label.setText("No image found, please choose right data path",
                                         color='#FF6A56')
        else:
            # print(len(files))
            # print(files[0])
            self.ImFolder = os.path.dirname(files[0])
            try:
                self.OpenDirDropped(os.path.basename(files[0]))
                print(files[0], self.ImNameSet[self.CurImId])
                self.ImPath = self.ImFolder + r'/' + self.ImNameSet[self.CurImId]
                iopart._load_image(self, filename=self.ImPath)
                self.initialize_listView()

                fname = os.path.basename(files[0])
                fsuffix = os.path.splitext(fname)[-1]
                if fsuffix in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.jfif']:
                    if '_mask' in fname:
                        self.state_label.setText("This is a mask file, autoload corresponding image", color='#FDC460')
                else:
                    self.state_label.setText("This format is not supported", color='#FF6A56')

            except ValueError:
                iopart._initialize_image_portable(self, iopart.imread(self.now_pyfile_path + '/assets/black.png'), resize=self.resize, X2=0)
                self.state_label.setText("No corresponding iamge for this mask file", color='#FF6A56')

    def toggle_masks(self):
        if self.MCheckBox.isChecked():
            self.masksOn = True
        else:
            self.masksOn = False
        if self.OCheckBox.isChecked():
            self.outlinesOn = True
        else:
            self.outlinesOn = False
        if not self.masksOn and not self.outlinesOn:
            self.p0.removeItem(self.layer)
            self.layer_off = True
        else:
            if self.layer_off:
                self.p0.addItem(self.layer)
            self.redraw_masks(masks=self.masksOn, outlines=self.outlinesOn)
        if self.loaded:
            self.update_plot()

    def draw_mask(self, z, ar, ac, vr, vc, color, idx=None):
        ''' draw single mask using outlines and area '''
        if idx is None:
            idx = self.ncells + 1
        self.cellpix[z][vr, vc] = idx
        self.cellpix[z][ar, ac] = idx
        self.outpix[z][vr, vc] = idx
        if self.masksOn:
            self.layers[z][ar, ac, :3] = color
            # print(self.layers.shape)
            # print(self.layers[z][ar, ac, :3].shape)
            # print(self.layers[z][ar,ac,:3])
            self.layers[z][ar, ac, -1] = self.opacity
            # print(z)
        if self.outlinesOn:
            self.layers[z][vr, vc] = np.array(self.outcolor)

    def draw_masks(self):
        self.cellcolors = np.array(self.cellcolors)
        self.layers[..., :3] = self.cellcolors[self.cellpix, :]
        self.layers[..., 3] = self.opacity * (self.cellpix > 0).astype(np.uint8)
        self.cellcolors = list(self.cellcolors)
        self.layers[self.outpix > 0] = np.array(self.outcolor)
        if self.selected > 0:
            self.layers[self.outpix == self.selected] = np.array([0, 0, 0, self.opacity])

    def redraw_masks(self, masks=True, outlines=True):
        if not outlines and masks:
            self.draw_masks()
            self.cellcolors = np.array(self.cellcolors)
            self.layers[..., :3] = self.cellcolors[self.cellpix, :]
            self.layers[..., 3] = self.opacity * (self.cellpix > 0).astype(np.uint8)
            self.cellcolors = list(self.cellcolors)
            if self.selected > 0:
                self.layers[self.cellpix == self.selected] = np.array([255, 255, 255, self.opacity])
        else:
            if masks:
                self.layers[..., 3] = self.opacity * (self.cellpix > 0).astype(np.uint8)
            else:
                self.layers[..., 3] = 0
            self.layers[self.outpix > 0] = np.array(self.outcolor).astype(np.uint8)

    def update_plot_without_mask(self):
        self.Ly, self.Lx, _ = self.stack[self.currentZ].shape
        if self.view == 0:
            image = self.stack[self.currentZ]
            if self.color == 0:
                if self.onechan:
                    # show single channel
                    image = self.stack[self.currentZ][:, :, 0]
                self.img.setImage(image, autoLevels=False, lut=None)
            elif self.color == 1:
                image = image.astype(np.float32).mean(axis=-1).astype(np.uint8)
                self.img.setImage(image, autoLevels=False, lut=None)
            elif self.color == 2:
                image = image.astype(np.float32).mean(axis=-1).astype(np.uint8)
                self.img.setImage(image, autoLevels=False, lut=self.cmap[0])
            elif self.color > 2:
                image = image[:, :, self.color - 3]
                self.img.setImage(image, autoLevels=False, lut=self.cmap[self.color - 2])
            self.img.setLevels(self.saturation[self.currentZ])
        else:
            image = np.zeros((self.Ly, self.Lx), np.uint8)
            if len(self.flows) >= self.view - 1 and len(self.flows[self.view - 1]) > 0:
                image = self.flows[self.view - 1][self.currentZ]
            if self.view > 1:
                self.img.setImage(image, autoLevels=False, lut=self.bwr)
            else:
                self.img.setImage(image, autoLevels=False, lut=None)
            self.img.setLevels([0.0, 255.0])
        self.scale.setImage(self.radii, autoLevels=False)
        self.scale.setLevels([0.0, 255.0])
        if self.masksOn or self.outlinesOn:
            self.layer.setImage(self.layers[self.currentZ], autoLevels=False)
        self.win.show()
        self.show()

    def update_plot(self):
        self.Ly, self.Lx, _ = self.stack[self.currentZ].shape
        if self.view == 0:
            image = self.stack[self.currentZ]
            if self.color == 0:
                if self.onechan:
                    # show single channel
                    image = self.stack[self.currentZ][:, :, 0]
                self.img.setImage(image, autoLevels=False, lut=None)
            elif self.color == 1:
                image = image.astype(np.float32).mean(axis=-1).astype(np.uint8)
                self.img.setImage(image, autoLevels=False, lut=None)
            elif self.color == 2:
                image = image.astype(np.float32).mean(axis=-1).astype(np.uint8)
                self.img.setImage(image, autoLevels=False, lut=self.cmap[0])
            elif self.color > 2:
                image = image[:, :, self.color - 3]
                self.img.setImage(image, autoLevels=False, lut=self.cmap[self.color - 2])
            self.img.setLevels(self.saturation[self.currentZ])
        else:
            image = np.zeros((self.Ly, self.Lx), np.uint8)
            if len(self.flows) >= self.view - 1 and len(self.flows[self.view - 1]) > 0:
                image = self.flows[self.view - 1][self.currentZ]
            if self.view > 1:
                self.img.setImage(image, autoLevels=False, lut=self.bwr)
            else:
                self.img.setImage(image, autoLevels=False, lut=None)
            self.img.setLevels([0.0, 255.0])
        self.scale.setImage(self.radii, autoLevels=False)
        self.scale.setLevels([0.0, 255.0])

        if self.masksOn or self.outlinesOn:
            self.layer.setImage(self.layers[self.currentZ], autoLevels=False)

        self.win.show()
        self.show()

    def compute_scale(self):

        self.diameter = float(self.Diameter.text())
        self.pr = int(float(self.Diameter.text()))

        self.radii = np.zeros((self.Ly + self.pr, self.Lx, 4), np.uint8)
        yy, xx = plot.disk([self.pr / 2 - 1, self.Ly - self.pr / 2 + 1],
                           self.pr / 2, self.Ly + self.pr, self.Lx)
        self.radii[yy, xx, 0] = 255
        self.radii[yy, xx, -1] = 255
        self.update_plot()
        self.p0.setYRange(0, self.Ly + self.pr)
        self.p0.setXRange(0, self.Lx)

    def auto_save(self):
        if self.autosaveOn:
            print('Autosaved')
            self.save_cell_list()
            iopart._save_sets(self)
            iopart._save_png(self)

    def save_all(self):
        self.save_cell_list()
        iopart._save_sets(self)
        iopart._save_png(self)
        self.state_label.setText('Save masks/npy/list successfully', color='#39B54A')


def make_cmap(cm=0):
    # make a single channel colormap
    r = np.arange(0, 256)
    color = np.zeros((256, 3))
    color[:, cm] = r
    color = color.astype(np.uint8)
    cmap = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return cmap


def interpZ(mask, zdraw):
    """ find nearby planes and average their values using grid of points
        zfill is in ascending order
    """
    ifill = np.ones(mask.shape[0], np.bool)
    zall = np.arange(0, mask.shape[0], 1, int)
    ifill[zdraw] = False
    zfill = zall[ifill]
    zlower = zdraw[np.searchsorted(zdraw, zfill, side='left') - 1]
    zupper = zdraw[np.searchsorted(zdraw, zfill, side='right')]
    for k, z in enumerate(zfill):
        Z = zupper[k] - zlower[k]
        zl = (z - zlower[k]) / Z
        plower = avg3d(mask[zlower[k]]) * (1 - zl)
        pupper = avg3d(mask[zupper[k]]) * zl
        mask[z] = (plower + pupper) > 0.33
        # Ml, norml = avg3d(mask[zlower[k]], zl)
        # Mu, normu = avg3d(mask[zupper[k]], 1-zl)
        # mask[z] = (Ml + Mu) / (norml + normu)  > 0.5
    return mask, zfill


def avg3d(C):
    """ smooth value of c across nearby points
        (c is center of grid directly below point)
        b -- a -- b
        a -- c -- a
        b -- a -- b
    """
    Ly, Lx = C.shape
    # pad T by 2
    T = np.zeros((Ly + 2, Lx + 2), np.float32)
    M = np.zeros((Ly, Lx), np.float32)
    T[1:-1, 1:-1] = C.copy()
    y, x = np.meshgrid(np.arange(0, Ly, 1, int), np.arange(0, Lx, 1, int), indexing='ij')
    y += 1
    x += 1
    a = 1. / 2  # /(z**2 + 1)**0.5
    b = 1. / (1 + 2 ** 0.5)  # (z**2 + 2)**0.5
    c = 1.
    M = (b * T[y - 1, x - 1] + a * T[y - 1, x] + b * T[y - 1, x + 1] +
         a * T[y, x - 1] + c * T[y, x] + a * T[y, x + 1] +
         b * T[y + 1, x - 1] + a * T[y + 1, x] + b * T[y + 1, x + 1])
    M /= 4 * a + 4 * b + c
    return M


def make_bwr():
    # make a bwr colormap
    b = np.append(255 * np.ones(128), np.linspace(0, 255, 128)[::-1])[:, np.newaxis]
    r = np.append(np.linspace(0, 255, 128), 255 * np.ones(128))[:, np.newaxis]
    g = np.append(np.linspace(0, 255, 128), np.linspace(0, 255, 128)[::-1])[:, np.newaxis]
    color = np.concatenate((r, g, b), axis=-1).astype(np.uint8)
    bwr = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return bwr


def make_spectral():
    # make spectral colormap
    r = np.array(
        [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108,
         112, 116, 120, 124, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 120,
         112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67,
         71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163,
         167, 171, 175, 179, 183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255])
    g = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 0, 7, 15, 23,
         31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111, 119, 127, 135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215,
         223, 231, 239, 247, 255, 247, 239, 231, 223, 215, 207, 199, 191, 183, 175, 167, 159, 151, 143, 135, 128, 129,
         131, 132, 134, 135, 137, 139, 140, 142, 143, 145, 147, 148, 150, 151, 153, 154, 156, 158, 159, 161, 162, 164,
         166, 167, 169, 170, 172, 174, 175, 177, 178, 180, 181, 183, 185, 186, 188, 189, 191, 193, 194, 196, 197, 199,
         201, 202, 204, 205, 207, 208, 210, 212, 213, 215, 216, 218, 220, 221, 223, 224, 226, 228, 229, 231, 232, 234,
         235, 237, 239, 240, 242, 243, 245, 247, 248, 250, 251, 253, 255, 251, 247, 243, 239, 235, 231, 227, 223, 219,
         215, 211, 207, 203, 199, 195, 191, 187, 183, 179, 175, 171, 167, 163, 159, 155, 151, 147, 143, 139, 135, 131,
         127, 123, 119, 115, 111, 107, 103, 99, 95, 91, 87, 83, 79, 75, 71, 67, 63, 59, 55, 51, 47, 43, 39, 35, 31, 27,
         23, 19, 15, 11, 7, 3, 0, 8, 16, 24, 32, 41, 49, 57, 65, 74, 82, 90, 98, 106, 115, 123, 131, 139, 148, 156, 164,
         172, 180, 189, 197, 205, 213, 222, 230, 238, 246, 254])
    b = np.array(
        [0, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111, 119, 127, 135, 143, 151, 159, 167, 175, 183, 191,
         199, 207, 215, 223, 231, 239, 247, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 251, 247, 243, 239,
         235, 231, 227, 223, 219, 215, 211, 207, 203, 199, 195, 191, 187, 183, 179, 175, 171, 167, 163, 159, 155, 151,
         147, 143, 139, 135, 131, 128, 126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94,
         92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38,
         36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 16, 24, 32, 41, 49, 57, 65, 74, 82, 90, 98, 106, 115, 123, 131,
         139, 148, 156, 164, 172, 180, 189, 197, 205, 213, 222, 230, 238, 246, 254])
    color = (np.vstack((r, g, b)).T).astype(np.uint8)
    spectral = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return spectral



if __name__ == '__main__':
    pass
