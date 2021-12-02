from PyQt5 import QtGui, QtCore, Qt, QtWidgets
from scellseg import models
import iopart as io

def mainmenu(parent):
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")

    parent.savePNG = QtGui.QAction("Save masks (*_cp_masks.png)", parent)
    parent.savePNG.setShortcut("Ctrl+M")
    parent.savePNG.triggered.connect(lambda: io._save_png_menu(parent))
    file_menu.addAction(parent.savePNG)
    parent.savePNG.setEnabled(False)

    parent.saveOutlines = QtGui.QAction("Save &outlines for imageJ (*_cp_outlines.txt)", parent)
    parent.saveOutlines.setShortcut("Ctrl+O")
    parent.saveOutlines.triggered.connect(lambda: io._save_outlines_menu(parent))
    file_menu.addAction(parent.saveOutlines)
    parent.saveOutlines.setEnabled(False)

    parent.saveSet = QtGui.QAction("&Save npy for image&&masks (*_seg.npy)", parent)
    parent.saveSet.setShortcut("Ctrl+N")
    parent.saveSet.triggered.connect(lambda: io._save_sets_menu(parent))
    file_menu.addAction(parent.saveSet)
    parent.saveSet.setEnabled(False)

    parent.saveCellList = QtGui.QAction("Save &instance list (*_instance_list.txt)", parent)
    parent.saveCellList.setShortcut("Ctrl+L")
    parent.saveCellList.triggered.connect(lambda:parent.save_cell_list_menu())
    file_menu.addAction(parent.saveCellList)
    parent.saveCellList.setEnabled(False)

    # load processed data
    loadImg = QtGui.QAction("&Load image (*.tif, *.png, *.jpg)", parent)
    loadImg.setShortcut("Ctrl+Shift+I")
    loadImg.triggered.connect(lambda: io._load_image(parent))
    file_menu.addAction(loadImg)

    parent.loadMasks = QtGui.QAction("Load &masks (*.tif, *.png, *.jpg)", parent)
    parent.loadMasks.setShortcut("Ctrl+Shift+M")
    parent.loadMasks.triggered.connect(lambda: io._load_masks(parent))
    file_menu.addAction(parent.loadMasks)
    parent.loadMasks.setEnabled(False)

    parent.loadManual = QtGui.QAction("Load npy for image&&masks (*_seg.npy)", parent)
    parent.loadManual.setShortcut("Ctrl+Shift+N")
    parent.loadManual.triggered.connect(lambda: io._load_seg(parent))
    file_menu.addAction(parent.loadManual)
    parent.loadManual.setEnabled(False)

    parent.loadCellList = QtGui.QAction("Load &instance list (*_instance_list.txt)", parent)
    parent.loadCellList.setShortcut("Ctrl+Shift+L")
    parent.loadCellList.triggered.connect(lambda: parent.load_cell_list())
    file_menu.addAction(parent.loadCellList)
    parent.loadCellList.setEnabled(False)

    #loadStack = QtGui.QAction("Load &numpy z-stack (*.npy nimgs x nchan x pixels x pixels)", parent)
    #loadStack.setShortcut("Ctrl+N")
    #loadStack.triggered.connect(lambda: parent.load_zstack(None))
    #file_menu.addAction(loadStack)


def editmenu(parent):
    main_menu = parent.menuBar()
    edit_menu = main_menu.addMenu("&Edit")

    parent.remcell = QtGui.QAction('Remove selected cell (Ctrl+CLICK)', parent)
    parent.remcell.setShortcut("Ctrl+Click")
    parent.remcell.triggered.connect(parent.remove_action)
    # parent.remcell.setEnabled(False)
    edit_menu.addAction(parent.remcell)

    parent.undo = QtGui.QAction('Undo previous mask/trace', parent)
    parent.undo.setShortcut("Ctrl+Z")
    parent.undo.triggered.connect(parent.undo_action)
    # parent.undo.setEnabled(False)
    edit_menu.addAction(parent.undo)

    parent.redo = QtGui.QAction('Undo remove mask', parent)
    parent.redo.setShortcut("Ctrl+Y")
    parent.redo.triggered.connect(parent.undo_remove_action)
    # parent.redo.setEnabled(False)
    edit_menu.addAction(parent.redo)

    parent.ClearButton = QtGui.QAction('Clear all masks', parent)
    parent.ClearButton.setShortcut("Ctrl+0")
    parent.ClearButton.triggered.connect(parent.clear_all)
    # parent.ClearButton.setEnabled(False)
    edit_menu.addAction(parent.ClearButton)


def helpmenu(parent):
    main_menu = parent.menuBar()
    help_menu = main_menu.addMenu("&Help")
    
    checkMKL = QtGui.QAction("Check CPU MKL", parent)
    checkMKL.triggered.connect(lambda: models.check_mkl(parent))
    help_menu.addAction(checkMKL)


    openHelp = QtGui.QAction("&Help window", parent)
    openHelp.setShortcut("Ctrl+H")
    openHelp.triggered.connect(parent.help_window)
    help_menu.addAction(openHelp)

    openGUI = QtGui.QAction("&GUI layout", parent)
    openGUI.setShortcut("Ctrl+G")
    openGUI.triggered.connect(parent.gui_window)
    help_menu.addAction(openGUI)