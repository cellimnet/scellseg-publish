import PyQt5.QtWidgets
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.Qt import *


import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph import Point
import numpy as np
import os
import pathlib

def horizontal_slider_style():
    return """QSlider::groove:horizontal {
            border: 1px solid #bbb;
            background: black;
            height: 10px;
            border-radius: 4px;
            }

            QSlider::sub-page:horizontal {
            background: qlineargradient(x1: 0, y1: 0,    x2: 0, y2: 1,
                stop: 0 black, stop: 1 rgb(150,255,150));
            background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,
                stop: 0 black, stop: 1 rgb(150,255,150));
            border: 1px solid #777;
            height: 10px;
            border-radius: 4px;
            }

            QSlider::add-page:horizontal {
            background: black;
            border: 1px solid #777;
            height: 10px;
            border-radius: 4px;
            }

            QSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #eee, stop:1 #ccc);
            border: 1px solid #777;
            width: 13px;
            margin-top: -2px;
            margin-bottom: -2px;
            border-radius: 4px;
            }

            QSlider::handle:horizontal:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #fff, stop:1 #ddd);
            border: 1px solid #444;
            border-radius: 4px;
            }

            QSlider::sub-page:horizontal:disabled {
            background: #bbb;
            border-color: #999;
            }

            QSlider::add-page:horizontal:disabled {
            background: #eee;
            border-color: #999;
            }

            QSlider::handle:horizontal:disabled {
            background: #eee;
            border: 1px solid #aaa;
            border-radius: 4px;
            }"""

class ExampleGUI(QtGui.QDialog):
    def __init__(self, parent=None):
        super(ExampleGUI, self).__init__(parent)
        self.setGeometry(375,200,1420,800)
        self.setFixedSize(1420,800)
        self.setWindowTitle('GUI layout')
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)

        layout = QtGui.QVBoxLayout(self)
        self.setLayout(layout)

        label = QtGui.QLabel()
        label.setScaledContents(True)
        pixmap = QtGui.QPixmap(os.path.dirname(os.path.abspath(__file__)).replace('\\', '/') + "/assets/GUI.png")
        # pixmap = pixmap.scaled(pixmap.width()/3, pixmap.height()/3)
        label.setPixmap(pixmap)
        # label.resize(pixmap.width()/3, pixmap.height()/3)
        layout.addWidget(label)

class HelpWindow(QtGui.QDialog):
    def __init__(self, parent=None):
        text = ('''
            <p class="has-line-data"> <span style="color: #5b0f00"><b>Declartion:</b></span> This software is heavily based on <a href="https://github.com/MouseLand/cellpose">Cellpose</a>. Some operations are same as Cellpose, and their description are directly copied from it.</p>
            <p class="has-line-data" style="color: #5b0f00"> <strong>View and Draw options</strong> </p>
            <p class="has-line-data"> <b>Main GUI mouse controls</b> </p>
            <ul>
            <li class="has-line-data">Pan  = left-click  + drag</li>
            <li class="has-line-data">Zoom = scroll wheel (or +/= and - buttons) </li>
            <li class="has-line-data">Full view = double left-click</li>
            <li class="has-line-data">Select mask = left-click on mask</li>
            <li class="has-line-data">Delete mask = Ctrl (or COMMAND on Mac) + left-click</li>
            <li class="has-line-data">Merge masks = Alt + left-click (will merge last two)</li>
            <li class="has-line-data">Start draw mask = right-click</li>
            <li class="has-line-data">End draw mask = right-click, or return to circle at beginning</li>
            <li class="has-line-data">End draw mask = right-click, or return to circle at beginning</li>
            </ul>
            <p class="has-line-data"><span style="color: #366c1a"><b>!NOTE1!: </b></span> Overlaps in masks are NOT allowed. If you draw a mask on top of another mask, it is cropped so that it doesn???t overlap with the old mask. Masks in 2D should be single strokes (single stroke is checked). If you want to draw masks in 3D (experimental), then you can turn this option off and draw a stroke on each plane with the cell and then press ENTER. 3D labelling will fill in planes that you have not labelled so that you do not have to as densely label.</p>
            <p class="has-line-data"><span style="color: #366c1a"><b>!NOTE2!: </b></span> The default save mode is autosaving masks/npy/list in the same folder as the loaded image when you switched to next image, you can close this mode with shotcut (P) or finding the checkbox in View & Draw. You can save masks/npy/list manully with shotcut Ctrl+S.</p>
            <p class="has-line-data"><span style="color: #366c1a"><b>!NOTE3!: </b></span> Drag a image/mask or a folder is supported, for a image, we autoload its parent directory, for a mask, we autoload its corresponding image and its parent directory.</p>
            <p class="has-line-data"> </p>
            <table class="table table-striped table-bordered">
            <thead>
            <tr>
            <th align='left'>Keyboard shortcuts</th>
            <th align='left'>  &nbsp;  </th>
            <th align='left'>Description</th>
            </tr>
            </thead>
            <tbody>
            <tr>
            <td>E</td>
            <td> </td>
            <td>turn edit mask ON, in this mode, you need firstly select a mask you wanted to edit, the selected mask will be highlighted, use right-click to add pixels and Shift+right-click to delete pixels</td>
            </tr>
            <tr>
            <td>+ / -</td>
            <td> </td>
            <td>zoom in / out</td>
            </tr>
            <tr>
            <td>Ctrl+Z</td>
            <td> </td>
            <td>undo previously drawn mask/stroke</td>
            </tr>
            <tr>
            <td>Ctrl+Y</td>
            <td> </td>
            <td>undo remove mask</td>
            </tr>
            <tr>
            <td>Ctrl+0</td>
            <td> </td>
            <td>clear all masks</td>
            </tr>
            <tr>
            <td>Ctrl+1</td>
            <td> </td>
            <td>focus View and Draw box</td>
            </tr>
            <tr>
            <td>Ctrl+2</td>
            <td> </td>
            <td>focus Fine-tune box</td>
            </tr>
            <tr>
            <td>Ctrl+3</td>
            <td> </td>
            <td>focus Inference box</td>
            </tr>
            <tr>
            <td>Ctrl + ???/???</td>
            <td> </td>
            <td>cycle through images in current directory</td>
            </tr>
            <tr>
            <td>??? / ???</td>
            <td> </td>
            <td>change color (RGB/gray/red/green/blue)</td>
            </tr>
            <tr>
            <td>PAGE-UP / PAGE-DOWN</td>
            <td> </td>
            <td>change to flows and cell prob views (if segmentation computed)</td>
            </tr>
            <tr>
            <td>[ / ]</td>
            <td> </td>
            <td>increase / decrease brush size for drawing masks</td>
            </tr>
            <tr>
            <td>X</td>
            <td> </td>
            <td>turn masks ON or OFF</td>
            </tr>
            <tr>
            <td>Z</td>
            <td> </td>
            <td>turn outlines ON or OFF</td>
            </tr>
            <tr>
            <td>S</td>
            <td> </td>
            <td>turn scale disk ON or OFF</td>
            </tr>
            <tr>
            <td>P</td>
            <td> </td>
            <td>turn autosave ON or OFF</td>
            </tr>
            <tr>
            <td>See Menu-File</td>
            <td> </td>
            <td>some load and save operations are also set shotcuts, see corresponding shotcut in the file menu</td>
            </tr>
            </tbody>
            </table>
            
            <p class="has-line-data" style="color: #5b0f00"><strong>Data organization </strong></p>
            <p class="has-line-data"> You should prepare your data in one folder with your experiment name like "mito-20211116", here we call it <b>parent folder</b>. Into this folder, it should contain a <b>shot subfolder</b> and a <b>query subfolder</b>. The shot subfolder contains images and the corresponding labelled masks, the query subfolder contains the images you want to segment. Into the shot subfolder, images should be named with <b>"_img" suffix</b> and masks should be named as <b>"_masks" suffix</b>. Except the suffix, the name of image and mask should be identical, for example, 001_img and 001_masks, notably, 001_masks should not be named as 001_cp_masks or 001_img_cp_masks (You should rename your masks name after annotation). Into the query subfolder, images should be named with <b>"_img" suffix</b>.</p>
            
            <p class="has-line-data" style="color: #5b0f00"><strong>Fine-tune options (2D only) </strong></p>
            <ul>
            <li class="has-line-data"><em>Use GPU:</em> &nbsp; If you have specially installed the cuda version of pytorch, you can activate this.</li>
            <li class="has-line-data"><em>Model architecture:</em> &nbsp; We provided three types of pre-trained model architecture: Scellseg, Cellpose and Hover. Try different model if you need.</li>
            <li class="has-line-data"><em>Chan to segment:</em> &nbsp; This is the channel in which the instance exist.</li>
            <li class="has-line-data"><em>Chan2 (optional):</em> &nbsp; You can provide anothor channel to help segmenting the instance.</li>
            <li class="has-line-data"><em>Fine-tune strategy:</em> &nbsp; We provide both contrastive and classic fine-tuning strageties, try different stragety if you need.</li>
            <li class="has-line-data"><em>Epoch:</em> &nbsp; The default epoch for fine-tuning is 100, which was used in our paper, you can input the appropriate epoch according to your own data.</li>
            <li class="has-line-data"><em>Batch size:</em> &nbsp; The default batch size for fine-tuning is 8, which was used in our paper, you can input the appropriate epoch according to your own GPU.</li>
            </ul>
            <p class="has-line-data"><span style="color: #366c1a"><b>!NOTE4!: </b></span> The saved path of model file after fine-tuning is shown in the bottom of display window.</p>
            
            <p class="has-line-data" style="color: #5b0f00"><strong>Inference options (2D only) </strong></p>
            <ul>
            <li class="has-line-data"><em>Dataset path button:</em> &nbsp; Click this button to choose the parent folder for fine-tuning.</li>
            <li class="has-line-data"><em>Use GPU:</em> &nbsp; If you have specially installed the cuda version of pytorch, you can activate this.</li>
            <li class="has-line-data"><em>Model architecture:</em> &nbsp; We provided three types of pre-trained model architecture: Scellseg, Cellpose and Hover. Try different model if you need.</li>
            <li class="has-line-data"><em>Chan to segment:</em> &nbsp; This is the channel in which the instance exist.</li>
            <li class="has-line-data"><em>Chan2 (optional):</em> &nbsp; You can provide anothor channel to help segmenting the instance.</li>
            <li class="has-line-data"><em>Model file button:</em> &nbsp; The default model file is the corresponding pre-trained model file of model architecture, you can use your own model file after fin-tuning by click the button.</li>
            <li class="has-line-data"><em>Reset pre-trained button:</em> &nbsp; You can reset to the corresponding pre-trained model file by click the button.</li>
            </ul>
            <p class="has-line-data"><span style="color: #366c1a"><b>!NOTE5!: </b></span> The model architecture you choose should be same as the model file used when fine-tuning. For example, when inference, if you load a model file of Scellseg model (choose this model architecture when fine-tuning), the corresponding model architecture you choose should also be Scellseg. The channels should also be same as settings when fine-tuning.</p>

            <p class="has-line-data"><strong>Run seg for image in window </strong></p>
            <ul>
            <li class="has-line-data"><em>Cell diameter:</em> &nbsp; You can manually enter the approximate diameter for your cells, or press ???calibrate??? to let the model estimate it. The size is represented by a disk at the bottom of the view window (can turn this disk of by unchecking ???Scale disk on???).</li>
            <li class="has-line-data"><em>Inference mode:</em> &nbsp; You can select whether using resample which may improve the performance but slower.</li>
            <li class="has-line-data"><em>Invert grayscale:</em> &nbsp; You can select whether using invert grayscale.</li>
            <li class="has-line-data"><em>Model match TH:</em> &nbsp; After running segmentation, you can slide it for better performance, you can get the value by hoving on the slider. Default value is 0.4</li>
            <li class="has-line-data"><em>Cell prob TH:</em> &nbsp; After running segmentation, you can slide it for better performance, you can get the value by hoving on the slider. Default value is 0.5</li>
            </ul>

            <p class="has-line-data"><strong>Batch segmentation / Get single instance: </strong> </p> 
            <li class="has-line-data"><em>Batch size:</em> &nbsp; The default batch size for ???Batch segmentation??? is 8, which was used in our paper, you can input the appropriate epoch according to your own GPU.</li>
            <p class="has-line-data"> You can conduct batch segmentation or get single instance by providing the right data path, for "batch segmentation", data should be organized like parent folder, and for "get single instance", data should be organized like shot subfolder, here shot subfolder is not the specific "shot" folder, if just mean data should be organized like that.</p>
            <p class="has-line-data"> <span style="color: #366c1a"><b>!NOTE6!: </b></span> We did not do any experiments on 3D images, there may be some bugs.</p>
            ''')

        super(HelpWindow, self).__init__(parent)

        self.setWindowTitle('Scellseg help')
        self.win = QtGui.QWidget(self)
        self.win.setContentsMargins(0, 0, 0, 0)

        self.scrollText = QtWidgets.QScrollArea(self.win)
        self.scrollTextWidgetContents = QtWidgets.QWidget()

        label = QtWidgets.QLabel(self.scrollTextWidgetContents)
        label.setText(text)
        label.setOpenExternalLinks(True)
        label.setFont(QtGui.QFont("Arial", 8))
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignTop)
        # label.setTextFormat(QtCore.Qt.AutoText)
        # string_sheet = [
        # 'QString {'
        #     'text-indent: 16px'
        # '}'
        # ]
        # label.setStyleSheet('\n'.join(string_sheet))
        self.verticalLayout = QtWidgets.QVBoxLayout(self.scrollTextWidgetContents)
        self.verticalLayout.addWidget(label)

        self.scrollText.setWidget(self.scrollTextWidgetContents)
        self.scrollText.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)


        # self.scrollTextWidgetContents.setSizePolicy(PyQt5.QtWidgets.QSizePolicy.Expanding, PyQt5.QtWidgets.QSizePolicy.Expanding)
        # self.scrollText.setSizePolicy(PyQt5.QtWidgets.QSizePolicy.Expanding, PyQt5.QtWidgets.QSizePolicy.Expanding)

        scroll_sheet = [
        'QScrollBar::handle:vertical {'
            'background: #6589c2'
        '}'
]
        self.scrollText.setStyleSheet('\n'.join(scroll_sheet))

        layout = QtWidgets.QHBoxLayout(self.win)
        layout.addWidget(self.scrollText)


        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setWindowModality(Qt.NonModal)
        self.setGeometry(100,50,767,525)
        # self.setFixedSize(767,525)
        # self.adjustSize()
        self.show()

class TypeRadioButtons(QtGui.QButtonGroup):
    def __init__(self, parent=None, row=0, col=0):
        super(TypeRadioButtons, self).__init__()
        parent.color = 0
        self.parent = parent
        self.bstr = self.parent.cell_types
        for b in range(len(self.bstr)):
            button = QtGui.QRadioButton(self.bstr[b])
            button.setStyleSheet('color: rgb(190,190,190);')
            button.setFont(QtGui.QFont("Arial", 10))
            if b==0:
                button.setChecked(True)
            self.addButton(button, b)
            button.toggled.connect(lambda: self.btnpress(parent))
            self.parent.l0.addWidget(button, row+b,col,1,2)
        self.setExclusive(True)


    def btnpress(self, parent):
       b = self.checkedId()
       self.parent.cell_type = b

class RGBRadioButtons(QtGui.QButtonGroup):
    def __init__(self, parent=None, row=0, col=0):
        super(RGBRadioButtons, self).__init__()
        parent.color = 0
        self.parent = parent
        self.bstr = ["image", "gradXY", "cellprob", "gradZ"]
        self.dropdown = []
        for b in range(len(self.bstr)):
            button = QtGui.QRadioButton(self.bstr[b])
            if b==0:
                button.setChecked(True)
            self.addButton(button, b)
            button.toggled.connect(lambda: self.btnpress(parent))
            self.parent.gridLayout.addWidget(button, row+b,col,1,1)
        self.setExclusive(True)
        #self.buttons.

    def btnpress(self, parent):
       b = self.checkedId()
       self.parent.view = b
       if self.parent.loaded:
           self.parent.update_plot()


class ViewBoxNoRightDrag(pg.ViewBox):
    def __init__(self, parent=None, border=None, lockAspect=True, enableMouse=True, invertY=False, enableMenu=True, name=None, invertX=False):
        pg.ViewBox.__init__(self, None, border, lockAspect, enableMouse,
                            invertY, enableMenu, name, invertX)
        self.parent = parent
        self.axHistoryPointer = -1

    def keyPressEvent(self, ev):
        """
        This routine should capture key presses in the current view box.
        The following events are implemented:
        +/= : moves forward in the zooming stack (if it exists)
        - : moves backward in the zooming stack (if it exists)

        """
        ev.accept()
        if ev.text() == '-':
            self.scaleBy([1.1, 1.1])
        elif ev.text() in ['+', '=']:
            self.scaleBy([0.9, 0.9])
        else:
            ev.ignore()
    
    def mouseDragEvent(self, ev, axis=None):
        ## if axis is specified, event will only affect that axis.
        if self.parent is None or (self.parent is not None and not self.parent.in_stroke):
            ev.accept()  ## we accept all buttons

            pos = ev.pos()
            lastPos = ev.lastPos()
            dif = pos - lastPos
            dif = dif * -1

            ## Ignore axes if mouse is disabled
            mouseEnabled = np.array(self.state['mouseEnabled'], dtype=np.float)
            mask = mouseEnabled.copy()
            if axis is not None:
                mask[1-axis] = 0.0

            ## Scale or translate based on mouse button
            if ev.button() & (QtCore.Qt.LeftButton | QtCore.Qt.MidButton):
                if self.state['mouseMode'] == pg.ViewBox.RectMode:
                    if ev.isFinish():  ## This is the final move in the drag; change the view scale now
                        #print "finish"
                        self.rbScaleBox.hide()
                        ax = QtCore.QRectF(Point(ev.buttonDownPos(ev.button())), Point(pos))
                        ax = self.childGroup.mapRectFromParent(ax)
                        self.showAxRect(ax)
                        self.axHistoryPointer += 1
                        self.axHistory = self.axHistory[:self.axHistoryPointer] + [ax]
                    else:
                        ## update shape of scale box
                        self.updateScaleBox(ev.buttonDownPos(), ev.pos())
                else:
                    tr = dif*mask
                    tr = self.mapToView(tr) - self.mapToView(Point(0,0))
                    x = tr.x() if mask[0] == 1 else None
                    y = tr.y() if mask[1] == 1 else None

                    self._resetTarget()
                    if x is not None or y is not None:
                        self.translateBy(x=x, y=y)
                    self.sigRangeChangedManually.emit(self.state['mouseEnabled'])

    def setBackgroundColor(self, color):
        """
        Set the background color of the ViewBox.

        If color is None, then no background will be drawn.

        Added in version 0.9.9
        """
        self.background.setVisible(color is not None)
        self.state['background'] = color
        self.updateBackground()


class ImageDraw(pg.ImageItem):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`
    GraphicsObject displaying an image. Optimized for rapid update (ie video display).
    This item displays either a 2D numpy array (height, width) or
    a 3D array (height, width, RGBa). This array is optionally scaled (see
    :func:`setLevels <pyqtgraph.ImageItem.setLevels>`) and/or colored
    with a lookup table (see :func:`setLookupTable <pyqtgraph.ImageItem.setLookupTable>`)
    before being displayed.
    ImageItem is frequently used in conjunction with
    :class:`HistogramLUTItem <pyqtgraph.HistogramLUTItem>` or
    :class:`HistogramLUTWidget <pyqtgraph.HistogramLUTWidget>` to provide a GUI
    for controlling the levels and lookup table used to display the image.
    """

    sigImageChanged = QtCore.Signal()

    def __init__(self, image=None, viewbox=None, parent=None, **kargs):
        super(ImageDraw, self).__init__()
        self.levels = np.array([0,255])
        self.lut = None
        self.autoDownsample = False
        self.axisOrder = 'row-major'
        self.removable = False
        self.parent = parent
        self.parent.eraser_selected = 0

        self.setDrawKernel(kernel_size=self.parent.brush_size)
        self.parent.current_stroke = []
        self.parent.in_stroke = False

    def mouseClickEvent(self, ev):
        self.eraser_model = self.parent.eraser_button.isChecked()

        if self.parent.masksOn or self.parent.outlinesOn:
            if not self.eraser_model:
                if  self.parent.loaded and (ev.button()==QtCore.Qt.RightButton):
                    if not self.parent.in_stroke:
                        ev.accept()
                        self.create_start(ev.pos())
                        self.parent.stroke_appended = False
                        self.parent.in_stroke = True
                        self.drawAt(ev.pos(), ev)
                    else:
                        ev.accept()
                        self.end_stroke()
                        self.parent.in_stroke = False

            else:
                y, x = int(ev.pos().y()), int(ev.pos().x())
                size = int(self.parent.brush_size / 2)
                if self.parent.loaded:
                    if ev.button() == QtCore.Qt.RightButton:
                        if ev.modifiers() == QtCore.Qt.ShiftModifier:
                            print(self.parent.selected)

                            if self.parent.cellpix[0][int(ev.pos().y()), int(ev.pos().x())] == self.parent.selected:
                                for i in range(y- size, y+size+1):
                                    for j in range(x-size,x+size+1):
                                        if self.parent.cellpix[0][i,j] == self.parent.selected:
                                            self.parent.cellpix[0][i,j] = 0
                                            self.parent.layers[0][i,j] = 0
                                self.parent.update_plot()

                        elif self.parent.selected>0:
                            print(self.parent.selected)
                            for i in range(y - size, y + size + 1):
                                for j in range(x - size, x + size + 1):
                                    if self.parent.layers[0][i,j,-1] == 0:
                                        self.parent.cellpix[0][i,j]=self.parent.selected
                                        self.parent.layers[0][i,j,0:3] =self.parent.cellcolors[self.parent.selected].tolist()
                                        self.parent.layers[0][i, j, -1] =255
                            self.parent.update_plot()

            if not self.parent.in_stroke:
                y, x = int(ev.pos().y()), int(ev.pos().x())
                if y >= 0 and y < self.parent.Ly and x >= 0 and x < self.parent.Lx:
                    if ev.button() == QtCore.Qt.LeftButton and not ev.double():
                        idx = self.parent.cellpix[self.parent.currentZ][y, x]
                        if idx > 0:
                            if ev.modifiers() == QtCore.Qt.ControlModifier:
                                # delete mask selected
                                self.parent.remove_cell(idx)
                            elif ev.modifiers() == QtCore.Qt.AltModifier:
                                self.parent.merge_cells(idx)
                            elif self.parent.masksOn:
                                self.parent.unselect_cell()
                                self.parent.select_cell(idx)
                        elif self.parent.masksOn:
                            self.parent.unselect_cell()
                    else:
                        ev.ignore()
                        return


    def mouseDragEvent(self, ev):
        ev.ignore()
        return

    def hoverEvent(self, ev):
        if self.parent.in_stroke:
            if self.parent.in_stroke:
                # continue stroke if not at start
                self.drawAt(ev.pos())
                if self.is_at_start(ev.pos()):
                    self.end_stroke()
                    self.parent.in_stroke = False

        else:
            ev.acceptClicks(QtCore.Qt.RightButton)

    def create_start(self, pos):
        self.scatter = pg.ScatterPlotItem([pos.x()], [pos.y()], pxMode=False,
                                        pen=pg.mkPen(color=(255,255,255), width=2),
                                        size=max(3*2, self.parent.brush_size*1.8*2), brush=None)
        self.parent.p0.addItem(self.scatter)

    def is_at_start(self, pos):
        thresh_out = max(6, self.parent.brush_size*3)
        thresh_in = max(3, self.parent.brush_size*1.8)
        if len(self.parent.current_stroke) > 3:
            stroke = np.array(self.parent.current_stroke)
            dist = (((stroke[1:,1:] - stroke[:1,1:][np.newaxis,:,:])**2).sum(axis=-1))**0.5
            dist = dist.flatten()
            has_left = (dist > thresh_out).nonzero()[0]
            if len(has_left) > 0:
                first_left = np.sort(has_left)[0]
                has_returned = (dist[max(4,first_left+1):] < thresh_in).sum()
                if has_returned > 0:

                    return True
                else:
                    return False
            else:
                return False

    def end_stroke(self):
        self.parent.p0.removeItem(self.scatter)
        if not self.parent.stroke_appended:
            self.parent.strokes.append(self.parent.current_stroke)
            self.parent.stroke_appended = True
            self.parent.current_stroke = np.array(self.parent.current_stroke)
            ioutline = self.parent.current_stroke[:,3]==1
            self.parent.current_point_set.extend(list(self.parent.current_stroke[ioutline]))
            self.parent.current_stroke = []
            if self.parent.sstroke_On:
                self.parent.add_set()
        if len(self.parent.current_point_set) > 0 and self.parent.sstroke_On:
            self.parent.add_set()

    def tabletEvent(self, ev):
        pass


    def drawAt(self, pos, ev=None):
        mask = self.greenmask
        set = self.parent.current_point_set
        stroke = self.parent.current_stroke
        pos = [int(pos.y()), int(pos.x())]
        dk = self.drawKernel
        kc = self.drawKernelCenter
        sx = [0,dk.shape[0]]
        sy = [0,dk.shape[1]]
        tx = [pos[0] - kc[0], pos[0] - kc[0]+ dk.shape[0]]
        ty = [pos[1] - kc[1], pos[1] - kc[1]+ dk.shape[1]]
        kcent = kc.copy()
        if tx[0]<=0:
            sx[0] = 0
            sx[1] = kc[0] + 1
            tx    = sx
            kcent[0] = 0
        if ty[0]<=0:
            sy[0] = 0
            sy[1] = kc[1] + 1
            ty    = sy
            kcent[1] = 0
        if tx[1] >= self.parent.Ly-1:
            sx[0] = dk.shape[0] - kc[0] - 1
            sx[1] = dk.shape[0]
            tx[0] = self.parent.Ly - kc[0] - 1
            tx[1] = self.parent.Ly
            kcent[0] = tx[1]-tx[0]-1
        if ty[1] >= self.parent.Lx-1:
            sy[0] = dk.shape[1] - kc[1] - 1
            sy[1] = dk.shape[1]
            ty[0] = self.parent.Lx - kc[1] - 1
            ty[1] = self.parent.Lx
            kcent[1] = ty[1]-ty[0]-1


        ts = (slice(tx[0],tx[1]), slice(ty[0],ty[1]))
        ss = (slice(sx[0],sx[1]), slice(sy[0],sy[1]))
        self.image[ts] = mask[ss]

        for ky,y in enumerate(np.arange(ty[0], ty[1], 1, int)):
            for kx,x in enumerate(np.arange(tx[0], tx[1], 1, int)):
                iscent = np.logical_and(kx==kcent[0], ky==kcent[1])
                stroke.append([self.parent.currentZ, x, y, iscent])
        self.updateImage()

    def setDrawKernel(self, kernel_size=3):
        bs = kernel_size
        kernel = np.ones((bs,bs), np.uint8)
        self.drawKernel = kernel
        self.drawKernelCenter = [int(np.floor(kernel.shape[0]/2)),
                                 int(np.floor(kernel.shape[1]/2))]
        onmask = 255 * kernel[:,:,np.newaxis]
        offmask = np.zeros((bs,bs,1))
        opamask = 100 * kernel[:,:,np.newaxis]
        self.redmask = np.concatenate((onmask,offmask,offmask,onmask), axis=-1)
        self.greenmask = np.concatenate((onmask,offmask,onmask,opamask), axis=-1)


class RangeSlider(QtGui.QSlider):
    """ A slider for ranges.

        This class provides a dual-slider for ranges, where there is a defined
        maximum and minimum, as is a normal slider, but instead of having a
        single slider value, there are 2 slider values.

        This class emits the same signals as the QSlider base class, with the
        exception of valueChanged

        Found this slider here: https://www.mail-archive.com/pyqt@riverbankcomputing.com/msg22889.html
        and modified it
    """
    def __init__(self, parent=None, *args):
        super(RangeSlider, self).__init__(*args)

        self._low = self.minimum()
        self._high = self.maximum()

        self.pressed_control = QtGui.QStyle.SC_None
        self.hover_control = QtGui.QStyle.SC_None
        self.click_offset = 0

        self.setOrientation(QtCore.Qt.Horizontal)
        self.setTickPosition(QtGui.QSlider.TicksRight)

        self.opt = QtGui.QStyleOptionSlider()
        self.opt.orientation=QtCore.Qt.Vertical
        self.initStyleOption(self.opt)
        self.active_slider = 0
        self.parent = parent
        self.show()


    def level_change(self):
        if self.parent is not None:
            if self.parent.loaded:
                for z in range(self.parent.NZ):
                    self.parent.saturation[z] = [self._low, self._high]
                self.parent.update_plot()

    def low(self):
        return self._low

    def setLow(self, low):
        self._low = low
        self.update()

    def high(self):
        return self._high

    def setHigh(self, high):
        self._high = high
        self.update()

    def paintEvent(self, event):
        # based on http://qt.gitorious.org/qt/qt/blobs/master/src/gui/widgets/qslider.cpp
        painter = QtGui.QPainter(self)
        style = QtGui.QApplication.style()

        for i, value in enumerate([self._low, self._high]):
            opt = QtWidgets.QStyleOptionSlider()
            self.initStyleOption(opt)
            if i == 0:
                opt.subControls = QtWidgets.QStyle.SC_SliderGroove | QtWidgets.QStyle.SC_SliderHandle
            else:
                opt.subControls = QtWidgets.QStyle.SC_SliderHandle

            if self.tickPosition() != self.NoTicks:
                opt.subControls |= QtWidgets.QStyle.SC_SliderTickmarks

            if self.pressed_control:
                opt.activeSubControls = self.pressed_control
                opt.state |= QtWidgets.QStyle.State_Sunken
            else:
                opt.activeSubControls = self.hover_control

            opt.sliderPosition = value
            opt.sliderValue = value

            if i==0:
                pen = QtGui.QPen()
                pen.setBrush(QtGui.QColor('#F0F0F0'))
                pen.setCapStyle(QtCore.Qt.RoundCap)
                pen.setWidth(3)
                painter.setPen(pen)
                painter.setBrush(QtGui.QColor('#F0F0F0'))
                x1,y1,x2,y2 = event.rect().getCoords()
                painter.drawRect(event.rect())
            style.drawComplexControl(QtWidgets.QStyle.CC_Slider, opt, painter, self)
            
    def mousePressEvent(self, event):
        event.accept()
        self.parent.autobtn.setChecked(False)

        style = QtGui.QApplication.style()
        print('style', style.CC_Slider)
        button = event.button()
        if button:
            opt = QtGui.QStyleOptionSlider()
            self.initStyleOption(opt)

            self.active_slider = -1

            for i, value in enumerate([self._low, self._high]):
                opt.sliderPosition = value
                hit = style.hitTestComplexControl(style.CC_Slider, opt, event.pos(), self)
                if hit == style.SC_SliderHandle:
                    self.active_slider = i

                    self.pressed_control = hit
                    self.triggerAction(self.SliderMove)
                    self.setRepeatAction(self.SliderNoAction)
                    self.setSliderDown(True)
                    break

            if self.active_slider < 0:
                self.pressed_control = QtGui.QStyle.SC_SliderHandle
                self.click_offset = self.__pixelPosToRangeValue(self.__pick(event.pos()))
                self.triggerAction(self.SliderMove)
                self.setRepeatAction(self.SliderNoAction)
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        self.parent.autobtn.setChecked(False)


        if self.pressed_control != QtGui.QStyle.SC_SliderHandle:
            event.ignore()
            return

        event.accept()
        new_pos = self.__pixelPosToRangeValue(self.__pick(event.pos()))
        opt = QtGui.QStyleOptionSlider()
        self.initStyleOption(opt)

        if self.active_slider < 0:
            offset = new_pos - self.click_offset
            self._high += offset
            self._low += offset
            if self._low < self.minimum():
                diff = self.minimum() - self._low
                self._low += diff
                self._high += diff
            if self._high > self.maximum():
                diff = self.maximum() - self._high
                self._low += diff
                self._high += diff
        elif self.active_slider == 0:
            if new_pos >= self._high:
                new_pos = self._high - 1
            self._low = new_pos
        else:
            if new_pos <= self._low:
                new_pos = self._low + 1
            self._high = new_pos

        self.click_offset = new_pos
        self.update()

    def mouseReleaseEvent(self, event):
        self.parent.autobtn.setChecked(False)
        self.level_change()

    def __pick(self, pt):
        if self.orientation() == QtCore.Qt.Horizontal:
            return pt.x()
        else:
            return pt.y()


    def __pixelPosToRangeValue(self, pos):
        opt = QtGui.QStyleOptionSlider()
        self.initStyleOption(opt)
        style = QtGui.QApplication.style()
        gr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderGroove, self)
        sr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderHandle, self)

        if self.orientation() == QtCore.Qt.Horizontal:
            slider_length = sr.width()
            slider_min = gr.x()
            slider_max = gr.right() - slider_length + 1
        else:
            slider_length = sr.height()
            slider_min = gr.y()
            slider_max = gr.bottom() - slider_length + 1

        return style.sliderValueFromPosition(self.minimum(), self.maximum(),
                                             pos-slider_min, slider_max-slider_min,
                                             opt.upsideDown)


