# -*- coding: utf-8 -*-
import os

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cellPoseUI.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

import os
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontDatabase

from scellsegUi  import Ui_MainWindow


class scellsegGui(Ui_MainWindow):
    def __init__(self, image=None, parent = None):
        super(scellsegGui, self).__init__(parent)
        self.setupUi(self)
        self.splitter.setSizes([500, 250])
        self.autosave = True
        self.splitter.handle(1).setAttribute(Qt.WA_Hover, True)
        self.splitter2.handle(1).setAttribute(Qt.WA_Hover, True)


    def autosave_on(self):
        if self.SCheckBox.isChecked():
            self.autosave = True
        else:
            self.autosave = False

    def closeEvent(self, event):
        answer = QtWidgets.QMessageBox.question(self, 'Close', 'Close Scellseg',
                                                QtWidgets.QMessageBox.Yes |
                                                QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if answer == QtWidgets.QMessageBox.Yes:
            event.accept()
        elif answer == QtWidgets.QMessageBox.No:
            event.ignore()


if __name__ == "__main__":
    import sys

    Translucent = 'rgba(255,255,255,0)'
    Primary = '#fafafa'
    PrimaryLight = '#C0C0C0'
    ListColor = '#F0F0F0'

    Secondary = '#D3D3D3'
    SecondaryLight = '#D3D3D3'
    SecondaryDark = '#D3D3D3'
    SecondaryText = '#000000'
    sheet = [
        'QWidget',
        '{',
        'outline: 0;',
        'font: 11pt "文泉驿微米黑";',
        'selection-color: {0:s};'.format(SecondaryText),
        'selection-background-color: {0:s};'.format(Secondary),
        ' } ',

        # 'QSlider::groove:vertical {',
        # 'background-color: {0:s};'.format(SecondaryLight),
        # 'position: absolute;',
        # 'left: 4px; right: 4px;',
        # '}',
        # '',
        # 'QSlider::groove:horizontal{',
        # 'background-color: {0:s};'.format(SecondaryLight),
        # 'position: absolute;',
        # 'top: 4px; bottom: 4px;',
        # '}',
        # '',
        # 'QSlider::handle:vertical {',
        # 'height: 10px;',
        # 'background-color: {0:s};'.format('#A9A9A9'),
        # 'margin: 0 -4px;',
        # '}',
        # '',
        # 'QSlider::handle:horizontal{',
        # 'width: 10px;',
        # 'background-color: {0:s};'.format('#A9A9A9'),
        # 'margin: -4px 0px -4px 0px;',
        # '}',
        # '',
        # 'QSlider::add-page {',
        # 'background-color: {0:s};'.format(SecondaryLight),
        # '}',
        # '',
        # 'QSlider::sub-page {',
        # 'background-color: {0:s};'.format(SecondaryDark),
        # '}',
        #
        # 'QSlider::sub-page:horizontal',
        # '{',
        # 'background:rgba(0,255,0,0.4);',
        # '}',
        #
        # 'QRangeSlider::sub-page:horizontal',
        # '{',
        # 'background:rgba(100,255,2,0,4);',
        # '}',

        # """
        # QRangeSlider * {
        #     border: 0px;
        #     padding: 0px;
        # }
        # QRangeSlider #Head {
        #     background: #222;
        # }
        # QRangeSlider #Span {
        #     background: #393;
        # }
        # QRangeSlider #Span:active {
        #     background: #282;
        # }
        # QRangeSlider #Tail {
        #     background: #222;
        # }
        # QRangeSlider > QSplitter::handle {
        #     background: #393;
        # }
        # QRangeSlider > QSplitter::handle:vertical {
        #     height: 4px;
        # }
        # QRangeSlider > QSplitter::handle:pressed {
        #     background: #ca5;
        # }""",

        'QSplitter::handle:horizontal',
        '{',
        'width: 10px;',
        '}',
        'QSplitter::handle:vertical',
        '{',
        'height: 10px;',
        '}',
        'QSplitter::handle',
        '{',
        'background-color: {0:s};'.format(Translucent),
        '}',
        'QSplitter::handle:hover',
        '{',
        'background-color: {0:s};'.format(Secondary),
        '}',
        'QSplitter::handle:pressed',
        '{',
        'background-color: {0:s};'.format(Secondary),
        '}',

        'QTableView',
        '{',
        'background-color: {0:s};'.format(ListColor),
        'border-style: none;',
        '}',
        'QHeaderView',
        '{',
        'background-color: {0:s};'.format(ListColor),
        '}',
        'QHeaderView::section',
        '{',
        'background-color: {0:s};'.format(ListColor),
        '}',

        'QMenuBar',
        '{',
        'background-color: {0:s};'.format(Primary),
        'border-width: 1px;',
        'border-style: none;',
        'border-color: {0:s};'.format(SecondaryDark),
        'color: {0:s};'.format(SecondaryText),
        'margin: 0px;',
        '}',
        'QMenuBar::item:selected',
        '{',
        'background-color: {0:s};'.format(Secondary),
        'color: {0:s};'.format(SecondaryText),
        '}',
        'QMenu',
        '{',
        'background-color:{0:s};'.format(PrimaryLight),
        'border-width: 2px;',
        'border-style: solid;',
        'border-color: {0:s};'.format(SecondaryDark),
        'margin: 0px;',
        '}',
        'QMenu::separator'
        '{',
        'height: 2px;'
        'background-color: {0:s};'.format(Primary),
        'margin: 0px  2px;',
        '}',
        'QMenu::icon:checked',
        '{',
        'background-color: {0:s};'.format(Secondary),
        'border-width: 1px;',
        'border-style: solid;',
        'border-color: {0:s};'.format(Primary),
        '}',
        'QMenu::item',
        '{',
        'padding: 4px 25px 4px 20px;',
        '}',
        'QMenu::item:selected',
        '{',
        'background-color: {0:s};'.format(Secondary),
        'color: {0:s};'.format(SecondaryText),
        '}',

        'QToolBox::tab',
        '{',
        'background-color: {0:s};'.format(SecondaryLight),
        'border: 2px solid #e3e3e3;',
        'padding: 5px;',
        '}',
        'QToolBox::tab:selected',
        '{',
        'background-color: {0:s};'.format(SecondaryDark),
        'color: {0:s};'.format(SecondaryText),
        'border: 2px solid #333;',
        '}',
        'QWidget#page,QWidget#page_2,QWidget#page_3',
        '{',
        'backgroundcolor:#F0F0F0;',
        # 'background-image: url(./Resource/background.jpg);',
        '}',

        'QProgressBar {',
        'border: 1px solid rgb(0,0,0);',
        'border-radius: 2px;',
        'background-color: {0:s};'.format(SecondaryLight),
        '}',

        'QProgressBar::chunk {',
        'border: 1px solid rgb(0,0,0);',
        'border-radius: 0px;',
        'background-color: {0:s};'.format(SecondaryDark),
        'width: 10px;',
        'margin: 2px;',
        '}',

        'QLabel#jLabelPicture',
        '{',
        'border-width: 2px;',
        'border-radius: 0px;',
        'border-style: solid;',
        'border-color: {0:s};'.format(SecondaryDark),
        '}',

        'QScrollBar,QScrollBar::add-line,QScrollBar::add-page,QScrollBar::sub-line,QScrollBar::sub-page',
        '{',
        'background-color: {0:s};'.format(Translucent),
        '}',
        'QScrollBar:horizontal',
        '{',
        'height: 10px;',
        '}',
        'QScrollBar:vertical',
        '{',
        'width: 10px;',
        '}',
        'QScrollBar::handle',
        '{',
        'background-color: {0:s};'.format(Translucent),
        '}',
        'QScrollBar::handle:hover',
        '{',
        'background-color: {0:s};'.format(Secondary),
        '}',
        'QScrollBar::handle:pressed',
        '{',
        'background-color: {0:s};'.format(Secondary),
        '}',
    ]
    app = QtWidgets.QApplication(sys.argv)
    loadedFontID = QFontDatabase.addApplicationFont(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Resource", "Font", "wqy-microhei.ttc"))


    print("Opening environment…")

    gui = scellsegGui()
    app.setStyleSheet('\n'.join(sheet))

    # gui.slider.setStyleSheet('QSlider::sub-page:horizontal{border:0px};')
    gui.show()
    sys.exit(app.exec_())