import sys
import numpy as np
import os
import cv2
import fitz
from pdf2image import convert_from_path

from infer_utils import detect_text_bboxes, detect_table_bboxes, normalize_table_detector_input, normalize_text_detector_input
from infer_utils import rescale_output, read_pdf_windowed, draw_table_struct, to_excel_file
from infer_utils import TABLE_DETECTION_CONFIG, TEXT_DETECTION_CONFIG

from PIL import Image
from PIL.ImageQt import ImageQt

from PyQt5 import QtWidgets,QtCore, QtGui, sip
from PyQt5.QtCore import QSize, Qt, pyqtSlot
from PyQt5.QtWidgets import (
    QListWidget,
    QApplication,
    QHBoxLayout,
    QLabel,
    QToolBar,
    QMainWindow,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QFormLayout,
    QScrollArea,
    QGroupBox,
    QCheckBox,
    QFrame,
    QMessageBox
)


def deleteLayout(layout):
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                deleteLayout(item.layout())
        sip.delete(layout)


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Extract Tables")
        self.setWindowIcon(QtGui.QIcon('logo.png'))
        self.selected_pages = {}
        self.checkboxes_list = []

        pagelayout = QVBoxLayout()
        button_layout = QHBoxLayout()

        self.button1 = QPushButton("Open")
        self.button1.setStyleSheet('''width:  20px;
                                    height: 30px;''')
        button_layout.addWidget(self.button1)
        self.button1.clicked.connect(self.on_click_open)

        self.lineEdit = QLineEdit()
        self.lineEdit.setEnabled(False)
        self.lineEdit.setStyleSheet('font-size: 22px;')
        button_layout.addWidget(self.lineEdit)

        self.check_all_button = QPushButton("ch/unch")
        self.check_all_button.setEnabled(True)
        self.check_all_button.setStyleSheet('''width:  20px;
                                    height: 30px;''')
        self.check_all_button.clicked.connect(self.on_click_check_all)

        button_layout.addWidget(self.check_all_button)
        
        self.extract_button = QPushButton("Extract")
        self.extract_button.setEnabled(False)
        self.extract_button.setStyleSheet('''width:  20px;
                                    height: 30px;''')
        self.extract_button.clicked.connect(self.on_click_extract)

        button_layout.addWidget(self.extract_button)

        pagelayout.addLayout(button_layout)

        scroll_layout = QHBoxLayout()

        self.formLayout = QFormLayout()
        self.groupBox = QGroupBox()
        self.groupBox.setStyleSheet("background-color:#ffffdf;")
        
        # self.listwidget = QtWidgets.QListWidget()
        # self.listwidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        self.setCentralWidget(self.groupBox)
        # add some items)
        scroll = QScrollArea()
        scroll.setWidget(self.groupBox)
        scroll.setWidgetResizable(True)
        #scroll.setFixedWidth(940)
        scroll_layout.addWidget(scroll)

        self.formLayout2 = QFormLayout()
        self.groupBox2 = QGroupBox()
        self.groupBox2.setStyleSheet("background-color:#9ddfff;")
        
        # self.listwidget = QtWidgets.QListWidget()
        # self.listwidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        self.setCentralWidget(self.groupBox2)
        # add some items)
        scroll = QScrollArea()
        scroll.setWidget(self.groupBox2)
        scroll.setWidgetResizable(True)
        #scroll.setFixedWidth(940)
        scroll_layout.addWidget(scroll)

        pagelayout.addLayout(scroll_layout)

        # Set the central widget of the Window.
        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)
        self.showMaximized()

    @pyqtSlot()
    def on_click_open(self):
        self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', QtCore.QDir().homePath(), '*.pdf')
        if not self.fileName:
            return

        self.fitz_doc = fitz.open(self.fileName)

        deleteLayout(self.formLayout2)
        self.formLayout2 = QFormLayout()
        self.extract_button.setEnabled(True)
        self.selected_pages = {}
        self.lineEdit.setText(self.fileName)

        deleteLayout(self.formLayout)
        self.formLayout = QFormLayout()

        self.pages = convert_from_path(pdf_path=f'{self.fileName}')
        for i in range(len(self.pages)):
            label = QLabel(self)
            page= np.array(self.pages[i])
            pImage = Image.fromarray(page)
            qtImage = ImageQt(pImage).scaled(940, int(page.shape[0] * 940 / page.shape[1]), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.pixmap = QtGui.QPixmap.fromImage(qtImage)
            self.pixmap=self.pixmap.scaledToWidth(900)
            label.setPixmap(self.pixmap)

            self.formLayout.addRow(label)

            self.checkbox = QCheckBox(f"Page {i+1}")
            self.checkbox.setCheckState(False)
            self.checkboxes_list.append(self.checkbox)

            self.checkbox.stateChanged.connect(self.clickBox(i))
            StyleSheet = '''
                                QCheckBox {
                                    spacing: 5px;
                                    font-size:20px;     /* <--- */
                                    margin-bottom:50%;
                                }

                                QCheckBox::indicator {
                                    width:  20px;
                                    height: 20px;
                                    color : white;
                                }
                                '''  
            self.checkbox.setStyleSheet(StyleSheet)
            # self.list_selecte_items[i].clicked.connect(self.list_selecte_items[i])
            # comboList[2*i])
            self.formLayout.addRow(self.checkbox)
            self.line = QFrame()
            self.line.setFrameShape(QFrame.HLine)
            self.line.setFrameShadow(QFrame.Sunken)
            self.line.setMidLineWidth(20)

            self.formLayout.addRow(self.line)

        self.groupBox.setLayout(self.formLayout)

    def on_click_check_all(self):
        if len(self.checkboxes_list) > 0:
            is_checked = not self.checkboxes_list[0].isChecked()
            for cb in self.checkboxes_list:
                cb.setChecked(is_checked)

        # for cb in self.checkboxes_list:
        #     cb.setChecked(False)

    def clickBox(self, index):
        def inner_func(state):
            if state == QtCore.Qt.Checked:
                self.selected_pages[index] = self.pages[index]
                # print(f'Checked {index}.')
            else:
                # if self.selected_pages.get(index):
                del self.selected_pages[index]
                # print(f'Unchecked {index}.')
            # print(self.list_of_pages)

        return inner_func

    @pyqtSlot()
    def on_click_extract(self):

        deleteLayout(self.formLayout2)
        self.formLayout2 = QFormLayout()
        # if self.list_of_pages == []:
        #     msgBox = QMessageBox()
        #     msgBox.setIcon(QMessageBox.Information)
        #     msgBox.setText("Please select one or more pages.")
        #     msgBox.setWindowTitle("No file to extract")
        #     msgBox.setStandardButtons(QMessageBox.Ok)
        #     msgBox.exec()
        #     return

        print("predicting...")
        for page_i, pil_img in self.selected_pages.items():
            orig_img = np.array(pil_img)
            normed_page_img = normalize_table_detector_input(cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY), resize_shape=TABLE_DETECTION_CONFIG["input_shape"])
            table_bboxes = detect_table_bboxes(normed_page_img)

            rescaled_bboxes = rescale_output(table_bboxes, normed_page_img.shape[:2], orig_img.shape)

            post_orig_img = orig_img.copy()

            for box_i, table_bbox in enumerate(rescaled_bboxes):
                table_img = orig_img[round(table_bbox[1]): round(table_bbox[3]), round(table_bbox[0]): round(table_bbox[2])]

                cv2.rectangle(post_orig_img, 
                    (round(table_bbox[0]), round(table_bbox[1])), 
                    (round(table_bbox[2]), round(table_bbox[3])),
                    color=(200, 0, 0), thickness=3
                )

            label = QLabel(self)
            pImage = Image.fromarray(post_orig_img)
            qtImage = ImageQt(pImage).scaled(940, int(post_orig_img.shape[0] * 940 / post_orig_img.shape[1]), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.pixmap = QtGui.QPixmap.fromImage(qtImage)
            self.pixmap = self.pixmap.scaledToWidth(900)
            label.setPixmap(self.pixmap)

            self.formLayout2.addRow(label)

                # normed_table_img = normalize_text_detector_input(table_img, TEXT_DETECTION_CONFIG["input_shape"])
                # # print(normed_talbe_img.shape)
                # # print(normed_table_img.shape, table_img.shape)
                
                # text_bboxes = detect_text_bboxes(normed_table_img)
                # rescaled_text_bboxes = rescale_output(text_bboxes, normed_table_img.shape[1:][::-1], table_img.shape)

                # for i in range(len(rescaled_text_bboxes)):
                #     rescaled_text_bboxes[i][0] += table_bbox[0]
                #     rescaled_text_bboxes[i][1] += table_bbox[1]
                #     rescaled_text_bboxes[i][2] += table_bbox[0]
                #     rescaled_text_bboxes[i][3] += table_bbox[1]

                # rescaled_pdf_text_bboxes = rescale_output(rescaled_text_bboxes, orig_img.shape, (self.fitz_doc[page_i].mediabox[3], self.fitz_doc[page_i].mediabox[2]))
                # for b1, b2 in zip(rescaled_text_bboxes, rescaled_pdf_text_bboxes):
                #     print(b1, b2)
                # print(rescaled_pdf_text_bboxes, fitz_doc[page_i-1].mediabox, table_img.shape)

                # to_excel_file(rescaled_pdf_text_bboxes, fitz_doc[page_i-1], bpdf_name, page_i, box_i+1)
                # print(f"Page {page_i} - saved table {box_i+1}.")

            table_page = QLineEdit()
            table_page.setEnabled(False)
            table_page.setStyleSheet('font-size: 20px;')
            table_page.setText(f"Page {page_i}.")

            self.formLayout2.addRow(table_page)

            self.line = QFrame()
            self.line.setFrameShape(QFrame.HLine)
            self.line.setFrameShadow(QFrame.Sunken)
            self.line.setMidLineWidth(20)

            self.formLayout2.addRow(self.line)

        self.groupBox2.setLayout(self.formLayout2)
    
app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()