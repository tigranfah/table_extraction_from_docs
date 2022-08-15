from fileinput import filename
import sys
import fitz
import cv2
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
from pdf2image import convert_from_path
import os
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
from PyQt5.QtGui import QPixmap, QImage


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
        self.list_of_pages = []

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


        self.button2 = QPushButton("Extract")
        self.button2.setEnabled(False)
        self.button2.setStyleSheet('''width:  20px;
                                    height: 30px;''')
        self.button2.clicked.connect(self.on_click_extract)

        button_layout.addWidget(self.button2)

        pagelayout.addLayout(button_layout)

        scroll_layout = QHBoxLayout()

        self.formLayout =QFormLayout()
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
        deleteLayout(self.formLayout2)
        self.formLayout2 = QFormLayout()
        self.button2.setEnabled(True)
        self.list_of_pages = []
        self.lineEdit.setText(self.fileName)
        deleteLayout(self.formLayout)
        self.formLayout = QFormLayout()
        comboList = []
        self.list_selecte_items = [QLabel().setPixmap(QPixmap('logo.png')) for i in range(50)]
        # filename = '0808.1802v1.1.pdf'
        self.pages = convert_from_path(pdf_path=rf'{self.fileName}',
                                        poppler_path=r"C:\poppler-0.68.0\bin")
        for i in range(len(self.pages)):
            label = QLabel(self)
            page= np.array(self.pages[i])
            pImage = Image.fromarray(page)
            qtImage = ImageQt(pImage).scaled(940, int(page.shape[0] * 940 / page.shape[1]), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.pixmap = QtGui.QPixmap.fromImage(qtImage)
            self.pixmap=self.pixmap.scaledToWidth(900)
            label.setPixmap(self.pixmap)
            # self.listwidget.addItem('')
            comboList.append(label)
            self.checkbox = QCheckBox(f"Page {i+1}")


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
            comboList.append(self.checkbox)
            # self.list_selecte_items[i].clicked.connect(self.list_selecte_items[i])
            self.formLayout.addRow(comboList[2*i])
            self.formLayout.addRow(comboList[2*i+1])
            self.line = QFrame()
            self.line.setFrameShape(QFrame.HLine)
            self.line.setFrameShadow(QFrame.Sunken)
            self.line.setMidLineWidth(20)

            self.formLayout.addRow(self.line)

        self.groupBox.setLayout(self.formLayout)

    def clickBox(self, index):
        def inner_func(state):
            if state == QtCore.Qt.Checked:
                self.list_of_pages.append(self.pages[index])
                print(f'Checked {index}.')
            else:
                self.list_of_pages.remove(self.pages[index])
                print(f'Unchecked {index}.')
            # print(self.list_of_pages)

        return inner_func

    @pyqtSlot()
    def on_click_extract(self):

        deleteLayout(self.formLayout2)
        self.formLayout2 = QFormLayout()
        print(self.list_of_pages)
        if self.list_of_pages == []:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText("Please select one or more pages.")
            msgBox.setWindowTitle("No file to extract")
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.exec()
            return

        filenames = [r'C:\Users\user\PycharmProjects\Get_text_from_tables\img\0704.2596v1.2.png', r'C:\Users\user\PycharmProjects\Get_text_from_tables\img\0705.1956v1.25.png']
        
        comboList = []

        for i in range(len(filenames)):
          label = QLabel(self)
          self.pixmap = QtGui.QPixmap(filenames[i])
          self.pixmap=self.pixmap.scaledToWidth(900)
          label.setPixmap(self.pixmap)
          comboList.append(label)
          table_page = QLineEdit()
          table_page.setEnabled(False)
          table_page.setStyleSheet('font-size: 20px;')
          table_page.setText(f"Page {1}, Table {i+1}")
          comboList.append(table_page)
          self.formLayout2.addRow(comboList[2*i])
          self.formLayout2.addRow(comboList[2*i+1])
          self.line = QFrame()
          self.line.setFrameShape(QFrame.HLine)
          self.line.setFrameShadow(QFrame.Sunken)
          self.line.setMidLineWidth(20)

          self.formLayout2.addRow(self.line)
        self.groupBox2.setLayout(self.formLayout2)
'''
        for i in range(len(self.list_of_pages)):
          label = QLabel(self)
          page= np.array(self.list_of_pages[i])
          pImage = Image.fromarray(page)
          qtImage = ImageQt(pImage).scaled(940, int(page.shape[0] * 940 / page.shape[1]), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
          self.pixmap = QtGui.QPixmap.fromImage(qtImage)
          self.pixmap=self.pixmap.scaledToWidth(900)
          label.setPixmap(self.pixmap)
          # self.listwidget.addItem('')
          comboList.append(label)
          # self.list_selecte_items[i].clicked.connect(self.list_selecte_items[i])
          table_page = QLineEdit()
          table_page.setEnabled(False)
          table_page.setStyleSheet('font-size: 20px;')
          table_page.setText(f"Page {i+1}")
          comboList.append(table_page)
          self.formLayout2.addRow(comboList[2*i])
          self.formLayout2.addRow(comboList[2*i+1])
          self.line = QFrame()
          self.line.setFrameShape(QFrame.HLine)
          self.line.setFrameShadow(QFrame.Sunken)
          self.line.setMidLineWidth(20)

          self.formLayout2.addRow(self.line)

        self.groupBox2.setLayout(self.formLayout2)
'''
    
app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()