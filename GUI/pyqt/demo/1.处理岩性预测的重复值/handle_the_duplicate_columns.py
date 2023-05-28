# -*- coding: utf-8 -*-
# @Time    : 2022/8/12 11:00
# @Author  : Dreamstar
# @File    : handle_the_duplicate_columns.py
# @Desc    : 主文件
# pyuic5 MainWindow.ui -o MainWindow.py
# PySide6-uic MainWindow.ui -o MainWindow.py

import os
import sys

import pandas as pd
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

from MainWindow import Ui_MainWindow

BASE_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))  # 找绝对路径


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.fname_path = None
        self.select_sheet = None

        self.ui = Ui_MainWindow()  # UI类的实例化()
        self.ui.setupUi(self)
        self.init_window()
        self.event_signal()

    def init_window(self):
        # 主界面
        # 选择文件和文件的目录，目录中包含默认值
        self.fname_path = os.path.join(BASE_DIR, '岩心分类方法.xlsx')
        self.ui.lineEdit_file_dir.setText(self.fname_path)
        # 选择已经读取的表格中的sheet
        self.init_get_sheet_list()
        # 一个图片显示
        pixmap = QPixmap("ref_img.png")
        self.ui.pic_label.setPixmap(pixmap)
        # 一个按钮，用于启动主要的功能，并且保存最终的结果

    def event_signal(self):
        # self.ui.___ACTION___.triggered.connect(___FUNCTION___)
        # self.ui.___BUTTON___.clicked.connect(___FUNCTION___)
        # self.ui.___COMBO_BOX___.currentIndexChanged.connect(___FUNCTION___)
        # self.ui.___SPIN_BOX___.valueChanged.connect(___FUNCTION___)
        # 自定义信号.属性名.connect(___FUNCTION___)
        self.ui.Button_select_file.clicked.connect(self.event_open_config_file_click)
        self.ui.Button_start.clicked.connect(self.event_handle_excel)
        self.ui.comboBox_select_sheet.activated[str].connect(self.event_upgrade_choose_sheet_list)

    def init_get_sheet_list(self):
        df = pd.read_excel(self.fname_path, sheet_name=None)
        sheet_list = list(df.keys())
        self.ui.comboBox_select_sheet.clear()
        for text in sheet_list:
            self.ui.comboBox_select_sheet.addItem(str(text))
        self.select_sheet = sheet_list[0]

    def event_open_config_file_click(self):
        fname = QFileDialog.getOpenFileName(self, '岩心分类方法.xlsx', BASE_DIR, filter='*.xls;*.xlsx')
        if fname[0]:
            self.ui.lineEdit_file_dir.setText(fname[0])  # 配置文件地址写到lineEdit控件中
            self.fname_path = fname[0]
            self.init_get_sheet_list()


    def event_upgrade_choose_sheet_list(self, text):
        print(text)
        self.select_sheet = text  # 更新已选择的sheet

    def event_handle_excel(self):
        try:
            df_data = pd.read_excel(self.fname_path, sheet_name=self.select_sheet, header=0)

            df_data = df_data.iloc[1:, :]

            temp_before = df_data['岩性'].iloc[0]
            start_depth = df_data['开始深度'].iloc[0]
            df_output = pd.DataFrame([], columns=df_data.columns)
            for idx in range(df_data.shape[0]):
                temp_now = df_data['岩性'].iloc[idx]
                if temp_now != temp_before:
                    df = df_data.iloc[idx-1:idx].copy()
                    df['开始深度'].loc[idx] = start_depth
                    df_output = pd.concat([df_output, df], axis=0)
                    start_depth = df_data['开始深度'].iloc[idx]
                    temp_before = df_data['岩性'].iloc[idx]
            df_output.to_csv("test.csv", index=False, encoding='utf_8_sig')
            print("成功保存至test.csv")
        except:
            print("请选择格式正确的sheet")


if __name__ == '__main__':
    app = QApplication([])  # 启动一个应用
    window = MainWindow()  # 实例化主窗口
    window.show()  # 展示主窗口
    app.exec()  # 避免程序执行到这一行后直接退出
