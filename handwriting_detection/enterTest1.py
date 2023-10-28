import sys

import enterTest2
import test1
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox
from PyQt5 import QtWidgets


# 这里定义的第一个界面的后端代码需要继承两个类
class FirstWindowActions(test1.Ui_MainWindow, QMainWindow):

    def __init__(self):
        super(test1.Ui_MainWindow, self).__init__()
        # 创建界面
        self.setupUi(self)
        self.pushButton.clicked.connect(self.click_login_button)

    def click_login_button(self):
        """点击登录按钮，跳转到相应界面"""

        # 实例化第二个界面的后端类，并对第二个界面进行显示
        self.scend_window = enterTest2.SecondWindowActions()
        # 显示第二个界面
        self.scend_window.show()
        # 关闭第一个界面
        self.close()

# if __name__ == '__main__':
#     # 界面的入口，在这里需要定义QApplication对象，之后界面跳转时不用重新定义，只需要调用show()函数jikt
#     app = QApplication(sys.argv)
#     # 显示创建的界面
#     MainWindow = FirstWindowActions()  # 创建窗体对象
#     MainWindow.show()  # 显示窗体
#
#     sys.exit(app.exec_())  # 程序关闭时退出进程
'''
enterTest1文件中有关联函数，是因为关联的函数就是打开第二个界面，可以直接写在同一个Python文件中，很简单；
enterTest2文件中没有关联函数，是因为关联的函数很多，就将关联的函数和关联函数的声明另存与另一个文件。
'''