import test2
from PyQt5.QtWidgets import QApplication, QMainWindow


class SecondWindowActions(test2.Ui_MainWindow, QMainWindow):

    def __init__(self):
        super(test2.Ui_MainWindow, self).__init__()
        #调用 setupUi 方法来初始化和设置这个窗口类的用户界面。这里的 self 指的是主窗口类的实例
        self.setupUi(self)