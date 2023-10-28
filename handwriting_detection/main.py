# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtWidgets import QApplication

from enterTest1 import FirstWindowActions
# 从enterTest1里导入FirstWindowActions功能(函数)

if __name__ == '__main__':
    # 界面的入口，在这里需要定义QApplication对象，之后界面跳转时不用重新定义，只需要调用show()函数jike
    app = QApplication(sys.argv)
    # 显示创建的界面
    MainWindow = FirstWindowActions()  # 创建窗体对象
    MainWindow.show()  # 显示窗体

    sys.exit(app.exec_())  # 程序关闭时退出进程