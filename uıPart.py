from PyQt5.QtWidgets import *


def start():
    app = QApplication([])

    home = QWidget()
    button1 = QPushButton("Open Camera",home)
    button1.move(100,60)

    home.setWindowTitle("Object Detection")
    home.resize(900,700)
    home.show()


    app.exec_()

start()

