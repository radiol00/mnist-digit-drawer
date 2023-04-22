from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtGui import QImage, qRgb, QPixmap, QMouseEvent, QPainter, QPen, QColor, QGradient
from PIL import Image

class MonoQWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.window_size = QSize(300, 450)
        self.setWindowTitle("Digit Drawer")
        self.setFixedSize(self.window_size)

class MonoQCanvas(QImage):
    LMB_PRESS = False
    def __init__(self, size):
        super().__init__(size, QImage.Format.Format_Grayscale8)
        self.widget = QLabel()
        self.widget.setFixedSize(size)

        self.widget.mousePressEvent = self.press
        self.widget.mouseReleaseEvent = self.release
        self.widget.mouseMoveEvent = self.move

        self.painter = QPainter(self)
        self.pen = QPen()
        self.pen.setWidth(10)
        self.pen.setColor(QColor(255, 255, 255))
        self.pen.setCapStyle(Qt.RoundCap)
        self.painter.setPen(self.pen)
            
        self.clear()
        
    def clear(self):
        self.fill(qRgb(0, 0, 0))
        self.update()
        
    def update(self):
        self.widget.setPixmap(QPixmap.fromImage(self))

    def press(self, ev: QMouseEvent):
        if ev.button() == Qt.LeftButton:
            self.LMB_PRESS = True
            x, y = ev.x(), ev.y()
            self.painter.drawPoint(x, y)
            self.update()

    def move(self, ev: QMouseEvent):
        if self.LMB_PRESS:
            x, y = ev.x(), ev.y()
            self.painter.drawPoint(x, y)
            self.update()

    def release(self, ev: QMouseEvent):
      if ev.button() == Qt.LeftButton:
            self.LMB_PRESS = False

class MonoQPredition:
    def __init__(self, label):
        self.widgetLabel = QLabel(label)
        self.preditionLabel = QLabel("-")

    def get_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.widgetLabel)
        self.widgetLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preditionLabel)
        self.preditionLabel.setAlignment(Qt.AlignCenter)

        return layout
    
    def update_predition(self, value):
        self.preditionLabel.setText(f"{value * 100:.0f}%")

class MonoQDigitDrawer:
    def __init__(self):
        self.app = QApplication([])

        self.window = MonoQWindow()
        self.canvas = MonoQCanvas(QSize(280, 280))
        self.predictButton = QPushButton("Predict")
        self.clearButton = QPushButton("Clear")
        self.predictionBarLayout = QHBoxLayout()
        self.mainLayout = QVBoxLayout()

        self.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.preditionLabels = []

        for c in self.classes:
            predition = MonoQPredition(c)
            self.predictionBarLayout.addLayout(predition.get_layout())
            self.preditionLabels.append(predition)

        self.clearButton.clicked.connect(self.clear_canvas)

        self.mainLayout.addWidget(self.canvas.widget)
        self.mainLayout.addWidget(self.clearButton)
        self.mainLayout.addStretch()
        self.mainLayout.addLayout(self.predictionBarLayout)
        self.mainLayout.addStretch()
        self.mainLayout.addWidget(self.predictButton)
        self.mainLayout.addWidget(QLabel("Tobiasz Pokorniecki 2023"))

        
        self.window.setLayout(self.mainLayout)
        
    def clear_canvas(self):
        self.canvas.clear()

    def get_image(self):
        bits = self.canvas.bits()
        bits = bits.asarray(280 * 280)
        image = Image.frombytes('L', (280, 280), bits, 'raw')
        image = image.resize((28, 28))
        return image
        
    def bind_predict_button(self, callback):
        self.predictButton.clicked.connect(callback)

    def update_predictions(self, preditions):
        for i in range(len(preditions)):
            self.preditionLabels[i].update_predition(preditions[i])

    def show(self):
        self.window.show()
        self.app.exec_()