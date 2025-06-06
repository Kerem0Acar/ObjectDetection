import sys
import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from ultralytics import YOLO
import Database


class ObjectDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection System")
        self.setGeometry(100, 100, 1200, 800)
        
        #Style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
            QComboBox {
                background-color: #424242;
                color: white;
                border: 1px solid #666666;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid none;
                border-right: 5px solid none;
                border-top: 5px solid white;
                width: 0;
                height: 0;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #333333;
                color: white;
                selection-background-color: #0d47a1;
                selection-color: white;
                border: 1px solid #666666;
                outline: none;
                padding: 5px;
            }
            QComboBox QAbstractItemView::item {
                min-height: 25px;
                padding: 5px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #1565c0;
                color: white;
            }
            QGroupBox {
                color: white;
                font-weight: bold;
                border: 2px solid #424242;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #424242;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #424242;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0d47a1;
                border: 1px solid #0d47a1;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #1565c0;
            }
            QMessageBox {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            QMessageBox QLabel {
                color: #ffffff;
            }
            QMessageBox QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QMessageBox QPushButton:hover {
                background-color: #1565c0;
            }
        """)
        
        # Initialize variables
        self.camera = None
        self.timer = None
        self.model = None
        self.is_detecting = False
        self.detected_objects = {}  # Dictionary to store detected objects and their confidences
        
        # Create the main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setSpacing(20)  # Add spacing between panels
        
        # Left panel (camera feed and controls)
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)  # Add spacing between elements
        
        # Camera feed
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(800, 600)
        self.camera_label.setStyleSheet("""
            QLabel {
                border: 2px solid #424242;
                border-radius: 10px;
                padding: 5px;
                background-color: #1a1a1a;
            }
        """)
        left_panel.addWidget(self.camera_label)
        
        # Controls panel
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)  # Add spacing between controls
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        model_label.setStyleSheet("font-weight: bold;")
        model_layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "v1-yolo8n-100-epoches(Phone)/best.pt",
            "v2-yolo11n-150-epoches(Phone)/best.pt",
            "v3-yolo11n-200-epoches(Phone and Glasses)/best.pt",
            "v4-yolo11n-150-epoches(ToothBrush Sneakers Phone)/best.pt",
            "yolo11n.pt",
            "yolov12n.pt"
        ])
        model_layout.addWidget(self.model_combo)
        controls_layout.addLayout(model_layout)
        
        # Camera controls
        self.camera_btn = QPushButton("Start Camera")
        self.camera_btn.clicked.connect(self.toggle_camera)
        controls_layout.addWidget(self.camera_btn)
        
        self.detect_btn = QPushButton("Start Detection")
        self.detect_btn.clicked.connect(self.toggle_detection)
        self.detect_btn.setEnabled(False)
        controls_layout.addWidget(self.detect_btn)
        
        left_panel.addLayout(controls_layout)
        layout.addLayout(left_panel)
        
        # Right panel (settings and results)
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)  # Add spacing between elements
        
        # Settings group
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(10)  # Add spacing between elements
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence Threshold:")
        conf_label.setStyleSheet("font-weight: bold;")
        conf_layout.addWidget(conf_label)
        
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(50)
        conf_layout.addWidget(self.conf_slider)
        
        self.conf_label = QLabel("0.50")
        self.conf_label.setStyleSheet("min-width: 45px;")
        conf_layout.addWidget(self.conf_label)
        settings_layout.addLayout(conf_layout)
        
        self.conf_slider.valueChanged.connect(
            lambda: self.conf_label.setText(f"{self.conf_slider.value()/100:.2f}")
        )
        
        settings_group.setLayout(settings_layout)
        right_panel.addWidget(settings_group)
        
        # Results group
        results_group = QGroupBox("Detection Results")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        right_panel.addWidget(results_group)
        
        # Save results button
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
        """)
        right_panel.addWidget(self.save_btn)
        
        # Add Gather Data button next to Save Results
        self.gather_data_btn = QPushButton("Gather Data")
        self.gather_data_btn.clicked.connect(self.gather_data)
        self.gather_data_btn.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
        """)
        right_panel.addWidget(self.gather_data_btn)
        
        layout.addLayout(right_panel)
        
    def toggle_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)

            if not self.camera.isOpened():
                QMessageBox.critical(self, "Error", "Could not open camera!")
                self.camera = None
                return
            
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)  # 30ms = ~30 fps
            self.camera_btn.setText("Stop Camera")
            self.detect_btn.setEnabled(True)
        else:
            self.timer.stop()
            self.camera.release()
            self.camera = None
            self.camera_label.clear()
            self.camera_btn.setText("Start Camera")
            self.detect_btn.setEnabled(False)
            self.is_detecting = False
            self.detect_btn.setText("Start Detection")
    
    def toggle_detection(self):
        if not self.is_detecting:
            model_path = self.model_combo.currentText()
            try:
                self.model = YOLO(model_path)
                self.is_detecting = True
                self.detect_btn.setText("Stop Detection")
                self.model_combo.setEnabled(False)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load model: {str(e)}")
        else:
            self.is_detecting = False
            self.detect_btn.setText("Start Detection")
            self.model_combo.setEnabled(True)
            self.model = None
    
    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            if self.is_detecting and self.model:
                # Perform detection
                results = self.model.predict(
                    frame,
                    conf=self.conf_slider.value()/100,
                    show=False
                )
                
                # Draw results
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        label = f"{result.names[int(cls)]} {conf:.2f}"
                        
                        # Draw rectangle and label
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, label, (int(x1), int(y1)-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Update results text
                self.update_results(results)

                        # Convert frame to Qt format and display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_image = qt_image.scaled(self.camera_label.size(), Qt.KeepAspectRatio)            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_image = qt_image.scaled(self.camera_label.size(), Qt.KeepAspectRatio)
            self.camera_label.setPixmap(QPixmap.fromImage(scaled_image))
    
    def update_results(self, results):
        # Update the detected_objects dictionary with new results
        for result in results:
            for box in result.boxes:
                cls = result.names[int(box.cls[0])]
                conf = float(box.conf[0])
                self.detected_objects[cls] = conf
        
        # Create text from the dictionary
        text = ""
        for obj, conf in self.detected_objects.items():
            text += f"{obj} : {conf:.2f}\n"
        
        self.results_text.setText(text)
    
    def save_results(self):
        Database.creating_table()
        text = self.results_text.toPlainText()
        if not text:
            QMessageBox.information(self, "Failed", "Text cannot be empty!")
        else:
            lines = text.splitlines()

            for line in lines:
                if ":" in line:
                    obj, acc = line.split(":", 1)
                    Database.inserting_table(obj, acc)

            QMessageBox.information(self, "Success", "Results saved successfully!")

    def gather_data(self):
        Database.gathering_objects()
        print("All data has been gathered.")
        QMessageBox.information(self,"Data gathering is successful")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = ObjectDetectionGUI()
    window.show()
    sys.exit(app.exec_())