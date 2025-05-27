import cv2
import AddNotepad
from ultralytics import YOLO
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox

class FrameProcessor:
    def __init__(self, gui):
        self.gui = gui
        self.camera = None
        self.timer = None
        self.model = None
        self.is_detecting = False

    def toggle_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                QMessageBox.critical(self.gui, "Error", "Could not open camera!")
                self.camera = None
                return
            
            self.timer = QTimer(self.gui) 
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)  # 30ms = ~30 fps
            self.gui.camera_btn.setText("Stop Camera")
            self.gui.detect_btn.setEnabled(True)
        else:
            if self.timer:
                self.timer.stop()
            if self.camera:
                self.camera.release()
            self.camera = None
            self.gui.camera_label.clear()
            self.gui.camera_btn.setText("Start Camera")
            self.gui.detect_btn.setEnabled(False)
            # If camera stops, detection should also stop and UI reset
            if self.is_detecting:
                self.is_detecting = False
                self.gui.detect_btn.setText("Start Detection")
                self.gui.model_combo.setEnabled(True)
                self.model = None # Release model

    def toggle_detection(self):
        if not self.gui.camera: # Camera must be active to start detection
            QMessageBox.warning(self.gui, "Warning", "Please start the camera first.")
            return

        if not self.is_detecting:
            model_path = self.gui.model_combo.currentText()
            try:
                self.model = YOLO(model_path)
                self.is_detecting = True
                self.gui.detect_btn.setText("Stop Detection")
                self.gui.model_combo.setEnabled(False)
            except Exception as e:
                QMessageBox.critical(self.gui, "Error", f"Could not load model: {str(e)}")
                self.model = None # Ensure model is None if loading failed
        else:
            self.is_detecting = False
            self.gui.detect_btn.setText("Start Detection")
            self.gui.model_combo.setEnabled(True)
            self.model = None # Release model

    def update_frame(self):
        if not self.camera or not self.camera.isOpened():
            # This case should ideally be handled by stopping the timer if camera becomes unavailable
            return
        
        ret, frame = self.camera.read()
        if not ret:
            # Handle frame read failure, e.g., stop timer and notify user
            if self.timer:
                self.timer.stop()
            self.camera.release()
            self.camera = None
            self.gui.camera_label.clear()
            self.gui.camera_btn.setText("Start Camera")
            self.gui.detect_btn.setEnabled(False)
            if self.is_detecting: # Reset detection state as well
                self.is_detecting = False
                self.gui.detect_btn.setText("Start Detection")
                self.gui.model_combo.setEnabled(True)
                self.model = None
            QMessageBox.warning(self.gui, "Camera Error", "Failed to read frame from camera. Camera has been stopped.")
            return

        display_frame = frame.copy() # Start with a copy of the raw frame

        if self.is_detecting and self.model:
            results = self.model.predict(
                source=frame, # Use 'source=' for clarity if supported, else just frame
                conf=self.gui.conf_slider.value()/100,
                show=False # Ensure plots are not shown by YOLO directly
            )

            # yolov8 results object is a list, take the first one for a single image
            if results and results[0]:
                annotated_frame_yolo = results[0].plot() # This returns a numpy array with annotations
                display_frame = annotated_frame_yolo # Use YOLO's annotated frame
                self.gui.update_results(results) 
            # Fallback if results[0].plot() is not as expected or no detections
            # The original code drew boxes manually, results[0].plot() is more convenient if it works well.
            # If results[0].plot() already includes labels and boxes, manual drawing is redundant.
            # Keeping manual drawing logic commented out for now, assuming .plot() is sufficient.
            """
            for result in results: # This loop might be for multiple images in a batch
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls_idx = int(box.cls[0].cpu().numpy())
                    label = f"{result.names[cls_idx]} {conf:.2f}"
                    
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(display_frame, label, (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            self.gui.update_results(results)
            """

        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        if self.gui.camera_label.width() > 0 and self.gui.camera_label.height() > 0:
            scaled_image = qt_image.scaled(self.gui.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.gui.camera_label.setPixmap(QPixmap.fromImage(scaled_image))
        else:
            # Fallback: if label size is 0, set pixmap directly, it might not be visible yet.
            self.gui.camera_label.setPixmap(QPixmap.fromImage(qt_image))

# Keep the existing run function if it's used elsewhere.
# If AddNotepad is needed by run(), its import should be uncommented.
import AddNotepad # Assuming this is needed for the run function below

def run(model_path_arg): # Renamed 'model' to 'model_path_arg' to avoid clash if self.model is used
    cap = cv2.VideoCapture(0)

    # Ensure this model loading is what's intended for this separate 'run' function
    # This 'model' is local to this function.
    model_yolo = YOLO(model_path_arg) 

    if not cap.isOpened():
        print("Camera didn't open")
        exit()

    cv2.namedWindow("Kamera", cv2.WINDOW_NORMAL)
    # Consider if fullscreen is always desired for this non-GUI run
    # cv2.setWindowProperty("Kamera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    AddNotepad.createCSV()

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model_yolo.predict(frame, conf=0.5, verbose=False) # Using local model_yolo
        annotated_frame = results[0].plot()

        # This part seems specific to AddNotepad functionality
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model_yolo.names[class_id] # Using local model_yolo
            confidence = float(box.conf[0])
            AddNotepad.addNote(class_name, confidence)

        cv2.imshow("Kamera", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()