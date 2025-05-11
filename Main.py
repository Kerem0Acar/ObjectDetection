import  cv2
import AddNotepad
from ultralytics import YOLO
import time

cap = cv2.VideoCapture(0)

model1 = r"v1-yolo8n-100-epoches/best.pt"
model2 = r"v2-yolo11n-150-epoches/best.pt"
model3 = r"v3-yolo11n-200-epoches/best.pt"


model = YOLO("yolo11n.pt")

if not cap.isOpened():
    print("Camera didn't open")
    exit()

cv2.namedWindow("Kamera",cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Kamera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

AddNotepad.createCSV()

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame, conf=0.5,verbose=False)
    annotated_frame = results[0].plot()

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        AddNotepad.addNote(class_name,confidence)

    cv2.imshow("Kamera",annotated_frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()