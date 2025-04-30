import  cv2
import AddNotepad
from ultralytics import YOLO

cap = cv2.VideoCapture(0)

model = YOLO("Model1/best.pt")

if not cap.isOpened():
    print("Camera didn't open")
    exit()

cv2.namedWindow("Kamera",cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Kamera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

AddNotepad.createCSV()
AddNotepad.addNote("Phone", 0.84)


while True:
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame, conf=0.4,         verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("Kamera",annotated_frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()