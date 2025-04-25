import  cv2
import AddNotepad

cap = cv2.VideoCapture(0)

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

    cv2.imshow("Kamera", frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()