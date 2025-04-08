import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera didn't open")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        break

    cv2.imshow("Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()