import cv2

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Kamera açılamadı!")
    exit()

while True:
    success, frame = cam.read()
    if not success:
        break

    cv2.imshow("Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()