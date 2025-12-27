import cv2

video_path = "data/test_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir el video")

ret, frame = cap.read()

if not ret:
    raise RuntimeError("No se pudo leer el primer frame")

print("Frame shape:", frame.shape)

cv2.imshow("Primer frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

cap.release()
