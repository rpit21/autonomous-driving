import cv2
import os

video_dir = "data/bddv/videos"
videos = os.listdir(video_dir)

print("Videos encontrados:", videos)

video_path = os.path.join(video_dir, videos[0])
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("No se pudo leer el video")

print("Primer frame shape:", frame.shape)
