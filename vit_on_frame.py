import cv2
import torch
from transformers import ViTImageProcessor, ViTModel

# 1. Cargar ViT preentrenado
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)
model.eval()

# 2. Leer un frame del video
video_path = "data/test_video.mp4"
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("No se pudo leer el frame")

# OpenCV usa BGR, ViT espera RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# 3. Preprocesar para ViT
inputs = processor(images=frame_rgb, return_tensors="pt")

# 4. Forward pass (sin gradientes)
with torch.no_grad():
    outputs = model(**inputs)

# 5. Extraer embedding
last_hidden_state = outputs.last_hidden_state
cls_embedding = last_hidden_state[:, 0, :]  # token CLS

print("Embedding shape:", cls_embedding.shape)
