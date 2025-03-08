import cv2
import numpy as np
import time
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# Load YOLOv8 model for face detection
face_model = YOLO("yolov8n-face.pt")  # You need to download this model

# Load InsightFace for recognition
face_recog = FaceAnalysis(name="buffalo_l")
face_recog.prepare(ctx_id=-1)  # Runs on CPU

# Track time user looks away
look_away_start = None
LOOK_AWAY_THRESHOLD = 20  # seconds

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # YOLOv8 face detection
    results = face_model(frame)
    faces = results[0].boxes.xyxy.cpu().numpy()  # Get face bounding boxes

    if len(faces) > 1:
        cv2.putText(frame, "Multiple Faces Detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for (x1, y1, x2, y2) in faces:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Face recognition using InsightFace
        face_crop = frame[y1:y2, x1:x2]
        face_info = face_recog.get(face_crop)

        if face_info:
            face_id = face_info[0]["embedding"]
            cv2.putText(frame, "Face Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Proctoring', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
