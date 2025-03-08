import cv2
import numpy as np
import time

# Load OpenCV's built-in face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Track time user looks away
look_away_start = None
LOOK_AWAY_THRESHOLD = 20  # seconds

# Simple face encoding function to replace InsightFace
def get_face_encoding(face_img):
    # Resize for consistency
    face_img = cv2.resize(face_img, (100, 100))
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    # Flatten and normalize
    return gray.flatten() / 255.0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 1:
        cv2.putText(frame, "Multiple Faces Detected!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    for (x, y, w, h) in faces:
        # Convert coordinates to match original format
        x1, y1, x2, y2 = x, y, x+w, y+h
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Get face region
        face_crop = frame[y1:y2, x1:x2]
        
        # Only process if face is big enough
        if face_crop.shape[0] > 20 and face_crop.shape[1] > 20:
            # Simple face encoding
            face_encoding = get_face_encoding(face_crop)
            
            # Here you could compare this encoding to a reference
            # For now, just display that a face was detected
            cv2.putText(frame, "Face Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Proctoring', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()