import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize FaceMesh (for landmarks) and Iris (for detailed eye tracking)
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,  # Enable iris landmarks
    max_num_faces=3,        # Detect up to 2 faces
    min_detection_confidence=0.5
) as face_mesh:
    
    cap = cv2.VideoCapture(0)  # Webcam input
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        violations = []
        
        if results.multi_face_landmarks:
            # Detect multiple faces (cheating)
            if len(results.multi_face_landmarks) > 1:
                violations.append("Multiple faces detected")
            
            for face_landmarks in results.multi_face_landmarks:
                # Extract eye landmarks
                left_eye = [face_landmarks.landmark[i] for i in mp_face_mesh.FACEMESH_LEFT_EYE]
                right_eye = [face_landmarks.landmark[i] for i in mp_face_mesh.FACEMESH_RIGHT_EYE]
                iris_left = [face_landmarks.landmark[i] for i in [468, 469, 470, 471, 472]]  # Iris landmarks
                iris_right = [face_landmarks.landmark[i] for i in [473, 474, 475, 476, 477]]
                
                # Calculate gaze direction (simplified)
                left_gaze = (iris_left[0].x - left_eye[0].x)  # Compare iris position to eye corner
                right_gaze = (iris_right[0].x - right_eye[0].x)
                
                if abs(left_gaze) > 0.1 or abs(right_gaze) > 0.1:
                    violations.append("Looking away from screen")
                
                # Detect closed eyes using EAR (Eye Aspect Ratio)
                ear_left = calculate_ear(left_eye)
                ear_right = calculate_ear(right_eye)
                if (ear_left + ear_right) / 2 < 0.2:
                    violations.append("Eyes closed")
        
        # Display violations
        cv2.putText(frame, f"Violations: {violations}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Proctoring', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
