import cv2
import numpy as np
import dlib

def analyze_lip_movements(frame):
    """
    Analyze lip movements to detect if someone might be talking or getting help
    
    Args:
        frame: The video frame to analyze
        
    Returns:
        List of detected violations with description and confidence
    """
    violations = []
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray)
        
        if len(faces) == 0:
            return []  # No face detected, can't analyze lips
        
        # Facial landmarks detection
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(predictor_path)
        
        # Loop through detected faces
        for face in faces:
            landmarks = predictor(gray, face)
            
            # Extract mouth landmarks (points 48-68 in 68-point model)
            mouth_points = []
            for i in range(48, 68):
                point = landmarks.part(i)
                mouth_points.append((point.x, point.y))
            
            mouth_points = np.array(mouth_points, dtype=np.int32)
            
            # Calculate mouth aspect ratio
            mar = calculate_mouth_aspect_ratio(mouth_points)
            
            # Check if mouth is open (possible talking)
            if mar > 0.5:  # Threshold determined empirically
                violations.append({
                    "description": "Significant lip movement detected (possibly talking)",
                    "confidence": min(mar * 0.8, 0.95)  # Scale confidence based on MAR
                })
    
    except Exception as e:
        # Handle exceptions
        print(f"Error in lip movement analysis: {str(e)}")
    
    return violations

def calculate_mouth_aspect_ratio(mouth_points):
    """
    Calculate the mouth aspect ratio to detect mouth opening
    """
    # Vertical mouth distances
    v1 = np.linalg.norm(mouth_points[2] - mouth_points[10])  # Upper lip to lower lip
    v2 = np.linalg.norm(mouth_points[4] - mouth_points[8])   # Upper lip to lower lip
    
    # Horizontal mouth distance
    h = np.linalg.norm(mouth_points[0] - mouth_points[6])    # Left to right corner
    
    # Calculate mouth aspect ratio
    mar = (v1 + v2) / (2.0 * h + 1e-6)
    
    return mar