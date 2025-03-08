import cv2
import numpy as np

def analyze_lip_movements(frame):
    """
    Analyze lip movements to detect if someone might be talking or getting help.

    Args:
        frame: The video frame to analyze.

    Returns:
        List of detected violations with description and confidence.
    """
    violations = []

    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Load the face detector
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Detect faces
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return []  # No face detected, can't analyze lips

        # Load the facial landmark detector
        landmark_detector = cv2.face.createFacemarkLBF()
        landmark_detector.loadModel("lbfmodel.yaml")  # Path to the downloaded model file

        # Detect facial landmarks
        _, landmarks = landmark_detector.fit(gray, np.array(faces))

        # Loop through detected faces and landmarks
        for landmark in landmarks:
            # Extract mouth landmarks (points 48-68 in the 68-point model)
            mouth_points = landmark[0][48:68]  # Points 48-67 (0-based indexing)

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
    Calculate the mouth aspect ratio to detect mouth opening.

    Args:
        mouth_points: A numpy array of mouth landmark points.

    Returns:
        Mouth aspect ratio (MAR).
    """
    # Vertical mouth distances
    v1 = np.linalg.norm(mouth_points[2] - mouth_points[10])  # Upper lip to lower lip
    v2 = np.linalg.norm(mouth_points[4] - mouth_points[8])   # Upper lip to lower lip

    # Horizontal mouth distance
    h = np.linalg.norm(mouth_points[0] - mouth_points[6])    # Left to right corner

    # Calculate mouth aspect ratio
    mar = (v1 + v2) / (2.0 * h + 1e-6)

    return mar