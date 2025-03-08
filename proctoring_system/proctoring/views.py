from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import generics
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
import json
import cv2
import numpy as np
import base64
import tempfile
import os

from .models import ExamSession, ViolationRecord
from .serializers import ExamSessionSerializer, ViolationRecordSerializer
from proctoring_system.utils.eye_tracking import analyze_eye_movements
from proctoring_system.utils.audio_analysis import analyze_audio
from proctoring_system.utils.object_detection import detect_objects
from proctoring_system.utils.lip_movement import analyze_lip_movements

class ExamSessionListCreate(generics.ListCreateAPIView):
    queryset = ExamSession.objects.all()
    serializer_class = ExamSessionSerializer
    permission_classes = [IsAuthenticated]
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class ExamSessionDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = ExamSession.objects.all()
    serializer_class = ExamSessionSerializer
    permission_classes = [IsAuthenticated]

class ViolationList(generics.ListAPIView):
    serializer_class = ViolationRecordSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        session_id = self.request.query_params.get('session', None)
        if session_id is not None:
            return ViolationRecord.objects.filter(session_id=session_id)
        return ViolationRecord.objects.filter(session__user=self.request.user)

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def analyze_video(request):
    """
    Endpoint to analyze uploaded video files for proctoring
    """
    try:
        if 'video_file' not in request.FILES:
            return Response({"error": "No video file provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        video_file = request.FILES['video_file']
        session_id = request.data.get('session_id')
        
        # Save the video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            for chunk in video_file.chunks():
                temp_video.write(chunk)
            temp_video_path = temp_video.name
        
        # Create or get exam session
        if session_id:
            try:
                session = ExamSession.objects.get(id=session_id, user=request.user)
            except ExamSession.DoesNotExist:
                return Response({"error": "Invalid session ID"}, status=status.HTTP_404_NOT_FOUND)
        else:
            session = ExamSession.objects.create(user=request.user)
            session.video_file = video_file
            session.save()
        
        # Process the video file
        violations = process_video(temp_video_path, session)
        
        # Clean up the temporary file
        os.unlink(temp_video_path)
        
        # Return the analysis results
        return Response({
            "session_id": session.id,
            "violations_count": len(violations),
            "violations": ViolationRecordSerializer(violations, many=True).data,
            "is_eligible": len(violations) < 6
        })
    
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def analyze_frame(request):
    """
    Endpoint to analyze a single frame from webcam
    """
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')
        frame_data = data.get('frame')  # Base64 encoded image
        timestamp = data.get('timestamp', 0)
        audio_data = data.get('audio')  # Base64 encoded audio chunk
        
        # Get or create session
        if session_id:
            try:
                session = ExamSession.objects.get(id=session_id, user=request.user)
            except ExamSession.DoesNotExist:
                return Response({"error": "Invalid session ID"}, status=status.HTTP_404_NOT_FOUND)
        else:
            session = ExamSession.objects.create(user=request.user)
        
        violations = []
        
        # Process video frame
        if frame_data:
            # Convert base64 to image
            img_data = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Analyze the frame
            violations.extend(analyze_frame_data(frame, session, timestamp))
        
        # Process audio data
        if audio_data:
            # Convert base64 to audio
            audio_bytes = base64.b64decode(audio_data.split(',')[1])
            
            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name
            
            # Analyze audio
            audio_violations = analyze_audio(temp_audio_path, timestamp)
            for violation in audio_violations:
                vr = ViolationRecord(
                    session=session,
                    violation_type='audio',
                    timestamp=timestamp,
                    description=violation['description'],
                    confidence=violation['confidence']
                )
                vr.save()
                violations.append(vr)
            
            # Clean up
            os.unlink(temp_audio_path)
        
        return Response({
            "session_id": session.id,
            "violations_detected": len(violations) > 0,
            "violations": ViolationRecordSerializer(violations, many=True).data,
            "total_violations": ViolationRecord.objects.filter(session=session).count()
        })
    
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def process_video(video_path, session):
    """
    Process the entire video file and detect violations
    """
    violations = []
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process frames at regular intervals (not every frame for efficiency)
    sample_interval = int(fps)  # Process one frame per second
    
    for frame_idx in range(0, frame_count, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Calculate timestamp in seconds
        timestamp = frame_idx / fps
        
        # Analyze the current frame
        frame_violations = analyze_frame_data(frame, session, timestamp)
        violations.extend(frame_violations)
    
    # Also analyze audio
    audio_violations = analyze_audio(video_path, 0)
    for violation in audio_violations:
        vr = ViolationRecord(
            session=session,
            violation_type='audio',
            timestamp=violation['timestamp'],
            description=violation['description'],
            confidence=violation['confidence']
        )
        vr.save()
        violations.append(vr)
    
    # Clean up
    cap.release()
    
    return violations

def analyze_frame_data(frame, session, timestamp):
    """
    Analyze a single frame for potential violations
    """
    violations = []
    
    # Detect eye movements
    eye_violations = analyze_eye_movements(frame)
    for violation in eye_violations:
        vr = ViolationRecord(
            session=session,
            violation_type='eye',
            timestamp=timestamp,
            description=violation['description'],
            confidence=violation['confidence']
        )
        vr.save()
        violations.append(vr)
    
    # Detect lip movements
    lip_violations = analyze_lip_movements(frame)
    for violation in lip_violations:
        vr = ViolationRecord(
            session=session,
            violation_type='lip',
            timestamp=timestamp,
            description=violation['description'],
            confidence=violation['confidence']
        )
        vr.save()
        violations.append(vr)
    
    # Detect objects
    object_violations = detect_objects(frame)
    for violation in object_violations:
        vr = ViolationRecord(
            session=session,
            violation_type='object',
            timestamp=timestamp,
            description=violation['description'],
            confidence=violation['confidence']
        )
        vr.save()
        violations.append(vr)
    
    return violations

from django.shortcuts import render

def index(request):
    return render(request, 'index.html')