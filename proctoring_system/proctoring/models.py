from django.db import models
from django.contrib.auth.models import User

class ExamSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    is_completed = models.BooleanField(default=False)
    video_file = models.FileField(upload_to='exam_videos/', null=True, blank=True)
    
    def __str__(self):
        return f"Exam Session {self.id} - {self.user.username}"

class ViolationRecord(models.Model):
    VIOLATION_TYPES = [
        ('eye', 'Eye Movement'),
        ('audio', 'Audio Irregularity'),
        ('object', 'Object Detection'),
        ('lip', 'Lip Movement'),
        ('other', 'Other Violation'),
    ]
    
    session = models.ForeignKey(ExamSession, on_delete=models.CASCADE, related_name='violations')
    violation_type = models.CharField(max_length=10, choices=VIOLATION_TYPES)
    timestamp = models.FloatField()  # Seconds from start of video
    description = models.TextField()
    confidence = models.FloatField()  # Confidence score from ML model
    screenshot = models.ImageField(upload_to='violation_screenshots/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.get_violation_type_display()} at {self.timestamp}s"
