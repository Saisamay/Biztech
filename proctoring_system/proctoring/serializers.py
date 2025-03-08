from rest_framework import serializers
from .models import ExamSession, ViolationRecord

class ViolationRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = ViolationRecord
        fields = ['id', 'violation_type', 'timestamp', 'description', 'confidence', 'screenshot', 'created_at']

class ExamSessionSerializer(serializers.ModelSerializer):
    violations_count = serializers.SerializerMethodField()
    
    class Meta:
        model = ExamSession
        fields = ['id', 'start_time', 'end_time', 'is_completed', 'video_file', 'violations_count']
    
    def get_violations_count(self, obj):
        return obj.violations.count()