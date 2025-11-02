from rest_framework import serializers
from .models import Conversation, AudioFile, EmotionFeedback

class ConversationSerializer(serializers.ModelSerializer):
    emotion_summary = serializers.SerializerMethodField()
    
    class Meta:
        model = Conversation
        fields = ['id', 'user_message', 'assistant_response', 
                 'emotion_data', 'audio_duration', 'created_at', 'emotion_summary']
        read_only_fields = ['id', 'created_at']
    
    def get_emotion_summary(self, obj):
        if obj.emotion_data:
            return {
                'dominant_emotion': obj.emotion_data.get('dominant_emotion', 'neutral'),
                'confidence': obj.emotion_data.get('confidence', 0)
            }
        return None

class AudioFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = AudioFile
        fields = ['id', 'audio_file', 'transcription', 'duration', 'file_size', 'created_at']
        read_only_fields = ['id', 'created_at']

class EmotionFeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = EmotionFeedback
        fields = ['id', 'conversation', 'detected_emotion', 'user_feedback_emotion', 
                 'confidence', 'feedback_notes', 'created_at']
        read_only_fields = ['id', 'created_at']