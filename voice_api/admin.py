from django.contrib import admin
from .models import Conversation, AudioFile, EmotionFeedback

@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('user', 'user_message_preview', 'assistant_response_preview', 'dominant_emotion', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('user__username', 'user_message', 'assistant_response')
    readonly_fields = ('created_at', 'updated_at')
    
    def user_message_preview(self, obj):
        return obj.user_message[:50] + '...' if len(obj.user_message) > 50 else obj.user_message
    user_message_preview.short_description = 'User Message'
    
    def assistant_response_preview(self, obj):
        return obj.assistant_response[:50] + '...' if len(obj.assistant_response) > 50 else obj.assistant_response
    assistant_response_preview.short_description = 'Assistant Response'
    
    def dominant_emotion(self, obj):
        return obj.emotion_data.get('dominant_emotion', 'N/A') if obj.emotion_data else 'N/A'
    dominant_emotion.short_description = 'Dominant Emotion'

@admin.register(AudioFile)
class AudioFileAdmin(admin.ModelAdmin):
    list_display = ('user', 'transcription_preview', 'duration', 'file_size', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('user__username', 'transcription')
    readonly_fields = ('created_at',)
    
    def transcription_preview(self, obj):
        return obj.transcription[:50] + '...' if len(obj.transcription) > 50 else obj.transcription
    transcription_preview.short_description = 'Transcription'

@admin.register(EmotionFeedback)
class EmotionFeedbackAdmin(admin.ModelAdmin):
    list_display = ('user', 'detected_emotion', 'user_feedback_emotion', 'confidence', 'created_at')
    list_filter = ('detected_emotion', 'user_feedback_emotion', 'created_at')
    search_fields = ('user__username', 'feedback_notes')
    readonly_fields = ('created_at',)