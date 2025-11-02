'''
from django.db import models
from django.contrib.auth import get_user_model
from django.core.validators import MinValueValidator, MaxValueValidator

User = get_user_model()

class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    user_message = models.TextField()
    assistant_response = models.TextField()
    emotion_data = models.JSONField(default=dict)  # Store emotion analysis results
    audio_duration = models.FloatField(null=True, blank=True)  # Duration in seconds
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'conversations'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Conversation {self.id} - {self.user.username}"

class AudioFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, null=True, blank=True)
    audio_file = models.FileField(upload_to='user_audio/')
    transcription = models.TextField()
    duration = models.FloatField(validators=[MinValueValidator(0)])
    file_size = models.IntegerField(validators=[MinValueValidator(0)])
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'audio_files'
        ordering = ['-created_at']

class EmotionFeedback(models.Model):
    EMOTION_CHOICES = [
        ('joy', 'Joy'),
        ('sadness', 'Sadness'),
        ('anger', 'Anger'),
        ('fear', 'Fear'),
        ('disgust', 'Disgust'),
        ('surprise', 'Surprise'),
        ('neutral', 'Neutral'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    detected_emotion = models.CharField(max_length=20, choices=EMOTION_CHOICES)
    user_feedback_emotion = models.CharField(max_length=20, choices=EMOTION_CHOICES)
    confidence = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(1)])
    feedback_notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'emotion_feedback'
'''
'''
from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator

class Conversation(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='conversations')
    user_message = models.TextField()
    assistant_response = models.TextField()
    emotion_data = models.JSONField(default=dict)
    audio_duration = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'conversations'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Conversation {self.id} - {self.user.username}"

class AudioFile(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, null=True, blank=True)
    audio_file = models.FileField(upload_to='user_audio/')
    transcription = models.TextField()
    duration = models.FloatField(validators=[MinValueValidator(0)])
    file_size = models.IntegerField(validators=[MinValueValidator(0)])
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'audio_files'
        ordering = ['-created_at']

class EmotionFeedback(models.Model):
    EMOTION_CHOICES = [
        ('joy', 'Joy'),
        ('sadness', 'Sadness'),
        ('anger', 'Anger'),
        ('fear', 'Fear'),
        ('disgust', 'Disgust'),
        ('surprise', 'Surprise'),
        ('neutral', 'Neutral'),
    ]
    
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    detected_emotion = models.CharField(max_length=20, choices=EMOTION_CHOICES)
    user_feedback_emotion = models.CharField(max_length=20, choices=EMOTION_CHOICES)
    confidence = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(1)])
    feedback_notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'emotion_feedback'
'''

from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator

class Conversation(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='conversations')
    user_message = models.TextField()
    assistant_response = models.TextField()
    emotion_data = models.JSONField(default=dict)
    audio_duration = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'conversations'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Conversation {self.id} - {self.user.username}"

class AudioFile(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, null=True, blank=True)
    audio_file = models.FileField(upload_to='user_audio/')
    transcription = models.TextField()
    duration = models.FloatField(validators=[MinValueValidator(0)], null=True, blank=True)  # Make nullable
    file_size = models.IntegerField(validators=[MinValueValidator(0)], null=True, blank=True)  # Make nullable
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'audio_files'
        ordering = ['-created_at']

class EmotionFeedback(models.Model):
    EMOTION_CHOICES = [
        ('joy', 'Joy'),
        ('sadness', 'Sadness'),
        ('anger', 'Anger'),
        ('fear', 'Fear'),
        ('disgust', 'Disgust'),
        ('surprise', 'Surprise'),
        ('neutral', 'Neutral'),
    ]
    
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    detected_emotion = models.CharField(max_length=20, choices=EMOTION_CHOICES)
    user_feedback_emotion = models.CharField(max_length=20, choices=EMOTION_CHOICES)
    confidence = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(1)])
    feedback_notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'emotion_feedback'