'''
from django.db import models
from django.contrib.auth import get_user_model
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid

User = get_user_model()

class MemoryType(models.TextChoices):
    SHORT_TERM = 'short_term', 'Short Term'
    LONG_TERM = 'long_term', 'Long Term'
    PERSONAL = 'personal', 'Personal'
    CONVERSATION = 'conversation', 'Conversation'

class MemoryEntry(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='memories')
    memory_type = models.CharField(max_length=20, choices=MemoryType.choices)
    content = models.TextField()
    embedding = models.BinaryField(null=True, blank=True)
    metadata = models.JSONField(default=dict)
    importance_score = models.FloatField(
        default=0.5,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    accessed_count = models.IntegerField(default=0)
    last_accessed = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'memory_entries'
        indexes = [
            models.Index(fields=['user', 'memory_type']),
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['user', 'importance_score']),
        ]
        ordering = ['-importance_score', '-created_at']
    
    def __str__(self):
        return f"{self.memory_type} Memory - {self.user.username}"

class MemoryAccessLog(models.Model):
    memory = models.ForeignKey(MemoryEntry, on_delete=models.CASCADE, related_name='access_logs')
    access_type = models.CharField(max_length=20, choices=[
        ('read', 'Read'),
        ('update', 'Update'),
        ('recall', 'Recall'),
        ('search', 'Search')
    ])
    query_context = models.TextField(blank=True)
    relevance_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'memory_access_logs'
        ordering = ['-created_at']

class UserPreferences(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='preferences')
    memory_retention_days = models.IntegerField(default=30)
    max_short_term_memories = models.IntegerField(default=100)
    max_long_term_memories = models.IntegerField(default=1000)
    auto_cleanup_enabled = models.BooleanField(default=True)
    learning_rate = models.FloatField(
        default=0.7,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    class Meta:
        db_table = 'user_preferences'
    
    def __str__(self):
        return f"Preferences - {self.user.username}"
'''

from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid

class MemoryType(models.TextChoices):
    SHORT_TERM = 'short_term', 'Short Term'
    LONG_TERM = 'long_term', 'Long Term'
    PERSONAL = 'personal', 'Personal'
    CONVERSATION = 'conversation', 'Conversation'

class MemoryEntry(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='memories')
    memory_type = models.CharField(max_length=20, choices=MemoryType.choices)
    content = models.TextField()
    embedding = models.BinaryField(null=True, blank=True)
    metadata = models.JSONField(default=dict)
    importance_score = models.FloatField(
        default=0.5,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    accessed_count = models.IntegerField(default=0)
    last_accessed = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'memory_entries'
        indexes = [
            models.Index(fields=['user', 'memory_type']),
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['user', 'importance_score']),
        ]
        ordering = ['-importance_score', '-created_at']
    
    def __str__(self):
        return f"{self.memory_type} Memory - {self.user.username}"

class MemoryAccessLog(models.Model):
    memory = models.ForeignKey(MemoryEntry, on_delete=models.CASCADE, related_name='access_logs')
    access_type = models.CharField(max_length=20, choices=[
        ('read', 'Read'),
        ('update', 'Update'),
        ('recall', 'Recall'),
        ('search', 'Search')
    ])
    query_context = models.TextField(blank=True)
    relevance_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'memory_access_logs'
        ordering = ['-created_at']

class UserPreferences(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='preferences')
    memory_retention_days = models.IntegerField(default=30)
    max_short_term_memories = models.IntegerField(default=100)
    max_long_term_memories = models.IntegerField(default=1000)
    auto_cleanup_enabled = models.BooleanField(default=True)
    learning_rate = models.FloatField(
        default=0.7,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    class Meta:
        db_table = 'user_preferences'
    
    def __str__(self):
        return f"Preferences - {self.user.username}"