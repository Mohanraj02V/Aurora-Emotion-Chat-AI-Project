from rest_framework import serializers
from .models import MemoryEntry, MemoryAccessLog, UserPreferences, MemoryType

class MemoryEntrySerializer(serializers.ModelSerializer):
    content_preview = serializers.SerializerMethodField()
    
    class Meta:
        model = MemoryEntry
        fields = [
            'id', 'memory_type', 'content', 'content_preview', 
            'metadata', 'importance_score', 'accessed_count',
            'last_accessed', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'accessed_count', 'last_accessed', 'created_at', 'updated_at']
    
    def get_content_preview(self, obj):
        """Return a preview of the content"""
        if len(obj.content) <= 100:
            return obj.content
        return obj.content[:100] + '...'

class MemoryCreateSerializer(serializers.Serializer):
    content = serializers.CharField(required=True, max_length=10000)
    memory_type = serializers.ChoiceField(
        choices=MemoryType.choices, 
        default=MemoryType.CONVERSATION
    )
    metadata = serializers.JSONField(required=False, default=dict)
    importance_score = serializers.FloatField(
        required=False, 
        default=0.5,
        min_value=0.0,
        max_value=1.0
    )

class MemoryAccessLogSerializer(serializers.ModelSerializer):
    memory_content = serializers.CharField(source='memory.content', read_only=True)
    
    class Meta:
        model = MemoryAccessLog
        fields = [
            'id', 'memory', 'memory_content', 'access_type',
            'query_context', 'relevance_score', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']

class UserPreferencesSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserPreferences
        fields = [
            'memory_retention_days', 'max_short_term_memories',
            'max_long_term_memories', 'auto_cleanup_enabled', 'learning_rate'
        ]