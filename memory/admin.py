from django.contrib import admin
from .models import MemoryEntry, MemoryAccessLog, UserPreferences

@admin.register(MemoryEntry)
class MemoryEntryAdmin(admin.ModelAdmin):
    list_display = ('user', 'memory_type', 'content_preview', 'importance_score', 'accessed_count', 'created_at')
    list_filter = ('memory_type', 'created_at', 'importance_score')
    search_fields = ('user__username', 'content')
    readonly_fields = ('created_at', 'updated_at', 'last_accessed')
    
    def content_preview(self, obj):
        return obj.content[:100] + '...' if len(obj.content) > 100 else obj.content
    content_preview.short_description = 'Content Preview'

@admin.register(MemoryAccessLog)
class MemoryAccessLogAdmin(admin.ModelAdmin):
    list_display = ('memory', 'access_type', 'relevance_score', 'created_at')
    list_filter = ('access_type', 'created_at')
    search_fields = ('memory__content', 'query_context')
    readonly_fields = ('created_at',)

@admin.register(UserPreferences)
class UserPreferencesAdmin(admin.ModelAdmin):
    list_display = ('user', 'memory_retention_days', 'auto_cleanup_enabled', 'learning_rate')
    list_filter = ('auto_cleanup_enabled',)
    search_fields = ('user__username',)