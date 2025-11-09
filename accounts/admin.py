from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser, UserSession, EncryptedData

@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'consent_given', 'voice_cloning_consent', 'memory_storage_consent', 'date_joined')
    list_filter = ('consent_given', 'voice_cloning_consent', 'memory_storage_consent', 'is_staff', 'is_active')
    search_fields = ('username', 'email')
    ordering = ('-date_joined',)
    
    fieldsets = UserAdmin.fieldsets + (
        ('Consent Settings', {
            'fields': ('consent_given', 'consent_given_at', 'voice_cloning_consent', 'memory_storage_consent')
        }),
    )

@admin.register(UserSession)
class UserSessionAdmin(admin.ModelAdmin):
    list_display = ('user', 'session_token', 'created_at', 'expires_at', 'is_active')
    list_filter = ('is_active', 'created_at')
    search_fields = ('user__username', 'session_token')
    readonly_fields = ('created_at',)

@admin.register(EncryptedData)
class EncryptedDataAdmin(admin.ModelAdmin):
    list_display = ('user', 'data_type', 'created_at')
    list_filter = ('data_type', 'created_at')
    search_fields = ('user__username', 'data_type')
    readonly_fields = ('created_at',)