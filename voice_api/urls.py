from django.urls import path
from . import views

urlpatterns = [
    path('process-audio/', views.process_audio_message, name='process_audio'),
    path('process-text/', views.process_text_message, name='process_text'),
    path('conversations/', views.get_conversation_history, name='conversation_history'),
    path('conversations/<int:conversation_id>/', views.delete_conversation, name='delete_conversation'),
    path('websocket-test/', views.websocket_test, name='websocket_test'),  # Add this
]