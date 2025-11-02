from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/voice/stream/$', consumers.VoiceStreamConsumer.as_asgi()),
    re_path(r'ws/voice/status/$', consumers.VoiceStatusConsumer.as_asgi()),
]