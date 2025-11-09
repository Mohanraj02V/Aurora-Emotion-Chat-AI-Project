'''
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import voice_api.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'aurora.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            voice_api.routing.websocket_urlpatterns
        )
    ),
})
'''

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import voice_api.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'aurora.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            voice_api.routing.websocket_urlpatterns
        )
    ),
})