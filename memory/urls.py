from django.urls import path
from . import views

urlpatterns = [
    path('create/', views.create_memory, name='create_memory'),
    path('recall/', views.recall_memories, name='recall_memories'),
    path('list/', views.list_memories, name='list_memories'),
    path('<uuid:memory_id>/importance/', views.update_memory_importance, name='update_memory_importance'),
    path('<uuid:memory_id>/delete/', views.delete_memory, name='delete_memory'),
    path('cleanup/', views.cleanup_memories, name='cleanup_memories'),
    path('stats/', views.get_memory_stats, name='memory_stats'),
    path('preferences/', views.user_preferences, name='user_preferences'),
]