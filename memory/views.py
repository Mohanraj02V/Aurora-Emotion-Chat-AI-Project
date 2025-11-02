from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.core.paginator import Paginator
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
import logging

from .models import MemoryEntry, MemoryAccessLog, UserPreferences
from .services import memory_service
from .serializers import (
    MemoryEntrySerializer, 
    MemoryCreateSerializer,
    UserPreferencesSerializer
)

logger = logging.getLogger(__name__)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_memory(request):
    """Create a new memory entry"""
    try:
        serializer = MemoryCreateSerializer(data=request.data)
        if serializer.is_valid():
            memory = memory_service.create_memory(
                user=request.user,
                content=serializer.validated_data['content'],
                memory_type=serializer.validated_data.get('memory_type', 'conversation'),
                metadata=serializer.validated_data.get('metadata', {}),
                importance_score=serializer.validated_data.get('importance_score', 0.5)
            )
            
            return Response({
                'message': 'Memory created successfully',
                'memory': MemoryEntrySerializer(memory).data
            }, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
    except Exception as e:
        logger.error(f"Error creating memory: {str(e)}")
        return Response(
            {'error': 'Failed to create memory'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def recall_memories(request):
    """Recall memories based on query"""
    try:
        query = request.GET.get('query', '').strip()
        limit = int(request.GET.get('limit', 5))
        min_similarity = float(request.GET.get('min_similarity', 0.3))
        
        if not query:
            return Response({'error': 'Query parameter is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        memories = memory_service.recall_memories(
            user=request.user,
            query=query,
            limit=limit,
            min_similarity=min_similarity
        )
        
        response_data = []
        for memory_data in memories:
            memory_obj = memory_data['memory']
            response_data.append({
                'memory': MemoryEntrySerializer(memory_obj).data,
                'similarity_score': memory_data['similarity_score']
            })
        
        return Response({
            'query': query,
            'memories_found': len(response_data),
            'memories': response_data
        })
        
    except Exception as e:
        logger.error(f"Error recalling memories: {str(e)}")
        return Response(
            {'error': 'Failed to recall memories'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
@cache_page(60 * 5)  # Cache for 5 minutes
def list_memories(request):
    """List user's memories with pagination"""
    try:
        memory_type = request.GET.get('type', '')
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 20))
        
        memories = MemoryEntry.objects.filter(user=request.user)
        
        if memory_type:
            memories = memories.filter(memory_type=memory_type)
        
        memories = memories.order_by('-created_at')
        
        paginator = Paginator(memories, page_size)
        page_obj = paginator.get_page(page)
        
        serializer = MemoryEntrySerializer(page_obj, many=True)
        
        return Response({
            'memories': serializer.data,
            'total_pages': paginator.num_pages,
            'current_page': page,
            'total_memories': paginator.count
        })
        
    except Exception as e:
        logger.error(f"Error listing memories: {str(e)}")
        return Response(
            {'error': 'Failed to list memories'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def update_memory_importance(request, memory_id):
    """Update memory importance score"""
    try:
        importance_score = request.data.get('importance_score')
        
        if importance_score is None or not (0 <= importance_score <= 1):
            return Response(
                {'error': 'Importance score must be between 0 and 1'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        memory = memory_service.promote_memory(memory_id, importance_score)
        
        return Response({
            'message': 'Memory importance updated successfully',
            'memory': MemoryEntrySerializer(memory).data
        })
        
    except MemoryEntry.DoesNotExist:
        return Response({'error': 'Memory not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Error updating memory importance: {str(e)}")
        return Response(
            {'error': 'Failed to update memory importance'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_memory(request, memory_id):
    """Delete a memory"""
    try:
        memory = MemoryEntry.objects.get(id=memory_id, user=request.user)
        
        # Delete from vector store
        from .vector_store import vector_store
        vector_store.delete_memory(memory_id)
        
        # Delete from database
        memory.delete()
        
        return Response({'message': 'Memory deleted successfully'})
        
    except MemoryEntry.DoesNotExist:
        return Response({'error': 'Memory not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error(f"Error deleting memory: {str(e)}")
        return Response(
            {'error': 'Failed to delete memory'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def cleanup_memories(request):
    """Clean up old memories"""
    try:
        days = int(request.data.get('days', 30))
        
        deleted_count = memory_service.cleanup_old_memories(request.user, days)
        
        return Response({
            'message': f'Cleanup completed successfully',
            'memories_deleted': deleted_count
        })
        
    except Exception as e:
        logger.error(f"Error cleaning up memories: {str(e)}")
        return Response(
            {'error': 'Failed to cleanup memories'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_memory_stats(request):
    """Get memory statistics"""
    try:
        stats = memory_service.get_memory_stats(request.user)
        
        return Response(stats)
        
    except Exception as e:
        logger.error(f"Error getting memory stats: {str(e)}")
        return Response(
            {'error': 'Failed to get memory statistics'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET', 'PUT'])
@permission_classes([IsAuthenticated])
def user_preferences(request):
    """Get or update user preferences"""
    try:
        if request.method == 'GET':
            preferences, created = UserPreferences.objects.get_or_create(user=request.user)
            serializer = UserPreferencesSerializer(preferences)
            return Response(serializer.data)
        
        elif request.method == 'PUT':
            preferences, created = UserPreferences.objects.get_or_create(user=request.user)
            serializer = UserPreferencesSerializer(preferences, data=request.data, partial=True)
            
            if serializer.is_valid():
                serializer.save()
                return Response({
                    'message': 'Preferences updated successfully',
                    'preferences': serializer.data
                })
            
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
    except Exception as e:
        logger.error(f"Error handling user preferences: {str(e)}")
        return Response(
            {'error': 'Failed to process preferences'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )