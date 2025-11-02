from django.db import transaction
from django.utils import timezone
from datetime import timedelta
import logging
from .models import MemoryEntry, MemoryAccessLog, UserPreferences, MemoryType
from .vector_store import vector_store

logger = logging.getLogger(__name__)

class MemoryService:
    def __init__(self):
        self.vector_store = vector_store
    
    def create_memory(self, user, content, memory_type=MemoryType.CONVERSATION, metadata=None, importance_score=0.5):
        """Create a new memory entry"""
        try:
            with transaction.atomic():
                # Create memory in database
                memory = MemoryEntry.objects.create(
                    user=user,
                    content=content,
                    memory_type=memory_type,
                    metadata=metadata or {},
                    importance_score=importance_score
                )
                
                # Store in vector database
                vector_metadata = {
                    'user_id': str(user.id),
                    'memory_type': memory_type,
                    'importance_score': importance_score,
                    'created_at': memory.created_at.isoformat()
                }
                if metadata:
                    vector_metadata.update(metadata)
                
                success = self.vector_store.store_memory(
                    memory.id, content, vector_metadata, memory_type
                )
                
                if not success:
                    logger.warning(f"Failed to store memory {memory.id} in vector store")
                
                logger.info(f"Created memory {memory.id} for user {user.username}")
                return memory
                
        except Exception as e:
            logger.error(f"Error creating memory: {str(e)}")
            raise e
    
    def recall_memories(self, user, query, limit=5, min_similarity=0.3):
        """Recall memories relevant to the query"""
        try:
            # Get similar memories from vector store
            similar_memories = self.vector_store.search_similar_memories(
                query, str(user.id), limit=limit, threshold=min_similarity
            )
            
            # Update access logs and counts
            memory_entries = []
            for memory_data in similar_memories:
                try:
                    memory = MemoryEntry.objects.get(id=memory_data['memory_id'])
                    memory.accessed_count += 1
                    memory.save()
                    
                    # Log this access
                    MemoryAccessLog.objects.create(
                        memory=memory,
                        access_type='recall',
                        query_context=query,
                        relevance_score=memory_data['similarity_score']
                    )
                    
                    memory_entries.append({
                        'memory': memory,
                        'similarity_score': memory_data['similarity_score']
                    })
                    
                except MemoryEntry.DoesNotExist:
                    continue
            
            return memory_entries
            
        except Exception as e:
            logger.error(f"Error recalling memories: {str(e)}")
            return []
    
    def get_conversation_context(self, user, current_message, conversation_history=None, limit=7):
        """Get relevant memories for conversation context"""
        try:
            contextual_memories = self.vector_store.get_contextual_memories(
                current_message, str(user.id), conversation_history, limit=limit
            )
            
            # Format for LLM context
            context_parts = []
            for memory_data in contextual_memories:
                if memory_data['similarity_score'] > 0.4:
                    context_parts.append(f"Previous memory: {memory_data['content']}")
            
            if context_parts:
                return "\n".join(context_parts)
            return ""
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {str(e)}")
            return ""
    
    def promote_memory(self, memory_id, new_importance_score):
        """Promote a memory to long-term or increase importance"""
        try:
            memory = MemoryEntry.objects.get(id=memory_id)
            
            # Update importance score
            memory.importance_score = new_importance_score
            
            # Promote to long-term if important enough
            if new_importance_score > 0.8 and memory.memory_type != MemoryType.LONG_TERM:
                memory.memory_type = MemoryType.LONG_TERM
            
            memory.save()
            
            # Update in vector store
            metadata = memory.metadata.copy()
            metadata['importance_score'] = new_importance_score
            metadata['memory_type'] = memory.memory_type
            
            self.vector_store.store_memory(
                memory.id, memory.content, metadata, memory.memory_type
            )
            
            logger.info(f"Promoted memory {memory_id} to importance {new_importance_score}")
            return memory
            
        except Exception as e:
            logger.error(f"Error promoting memory: {str(e)}")
            raise e
    
    def cleanup_old_memories(self, user, days=30):
        """Clean up old, unimportant memories"""
        try:
            cutoff_date = timezone.now() - timedelta(days=days)
            
            # Get old, unimportant memories
            old_memories = MemoryEntry.objects.filter(
                user=user,
                created_at__lt=cutoff_date,
                importance_score__lt=0.3,
                memory_type=MemoryType.SHORT_TERM
            )
            
            deleted_count = 0
            for memory in old_memories:
                # Delete from vector store
                self.vector_store.delete_memory(memory.id)
                # Delete from database
                memory.delete()
                deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old memories for user {user.username}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old memories: {str(e)}")
            return 0
    
    def get_memory_stats(self, user):
        """Get memory statistics for user"""
        try:
            from django.db.models import Avg
            
            stats = {
                'total_memories': MemoryEntry.objects.filter(user=user).count(),
                'short_term_memories': MemoryEntry.objects.filter(
                    user=user, memory_type=MemoryType.SHORT_TERM
                ).count(),
                'long_term_memories': MemoryEntry.objects.filter(
                    user=user, memory_type=MemoryType.LONG_TERM
                ).count(),
                'personal_memories': MemoryEntry.objects.filter(
                    user=user, memory_type=MemoryType.PERSONAL
                ).count(),
                'total_accesses': MemoryAccessLog.objects.filter(
                    memory__user=user
                ).count(),
                'avg_importance': MemoryEntry.objects.filter(
                    user=user
                ).aggregate(avg_importance=Avg('importance_score'))['avg_importance'] or 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}")
            return {}

# Global service instance
memory_service = MemoryService()