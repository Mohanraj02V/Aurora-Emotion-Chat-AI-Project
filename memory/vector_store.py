import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from django.core.cache import cache
import logging
import pickle
import os
from django.conf import settings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    _instance = None
    
    def __init__(self):
        if VectorStoreManager._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.embedding_model = None
            self.chroma_client = None
            self.chroma_collection = None
            self.load_models()
            VectorStoreManager._instance = self
    
    @staticmethod
    def get_instance():
        if VectorStoreManager._instance is None:
            VectorStoreManager()
        return VectorStoreManager._instance
    
    def load_models(self):
        """Load embedding model and initialize vector stores"""
        try:
            # Load embedding model
            cache_key = 'embedding_model'
            if cache.get(cache_key) is None:
                logger.info("Loading sentence transformer model...")
                self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                cache.set(cache_key, 'loaded', 3600)
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.Client()
            self.chroma_collection = self.chroma_client.get_or_create_collection("aurora_memories")
            
            logger.info("Vector store manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            # Don't raise the exception, just log it
            pass
    
    def get_embedding(self, text):
        """Generate embedding for text"""
        try:
            if not text or len(text.strip()) == 0:
                return np.zeros(384, dtype=np.float32)
            
            # Cache embeddings for same text
            cache_key = f"embedding_{hash(text)}"
            cached_embedding = cache.get(cache_key)
            
            if cached_embedding:
                return pickle.loads(cached_embedding)
            
            if self.embedding_model is None:
                self.load_models()
                if self.embedding_model is None:
                    return np.zeros(384, dtype=np.float32)
            
            embedding = self.embedding_model.encode(text)
            
            # Cache for 1 hour
            cache.set(cache_key, pickle.dumps(embedding), 3600)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return np.zeros(384, dtype=np.float32)
    
    def store_memory(self, memory_id, content, metadata=None, memory_type='conversation'):
        """Store memory in vector database"""
        try:
            if self.chroma_collection is None:
                self.load_models()
                if self.chroma_collection is None:
                    return False
            
            embedding = self.get_embedding(content)
            
            # Store in ChromaDB
            self.chroma_collection.add(
                embeddings=[embedding.tolist()],
                documents=[content],
                metadatas=[metadata or {}],
                ids=[str(memory_id)]
            )
            
            logger.info(f"Stored memory {memory_id} in vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error storing memory in vector store: {str(e)}")
            return False
    
    def search_similar_memories(self, query, user_id, limit=5, threshold=0.3):
        """Search for similar memories using semantic search"""
        try:
            if self.chroma_collection is None:
                self.load_models()
                if self.chroma_collection is None:
                    return []
            
            query_embedding = self.get_embedding(query)
            
            # Search in ChromaDB
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=limit
            )
            
            similar_memories = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    similarity_score = 1 - distance
                    if similarity_score >= threshold:
                        similar_memories.append({
                            'content': doc,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'memory_id': results['ids'][0][i]
                        })
            
            # Sort by similarity score
            similar_memories.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return similar_memories
            
        except Exception as e:
            logger.error(f"Error searching memories: {str(e)}")
            return []
    
    def get_contextual_memories(self, query, user_id, conversation_history=None, limit=10):
        """Get relevant memories for context in conversation"""
        try:
            # Search for directly similar memories
            similar_memories = self.search_similar_memories(query, user_id, limit=limit)
            
            # If we have conversation history, also search based on recent context
            context_memories = []
            if conversation_history and len(conversation_history) > 0:
                recent_context = " ".join([msg.get('content', '') for msg in conversation_history[-3:]])
                context_memories = self.search_similar_memories(recent_context, user_id, limit=limit//2)
            
            # Combine and deduplicate
            all_memories = {}
            for memory in similar_memories + context_memories:
                mem_id = memory.get('memory_id')
                if mem_id not in all_memories or memory['similarity_score'] > all_memories[mem_id]['similarity_score']:
                    all_memories[mem_id] = memory
            
            # Return top memories
            sorted_memories = sorted(all_memories.values(), key=lambda x: x['similarity_score'], reverse=True)
            return sorted_memories[:limit]
            
        except Exception as e:
            logger.error(f"Error getting contextual memories: {str(e)}")
            return []
    
    def delete_memory(self, memory_id):
        """Delete memory from vector store"""
        try:
            if self.chroma_collection is None:
                return False
                
            self.chroma_collection.delete(ids=[str(memory_id)])
            logger.info(f"Deleted memory {memory_id} from vector store")
            return True
        except Exception as e:
            logger.error(f"Error deleting memory from vector store: {str(e)}")
            return False

# Singleton instance with error handling
try:
    vector_store = VectorStoreManager.get_instance()
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    vector_store = None