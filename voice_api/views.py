import os
import uuid
import wave
import tempfile
import contextlib
from datetime import datetime
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
import logging

from .services import voice_ai_service
from .models import Conversation, AudioFile
from .serializers import ConversationSerializer

logger = logging.getLogger(__name__)

def get_audio_duration(file_path):
    """Get audio duration in seconds"""
    try:
        if file_path.endswith('.wav'):
            with contextlib.closing(wave.open(file_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return frames / float(rate)
        else:
            # For other formats, use a fallback or return None
            return None
    except Exception as e:
        logger.error(f"Error getting audio duration: {e}")
        return None

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def process_audio_message(request):
    """Process audio message and return text response with emotion analysis"""
    temp_path = None
    
    try:
        logger.info("Audio processing request received")
        
        if 'audio' not in request.FILES:
            logger.error("No audio file provided in request")
            return Response({'error': 'No audio file provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        audio_file = request.FILES['audio']
        user = request.user
        
        logger.info(f"Processing audio file: {audio_file.name}, size: {audio_file.size} bytes")
        
        # Validate file type
        valid_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm']
        file_extension = os.path.splitext(audio_file.name.lower())[1]
        if file_extension not in valid_extensions:
            logger.error(f"Invalid audio format: {file_extension}")
            return Response({'error': f'Invalid audio format. Supported formats: {", ".join(valid_extensions)}'}, 
                          status=status.HTTP_400_BAD_REQUEST)
        
        # Check file size (max 10MB)
        if audio_file.size > 10 * 1024 * 1024:
            logger.error(f"Audio file too large: {audio_file.size} bytes")
            return Response({'error': 'Audio file too large (max 10MB)'}, status=status.HTTP_400_BAD_REQUEST)
        
        if audio_file.size == 0:
            logger.error("Audio file is empty")
            return Response({'error': 'Audio file is empty'}, status=status.HTTP_400_BAD_REQUEST)

        # Create a proper temporary file with correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Write the uploaded file to temporary file
            for chunk in audio_file.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name
        
        logger.info(f"Temporary file created: {temp_path}")
        logger.info(f"Temporary file exists: {os.path.exists(temp_path)}")
        logger.info(f"Temporary file size: {os.path.getsize(temp_path)} bytes")

        # Check if voice service is available
        if voice_ai_service is None:
            logger.error("Voice AI service not initialized")
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            return Response({'error': 'Voice service not available'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Transcribe audio
        logger.info("Starting audio transcription...")
        transcription = voice_ai_service.transcribe_audio(temp_path)
        logger.info(f"Transcription result: {transcription}")
        
        if transcription.startswith("Error") or transcription.startswith("Speech recognition"):
            logger.error(f"Transcription failed: {transcription}")
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            return Response({'error': transcription}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Get conversation history
        recent_conversations = Conversation.objects.filter(
            user=user
        ).order_by('-created_at')[:5]
        
        conversation_history = [
            f"User: {conv.user_message}\nAssistant: {conv.assistant_response}"
            for conv in recent_conversations
        ]
        
        # Process message with emotion detection
        logger.info("Processing message with emotion detection...")
        result = voice_ai_service.process_user_message(
            transcription, 
            user,
            conversation_history
        )
        
        # Save conversation to database
        conversation = Conversation.objects.create(
            user=user,
            user_message=result['user_message'],
            assistant_response=result['assistant_response'],
            emotion_data=result['emotion_analysis'],
            audio_duration=0
        )
        
        # Save audio file record if user consented
        if user.memory_storage_consent:
            # Get audio duration if possible
            duration = get_audio_duration(temp_path)
            file_size = audio_file.size
            
            # Reset file pointer before saving
            audio_file.seek(0)
            AudioFile.objects.create(
                user=user,
                conversation=conversation,
                audio_file=audio_file,
                transcription=transcription,
                duration=duration,  # Can be None
                file_size=file_size  # Can be None
            )
            logger.info(f"Audio file saved with duration: {duration}, file_size: {file_size}")
        
        serializer = ConversationSerializer(conversation)
        
        response_data = {
            'conversation': serializer.data,
            'emotion_analysis': result['emotion_analysis'],
            'transcription': transcription
        }
        
        logger.info("Audio processing completed successfully")
        return Response(response_data)
        
    except Exception as e:
        logger.error(f"Error in process_audio_message: {str(e)}", exc_info=True)
        return Response(
            {'error': f'Failed to process audio message: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        # Clean up temporary file only after everything is done
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.info(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                logger.error(f"Error cleaning up temp file: {str(e)}")

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def process_text_message(request):
    """Process text message and return emotional response"""
    try:
        text = request.data.get('text', '').strip()
        if not text:
            return Response({'error': 'No text provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        user = request.user
        
        # Get conversation history
        recent_conversations = Conversation.objects.filter(
            user=user
        ).order_by('-created_at')[:5]
        
        conversation_history = [
            f"User: {conv.user_message}\nAssistant: {conv.assistant_response}"
            for conv in recent_conversations
        ]
        
        # Process message with emotion detection
        result = voice_ai_service.process_user_message(text, conversation_history)
        
        # Save conversation to database
        conversation = Conversation.objects.create(
            user=user,
            user_message=result['user_message'],
            assistant_response=result['assistant_response'],
            emotion_data=result['emotion_analysis']
        )
        
        serializer = ConversationSerializer(conversation)
        
        return Response({
            'conversation': serializer.data,
            'emotion_analysis': result['emotion_analysis']
        })
        
    except Exception as e:
        logger.error(f"Error processing text message: {str(e)}")
        return Response(
            {'error': 'Failed to process text message'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
@cache_page(60 * 2)  # Cache for 2 minutes
def get_conversation_history(request):
    """Get user's conversation history"""
    try:
        conversations = Conversation.objects.filter(
            user=request.user
        ).order_by('-created_at')[:20]  # Last 20 conversations
        
        serializer = ConversationSerializer(conversations, many=True)
        return Response(serializer.data)
        
    except Exception as e:
        logger.error(f"Error fetching conversation history: {str(e)}")
        return Response(
            {'error': 'Failed to fetch conversation history'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_conversation(request, conversation_id):
    """Delete specific conversation"""
    try:
        conversation = Conversation.objects.get(
            id=conversation_id, 
            user=request.user
        )
        conversation.delete()
        
        return Response({'message': 'Conversation deleted successfully'})
        
    except Conversation.DoesNotExist:
        return Response(
            {'error': 'Conversation not found'}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        return Response(
            {'error': 'Failed to delete conversation'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def websocket_test(request):
    """Test WebSocket connection"""
    return Response({
        'message': 'WebSocket endpoints are available at /ws/voice/stream/ and /ws/voice/status/',
        'websocket_urls': {
            'voice_stream': 'ws://localhost:8000/ws/voice/stream/',
            'voice_status': 'ws://localhost:8000/ws/voice/status/'
        }
    })