'''
import json
import base64
import wave
import tempfile
import os
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.core.files.base import ContentFile
import logging
from .services import voice_ai_service
from .models import Conversation
from .emotion_detector import emotion_detector

logger = logging.getLogger(__name__)

class VoiceStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope["user"]
        
        if self.user.is_authenticated:
            await self.accept()
            self.audio_chunks = []
            self.sample_rate = 16000
            self.channels = 1
            self.sample_width = 2  # 16-bit PCM
            
            await self.send(text_data=json.dumps({
                'type': 'connection_established',
                'message': 'WebSocket connection established for voice streaming'
            }))
        else:
            await self.close(code=4001)

    async def disconnect(self, close_code):
        logger.info(f"WebSocket disconnected with code: {close_code}")
        # Process any remaining audio data
        if hasattr(self, 'audio_chunks') and self.audio_chunks:
            await self.process_audio_data()

    async def receive(self, text_data=None, bytes_data=None):
        try:
            if text_data:
                data = json.loads(text_data)
                message_type = data.get('type')
                
                if message_type == 'audio_chunk':
                    # Handle base64 encoded audio data
                    audio_data = data.get('data')
                    if audio_data:
                        await self.handle_audio_chunk(audio_data)
                
                elif message_type == 'audio_end':
                    # Process the complete audio
                    await self.process_audio_data()
                
                elif message_type == 'ping':
                    await self.send(text_data=json.dumps({
                        'type': 'pong',
                        'timestamp': data.get('timestamp')
                    }))

        except Exception as e:
            logger.error(f"Error in WebSocket receive: {str(e)}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Error processing audio data'
            }))

    async def handle_audio_chunk(self, audio_data):
        """Handle incoming audio chunks"""
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data.split(',')[1])
            self.audio_chunks.append(audio_bytes)
            
            # Send acknowledgment
            await self.send(text_data=json.dumps({
                'type': 'audio_ack',
                'chunks_received': len(self.audio_chunks)
            }))
            
        except Exception as e:
            logger.error(f"Error handling audio chunk: {str(e)}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Error processing audio chunk'
            }))

    async def process_audio_data(self):
        """Process complete audio data"""
        try:
            if not self.audio_chunks:
                return

            # Combine all audio chunks
            combined_audio = b''.join(self.audio_chunks)
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                # Write WAV file header and data
                with wave.open(temp_audio.name, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(self.sample_width)
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(combined_audio)
                
                temp_path = temp_audio.name

            try:
                # Send processing status
                await self.send(text_data=json.dumps({
                    'type': 'processing_start',
                    'message': 'Processing audio...'
                }))

                # Transcribe audio
                transcription = await database_sync_to_async(voice_ai_service.transcribe_audio)(temp_path)
                
                await self.send(text_data=json.dumps({
                    'type': 'transcription_result',
                    'transcription': transcription
                }))

                # Get conversation history
                recent_conversations = await self.get_recent_conversations()
                conversation_history = [
                    f"User: {conv.user_message}\nAssistant: {conv.assistant_response}"
                    for conv in recent_conversations
                ]

                # Process message with emotion detection
                result = await database_sync_to_async(voice_ai_service.process_user_message)(
                    transcription, self.user, conversation_history
                )

                # Save conversation to database
                conversation = await self.save_conversation(result)

                # Send final result
                await self.send(text_data=json.dumps({
                    'type': 'processing_complete',
                    'conversation': {
                        'id': conversation.id,
                        'user_message': result['user_message'],
                        'assistant_response': result['assistant_response'],
                        'created_at': conversation.created_at.isoformat()
                    },
                    'emotion_analysis': result['emotion_analysis'],
                    'transcription': transcription
                }))

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
                # Clear audio chunks for next recording
                self.audio_chunks = []

        except Exception as e:
            logger.error(f"Error processing audio data: {str(e)}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Error processing audio: {str(e)}'
            }))

    @database_sync_to_async
    def get_recent_conversations(self):
        """Get recent conversations from database"""
        return list(Conversation.objects.filter(
            user=self.user
        ).order_by('-created_at')[:5])

    @database_sync_to_async
    def save_conversation(self, result):
        """Save conversation to database"""
        conversation = Conversation.objects.create(
            user=self.user,
            user_message=result['user_message'],
            assistant_response=result['assistant_response'],
            emotion_data=result['emotion_analysis']
        )
        return conversation


class VoiceStatusConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope["user"]
        
        if self.user.is_authenticated:
            await self.accept()
            await self.send(text_data=json.dumps({
                'type': 'status_connected',
                'message': 'Status WebSocket connected'
            }))
        else:
            await self.close(code=4001)

    async def disconnect(self, close_code):
        logger.info(f"Status WebSocket disconnected with code: {close_code}")

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'get_status':
                # Send current system status
                status = await self.get_system_status()
                await self.send(text_data=json.dumps({
                    'type': 'system_status',
                    'status': status
                }))
            
            elif message_type == 'ping':
                await self.send(text_data=json.dumps({
                    'type': 'pong',
                    'timestamp': data.get('timestamp')
                }))

        except Exception as e:
            logger.error(f"Error in status WebSocket: {str(e)}")

    @database_sync_to_async
    def get_system_status(self):
        """Get system status information"""
        from .services import voice_ai_service
        
        status = {
            'asr_loaded': voice_ai_service.asr_model is not None,
            'chat_model_loaded': voice_ai_service.chat_model is not None,
            'emotion_detector_loaded': emotion_detector.emotion_classifier is not None,
            'cuda_available': getattr(voice_ai_service.chat_model, 'device', None) is not None and 'cuda' in str(voice_ai_service.chat_model.device),
            'user_conversations_count': Conversation.objects.filter(user=self.user).count()
        }
        return status
'''

import json
import base64
import wave
import tempfile
import os
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.core.files.base import ContentFile
import logging
from .services import voice_ai_service
from .voice_synthesizer import voice_synthesizer
from .models import Conversation
from .emotion_detector import emotion_detector

logger = logging.getLogger(__name__)

class VoiceStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope["user"]
        
        if self.user.is_authenticated:
            await self.accept()
            self.audio_chunks = []
            self.sample_rate = 16000
            self.channels = 1
            self.sample_width = 2  # 16-bit PCM
            
            await self.send(text_data=json.dumps({
                'type': 'connection_established',
                'message': 'WebSocket connection established for voice streaming',
                'voice_enabled': True
            }))
        else:
            await self.close(code=4001)

    async def disconnect(self, close_code):
        logger.info(f"WebSocket disconnected with code: {close_code}")
        # Stop any ongoing speech
        voice_synthesizer.stop_speaking()
        # Process any remaining audio data
        if hasattr(self, 'audio_chunks') and self.audio_chunks:
            await self.process_audio_data()

    async def receive(self, text_data=None, bytes_data=None):
        try:
            if text_data:
                data = json.loads(text_data)
                message_type = data.get('type')
                
                if message_type == 'audio_chunk':
                    # Handle base64 encoded audio data
                    audio_data = data.get('data')
                    if audio_data:
                        await self.handle_audio_chunk(audio_data)
                
                elif message_type == 'audio_end':
                    # Process the complete audio
                    await self.process_audio_data()
                
                elif message_type == 'stop_speaking':
                    # Stop current speech
                    voice_synthesizer.stop_speaking()
                    await self.send(text_data=json.dumps({
                        'type': 'speech_stopped',
                        'message': 'Speech stopped'
                    }))
                
                elif message_type == 'ping':
                    await self.send(text_data=json.dumps({
                        'type': 'pong',
                        'timestamp': data.get('timestamp')
                    }))

        except Exception as e:
            logger.error(f"Error in WebSocket receive: {str(e)}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Error processing audio data'
            }))

    async def handle_audio_chunk(self, audio_data):
        """Handle incoming audio chunks"""
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data.split(',')[1])
            self.audio_chunks.append(audio_bytes)
            
            # Send acknowledgment
            await self.send(text_data=json.dumps({
                'type': 'audio_ack',
                'chunks_received': len(self.audio_chunks)
            }))
            
        except Exception as e:
            logger.error(f"Error handling audio chunk: {str(e)}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Error processing audio chunk'
            }))

    async def process_audio_data(self):
        """Process complete audio data"""
        try:
            if not self.audio_chunks:
                return

            # Combine all audio chunks
            combined_audio = b''.join(self.audio_chunks)
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                # Write WAV file header and data
                with wave.open(temp_audio.name, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(self.sample_width)
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(combined_audio)
                
                temp_path = temp_audio.name

            try:
                # Send processing status
                await self.send(text_data=json.dumps({
                    'type': 'processing_start',
                    'message': 'Processing audio...'
                }))

                # Transcribe audio
                transcription = await database_sync_to_async(voice_ai_service.transcribe_audio)(temp_path)
                
                await self.send(text_data=json.dumps({
                    'type': 'transcription_result',
                    'transcription': transcription
                }))

                # Get conversation history
                recent_conversations = await self.get_recent_conversations()
                conversation_history = [
                    {
                        'user_message': conv.user_message,
                        'assistant_response': conv.assistant_response
                    }
                    for conv in recent_conversations
                ]

                # Process message with emotion detection and voice output
                result = await database_sync_to_async(voice_ai_service.process_user_message)(
                    transcription, self.user, conversation_history
                )

                # Save conversation to database
                conversation = await self.save_conversation(result)

                # Send final result
                await self.send(text_data=json.dumps({
                    'type': 'processing_complete',
                    'conversation': {
                        'id': conversation.id,
                        'user_message': result['user_message'],
                        'assistant_response': result['assistant_response'],
                        'created_at': conversation.created_at.isoformat()
                    },
                    'emotion_analysis': result['emotion_analysis'],
                    'transcription': transcription,
                    'voice_output': result.get('voice_output', False),
                    'command_executed': result.get('command_executed', False),
                    'command_result': result.get('command_result')
                }))

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
                # Clear audio chunks for next recording
                self.audio_chunks = []

        except Exception as e:
            logger.error(f"Error processing audio data: {str(e)}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Error processing audio: {str(e)}'
            }))

    @database_sync_to_async
    def get_recent_conversations(self):
        """Get recent conversations from database"""
        return list(Conversation.objects.filter(
            user=self.user
        ).order_by('-created_at')[:5])

    @database_sync_to_async
    def save_conversation(self, result):
        """Save conversation to database"""
        conversation = Conversation.objects.create(
            user=self.user,
            user_message=result['user_message'],
            assistant_response=result['assistant_response'],
            emotion_data=result['emotion_analysis']
        )
        return conversation


class VoiceStatusConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope["user"]
        
        if self.user.is_authenticated:
            await self.accept()
            await self.send(text_data=json.dumps({
                'type': 'status_connected',
                'message': 'Status WebSocket connected',
                'voice_enabled': True
            }))
        else:
            await self.close(code=4001)

    async def disconnect(self, close_code):
        logger.info(f"Status WebSocket disconnected with code: {close_code}")

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'get_status':
                # Send current system status
                status = await self.get_system_status()
                await self.send(text_data=json.dumps({
                    'type': 'system_status',
                    'status': status
                }))
            
            elif message_type == 'test_voice':
                # Test voice synthesis
                test_text = data.get('text', 'Hello, this is a voice test.')
                success = voice_synthesizer.speak_text(test_text)
                await self.send(text_data=json.dumps({
                    'type': 'voice_test_result',
                    'success': success,
                    'text': test_text
                }))
            
            elif message_type == 'stop_speaking':
                voice_synthesizer.stop_speaking()
                await self.send(text_data=json.dumps({
                    'type': 'speech_stopped',
                    'message': 'All speech stopped'
                }))
            
            elif message_type == 'ping':
                await self.send(text_data=json.dumps({
                    'type': 'pong',
                    'timestamp': data.get('timestamp')
                }))

        except Exception as e:
            logger.error(f"Error in status WebSocket: {str(e)}")

    @database_sync_to_async
    def get_system_status(self):
        """Get system status information"""
        from .services import voice_ai_service
        
        status = {
            'asr_loaded': voice_ai_service.asr_model is not None,
            'chat_model_loaded': voice_ai_service.chat_model is not None,
            'emotion_detector_loaded': emotion_detector.emotion_classifier is not None,
            'voice_synthesizer_loaded': voice_synthesizer.engine is not None,
            'cuda_available': getattr(voice_ai_service.chat_model, 'device', None) is not None and 'cuda' in str(voice_ai_service.chat_model.device),
            'user_conversations_count': Conversation.objects.filter(user=self.user).count(),
            'voice_enabled': True,
            'command_execution_enabled': True
        }
        return status