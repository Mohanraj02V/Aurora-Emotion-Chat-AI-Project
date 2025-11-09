'''
import whisper
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import numpy as np
from django.core.cache import cache
import logging
import tempfile
import os
from datetime import datetime
from .emotion_detector import emotion_detector
from memory.services import memory_service

logger = logging.getLogger(__name__)

# Enhanced error handling for model loading
try:
    import whisper
    WHISPER_AVAILABLE = True
    logger.info("Whisper imported successfully")
except ImportError as e:
    logger.error(f"Whisper import failed: {e}")
    WHISPER_AVAILABLE = False

class VoiceAIService:
    def __init__(self):
        self.asr_model = None
        self.chat_model = None
        self.chat_tokenizer = None
        self.models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load AI models with enhanced error handling"""
        try:
            # Load ASR model only if whisper is available
            if WHISPER_AVAILABLE:
                try:
                    logger.info("Loading Whisper ASR model...")
                    # Use base model for better accuracy
                    self.asr_model = whisper.load_model("base")
                    logger.info("Whisper model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load Whisper model: {e}")
                    self.asr_model = None
            
            # Load chat model
            try:
                logger.info("Loading DialoGPT model...")
                self.chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
                self.chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
                logger.info("DialoGPT model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load DialoGPT model: {e}")
                # Fallback to smaller model
                try:
                    logger.info("Trying fallback model: microsoft/DialoGPT-small")
                    self.chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
                    self.chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
                    logger.info("Fallback DialoGPT model loaded successfully")
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")
                    self.chat_model = None
                    self.chat_tokenizer = None
            
            self.models_loaded = True
            logger.info("Voice AI Service initialized")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.models_loaded = False

    def transcribe_audio(self, audio_file_path):
        """Transcribe audio to text using Whisper"""
        if not WHISPER_AVAILABLE or self.asr_model is None:
            logger.error("Whisper is not available for transcription")
            return "Speech recognition is currently unavailable. Please try typing your message."
        
        try:
            if not os.path.exists(audio_file_path):
                logger.error(f"Audio file not found: {audio_file_path}")
                return "Audio file not found. Please try recording again."
            
            # Check file size
            file_size = os.path.getsize(audio_file_path)
            if file_size == 0:
                logger.error("Audio file is empty")
                return "No audio detected. Please try recording again."
            
            logger.info(f"Transcribing audio file: {audio_file_path} (size: {file_size} bytes)")
            
            # Perform transcription
            result = self.asr_model.transcribe(audio_file_path)
            transcription = result["text"].strip()
            
            if not transcription:
                transcription = "I couldn't detect any speech in the audio. Please try again in a quiet environment."
                logger.warning("No speech detected in audio file")
            else:
                logger.info(f"Transcription successful: {transcription}")
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error in audio transcription: {str(e)}")
            return f"Error processing audio: {str(e)}"

    def process_user_message(self, text, user, conversation_history=None):
        """Process user message with comprehensive error handling"""
        try:
            logger.info(f"Processing user message: {text}")
            
            # Check if this is an error message from transcription
            if any(keyword in text.lower() for keyword in ['error', 'unavailable', 'not found', 'no audio', 'couldn\'t detect']):
                response_text = "I had trouble understanding your voice message. You can try typing your message or recording again in a quiet environment."
                return {
                    'user_message': text,
                    'assistant_response': response_text,
                    'emotion_analysis': {'dominant_emotion': 'neutral', 'scores': {}},
                    'memory_context_used': False,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Validate input
            if not text or len(text.strip()) == 0:
                response_text = "I didn't receive any message. Could you please try again?"
                return {
                    'user_message': text,
                    'assistant_response': response_text,
                    'emotion_analysis': {'dominant_emotion': 'neutral', 'scores': {}},
                    'memory_context_used': False,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Detect emotion
            emotion_result = {}
            try:
                if emotion_detector and hasattr(emotion_detector, 'detect_emotion'):
                    emotion_result = emotion_detector.detect_emotion(text)
                    logger.info(f"Emotion detected: {emotion_result.get('dominant_emotion', 'unknown')}")
                else:
                    emotion_result = {'dominant_emotion': 'neutral', 'scores': {}}
                    logger.warning("Emotion detector not available")
            except Exception as e:
                logger.error(f"Emotion detection failed: {e}")
                emotion_result = {'dominant_emotion': 'neutral', 'scores': {}}
            
            # Get memory context
            memory_context = ""
            try:
                if memory_service:
                    memory_context = memory_service.get_conversation_context(
                        user=user,
                        current_message=text,
                        conversation_history=conversation_history
                    )
                    if memory_context:
                        logger.info("Memory context retrieved successfully")
            except Exception as e:
                logger.error(f"Error getting memory context: {str(e)}")
            
            # Generate response
            try:
                emotional_prompt = self._build_contextual_prompt(
                    text, emotion_result.get('dominant_emotion', 'neutral'), memory_context
                )
                response = self._generate_response(emotional_prompt, conversation_history)
                logger.info("Response generated successfully")
                
            except Exception as e:
                logger.error(f"Response generation failed: {e}")
                response = self._get_fallback_response(emotion_result.get('dominant_emotion', 'neutral'))
            
            # Store in memory (only if successful and user consented)
            try:
                if (user.memory_storage_consent and memory_service and 
                    response != self._get_fallback_response('neutral')):
                    memory_service.create_memory(
                        user=user,
                        content=f"User: {text}\nAssistant: {response}",
                        memory_type='conversation',
                        metadata={
                            'emotion': emotion_result.get('dominant_emotion', 'neutral'),
                            'confidence': emotion_result.get('confidence', 0),
                        },
                        importance_score=0.5
                    )
                    logger.info("Conversation stored in memory")
            except Exception as e:
                logger.error(f"Error storing memory: {str(e)}")
            
            return {
                'user_message': text,
                'assistant_response': response,
                'emotion_analysis': emotion_result,
                'memory_context_used': bool(memory_context),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing user message: {str(e)}")
            return {
                'user_message': text,
                'assistant_response': "I'm experiencing some technical difficulties. Please try again in a moment.",
                'emotion_analysis': {'dominant_emotion': 'neutral', 'scores': {}},
                'memory_context_used': False,
                'timestamp': datetime.now().isoformat()
            }

    def _build_contextual_prompt(self, user_text, detected_emotion, memory_context):
        """Build enhanced prompt with emotion and memory context"""
        emotion_prompts = {
            'joy': "The user is feeling happy and joyful. Respond in an enthusiastic, positive tone. Suggest fun activities or share in their excitement.",
            'sadness': "The user seems sad. Respond with empathy, comfort, and support. Offer gentle suggestions for mood improvement.",
            'anger': "The user appears angry. Respond calmly, acknowledge their feelings, and try to de-escalate. Offer practical solutions.",
            'fear': "The user seems anxious or fearful. Respond with reassurance and support. Help them feel safe and understood.",
            'disgust': "The user seems displeased or disgusted. Respond diplomatically and try to understand the cause.",
            'surprise': "The user seems surprised. Respond with appropriate interest and engagement.",
            'neutral': "Respond in a friendly, helpful tone. Be conversational and engaging."
        }
        
        base_prompt = emotion_prompts.get(detected_emotion, emotion_prompts['neutral'])
        
        # Add memory context if available
        if memory_context:
            context_prompt = f"\n\nRelevant context from previous conversations:\n{memory_context}\n\nUse this context to provide more personalized responses."
        else:
            context_prompt = ""
        
        return f"{base_prompt}{context_prompt}\n\nUser message: {user_text}\nAssistant:"

    def _generate_response(self, prompt, conversation_history=None):
        """Generate response using DialoGPT"""
        if self.chat_model is None or self.chat_tokenizer is None:
            logger.error("Chat model not loaded")
            return "I'm currently unavailable. Please try again later."
        
        try:
            # Prepare input with conversation history
            if conversation_history and len(conversation_history) > 0:
                # Use the last 2 conversations for context
                recent_context = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history
                context = "\n".join(recent_context) + f"\n{prompt}"
            else:
                context = prompt
            
            # Tokenize input
            inputs = self.chat_tokenizer.encode(context, return_tensors="pt")
            
            # Generate response with safe parameters
            with torch.no_grad():
                outputs = self.chat_model.generate(
                    inputs,
                    max_length=min(len(inputs[0]) + 100, 512),
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.85,
                    top_k=40,
                    repetition_penalty=1.1,
                    pad_token_id=self.chat_tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            response = self.chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "Assistant:" in response:
                response_parts = response.split("Assistant:")
                if len(response_parts) > 1:
                    response = response_parts[-1].strip()
            
            # Clean up the response - take first sentence or first 200 characters
            response = response.split('.')[0] + '.' if '.' in response else response
            response = response[:200].strip()
            
            # Ensure response is not empty
            if not response or len(response) < 2:
                response = self._get_fallback_response('neutral')
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._get_fallback_response('neutral')

    def _get_fallback_response(self, emotion):
        """Get appropriate fallback response based on emotion"""
        fallback_responses = {
            'joy': "That sounds wonderful! Tell me more about what's making you happy.",
            'sadness': "I'm here to listen and help. Would you like to talk about what's bothering you?",
            'anger': "I understand you're upset. Let's work through this together.",
            'fear': "It's okay to feel scared sometimes. I'm here to help you feel safe.",
            'neutral': "How can I help you today?",
        }
        return fallback_responses.get(emotion, "How can I help you today?")

# Global service instance
try:
    voice_ai_service = VoiceAIService()
    if voice_ai_service.models_loaded:
        logger.info("Voice AI Service initialized successfully")
    else:
        logger.warning("Voice AI Service initialized but some models failed to load")
except Exception as e:
    logger.error(f"Failed to initialize Voice AI Service: {e}")
    voice_ai_service = None
'''

import whisper
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import numpy as np
from django.core.cache import cache
import logging
import tempfile
import os
from datetime import datetime
from .emotion_detector import emotion_detector
from memory.services import memory_service
import shutil

logger = logging.getLogger(__name__)

# Enhanced error handling for model loading
try:
    import whisper
    WHISPER_AVAILABLE = True
    logger.info("Whisper imported successfully")
except ImportError as e:
    logger.error(f"Whisper import failed: {e}")
    WHISPER_AVAILABLE = False

class VoiceAIService:
    def __init__(self):
        self.asr_model = None
        self.chat_model = None
        self.chat_tokenizer = None
        self.models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load AI models with enhanced error handling"""
        try:
            # Load ASR model only if whisper is available
            if WHISPER_AVAILABLE:
                try:
                    logger.info("Loading Whisper ASR model...")
                    # Use base model for better accuracy
                    self.asr_model = whisper.load_model("base")
                    logger.info("Whisper model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load Whisper model: {e}")
                    self.asr_model = None
            
            # Load chat model
            try:
                logger.info("Loading DialoGPT model...")
                self.chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
                self.chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
                logger.info("DialoGPT model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load DialoGPT model: {e}")
                # Fallback to smaller model
                try:
                    logger.info("Trying fallback model: microsoft/DialoGPT-small")
                    self.chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
                    self.chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
                    logger.info("Fallback DialoGPT model loaded successfully")
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")
                    self.chat_model = None
                    self.chat_tokenizer = None
            
            self.models_loaded = True
            logger.info("Voice AI Service initialized")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.models_loaded = False

    def transcribe_audio(self, audio_file_path):
        """Transcribe audio to text using Whisper with robust file handling"""
        if not WHISPER_AVAILABLE or self.asr_model is None:
            logger.error("Whisper is not available for transcription")
            return "Speech recognition is currently unavailable. Please try typing your message."
        
        converted_path = None
        
        try:
            # Verify file exists and is accessible
            if not os.path.exists(audio_file_path):
                logger.error(f"Audio file not found: {audio_file_path}")
                return "Audio file not found. Please try recording again."
            
            # Check file size
            file_size = os.path.getsize(audio_file_path)
            if file_size == 0:
                logger.error("Audio file is empty")
                return "No audio detected. Please try recording again."
            
            logger.info(f"Transcribing audio file: {audio_file_path} (size: {file_size} bytes)")
            
            # Convert to WAV format for better Whisper compatibility
            converted_path = self._convert_to_wav(audio_file_path)
            if not converted_path:
                logger.warning("Could not convert to WAV, using original file")
                converted_path = audio_file_path
            
            # Verify converted file exists
            if not os.path.exists(converted_path):
                logger.error(f"Converted file not found: {converted_path}")
                return "Error processing audio file. Please try again."
            
            # Perform transcription
            result = self.asr_model.transcribe(converted_path)
            transcription = result["text"].strip()
            
            if not transcription:
                transcription = "I couldn't detect any speech in the audio. Please try again in a quiet environment."
                logger.warning("No speech detected in audio file")
            else:
                logger.info(f"Transcription successful: {transcription}")
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error in audio transcription: {str(e)}")
            return "I had trouble processing your audio. Please try typing your message instead."
        
        finally:
            # Clean up converted file if it was created
            if converted_path and converted_path != audio_file_path and os.path.exists(converted_path):
                try:
                    os.unlink(converted_path)
                    logger.info(f"Cleaned up converted file: {converted_path}")
                except Exception as e:
                    logger.warning(f"Could not delete converted file: {e}")

    def _convert_to_wav(self, audio_file_path):
        """Convert audio file to WAV format for better Whisper compatibility"""
        try:
            # Check if already WAV
            if audio_file_path.lower().endswith('.wav'):
                return audio_file_path
            
            # Try using pydub if available
            try:
                from pydub import AudioSegment
                
                # Create output path
                wav_path = audio_file_path + '.wav'
                
                # Convert to WAV
                audio = AudioSegment.from_file(audio_file_path)
                audio = audio.set_frame_rate(16000)  # Set to 16kHz
                audio = audio.set_channels(1)        # Convert to mono
                audio.export(wav_path, format="wav")
                
                logger.info(f"Converted {audio_file_path} to WAV format")
                return wav_path
                
            except ImportError:
                logger.warning("pydub not available, using original file format")
                return audio_file_path
            except Exception as e:
                logger.error(f"Audio conversion failed: {e}")
                return audio_file_path
                
        except Exception as e:
            logger.error(f"Error in audio conversion: {e}")
            return audio_file_path

    def process_user_message(self, text, user, conversation_history=None):
        """Process user message with comprehensive error handling"""
        try:
            logger.info(f"Processing user message: {text}")
            
            # Check if this is an error message from transcription
            if any(keyword in text.lower() for keyword in ['error', 'unavailable', 'not found', 'no audio', 'couldn\'t detect', 'trouble processing']):
                response_text = "I had trouble understanding your voice message. You can try typing your message or recording again in a quiet environment."
                return {
                    'user_message': text,
                    'assistant_response': response_text,
                    'emotion_analysis': {'dominant_emotion': 'neutral', 'scores': {}},
                    'memory_context_used': False,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Validate input
            if not text or len(text.strip()) == 0:
                response_text = "I didn't receive any message. Could you please try again?"
                return {
                    'user_message': text,
                    'assistant_response': response_text,
                    'emotion_analysis': {'dominant_emotion': 'neutral', 'scores': {}},
                    'memory_context_used': False,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Detect emotion
            emotion_result = {}
            try:
                if emotion_detector and hasattr(emotion_detector, 'detect_emotion'):
                    emotion_result = emotion_detector.detect_emotion(text)
                    logger.info(f"Emotion detected: {emotion_result.get('dominant_emotion', 'unknown')}")
                else:
                    emotion_result = {'dominant_emotion': 'neutral', 'scores': {}}
                    logger.warning("Emotion detector not available")
            except Exception as e:
                logger.error(f"Emotion detection failed: {e}")
                emotion_result = {'dominant_emotion': 'neutral', 'scores': {}}
            
            # Get memory context
            memory_context = ""
            try:
                if memory_service:
                    memory_context = memory_service.get_conversation_context(
                        user=user,
                        current_message=text,
                        conversation_history=conversation_history
                    )
                    if memory_context:
                        logger.info("Memory context retrieved successfully")
            except Exception as e:
                logger.error(f"Error getting memory context: {str(e)}")
            
            # Generate response
            try:
                emotional_prompt = self._build_contextual_prompt(
                    text, emotion_result.get('dominant_emotion', 'neutral'), memory_context
                )
                response = self._generate_response(emotional_prompt, conversation_history)
                logger.info("Response generated successfully")
                
            except Exception as e:
                logger.error(f"Response generation failed: {e}")
                response = self._get_fallback_response(emotion_result.get('dominant_emotion', 'neutral'))
            
            # Store in memory (only if successful and user consented)
            try:
                if (user.memory_storage_consent and memory_service and 
                    response != self._get_fallback_response('neutral')):
                    memory_service.create_memory(
                        user=user,
                        content=f"User: {text}\nAssistant: {response}",
                        memory_type='conversation',
                        metadata={
                            'emotion': emotion_result.get('dominant_emotion', 'neutral'),
                            'confidence': emotion_result.get('confidence', 0),
                        },
                        importance_score=0.5
                    )
                    logger.info("Conversation stored in memory")
            except Exception as e:
                logger.error(f"Error storing memory: {str(e)}")
            
            return {
                'user_message': text,
                'assistant_response': response,
                'emotion_analysis': emotion_result,
                'memory_context_used': bool(memory_context),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing user message: {str(e)}")
            return {
                'user_message': text,
                'assistant_response': "I'm experiencing some technical difficulties. Please try again in a moment.",
                'emotion_analysis': {'dominant_emotion': 'neutral', 'scores': {}},
                'memory_context_used': False,
                'timestamp': datetime.now().isoformat()
            }

    def _build_contextual_prompt(self, user_text, detected_emotion, memory_context):
        """Build enhanced prompt with emotion and memory context"""
        emotion_prompts = {
            'joy': "The user is feeling happy and joyful. Respond in an enthusiastic, positive tone. Suggest fun activities or share in their excitement.",
            'sadness': "The user seems sad. Respond with empathy, comfort, and support. Offer gentle suggestions for mood improvement.",
            'anger': "The user appears angry. Respond calmly, acknowledge their feelings, and try to de-escalate. Offer practical solutions.",
            'fear': "The user seems anxious or fearful. Respond with reassurance and support. Help them feel safe and understood.",
            'disgust': "The user seems displeased or disgusted. Respond diplomatically and try to understand the cause.",
            'surprise': "The user seems surprised. Respond with appropriate interest and engagement.",
            'neutral': "Respond in a friendly, helpful tone. Be conversational and engaging."
        }
        
        base_prompt = emotion_prompts.get(detected_emotion, emotion_prompts['neutral'])
        
        # Add memory context if available
        if memory_context:
            context_prompt = f"\n\nRelevant context from previous conversations:\n{memory_context}\n\nUse this context to provide more personalized responses."
        else:
            context_prompt = ""
        
        return f"{base_prompt}{context_prompt}\n\nUser message: {user_text}\nAssistant:"

    def _generate_response(self, prompt, conversation_history=None):
        """Generate response using DialoGPT"""
        if self.chat_model is None or self.chat_tokenizer is None:
            logger.error("Chat model not loaded")
            return "I'm currently unavailable. Please try again later."
        
        try:
            # Prepare input with conversation history
            if conversation_history and len(conversation_history) > 0:
                # Use the last 2 conversations for context
                recent_context = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history
                context = "\n".join(recent_context) + f"\n{prompt}"
            else:
                context = prompt
            
            # Tokenize input
            inputs = self.chat_tokenizer.encode(context, return_tensors="pt")
            
            # Generate response with safe parameters
            with torch.no_grad():
                outputs = self.chat_model.generate(
                    inputs,
                    max_length=min(len(inputs[0]) + 100, 512),
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.85,
                    top_k=40,
                    repetition_penalty=1.1,
                    pad_token_id=self.chat_tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            response = self.chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "Assistant:" in response:
                response_parts = response.split("Assistant:")
                if len(response_parts) > 1:
                    response = response_parts[-1].strip()
            
            # Clean up the response - take first sentence or first 200 characters
            response = response.split('.')[0] + '.' if '.' in response else response
            response = response[:200].strip()
            
            # Ensure response is not empty
            if not response or len(response) < 2:
                response = self._get_fallback_response('neutral')
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._get_fallback_response('neutral')

    def _get_fallback_response(self, emotion):
        """Get appropriate fallback response based on emotion"""
        fallback_responses = {
            'joy': "That sounds wonderful! Tell me more about what's making you happy.",
            'sadness': "I'm here to listen and help. Would you like to talk about what's bothering you?",
            'anger': "I understand you're upset. Let's work through this together.",
            'fear': "It's okay to feel scared sometimes. I'm here to help you feel safe.",
            'neutral': "How can I help you today?",
        }
        return fallback_responses.get(emotion, "How can I help you today?")

# Global service instance
try:
    voice_ai_service = VoiceAIService()
    if voice_ai_service.models_loaded:
        logger.info("Voice AI Service initialized successfully")
    else:
        logger.warning("Voice AI Service initialized but some models failed to load")
except Exception as e:
    logger.error(f"Failed to initialize Voice AI Service: {e}")
    voice_ai_service = None