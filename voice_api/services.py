'''
import whisper
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import soundfile as sf
import librosa
from django.core.cache import cache
import logging
import tempfile
import os
from datetime import datetime
from .emotion_detector import emotion_detector
from memory.services import memory_service  # Import memory service

logger = logging.getLogger(__name__)

class VoiceAIService:
    def __init__(self):
        self.asr_model = None
        self.chat_model = None
        self.chat_tokenizer = None
        self.load_models()
    
    def load_models(self):
        """Load AI models with caching and error handling"""
        try:
            # Load ASR model
            cache_key = 'asr_model'
            if cache.get(cache_key) is None:
                logger.info("Loading Whisper ASR model...")
                self.asr_model = whisper.load_model("base")
                cache.set(cache_key, 'loaded', 3600)
            
            # Load chat model
            chat_cache_key = 'chat_model'
            if cache.get(chat_cache_key) is None:
                logger.info("Loading DialoGPT model...")
                self.chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
                self.chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
                if torch.cuda.is_available():
                    self.chat_model = self.chat_model.to('cuda')
                cache.set(chat_cache_key, 'loaded', 3600)
                
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise e
    
    def transcribe_audio(self, audio_file_path):
        """Transcribe audio to text using Whisper"""
        try:
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # Use cache for same audio files
            file_hash = self._get_file_hash(audio_file_path)
            cache_key = f"transcription_{file_hash}"
            cached_result = cache.get(cache_key)
            
            if cached_result:
                return cached_result
            
            result = self.asr_model.transcribe(audio_file_path)
            transcription = result["text"].strip()
            
            # Cache transcription for 10 minutes
            cache.set(cache_key, transcription, 600)
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error in audio transcription: {str(e)}")
            raise e
    
    def process_user_message(self, text, user, conversation_history=None):
        """Process user message with emotion detection, memory, and generate response"""
        try:
            # Detect emotion
            emotion_result = emotion_detector.detect_emotion(text)
            
            # Get relevant memories for context
            memory_context = memory_service.get_conversation_context(
                user=user,
                current_message=text,
                conversation_history=conversation_history
            )
            
            # Generate emotional context prompt with memory
            emotional_prompt = self._build_contextual_prompt(
                text, emotion_result['dominant_emotion'], memory_context
            )
            
            # Generate response
            response = self._generate_response(emotional_prompt, conversation_history)
            
            # Store important conversations in memory
            if emotion_result['confidence'] > 0.7 or emotion_result['dominant_emotion'] in ['joy', 'sadness', 'anger']:
                memory_service.create_memory(
                    user=user,
                    content=f"User: {text}\nAssistant: {response}",
                    memory_type='conversation',
                    metadata={
                        'emotion': emotion_result['dominant_emotion'],
                        'confidence': emotion_result['confidence'],
                        'emotional_intensity': max(emotion_result['scores'].values()) if emotion_result['scores'] else 0
                    },
                    importance_score=min(0.3 + (emotion_result['confidence'] * 0.4), 0.8)
                )
            
            return {
                'user_message': text,
                'assistant_response': response,
                'emotion_analysis': emotion_result,
                'memory_context_used': bool(memory_context),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing user message: {str(e)}")
            raise e
    
    def _build_contextual_prompt(self, user_text, detected_emotion, memory_context):
        """Build enhanced prompt with emotion and memory context"""
        emotion_prompts = {
            'joy': "The user is feeling happy and joyful. Respond in an enthusiastic, positive tone that matches their energy.",
            'sadness': "The user seems sad or down. Respond with genuine empathy, comfort, and emotional support. Be gentle and understanding.",
            'anger': "The user appears angry or frustrated. Respond calmly, acknowledge their feelings without judgment, and try to de-escalate the situation.",
            'fear': "The user seems anxious or fearful. Respond with reassurance, support, and practical comfort.",
            'disgust': "The user seems displeased or disgusted. Respond diplomatically and try to understand the cause of their reaction.",
            'surprise': "The user seems surprised. Respond with appropriate interest, engagement, and curiosity.",
            'neutral': "Respond in a friendly, helpful, and engaging tone."
        }
        
        base_prompt = emotion_prompts.get(detected_emotion, emotion_prompts['neutral'])
        
        # Add memory context if available
        if memory_context:
            context_prompt = f"\n\nRelevant previous context:\n{memory_context}\n\nUse this context to provide more personalized and relevant responses, but don't explicitly mention that you're recalling memories."
        else:
            context_prompt = ""
        
        return f"{base_prompt}{context_prompt}\n\nCurrent user message: {user_text}\nAssistant response:"
    
    def _generate_response(self, prompt, conversation_history=None):
        """Generate response using DialoGPT"""
        try:
            # Prepare input with conversation history
            if conversation_history:
                # Use last few messages for context
                context = "\n".join(conversation_history[-3:]) + f"\n{prompt}"
            else:
                context = prompt
            
            # Tokenize input
            inputs = self.chat_tokenizer.encode(context, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            
            # Generate response
            with torch.no_grad():
                outputs = self.chat_model.generate(
                    inputs,
                    max_length=len(inputs[0]) + 150,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.chat_tokenizer.eos_token_id
                )
            
            response = self.chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "Assistant response:" in response:
                response = response.split("Assistant response:")[-1].strip()
            elif "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            # Clean up the response
            response = response.split('\n')[0].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I understand. How can I help you further?"
    
    def _get_file_hash(self, file_path):
        """Generate simple hash for file caching"""
        import hashlib
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def cleanup_temp_file(self, file_path):
        """Clean up temporary audio files"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up temp file: {str(e)}")

# Global service instance
voice_ai_service = VoiceAIService()
'''

import whisper
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import soundfile as sf
import librosa
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
                    # Use base model for better accuracy, force CPU to avoid GPU issues
                    self.asr_model = whisper.load_model("base", device="cpu")
                    logger.info("Whisper model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load Whisper model: {e}")
                    self.asr_model = None
            
            # Load chat model
            try:
                logger.info("Loading DialoGPT model...")
                self.chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
                self.chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
                
                # Always use CPU for now to avoid CUDA issues
                self.chat_model = self.chat_model.to('cpu')
                logger.info("DialoGPT model loaded successfully on CPU")
                
            except Exception as e:
                logger.error(f"Failed to load DialoGPT model: {e}")
                # Fallback to a smaller model
                try:
                    logger.info("Trying fallback model: microsoft/DialoGPT-small")
                    self.chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
                    self.chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
                    self.chat_model = self.chat_model.to('cpu')
                    logger.info("Fallback DialoGPT-small model loaded successfully")
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")
                    self.chat_model = None
                    self.chat_tokenizer = None
            
            self.models_loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.models_loaded = False
    
    def transcribe_audio(self, audio_file_path):
        """Transcribe audio to text using Whisper with enhanced error handling"""
        if not WHISPER_AVAILABLE or self.asr_model is None:
            logger.error("Whisper is not available for transcription")
            return "Speech recognition is currently unavailable."
        
        try:
            if not os.path.exists(audio_file_path):
                logger.error(f"Audio file not found: {audio_file_path}")
                return f"Audio file not found: {audio_file_path}"
            
            # Check file size
            file_size = os.path.getsize(audio_file_path)
            if file_size == 0:
                logger.error("Audio file is empty")
                return "Audio file is empty."
            
            logger.info(f"Transcribing audio file: {audio_file_path} (size: {file_size} bytes)")
            
            # Use cache for same audio files
            file_hash = self._get_file_hash(audio_file_path)
            cache_key = f"transcription_{file_hash}"
            cached_result = cache.get(cache_key)
            
            if cached_result:
                logger.info("Using cached transcription")
                return cached_result
            
            # Perform transcription with error handling
            try:
                result = self.asr_model.transcribe(audio_file_path)
                transcription = result["text"].strip()
                
                if not transcription:
                    transcription = "No speech detected in audio."
                    logger.warning("No speech detected in audio file")
                else:
                    logger.info(f"Transcription successful: {transcription}")
                
                # Cache transcription for 10 minutes
                cache.set(cache_key, transcription, 600)
                
                return transcription
                
            except Exception as transcribe_error:
                logger.error(f"Whisper transcription error: {transcribe_error}")
                return f"Transcription error: {str(transcribe_error)}"
            
        except Exception as e:
            logger.error(f"Error in audio transcription: {str(e)}")
            return f"Error transcribing audio: {str(e)}"

    def process_user_message(self, text, user, conversation_history=None):
        """Process user message with comprehensive error handling"""
        try:
            logger.info(f"Processing user message: {text}")
            
            # Validate input
            if not text or len(text.strip()) == 0:
                return {
                    'user_message': text,
                    'assistant_response': "I didn't receive any message. Could you please try again?",
                    'emotion_analysis': {'dominant_emotion': 'neutral', 'scores': {}},
                    'memory_context_used': False,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Detect emotion with error handling
            emotion_result = {}
            try:
                if emotion_detector and hasattr(emotion_detector, 'detect_emotion'):
                    emotion_result = emotion_detector.detect_emotion(text)
                    logger.info(f"Emotion detected: {emotion_result.get('dominant_emotion', 'unknown')}")
                else:
                    emotion_result = {'dominant_emotion': 'neutral', 'scores': {}, 'error': 'Emotion detector not available'}
                    logger.warning("Emotion detector not available")
            except Exception as e:
                logger.error(f"Emotion detection failed: {e}")
                emotion_result = {'dominant_emotion': 'neutral', 'scores': {}, 'error': str(e)}
            
            # Get relevant memories for context
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
                else:
                    logger.warning("Memory service not available")
            except Exception as e:
                logger.error(f"Error getting memory context: {str(e)}")
            
            # Generate response with fallback
            try:
                emotional_prompt = self._build_contextual_prompt(
                    text, emotion_result.get('dominant_emotion', 'neutral'), memory_context
                )
                response = self._generate_response(emotional_prompt, conversation_history)
                logger.info("Response generated successfully")
            except Exception as e:
                logger.error(f"Response generation failed: {e}")
                response = "I understand. How can I help you with that?"
            
            # Store conversation in memory (if enabled)
            try:
                if user.memory_storage_consent and memory_service:
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
            # Return a safe fallback response
            return {
                'user_message': text,
                'assistant_response': "I'm experiencing some technical difficulties. Please try again in a moment.",
                'emotion_analysis': {'dominant_emotion': 'neutral', 'scores': {}, 'error': str(e)},
                'memory_context_used': False,
                'timestamp': datetime.now().isoformat()
            }

    def _build_contextual_prompt(self, user_text, detected_emotion, memory_context):
        """Build enhanced prompt with emotion and memory context"""
        emotion_prompts = {
            'joy': "The user is feeling happy and joyful. Respond in an enthusiastic, positive tone.",
            'sadness': "The user seems sad. Respond with empathy, comfort, and support.",
            'anger': "The user appears angry. Respond calmly, acknowledge their feelings, and try to de-escalate.",
            'fear': "The user seems anxious or fearful. Respond with reassurance and support.",
            'disgust': "The user seems displeased or disgusted. Respond diplomatically and try to understand the cause.",
            'surprise': "The user seems surprised. Respond with appropriate interest and engagement.",
            'neutral': "Respond in a friendly, helpful tone."
        }
        
        base_prompt = emotion_prompts.get(detected_emotion, emotion_prompts['neutral'])
        
        # Add memory context if available
        if memory_context:
            context_prompt = f"\n\nRelevant context from previous conversations:\n{memory_context}\n\nUse this context to provide more personalized responses."
        else:
            context_prompt = ""
        
        return f"{base_prompt}{context_prompt}\n\nUser message: {user_text}\nAssistant:"

    def _generate_response(self, prompt, conversation_history=None):
        """Generate response using DialoGPT with enhanced error handling"""
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
                    max_length=min(len(inputs[0]) + 100, 512),  # Reduced max length
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
                response = "I understand. How can I help you further?"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I understand. How can I help you with that?"

    def _get_file_hash(self, file_path):
        """Generate simple hash for file caching"""
        import hashlib
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return str(os.path.getsize(file_path))

# Global service instance with initialization check
try:
    voice_ai_service = VoiceAIService()
    if voice_ai_service.models_loaded:
        logger.info("Voice AI Service initialized successfully")
    else:
        logger.warning("Voice AI Service initialized but some models failed to load")
except Exception as e:
    logger.error(f"Failed to initialize Voice AI Service: {e}")
    voice_ai_service = None