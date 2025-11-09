import logging
import tempfile
import os
from django.conf import settings
from gtts import gTTS
import pygame
import threading
import queue

logger = logging.getLogger(__name__)

class VoiceSynthesizer:
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        if VoiceSynthesizer._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.is_speaking = False
            self.speech_queue = queue.Queue()
            pygame.mixer.init()
            VoiceSynthesizer._instance = self
    
    @staticmethod
    def get_instance():
        if VoiceSynthesizer._instance is None:
            with VoiceSynthesizer._lock:
                if VoiceSynthesizer._instance is None:
                    VoiceSynthesizer._instance = VoiceSynthesizer()
        return VoiceSynthesizer._instance
    
    def speak_text(self, text, emotion=None):
        """Speak text using Google Text-to-Speech"""
        if not getattr(settings, 'VOICE_SYNTHESIS_ENABLED', True):
            logger.warning("Voice synthesis is disabled")
            return False
        
        try:
            if not text or len(text.strip()) == 0:
                return False
            
            # Limit text length for TTS
            text = text[:500]  # Google TTS has character limits
            
            logger.info(f"Generating speech for: {text[:50]}...")
            
            # Create temporary file for speech
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Generate speech using gTTS
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_path)
            
            # Play the audio
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            
            # Clean up
            pygame.mixer.music.unload()
            os.unlink(temp_path)
            
            logger.info("Speech completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in speech synthesis: {str(e)}")
            # Clean up on error
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass
            return False
    
    def stop_speaking(self):
        """Stop current speech"""
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                self.is_speaking = False
                logger.info("Speech stopped")
        except Exception as e:
            logger.error(f"Error stopping speech: {str(e)}")
    
    def save_speech_to_file(self, text, filename=None):
        """Save synthesized speech to file"""
        try:
            if not text:
                return None
            
            if filename is None:
                import uuid
                filename = f"speech_{uuid.uuid4().hex}.mp3"
            
            filepath = os.path.join(settings.MEDIA_ROOT, 'user_audio', filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Generate and save speech
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(filepath)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving speech to file: {str(e)}")
            return None

# Global instance
voice_synthesizer = VoiceSynthesizer.get_instance()