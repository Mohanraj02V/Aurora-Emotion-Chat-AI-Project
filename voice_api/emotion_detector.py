'''
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from django.core.cache import cache
import logging

logger = logging.getLogger(__name__)

class EmotionDetector:
    _instance = None
    
    def __init__(self):
        if EmotionDetector._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.emotion_classifier = None
            self.tokenizer = None
            self.model = None
            self.load_model()
            EmotionDetector._instance = self
    
    @staticmethod
    def get_instance():
        if EmotionDetector._instance is None:
            EmotionDetector()
        return EmotionDetector._instance
    
    def load_model(self):
        """Load emotion detection model with caching"""
        try:
            cache_key = 'emotion_model'
            cached_model = cache.get(cache_key)
            
            if cached_model is None:
                logger.info("Loading emotion detection model...")
                self.emotion_classifier = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True,
                    device=0 if torch.cuda.is_available() else -1
                )
                # Cache the model for 1 hour
                cache.set(cache_key, 'loaded', 3600)
                logger.info("Emotion detection model loaded successfully")
            else:
                logger.info("Using cached emotion detection model")
                
        except Exception as e:
            logger.error(f"Error loading emotion model: {str(e)}")
            raise e
    
    def detect_emotion(self, text):
        """Detect emotion from text with error handling"""
        try:
            if not text or len(text.strip()) == 0:
                return {'dominant_emotion': 'neutral', 'scores': {}}
            
            # Cache results for similar texts
            cache_key = f"emotion_{hash(text)}"
            cached_result = cache.get(cache_key)
            
            if cached_result:
                return cached_result
            
            results = self.emotion_classifier(text[:512])  # Limit text length
            emotion_scores = {}
            
            for result in results[0]:
                emotion_scores[result['label']] = float(result['score'])
            
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            
            result = {
                'dominant_emotion': dominant_emotion,
                'scores': emotion_scores,
                'confidence': emotion_scores[dominant_emotion]
            }
            
            # Cache result for 5 minutes
            cache.set(cache_key, result, 300)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")
            return {'dominant_emotion': 'neutral', 'scores': {}, 'error': str(e)}
    
    def get_emotional_response_prompt(self, user_text, detected_emotion):
        """Generate emotional context for response generation"""
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
        
        return f"{base_prompt} User message: {user_text}"

# Singleton instance
emotion_detector = EmotionDetector.get_instance()
'''

import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from django.core.cache import cache
import logging
import random

logger = logging.getLogger(__name__)

class EmotionDetector:
    _instance = None
    
    def __init__(self):
        if EmotionDetector._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.emotion_classifier = None
            self.tokenizer = None
            self.model = None
            self.load_model()
            EmotionDetector._instance = self
    
    @staticmethod
    def get_instance():
        if EmotionDetector._instance is None:
            EmotionDetector()
        return EmotionDetector._instance
    
    def load_model(self):
        """Load emotion detection model with caching"""
        try:
            cache_key = 'emotion_model'
            cached_model = cache.get(cache_key)
            
            if cached_model is None:
                logger.info("Loading emotion detection model...")
                self.emotion_classifier = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True,
                    device=0 if torch.cuda.is_available() else -1
                )
                # Cache the model for 1 hour
                cache.set(cache_key, 'loaded', 3600)
                logger.info("Emotion detection model loaded successfully")
            else:
                logger.info("Using cached emotion detection model")
                
        except Exception as e:
            logger.error(f"Error loading emotion model: {str(e)}")
            raise e
    
    def detect_emotion(self, text):
        """Detect emotion from text with error handling"""
        try:
            if not text or len(text.strip()) == 0:
                return {'dominant_emotion': 'neutral', 'scores': {}}
            
            # Cache results for similar texts
            cache_key = f"emotion_{hash(text)}"
            cached_result = cache.get(cache_key)
            
            if cached_result:
                return cached_result
            
            results = self.emotion_classifier(text[:512])  # Limit text length
            emotion_scores = {}
            
            for result in results[0]:
                emotion_scores[result['label']] = float(result['score'])
            
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            
            result = {
                'dominant_emotion': dominant_emotion,
                'scores': emotion_scores,
                'confidence': emotion_scores[dominant_emotion],
                'recommendations': self._generate_recommendations(dominant_emotion, emotion_scores)
            }
            
            # Cache result for 5 minutes
            cache.set(cache_key, result, 300)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")
            return {'dominant_emotion': 'neutral', 'scores': {}, 'error': str(e)}
    
    def _generate_recommendations(self, dominant_emotion, emotion_scores):
        """Generate personalized recommendations based on emotion"""
        recommendations = {
            'joy': [
                "Share your happiness with others - call a friend!",
                "Try a new hobby or creative activity",
                "Listen to upbeat music and dance",
                "Plan something fun for the weekend",
                "Write down what made you happy today"
            ],
            'sadness': [
                "Take a walk in nature - fresh air helps",
                "Listen to calming music or a podcast",
                "Reach out to a friend or loved one",
                "Try some gentle exercise or stretching",
                "Watch your favorite comfort movie"
            ],
            'anger': [
                "Take deep breaths - count to 10 slowly",
                "Go for a brisk walk to release energy",
                "Write down your feelings in a journal",
                "Try some physical activity or exercise",
                "Listen to calming instrumental music"
            ],
            'fear': [
                "Practice mindfulness or meditation",
                "Break down worries into smaller pieces",
                "Talk to someone you trust about your concerns",
                "Create a comforting routine",
                "Focus on what you can control right now"
            ],
            'disgust': [
                "Change your environment - go to a different room",
                "Focus on positive aspects around you",
                "Try a calming tea or warm beverage",
                "Engage in a distracting activity",
                "Practice gratitude - list three good things"
            ],
            'surprise': [
                "Take a moment to process what happened",
                "Share the surprise with someone else",
                "Embrace the unexpected positively",
                "Use this energy for something creative",
                "Document this surprising moment"
            ],
            'neutral': [
                "Try learning something new today",
                "Reach out to an old friend",
                "Plan your next personal project",
                "Take time for self-reflection",
                "Explore a new podcast or book"
            ]
        }
        
        # Get recommendations for the dominant emotion
        emotion_recs = recommendations.get(dominant_emotion, recommendations['neutral'])
        
        # Add some variety by shuffling and selecting 2-3 recommendations
        random.shuffle(emotion_recs)
        return emotion_recs[:3]
    
    def get_emotional_response_prompt(self, user_text, detected_emotion):
        """Generate emotional context for response generation"""
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
        
        # Add recommendation context
        if detected_emotion in ['sadness', 'anger', 'fear']:
            base_prompt += " Gently suggest some calming activities or positive distractions."
        elif detected_emotion == 'joy':
            base_prompt += " Encourage them to make the most of their positive mood."
        
        return f"{base_prompt} User message: {user_text}"

# Singleton instance
emotion_detector = EmotionDetector.get_instance()