import os
import subprocess
import platform
import logging
import requests
import json
from django.conf import settings

logger = logging.getLogger(__name__)

class CommandExecutor:
    def __init__(self):
        self.system = platform.system().lower()
        self.allowed_commands = getattr(settings, 'ALLOWED_COMMANDS', [])
        self.allow_system_commands = getattr(settings, 'ALLOW_SYSTEM_COMMANDS', True)
    
    def execute_command(self, command_description, user_intent):
        """Execute system commands based on user intent"""
        if not self.allow_system_commands:
            return {"success": False, "message": "System commands are disabled", "output": None}
        
        try:
            # Map common applications to system commands
            command_map = self._get_command_map()
            
            # Find the best matching command
            command_key = self._find_matching_command(command_description, command_map)
            
            if command_key and command_key in self.allowed_commands:
                command = command_map[command_key]
                result = self._run_command(command)
                
                if result['success']:
                    return {
                        "success": True,
                        "message": f"Successfully opened {command_key}",
                        "output": result['output'],
                        "action": "command_executed"
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Failed to open {command_key}",
                        "output": result['output'],
                        "action": "command_failed"
                    }
            else:
                return {
                    "success": False,
                    "message": f"Command '{command_description}' not allowed or not recognized",
                    "output": None,
                    "action": "command_not_allowed"
                }
                
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            return {
                "success": False,
                "message": f"Error executing command: {str(e)}",
                "output": None,
                "action": "error"
            }
    
    def _get_command_map(self):
        """Get system-specific command map"""
        if self.system == "windows":
            return {
                'notepad': 'notepad.exe',
                'calculator': 'calc.exe',
                'paint': 'mspaint.exe',
                'chrome': 'chrome.exe',
                'firefox': 'firefox.exe',
                'explorer': 'explorer.exe',
                'cmd': 'cmd.exe',
                'powershell': 'powershell.exe'
            }
        elif self.system == "darwin":  # macOS
            return {
                'textedit': 'open -a TextEdit',
                'calculator': 'open -a Calculator',
                'safari': 'open -a Safari',
                'chrome': 'open -a "Google Chrome"',
                'firefox': 'open -a Firefox'
            }
        else:  # Linux
            return {
                'texteditor': 'gedit',
                'calculator': 'gnome-calculator',
                'browser': 'firefox',
                'chrome': 'google-chrome',
                'filemanager': 'nautilus'
            }
    
    def _find_matching_command(self, description, command_map):
        """Find the best matching command from description"""
        description_lower = description.lower()
        
        for command_key in self.allowed_commands:
            if command_key in description_lower:
                return command_key
        
        # If no direct match, check for synonyms
        synonym_map = {
            'notepad': ['text editor', 'write', 'note'],
            'calculator': ['calc', 'calculate', 'math'],
            'paint': ['draw', 'paint', 'drawing'],
            'chrome': ['browser', 'web', 'internet'],
            'firefox': ['browser', 'web'],
            'explorer': ['files', 'file manager', 'folder']
        }
        
        for command_key, synonyms in synonym_map.items():
            if any(synonym in description_lower for synonym in synonyms):
                return command_key
        
        return None
    
    def _run_command(self, command):
        """Execute the system command"""
        try:
            if self.system == "windows":
                # Use subprocess without shell for security
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False
                )
            else:
                # For Unix-like systems, might need shell=True for some commands
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True
                )
            
            # Don't wait for process to complete for GUI applications
            if any(app in command for app in ['.exe', 'open -a', 'gedit', 'calculator']):
                return {"success": True, "output": "Application launched"}
            else:
                stdout, stderr = process.communicate(timeout=10)
                return {
                    "success": process.returncode == 0,
                    "output": stdout.decode() if stdout else stderr.decode()
                }
                
        except subprocess.TimeoutExpired:
            process.kill()
            return {"success": True, "output": "Process started"}
        except Exception as e:
            return {"success": False, "output": str(e)}


class OnlineLLMClient:
    def __init__(self):
        self.api_key = getattr(settings, 'OPENAI_API_KEY', None)
        self.use_openai = getattr(settings, 'USE_OPENAI_FOR_RESPONSES', False)
    
    def get_enhanced_response(self, user_message, emotion_context, conversation_history=None):
        """Get enhanced response using online LLM"""
        if not self.use_openai or not self.api_key:
            return None
        
        try:
            import openai
            openai.api_key = self.api_key
            
            # Build context for the LLM
            system_message = self._build_system_prompt(emotion_context)
            
            messages = [
                {"role": "system", "content": system_message}
            ]
            
            # Add conversation history
            if conversation_history:
                for conv in conversation_history[-6:]:  # Last 3 exchanges
                    messages.extend([
                        {"role": "user", "content": conv.get('user_message', '')},
                        {"role": "assistant", "content": conv.get('assistant_response', '')}
                    ])
            
            # Add current message
            messages.append({"role": "user", "content": user_message})
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return None
    
    def _build_system_prompt(self, emotion_context):
        """Build system prompt with emotion context"""
        emotion = emotion_context.get('dominant_emotion', 'neutral')
        confidence = emotion_context.get('confidence', 0)
        
        emotion_prompts = {
            'joy': "The user is feeling happy and joyful. Respond in an enthusiastic, positive, and engaging tone. Suggest fun activities or share in their excitement.",
            'sadness': "The user seems sad or down. Respond with empathy, comfort, and support. Offer gentle suggestions for mood improvement and be a good listener.",
            'anger': "The user appears angry or frustrated. Respond calmly, acknowledge their feelings, and try to de-escalate the situation. Offer practical solutions.",
            'fear': "The user seems anxious or fearful. Respond with reassurance, support, and practical advice. Help them feel safe and understood.",
            'disgust': "The user seems displeased or disgusted. Respond diplomatically and try to understand the cause. Offer alternative perspectives.",
            'surprise': "The user seems surprised. Respond with appropriate interest and engagement. Match their energy level appropriately.",
            'neutral': "Respond in a friendly, helpful, and engaging tone. Be conversational and natural."
        }
        
        base_prompt = emotion_prompts.get(emotion, emotion_prompts['neutral'])
        
        # Add instruction handling capability
        instruction_prompt = """
        If the user asks you to perform any system commands (like opening applications, files, or executing tasks), 
        analyze their intent and provide appropriate guidance. If it's a simple system command you can help with, 
        mention that you can assist with it.
        """
        
        return f"{base_prompt}\n\n{instruction_prompt}\n\nKeep responses conversational and under 200 words."


# Global instances
command_executor = CommandExecutor()
online_llm_client = OnlineLLMClient()