import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import dotenv
import yaml
from core.llm import call_llm_with_tools, call_llm, LLMError
from core.imgen import generate_image_with_retry, generate_image_prompt
from core.voice import transcribe_audio, speak_text
import threading
from queue import Queue
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()
# Constants
HEURIST_BASE_URL = "https://llm-gateway.heurist.xyz"
HEURIST_API_KEY = os.getenv("HEURIST_API_KEY")
LARGE_MODEL_ID = os.getenv("LARGE_MODEL_ID")
SMALL_MODEL_ID = os.getenv("SMALL_MODEL_ID")
TWEET_WORD_LIMITS = [15, 20, 30, 35]
IMAGE_GENERATION_PROBABILITY = 0.3
DRYRUN = os.getenv("DRYRUN")
BASE_IMAGE_PROMPT = ""


# Add new constants
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image based on a text prompt, any request to create should be handled by this tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to generate the image from"
                    }
                },
                "required": ["prompt"]
            }
        }
    },
]

class PromptConfig:
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Get the project root directory (2 levels up from the current file)
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "prompts.yaml"
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at {self.config_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def get_system_prompt(self) -> str:
        return self.config['system']['base']

    def get_basic_settings(self) -> list:
        return self.config['character']['basic_settings']

    def get_interaction_styles(self) -> list:
        return self.config['character']['interaction_styles']

    def get_basic_prompt_template(self) -> str:
        return self.config['templates']['basic_prompt']

    def get_tweet_instruction_template(self) -> str:
        return self.config['templates']['tweet_instruction']

    def get_context_twitter_template(self) -> str:
        return self.config['templates']['context_twitter']

    def get_tweet_ideas(self) -> list:
        return self.config['tweet_ideas']['options']

    def get_twitter_rules(self) -> str:
        return self.config['rules']['twitter']
    
    def get_reply_to_comment_template(self) -> str:
        return self.config['reply']['reply_to_comment_template']

    def get_reply_to_tweet_template(self) -> str:
        return self.config['reply']['reply_to_tweet_template']
    
    def get_telegram_rules(self) -> str:
        return self.config['rules']['telegram']

class CoreAgent:
    def __init__(self):
        self.prompt_config = PromptConfig()
        self.interfaces = {}
        self._message_queue = Queue()
        self._lock = threading.Lock()
    
    def register_interface(self, name, interface):
        with self._lock:
            self.interfaces[name] = interface
        
    async def pre_validation(self, message: str) -> bool:
        """
        Pre-validation of the message
        
        Args:
            message: The user's message
            
        Returns:
            True if the message is valid, False otherwise
        """
        try:
            response = call_llm(
                HEURIST_BASE_URL,
                HEURIST_API_KEY, 
                SMALL_MODEL_ID,
                self.prompt_config.get_telegram_rules(),
                message,
                temperature=0.5,
            )
            print(response)
            response = response.lower()
            validation = False if "false" in response else True if "true" in response else False
            print(validation)
            return validation
        except Exception as e:
            logger.error(f"Pre-validation failed: {str(e)}")
            return False

    async def handle_image_generation(self, prompt: str, base_prompt: str = "") -> Optional[str]:
        """
        Handle image generation requests with retry logic
        
        Args:
            prompt: The image generation prompt
            base_prompt: Optional base prompt to prepend
            
        Returns:
            Generated image URL or None if failed
        """
        try:
            full_prompt = base_prompt + prompt if base_prompt else prompt
            result = generate_image_with_retry(prompt=full_prompt)
            print(result)
            return result
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            return None



    async def transcribe_audio(self, audio_file_path: Path) -> str:
        """
        Handle voice transcription requests
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            return transcribe_audio(audio_file_path)
        except Exception as e:
            logger.error(f"Voice transcription failed: {str(e)}")
            raise

    async def handle_text_to_speech(self, text: str) -> Path:
        """
        Handle text-to-speech conversion
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Path to generated audio file
        """
        try:
            return speak_text(text)
        except Exception as e:
            logger.error(f"Text-to-speech conversion failed: {str(e)}")
            raise

    async def handle_message(self, message: str, source_interface: str = None, chat_id: str = None):
        """
        Handle message and optionally notify other interfaces.        
        Args:
            message: The message to process
            source_interface: Optional name of the interface that sent the message
            
        Returns:
            tuple: (text_response, image_url)
        """
        logger.info(f"Handling message from {source_interface}")
        logger.info(f"registered interfaces: {self.interfaces}")

        preValidation = False if source_interface in ["api"] else True
        preValidationResult = True
        if preValidation:
            preValidationResult = await self.pre_validation(message)

        if not preValidationResult:
            return "Ignoring message.", None
        try:
            # Call LLM with tools to process the message
            response = call_llm_with_tools(
                HEURIST_BASE_URL,
                HEURIST_API_KEY, 
                LARGE_MODEL_ID,
                self.prompt_config.get_system_prompt(),
                message,
                temperature=0.01,
                tools=TOOLS
            )
            print(response)
            # Check if response is valid
            if not response:
                return "Sorry, I couldn't process your message.", None
                
            # Extract text response
            text_response = ""
            if hasattr(response, 'content') and response.content:
                text_response = response.content
                
            image_url = None
            
            # Check if image generation was requested
            if 'tool_calls' in response and response['tool_calls']:
                tool_call = response['tool_calls']
                if tool_call.function.name == 'generate_image':
                    args = json.loads(tool_call.function.arguments)
                    image_url = await self.handle_image_generation(args['prompt'])

            
            # Notify other interfaces if needed
            if source_interface and chat_id:
                for interface_name, interface in self.interfaces.items():
                    if interface_name != source_interface:
                        await self.send_to_interface(interface_name, {
                            'type': 'message',
                            'content': text_response,
                            'image_url': image_url,
                            'source': source_interface,
                            'chat_id': chat_id
                        })
            
            return text_response, image_url
            
        except LLMError as e:
            logger.error(f"LLM processing failed: {str(e)}")
            return "Sorry, I encountered an error processing your message.", None
        except Exception as e:
            logger.error(f"Message handling failed: {str(e)}")
            return "Sorry, something went wrong.", None

    async def send_to_interface(self, target_interface: str, message: dict):
        """
        Send a message to a specific interface
        
        Args:
            target_interface (str): Name of the interface to send to
            message (dict): Message data containing at minimum:
                {
                    'type': str,  # Type of message (e.g., 'message', 'image', 'voice')
                    'content': str,  # Main content
                    'image_url': Optional[str],  # Optional image URL
                    'source': Optional[str]  # Source interface name
                }
        
        Returns:
            bool: True if message was queued successfully, False otherwise
        """
        try:
            with self._lock:
                if target_interface not in self.interfaces:
                    logger.error(f"Interface {target_interface} not registered")
                    return False
                
                # Validate message format
                if not isinstance(message, dict) or 'type' not in message or 'content' not in message:
                    logger.error("Invalid message format")
                    return False
                
                # Add timestamp and target
                message['timestamp'] = datetime.now().isoformat()
                message['target'] = target_interface
                
                # Queue the message
                self._message_queue.put(message)
                
                # Get interface instance
                interface = self.interfaces[target_interface]
                logger.info(f"Interface: {interface}")
                logger.info(f"Message: {message}")
                logger.info("trying to send message")
                # Handle different message types
                logger.info(f"Message type: {message['type']}")
                if message['type'] == 'message':
                    logger.info(f"interface has method {hasattr(interface, 'send_message')}")
                    if hasattr(interface, 'send_message'):
                        try:
                            logger.info(f"Attempting to send message via {target_interface} interface")
                            await interface.send_message(
                                chat_id=message['chat_id'],
                                message=message['content'],
                                image_url=message['image_url']
                            )
                            logger.info("Message sent successfully")
                        except Exception as e:
                            logger.error(f"Failed to send message via {target_interface}: {str(e)}")
                            raise
                # Log successful queue
                logger.info(f"Message queued for {target_interface}: {message['type']}")
                return True
                
        except Exception as e:
            logger.error(f"Error sending message to {target_interface}: {str(e)}")
            return False