"""
Gemini AI Chatbot Integration for NeuroAid
Handles conversations with Google's Gemini API
"""

import os
import google.generativeai as genai
from typing import Optional
from dotenv import load_dotenv

# CRITICAL: Load environment variables from .env file
load_dotenv(override=True)
# ============================================================
# CONFIGURATION
# ============================================================

class ChatbotConfig:
    """Configuration for Gemini chatbot"""
    
    # Get API key from environment variable
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Model settings - Use the actual available models from your API
    MODELS_TO_TRY = [
        "models/gemini-2.5-flash",           # Latest stable flash model
        "models/gemini-2.0-flash",           # Stable 2.0 flash
        "models/gemini-flash-latest",        # Always points to latest flash
        "models/gemini-pro-latest",          # Latest pro model
        "models/gemini-2.5-pro",             # Latest 2.5 pro
        "models/gemini-2.0-flash-exp",       # Experimental 2.0
    ]
    
    # Generation settings
    TEMPERATURE = 0.7  # Creativity level (0.0 to 1.0)
    MAX_TOKENS = 2048  # Maximum response length
    TOP_P = 0.95
    TOP_K = 40
    
    # System instruction for health-focused responses
    SYSTEM_PROMPT = """You are a helpful AI health assistant for NeuroAid, a medical information platform. 

Your role:
- Provide accurate, evidence-based health information
- Explain medical concepts in simple, understandable language
- Be empathetic and supportive when discussing health concerns
- Always remind users that you provide general information, not medical advice
- Encourage users to consult healthcare professionals for diagnosis and treatment

Guidelines:
- Be clear and concise in your responses
- Use bullet points and structure for better readability
- Provide context and explanations for medical terms
- Be honest about limitations - say "I don't know" when uncertain
- Never diagnose or prescribe medication
- Always emphasize the importance of professional medical care

Tone: Professional, friendly, and supportive"""


# ============================================================
# GEMINI CHATBOT CLASS
# ============================================================

class GeminiChatbot:
    """Handles interactions with Google Gemini API"""
    
    def __init__(self):
        """Initialize the chatbot with API configuration"""
        self.api_key = ChatbotConfig.GEMINI_API_KEY
        self.model = None
        self.chat_session = None
        self.model_name = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Gemini API and model"""
        print("\n" + "="*60)
        print("INITIALIZING GEMINI CHATBOT")
        print("="*60)
        
        if not self.api_key:
            print("❌ GEMINI_API_KEY not found in environment variables")
            print("\n📝 To fix this:")
            print("   1. Create/edit .env file in your project root")
            print("   2. Add this line: GEMINI_API_KEY=your-actual-key")
            print("   3. Get your key from: https://aistudio.google.com/app/apikey")
            print("   4. Restart your Flask app")
            print("="*60 + "\n")
            return False
        
        print(f"✅ API Key found: {self.api_key[:15]}...")
        
        try:
            # Configure API
            genai.configure(api_key=self.api_key)
            
            # Generation config
            generation_config = {
                "temperature": ChatbotConfig.TEMPERATURE,
                "top_p": ChatbotConfig.TOP_P,
                "top_k": ChatbotConfig.TOP_K,
                "max_output_tokens": ChatbotConfig.MAX_TOKENS,
            }
            
            # Try each model until one works
            for model_name in ChatbotConfig.MODELS_TO_TRY:
                try:
                    print(f"   Trying model: {model_name}...")
                    
                    # Create model with safety settings to avoid blocks
                    safety_settings = [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE",
                        },
                    ]
                    
                    # For all models, try with system instruction
                    try:
                        self.model = genai.GenerativeModel(
                            model_name=model_name,
                            generation_config=generation_config,
                            safety_settings=safety_settings,
                            system_instruction=ChatbotConfig.SYSTEM_PROMPT
                        )
                    except:
                        # If system_instruction fails, try without it
                        self.model = genai.GenerativeModel(
                            model_name=model_name,
                            generation_config=generation_config,
                            safety_settings=safety_settings
                        )
                    
                    # Test the model with a simple message
                    test_chat = self.model.start_chat(history=[])
                    test_response = test_chat.send_message("Hi")
                    
                    # If we get here, the model works!
                    self.model_name = model_name
                    self.chat_session = self.model.start_chat(history=[])
                    
                    print(f"✅ Gemini chatbot initialized with model: {model_name}")
                    print("="*60 + "\n")
                    return True
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"   ❌ {model_name} failed: {error_msg[:80]}...")
                    
                    # Check for common errors
                    if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
                        print("\n❌ ERROR: Your API key is invalid!")
                        print("   1. Check your .env file")
                        print("   2. Get a new key: https://aistudio.google.com/app/apikey")
                        print("   3. Make sure there are no extra spaces")
                        print("="*60 + "\n")
                        return False
                    
                    if "403" in error_msg and "leaked" in error_msg.lower():
                        print("\n❌ ERROR: Your API key is flagged as leaked!")
                        print("   1. Delete this key at: https://aistudio.google.com/app/apikey")
                        print("   2. Create a BRAND NEW key")
                        print("   3. Update your .env file")
                        print("   4. Restart the application")
                        print("="*60 + "\n")
                        return False
                    
                    continue
            
            # If we get here, no model worked
            print("\n❌ All models failed. Possible issues:")
            print("   1. API key is invalid")
            print("   2. No internet connection")
            print("   3. Gemini API is down")
            print("   4. API key doesn't have access to these models")
            print("\n🔧 Try:")
            print("   1. Check API key at: https://aistudio.google.com/app/apikey")
            print("   2. Make sure API key is enabled")
            print("   3. Test internet connection")
            print("="*60 + "\n")
            return False
            
        except Exception as e:
            print(f"❌ Failed to initialize Gemini chatbot: {e}")
            print("="*60 + "\n")
            return False
    
    def is_available(self) -> bool:
        """Check if chatbot is ready to use"""
        return self.model is not None and self.chat_session is not None
    
    def get_response(self, user_message: str) -> tuple[bool, str]:
        """
        Get response from Gemini for user message
        
        Args:
            user_message: User's input message
            
        Returns:
            Tuple of (success: bool, response: str)
        """
        if not self.is_available():
            return False, "Chatbot is not properly initialized. Please check API key configuration."
        
        if not user_message or not user_message.strip():
            return False, "Please enter a message."
        
        try:
            # Send message and get response
            response = self.chat_session.send_message(user_message.strip())
            
            # Extract text from response
            response_text = response.text
            
            if not response_text:
                return False, "I couldn't generate a response. Please try again."
            
            return True, response_text
            
        except Exception as e:
            error_message = str(e)
            print(f"❌ Gemini API error: {error_message}")
            
            # Handle common errors
            if "API_KEY_INVALID" in error_message or "Invalid API key" in error_message:
                return False, "API key is invalid. Please check your configuration."
            elif "API key not valid" in error_message:
                return False, "API key is not valid. Please check your GEMINI_API_KEY."
            elif "quota" in error_message.lower():
                return False, "API quota exceeded. Please try again later."
            elif "rate" in error_message.lower():
                return False, "Rate limit exceeded. Please wait a moment and try again."
            elif "safety" in error_message.lower() or "blocked" in error_message.lower():
                return False, "Response was blocked due to safety concerns. Please rephrase your question."
            elif "404" in error_message:
                return False, f"Model '{self.model_name}' not found. This is an API configuration issue."
            else:
                return False, f"An error occurred: {error_message}"
    
    def reset_conversation(self):
        """Start a new conversation (clear history)"""
        if self.model:
            self.chat_session = self.model.start_chat(history=[])
            return True
        return False
    
    def get_conversation_history(self) -> list:
        """Get the current conversation history"""
        if self.chat_session:
            return self.chat_session.history
        return []


# ============================================================
# SINGLETON INSTANCE
# ============================================================

# Create a single chatbot instance to be used across the app
_chatbot_instance = None

def get_chatbot() -> GeminiChatbot:
    """Get or create the chatbot singleton instance"""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = GeminiChatbot()
    return _chatbot_instance


# ============================================================
# SIMPLE FUNCTION INTERFACE
# ============================================================

def get_response(user_input: str) -> str:
    """
    Simple function to get chatbot response (for backward compatibility)
    
    Args:
        user_input: User's message
        
    Returns:
        Bot's response string
    """
    chatbot = get_chatbot()
    success, response = chatbot.get_response(user_input)
    return response if success else f"Error: {response}"


# ============================================================
# TEST FUNCTION
# ============================================================

def test_chatbot():
    """Test the chatbot functionality"""
    print("\n" + "=" * 60)
    print("TESTING GEMINI CHATBOT")
    print("=" * 60)
    
    chatbot = get_chatbot()
    
    if not chatbot.is_available():
        print("❌ Chatbot is not available. Check API key configuration.")
        print("\nTroubleshooting:")
        print("1. Make sure you have a valid API key")
        print("2. Get key from: https://aistudio.google.com/app/apikey")
        print("3. Add to .env file: GEMINI_API_KEY=your-key")
        print("4. Restart the application")
        return
    
    print(f"✅ Using model: {chatbot.model_name}")
    print()
    
    # Test messages
    test_messages = [
        "Hello! What can you help me with?",
        "What are common symptoms of flu?",
        "How can I maintain good health?"
    ]
    
    for i, msg in enumerate(test_messages, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/3")
        print(f"{'='*60}")
        print(f"👤 User: {msg}")
        success, response = chatbot.get_response(msg)
        if success:
            # Show first 300 chars
            preview = response[:300] + "..." if len(response) > 300 else response
            print(f"🤖 Bot: {preview}")
        else:
            print(f"❌ Error: {response}")
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Run test if executed directly
    test_chatbot()