# Open AI:https://platform.openai.com/docs/guides/fine-tuning,  https://medium.com/@abed63/flask-application-with-openai-chatgpt-integration-tutorial-958588ac6bdf
# Google Gemini: https://ai.google.dev/gemini-api/docs
# Anthropic: https://docs.anthropic.com/en/docs/build-with-claude/text-generation
<<<<<<< HEAD

import os
# from flask import json
from google.cloud import aiplatform
import google.generativeai as genai 
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
=======
# Mistral :https://docs.mistral.ai/api/
# langchain reference: https://python.langchain.com/api_reference/

import os
# from flask import json
# from google.cloud import aiplatform
# import google.generativeai as genai 
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
>>>>>>> feature-branch
from langchain_anthropic import ChatAnthropic
from langchain_mistralai.chat_models import ChatMistralAI
from service.openai_api import get_open_api_key
from service.base import BOT_REGISTRY, BaseEngine, EngineResponse  

class OpenAIEngine(BaseEngine):
    NAME = "openai"
<<<<<<< HEAD
    DISPLAY_NAME = "OpenAI GPT"
=======
    DISPLAY_NAME = "GPT 4o Mini"
>>>>>>> feature-branch

    def send(self, message: str) -> EngineResponse:
        openai_api_key = get_open_api_key()
        
        if not openai_api_key:
            return EngineResponse(text="Error: Failed to retrieve OpenAI API key.")
        
        openai_chain = ChatOpenAI(
            api_key=openai_api_key, 
            model="gpt-4o-mini-2024-07-18",
<<<<<<< HEAD
            temperature=0.7, 
            max_tokens=150,   
            top_p=1.0        
=======
            temperature=0.5, 
            max_tokens=512,   
            top_p=0.9      
>>>>>>> feature-branch
        )
        
        try:
            # Use invoke() instead of __call__() to comply with the new method
            response = openai_chain.invoke(message)
<<<<<<< HEAD
            print(f"Chat GPT Response: {response}") 
=======
>>>>>>> feature-branch
            response_content = response.content if hasattr(response, 'content') else "No content found"
                
            return EngineResponse(text=response_content)

        except Exception as e:
            return EngineResponse(text=f"Error: {str(e)}")

class GoogleVertexAIEngine(BaseEngine):
<<<<<<< HEAD
    NAME = "googlevertex"
    DISPLAY_NAME = "Google Vertex AI"
=======
    NAME = "Gemini-1.5-Flash"
    DISPLAY_NAME = "Gemini-1.5-Flash"
>>>>>>> feature-branch

    def send(self, message: str) -> EngineResponse:
        # Retrieve Google API Key from environment
        google_api_key = os.getenv('GOOGLE_API_KEY')
<<<<<<< HEAD

        # Initialize Google AI Platform with your project and location
        aiplatform.init(
            project="284457257300",  # Your Google Cloud Project ID
            location="us-central1"   # Your Google Cloud Region
        )
        
        if not google_api_key:
            return EngineResponse(text="Error: Google Vertex AI API key not found.")
        
        print(f"Gemini response: {google_api_key}")

        # Create ChatVertexAI chain
        google_chain = ChatVertexAI(
            api_key=google_api_key,
            model="gemini-1.5-flash",  
            temperature=1.0,           
            max_tokens=100,             
            candidate_count=1         
        )

        try:
            # Invoke the model with the message
            response = google_chain.invoke(message)
            response_text = response.content  # Accessing the content of the response
            return EngineResponse(text=response_text)
        
        except Exception as e:
            return EngineResponse(text=f"Error: {str(e)}")
        
class AnthropicEngine(BaseEngine):
    NAME = "anthropic"
    DISPLAY_NAME = "Anthropic Claude"

    def send(self, message: str) -> EngineResponse:
        authropic_api_key = os.getenv('AUTHROPIC_API_KEY')
        anthropic_chain = ChatAnthropic(
            api_key=authropic_api_key,
            model="claude-3-sonnet-20240229",
            temperature=0,
            max_tokens=1024,
            timeout=None,
            # max_retries=2,                 
        )
        try:
            response_text = anthropic_chain.invoke(message)
            return EngineResponse(text=response_text)
        except Exception as e:
            return EngineResponse(text=f"Error: {str(e)}")


class MistralAIEngine(BaseEngine):
    NAME = "mistral"
    DISPLAY_NAME = "Mistral AI"
=======
        
        if not google_api_key:
            return EngineResponse(text="Error: Google Gemini API key not found.")
        
        try:
            # Initialize ChatGoogleGenerativeAI with API key
            google_chain = ChatGoogleGenerativeAI(
                api_key=google_api_key,
                model="gemini-1.5-flash",  
                temperature=0.7,           
                max_tokens=150,           
                top_p=0.9                  
            )

            # Generate response using the LLM
            response = google_chain.invoke(message)
            response_content = response.content if hasattr(response, 'content') else "No content found"
            return EngineResponse(text=response_content)

        except Exception as e:
            # Handle errors and return an appropriate error message
            return EngineResponse(text=f"Error: {str(e)}")
        
class MistralAIEngine(BaseEngine):
    NAME = "Mistral AI small latest"
    DISPLAY_NAME = "Mistral AI small latest"
>>>>>>> feature-branch

    def send(self, message: str) -> EngineResponse:
        mistral_api_key = os.getenv('MISTRALAI_API_KEY')
        print(f"Mistral : {mistral_api_key}")
        
        if not mistral_api_key:
            return EngineResponse(text="Error: Mistral AI API key not found.")
        
        mistral_chain = ChatMistralAI(
            api_key=mistral_api_key,
            model="mistral-small-latest",  
            temperature=0.7,
            max_tokens=512,     
            top_p=1.0
        )
        
        try:

            response = mistral_chain.invoke(message)
            response_text = response['text'] if isinstance(response, dict) else response.content
            return EngineResponse(text=response_text)
        except Exception as e:
            return EngineResponse(text=f"Error: {str(e)}")
<<<<<<< HEAD
=======
        
class AnthropicEngine(BaseEngine):
    NAME = "anthropic"
    DISPLAY_NAME = "Anthropic Claude unpaid"

    def send(self, message: str) -> EngineResponse:
        authropic_api_key = os.getenv('AUTHROPIC_API_KEY')
        if not authropic_api_key:
            print("Error: Authropic API key not found.")
            return EngineResponse(text="Error: Authropic API key not found.")

        # Logging the API key (Note: Don't log API keys in production!)
        print(f"Using Anthropic API key: {authropic_api_key}")
        anthropic_chain = ChatAnthropic(
            api_key=authropic_api_key,
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            max_tokens=1024,
            timeout=None,
            max_retries=2,                 
        )
        try:
            response = anthropic_chain.invoke(message)
            print(f"Chat Claude Response: {response}") 
            response_content = response.content if hasattr(response, 'content') else "No content found"
            return EngineResponse(text=response_content)
        except Exception as e:
            return EngineResponse(text=f"Error: {str(e)}")
>>>>>>> feature-branch


## EXTRASSS
# Function to call multiple LLM models and return their responses
def call_llm_models(models, user_query):
    responses = {}
    for model_name in models:
        engine_class = BOT_REGISTRY.get(model_name)
        if engine_class:
            engine = engine_class()
            response = engine.send(user_query)
            responses[model_name] = response.text
        else:
            responses[model_name] = f"Error: Model {model_name} not found or failed to respond"
    return responses