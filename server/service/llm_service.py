# Open AI:https://platform.openai.com/docs/guides/fine-tuning,  https://medium.com/@abed63/flask-application-with-openai-chatgpt-integration-tutorial-958588ac6bdf
# Google Gemini: https://ai.google.dev/gemini-api/docs
# Anthropic: https://docs.anthropic.com/en/docs/build-with-claude/text-generation
# Mistral :https://docs.mistral.ai/api/
# langchain reference: https://python.langchain.com/api_reference/

import os
# from flask import json
# from google.cloud import aiplatform
# import google.generativeai as genai 
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai.chat_models import ChatMistralAI
from service.openai_api import get_open_api_key
from service.base import BOT_REGISTRY, BaseEngine, EngineResponse  

class OpenAIEngine(BaseEngine):
    NAME = "openai"
    DISPLAY_NAME = "GPT 4o Mini"

    def send(self, message: str) -> EngineResponse:
        openai_api_key = get_open_api_key()
        
        if not openai_api_key:
            return EngineResponse(text="Error: Failed to retrieve OpenAI API key.")
        
        openai_chain = ChatOpenAI(
            api_key=openai_api_key, 
            model="gpt-4o-mini-2024-07-18",
            temperature=0.5, 
            max_tokens=1024,   
            top_p=0.9      
        )
        
        try:
            response = openai_chain.invoke(message)
            response_content = response.content if hasattr(response, 'content') else "No content found"
            return EngineResponse(text=response_content)

        except Exception as e:
            return EngineResponse(text=f"Error: {str(e)}")

class GoogleVertexAIEngine(BaseEngine):
    NAME = "Gemini-1.5-Flash"
    DISPLAY_NAME = "Gemini-1.5-Flash"

    def send(self, message: str) -> EngineResponse:
        google_api_key = os.getenv('GOOGLE_API_KEY')
        
        if not google_api_key:
            return EngineResponse(text="Error: Google Gemini API key not found.")
        
        try:
            google_chain = ChatGoogleGenerativeAI(
                api_key=google_api_key,
                model="gemini-1.5-flash",  
                temperature=0.7,           
                max_tokens=150,           
                top_p=0.9                  
            )

            response = google_chain.invoke(message)
            response_content = response.content if hasattr(response, 'content') else "No content found"
            return EngineResponse(text=response_content)

        except Exception as e:

            return EngineResponse(text=f"Error: {str(e)}")
        
class MistralAIEngine(BaseEngine):
    NAME = "Mistral AI small latest"
    DISPLAY_NAME = "Mistral AI small latest"

    def send(self, message: str) -> EngineResponse:
        mistral_api_key = os.getenv('MISTRALAI_API_KEY')
        print(f"Mistral : {mistral_api_key}")
        
        if not mistral_api_key:
            return EngineResponse(text="Error: Mistral AI API key not found.")
        
        mistral_chain = ChatMistralAI(
            api_key=mistral_api_key,
            model="mistral-small-latest",  
            temperature=0.7,
            max_tokens=2048,     
            top_p=1.0
        )
        
        try:

            response = mistral_chain.invoke(message)
            response_text = response['text'] if isinstance(response, dict) else response.content
            return EngineResponse(text=response_text)
        except Exception as e:
            return EngineResponse(text=f"Error: {str(e)}")
        
class AnthropicEngine(BaseEngine):
    NAME = "anthropic"
    DISPLAY_NAME = "Anthropic Claude unpaid"

    def send(self, message: str) -> EngineResponse:
        authropic_api_key = os.getenv('AUTHROPIC_API_KEY')
        if not authropic_api_key:
            print("Error: Authropic API key not found.")
            return EngineResponse(text="Error: Authropic API key not found.")

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
