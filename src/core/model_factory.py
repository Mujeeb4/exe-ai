"""
Model Factory for creating ADK-compatible model instances.
Supports native Google GenAI models and third-party models via LiteLLM.
"""

from typing import Any, Optional
from google.genai import Client
from litellm import completion
import os

class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create_model(model_name: str, api_key: Optional[str] = None) -> Any:
        """
        Create a model instance based on the model name.
        
        Args:
            model_name: Name of the model (e.g., 'gemini-1.5-flash', 'gpt-4o')
            api_key: Optional API key (if not in env vars)
            
        Returns:
            A model object compatible with ADK LlmAgent (string for Gemini, or wrapper for others)
        """
        # For Google models, ADK often just takes the model name string if configured correctly,
        # or we might need to pass the client.
        # Based on ADK docs/usage, LlmAgent usually takes a 'model' string for Gemini.
        
        if model_name.startswith("gemini-"):
            # Return the model name string for native ADK support
            return model_name
            
        # For non-Google models, we use LiteLLM.
        # ADK's LlmAgent might expect a specific interface. 
        # If ADK doesn't natively support LiteLLM objects, we might need a wrapper.
        # However, recent ADK versions might support custom model callables or objects.
        # Assuming we can pass a LiteLLM wrapper or that we need to wrap it to look like a Google model.
        
        # Since I cannot see the ADK source code for LlmAgent internals regarding custom models,
        # I will create a wrapper that mimics the expected behavior if possible,
        # or simply return the model name if we are patching the agent to use LiteLLM.
        
        # Strategy: Return a LiteLLMWrapper that implements generate_content
        return LiteLLMWrapper(model_name)

class LiteLLMWrapper:
    """Wrapper to make LiteLLM models look like Google GenAI models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def generate_content(self, contents: str, config: Any = None) -> Any:
        """
        Mimic google.genai.models.GenerativeModel.generate_content
        """
        # Extract text from contents (which might be complex ADK objects)
        prompt = str(contents)
        
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            # Map config if possible
        )
        
        # Convert response to look like Google's GenerateContentResponse
        return LiteLLMResponseAdapter(response)

class LiteLLMResponseAdapter:
    """Adapter to make LiteLLM response look like Google GenAI response."""
    
    def __init__(self, response: Any):
        self.response = response
        self.text = response.choices[0].message.content
        
    @property
    def parts(self):
        from google.genai import types
        return [types.Part(text=self.text)]
