"""
Unified Interface for LLM Providers.
Defines the contract that all model providers (Google, Anthropic, OpenAI) must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel

@dataclass
class ModelInfo:
    """Information about a specific AI model."""
    name: str
    provider: str
    display_name: str
    description: str
    input_token_limit: int = 0
    output_token_limit: int = 0


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the provider (e.g., 'google', 'anthropic')."""
        pass
        
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is configured and available (e.g., has API key)."""
        pass
        
    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """List available models from this provider."""
        pass
        
    @abstractmethod
    def generate_content(self, model_name: str, prompt: str, system_instruction: Optional[str] = None, **kwargs) -> str:
        """
        Generate content using the specified model.
        
        Args:
            model_name: Name of the model to use
            prompt: The user prompt
            system_instruction: Optional system instruction/context
            **kwargs: Additional generation config (temperature, etc.)
            
        Returns:
            Generated text content
        """
        pass
