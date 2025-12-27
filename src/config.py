"""
Configuration loader for Exe.
Handles API keys, settings, and user preferences.
"""

import json
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel


class ExeConfig(BaseModel):
    """Configuration model for Exe."""
    # API Keys per provider
    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Deprecated: kept for backward compatibility (maps to google_api_key)
    api_key: Optional[str] = None
    
    # Model configuration
    router_model: str = "gemini-1.5-flash"  # Fast model for intent classification
    coder_model: str = "gemini-2.0-flash-exp"   # Advanced model for code generation
    embedding_model: str = "text-embedding-004"  # Default Google embedding
    embedding_provider: str = "google"  # "google" | "openai"
    
    auto_apply: bool = False
    focus_path: Optional[str] = None
    
    # Deprecated: kept for backward compatibility
    model: Optional[str] = None
    
    def model_post_init(self, __context) -> None:
        """Handle backward compatibility with old fields."""
        # Migrate old api_key to google_api_key
        if self.api_key and not self.google_api_key:
            self.google_api_key = self.api_key
        
        # Handle old model field
        if self.model and not self.coder_model:
            self.coder_model = self.model
        if self.model and not self.router_model:
            self.router_model = self.model
    
    def get_api_key_for_model(self, model_name: str) -> Optional[str]:
        """Return the appropriate API key for a given model."""
        provider = self.get_provider_for_model(model_name)
        return self.get_api_key_for_provider(provider)
    
    def get_provider_for_model(self, model_name: str) -> str:
        """Determine the provider from model name."""
        if model_name.startswith("gemini-") or model_name.startswith("text-embedding-004"):
            return "google"
        elif model_name.startswith(("gpt-", "o1-", "text-embedding-3")):
            return "openai"
        elif model_name.startswith("claude-"):
            return "anthropic"
        return "google"  # Default
    
    def get_api_key_for_provider(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider."""
        if provider == "google":
            return self.google_api_key or os.environ.get("GOOGLE_API_KEY")
        elif provider == "openai":
            return self.openai_api_key or os.environ.get("OPENAI_API_KEY")
        elif provider == "anthropic":
            return self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        return None
    
    def get_required_providers(self) -> set:
        """Get set of providers needed based on selected models."""
        providers = set()
        providers.add(self.get_provider_for_model(self.router_model))
        providers.add(self.get_provider_for_model(self.coder_model))
        providers.add(self.embedding_provider)
        return providers
    
    def setup_environment(self) -> None:
        """Set up environment variables for all configured providers."""
        if self.google_api_key:
            os.environ["GOOGLE_API_KEY"] = self.google_api_key
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key


class ConfigManager:
    """Manages configuration loading and saving."""
    
    def __init__(self, config_dir: Path = Path.home() / ".exe"):
        self.config_dir = config_dir
        self.config_file = config_dir / "config.json"
        
    def load(self) -> Optional[ExeConfig]:
        """Load configuration from disk."""
        if not self.config_file.exists():
            return None
        
        with open(self.config_file, 'r') as f:
            data = json.load(f)
        
        return ExeConfig(**data)
    
    def save(self, config: ExeConfig) -> None:
        """Save configuration to disk."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_file, 'w') as f:
            json.dump(config.model_dump(), f, indent=2)
    
    def exists(self) -> bool:
        """Check if config exists."""
        return self.config_file.exists()
