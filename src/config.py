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
    api_key: str
    auto_apply: bool = False
    model: str = "gpt-4"
    focus_path: Optional[str] = None


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
