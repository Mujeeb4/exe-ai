"""
Tests for configuration and session management (Day 1).
"""

import pytest
import json
from pathlib import Path
from src.config import ConfigManager, ExeConfig
from src.session import SessionState


def test_config_manager_save_and_load(tmp_path):
    """Test that config can be saved and loaded."""
    config_manager = ConfigManager(config_dir=tmp_path / ".exe")
    
    # Create and save config
    config = ExeConfig(
        api_key="test-key-123",
        auto_apply=True,
        model="gpt-4",
        focus_path="src/core"
    )
    config_manager.save(config)
    
    # Verify file exists
    assert config_manager.exists()
    
    # Load and verify
    loaded_config = config_manager.load()
    assert loaded_config is not None
    assert loaded_config.api_key == "test-key-123"
    assert loaded_config.auto_apply == True
    assert loaded_config.model == "gpt-4"
    assert loaded_config.focus_path == "src/core"


def test_config_uses_model_dump(tmp_path):
    """Test that config uses Pydantic v2 model_dump method."""
    config_manager = ConfigManager(config_dir=tmp_path / ".exe")
    
    config = ExeConfig(api_key="test-key")
    config_manager.save(config)
    
    # Read the JSON file directly
    with open(config_manager.config_file, 'r') as f:
        data = json.load(f)
    
    # Verify structure
    assert "api_key" in data
    assert "auto_apply" in data
    assert "model" in data


def test_session_state_focus():
    """Test session state focus management."""
    state = SessionState()
    
    # Initially no focus
    assert not state.is_focused()
    assert state.get_focus() is None
    
    # Set focus (using current directory as it exists)
    state.project_root = Path.cwd()
    state.set_focus(".")
    
    assert state.is_focused()
    assert state.get_focus() is not None
    
    # Clear focus
    state.clear_focus()
    assert not state.is_focused()


def test_session_state_invalid_path():
    """Test that session state rejects invalid paths."""
    state = SessionState()
    state.project_root = Path.cwd()
    
    with pytest.raises(ValueError, match="Path does not exist"):
        state.set_focus("nonexistent/path/that/does/not/exist")


def test_session_state_relative_path(tmp_path):
    """Test that session state handles relative paths."""
    state = SessionState()
    state.project_root = tmp_path
    
    # Create a test directory
    test_dir = tmp_path / "test_folder"
    test_dir.mkdir()
    
    # Set focus with relative path
    state.set_focus("test_folder")
    
    assert state.is_focused()
    assert "test_folder" in state.get_focus()


def test_config_defaults():
    """Test that ExeConfig has correct defaults."""
    config = ExeConfig(api_key="test-key")
    
    assert config.auto_apply == False
    assert config.router_model == "gemini-1.5-flash-8b"
    assert config.coder_model == "gemini-2.0-flash-exp"
    assert config.focus_path is None
