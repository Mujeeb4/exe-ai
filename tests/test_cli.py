"""
Integration tests for Day 1 CLI commands.
"""

from typer.testing import CliRunner
from pathlib import Path
import json
from src.main import app
from src.config import ConfigManager

runner = CliRunner()


def test_init_command(tmp_path, monkeypatch):
    """Test that exe init creates config correctly."""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)
    
    # Mock ConfigManager to use temp directory
    original_init = ConfigManager.__init__
    def mock_init(self, config_dir=None):
        original_init(self, config_dir=tmp_path / ".exe")
    monkeypatch.setattr(ConfigManager, "__init__", mock_init)
    
    # Simulate user input
    result = runner.invoke(app, ["init"], input="test-api-key\nn\n")
    
    assert result.exit_code == 0
    assert "Exe initialized successfully" in result.stdout


def test_init_already_initialized(tmp_path, monkeypatch):
    """Test that exe init shows warning if already initialized."""
    monkeypatch.chdir(tmp_path)
    
    # Mock ConfigManager to use temp directory
    original_init = ConfigManager.__init__
    def mock_init(self, config_dir=None):
        original_init(self, config_dir=tmp_path / ".exe")
    monkeypatch.setattr(ConfigManager, "__init__", mock_init)
    
    # First init
    runner.invoke(app, ["init"], input="test-api-key\nn\n")
    
    # Second init should show warning
    result = runner.invoke(app, ["init"])
    
    assert "already initialized" in result.stdout


def test_focus_command_requires_init(tmp_path, monkeypatch):
    """Test that focus command requires initialization."""
    monkeypatch.chdir(tmp_path)
    
    # Mock ConfigManager to use temp directory
    original_init = ConfigManager.__init__
    def mock_init(self, config_dir=None):
        original_init(self, config_dir=tmp_path / ".exe")
    monkeypatch.setattr(ConfigManager, "__init__", mock_init)
    
    result = runner.invoke(app, ["focus", "src/"])
    
    assert result.exit_code == 1
    assert "not initialized" in result.stdout


def test_focus_command_validates_path(tmp_path, monkeypatch):
    """Test that focus command validates path exists."""
    monkeypatch.chdir(tmp_path)
    
    # Mock ConfigManager to use temp directory
    original_init = ConfigManager.__init__
    def mock_init(self, config_dir=None):
        original_init(self, config_dir=tmp_path / ".exe")
    monkeypatch.setattr(ConfigManager, "__init__", mock_init)
    
    # Initialize first
    runner.invoke(app, ["init"], input="test-api-key\nn\n")
    
    # Try to focus on non-existent path
    result = runner.invoke(app, ["focus", "nonexistent/path"])
    
    assert result.exit_code == 1
    assert "does not exist" in result.stdout


def test_focus_command_success(tmp_path, monkeypatch):
    """Test successful focus command."""
    monkeypatch.chdir(tmp_path)
    
    # Mock ConfigManager to use temp directory
    original_init = ConfigManager.__init__
    def mock_init(self, config_dir=None):
        original_init(self, config_dir=tmp_path / ".exe")
    monkeypatch.setattr(ConfigManager, "__init__", mock_init)
    
    # Create a test directory
    test_dir = tmp_path / "src"
    test_dir.mkdir()
    
    # Initialize
    runner.invoke(app, ["init"], input="test-api-key\nn\n")
    
    # Set focus
    result = runner.invoke(app, ["focus", "src"])
    
    assert result.exit_code == 0
    assert "Focus mode set" in result.stdout


def test_clear_focus_command(tmp_path, monkeypatch):
    """Test clear-focus command."""
    monkeypatch.chdir(tmp_path)
    
    # Mock ConfigManager to use temp directory
    original_init = ConfigManager.__init__
    def mock_init(self, config_dir=None):
        original_init(self, config_dir=tmp_path / ".exe")
    monkeypatch.setattr(ConfigManager, "__init__", mock_init)
    
    # Create a test directory
    test_dir = tmp_path / "src"
    test_dir.mkdir()
    
    # Initialize and set focus
    runner.invoke(app, ["init"], input="test-api-key\nn\n")
    runner.invoke(app, ["focus", "src"])
    
    # Clear focus
    result = runner.invoke(app, ["clear-focus"])
    
    assert result.exit_code == 0
    assert "Focus mode cleared" in result.stdout
