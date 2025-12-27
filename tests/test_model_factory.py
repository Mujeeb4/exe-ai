"""
Tests for ModelFactory.
"""

import sys
from unittest.mock import MagicMock

# Mock dependencies before importing module under test
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = MagicMock()
sys.modules["litellm"] = MagicMock()

import pytest
from src.core.model_factory import ModelFactory, LiteLLMWrapper

def test_create_gemini_model():
    """Test that Gemini models return the model name string."""
    model = ModelFactory.create_model("gemini-1.5-flash")
    assert isinstance(model, str)
    assert model == "gemini-1.5-flash"

def test_create_gpt_model():
    """Test that GPT models return a LiteLLMWrapper."""
    model = ModelFactory.create_model("gpt-4o")
    assert isinstance(model, LiteLLMWrapper)
    assert model.model_name == "gpt-4o"

def test_create_claude_model():
    """Test that Claude models return a LiteLLMWrapper."""
    model = ModelFactory.create_model("claude-3-5-sonnet-latest")
    assert isinstance(model, LiteLLMWrapper)
    assert model.model_name == "claude-3-5-sonnet-latest"
