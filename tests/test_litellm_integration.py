"""
Tests for LiteLLM Integration.
"""

import sys
from unittest.mock import MagicMock, patch

# Mock dependencies before importing module under test
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = MagicMock()
sys.modules["litellm"] = MagicMock()

import pytest
from src.core.model_factory import LiteLLMWrapper

# We need to patch the 'completion' function that is imported in model_factory
# Since we mocked litellm above, the import in model_factory got a Mock.
# We need to make sure we are testing the right thing.

def test_litellm_wrapper_generate_content():
    """Test that LiteLLMWrapper calls completion and adapts response."""
    
    # Setup the mock for completion
    # We need to patch 'src.core.model_factory.completion' which is where it's imported
    with patch('src.core.model_factory.completion') as mock_completion:
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_completion.return_value = mock_response
        
        wrapper = LiteLLMWrapper("gpt-4o")
        response = wrapper.generate_content("Hello")
        
        # Verify completion call
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        assert call_args.kwargs['model'] == "gpt-4o"
        assert call_args.kwargs['messages'] == [{"role": "user", "content": "Hello"}]
        
        # Verify response adaptation
        assert response.text == "Test response"
        # Note: We can't easily test parts because it depends on google.genai.types which is mocked
        # But we can verify the text attribute which is what we care about most
