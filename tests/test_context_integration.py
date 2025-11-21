import pytest
from src.core.coder import Coder
from src.core.models import RouterOutput, CodeChunk
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_genai():
    with patch('src.core.coder.genai.Client') as mock:
        yield mock

def test_coder_with_conversation_context(mock_genai):
    """Test that conversation history is included in prompts"""
    mock_response = MagicMock()
    mock_response.text = "This is an answer with context."
    mock_genai.return_value.models.generate_content.return_value = mock_response
    
    coder = Coder(api_key="test")
    
    router_output = RouterOutput(intent="question", relevant_files=[])
    chunks = []
    conversation_history = [
        {"role": "user", "content": "What is the main function?", "timestamp": "2024-01-01"},
        {"role": "assistant", "content": "The main function is in main.py", "timestamp": "2024-01-01"}
    ]
    
    # Execute
    output = coder.process("Tell me more about it", router_output, chunks, conversation_history)
    
    # Verify
    assert output.type == "answer"
    
    # Check that conversation context was included
    call_args = mock_genai.return_value.models.generate_content.call_args
    prompt = call_args.kwargs['contents']
    
    assert "Recent conversation:" in prompt
    assert "USER: What is the main function?" in prompt

def test_coder_without_conversation_context(mock_genai):
    """Test that coder works without conversation history"""
    mock_response = MagicMock()
    mock_response.text = "This is an answer."
    mock_genai.return_value.models.generate_content.return_value = mock_response
    
    coder = Coder(api_key="test")
    
    router_output = RouterOutput(intent="question", relevant_files=[])
    chunks = []
    
    # Execute without conversation history
    output = coder.process("What is the main function?", router_output, chunks)
    
    # Verify
    assert output.type == "answer"
    assert output.content == "This is an answer."
