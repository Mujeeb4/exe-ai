import pytest
from src.core.coder import Coder
from src.core.models import RouterOutput, CodeChunk
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_openai():
    with patch('src.core.coder.openai.OpenAI') as mock:
        yield mock

def test_coder_with_conversation_context(mock_openai):
    """Test that conversation history is included in prompts"""
    coder = Coder(api_key="test")
    
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is an answer with context."
    mock_openai.return_value.chat.completions.create.return_value = mock_response
    
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
    call_args = mock_openai.return_value.chat.completions.create.call_args
    messages = call_args.kwargs['messages']
    user_message = messages[1]['content']
    
    assert "Recent conversation:" in user_message
    assert "USER: What is the main function?" in user_message

def test_coder_without_conversation_context(mock_openai):
    """Test that coder works without conversation history"""
    coder = Coder(api_key="test")
    
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is an answer."
    mock_openai.return_value.chat.completions.create.return_value = mock_response
    
    router_output = RouterOutput(intent="question", relevant_files=[])
    chunks = []
    
    # Execute without conversation history
    output = coder.process("What is the main function?", router_output, chunks)
    
    # Verify
    assert output.type == "answer"
    assert output.content == "This is an answer."
