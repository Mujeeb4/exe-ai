"""
Integration tests for context handling in coder.
These tests use real ADK agents - ensure API key is configured.
"""
import pytest
from src.core.adk_adapters import CoderAdapter
from src.core.models import RouterOutput, CodeChunk


@pytest.mark.skip(reason="Requires real API key - run manually with valid credentials")
def test_coder_with_conversation_context():
    """Test that conversation history is included in prompts"""
    coder = CoderAdapter(api_key="test")
    
    router_output = RouterOutput(intent="question", relevant_files=[])
    chunks = [
        CodeChunk(
            file_path="src/main.py",
            chunk_id="main:1",
            content="def main():\n    print('Hello')",
            start_line=1,
            end_line=2,
            chunk_type="function",
            name="main",
            language="python"
        )
    ]
    conversation_history = [
        {"role": "user", "content": "What is the main function?", "timestamp": "2024-01-01"},
        {"role": "assistant", "content": "The main function is in main.py", "timestamp": "2024-01-01"}
    ]
    
    # Execute
    output = coder.process("Tell me more about it", router_output, chunks, conversation_history)
    
    # Verify
    assert output.type == "answer"
    assert len(output.content) > 0


@pytest.mark.skip(reason="Requires real API key - run manually with valid credentials")
def test_coder_without_conversation_context():
    """Test that coder works without conversation history"""
    coder = CoderAdapter(api_key="test")
    
    router_output = RouterOutput(intent="question", relevant_files=[])
    chunks = []
    
    # Execute without conversation history
    output = coder.process("What is the main function?", router_output, chunks)
    
    # Verify
    assert output.type == "answer"
    assert len(output.content) > 0
