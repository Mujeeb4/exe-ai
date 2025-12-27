"""
Tests for CoderAdapter - requires real ADK agents.
Mark as skipped to run with real API credentials only.
"""
import pytest
from src.core.adk_adapters import CoderAdapter
from src.core.models import RouterOutput, CodeChunk


@pytest.mark.skip(reason="Requires real API key - run manually with valid credentials")
def test_coder_strips_markdown():
    coder = CoderAdapter(api_key="test")
    
    router_output = RouterOutput(intent="code_edit", relevant_files=["file.py"])
    chunks = [
        CodeChunk(
            file_path="file.py",
            chunk_id="1",
            content="old code",
            start_line=1,
            end_line=1,
            chunk_type="function",
            name="test",
            language="python"
        )
    ]
    
    # Execute
    output = coder.process("change old to new", router_output, chunks)
    
    # Verify
    assert output.type == "patch"
    assert len(output.content) > 0


@pytest.mark.skip(reason="Requires real API key - run manually with valid credentials")
def test_coder_answer():
    coder = CoderAdapter(api_key="test")
    
    router_output = RouterOutput(intent="question", relevant_files=[])
    chunks = []
    
    # Execute
    output = coder.process("question", router_output, chunks)
    
    # Verify
    assert output.type == "answer"
    assert len(output.content) > 0
