import pytest
from unittest.mock import MagicMock, patch
from src.core.coder import Coder
from src.core.models import RouterOutput, CodeChunk

@pytest.fixture
def mock_openai():
    with patch('src.core.coder.openai.OpenAI') as mock:
        yield mock

def test_coder_strips_markdown(mock_openai):
    # Setup
    coder = Coder(api_key="test")
    
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "```diff\n--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-old\n+new\n```"
    mock_openai.return_value.chat.completions.create.return_value = mock_response
    
    router_output = RouterOutput(intent="code_edit", relevant_files=["file.py"])
    chunks = []
    
    # Execute
    output = coder.process("change code", router_output, chunks)
    
    # Verify
    assert output.type == "patch"
    assert output.content == "--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-old\n+new"

def test_coder_answer(mock_openai):
    # Setup
    coder = Coder(api_key="test")
    
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is an answer."
    mock_openai.return_value.chat.completions.create.return_value = mock_response
    
    router_output = RouterOutput(intent="question", relevant_files=[])
    chunks = []
    
    # Execute
    output = coder.process("question", router_output, chunks)
    
    # Verify
    assert output.type == "answer"
    assert output.content == "This is an answer."
