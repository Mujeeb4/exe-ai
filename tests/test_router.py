import pytest
from unittest.mock import MagicMock, patch
from src.core.router import Router
from src.core.models import CodeChunk, RouterOutput
from src.memory.db import VectorDB
from src.memory.embedder import Embedder

@pytest.fixture
def mock_db():
    db = MagicMock(spec=VectorDB)
    return db

@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=Embedder)
    embedder.embed_single.return_value = [0.1, 0.2, 0.3]
    return embedder

@pytest.fixture
def mock_openai():
    with patch('src.core.router.openai.OpenAI') as mock:
        client = mock.return_value
        client.chat.completions.create.return_value.choices[0].message.content = '{"intent": "question"}'
        yield mock

def test_router_initialization(mock_openai):
    router = Router(api_key="test_key", focus_path="src/utils")
    assert router.focus_path == "src/utils"

def test_router_route_respects_focus_path(mock_db, mock_embedder, mock_openai):
    # Setup
    router = Router(api_key="test_key", focus_path="src/utils")
    
    mock_chunk = CodeChunk(
        file_path="src/utils/helper.py",
        chunk_id="1",
        content="def help(): pass",
        start_line=1,
        end_line=2,
        chunk_type="function"
    )
    mock_db.search.return_value = [mock_chunk]
    
    # Execute
    output = router.route("how to help", mock_db, mock_embedder)
    
    # Verify
    mock_embedder.embed_single.assert_called_once_with("how to help")
    mock_db.search.assert_called_once()
    
    # Check arguments passed to db.search
    call_args = mock_db.search.call_args
    assert call_args.kwargs['focus_path'] == "src/utils"
    
    # Verify output
    assert output.intent == "question"
    assert output.relevant_files == ["src/utils/helper.py"]
    assert output.focus_area == "src/utils"
    assert output.relevant_chunks == [mock_chunk]

def test_router_route_no_focus_path(mock_db, mock_embedder, mock_openai):
    # Setup
    router = Router(api_key="test_key")
    
    mock_db.search.return_value = []
    
    # Execute
    router.route("query", mock_db, mock_embedder)
    
    # Verify
    call_args = mock_db.search.call_args
    assert call_args.kwargs['focus_path'] is None
