import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.core.adk_adapters import RouterAdapter
from src.memory.db import VectorDB
from src.memory.embedder import Embedder
from src.core.models import CodeChunk

@pytest.fixture
def temp_db_path(tmp_path):
    db_path = tmp_path / "lancedb_test"
    yield str(db_path)

@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=Embedder)
    embedder.embed_single.return_value = [0.1] * 1536
    return embedder

@pytest.fixture
def mock_runner():
    with patch('src.core.adk_adapters.InMemoryRunner') as mock:
        mock_response = MagicMock()
        mock_response.text = '{"intent": "question"}'
        mock.return_value.models.generate_content.return_value = mock_response
        yield mock

def test_focus_mode_filters_results(temp_db_path, mock_embedder, ):
    """Test that focus mode only returns chunks from focused path"""
    # Setup DB with chunks from different directories
    db = VectorDB(db_path=temp_db_path)
    
    chunks = [
        CodeChunk(
            file_path="src/utils/helper.py",
            chunk_id="utils:1",
            content="def helper(): pass",
            start_line=1,
            end_line=1,
            chunk_type="function",
            name="helper",
            language="python"
        ),
        CodeChunk(
            file_path="src/core/main.py",
            chunk_id="core:1",
            content="def main(): pass",
            start_line=1,
            end_line=1,
            chunk_type="function",
            name="main",
            language="python"
        ),
        CodeChunk(
            file_path="tests/test_main.py",
            chunk_id="tests:1",
            content="def test_main(): pass",
            start_line=1,
            end_line=1,
            chunk_type="function",
            name="test_main",
            language="python"
        )
    ]
    
    embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
    db.add_chunks(chunks, embeddings)
    
    # Test 1: Focus on src/utils
    router_utils = RouterAdapter(api_key="test", focus_path="src/utils")
    output_utils = router_utils.route("query", db, mock_embedder)
    
    # Should only return utils files
    assert len(output_utils.relevant_chunks) >= 1
    for chunk in output_utils.relevant_chunks:
        assert chunk.file_path.startswith("src/utils")
    
    # Test 2: Focus on src/core
    router_core = RouterAdapter(api_key="test", focus_path="src/core")
    output_core = router_core.route("query", db, mock_embedder)
    
    # Should only return core files
    assert len(output_core.relevant_chunks) >= 1
    for chunk in output_core.relevant_chunks:
        assert chunk.file_path.startswith("src/core")
    
    # Test 3: No focus (should return all)
    router_all = RouterAdapter(api_key="test", focus_path=None)
    output_all = router_all.route("query", db, mock_embedder)
    
    # Should have more results
    assert len(output_all.relevant_chunks) >= len(output_utils.relevant_chunks)

def test_focus_mode_with_nested_paths(temp_db_path, mock_embedder, ):
    """Test focus mode with nested directory structures"""
    db = VectorDB(db_path=temp_db_path)
    
    chunks = [
        CodeChunk(
            file_path="src/api/v1/users.py",
            chunk_id="1",
            content="def get_users(): pass",
            start_line=1,
            end_line=1,
            chunk_type="function",
            language="python"
        ),
        CodeChunk(
            file_path="src/api/v2/users.py",
            chunk_id="2",
            content="def get_users_v2(): pass",
            start_line=1,
            end_line=1,
            chunk_type="function",
            language="python"
        ),
        CodeChunk(
            file_path="src/db/models.py",
            chunk_id="3",
            content="class User: pass",
            start_line=1,
            end_line=1,
            chunk_type="class",
            language="python"
        )
    ]
    
    embeddings = [[0.1] * 1536] * 3
    db.add_chunks(chunks, embeddings)
    
    # Focus on src/api (should include both v1 and v2)
    router = RouterAdapter(api_key="test", focus_path="src/api")
    output = router.route("users", db, mock_embedder)
    
    # Should return API files but not db files
    api_files = [c.file_path for c in output.relevant_chunks if c.file_path.startswith("src/api")]
    db_files = [c.file_path for c in output.relevant_chunks if c.file_path.startswith("src/db")]
    
    assert len(api_files) >= 1
    assert len(db_files) == 0

def test_focus_mode_performance_improvement(temp_db_path, mock_embedder, ):
    """Test that focus mode reduces search space"""
    db = VectorDB(db_path=temp_db_path)
    
    # Create many chunks across different directories
    chunks = []
    for i in range(50):
        chunks.append(CodeChunk(
            file_path=f"src/module{i}/file.py",
            chunk_id=f"chunk:{i}",
            content=f"def func{i}(): pass",
            start_line=1,
            end_line=1,
            chunk_type="function",
            language="python"
        ))
    
    embeddings = [[0.1] * 1536] * 50
    db.add_chunks(chunks, embeddings)
    
    # Focus on single module
    router_focused = RouterAdapter(api_key="test", focus_path="src/module5")
    output_focused = router_focused.route("query", db, mock_embedder)
    
    # Router without focus
    router_unfocused = RouterAdapter(api_key="test", focus_path=None)
    output_unfocused = router_unfocused.route("query", db, mock_embedder)
    
    # Focused should return fewer results (more precise)
    assert len(output_focused.relevant_chunks) < len(output_unfocused.relevant_chunks)

def test_intent_classification_accuracy(mock_embedder, ):
    """Test that router correctly classifies different intents"""
    from unittest.mock import patch
    
    db = MagicMock(spec=VectorDB)
    db.search.return_value = []
    
    test_cases = [
        ("How does authentication work?", "question"),
        ("Refactor the login function", "refactor"),
        ("Add error handling to api.py", "code_edit"),
        ("Explain the database schema", "explain")
    ]
    
    # This test requires real ADK - skip for now
    pytest.skip("Requires real API key for intent classification")


@pytest.mark.skip(reason="Requires real API key - run manually with valid credentials")
def test_focus_mode_with_repo_context(mock_embedder):
    """Test that router includes repository context when available"""
    db = MagicMock(spec=VectorDB)
    db.search.return_value = []
    
    repo_context = """
    # REPOSITORY CONTEXT
    src/
      - main.py
      - utils.py
    tests/
      - test_main.py
    """
    
    router = RouterAdapter(api_key="test", focus_path="src", repo_context=repo_context)
    output = router.route("test query", db, mock_embedder)
    
    # Verify output is generated
    assert isinstance(output, RouterOutput)
