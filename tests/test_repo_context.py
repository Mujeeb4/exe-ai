import pytest
from pathlib import Path
from src.memory.repo_context import RepositoryContext
from src.core.models import CodeChunk

def test_repository_context_building(tmp_path):
    """Test building comprehensive repository context"""
    # Create repository context manager
    cache_path = tmp_path / "repo_context.json"
    repo_context = RepositoryContext(cache_path=str(cache_path))
    
    # Create sample chunks
    chunks = [
        CodeChunk(
            file_path="src/main.py",
            chunk_id="main:1",
            content="def main():\n    print('hello')",
            start_line=1,
            end_line=2,
            chunk_type="function",
            name="main",
            language="python",
            imports=["os", "sys"]
        ),
        CodeChunk(
            file_path="src/utils.py",
            chunk_id="utils:1",
            content="class Helper:\n    pass",
            start_line=1,
            end_line=2,
            chunk_type="class",
            name="Helper",
            language="python",
            imports=["typing"]
        ),
        CodeChunk(
            file_path="tests/test_main.py",
            chunk_id="test:1",
            content="def test_main():\n    assert True",
            start_line=1,
            end_line=2,
            chunk_type="function",
            name="test_main",
            language="python"
        )
    ]
    
    # Build context
    context_str = repo_context.build_context(tmp_path, chunks)
    
    # Verify context contains key information
    assert "REPOSITORY CONTEXT" in context_str
    assert "DIRECTORY STRUCTURE" in context_str
    assert "CODEBASE STATISTICS" in context_str
    assert "FILE OVERVIEW" in context_str
    
    # Verify stats
    assert "Total Files: 3" in context_str
    assert "Total Chunks: 3" in context_str
    assert "Total Functions: 2" in context_str
    assert "Total Classes: 1" in context_str
    assert "Languages: python" in context_str
    
    # Verify file details
    assert "src/main.py" in context_str
    assert "Functions: main" in context_str
    assert "Imports: os, sys" in context_str
    
    assert "src/utils.py" in context_str
    assert "Classes: Helper" in context_str
    
    # Verify cache was created
    assert cache_path.exists()

def test_get_file_context(tmp_path):
    """Test retrieving context for a specific file"""
    cache_path = tmp_path / "repo_context.json"
    repo_context = RepositoryContext(cache_path=str(cache_path))
    
    chunks = [
        CodeChunk(
            file_path="src/main.py",
            chunk_id="main:1",
            content="def main():\n    pass",
            start_line=1,
            end_line=2,
            chunk_type="function",
            name="main",
            language="python"
        )
    ]
    
    repo_context.build_context(tmp_path, chunks)
    
    # Get file-specific context
    file_ctx = repo_context.get_file_context("src/main.py")
    
    assert file_ctx is not None
    assert "Functions: main" in file_ctx

def test_context_caching(tmp_path):
    """Test that context is properly cached and loaded"""
    cache_path = tmp_path / "repo_context.json"
    
    # Create and build context
    repo_context1 = RepositoryContext(cache_path=str(cache_path))
    chunks = [
        CodeChunk(
            file_path="test.py",
            chunk_id="1",
            content="pass",
            start_line=1,
            end_line=1,
            chunk_type="module",
            language="python"
        )
    ]
    context_str = repo_context1.build_context(tmp_path, chunks)
    
    # Create new instance and verify cache is loaded
    repo_context2 = RepositoryContext(cache_path=str(cache_path))
    cached_context = repo_context2.get_context()
    
    assert cached_context == context_str
