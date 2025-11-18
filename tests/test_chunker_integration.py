"""
Test chunker on real project files to verify Day 2 implementation.
"""

import pytest
from pathlib import Path
from src.memory.chunker import PythonChunker


def test_chunker_on_config_py():
    """Test chunker on our actual config.py file."""
    config_file = Path("src/config.py")
    
    if not config_file.exists():
        pytest.skip("config.py not found")
    
    chunker = PythonChunker()
    chunks = chunker.chunk_file(config_file)
    
    # Should extract ExeConfig and ConfigManager classes
    chunk_names = [c.name for c in chunks if c.name is not None]
    
    assert any("ExeConfig" in name for name in chunk_names), "ExeConfig class not found"
    assert any("ConfigManager" in name for name in chunk_names), "ConfigManager class not found"
    
    # Should extract methods from ConfigManager
    assert any("load" in name for name in chunk_names), "load method not found"
    assert any("save" in name for name in chunk_names), "save method not found"
    
    print(f"\n✓ Found {len(chunks)} chunks in config.py")
    print(f"  Classes/Methods: {chunk_names}")


def test_chunker_on_main_py():
    """Test chunker on our actual main.py file."""
    main_file = Path("src/main.py")
    
    if not main_file.exists():
        pytest.skip("main.py not found")
    
    chunker = PythonChunker()
    chunks = chunker.chunk_file(main_file)
    
    # Should extract CLI command functions
    chunk_names = [c.name for c in chunks if c.name is not None]
    
    assert any("init" in name for name in chunk_names), "init function not found"
    assert any("start" in name for name in chunk_names), "start function not found"
    assert any("focus" in name for name in chunk_names), "focus function not found"
    
    print(f"\n✓ Found {len(chunks)} chunks in main.py")
    print(f"  Commands: {chunk_names}")


def test_chunker_on_router_py():
    """Test chunker on router.py which has a class with methods."""
    router_file = Path("src/core/router.py")
    
    if not router_file.exists():
        pytest.skip("router.py not found")
    
    chunker = PythonChunker()
    chunks = chunker.chunk_file(router_file)
    
    chunk_names = [c.name for c in chunks if c.name is not None]
    
    # Should find Router class and its methods
    assert any("Router" in name for name in chunk_names), "Router class not found"
    assert any("route" in name for name in chunk_names), "route method not found"
    
    # Check that methods are properly qualified
    method_chunks = [c for c in chunks if c.chunk_type == "method"]
    assert len(method_chunks) > 0, "No methods found"
    
    print(f"\n✓ Found {len(chunks)} chunks in router.py")
    print(f"  Router chunks: {[c.name for c in chunks if c.name is not None and 'Router' in c.name]}")


def test_chunker_on_chunker_py():
    """Test chunker on itself (meta!)."""
    chunker_file = Path("src/memory/chunker.py")
    
    if not chunker_file.exists():
        pytest.skip("chunker.py not found")
    
    chunker = PythonChunker()
    chunks = chunker.chunk_file(chunker_file)
    
    chunk_names = [c.name for c in chunks if c.name is not None]
    
    # Should find PythonChunker class and its many methods
    assert any("PythonChunker" in name for name in chunk_names), "PythonChunker class not found"
    assert any("chunk_file" in name for name in chunk_names), "chunk_file method not found"
    assert any("_extract_function_chunk" in name for name in chunk_names), "_extract_function_chunk not found"
    assert any("_extract_class_chunks" in name for name in chunk_names), "_extract_class_chunks not found"
    
    # Count methods
    method_chunks = [c for c in chunks if c.chunk_type == "method"]
    assert len(method_chunks) >= 5, f"Expected at least 5 methods, found {len(method_chunks)}"
    
    print(f"\n✓ Found {len(chunks)} chunks in chunker.py (testing itself!)")
    print(f"  Methods: {len(method_chunks)}")
    print(f"  All chunks: {chunk_names}")


def test_chunker_chunk_quality():
    """Test that chunks have proper content and metadata."""
    chunker_file = Path("src/config.py")
    
    if not chunker_file.exists():
        pytest.skip("config.py not found")
    
    chunker = PythonChunker()
    chunks = chunker.chunk_file(chunker_file)
    
    for chunk in chunks:
        # Each chunk should have required fields
        assert chunk.file_path, "Chunk missing file_path"
        assert chunk.chunk_id, "Chunk missing chunk_id"
        assert chunk.content, "Chunk missing content"
        assert chunk.start_line > 0, "Invalid start_line"
        assert chunk.end_line >= chunk.start_line, "end_line before start_line"
        assert chunk.chunk_type, "Chunk missing chunk_type"
        assert chunk.name, "Chunk missing name"
        
        # Content should not be empty
        assert len(chunk.content.strip()) > 0, f"Chunk {chunk.name} has empty content"
    
    print(f"\n✓ All {len(chunks)} chunks have proper structure")


def test_chunker_statistics():
    """Generate statistics about chunking across project files."""
    src_dir = Path("src")
    
    if not src_dir.exists():
        pytest.skip("src directory not found")
    
    chunker = PythonChunker()
    
    total_files = 0
    total_chunks = 0
    chunk_types = {}
    
    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        
        chunks = chunker.chunk_file(py_file)
        total_files += 1
        total_chunks += len(chunks)
        
        for chunk in chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
    
    print(f"\n📊 Chunking Statistics:")
    print(f"  Total Python files: {total_files}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Average chunks per file: {total_chunks / total_files:.1f}")
    print(f"\n  Chunk types:")
    for chunk_type, count in sorted(chunk_types.items()):
        print(f"    {chunk_type}: {count}")
    
    assert total_files > 0, "No Python files found"
    assert total_chunks > 0, "No chunks generated"
