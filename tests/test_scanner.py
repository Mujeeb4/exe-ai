"""
Tests for the directory scanner.
"""

import pytest
from pathlib import Path
from src.ingestion.scanner import Scanner


def test_scanner_finds_python_files(tmp_path):
    """Test that scanner finds Python files."""
    # Create test files
    (tmp_path / "test.py").write_text("print('hello')")
    (tmp_path / "module.py").write_text("def func(): pass")
    (tmp_path / "README.md").write_text("# README")
    
    scanner = Scanner()
    python_files = scanner._find_python_files(tmp_path)
    
    assert len(python_files) == 2
    assert all(f.suffix == '.py' for f in python_files)


def test_scanner_respects_gitignore(tmp_path):
    """Test that scanner respects .gitignore patterns."""
    # Create .gitignore
    (tmp_path / ".gitignore").write_text("__pycache__/\n*.pyc")
    
    # Create files
    (tmp_path / "main.py").write_text("pass")
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "cached.pyc").write_text("")
    
    scanner = Scanner()
    python_files = scanner._find_python_files(tmp_path)
    
    # Should only find main.py, not cached.pyc
    assert len(python_files) == 1
    assert python_files[0].name == "main.py"
