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
    files = scanner._find_files(tmp_path)
    
    assert len(files) == 2
    assert all(f.suffix == '.py' for f in files)


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
    files = scanner._find_files(tmp_path)
    
    # Should only find main.py, not cached.pyc
    assert len(files) == 1
    assert files[0].name == "main.py"

def test_scanner_finds_js_ts_files(tmp_path):
    """Test that scanner finds JS/TS files."""
    (tmp_path / "script.js").write_text("console.log('hi')")
    (tmp_path / "app.ts").write_text("const x: int = 1;")
    (tmp_path / "comp.jsx").write_text("const C = () => <div/>")
    (tmp_path / "other.txt").write_text("text")
    
    scanner = Scanner()
    files = scanner._find_files(tmp_path)
    
    assert len(files) == 3
    extensions = {f.suffix for f in files}
    assert extensions == {'.js', '.ts', '.jsx'}
