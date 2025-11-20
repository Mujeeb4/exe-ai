
import pytest
from pathlib import Path
from src.memory.chunker import PythonChunker, UniversalChunker

@pytest.fixture
def py_chunker():
    return PythonChunker()

@pytest.fixture
def uni_chunker():
    return UniversalChunker()

def test_python_context_injection(py_chunker, tmp_path):
    content = """
import os
from typing import List

class FileManager:
    def read(self, path: str):
        return os.read(path)
"""
    f = tmp_path / "file_manager.py"
    f.write_text(content, encoding="utf-8")
    
    chunks = py_chunker.chunk_file(f)
    
    # Find the 'read' method chunk
    read_chunk = next(c for c in chunks if c.name == "FileManager.read")
    
    # Check if imports are included
    assert "import os" in read_chunk.content
    assert "from typing import List" in read_chunk.content
    
    # Check if class context is included
    assert "class FileManager" in read_chunk.content

def test_universal_import_injection(uni_chunker, tmp_path):
    content = """
import React from 'react';
import { useState } from 'react';

function Counter() {
    const [count, setCount] = useState(0);
    return <div>{count}</div>;
}
"""
    f = tmp_path / "Counter.jsx"
    f.write_text(content, encoding="utf-8")
    
    chunks = uni_chunker.chunk_file(f)
    
    counter_chunk = next(c for c in chunks if c.name == "Counter")
    
    # Check if imports are included
    assert "import React from 'react';" in counter_chunk.content
    assert "import { useState } from 'react';" in counter_chunk.content

def test_robust_brace_counting(uni_chunker, tmp_path):
    content = """
function tricky() {
    const str = "This string has a closing brace } inside it";
    const comment = "// This comment has a brace { inside it";
    /* 
       Multi-line comment with braces { } 
    */
    return true;
}
"""
    f = tmp_path / "tricky.js"
    f.write_text(content, encoding="utf-8")
    
    chunks = uni_chunker.chunk_file(f)
    
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.name == "tricky"
    # If brace counting was wrong, it might have ended early at the string's }
    assert 'return true;' in chunk.content
    assert '}' in chunk.content # The final closing brace

def test_java_imports(uni_chunker, tmp_path):
    content = """
package com.example;
import java.util.List;
import java.util.ArrayList;

public class UserList {
    public List<String> getUsers() {
        return new ArrayList<>();
    }
}
"""
    f = tmp_path / "UserList.java"
    f.write_text(content, encoding="utf-8")
    
    chunks = uni_chunker.chunk_file(f)
    
    chunk = chunks[0]
    assert "import java.util.List;" in chunk.content
    assert "package com.example;" in chunk.content
