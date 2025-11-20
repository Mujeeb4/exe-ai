
import pytest
from pathlib import Path
from src.memory.chunker import JSChunker

@pytest.fixture
def js_chunker():
    return JSChunker()

def test_chunk_simple_function(js_chunker, tmp_path):
    content = """
function add(a, b) {
    return a + b;
}
"""
    f = tmp_path / "test.js"
    f.write_text(content, encoding="utf-8")
    
    chunks = js_chunker.chunk_file(f)
    assert len(chunks) == 1
    assert chunks[0].name == "add"
    assert chunks[0].chunk_type == "function"
    assert "return a + b;" in chunks[0].content

def test_chunk_class(js_chunker, tmp_path):
    content = """
class Calculator {
    constructor() {
        this.value = 0;
    }
    
    add(n) {
        this.value += n;
    }
}
"""
    f = tmp_path / "test.ts"
    f.write_text(content, encoding="utf-8")
    
    chunks = js_chunker.chunk_file(f)
    assert len(chunks) == 1
    assert chunks[0].name == "Calculator"
    assert chunks[0].chunk_type == "class"
    assert "add(n)" in chunks[0].content

def test_chunk_arrow_function(js_chunker, tmp_path):
    content = """
const multiply = (a, b) => {
    return a * b;
}
"""
    f = tmp_path / "test.jsx"
    f.write_text(content, encoding="utf-8")
    
    chunks = js_chunker.chunk_file(f)
    assert len(chunks) == 1
    assert chunks[0].name == "multiply"
    assert chunks[0].chunk_type == "arrow_function"

def test_chunk_nested_braces(js_chunker, tmp_path):
    content = """
function complex(x) {
    if (x > 0) {
        return {
            val: x
        };
    }
    return null;
}
"""
    f = tmp_path / "test.tsx"
    f.write_text(content, encoding="utf-8")
    
    chunks = js_chunker.chunk_file(f)
    assert len(chunks) == 1
    assert chunks[0].name == "complex"
    assert "return null;" in chunks[0].content

def test_chunk_multiple_items(js_chunker, tmp_path):
    content = """
function one() {
    return 1;
}

class Two {
    val = 2;
}

const three = () => {
    return 3;
}
"""
    f = tmp_path / "multi.js"
    f.write_text(content, encoding="utf-8")
    
    chunks = js_chunker.chunk_file(f)
    assert len(chunks) == 3
    names = {c.name for c in chunks}
    assert names == {"one", "Two", "three"}
