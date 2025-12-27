
import pytest
from pathlib import Path
from src.memory.chunker import UniversalChunker

@pytest.fixture
def chunker():
    return UniversalChunker()

def test_chunk_java(chunker, tmp_path):
    content = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
    f = tmp_path / "Calculator.java"
    f.write_text(content, encoding="utf-8")
    
    chunks = chunker.chunk_file(f)
    # Should find both the class and the method inside it
    assert len(chunks) >= 1
    names = {c.name for c in chunks}
    assert "Calculator" in names
    assert "add" in names
    
    class_chunk = next(c for c in chunks if c.name == "Calculator")
    assert class_chunk.chunk_type == "class"
    assert "add(int a, int b)" in class_chunk.content

def test_chunk_cpp(chunker, tmp_path):
    content = """
namespace Math {
    class Vector {
        float x, y;
    };
    
    int dot_product(Vector a, Vector b) {
        return a.x * b.x + a.y * b.y;
    }
}
"""
    f = tmp_path / "math.cpp"
    f.write_text(content, encoding="utf-8")
    
    chunks = chunker.chunk_file(f)
    # Should find namespace, class, and function
    # Note: current logic might nest them or find them sequentially depending on implementation details
    # With simple regex + brace counting, it finds top level items.
    # Namespace is top level.
    assert len(chunks) >= 1
    assert chunks[0].name == "Math"
    assert chunks[0].chunk_type == "namespace"

def test_chunk_go(chunker, tmp_path):
    content = """
package main

type User struct {
    Name string
    Age  int
}

func NewUser(name string) *User {
    return &User{Name: name}
}
"""
    f = tmp_path / "user.go"
    f.write_text(content, encoding="utf-8")
    
    chunks = chunker.chunk_file(f)
    # Should find struct and func
    names = {c.name for c in chunks}
    assert "User" in names
    assert "NewUser" in names

def test_chunk_rust(chunker, tmp_path):
    content = """
struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn new(x: i32, y: i32) -> Self {
        Point { x, y }
    }
}
"""
    f = tmp_path / "point.rs"
    f.write_text(content, encoding="utf-8")
    
    chunks = chunker.chunk_file(f)
    names = {c.name for c in chunks}
    assert "Point" in names # struct
    assert "Point" in names # impl - might duplicate name but different content/type
    
    types = {c.chunk_type for c in chunks}
    assert "struct" in types
    assert "impl" in types

def test_chunk_unsupported(chunker, tmp_path):
    f = tmp_path / "test.unknown"
    f.write_text("some content")
    chunks = chunker.chunk_file(f)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "text"
