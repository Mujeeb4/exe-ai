
import pytest
from pathlib import Path
from src.memory.chunker import UniversalChunker

@pytest.fixture
def chunker():
    return UniversalChunker()

def test_chunk_sql(chunker, tmp_path):
    content = """
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100)
);

CREATE VIEW active_users AS (
    SELECT * FROM users WHERE active = 1
);
"""
    f = tmp_path / "schema.sql"
    f.write_text(content, encoding="utf-8")
    
    chunks = chunker.chunk_file(f)
    assert len(chunks) == 2
    names = {c.name for c in chunks}
    assert "users" in names
    assert "active_users" in names
    assert chunks[0].chunk_type == "table"

def test_chunk_swift(chunker, tmp_path):
    content = """
class Person {
    var name: String
    
    init(name: String) {
        self.name = name
    }
}

struct Point {
    var x: Int
    var y: Int
}
"""
    f = tmp_path / "models.swift"
    f.write_text(content, encoding="utf-8")
    
    chunks = chunker.chunk_file(f)
    names = {c.name for c in chunks}
    assert "Person" in names
    assert "Point" in names

def test_chunk_kotlin(chunker, tmp_path):
    content = """
class User(val name: String) {
    fun greet() {
        println("Hello $name")
    }
}

fun main() {
    val user = User("Alice")
    user.greet()
}
"""
    f = tmp_path / "app.kt"
    f.write_text(content, encoding="utf-8")
    
    chunks = chunker.chunk_file(f)
    names = {c.name for c in chunks}
    assert "User" in names
    assert "main" in names
    
    types = {c.chunk_type for c in chunks}
    assert "class" in types
    assert "function" in types
