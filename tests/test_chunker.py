"""
Tests for the Python code chunker (Day 2).
Verifies AST parsing works on nested classes/functions and complex structures.
"""

import pytest
from pathlib import Path
from src.memory.chunker import PythonChunker


def test_chunker_extracts_functions(tmp_path):
    """Test that chunker extracts function definitions."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
def hello():
    return "world"

def goodbye():
    return "farewell"
""")
    
    chunker = PythonChunker()
    chunks = chunker.chunk_file(test_file)
    
    # Should find 2 functions
    function_chunks = [c for c in chunks if "function" in c.chunk_type]
    assert len(function_chunks) == 2
    
    names = [c.name for c in function_chunks]
    assert "hello" in names
    assert "goodbye" in names


def test_chunker_extracts_classes(tmp_path):
    """Test that chunker extracts class definitions."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
class Person:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, {self.name}"
""")
    
    chunker = PythonChunker()
    chunks = chunker.chunk_file(test_file)
    
    # Should find 1 class definition + 2 methods
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) == 1
    assert class_chunks[0].name == "Person"
    
    # Methods should be extracted separately
    method_chunks = [c for c in chunks if c.chunk_type == "method"]
    assert len(method_chunks) == 2
    
    method_names = [c.name for c in method_chunks]
    assert "Person.__init__" in method_names
    assert "Person.greet" in method_names


def test_chunker_handles_nested_classes(tmp_path):
    """Test that chunker handles nested class definitions."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
class Outer:
    class Inner:
        def inner_method(self):
            pass
    
    def outer_method(self):
        pass
""")
    
    chunker = PythonChunker()
    chunks = chunker.chunk_file(test_file)
    
    # Should find Outer class, Inner class (nested), and methods
    chunk_names = [c.name for c in chunks]
    
    assert "Outer" in chunk_names
    assert "Outer.Inner" in chunk_names
    assert "Outer.outer_method" in chunk_names
    assert "Outer.Inner.inner_method" in chunk_names
    
    # Check nested class is marked as nested
    nested_classes = [c for c in chunks if c.chunk_type == "nested_class"]
    assert len(nested_classes) >= 1


def test_chunker_handles_nested_functions(tmp_path):
    """Test that chunker handles nested function definitions."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
def outer():
    def inner():
        return "nested"
    return inner()
""")
    
    chunker = PythonChunker()
    chunks = chunker.chunk_file(test_file)
    
    # Should find outer function and nested inner function
    function_chunks = [c for c in chunks if "function" in c.chunk_type]
    assert len(function_chunks) >= 1
    
    # Check that nested function is detected
    # Note: nested functions appear in the outer function's content
    outer_chunk = next((c for c in chunks if c.name == "outer"), None)
    assert outer_chunk is not None
    assert "def inner():" in outer_chunk.content


def test_chunker_handles_async_functions(tmp_path):
    """Test that chunker handles async function definitions."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
async def fetch_data():
    return await get_data()

class AsyncHandler:
    async def process(self):
        await self.do_work()
""")
    
    chunker = PythonChunker()
    chunks = chunker.chunk_file(test_file)
    
    # Should find async function
    async_funcs = [c for c in chunks if "async" in c.chunk_type.lower()]
    assert len(async_funcs) >= 1
    
    # Check async function is properly identified
    fetch_chunk = next((c for c in chunks if c.name == "fetch_data"), None)
    assert fetch_chunk is not None
    assert "async" in fetch_chunk.chunk_type or "async" in fetch_chunk.content.lower()


def test_chunker_handles_decorators(tmp_path):
    """Test that chunker includes decorators in chunks."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
@property
def my_property(self):
    return self._value

@staticmethod
def static_func():
    pass

@classmethod
def class_func(cls):
    pass
""")
    
    chunker = PythonChunker()
    chunks = chunker.chunk_file(test_file)
    
    # Decorators should be included in the chunk content
    property_chunk = next((c for c in chunks if "my_property" in c.name), None)
    assert property_chunk is not None
    assert "@property" in property_chunk.content


def test_chunker_extracts_class_with_base_classes(tmp_path):
    """Test that chunker handles class inheritance."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
class Base:
    pass

class Derived(Base):
    pass

class Multiple(Base, object):
    pass
""")
    
    chunker = PythonChunker()
    chunks = chunker.chunk_file(test_file)
    
    # Should extract all classes
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) == 3
    
    # Check that base classes are noted in content
    derived_chunk = next((c for c in chunks if c.name == "Derived"), None)
    assert derived_chunk is not None
    # Base class should be mentioned in the signature comment
    assert "Base" in derived_chunk.content


def test_chunker_handles_complex_nested_structure(tmp_path):
    """Test chunker on complex nested structure with multiple levels."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
class OuterClass:
    '''Outer class docstring'''
    
    class MiddleClass:
        '''Middle class docstring'''
        
        class InnerClass:
            '''Inner class docstring'''
            
            def deepest_method(self):
                return "deep"
        
        def middle_method(self):
            return "middle"
    
    def outer_method(self):
        return "outer"
""")
    
    chunker = PythonChunker()
    chunks = chunker.chunk_file(test_file)
    
    # Should handle three levels of nesting
    chunk_names = [c.name for c in chunks]
    
    assert "OuterClass" in chunk_names
    assert "OuterClass.MiddleClass" in chunk_names
    assert "OuterClass.MiddleClass.InnerClass" in chunk_names
    assert "OuterClass.MiddleClass.InnerClass.deepest_method" in chunk_names


def test_chunker_handles_syntax_errors(tmp_path):
    """Test that chunker handles files with syntax errors gracefully."""
    test_file = tmp_path / "broken.py"
    test_file.write_text("def incomplete(")
    
    chunker = PythonChunker()
    chunks = chunker.chunk_file(test_file)
    
    # Should fall back to text chunking
    assert len(chunks) >= 1
    assert chunks[0].chunk_type == "text"
    # Should include error information
    assert "error" in chunks[0].content.lower() or "unparseable" in chunks[0].name.lower()


def test_chunker_preserves_function_signatures(tmp_path):
    """Test that chunker preserves function signatures with various argument types."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
def func1(a, b, c):
    pass

def func2(a, b=10, *args, **kwargs):
    pass

def func3(x: int, y: str = "default") -> bool:
    return True
""")
    
    chunker = PythonChunker()
    chunks = chunker.chunk_file(test_file)
    
    # Check that function signatures are preserved in content
    func2_chunk = next((c for c in chunks if c.name == "func2"), None)
    assert func2_chunk is not None
    assert "args" in func2_chunk.content or "*args" in func2_chunk.content
    assert "kwargs" in func2_chunk.content or "**kwargs" in func2_chunk.content


def test_chunker_extracts_module_docstring(tmp_path):
    """Test that chunker extracts module-level docstrings."""
    test_file = tmp_path / "test.py"
    test_file.write_text('''
"""
This is a module docstring.
It describes the module.
"""

def some_function():
    pass
''')
    
    chunker = PythonChunker()
    chunks = chunker.chunk_file(test_file)
    
    # Should have module docstring chunk
    module_chunks = [c for c in chunks if c.chunk_type == "module"]
    assert len(module_chunks) == 1
    assert "module docstring" in module_chunks[0].content.lower()


def test_chunker_no_duplicate_chunks(tmp_path):
    """Test that chunker doesn't create duplicate chunks."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
class TestClass:
    def method1(self):
        pass
    
    def method2(self):
        pass
""")
    
    chunker = PythonChunker()
    chunks = chunker.chunk_file(test_file)
    
    # Check for unique chunk IDs
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Found duplicate chunk IDs"

