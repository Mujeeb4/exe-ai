"""
AST-based code chunker for Python files.
Preserves function and class boundaries with support for nested structures.
"""

import ast
from pathlib import Path
from typing import List, Optional, Set
from ..core.models import CodeChunk


class PythonChunker:
    """Chunks Python code using AST parsing with support for nested structures."""
    
    def __init__(self, max_chunk_size: int = 1000):
        """
        Initialize the chunker.
        
        Args:
            max_chunk_size: Maximum number of lines per chunk (for very large functions/classes)
        """
        self.max_chunk_size = max_chunk_size
        self._processed_nodes: Set[int] = set()
    
    def chunk_file(self, file_path: Path) -> List[CodeChunk]:
        """
        Parse a Python file and extract chunks.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of CodeChunk objects
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            chunks = []
            lines = source.splitlines()
            
            # Reset processed nodes for this file
            self._processed_nodes.clear()
            
            # Module-level docstring
            docstring = ast.get_docstring(tree)
            if docstring:
                chunks.append(CodeChunk(
                    file_path=str(file_path),
                    chunk_id=f"{file_path}:module_docstring",
                    content=docstring,
                    start_line=1,
                    end_line=self._count_docstring_lines(docstring),
                    chunk_type="module",
                    name=f"{file_path.stem}_docstring"
                ))
            
            # Extract top-level elements (classes and functions)
            # Use iter_child_nodes instead of walk to control traversal
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunk = self._extract_function_chunk(node, lines, file_path, level=0)
                    if chunk:
                        chunks.append(chunk)
                elif isinstance(node, ast.ClassDef):
                    # Extract class chunk and its methods
                    chunks.extend(self._extract_class_chunks(node, lines, file_path, level=0))
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    # Track imports for context (optional enhancement)
                    pass
            
            return chunks
        
        except SyntaxError as e:
            # If file has syntax errors, treat as plain text
            return self._fallback_chunk(file_path, error=str(e))
        except Exception as e:
            # Catch other parsing errors
            return self._fallback_chunk(file_path, error=str(e))
    
    def _extract_function_chunk(
        self, 
        node: ast.FunctionDef | ast.AsyncFunctionDef, 
        lines: List[str], 
        file_path: Path,
        level: int = 0,
        parent_name: Optional[str] = None
    ) -> Optional[CodeChunk]:
        """
        Extract a chunk from a function node.
        
        Args:
            node: AST function node
            lines: Source code lines
            file_path: Path to the file
            level: Nesting level (0 for top-level)
            parent_name: Name of parent class/function if nested
        """
        if id(node) in self._processed_nodes:
            return None
        
        self._processed_nodes.add(id(node))
        
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        # Include decorators
        if node.decorator_list:
            first_decorator = node.decorator_list[0]
            start_line = min(start_line, first_decorator.lineno)
        
        content = "\n".join(lines[start_line - 1:end_line])
        
        # Build full qualified name
        if parent_name:
            full_name = f"{parent_name}.{node.name}"
        else:
            full_name = node.name
        
        chunk_type = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
        if level > 0:
            chunk_type = f"nested_{chunk_type}"
        
        # Extract function signature for better context
        args_str = self._extract_function_signature(node)
        
        return CodeChunk(
            file_path=str(file_path),
            chunk_id=f"{file_path}:{full_name}:{start_line}",
            content=f"# {args_str}\n{content}",
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            name=full_name
        )
    
    def _extract_class_chunks(
        self, 
        node: ast.ClassDef, 
        lines: List[str], 
        file_path: Path,
        level: int = 0,
        parent_name: Optional[str] = None
    ) -> List[CodeChunk]:
        """
        Extract chunks from a class and its methods.
        
        Args:
            node: AST class node
            lines: Source code lines
            file_path: Path to the file
            level: Nesting level
            parent_name: Parent class name if nested
        """
        if id(node) in self._processed_nodes:
            return []
        
        self._processed_nodes.add(id(node))
        chunks = []
        
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        # Include decorators
        if node.decorator_list:
            first_decorator = node.decorator_list[0]
            start_line = min(start_line, first_decorator.lineno)
        
        # Build full qualified name
        if parent_name:
            full_name = f"{parent_name}.{node.name}"
        else:
            full_name = node.name
        
        # Extract class-level chunk (class definition + docstring)
        class_header_end = start_line
        docstring = ast.get_docstring(node)
        
        if docstring:
            # Find where docstring ends
            for i, child in enumerate(node.body):
                if isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant):
                    class_header_end = child.end_lineno or start_line
                    break
        
        # Create chunk for class header and docstring
        class_intro = "\n".join(lines[start_line - 1:class_header_end])
        
        # Extract base classes
        bases_str = self._extract_class_bases(node)
        
        chunk_type = "class" if level == 0 else "nested_class"
        
        chunks.append(CodeChunk(
            file_path=str(file_path),
            chunk_id=f"{file_path}:{full_name}:class_def:{start_line}",
            content=f"# class {node.name}({bases_str})\n{class_intro}",
            start_line=start_line,
            end_line=class_header_end,
            chunk_type=chunk_type,
            name=full_name
        ))
        
        # Extract methods and nested classes
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_chunk = self._extract_function_chunk(
                    child, 
                    lines, 
                    file_path, 
                    level=level + 1,
                    parent_name=full_name
                )
                if method_chunk:
                    # Mark as method instead of function
                    method_chunk.chunk_type = "method"
                    if isinstance(child, ast.AsyncFunctionDef):
                        method_chunk.chunk_type = "async_method"
                    chunks.append(method_chunk)
            
            elif isinstance(child, ast.ClassDef):
                # Handle nested classes
                nested_chunks = self._extract_class_chunks(
                    child, 
                    lines, 
                    file_path, 
                    level=level + 1,
                    parent_name=full_name
                )
                chunks.extend(nested_chunks)
        
        return chunks
    
    def _extract_function_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Extract function signature as a string."""
        args = []
        
        # Regular arguments
        for arg in node.args.args:
            args.append(arg.arg)
        
        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        
        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        return f"{async_prefix}def {node.name}({', '.join(args)})"
    
    def _extract_class_bases(self, node: ast.ClassDef) -> str:
        """Extract base classes as a string."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                # Handle cases like parent.BaseClass
                bases.append(ast.unparse(base))
        
        return ", ".join(bases) if bases else ""
    
    def _count_docstring_lines(self, docstring: str) -> int:
        """Count lines in a docstring."""
        return len(docstring.splitlines()) + 2  # +2 for triple quotes
    
    def _fallback_chunk(self, file_path: Path, error: Optional[str] = None) -> List[CodeChunk]:
        """Fallback for files that can't be parsed."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunk_name = f"{file_path.name}_unparseable"
            if error:
                content = f"# Parsing error: {error}\n\n{content}"
            
            return [CodeChunk(
                file_path=str(file_path),
                chunk_id=f"{file_path}:full",
                content=content,
                start_line=1,
                end_line=len(content.splitlines()),
                chunk_type="text",
                name=chunk_name
            )]
        except Exception:
            return []

