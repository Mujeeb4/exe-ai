"""
AST-based code chunker for Python files.
Preserves function and class boundaries with support for nested structures.
"""

import ast
import re
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
            
            # Extract imports for context
            imports = []
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.unparse(node))
            import_context = "\n".join(imports)
            
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
                    chunk = self._extract_function_chunk(node, lines, file_path, level=0, context=import_context)
                    if chunk:
                        chunks.append(chunk)
                elif isinstance(node, ast.ClassDef):
                    # Extract class chunk and its methods
                    chunks.extend(self._extract_class_chunks(node, lines, file_path, level=0, context=import_context))
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    # Already handled
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
        parent_name: Optional[str] = None,
        context: str = ""
    ) -> Optional[CodeChunk]:
        """
        Extract a chunk from a function node.
        
        Args:
            node: AST function node
            lines: Source code lines
            file_path: Path to the file
            level: Nesting level (0 for top-level)
            parent_name: Name of parent class/function if nested
            context: Additional context to prepend (imports, class header)
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
        
        # Prepend context if available
        if context:
            content = f"{context}\n\n{content}"
        
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
        parent_name: Optional[str] = None,
        context: str = ""
    ) -> List[CodeChunk]:
        """
        Extract chunks from a class and its methods.
        
        Args:
            node: AST class node
            lines: Source code lines
            file_path: Path to the file
            level: Nesting level
            parent_name: Parent class name if nested
            context: Additional context to prepend (imports)
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
        if context:
            class_intro = f"{context}\n\n{class_intro}"
            
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
        
        # Prepare class context for methods (imports + class definition)
        class_def_line = f"class {node.name}({bases_str}):"
        method_context = f"{context}\n\n{class_def_line}" if context else class_def_line
        
        # Extract methods and nested classes
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_chunk = self._extract_function_chunk(
                    child, 
                    lines, 
                    file_path, 
                    level=level + 1,
                    parent_name=full_name,
                    context=method_context
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
                    parent_name=full_name,
                    context=method_context
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


class UniversalChunker:
    """Chunks code for multiple languages using regex and brace counting."""
    
    SUPPORTED_LANGUAGES = {
        # JavaScript / TypeScript
        '.js': 'javascript', '.jsx': 'javascript', '.ts': 'javascript', '.tsx': 'javascript',
        # Java
        '.java': 'java',
        # C#
        '.cs': 'c_sharp',
        # C++
        '.cpp': 'cpp', '.hpp': 'cpp', '.c': 'cpp', '.h': 'cpp', '.cc': 'cpp',
        # Go
        '.go': 'go',
        # Rust
        '.rs': 'rust',
        # PHP
        '.php': 'php',
        # Swift
        '.swift': 'swift',
        # Kotlin
        '.kt': 'kotlin', '.kts': 'kotlin',
        # SQL
        '.sql': 'sql',
    }

    PATTERNS = {
        'javascript': [
            (r'function\s+(\w+)\s*\(', 'function'),
            (r'class\s+(\w+)', 'class'),
            (r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[^=]+)\s*=>\s*\{', 'arrow_function'),
            (r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?function\s*\(', 'expression_function'),
            (r'interface\s+(\w+)', 'interface'),
            (r'type\s+(\w+)\s*=\s*\{', 'type'),
        ],
        'java': [
            (r'class\s+(\w+)', 'class'),
            (r'interface\s+(\w+)', 'interface'),
            (r'enum\s+(\w+)', 'enum'),
            (r'(?:public|protected|private|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *\{', 'method'),
        ],
        'c_sharp': [
            (r'class\s+(\w+)', 'class'),
            (r'interface\s+(\w+)', 'interface'),
            (r'enum\s+(\w+)', 'enum'),
            (r'namespace\s+(\w+)', 'namespace'),
            (r'(?:public|protected|private|static|internal|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *\{', 'method'),
        ],
        'cpp': [
            (r'class\s+(\w+)', 'class'),
            (r'struct\s+(\w+)', 'struct'),
            (r'namespace\s+(\w+)', 'namespace'),
            (r'\w+\s+(\w+)\s*\([^\)]*\)\s*\{', 'function'),
        ],
        'go': [
            (r'func\s+(\w+)\s*\(', 'function'),
            (r'type\s+(\w+)\s+struct', 'struct'),
            (r'type\s+(\w+)\s+interface', 'interface'),
        ],
        'rust': [
            (r'fn\s+(\w+)\s*\(', 'function'),
            (r'struct\s+(\w+)', 'struct'),
            (r'enum\s+(\w+)', 'enum'),
            (r'trait\s+(\w+)', 'trait'),
            (r'impl\s+(\w+)', 'impl'),
            (r'mod\s+(\w+)', 'module'),
        ],
        'php': [
            (r'function\s+(\w+)\s*\(', 'function'),
            (r'class\s+(\w+)', 'class'),
            (r'trait\s+(\w+)', 'trait'),
            (r'interface\s+(\w+)', 'interface'),
        ],
        'swift': [
            (r'class\s+(\w+)', 'class'),
            (r'struct\s+(\w+)', 'struct'),
            (r'enum\s+(\w+)', 'enum'),
            (r'protocol\s+(\w+)', 'protocol'),
            (r'extension\s+(\w+)', 'extension'),
            (r'func\s+(\w+)\s*\(', 'function'),
        ],
        'kotlin': [
            (r'class\s+(\w+)', 'class'),
            (r'interface\s+(\w+)', 'interface'),
            (r'fun\s+(\w+)\s*\(', 'function'),
            (r'object\s+(\w+)', 'object'),
        ],
        'sql': [
            (r'CREATE\s+TABLE\s+(\w+)', 'table'),
            (r'CREATE\s+VIEW\s+(\w+)', 'view'),
            (r'CREATE\s+PROCEDURE\s+(\w+)', 'procedure'),
            (r'CREATE\s+FUNCTION\s+(\w+)', 'function'),
        ],
    }
    
    IMPORT_PATTERNS = {
        'javascript': [r'import\s+.*?;', r'require\(.*?\);'],
        'java': [r'import\s+.*?;', r'package\s+.*?;'],
        'c_sharp': [r'using\s+.*?;', r'namespace\s+.*?;'],
        'cpp': [r'#include\s+.*', r'using\s+namespace\s+.*;'],
        'go': [r'import\s+\(.*\)', r'import\s+.*'],
        'rust': [r'use\s+.*;', r'mod\s+.*;', r'extern\s+crate\s+.*;'],
        'php': [r'use\s+.*;', r'require\s+.*;', r'include\s+.*;'],
        'swift': [r'import\s+.*'],
        'kotlin': [r'import\s+.*', r'package\s+.*'],
    }
    
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
        
    def chunk_file(self, file_path: Path) -> List[CodeChunk]:
        """
        Parse a file and extract chunks based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of CodeChunk objects
        """
        ext = file_path.suffix.lower()
        lang = self.SUPPORTED_LANGUAGES.get(ext)
        
        if not lang:
            return self._fallback_chunk(file_path, error=f"Unsupported extension: {ext}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = []
            lines = content.splitlines()
            patterns = self.PATTERNS.get(lang, [])
            
            # Extract imports
            import_context = self._extract_imports(content, lang)
            
            # Determine delimiters based on language
            start_char = '{'
            end_char = '}'
            if lang == 'sql':
                start_char = '('
                end_char = ')'
            
            for i, line in enumerate(lines):
                # Skip comment lines for start detection
                if line.strip().startswith('//') or line.strip().startswith('/*') or line.strip().startswith('#') or line.strip().startswith('--'):
                    continue
                    
                for pattern, type_name in patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        name = match.group(1)
                        # Check if this line has an opening delimiter
                        if start_char in line:
                            end_line = self._find_closing_brace(lines, i, start_char, end_char)
                            if end_line:
                                chunk_content = "\n".join(lines[i:end_line])
                                if import_context:
                                    chunk_content = f"{import_context}\n\n{chunk_content}"
                                    
                                chunks.append(CodeChunk(
                                    file_path=str(file_path),
                                    chunk_id=f"{file_path}:{name}:{i+1}",
                                    content=chunk_content,
                                    start_line=i + 1,
                                    end_line=end_line,
                                    chunk_type=type_name,
                                    name=name
                                ))
            
            if not chunks:
                 # Fallback to whole file if no structure found
                return [CodeChunk(
                    file_path=str(file_path),
                    chunk_id=f"{file_path}:full",
                    content=content,
                    start_line=1,
                    end_line=len(lines),
                    chunk_type="module",
                    name=file_path.stem
                )]
                
            return chunks
            
        except Exception as e:
            # Fallback
            return self._fallback_chunk(file_path, error=str(e))

    def _extract_imports(self, content: str, lang: str) -> str:
        """Extract import statements from code content."""
        patterns = self.IMPORT_PATTERNS.get(lang, [])
        if not patterns:
            return ""
            
        imports = []
        for line in content.splitlines():
            for pattern in patterns:
                if re.match(pattern, line.strip()):
                    imports.append(line.strip())
                    break
        
        return "\n".join(imports)

    def _find_closing_brace(self, lines: List[str], start_line_idx: int, start_char: str = '{', end_char: str = '}') -> Optional[int]:
        """Find the line number of the matching closing brace, ignoring strings and comments."""
        brace_count = 0
        found_start = False
        
        in_string = False
        string_char = ''
        in_comment = False # Single line comment
        in_block_comment = False
        
        for i in range(start_line_idx, len(lines)):
            line = lines[i]
            
            # Reset single line comment state
            in_comment = False
            
            for j, char in enumerate(line):
                # Handle comments
                if not in_string and not in_block_comment:
                    if char == '/' and j + 1 < len(line):
                        if line[j+1] == '/':
                            in_comment = True
                            break # Ignore rest of line
                        elif line[j+1] == '*':
                            in_block_comment = True
                
                if in_block_comment:
                    if char == '*' and j + 1 < len(line) and line[j+1] == '/':
                        in_block_comment = False
                    continue
                    
                # Handle strings
                if not in_comment:
                    if char in ('"', "'", '`'):
                        if not in_string:
                            in_string = True
                            string_char = char
                        elif char == string_char:
                            # Check for escape
                            if j > 0 and line[j-1] == '\\':
                                pass
                            else:
                                in_string = False
                
                if in_string or in_comment:
                    continue
                
                # Count braces
                if char == start_char:
                    brace_count += 1
                    found_start = True
                elif char == end_char:
                    brace_count -= 1
            
            if brace_count > 0:
                found_start = True
            
            if found_start and brace_count == 0:
                return i + 1 # Return 1-based line number (inclusive)
                
        return None

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

# Alias for backward compatibility if needed, though we'll update usages
JSChunker = UniversalChunker