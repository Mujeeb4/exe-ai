"""
Directory walker for codebase indexing.
"""

from pathlib import Path
from typing import List, Optional
import pathspec
from ..memory.db import VectorDB
from ..memory.chunker import PythonChunker, UniversalChunker
from ..memory.embedder import Embedder
from ..memory.repo_context import RepositoryContext


class Scanner:
    """Scans directories and indexes code files."""
    
    def __init__(self):
        self.py_chunker = PythonChunker()
        self.universal_chunker = UniversalChunker()
        self.ignored_patterns = self._load_gitignore()
    
    def _load_gitignore(self) -> pathspec.PathSpec:
        """Load .gitignore patterns."""
        gitignore_path = Path.cwd() / ".gitignore"
        
        default_patterns = [
            ".git/",
            ".exe/",
            "__pycache__/",
            "*.pyc",
            ".venv/",
            "venv/",
            "node_modules/",
            ".env"
        ]
        
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                patterns = default_patterns + f.read().splitlines()
        else:
            patterns = default_patterns
        
        return pathspec.PathSpec.from_lines('gitwildmatch', patterns)
    
    def scan_and_index(self, root_path: Path, db: VectorDB, embedder: Embedder, 
                      build_repo_context: bool = True) -> Optional[str]:
        """
        Scan a directory and index all code files.
        
        Args:
            root_path: Root directory to scan
            db: Vector database instance
            embedder: Embedder instance
            build_repo_context: Whether to build repository context
            
        Returns:
            Repository context string if build_repo_context=True
        """
        code_files = self._find_files(root_path)
        
        for file_path in code_files:
            if file_path.suffix == '.py':
                chunks = self.py_chunker.chunk_file(file_path)
            elif file_path.suffix in self.universal_chunker.SUPPORTED_LANGUAGES:
                chunks = self.universal_chunker.chunk_file(file_path)
            else:
                continue
            
            if chunks:
                # Generate embeddings
                texts = [chunk.content for chunk in chunks]
                embeddings = embedder.embed(texts)
                
                # Store in database
                db.add_chunks(chunks, embeddings)
        
        # Build repository context
        if build_repo_context:
            repo_context = RepositoryContext()
            all_chunks = db.get_all_chunks()
            context_str = repo_context.build_context(root_path, all_chunks)
            return context_str
        
        return None
    
    def _find_files(self, root_path: Path) -> List[Path]:
        """Find all supported code files in a directory."""
        code_files = []
        # Get all supported extensions
        extensions = ['*.py'] + [f"*{ext}" for ext in self.universal_chunker.SUPPORTED_LANGUAGES.keys()]
        
        for ext in extensions:
            for file_path in root_path.rglob(ext):
                # Check if file should be ignored
                try:
                    relative_path = file_path.relative_to(root_path)
                    if not self.ignored_patterns.match_file(str(relative_path)):
                        code_files.append(file_path)
                except ValueError:
                    continue
        
        return code_files
