"""
Directory walker for codebase indexing.
"""

from pathlib import Path
from typing import List
import pathspec
from ..memory.db import VectorDB
from ..memory.chunker import PythonChunker
from ..memory.embedder import Embedder


class Scanner:
    """Scans directories and indexes code files."""
    
    def __init__(self):
        self.chunker = PythonChunker()
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
    
    def scan_and_index(self, root_path: Path, db: VectorDB, embedder: Embedder):
        """
        Scan a directory and index all Python files.
        
        Args:
            root_path: Root directory to scan
            db: Vector database instance
            embedder: Embedder instance
        """
        python_files = self._find_python_files(root_path)
        
        for file_path in python_files:
            chunks = self.chunker.chunk_file(file_path)
            
            if chunks:
                # Generate embeddings
                texts = [chunk.content for chunk in chunks]
                embeddings = embedder.embed(texts)
                
                # Store in database
                db.add_chunks(chunks, embeddings)
    
    def _find_python_files(self, root_path: Path) -> List[Path]:
        """Find all Python files in a directory."""
        python_files = []
        
        for file_path in root_path.rglob("*.py"):
            # Check if file should be ignored
            relative_path = file_path.relative_to(root_path)
            
            if not self.ignored_patterns.match_file(str(relative_path)):
                python_files.append(file_path)
        
        return python_files
