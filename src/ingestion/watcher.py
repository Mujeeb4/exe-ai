"""
Watchdog-based file watcher with loop prevention.
"""

from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
from ..memory.db import VectorDB
from ..memory.chunker import PythonChunker
from ..memory.embedder import Embedder


class CodeFileHandler(FileSystemEventHandler):
    """Handles file system events for code files."""
    
    def __init__(self, db: VectorDB, embedder: Embedder, editor):
        self.db = db
        self.embedder = embedder
        self.editor = editor
        self.chunker = PythonChunker()
    
    def on_modified(self, event):
        """Handle file modification events."""
        # Prevent loop: ignore events triggered by our own modifications
        if self.editor.is_modifying:
            return
        
        if event.is_directory:
            return
        
        # Handle both str and bytes from watchdog
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode('utf-8')
        file_path = Path(str(src_path))
        
        # Only process Python files
        if file_path.suffix != '.py':
            return
        
        # Re-index the modified file
        self._reindex_file(file_path)
    
    def _reindex_file(self, file_path: Path):
        """Re-index a single file."""
        try:
            chunks = self.chunker.chunk_file(file_path)
            
            if chunks:
                texts = [chunk.content for chunk in chunks]
                embeddings = self.embedder.embed(texts)
                
                # Update database
                self.db.update_file(str(file_path), chunks, embeddings)
                print(f"Re-indexed: {file_path}")
        
        except Exception as e:
            print(f"Error re-indexing {file_path}: {e}")


class FileWatcher:
    """Watches for file changes and updates the index."""
    
    def __init__(self, root_path: Path, db: VectorDB, embedder: Embedder, editor):
        self.observer = Observer()
        self.handler = CodeFileHandler(db, embedder, editor)
        self.root_path = root_path
    
    def start(self):
        """Start watching for file changes."""
        self.observer.schedule(self.handler, str(self.root_path), recursive=True)
        self.observer.start()
    
    def stop(self):
        """Stop watching for file changes."""
        self.observer.stop()
        self.observer.join()
