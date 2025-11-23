"""
Watchdog-based file watcher with loop prevention.
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
from ..memory.db import VectorDB
from ..memory.chunker import PythonChunker
from ..memory.embedder import Embedder


class CodeFileHandler(FileSystemEventHandler):
    """Handles file system events for code files with debouncing and loop prevention."""
    
    def __init__(self, db: VectorDB, embedder: Embedder, editor, cooldown_seconds: float = 2.0):
        self.db = db
        self.embedder = embedder
        self.editor = editor
        self.chunker = PythonChunker()
        self.cooldown_seconds = cooldown_seconds
        self._last_modified: Dict[str, datetime] = {}
        self._processing_files: set = set()
    
    def on_modified(self, event):
        """Handle file modification events with debouncing and loop prevention."""
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
        
        # Check if file is already being processed
        file_key = str(file_path)
        if file_key in self._processing_files:
            return
        
        # Debouncing: ignore rapid successive events for the same file
        now = datetime.now()
        if file_key in self._last_modified:
            time_since_last = (now - self._last_modified[file_key]).total_seconds()
            if time_since_last < self.cooldown_seconds:
                return
        
        # Update last modified time
        self._last_modified[file_key] = now
        
        # Re-index the modified file
        self._reindex_file(file_path)
    
    def _reindex_file(self, file_path: Path):
        """Re-index a single file with proper error handling and processing lock."""
        file_key = str(file_path)
        
        # Mark file as being processed
        self._processing_files.add(file_key)
        
        try:
            # Verify file exists and is readable
            if not file_path.exists():
                return
            
            chunks = self.chunker.chunk_file(file_path)
            
            if chunks:
                texts = [chunk.content for chunk in chunks]
                embeddings = self.embedder.embed(texts)
                
                # Update database
                self.db.update_file(str(file_path), chunks, embeddings)
                print(f"✓ Re-indexed: {file_path}")
            else:
                print(f"⚠ No chunks found in: {file_path}")
        
        except FileNotFoundError:
            print(f"⚠ File not found (may have been deleted): {file_path}")
        except Exception as e:
            print(f"✗ Error re-indexing {file_path}: {e}")
        finally:
            # Always remove from processing set
            self._processing_files.discard(file_key)


class FileWatcher:
    """Watches for file changes and updates the index."""
    
    def __init__(self, root_path: Path, db: VectorDB, embedder: Embedder, editor, cooldown_seconds: float = 2.0):
        self.observer = Observer()
        self.handler = CodeFileHandler(db, embedder, editor, cooldown_seconds)
        self.root_path = root_path
        self._is_running = False
    
    def start(self):
        """Start watching for file changes."""
        if self._is_running:
            return
        
        self.observer.schedule(self.handler, str(self.root_path), recursive=True)
        self.observer.start()
        self._is_running = True
    
    def stop(self):
        """Stop watching for file changes."""
        if not self._is_running:
            return
        
        self.observer.stop()
        self.observer.join()
        self._is_running = False
    
    def is_running(self) -> bool:
        """Check if the watcher is currently running."""
        return self._is_running
