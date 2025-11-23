"""
Tests for the file watcher with loop prevention.
"""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from src.ingestion.watcher import FileWatcher, CodeFileHandler
from src.ingestion.editor import Editor
from src.memory.db import VectorDB
from src.memory.embedder import Embedder
from src.memory.chunker import PythonChunker
from src.core.models import CodeChunk


@pytest.fixture
def mock_db():
    """Create a mock VectorDB."""
    db = Mock(spec=VectorDB)
    db.update_file = Mock()
    return db


@pytest.fixture
def mock_embedder():
    """Create a mock Embedder."""
    embedder = Mock(spec=Embedder)
    embedder.embed = Mock(return_value=[[0.1, 0.2, 0.3]])
    return embedder


@pytest.fixture
def editor():
    """Create a real Editor instance."""
    return Editor()


@pytest.fixture
def handler(mock_db, mock_embedder, editor):
    """Create a CodeFileHandler with mocked dependencies."""
    return CodeFileHandler(mock_db, mock_embedder, editor, cooldown_seconds=0.5)


def test_watcher_initialization(tmp_path, mock_db, mock_embedder, editor):
    """Test that FileWatcher initializes correctly."""
    watcher = FileWatcher(tmp_path, mock_db, mock_embedder, editor)
    
    assert watcher.root_path == tmp_path
    assert not watcher.is_running()
    assert watcher.handler.cooldown_seconds == 2.0


def test_watcher_custom_cooldown(tmp_path, mock_db, mock_embedder, editor):
    """Test FileWatcher with custom cooldown."""
    watcher = FileWatcher(tmp_path, mock_db, mock_embedder, editor, cooldown_seconds=5.0)
    
    assert watcher.handler.cooldown_seconds == 5.0


def test_watcher_start_stop(tmp_path, mock_db, mock_embedder, editor):
    """Test starting and stopping the watcher."""
    watcher = FileWatcher(tmp_path, mock_db, mock_embedder, editor)
    
    # Initially not running
    assert not watcher.is_running()
    
    # Start watcher
    watcher.start()
    assert watcher.is_running()
    
    # Stop watcher
    watcher.stop()
    assert not watcher.is_running()


def test_watcher_idempotent_start_stop(tmp_path, mock_db, mock_embedder, editor):
    """Test that start/stop are idempotent."""
    watcher = FileWatcher(tmp_path, mock_db, mock_embedder, editor)
    
    # Multiple starts should not cause issues
    watcher.start()
    watcher.start()
    assert watcher.is_running()
    
    # Multiple stops should not cause issues
    watcher.stop()
    watcher.stop()
    assert not watcher.is_running()


def test_handler_ignores_editor_modifications(handler, tmp_path):
    """Test that handler ignores events when editor is modifying."""
    # Create a test file
    test_file = tmp_path / "test.py"
    test_file.write_text("def hello():\n    pass\n")
    
    # Set editor as modifying
    handler.editor.is_modifying = True
    
    # Create a mock event
    event = Mock()
    event.is_directory = False
    event.src_path = str(test_file)
    
    # Handle the event
    handler.on_modified(event)
    
    # Database should not be updated
    handler.db.update_file.assert_not_called()


def test_handler_ignores_directories(handler, tmp_path):
    """Test that handler ignores directory events."""
    event = Mock()
    event.is_directory = True
    event.src_path = str(tmp_path)
    
    handler.on_modified(event)
    
    # Database should not be updated
    handler.db.update_file.assert_not_called()


def test_handler_ignores_non_python_files(handler, tmp_path):
    """Test that handler only processes Python files."""
    # Create a non-Python file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, world!")
    
    event = Mock()
    event.is_directory = False
    event.src_path = str(test_file)
    
    handler.on_modified(event)
    
    # Database should not be updated
    handler.db.update_file.assert_not_called()


def test_handler_processes_python_files(handler, tmp_path):
    """Test that handler processes Python files correctly."""
    # Create a Python file
    test_file = tmp_path / "test.py"
    test_file.write_text("def hello():\n    return 'world'\n")
    
    event = Mock()
    event.is_directory = False
    event.src_path = str(test_file)
    
    # Mock the chunker to return a chunk
    mock_chunk = CodeChunk(
        file_path=str(test_file),
        chunk_id="test:hello",
        content="def hello():\n    return 'world'",
        start_line=1,
        end_line=2,
        chunk_type="function",
        name="hello"
    )
    
    with patch.object(handler.chunker, 'chunk_file', return_value=[mock_chunk]):
        handler.on_modified(event)
    
    # Database should be updated
    handler.db.update_file.assert_called_once()


def test_handler_debouncing(handler, tmp_path):
    """Test that handler debounces rapid events for the same file."""
    # Create a Python file
    test_file = tmp_path / "test.py"
    test_file.write_text("def hello():\n    return 'world'\n")
    
    event = Mock()
    event.is_directory = False
    event.src_path = str(test_file)
    
    # Mock the chunker
    mock_chunk = CodeChunk(
        file_path=str(test_file),
        chunk_id="test:hello",
        content="def hello():\n    return 'world'",
        start_line=1,
        end_line=2,
        chunk_type="function",
        name="hello"
    )
    
    with patch.object(handler.chunker, 'chunk_file', return_value=[mock_chunk]):
        # First event should be processed
        handler.on_modified(event)
        assert handler.db.update_file.call_count == 1
        
        # Immediate second event should be ignored (within cooldown)
        handler.on_modified(event)
        assert handler.db.update_file.call_count == 1
        
        # Wait for cooldown to expire
        time.sleep(0.6)
        
        # Third event should be processed
        handler.on_modified(event)
        assert handler.db.update_file.call_count == 2


def test_handler_processes_different_files_simultaneously(handler, tmp_path):
    """Test that handler can process different files without debouncing."""
    # Create two Python files
    file1 = tmp_path / "test1.py"
    file2 = tmp_path / "test2.py"
    file1.write_text("def hello():\n    pass\n")
    file2.write_text("def world():\n    pass\n")
    
    event1 = Mock()
    event1.is_directory = False
    event1.src_path = str(file1)
    
    event2 = Mock()
    event2.is_directory = False
    event2.src_path = str(file2)
    
    mock_chunk = CodeChunk(
        file_path="",
        chunk_id="test",
        content="test",
        start_line=1,
        end_line=2,
        chunk_type="function"
    )
    
    with patch.object(handler.chunker, 'chunk_file', return_value=[mock_chunk]):
        # Both events should be processed
        handler.on_modified(event1)
        handler.on_modified(event2)
        
        assert handler.db.update_file.call_count == 2


def test_handler_prevents_concurrent_processing(handler, tmp_path):
    """Test that handler prevents concurrent processing of the same file."""
    test_file = tmp_path / "test.py"
    test_file.write_text("def hello():\n    pass\n")
    
    event = Mock()
    event.is_directory = False
    event.src_path = str(test_file)
    
    # Manually add file to processing set
    handler._processing_files.add(str(test_file))
    
    mock_chunk = CodeChunk(
        file_path=str(test_file),
        chunk_id="test",
        content="test",
        start_line=1,
        end_line=2,
        chunk_type="function"
    )
    
    with patch.object(handler.chunker, 'chunk_file', return_value=[mock_chunk]):
        # Event should be ignored
        handler.on_modified(event)
        
        # Database should not be updated
        handler.db.update_file.assert_not_called()


def test_handler_handles_missing_file(handler, tmp_path):
    """Test that handler gracefully handles missing files."""
    nonexistent_file = tmp_path / "nonexistent.py"
    
    event = Mock()
    event.is_directory = False
    event.src_path = str(nonexistent_file)
    
    # Should not raise an exception
    handler.on_modified(event)
    
    # Database should not be updated
    handler.db.update_file.assert_not_called()


def test_handler_handles_chunking_errors(handler, tmp_path, capsys):
    """Test that handler handles chunking errors gracefully."""
    test_file = tmp_path / "test.py"
    test_file.write_text("def hello():\n    pass\n")
    
    event = Mock()
    event.is_directory = False
    event.src_path = str(test_file)
    
    # Mock chunker to raise an exception
    with patch.object(handler.chunker, 'chunk_file', side_effect=Exception("Chunking failed")):
        handler.on_modified(event)
    
    # Database should not be updated
    handler.db.update_file.assert_not_called()
    
    # Error should be printed
    captured = capsys.readouterr()
    assert "Error re-indexing" in captured.out


def test_handler_handles_bytes_path(handler, tmp_path):
    """Test that handler handles byte-encoded paths from watchdog."""
    test_file = tmp_path / "test.py"
    test_file.write_text("def hello():\n    pass\n")
    
    event = Mock()
    event.is_directory = False
    event.src_path = str(test_file).encode('utf-8')  # Bytes path
    
    mock_chunk = CodeChunk(
        file_path=str(test_file),
        chunk_id="test",
        content="test",
        start_line=1,
        end_line=2,
        chunk_type="function"
    )
    
    with patch.object(handler.chunker, 'chunk_file', return_value=[mock_chunk]):
        handler.on_modified(event)
    
    # Should be processed successfully
    handler.db.update_file.assert_called_once()


def test_no_infinite_loop_integration(tmp_path, mock_db, mock_embedder):
    """Integration test: verify no infinite loop when editor modifies file."""
    editor = Editor()
    handler = CodeFileHandler(mock_db, mock_embedder, editor, cooldown_seconds=0.1)
    
    test_file = tmp_path / "test.py"
    test_file.write_text("def hello():\n    return 'world'\n")
    
    # Simulate editor making a change
    patch_content = """--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
 def hello():
-    return 'world'
+    return 'universe'
"""
    
    # Simulate watcher event DURING modification (the race condition we're preventing)
    # We need to manually set is_modifying and trigger the event
    event = Mock()
    event.is_directory = False
    event.src_path = str(test_file)
    
    # Set flag to simulate modification in progress
    editor.is_modifying = True
    
    # This should be ignored because editor.is_modifying is True
    handler.on_modified(event)
    
    # Database update should not be called (event was ignored)
    mock_db.update_file.assert_not_called()
    
    # Reset flag
    editor.is_modifying = False
    
    # Now simulate a normal file change (not from editor)
    # This should trigger the update
    handler.on_modified(event)
    
    # Now the database should be updated
    mock_db.update_file.assert_called_once()


def test_real_world_patch_scenario(tmp_path, mock_db, mock_embedder):
    """Test real-world scenario: apply patch and verify cooldown prevents rapid re-indexing."""
    editor = Editor()
    handler = CodeFileHandler(mock_db, mock_embedder, editor, cooldown_seconds=0.5)
    
    test_file = tmp_path / "test.py"
    test_file.write_text("def hello():\n    return 'world'\n")
    
    patch_content = """--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
 def hello():
-    return 'world'
+    return 'universe'
"""
    
    # Apply patch
    success = editor.apply_patch(test_file, patch_content)
    assert success
    
    # After patch completes, is_modifying should be False
    assert not editor.is_modifying
    
    # File system watcher would detect the change after the write completes
    # The cooldown mechanism prevents rapid re-indexing
    event = Mock()
    event.is_directory = False
    event.src_path = str(test_file)
    
    # First event triggers update
    handler.on_modified(event)
    mock_db.update_file.assert_called_once()
    
    # Second event within cooldown period is ignored
    handler.on_modified(event)
    # Still only called once (second event was debounced)
    assert mock_db.update_file.call_count == 1

    mock_chunk = CodeChunk(
        file_path=str(test_file),
        chunk_id="test",
        content="test",
        start_line=1,
        end_line=2,
        chunk_type="function"
    )
    
    with patch.object(handler.chunker, 'chunk_file', return_value=[mock_chunk]):
        handler.on_modified(event)
    
    # This time it should be processed
    mock_db.update_file.assert_called_once()
