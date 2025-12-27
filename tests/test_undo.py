"""
Tests for undo functionality in the REPL.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from src.interface.repl import PatchBackup, ReplSession
from src.config import ExeConfig


def test_backup_creation():
    """Test that backups are created successfully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backup_dir = Path(tmpdir) / "backups"
        backup_manager = PatchBackup(backup_dir)
        
        # Create a test file
        test_file = Path(tmpdir) / "test.py"
        test_file.write_text("print('hello')\n")
        
        # Create backup
        backup_id = backup_manager.create_backup(test_file)
        
        assert backup_id is not None
        assert len(backup_manager.backup_history) == 1
        assert backup_manager.backup_history[0]["original_path"] == str(test_file)
        
        # Check backup file exists
        backup_path = Path(backup_manager.backup_history[0]["backup_path"])
        assert backup_path.exists()
        assert backup_path.read_text() == "print('hello')\n"


def test_backup_restore():
    """Test that backups can be restored."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backup_dir = Path(tmpdir) / "backups"
        backup_manager = PatchBackup(backup_dir)
        
        # Create a test file
        test_file = Path(tmpdir) / "test.py"
        test_file.write_text("print('hello')\n")
        
        # Create backup
        backup_id = backup_manager.create_backup(test_file)
        
        # Modify original file
        test_file.write_text("print('world')\n")
        assert test_file.read_text() == "print('world')\n"
        
        # Restore backup
        success = backup_manager.restore_backup(backup_id)
        
        assert success
        assert test_file.read_text() == "print('hello')\n"


def test_backup_nonexistent_file():
    """Test backup creation for nonexistent file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backup_dir = Path(tmpdir) / "backups"
        backup_manager = PatchBackup(backup_dir)
        
        # Try to backup nonexistent file
        nonexistent = Path(tmpdir) / "nonexistent.py"
        backup_id = backup_manager.create_backup(nonexistent)
        
        assert backup_id is None
        assert len(backup_manager.backup_history) == 0


def test_backup_cleanup():
    """Test that old backups are cleaned up."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backup_dir = Path(tmpdir) / "backups"
        backup_manager = PatchBackup(backup_dir)
        
        # Create multiple test files and backups
        for i in range(15):
            test_file = Path(tmpdir) / f"test{i}.py"
            test_file.write_text(f"print({i})\n")
            backup_manager.create_backup(test_file)
        
        assert len(backup_manager.backup_history) == 15
        
        # Cleanup, keeping only last 10
        backup_manager.cleanup_old_backups(keep_last=10)
        
        assert len(backup_manager.backup_history) == 10


def test_repl_session_undo():
    """Test REPL session undo functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config
        config = ExeConfig(
            api_key="test_key",
            router_model="gemini-1.5-flash-8b",
            coder_model="gemini-2.0-flash-exp",
            auto_apply=False
        )
        
        # Override backup directory for test
        session = ReplSession(config, session_id=1)
        session.backup_manager.backup_dir = Path(tmpdir) / "backups"
        session.backup_manager.backup_dir.mkdir()
        
        # Create test files
        test_file1 = Path(tmpdir) / "test1.py"
        test_file2 = Path(tmpdir) / "test2.py"
        test_file1.write_text("original1")
        test_file2.write_text("original2")
        
        # Create backups and record patch
        backup_id1 = session.backup_manager.create_backup(test_file1)
        backup_id2 = session.backup_manager.create_backup(test_file2)
        
        # Simulate patching
        test_file1.write_text("modified1")
        test_file2.write_text("modified2")
        
        session.record_patch_batch(
            [str(test_file1), str(test_file2)],
            [backup_id1, backup_id2]
        )
        session.patch_count = 2
        session.modified_files = [str(test_file1), str(test_file2)]
        
        # Verify modifications
        assert test_file1.read_text() == "modified1"
        assert test_file2.read_text() == "modified2"
        
        # Undo patch
        success = session.undo_last_patch()
        
        assert success
        assert test_file1.read_text() == "original1"
        assert test_file2.read_text() == "original2"
        assert session.patch_count == 0
        assert len(session.modified_files) == 0
        assert len(session.patch_history) == 0


def test_undo_empty_history():
    """Test undo with no patch history."""
    config = ExeConfig(
        api_key="test_key",
        router_model="gemini-1.5-flash-8b",
        coder_model="gemini-2.0-flash-exp",
        auto_apply=False
    )
    
    session = ReplSession(config, session_id=1)
    
    # Try to undo with no history
    success = session.undo_last_patch()
    
    assert not success
    assert len(session.patch_history) == 0


def test_get_last_backup():
    """Test retrieving the last backup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backup_dir = Path(tmpdir) / "backups"
        backup_manager = PatchBackup(backup_dir)
        
        # Initially no backups
        assert backup_manager.get_last_backup() is None
        
        # Create backups
        test_file1 = Path(tmpdir) / "test1.py"
        test_file2 = Path(tmpdir) / "test2.py"
        test_file1.write_text("content1")
        test_file2.write_text("content2")
        
        backup_manager.create_backup(test_file1)
        backup_manager.create_backup(test_file2)
        
        last_backup = backup_manager.get_last_backup()
        
        assert last_backup is not None
        assert last_backup["original_path"] == str(test_file2)
