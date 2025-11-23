"""
Tests for Day 8 polish features: diff formatting and enhanced error handling.
"""

import pytest
from pathlib import Path
import tempfile
from src.interface.repl import format_diff_with_colors
from src.ingestion.editor import Editor
from rich.text import Text


def test_diff_color_formatting():
    """Test that diffs are formatted with proper colors."""
    diff = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 def hello():
-    return 'world'
+    return 'universe'
"""
    
    result = format_diff_with_colors(diff)
    
    # Should return Rich Text object
    assert isinstance(result, Text)
    
    # Should contain the diff content
    assert '--- a/test.py' in result.plain
    assert '+++ b/test.py' in result.plain
    assert "return 'universe'" in result.plain


def test_diff_color_additions():
    """Test that additions are formatted in green."""
    diff = "+    return 'new line'"
    
    result = format_diff_with_colors(diff)
    
    # Check that style is applied
    assert len(result._spans) > 0
    # First span should have green style for additions
    assert any('green' in str(span.style) for span in result._spans)


def test_diff_color_deletions():
    """Test that deletions are formatted in red."""
    diff = "-    return 'old line'"
    
    result = format_diff_with_colors(diff)
    
    # Check that style is applied
    assert len(result._spans) > 0
    # Should have red style for deletions
    assert any('red' in str(span.style) for span in result._spans)


def test_diff_color_hunk_headers():
    """Test that hunk headers are formatted in cyan."""
    diff = "@@ -1,3 +1,4 @@"
    
    result = format_diff_with_colors(diff)
    
    # Check that style is applied
    assert len(result._spans) > 0
    # Should have cyan style for hunk headers
    assert any('cyan' in str(span.style) for span in result._spans)


def test_editor_apply_patch_with_details_success():
    """Test successful patch application with details."""
    with tempfile.TemporaryDirectory() as tmpdir:
        editor = Editor()
        test_file = Path(tmpdir) / "test.py"
        test_file.write_text("def hello():\n    return 'world'\n")
        
        patch = """--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
 def hello():
-    return 'world'
+    return 'universe'
"""
        
        result = editor.apply_patch_with_details(test_file, patch)
        
        assert result["success"] is True
        assert "error" not in result
        assert test_file.read_text() == "def hello():\n    return 'universe'\n"


def test_editor_apply_patch_with_details_file_not_found():
    """Test patch application when file doesn't exist."""
    editor = Editor()
    nonexistent = Path("/nonexistent/file.py")
    
    result = editor.apply_patch_with_details(nonexistent, "dummy patch")
    
    assert result["success"] is False
    assert result["error"] == "File not found"


def test_editor_apply_patch_with_details_encoding_error():
    """Test patch application with encoding error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        editor = Editor()
        test_file = Path(tmpdir) / "binary.dat"
        
        # Write binary data
        test_file.write_bytes(b'\x80\x81\x82\x83')
        
        patch = "dummy patch"
        result = editor.apply_patch_with_details(test_file, patch)
        
        assert result["success"] is False
        assert "encoding error" in result["error"].lower()


def test_editor_apply_patch_with_details_malformed_patch():
    """Test patch application with malformed patch content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        editor = Editor()
        test_file = Path(tmpdir) / "test.py"
        test_file.write_text("def hello():\n    return 'world'\n")
        
        # Completely malformed patch that can't be parsed
        patch = "this is not a valid unified diff format at all"
        
        result = editor.apply_patch_with_details(test_file, patch)
        
        # Should fail because patch is malformed
        assert result["success"] is False
        assert "does not apply cleanly" in result["error"]


def test_editor_apply_patch_with_details_permission_error(tmp_path):
    """Test patch application with permission error."""
    import os
    import stat
    
    editor = Editor()
    test_file = tmp_path / "readonly.py"
    test_file.write_text("def hello():\n    return 'world'\n")
    
    # Make file read-only
    os.chmod(test_file, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    
    patch = """--- a/readonly.py
+++ b/readonly.py
@@ -1,2 +1,2 @@
 def hello():
-    return 'world'
+    return 'universe'
"""
    
    try:
        result = editor.apply_patch_with_details(test_file, patch)
        
        # On Windows, this might still succeed, on Unix it should fail
        if result["success"] is False:
            assert "permission" in result["error"].lower() or "read-only" in result["error"].lower()
    finally:
        # Restore permissions for cleanup
        os.chmod(test_file, stat.S_IWUSR | stat.S_IRUSR)


def test_editor_backward_compatibility():
    """Test that apply_patch() still works for backward compatibility."""
    with tempfile.TemporaryDirectory() as tmpdir:
        editor = Editor()
        test_file = Path(tmpdir) / "test.py"
        test_file.write_text("def hello():\n    return 'world'\n")
        
        patch = """--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
 def hello():
-    return 'world'
+    return 'universe'
"""
        
        # Old method should still work
        success = editor.apply_patch(test_file, patch)
        
        assert success is True
        assert test_file.read_text() == "def hello():\n    return 'universe'\n"


def test_empty_diff_formatting():
    """Test formatting of empty diff."""
    result = format_diff_with_colors("")
    
    assert isinstance(result, Text)
    assert result.plain == "\n"


def test_multiline_diff_formatting():
    """Test formatting of complex multi-file diff."""
    diff = """--- a/file1.py
+++ b/file1.py
@@ -1,5 +1,6 @@
 import os
+import sys
 
 def main():
-    print("old")
+    print("new")
     return 0
--- a/file2.py
+++ b/file2.py
@@ -1,2 +1,2 @@
-VERSION = "1.0"
+VERSION = "2.0"
"""
    
    result = format_diff_with_colors(diff)
    
    assert isinstance(result, Text)
    assert "file1.py" in result.plain
    assert "file2.py" in result.plain
    assert "+import sys" in result.plain
    assert '-VERSION = "1.0"' in result.plain
