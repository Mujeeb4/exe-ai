"""
Tests for the diff-based editor.
"""

import pytest
from pathlib import Path
from src.ingestion.editor import Editor


def test_apply_simple_patch(tmp_path):
    """Test applying a simple unified diff patch."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""def hello():
    return "world"
""")
    
    # Simple patch: change "world" to "universe"
    patch = """--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
 def hello():
-    return "world"
+    return "universe"
"""
    
    editor = Editor()
    success = editor.apply_patch(test_file, patch)
    
    assert success
    assert "universe" in test_file.read_text()


def test_editor_sets_modification_flag():
    """Test that editor sets is_modifying flag."""
    editor = Editor()
    
    assert editor.is_modifying == False
    
    # Flag should be reset even if patch fails
    editor.apply_patch(Path("nonexistent.py"), "")
    
    assert editor.is_modifying == False
