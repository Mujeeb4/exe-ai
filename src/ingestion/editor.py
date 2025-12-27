"""
Handles safe patching of files using unified diffs.
"""

from pathlib import Path
from typing import Optional, Dict
import difflib
import hashlib
from contextlib import contextmanager


class Editor:
    """Applies unified diff patches to files safely with loop prevention."""
    
    def __init__(self):
        self.is_modifying = False  # Flag for watcher loop prevention
        self._modification_count = 0
    
    @contextmanager
    def modify_context(self):
        """Context manager for safe file modifications with loop prevention."""
        self.is_modifying = True
        self._modification_count += 1
        try:
            yield
        finally:
            self.is_modifying = False
    
    def apply_patch(self, file_path: Path, patch_content: str) -> bool:
        """
        Apply a unified diff patch to a file.
        
        Args:
            file_path: Path to the file to patch
            patch_content: Unified diff content
            
        Returns:
            True if patch applied successfully, False otherwise
        """
        result = self.apply_patch_with_details(file_path, patch_content)
        return result["success"]
    
    def apply_patch_with_details(self, file_path: Path, patch_content: str) -> Dict:
        """
        Apply a unified diff patch to a file with detailed error information.
        
        Args:
            file_path: Path to the file to patch
            patch_content: Unified diff content
            
        Returns:
            Dict with 'success' (bool) and optional 'error' (str) keys
        """
        with self.modify_context():
            try:
                # Verify file exists
                if not file_path.exists():
                    return {"success": False, "error": "File not found"}
                
                # Read current file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_lines = f.readlines()
                except UnicodeDecodeError:
                    return {"success": False, "error": "File encoding error (not UTF-8)"}
                
                # Calculate file hash to detect external modifications
                original_hash = hashlib.md5(''.join(original_lines).encode()).hexdigest()
                
                # Parse and apply patch
                patched_lines = self._apply_diff(original_lines, patch_content)
                
                if patched_lines is None:
                    return {"success": False, "error": "Patch does not apply cleanly (file may have changed)"}
                
                # Write patched content
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(patched_lines)
                except PermissionError:
                    return {"success": False, "error": "Permission denied (file may be read-only)"}
                
                return {"success": True}
            
            except Exception as e:
                return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    def _apply_diff(self, original_lines: list, patch_content: str) -> Optional[list]:
        """
        Parse unified diff and apply to original lines.
        
        Args:
            original_lines: Original file lines
            patch_content: Unified diff string
            
        Returns:
            Patched lines or None if patch fails
        """
        try:
            # Clean patch content
            patch_content = patch_content.strip()
            if not patch_content:
                return None
            
            patch_lines = patch_content.splitlines()
            result = original_lines.copy()
            offset = 0
            hunks_applied = 0
            
            # Parse diff hunks
            i = 0
            while i < len(patch_lines):
                line = patch_lines[i]
                
                # Skip header lines (---, +++, diff, index)
                if line.startswith('---') or line.startswith('+++') or \
                   line.startswith('diff') or line.startswith('index'):
                    i += 1
                    continue
                
                if line.startswith('@@'):
                    # Parse hunk header: @@ -start,count +start,count @@
                    try:
                        parts = line.split()
                        if len(parts) < 3:
                            print(f"⚠ Malformed hunk header: {line}")
                            i += 1
                            continue
                        
                        old_info = parts[1]
                        old_start = int(old_info.split(',')[0][1:])
                        
                        # Apply hunk
                        i += 1
                        line_num = old_start - 1 + offset
                        
                        while i < len(patch_lines) and not patch_lines[i].startswith('@@'):
                            diff_line = patch_lines[i]
                            
                            if diff_line.startswith('-'):
                                # Remove line
                                if line_num < len(result):
                                    result.pop(line_num)
                                    offset -= 1
                                else:
                                    print(f"⚠ Line {line_num} out of range during removal")
                            elif diff_line.startswith('+'):
                                # Add line
                                content = diff_line[1:]
                                if not content.endswith('\n'):
                                    content += '\n'
                                result.insert(line_num, content)
                                line_num += 1
                                offset += 1
                            elif diff_line.startswith(' '):
                                # Context line
                                line_num += 1
                            # Ignore other lines (e.g., \\ No newline at end of file)
                            
                            i += 1
                        
                        hunks_applied += 1
                    except (ValueError, IndexError) as e:
                        print(f"⚠ Error parsing hunk at line {i}: {e}")
                        i += 1
                        continue
                else:
                    i += 1
            
            if hunks_applied == 0:
                print("⚠ No valid hunks found in patch")
                return None
            
            return result
        
        except Exception as e:
            print(f"✗ Failed to parse diff: {e}")
            return None
    
    def get_modification_count(self) -> int:
        """Get the total number of modifications made by this editor."""
        return self._modification_count
