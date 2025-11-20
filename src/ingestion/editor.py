"""
Handles safe patching of files using unified diffs.
"""

from pathlib import Path
from typing import Optional
import difflib


class Editor:
    """Applies unified diff patches to files safely."""
    
    def __init__(self):
        self.is_modifying = False  # Flag for watcher loop prevention
    
    def apply_patch(self, file_path: Path, patch_content: str) -> bool:
        """
        Apply a unified diff patch to a file.
        
        Args:
            file_path: Path to the file to patch
            patch_content: Unified diff content
            
        Returns:
            True if patch applied successfully, False otherwise
        """
        self.is_modifying = True
        
        try:
            # Read current file content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()
            
            # Parse and apply patch
            patched_lines = self._apply_diff(original_lines, patch_content)
            
            if patched_lines is None:
                return False
            
            # Write patched content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(patched_lines)
            
            return True
        
        except Exception as e:
            print(f"Error applying patch: {e}")
            return False
        
        finally:
            self.is_modifying = False
    
    def _apply_diff(self, original_lines: list, patch_content: str) -> Optional[list]:
        """
        Parse unified diff and apply to original lines.
        
        Args:
            original_lines: Original file lines
            patch_content: Unified diff string
            
        Returns:
            Patched lines or None if patch fails
        """
        # Simple implementation: parse unified diff format
        # This is a basic version - production would use a robust library
        
        try:
            patch_lines = patch_content.splitlines()
            result = original_lines.copy()
            offset = 0
            
            # Parse diff hunks
            i = 0
            while i < len(patch_lines):
                line = patch_lines[i]
                
                if line.startswith('@@'):
                    # Parse hunk header: @@ -start,count +start,count @@
                    parts = line.split()
                    old_start = int(parts[1].split(',')[0][1:])
                    
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
                        elif diff_line.startswith('+'):
                            # Add line
                            # Ensure newline if missing
                            content = diff_line[1:]
                            if not content.endswith('\n'):
                                content += '\n'
                            result.insert(line_num, content)
                            line_num += 1
                            offset += 1
                        else:
                            # Context line
                            line_num += 1
                        
                        i += 1
                else:
                    i += 1
            
            return result
        
        except Exception as e:
            print(f"Failed to parse diff: {e}")
            return None
