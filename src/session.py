"""
Session state management for Exo.
Handles runtime state like focus path during active sessions.
"""

from pathlib import Path
from typing import Optional


class SessionState:
    """Manages runtime session state for Exe."""
    
    def __init__(self):
        self.focus_path: Optional[str] = None
        self.project_root: Path = Path.cwd()
        self.active: bool = False
    
    def set_focus(self, path: str) -> None:
        """
        Set the focus path for the current session.
        
        Args:
            path: Path to focus on (relative or absolute)
        """
        # Normalize path to be relative to project root
        focus_path = Path(path)
        if not focus_path.is_absolute():
            focus_path = self.project_root / focus_path
        
        # Verify path exists
        if not focus_path.exists():
            raise ValueError(f"Path does not exist: {path}")
        
        self.focus_path = str(focus_path.relative_to(self.project_root))
    
    def clear_focus(self) -> None:
        """Clear the focus path."""
        self.focus_path = None
    
    def get_focus(self) -> Optional[str]:
        """Get the current focus path."""
        return self.focus_path
    
    def is_focused(self) -> bool:
        """Check if focus mode is active."""
        return self.focus_path is not None


# Global session state instance
session_state = SessionState()
