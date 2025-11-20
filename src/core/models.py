"""
Pydantic models for structured data in Exo.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel


class RouterOutput(BaseModel):
    """Output from the Router agent."""
    intent: Literal["question", "code_edit", "refactor", "explain"]
    relevant_files: List[str]
    focus_area: Optional[str] = None
    relevant_chunks: Optional[List['CodeChunk']] = None



class CoderOutput(BaseModel):
    """Output from the Coder agent."""
    type: Literal["answer", "patch"]
    content: str
    files_to_modify: Optional[List[str]] = None


class CodeChunk(BaseModel):
    """Represents a chunk of code for vector storage."""
    file_path: str
    chunk_id: str
    content: str
    start_line: int
    end_line: int
    chunk_type: str  # "function", "class", "module"
    name: Optional[str] = None
    language: Optional[str] = None  # "python", "javascript", etc.
    imports: Optional[List[str]] = None  # List of imported modules
    last_modified: Optional[str] = None  # Timestamp for recency scoring
