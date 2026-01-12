"""
ADK Tools for Exe.
Provides function tools with ToolContext integration for file editing, code search, etc.
"""

from google.adk.tools import FunctionTool, ToolContext
from pathlib import Path
from typing import Dict, List, Optional
import json


def edit_file(file_path: str, patch_content: str, tool_context: ToolContext) -> Dict[str, str]:
    """
    Apply a unified diff patch to a file.
    
    Args:
        file_path: Relative path to the file to edit
        patch_content: Unified diff format patch content
        tool_context: ADK ToolContext for session state access
        
    Returns:
        Dict with status, message, and file path
    """
    from ..ingestion.editor import Editor
    
    # Get session state for safety checks
    focus_path = tool_context.session.state.get("focus_path")
    project_root = tool_context.session.state.get("project_root", ".")
    
    # Validate file is within focus if focus is set
    if focus_path and not file_path.startswith(focus_path):
        return {
            "status": "error",
            "message": f"File '{file_path}' is outside focus path '{focus_path}'",
            "file": file_path
        }
    
    # Resolve file path
    full_path = Path(project_root) / file_path
    
    if not full_path.exists():
        return {
            "status": "error",
            "message": f"File not found: {file_path}",
            "file": file_path
        }
    
    # Apply patch using existing Editor
    editor = Editor()
    success = editor.apply_patch(full_path, patch_content)
    
    if success:
        # Update session state to track modified files
        modified_files = tool_context.session.state.get("modified_files", [])
        if file_path not in modified_files:
            modified_files.append(file_path)
            tool_context.session.state["modified_files"] = modified_files
        
        return {
            "status": "success",
            "message": f"Successfully patched {file_path}",
            "file": file_path
        }
    else:
        return {
            "status": "error",
            "message": f"Failed to apply patch to {file_path}",
            "file": file_path
        }


def search_code(query: str, limit: int = 10, tool_context: ToolContext = None) -> Dict[str, List[Dict]]:
    """
    Search codebase using semantic similarity.
    
    Args:
        query: Search query
        limit: Maximum number of results
        tool_context: ADK ToolContext for session state access
        
    Returns:
        Dict with search results (file paths, content snippets, scores)
    """
    from ..memory.db import VectorDB
    from ..memory.embedder import Embedder
    
    # Get session state
    focus_path = tool_context.session.state.get("focus_path") if tool_context else None
    api_key = tool_context.session.state.get("api_key") if tool_context else None
    
    # Initialize components
    embedder = Embedder(api_key)
    db = VectorDB()
    
    # Embed query
    query_embedding = embedder.embed_single(query)
    
    # Search with focus if set
    chunks = db.search(query_embedding, limit=limit, focus_path=focus_path)
    
    # Format results
    results = [
        {
            "file": chunk.file_path,
            "content": chunk.content,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "score": getattr(chunk, 'score', 1.0)
        }
        for chunk in chunks
    ]
    
    return {
        "query": query,
        "results": results,
        "count": len(results),
        "focus_path": focus_path
    }


def list_files(directory: str = ".", pattern: str = "*.py", tool_context: ToolContext = None) -> Dict[str, List[str]]:
    """
    List files in a directory matching a pattern.
    
    Args:
        directory: Directory to search (relative to project root)
        pattern: Glob pattern for matching files
        tool_context: ADK ToolContext for session state access
        
    Returns:
        Dict with list of matching file paths
    """
    project_root = tool_context.session.state.get("project_root", ".") if tool_context else "."
    focus_path = tool_context.session.state.get("focus_path") if tool_context else None
    
    # Resolve directory
    search_dir = Path(project_root) / directory
    
    if not search_dir.exists() or not search_dir.is_dir():
        return {
            "status": "error",
            "message": f"Directory not found: {directory}",
            "files": []
        }
    
    # Find files
    files = []
    for file_path in search_dir.rglob(pattern):
        if file_path.is_file():
            # Make path relative to project root
            rel_path = str(file_path.relative_to(project_root))
            
            # Filter by focus if set
            if focus_path and not rel_path.startswith(focus_path):
                continue
            
            files.append(rel_path)
    
    return {
        "directory": directory,
        "pattern": pattern,
        "files": sorted(files),
        "count": len(files),
        "focus_path": focus_path
    }


def read_file(file_path: str, start_line: Optional[int] = None, 
              end_line: Optional[int] = None, tool_context: ToolContext = None) -> Dict[str, str]:
    """
    Read contents of a file or specific line range.
    
    Args:
        file_path: Relative path to file
        start_line: Optional starting line number (1-indexed)
        end_line: Optional ending line number (1-indexed)
        tool_context: ADK ToolContext for session state access
        
    Returns:
        Dict with file contents and metadata
    """
    project_root = tool_context.session.state.get("project_root", ".") if tool_context else "."
    focus_path = tool_context.session.state.get("focus_path") if tool_context else None
    
    # Validate focus
    if focus_path and not file_path.startswith(focus_path):
        return {
            "status": "error",
            "message": f"File '{file_path}' is outside focus path '{focus_path}'",
            "content": ""
        }
    
    # Resolve path
    full_path = Path(project_root) / file_path
    
    if not full_path.exists():
        return {
            "status": "error",
            "message": f"File not found: {file_path}",
            "content": ""
        }
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Extract line range if specified
        if start_line is not None or end_line is not None:
            start = (start_line - 1) if start_line else 0
            end = end_line if end_line else len(lines)
            content = ''.join(lines[start:end])
        else:
            content = ''.join(lines)
        
        return {
            "status": "success",
            "file": file_path,
            "content": content,
            "total_lines": len(lines),
            "start_line": start_line,
            "end_line": end_line
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reading file: {str(e)}",
            "content": ""
        }


def get_focus_info(tool_context: ToolContext) -> Dict[str, str]:
    """
    Get current focus path and statistics.
    
    Args:
        tool_context: ADK ToolContext for session state access
        
    Returns:
        Dict with focus information
    """
    focus_path = tool_context.session.state.get("focus_path")
    modified_files = tool_context.session.state.get("modified_files", [])
    
    return {
        "focus_path": focus_path or "Not set (searching entire codebase)",
        "is_focused": focus_path is not None,
        "modified_files_count": len(modified_files),
        "modified_files": modified_files
    }


# Create FunctionTool instances
edit_file_tool = FunctionTool(edit_file)
search_code_tool = FunctionTool(search_code)
list_files_tool = FunctionTool(list_files)
read_file_tool = FunctionTool(read_file)
get_focus_info_tool = FunctionTool(get_focus_info)

# Export all tools as a list for easy agent configuration
ALL_TOOLS = [
    edit_file_tool,
    search_code_tool,
    list_files_tool,
    read_file_tool,
    get_focus_info_tool
]

# Export commonly used tool sets
EDITOR_TOOLS = [edit_file_tool, read_file_tool]
SEARCH_TOOLS = [search_code_tool, list_files_tool]
INFO_TOOLS = [get_focus_info_tool]
