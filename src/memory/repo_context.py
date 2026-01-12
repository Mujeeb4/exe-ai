"""
Repository context manager for maintaining complete codebase overview.
Generates and caches a comprehensive summary of the repository structure.
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime, UTC
from ..core.models import CodeChunk


class RepositoryContext:
    """Maintains complete repository context for AI agents."""
    
    def __init__(self, cache_path: str = ".exo/repo_context.json"):
        self.cache_path = cache_path
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self.context: Dict = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cached context if exists."""
        cache_file = Path(self.cache_path)
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.context = json.load(f)
    
    def _save_cache(self):
        """Save context to cache."""
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.context, f, indent=2)
    
    def build_context(self, root_path: Path, chunks: List[CodeChunk]) -> str:
        """
        Build comprehensive repository context from all chunks.
        
        Args:
            root_path: Root directory of the repository
            chunks: All code chunks from the repository
            
        Returns:
            Formatted context string for AI agents
        """
        # Organize chunks by file
        files_map: Dict[str, List[CodeChunk]] = {}
        for chunk in chunks:
            if chunk.file_path not in files_map:
                files_map[chunk.file_path] = []
            files_map[chunk.file_path].append(chunk)
        
        # Build directory structure
        directory_tree = self._build_directory_tree(list(files_map.keys()))
        
        # Analyze codebase
        stats = self._analyze_codebase(chunks)
        
        # Build context sections
        context_parts = [
            "# REPOSITORY CONTEXT",
            f"Root: {root_path}",
            f"Last Updated: {datetime.now(UTC).isoformat()}",
            "",
            "## DIRECTORY STRUCTURE",
            directory_tree,
            "",
            "## CODEBASE STATISTICS",
            f"Total Files: {len(files_map)}",
            f"Total Chunks: {len(chunks)}",
            f"Languages: {', '.join(stats['languages'])}",
            f"Total Functions: {stats['functions']}",
            f"Total Classes: {stats['classes']}",
            "",
            "## FILE OVERVIEW",
        ]
        
        # Add file summaries
        for file_path in sorted(files_map.keys()):
            file_chunks = files_map[file_path]
            context_parts.append(f"\n### {file_path}")
            context_parts.append(f"Chunks: {len(file_chunks)}")
            
            # List all functions and classes
            functions = [c.name for c in file_chunks if c.chunk_type == "function" and c.name]
            classes = [c.name for c in file_chunks if c.chunk_type == "class" and c.name]
            
            if classes:
                context_parts.append(f"Classes: {', '.join(classes)}")
            if functions:
                context_parts.append(f"Functions: {', '.join(functions)}")
            
            # Show imports
            all_imports = set()
            for chunk in file_chunks:
                if chunk.imports:
                    all_imports.update(chunk.imports)
            if all_imports:
                context_parts.append(f"Imports: {', '.join(sorted(all_imports))}")
        
        context_str = "\n".join(context_parts)
        
        # Cache the context
        self.context = {
            "generated_at": datetime.now(UTC).isoformat(),
            "root_path": str(root_path),
            "content": context_str,
            "stats": stats
        }
        self._save_cache()
        
        return context_str
    
    def build_lightweight_context(self, root_path: Path, chunks: List[CodeChunk]) -> str:
        """
        Build a minimal lightweight context to reduce token usage.
        The agent should use tools (search_code, read_file) for detailed context.
        
        Args:
            root_path: Root directory of the repository
            chunks: All code chunks from the repository
            
        Returns:
            Minimal context string with just overview statistics
        """
        # Organize chunks by file  
        files_map: Dict[str, List[CodeChunk]] = {}
        for chunk in chunks:
            if chunk.file_path not in files_map:
                files_map[chunk.file_path] = []
            files_map[chunk.file_path].append(chunk)
        
        # Analyze codebase
        stats = self._analyze_codebase(chunks)
        
        # Build top-level directory structure only
        top_dirs = self._get_top_level_structure(list(files_map.keys()))
        
        # Create minimal context
        context_parts = [
            "# REPOSITORY OVERVIEW",
            f"Root: {root_path.name}",
            "",
            "## STATISTICS",
            f"- Files: {len(files_map)}",
            f"- Functions: {stats['functions']}",
            f"- Classes: {stats['classes']}",
            f"- Languages: {', '.join(stats['languages']) or 'Python'}",
            "",
            "## TOP-LEVEL STRUCTURE",
            top_dirs,
            "",
            "## HOW TO GET MORE CONTEXT",
            "Use the search_code tool to find relevant code by semantic search.",
            "Use the read_file tool to view specific file contents.",
            "Use the list_files tool to explore directory contents.",
        ]
        
        return "\n".join(context_parts)
    
    def _get_top_level_structure(self, file_paths: List[str]) -> str:
        """Get only top-level directories and file counts."""
        top_dirs: Dict[str, int] = {}
        root_files: List[str] = []
        
        for path in file_paths:
            p = Path(path)
            parts = p.parts
            
            if len(parts) == 1:
                # Root level file
                root_files.append(parts[0])
            else:
                # Get top-level directory
                top_dir = parts[0]
                top_dirs[top_dir] = top_dirs.get(top_dir, 0) + 1
        
        lines = []
        for dir_name in sorted(top_dirs.keys()):
            lines.append(f"- {dir_name}/ ({top_dirs[dir_name]} files)")
        
        if root_files:
            lines.append(f"- (root): {', '.join(sorted(root_files)[:5])}")
            if len(root_files) > 5:
                lines.append(f"  ... and {len(root_files) - 5} more")
        
        return "\n".join(lines) if lines else "No files found"
    
    def _build_directory_tree(self, file_paths: List[str]) -> str:
        """Build a visual directory tree."""
        tree_lines = []
        sorted_paths = sorted(file_paths)
        
        # Group by directory
        dirs: Dict[str, List[str]] = {}
        for path in sorted_paths:
            p = Path(path)
            dir_name = str(p.parent) if p.parent != Path('.') else "."
            if dir_name not in dirs:
                dirs[dir_name] = []
            dirs[dir_name].append(p.name)
        
        for dir_name in sorted(dirs.keys()):
            tree_lines.append(f"{dir_name}/")
            for file_name in sorted(dirs[dir_name]):
                tree_lines.append(f"  - {file_name}")
        
        return "\n".join(tree_lines)
    
    def _analyze_codebase(self, chunks: List[CodeChunk]) -> Dict:
        """Analyze codebase statistics."""
        languages = set()
        functions = 0
        classes = 0
        
        for chunk in chunks:
            if chunk.language:
                languages.add(chunk.language)
            if chunk.chunk_type == "function":
                functions += 1
            elif chunk.chunk_type == "class":
                classes += 1
        
        return {
            "languages": sorted(list(languages)),
            "functions": functions,
            "classes": classes
        }
    
    def get_context(self) -> Optional[str]:
        """Get cached repository context."""
        return self.context.get("content")
    
    def is_stale(self, max_age_seconds: int = 3600) -> bool:
        """Check if cached context is stale."""
        if not self.context or "generated_at" not in self.context:
            return True
        
        generated_at = datetime.fromisoformat(self.context["generated_at"])
        age = (datetime.now(UTC) - generated_at).total_seconds()
        return age > max_age_seconds
    
    def get_file_context(self, file_path: str) -> Optional[str]:
        """Get context for a specific file."""
        context = self.get_context()
        if not context:
            return None
        
        # Extract file section from context
        lines = context.split("\n")
        file_section = []
        in_section = False
        
        for line in lines:
            if line.startswith(f"### {file_path}"):
                in_section = True
            elif in_section and line.startswith("### "):
                break
            elif in_section:
                file_section.append(line)
        
        return "\n".join(file_section) if file_section else None
