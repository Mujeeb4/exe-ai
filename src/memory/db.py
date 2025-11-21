"""
LanceDB wrapper for vector storage.
"""

from typing import List, Optional
from pathlib import Path
import lancedb
from ..core.models import CodeChunk


class VectorDB:
    """Manages vector database operations using LanceDB."""
    
    def __init__(self, db_path: str = ".exo/lancedb"):
        self.db_path = db_path
        self.db = lancedb.connect(db_path)
        self._init_table()
    
    def _init_table(self):
        """Initialize or connect to the code chunks table."""
        try:
            self.table = self.db.open_table("code_chunks")
        except:
            # Table doesn't exist yet, will be created on first insert
            self.table = None
    
    def add_chunks(self, chunks: List[CodeChunk], embeddings: List[List[float]]):
        """
        Add code chunks with their embeddings to the database.
        
        Args:
            chunks: List of CodeChunk objects
            embeddings: List of embedding vectors
        """
        from datetime import datetime, UTC
        data = []
        for chunk, embedding in zip(chunks, embeddings):
            data.append({
                "file_path": chunk.file_path,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chunk_type": chunk.chunk_type,
                "name": chunk.name or "",
                "language": chunk.language or "unknown",
                "imports": ",".join(chunk.imports or []),
                "last_modified": chunk.last_modified or datetime.now(UTC).isoformat(),
                "vector": embedding
            })
        
        if self.table is None:
            self.table = self.db.create_table("code_chunks", data)
        else:
            self.table.add(data)
    
    def search(self, query_embedding: List[float], limit: int = 10, focus_path: Optional[str] = None, 
               chunk_type: Optional[str] = None, language: Optional[str] = None) -> List[CodeChunk]:
        """
        Search for similar code chunks with optional metadata filtering.
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            focus_path: Optional path to filter results
            chunk_type: Optional filter by chunk type ("function", "class", etc.)
            language: Optional filter by language ("python", "javascript", etc.)
            
        Returns:
            List of relevant CodeChunk objects
        """
        if self.table is None:
            return []
        
        results = self.table.search(query_embedding).limit(limit)
        
        if focus_path:
            results = results.where(f"file_path LIKE '{focus_path}%'")
        
        if chunk_type:
            results = results.where(f"chunk_type = '{chunk_type}'")
            
        if language:
            results = results.where(f"language = '{language}'")
        
        chunks = []
        for row in results.to_list():
            chunks.append(CodeChunk(
                file_path=row["file_path"],
                chunk_id=row["chunk_id"],
                content=row["content"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                chunk_type=row["chunk_type"],
                name=row.get("name"),
                language=row.get("language"),
                imports=row.get("imports", "").split(",") if row.get("imports") else None,
                last_modified=row.get("last_modified")
            ))
        
        return chunks
    
    def get_all_chunks(self) -> List[CodeChunk]:
        """
        Retrieve all chunks from the database.
        
        Returns:
            List of all CodeChunk objects
        """
        if self.table is None:
            return []
        
        chunks = []
        for row in self.table.to_pandas().to_dict('records'):
            chunks.append(CodeChunk(
                file_path=row["file_path"],
                chunk_id=row["chunk_id"],
                content=row["content"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                chunk_type=row["chunk_type"],
                name=row.get("name"),
                language=row.get("language"),
                imports=row.get("imports", "").split(",") if row.get("imports") else None,
                last_modified=row.get("last_modified")
            ))
        
        return chunks
    
    def update_file(self, file_path: str, chunks: List[CodeChunk], embeddings: List[List[float]]):
        """
        Update chunks for a specific file (delete old, add new).
        
        Args:
            file_path: Path to the file
            chunks: New chunks for the file
            embeddings: Embeddings for the new chunks
        """
        if self.table is not None:
            # Delete old chunks for this file
            self.table.delete(f"file_path = '{file_path}'")
        
        # Add new chunks
        self.add_chunks(chunks, embeddings)
