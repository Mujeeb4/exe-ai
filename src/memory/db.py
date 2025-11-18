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
                "vector": embedding
            })
        
        if self.table is None:
            self.table = self.db.create_table("code_chunks", data)
        else:
            self.table.add(data)
    
    def search(self, query_embedding: List[float], limit: int = 10, focus_path: Optional[str] = None) -> List[CodeChunk]:
        """
        Search for similar code chunks.
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            focus_path: Optional path to filter results
            
        Returns:
            List of relevant CodeChunk objects
        """
        if self.table is None:
            return []
        
        results = self.table.search(query_embedding).limit(limit)
        
        if focus_path:
            results = results.where(f"file_path LIKE '{focus_path}%'")
        
        chunks = []
        for row in results.to_list():
            chunks.append(CodeChunk(
                file_path=row["file_path"],
                chunk_id=row["chunk_id"],
                content=row["content"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                chunk_type=row["chunk_type"],
                name=row.get("name")
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
