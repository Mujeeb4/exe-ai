"""
Router Agent: Low-level intent classifier.
Analyzes user queries and determines relevant context.
"""

from typing import Optional, TYPE_CHECKING
import openai
from .models import RouterOutput
from ..memory.db import VectorDB

if TYPE_CHECKING:
    from ..memory.embedder import Embedder


class Router:
    """Routes user queries and retrieves relevant context."""
    
    def __init__(self, api_key: str, focus_path: Optional[str] = None):
        self.client = openai.OpenAI(api_key=api_key)
        self.focus_path = focus_path
    
    def route(self, query: str, db: VectorDB, embedder: 'Embedder') -> RouterOutput:
        """
        Classify user intent and retrieve relevant code chunks.
        
        Args:
            query: User's input query
            db: Vector database instance
            embedder: Embedder instance for query embedding
            
        Returns:
            RouterOutput with intent and relevant files
        """
        # Embed the query
        query_embedding = embedder.embed_single(query)
        
        # Search for relevant chunks
        chunks = db.search(query_embedding, limit=5, focus_path=self.focus_path)
        
        # Extract file paths
        relevant_files = list(set([chunk.file_path for chunk in chunks]))
        
        # Use LLM to classify intent
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Classify the user's intent. Respond with JSON: {\"intent\": \"question|code_edit|refactor|explain\"}"},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"}
        )
        
        import json
        content = response.choices[0].message.content
        if content is None:
            intent = "question"
        else:
            try:
                data = json.loads(content)
                intent = data.get("intent", "question")
            except json.JSONDecodeError:
                intent = "question"
                
        return RouterOutput(
            intent=intent,
            relevant_files=relevant_files,
            focus_area=self.focus_path,
            relevant_chunks=chunks
        )
        result = json.loads(content)
        
        return RouterOutput(
            intent=result["intent"],
            relevant_files=relevant_files,
            focus_area=self.focus_path
        )
