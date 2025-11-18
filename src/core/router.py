"""
Router Agent: Low-level intent classifier.
Analyzes user queries and determines relevant context.
"""

from typing import Optional
import openai
from .models import RouterOutput
from ..memory.db import VectorDB


class Router:
    """Routes user queries and retrieves relevant context."""
    
    def __init__(self, api_key: str, focus_path: Optional[str] = None):
        self.client = openai.OpenAI(api_key=api_key)
        self.focus_path = focus_path
    
    def route(self, query: str, db: VectorDB) -> RouterOutput:
        """
        Classify user intent and retrieve relevant code chunks.
        
        Args:
            query: User's input query
            db: Vector database instance
            
        Returns:
            RouterOutput with intent and relevant files
        """
        # Note: This is a placeholder. In real implementation, we'd embed the query first.
        # For now, return empty since embedding requires initialization
        chunks = []
        
        # Extract file paths
        relevant_files = list(set([chunk.file_path for chunk in chunks]))
        
        # Use LLM to classify intent
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Classify the user's intent. Respond with JSON: {\"intent\": \"question|code_edit|refactor|explain\"}"},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"}
        )
        
        import json
        content = response.choices[0].message.content
        if content is None:
            content = '{"intent": "question"}'
        result = json.loads(content)
        
        return RouterOutput(
            intent=result["intent"],
            relevant_files=relevant_files,
            focus_area=self.focus_path
        )
