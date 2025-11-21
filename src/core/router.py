"""
Router Agent: Low-level intent classifier.
Analyzes user queries and determines relevant context.
"""

from typing import Optional, TYPE_CHECKING, List
import json
from google import genai
from google.genai import types
from .models import RouterOutput
from ..memory.db import VectorDB
from .router_analytics import RouterAnalytics

if TYPE_CHECKING:
    from ..memory.embedder import Embedder


class Router:
    """Routes user queries and retrieves relevant context."""
    
    def __init__(self, api_key: str = None, focus_path: Optional[str] = None, 
                 repo_context: Optional[str] = None, model: str = "gemini-1.5-flash-8b"):
        # Initialize Google ADK client
        self.client = genai.Client(api_key=api_key)
        self.api_key = api_key
        self.focus_path = focus_path
        self.repo_context = repo_context
        self.model = model
        self.analytics = RouterAnalytics()
        self._focus_suggestions = []
        self.repo_context = repo_context
        self.analytics = RouterAnalytics()
    
    def suggest_focus(self, query: str) -> List[str]:
        """
        Suggest relevant focus paths based on query content.
        
        Args:
            query: User's query
            
        Returns:
            List of suggested focus paths
        """
        return self.analytics.get_focus_suggestions(query)
    
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
        try:
            # Suggest focus if not set
            if not self.focus_path:
                suggestions = self.suggest_focus(query)
                if suggestions:
                    # Store suggestions for potential use
                    self._focus_suggestions = suggestions
            
            # Embed the query
            query_embedding = embedder.embed_single(query)
            
            # Search for relevant chunks with increased limit for better context
            search_limit = 10 if self.focus_path else 15
            chunks = db.search(query_embedding, limit=search_limit, focus_path=self.focus_path)
            
            # Extract file paths
            relevant_files = list(set([chunk.file_path for chunk in chunks]))
            
            # Use LLM to classify intent with enhanced prompt
            system_prompt = """Classify the user's intent. Respond with JSON: {"intent": "question|code_edit|refactor|explain"}
            
Intent definitions:
- question: User asking about code, architecture, or functionality
- code_edit: User wants to modify specific code (add/change/fix)
- refactor: User wants to restructure/improve code without changing behavior
- explain: User wants detailed explanation of how something works"""
            
            if self.repo_context:
                system_prompt += f"\n\nRepository Context:\n{self.repo_context[:2000]}..."  # Include first 2000 chars
            
            prompt = f"{system_prompt}\n\nUser query: {query}"
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="application/json"
                )
            )
            
            content = response.text
            if content is None:
                intent = "question"
            else:
                try:
                    data = json.loads(content)
                    intent = data.get("intent", "question")
                except json.JSONDecodeError:
                    intent = "question"
            
            # Log analytics
            self.analytics.log_query(query, intent, self.focus_path, len(chunks))
            
            return RouterOutput(
                intent=intent,
                relevant_files=relevant_files,
                focus_area=self.focus_path,
                relevant_chunks=chunks
            )
        
        except Exception as e:
            # Fallback to basic routing on error
            print(f"Router error: {e}")
            return RouterOutput(
                intent="question",
                relevant_files=[],
                focus_area=self.focus_path,
                relevant_chunks=[]
            )
    
    def get_statistics(self):
        """Get router usage statistics."""
        return self.analytics.get_statistics()
        result = json.loads(content)
        
        return RouterOutput(
            intent=result["intent"],
            relevant_files=relevant_files,
            focus_area=self.focus_path
        )
