"""
Enhanced router with intelligent query analysis and focus suggestions.
"""

from typing import Optional, List, Dict, TYPE_CHECKING
import openai
from .models import RouterOutput
from ..memory.db import VectorDB

if TYPE_CHECKING:
    from ..memory.embedder import Embedder


class RouterAnalytics:
    """Analytics and insights for router performance."""
    
    def __init__(self):
        self.query_history: List[Dict] = []
    
    def log_query(self, query: str, intent: str, focus_area: Optional[str], num_results: int):
        """Log a query for analytics."""
        self.query_history.append({
            "query": query,
            "intent": intent,
            "focus_area": focus_area,
            "num_results": num_results
        })
    
    def get_focus_suggestions(self, query: str) -> List[str]:
        """Suggest focus paths based on query analysis."""
        suggestions = []
        
        # Keyword-based suggestions
        if "test" in query.lower():
            suggestions.append("tests/")
        if "api" in query.lower():
            suggestions.append("src/api/")
        if "database" in query.lower() or "db" in query.lower():
            suggestions.append("src/db/")
        if "model" in query.lower():
            suggestions.append("src/models/")
        if "util" in query.lower():
            suggestions.append("src/utils/")
        
        return suggestions
    
    def get_statistics(self) -> Dict:
        """Get router usage statistics."""
        if not self.query_history:
            return {}
        
        total_queries = len(self.query_history)
        intents = {}
        focus_areas = {}
        
        for entry in self.query_history:
            intent = entry["intent"]
            intents[intent] = intents.get(intent, 0) + 1
            
            focus = entry.get("focus_area")
            if focus:
                focus_areas[focus] = focus_areas.get(focus, 0) + 1
        
        return {
            "total_queries": total_queries,
            "intent_distribution": intents,
            "focus_usage": focus_areas,
            "avg_results": sum(e["num_results"] for e in self.query_history) / total_queries
        }
