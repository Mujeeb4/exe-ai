"""
ADK Runner implementation for Exe.
Manages agent execution, sessions, and event streaming.
"""

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, DatabaseSessionService
from google.genai import types
from typing import AsyncGenerator, Optional, Dict
from pathlib import Path
import time
from .coordinator_agent import create_coordinator_agent, create_simple_agent
from ..config import ExeConfig


class ExeRunner:
    """
    ADK Runner wrapper for Exe with session management and event streaming.
    
    This class provides:
    - Session lifecycle management
    - Event streaming for real-time feedback
    - Tool execution orchestration
    - Error handling and recovery
    """
    
    def __init__(
        self,
        config: ExeConfig,
        repo_context: Optional[str] = None,
        use_database: bool = True,
        use_simple_agent: bool = False
    ):
        """
        Initialize the Exe runner.
        
        Args:
            config: Exe configuration
            repo_context: Optional repository context/overview
            use_database: Whether to use persistent database for sessions
            use_simple_agent: Use simple single-agent instead of coordinator
        """
        self.config = config
        self.repo_context = repo_context
        
        # Create session service
        if use_database:
            db_path = Path.home() / ".exe" / "adk_sessions.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.session_service = DatabaseSessionService(
                db_url=f"sqlite:///{db_path}"
            )
        else:
            self.session_service = InMemorySessionService()
        
        # Create agent (coordinator or simple)
        if use_simple_agent:
            self.agent = create_simple_agent(config, repo_context)
        else:
            self.agent = create_coordinator_agent(config, repo_context)
        
        # Create ADK runner
        self.runner = Runner(
            agent=self.agent,
            app_name="exe",
            session_service=self.session_service
        )
    
    async def create_session(
        self,
        user_id: str,
        session_id: str,
        initial_state: Optional[Dict] = None
    ):
        """
        Create a new session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            initial_state: Optional initial session state
            
        Returns:
            Created session object
        """
        # Build initial state
        state = {
            "focus_path": self.config.focus_path,
            "project_root": str(Path.cwd()),
            "api_key": self.config.api_key,
            "modified_files": [],
            "conversation_turn": 0,
            **(initial_state or {})
        }
        
        return await self.session_service.create_session(
            app_name="exe",
            user_id=user_id,
            session_id=session_id,
            state=state
        )
    
    async def get_session(self, user_id: str, session_id: str):
        """
        Get an existing session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Session object or None if not found
        """
        try:
            return await self.session_service.get_session(
                app_name="exe",
                user_id=user_id,
                session_id=session_id
            )
        except Exception:
            return None
    
    async def query(
        self,
        user_id: str,
        session_id: str,
        query: str,
        stream: bool = True
    ) -> AsyncGenerator:
        """
        Execute a query and stream events.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            query: User's query text
            stream: Whether to stream events (vs. wait for final response)
            
        Yields:
            Event objects with real-time updates
        """
        # Create message content
        content = types.Content(
            role='user',
            parts=[types.Part(text=query)]
        )
        
        # Execute and stream events
        async for event in self.runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content
        ):
            yield event
    
    async def query_simple(
        self,
        user_id: str,
        session_id: str,
        query: str
    ) -> str:
        """
        Execute a query and return only the final response.
        
        Useful for non-interactive scenarios or testing.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            query: User's query text
            
        Returns:
            Final response text
        """
        final_response = ""
        
        async for event in self.query(user_id, session_id, query):
            if event.is_final_response():
                final_response = event.content.parts[0].text
                break
        
        return final_response
    
    async def update_focus(
        self,
        user_id: str,
        session_id: str,
        focus_path: Optional[str]
    ):
        """
        Update the focus path for a session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            focus_path: New focus path (None to clear)
        """
        session = await self.get_session(user_id, session_id)
        if session:
            session.state["focus_path"] = focus_path
            await self.session_service.update_session(session)
    
    async def get_modified_files(
        self,
        user_id: str,
        session_id: str
    ) -> list:
        """
        Get list of files modified in this session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            List of modified file paths
        """
        session = await self.get_session(user_id, session_id)
        if session:
            return session.state.get("modified_files", [])
        return []
    
    async def close_session(self, user_id: str, session_id: str):
        """
        Close and cleanup a session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
        """
        session = await self.get_session(user_id, session_id)
        if session:
            await self.session_service.close_session(session=session)


class EventCollector:
    """
    Helper class to collect and categorize events from ADK runner.
    
    Useful for testing and debugging.
    """
    
    def __init__(self):
        self.events = []
        self.tool_calls = []
        self.responses = []
        self.errors = []
    
    def add_event(self, event):
        """Add an event to the collector."""
        self.events.append(event)
        
        # Categorize event
        if hasattr(event, 'type'):
            if event.type == "tool_call":
                self.tool_calls.append({
                    "tool": event.tool_name,
                    "args": event.tool_args,
                    "timestamp": time.time()
                })
            elif event.type == "error":
                self.errors.append({
                    "message": str(event.content),
                    "timestamp": time.time()
                })
        
        if event.is_final_response():
            self.responses.append({
                "content": event.content.parts[0].text,
                "timestamp": time.time()
            })
    
    def get_summary(self) -> Dict:
        """Get a summary of collected events."""
        return {
            "total_events": len(self.events),
            "tool_calls": len(self.tool_calls),
            "responses": len(self.responses),
            "errors": len(self.errors),
            "tools_used": list(set(tc["tool"] for tc in self.tool_calls))
        }
    
    def get_final_response(self) -> Optional[str]:
        """Get the final response text."""
        if self.responses:
            return self.responses[-1]["content"]
        return None


async def create_and_run_simple(
    config: ExeConfig,
    query: str,
    repo_context: Optional[str] = None
) -> str:
    """
    Convenience function for simple one-off queries.
    
    Args:
        config: Exe configuration
        query: Query text
        repo_context: Optional repository context
        
    Returns:
        Response text
    """
    runner = ExeRunner(config, repo_context, use_database=False)
    
    user_id = "temp_user"
    session_id = f"temp_{int(time.time())}"
    
    await runner.create_session(user_id, session_id)
    response = await runner.query_simple(user_id, session_id, query)
    
    return response
