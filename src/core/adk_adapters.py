"""
Adapter classes to make ADK-based agents compatible with existing REPL interface.
These wrappers provide the same API as the simple Router/Coder classes.
"""

import asyncio
from typing import List, Optional, Dict
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from .router_agent import create_router_agent, parse_router_output
from .coder_agent import create_coder_agent
from .models import RouterOutput, CoderOutput, CodeChunk
from ..memory.db import VectorDB
from ..memory.embedder import Embedder


def _run_agent_sync(runner: InMemoryRunner, query: str, user_id: str = "default", session_id: str = "default") -> str:
    """
    Synchronous wrapper for ADK runner.run_async().
    Extracts the final response text from agent events.
    """
    async def _run():
        # Ensure session exists before running
        session_service = runner.session_service
        try:
            # Try to get existing session
            session = await session_service.get_session(
                app_name=runner.app_name,
                user_id=user_id,
                session_id=session_id
            )
        except Exception:
            session = None
        
        # Create session if it doesn't exist
        if session is None:
            session = await session_service.create_session(
                app_name=runner.app_name,
                user_id=user_id,
                session_id=session_id
            )
        
        content = types.Content(role='user', parts=[types.Part(text=query)])
        result_text = ""
        
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content
        ):
            if event.is_final_response():
                if hasattr(event.content, 'parts') and event.content.parts:
                    result_text = event.content.parts[0].text
                    break
        
        return result_text if result_text else "No response generated."
    
    # Run the async function in the event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new loop
            import nest_asyncio
            try:
                nest_asyncio.apply()
            except Exception:
                pass
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(_run())


class RouterAdapter:
    """Adapter to make ADK router_agent compatible with Router interface."""
    
    def __init__(self, api_key: str, focus_path: Optional[str] = None,
                 repo_context: Optional[str] = None, model: str = "gemini-1.5-flash-8b"):
        self.api_key = api_key
        self.focus_path = focus_path
        self.repo_context = repo_context
        self.model = model
        self.agent = create_router_agent(model=model)
        self.runner = InMemoryRunner(agent=self.agent, app_name="exe")
        
    def route(self, query: str, db: VectorDB, embedder: Embedder) -> RouterOutput:
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
            # Run the router agent
            result_text = _run_agent_sync(self.runner, query)
            
            # Parse the structured output
            router_output = parse_router_output(result_text)
            
            # Embed the query and search for relevant chunks
            query_embedding = embedder.embed_single(query)
            search_limit = 10 if self.focus_path else 15
            chunks = db.search(query_embedding, limit=search_limit, focus_path=self.focus_path)
            
            # Extract file paths
            relevant_files = list(set([chunk.file_path for chunk in chunks]))
            
            # Return RouterOutput with chunks
            return RouterOutput(
                intent=router_output.intent,
                relevant_files=relevant_files,
                focus_area=self.focus_path,
                relevant_chunks=chunks
            )
            
        except Exception as e:
            print(f"Router error: {e}")
            # Fallback to basic routing
            query_embedding = embedder.embed_single(query)
            chunks = db.search(query_embedding, limit=10, focus_path=self.focus_path)
            relevant_files = list(set([chunk.file_path for chunk in chunks]))
            
            return RouterOutput(
                intent="question",
                relevant_files=relevant_files,
                focus_area=self.focus_path,
                relevant_chunks=chunks
            )


class CoderAdapter:
    """Adapter to make ADK coder_agent compatible with Coder interface."""
    
    def __init__(self, api_key: str, repo_context: Optional[str] = None,
                 model: str = "gemini-2.0-flash-exp"):
        self.api_key = api_key
        self.repo_context = repo_context
        self.model = model
        self.agent = create_coder_agent(model=model, repo_context=repo_context)
        self.runner = InMemoryRunner(agent=self.agent, app_name="exe")
        
    def process(self, query: str, router_output: RouterOutput, chunks: List[CodeChunk],
                conversation_history: Optional[List[Dict[str, str]]] = None) -> CoderOutput:
        """
        Generate a response based on router output and retrieved context.
        
        Args:
            query: User's input query
            router_output: Output from Router agent
            chunks: Relevant code chunks from vector DB
            conversation_history: Optional recent conversation messages
            
        Returns:
            CoderOutput with answer or patch
        """
        try:
            # Build context message from chunks
            context_parts = []
            if chunks:
                context_parts.append("Relevant code context:")
                for chunk in chunks[:5]:  # Limit to avoid token overflow
                    context_parts.append(f"\n# {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
                    context_parts.append(chunk.content)
            
            # Add conversation history if provided
            if conversation_history:
                context_parts.append("\nRecent conversation:")
                for msg in conversation_history[-4:]:  # Last 2 turns
                    context_parts.append(f"{msg['role'].upper()}: {msg['content'][:200]}")
            
            context_message = "\n".join(context_parts)
            
            # Build full query with context
            full_query = f"{context_message}\n\nUser request: {query}"
            
            # Run the coder agent synchronously
            response_text = _run_agent_sync(self.runner, full_query)
            
            # Determine if response is a patch or answer
            is_patch = (
                router_output.intent in ["code_edit", "refactor"] or
                "---" in response_text and "+++" in response_text or
                "@@" in response_text
            )
            
            if is_patch:
                # Clean up patch content
                patch_content = response_text
                if "```diff" in patch_content:
                    patch_content = patch_content.replace("```diff", "").replace("```", "").strip()
                elif "```" in patch_content:
                    patch_content = patch_content.replace("```", "").strip()
                
                return CoderOutput(
                    type="patch",
                    content=patch_content,
                    files_to_modify=router_output.relevant_files
                )
            else:
                return CoderOutput(
                    type="answer",
                    content=response_text
                )
                
        except Exception as e:
            print(f"Coder error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback response
            return CoderOutput(
                type="answer",
                content=f"I encountered an error processing your request: {str(e)}"
            )
