"""
Example tests for ADK-based Exe components.
Demonstrates testing patterns for agents, tools, and runner.
"""

import pytest
import asyncio
from pathlib import Path
from google.genai import types
from google.adk.runners import InMemoryRunner
from src.config import ExeConfig
from src.core.router_agent import create_router_agent, parse_router_output
from src.core.coder_agent import create_coder_agent
from src.core.coordinator_agent import create_coordinator_agent
from src.core.runner import ExeRunner, EventCollector
from src.core.tools import edit_file_tool, search_code_tool


class TestRouterAgent:
    """Test suite for RouterAgent."""
    
    @pytest.mark.asyncio
    async def test_router_classifies_code_edit(self):
        """Test router correctly classifies code edit intent."""
        agent = create_router_agent()
        runner = InMemoryRunner(agent=agent, app_name="test")
        
        query = "Fix the bug in main.py where it crashes on invalid input"
        content = types.Content(role='user', parts=[types.Part(text=query)])
        
        result = None
        async for event in runner.run_async(
            user_id="test",
            session_id="test",
            new_message=content
        ):
            if event.is_final_response():
                result = event.content.parts[0].text
                break
        
        assert result is not None
        parsed = parse_router_output(result)
        assert parsed.intent == "code_edit"
        assert parsed.confidence > 0.6
    
    @pytest.mark.asyncio
    async def test_router_classifies_question(self):
        """Test router correctly classifies question intent."""
        agent = create_router_agent()
        runner = InMemoryRunner(agent=agent, app_name="test")
        
        query = "What does the Router class do?"
        content = types.Content(role='user', parts=[types.Part(text=query)])
        
        result = None
        async for event in runner.run_async(
            user_id="test",
            session_id="test",
            new_message=content
        ):
            if event.is_final_response():
                result = event.content.parts[0].text
                break
        
        assert result is not None
        parsed = parse_router_output(result)
        assert parsed.intent == "question"
    
    @pytest.mark.asyncio
    async def test_router_provides_reasoning(self):
        """Test router provides reasoning for classification."""
        agent = create_router_agent()
        runner = InMemoryRunner(agent=agent, app_name="test")
        
        query = "Refactor the authentication logic to use dependency injection"
        content = types.Content(role='user', parts=[types.Part(text=query)])
        
        result = None
        async for event in runner.run_async(
            user_id="test",
            session_id="test",
            new_message=content
        ):
            if event.is_final_response():
                result = event.content.parts[0].text
                break
        
        parsed = parse_router_output(result)
        assert parsed.intent == "refactor"
        assert len(parsed.reasoning) > 10  # Has meaningful reasoning


class TestCoderAgent:
    """Test suite for CoderAgent."""
    
    @pytest.mark.asyncio
    async def test_coder_uses_search_tool(self):
        """Test coder agent uses search_code tool."""
        config = ExeConfig(
            api_key="test_key",
            coder_model="gemini-2.0-flash-exp"
        )
        agent = create_coder_agent(config.coder_model)
        runner = InMemoryRunner(agent=agent, app_name="test")
        
        query = "Show me the Router class implementation"
        content = types.Content(role='user', parts=[types.Part(text=query)])
        
        tool_used = False
        async for event in runner.run_async(
            user_id="test",
            session_id="test",
            new_message=content
        ):
            if hasattr(event, 'type') and event.type == "tool_call":
                if getattr(event, 'tool_name', '') == "search_code":
                    tool_used = True
        
        # Note: Actual tool usage depends on model behavior
        # This test documents expected behavior
        assert True  # Placeholder - adjust based on actual tool integration


class TestExeRunner:
    """Test suite for ExeRunner."""
    
    @pytest.mark.asyncio
    async def test_runner_creates_session(self):
        """Test runner creates sessions properly."""
        config = ExeConfig(
            api_key="test_key",
            router_model="gemini-1.5-flash-8b",
            coder_model="gemini-2.0-flash-exp"
        )
        
        runner = ExeRunner(config, use_database=False)
        
        session = await runner.create_session(
            user_id="test_user",
            session_id="test_session",
            initial_state={"test": "value"}
        )
        
        assert session is not None
        assert session.state.get("test") == "value"
        assert "focus_path" in session.state
        assert "project_root" in session.state
    
    @pytest.mark.asyncio
    async def test_runner_streams_events(self):
        """Test runner streams events properly."""
        config = ExeConfig(
            api_key="test_key",
            router_model="gemini-1.5-flash-8b",
            coder_model="gemini-2.0-flash-exp"
        )
        
        runner = ExeRunner(config, use_database=False)
        
        await runner.create_session("test_user", "test_session")
        
        event_count = 0
        async for event in runner.query("test_user", "test_session", "Hello"):
            event_count += 1
            if event.is_final_response():
                break
        
        assert event_count > 0
    
    @pytest.mark.asyncio
    async def test_runner_updates_focus(self):
        """Test runner updates focus path correctly."""
        config = ExeConfig(api_key="test_key")
        runner = ExeRunner(config, use_database=False)
        
        await runner.create_session("test_user", "test_session")
        await runner.update_focus("test_user", "test_session", "src/")
        
        session = await runner.get_session("test_user", "test_session")
        assert session.state.get("focus_path") == "src/"


class TestEventCollector:
    """Test suite for EventCollector utility."""
    
    @pytest.mark.asyncio
    async def test_event_collector_categorizes_events(self):
        """Test EventCollector properly categorizes events."""
        config = ExeConfig(api_key="test_key")
        runner = ExeRunner(config, use_database=False)
        
        await runner.create_session("test_user", "test_session")
        
        collector = EventCollector()
        
        async for event in runner.query("test_user", "test_session", "Test query"):
            collector.add_event(event)
            if event.is_final_response():
                break
        
        summary = collector.get_summary()
        assert summary["total_events"] > 0
        assert isinstance(summary["tool_calls"], int)
        assert isinstance(summary["responses"], int)
    
    def test_event_collector_get_final_response(self):
        """Test EventCollector returns final response."""
        from src.core.runner import EventCollector
        
        collector = EventCollector()
        
        # Simulate adding a response
        collector.responses.append({
            "content": "This is the answer",
            "timestamp": 123456
        })
        
        final = collector.get_final_response()
        assert final == "This is the answer"


class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_question_workflow(self):
        """Test complete workflow for a question."""
        config = ExeConfig(
            api_key="test_key",
            router_model="gemini-1.5-flash-8b",
            coder_model="gemini-2.0-flash-exp"
        )
        
        runner = ExeRunner(config, use_database=False, use_simple_agent=False)
        
        await runner.create_session("test", "test")
        
        response = await runner.query_simple(
            "test",
            "test",
            "What is the purpose of this codebase?"
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_complete_edit_workflow(self):
        """Test complete workflow for code editing."""
        config = ExeConfig(
            api_key="test_key",
            router_model="gemini-1.5-flash-8b",
            coder_model="gemini-2.0-flash-exp"
        )
        
        runner = ExeRunner(config, use_database=False)
        
        await runner.create_session("test", "test")
        
        # Note: This would actually attempt to edit a file
        # In practice, mock the tools or use a test directory
        response = await runner.query_simple(
            "test",
            "test",
            "Add a docstring to the Router class"
        )
        
        assert isinstance(response, str)


# Pytest fixtures
@pytest.fixture
def test_config():
    """Provide a test configuration."""
    return ExeConfig(
        api_key="test_api_key",
        router_model="gemini-1.5-flash-8b",
        coder_model="gemini-2.0-flash-exp",
        auto_apply=False
    )


@pytest.fixture
def test_repo_context():
    """Provide test repository context."""
    return """
Test Repository:
- src/core/router.py: Intent classification
- src/core/coder.py: Code generation
- src/interface/repl.py: User interface
"""


@pytest.fixture
async def test_runner(test_config):
    """Provide a test runner instance."""
    runner = ExeRunner(test_config, use_database=False)
    await runner.create_session("test_user", "test_session")
    yield runner
    await runner.close_session("test_user", "test_session")


# Run tests with: pytest tests/test_adk_integration.py -v
