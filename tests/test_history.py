"""
Tests for chat history persistence.
"""

import pytest
from pathlib import Path
from src.memory.history import ChatHistory


def test_create_session(tmp_path):
    """Test creating a new chat session."""
    db_path = tmp_path / "test_sessions.db"
    history = ChatHistory(str(db_path))
    
    session_id = history.create_session("/path/to/project")
    
    assert session_id > 0
    history.close()


def test_add_and_retrieve_messages(tmp_path):
    """Test adding and retrieving messages."""
    db_path = tmp_path / "test_sessions.db"
    history = ChatHistory(str(db_path))
    
    session_id = history.create_session("/path/to/project")
    
    # Add messages
    history.add_message(session_id, "user", "Hello")
    history.add_message(session_id, "assistant", "Hi there!")
    
    # Retrieve messages
    messages = history.get_session_messages(session_id)
    
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Hi there!"
    
    history.close()
