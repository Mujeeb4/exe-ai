"""
SQLite wrapper for chat history persistence.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class ChatHistory:
    """Manages chat session history using SQLite."""
    
    def __init__(self, db_path: str = ".exo/sessions.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        self.conn.commit()
    
    def create_session(self, project_path: str) -> int:
        """
        Create a new chat session.
        
        Args:
            project_path: Path to the project
            
        Returns:
            Session ID
        """
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO sessions (project_path) VALUES (?)", (project_path,))
        self.conn.commit()
        return cursor.lastrowid or 0
    
    def add_message(self, session_id: int, role: str, content: str):
        """
        Add a message to a session.
        
        Args:
            session_id: ID of the session
            role: "user" or "assistant"
            content: Message content
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        self.conn.commit()
    
    def get_session_messages(self, session_id: int, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Retrieve all messages for a session.
        
        Args:
            session_id: ID of the session
            limit: Optional limit on number of recent messages to retrieve
            
        Returns:
            List of message dictionaries
        """
        cursor = self.conn.cursor()
        
        if limit:
            cursor.execute(
                "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit)
            )
            messages = []
            for row in reversed(cursor.fetchall()):
                messages.append({
                    "role": row[0],
                    "content": row[1],
                    "timestamp": row[2]
                })
        else:
            cursor.execute(
                "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            )
            messages = []
            for row in cursor.fetchall():
                messages.append({
                    "role": row[0],
                    "content": row[1],
                    "timestamp": row[2]
                })
        
        return messages
    
    def close(self):
        """Close database connection."""
        self.conn.close()
