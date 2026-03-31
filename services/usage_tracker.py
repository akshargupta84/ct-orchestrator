"""
Usage Tracker — SQLite-backed tracking for queries, sessions, and chat history.

Tracks:
- Login events and session duration
- Per-user query counts and content
- API token usage and estimated cost
- Chat history persistence across sessions
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


# Default DB path
_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "usage.db"
)

# Singleton instance
_tracker_instance = None


def get_tracker(db_path: str = None) -> "UsageTracker":
    """Get or create the singleton UsageTracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = UsageTracker(db_path or _DEFAULT_DB_PATH)
    return _tracker_instance


class UsageTracker:
    """SQLite-backed usage tracking and chat history persistence."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_conn(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    username TEXT NOT NULL,
                    role TEXT NOT NULL,
                    login_time TEXT NOT NULL,
                    logout_time TEXT,
                    query_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    action TEXT NOT NULL,
                    query TEXT,
                    response_preview TEXT,
                    tokens_used INTEGER DEFAULT 0,
                    estimated_cost REAL DEFAULT 0.0,
                    timestamp TEXT NOT NULL,
                    page TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    page TEXT NOT NULL DEFAULT 'agent_hub',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_usage_username ON usage_log(username);
                CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_chat_username ON chat_history(username);
                CREATE INDEX IF NOT EXISTS idx_sessions_username ON sessions(username);
            """)

    # =========================================================================
    # Session Tracking
    # =========================================================================

    def log_login(self, session_id: str, username: str, role: str):
        """Record a login event."""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, username, role, login_time) VALUES (?, ?, ?, ?)",
                (session_id, username, role, datetime.now().isoformat()),
            )

    def log_logout(self, session_id: str):
        """Record a logout event."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE sessions SET logout_time = ? WHERE session_id = ?",
                (datetime.now().isoformat(), session_id),
            )

    # =========================================================================
    # Query / Action Tracking
    # =========================================================================

    def log_query(
        self,
        session_id: str,
        username: str,
        action: str,
        query: str = "",
        response_preview: str = "",
        tokens_used: int = 0,
        page: str = "",
    ):
        """Log a user action (query, file upload, plan generation, etc.)."""
        # Estimate cost: ~$3/M input tokens, ~$15/M output tokens (Sonnet pricing)
        estimated_cost = (tokens_used / 1_000_000) * 9  # rough average

        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO usage_log 
                   (session_id, username, action, query, response_preview, tokens_used, estimated_cost, timestamp, page)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    username,
                    action,
                    query[:500],  # Truncate long queries
                    response_preview[:200],  # Preview only
                    tokens_used,
                    estimated_cost,
                    datetime.now().isoformat(),
                    page,
                ),
            )
            # Increment session query count
            conn.execute(
                "UPDATE sessions SET query_count = query_count + 1 WHERE session_id = ?",
                (session_id,),
            )

    def get_session_query_count(self, session_id: str) -> int:
        """Get query count for the current session."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT query_count FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row["query_count"] if row else 0

    # =========================================================================
    # Chat History Persistence
    # =========================================================================

    def save_chat_message(self, username: str, role: str, content: str, page: str = "agent_hub"):
        """Save a chat message for a user."""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO chat_history (username, page, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
                (username, page, role, content, datetime.now().isoformat()),
            )

    def load_chat_history(self, username: str, page: str = "agent_hub", limit: int = 50) -> list:
        """Load chat history for a user, most recent messages."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT role, content, timestamp FROM chat_history 
                   WHERE username = ? AND page = ? 
                   ORDER BY id DESC LIMIT ?""",
                (username, page, limit),
            ).fetchall()
            # Reverse to get chronological order
            messages = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
            return messages

    def clear_chat_history(self, username: str, page: str = "agent_hub"):
        """Clear chat history for a user."""
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM chat_history WHERE username = ? AND page = ?",
                (username, page),
            )

    # =========================================================================
    # Admin Analytics
    # =========================================================================

    def get_usage_stats(self) -> dict:
        """Get aggregated usage statistics for admin dashboard."""
        with self._get_conn() as conn:
            # Total queries
            total = conn.execute("SELECT COUNT(*) as cnt FROM usage_log").fetchone()["cnt"]

            # Unique users
            users = conn.execute(
                "SELECT COUNT(DISTINCT username) as cnt FROM usage_log"
            ).fetchone()["cnt"]

            # Total sessions
            sessions = conn.execute("SELECT COUNT(*) as cnt FROM sessions").fetchone()["cnt"]

            # Total tokens and cost
            tokens_row = conn.execute(
                "SELECT COALESCE(SUM(tokens_used), 0) as tokens, COALESCE(SUM(estimated_cost), 0) as cost FROM usage_log"
            ).fetchone()

            # Queries per day (last 7 days)
            daily = conn.execute(
                """SELECT DATE(timestamp) as day, COUNT(*) as cnt 
                   FROM usage_log 
                   WHERE timestamp >= DATE('now', '-7 days')
                   GROUP BY DATE(timestamp) ORDER BY day"""
            ).fetchall()

            # Top users
            top_users = conn.execute(
                """SELECT username, COUNT(*) as queries, SUM(tokens_used) as tokens
                   FROM usage_log GROUP BY username ORDER BY queries DESC LIMIT 10"""
            ).fetchall()

            # Recent queries
            recent = conn.execute(
                """SELECT username, action, query, timestamp 
                   FROM usage_log ORDER BY id DESC LIMIT 20"""
            ).fetchall()

            return {
                "total_queries": total,
                "unique_users": users,
                "total_sessions": sessions,
                "total_tokens": tokens_row["tokens"],
                "total_cost": tokens_row["cost"],
                "daily_queries": [{"day": r["day"], "count": r["cnt"]} for r in daily],
                "top_users": [
                    {"username": r["username"], "queries": r["queries"], "tokens": r["tokens"] or 0}
                    for r in top_users
                ],
                "recent_queries": [
                    {"username": r["username"], "action": r["action"], "query": r["query"][:80], "timestamp": r["timestamp"]}
                    for r in recent
                ],
            }
