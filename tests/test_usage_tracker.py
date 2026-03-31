"""
Tests for the Usage Tracker Service.

Covers:
- Database initialization
- Session logging (login/logout)
- Query logging and counting
- Chat history persistence
- Admin analytics
"""

import pytest


class TestUsageTrackerInit:
    """Tests for database initialization."""

    def test_creates_database(self, tmp_db):
        """Should create the SQLite database file."""
        import os
        from services.usage_tracker import UsageTracker

        tracker = UsageTracker(tmp_db)
        assert os.path.exists(tmp_db)

    def test_creates_tables(self, tmp_db):
        """Should create all required tables."""
        import sqlite3
        from services.usage_tracker import UsageTracker

        tracker = UsageTracker(tmp_db)
        conn = sqlite3.connect(tmp_db)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "sessions" in tables
        assert "usage_log" in tables
        assert "chat_history" in tables


class TestSessionTracking:
    """Tests for login/logout tracking."""

    def test_log_login(self, tmp_db):
        """Should record a login event."""
        from services.usage_tracker import UsageTracker

        tracker = UsageTracker(tmp_db)
        tracker.log_login("session_123", "demo", "viewer")

        # Verify
        import sqlite3
        conn = sqlite3.connect(tmp_db)
        row = conn.execute("SELECT * FROM sessions WHERE session_id='session_123'").fetchone()
        conn.close()
        assert row is not None

    def test_log_logout(self, tmp_db):
        """Should update logout time."""
        from services.usage_tracker import UsageTracker

        tracker = UsageTracker(tmp_db)
        tracker.log_login("session_456", "admin", "admin")
        tracker.log_logout("session_456")

        import sqlite3
        conn = sqlite3.connect(tmp_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM sessions WHERE session_id='session_456'").fetchone()
        conn.close()
        assert row["logout_time"] is not None


class TestQueryTracking:
    """Tests for query logging."""

    def test_log_query(self, tmp_db):
        """Should record a query."""
        from services.usage_tracker import UsageTracker

        tracker = UsageTracker(tmp_db)
        tracker.log_login("sess_1", "demo", "viewer")
        tracker.log_query(
            session_id="sess_1",
            username="demo",
            action="agent_hub_query",
            query="What drives brand recall?",
            response_preview="Logo placement is key...",
            page="agent_hub",
        )

        count = tracker.get_session_query_count("sess_1")
        assert count == 1

    def test_multiple_queries_counted(self, tmp_db):
        """Session query count should increment."""
        from services.usage_tracker import UsageTracker

        tracker = UsageTracker(tmp_db)
        tracker.log_login("sess_2", "demo", "viewer")

        for i in range(5):
            tracker.log_query("sess_2", "demo", "query", query=f"Question {i}")

        count = tracker.get_session_query_count("sess_2")
        assert count == 5

    def test_query_truncation(self, tmp_db):
        """Long queries should be truncated."""
        from services.usage_tracker import UsageTracker

        tracker = UsageTracker(tmp_db)
        tracker.log_login("sess_3", "demo", "viewer")

        long_query = "x" * 1000
        tracker.log_query("sess_3", "demo", "query", query=long_query)

        import sqlite3
        conn = sqlite3.connect(tmp_db)
        row = conn.execute("SELECT query FROM usage_log").fetchone()
        conn.close()
        assert len(row[0]) == 500  # Truncated to 500 chars


class TestChatHistoryPersistence:
    """Tests for chat history save/load/clear."""

    def test_save_and_load(self, tmp_db):
        """Should save and load chat messages."""
        from services.usage_tracker import UsageTracker

        tracker = UsageTracker(tmp_db)
        tracker.save_chat_message("demo", "user", "Hello")
        tracker.save_chat_message("demo", "assistant", "Hi there!")

        messages = tracker.load_chat_history("demo")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"

    def test_load_respects_user_isolation(self, tmp_db):
        """User A should not see User B's messages."""
        from services.usage_tracker import UsageTracker

        tracker = UsageTracker(tmp_db)
        tracker.save_chat_message("alice", "user", "Alice's message")
        tracker.save_chat_message("bob", "user", "Bob's message")

        alice_msgs = tracker.load_chat_history("alice")
        assert len(alice_msgs) == 1
        assert alice_msgs[0]["content"] == "Alice's message"

    def test_clear_history(self, tmp_db):
        """Should clear chat history for a user."""
        from services.usage_tracker import UsageTracker

        tracker = UsageTracker(tmp_db)
        tracker.save_chat_message("demo", "user", "Hello")
        tracker.save_chat_message("demo", "assistant", "Hi!")

        tracker.clear_chat_history("demo")
        messages = tracker.load_chat_history("demo")
        assert len(messages) == 0

    def test_load_limit(self, tmp_db):
        """Should respect the limit parameter."""
        from services.usage_tracker import UsageTracker

        tracker = UsageTracker(tmp_db)
        for i in range(20):
            tracker.save_chat_message("demo", "user", f"Message {i}")

        messages = tracker.load_chat_history("demo", limit=5)
        assert len(messages) == 5

    def test_load_chronological_order(self, tmp_db):
        """Messages should be in chronological order."""
        from services.usage_tracker import UsageTracker

        tracker = UsageTracker(tmp_db)
        tracker.save_chat_message("demo", "user", "First")
        tracker.save_chat_message("demo", "assistant", "Second")
        tracker.save_chat_message("demo", "user", "Third")

        messages = tracker.load_chat_history("demo")
        assert messages[0]["content"] == "First"
        assert messages[1]["content"] == "Second"
        assert messages[2]["content"] == "Third"


class TestAdminAnalytics:
    """Tests for admin dashboard analytics."""

    def test_usage_stats_empty(self, tmp_db):
        """Empty database should return zero stats."""
        from services.usage_tracker import UsageTracker

        tracker = UsageTracker(tmp_db)
        stats = tracker.get_usage_stats()
        assert stats["total_queries"] == 0
        assert stats["unique_users"] == 0
        assert stats["total_sessions"] == 0

    def test_usage_stats_with_data(self, tmp_db):
        """Should aggregate stats correctly."""
        from services.usage_tracker import UsageTracker

        tracker = UsageTracker(tmp_db)
        tracker.log_login("s1", "alice", "viewer")
        tracker.log_login("s2", "bob", "viewer")
        tracker.log_query("s1", "alice", "query", query="Q1")
        tracker.log_query("s1", "alice", "query", query="Q2")
        tracker.log_query("s2", "bob", "query", query="Q3")

        stats = tracker.get_usage_stats()
        assert stats["total_queries"] == 3
        assert stats["unique_users"] == 2
        assert stats["total_sessions"] == 2
        assert len(stats["top_users"]) == 2
        assert len(stats["recent_queries"]) == 3
