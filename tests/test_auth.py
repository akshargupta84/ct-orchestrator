"""
Tests for the Authentication Service.

Covers:
- Successful authentication (both demo accounts)
- Failed authentication (wrong password, unknown user)
- Role assignment
- Session ID generation
- Login page configuration
"""

import pytest


class TestAuthentication:
    """Tests for the authenticate() function."""

    def test_demo_user_login(self):
        """demo/demo2024 should authenticate successfully."""
        from services.auth import authenticate

        result = authenticate("demo", "demo2024")
        assert result is not None
        assert result["username"] == "demo"
        assert result["role"] == "viewer"
        assert result["display_name"] == "Demo User"

    def test_admin_user_login(self):
        """admin/admin2024 should authenticate successfully."""
        from services.auth import authenticate

        result = authenticate("admin", "admin2024")
        assert result is not None
        assert result["username"] == "admin"
        assert result["role"] == "admin"
        assert result["display_name"] == "Admin"

    def test_wrong_password(self):
        """Wrong password should return None."""
        from services.auth import authenticate

        result = authenticate("demo", "wrongpassword")
        assert result is None

    def test_unknown_user(self):
        """Unknown username should return None."""
        from services.auth import authenticate

        result = authenticate("nonexistent", "demo2024")
        assert result is None

    def test_empty_credentials(self):
        """Empty credentials should return None."""
        from services.auth import authenticate

        assert authenticate("", "") is None
        assert authenticate("demo", "") is None
        assert authenticate("", "demo2024") is None

    def test_case_insensitive_username(self):
        """Username should be case-insensitive."""
        from services.auth import authenticate

        result = authenticate("Demo", "demo2024")
        assert result is not None
        assert result["username"] == "demo"

    def test_session_id_generated(self):
        """Successful login should include a unique session ID."""
        from services.auth import authenticate

        result1 = authenticate("demo", "demo2024")
        result2 = authenticate("demo", "demo2024")
        assert result1["session_id"] is not None
        assert result2["session_id"] is not None
        assert result1["session_id"] != result2["session_id"]  # Should be unique

    def test_login_time_included(self):
        """Successful login should include a timestamp."""
        from services.auth import authenticate

        result = authenticate("demo", "demo2024")
        assert "login_time" in result
        assert len(result["login_time"]) > 0


class TestRoleChecking:
    """Tests for role-related functions."""

    def test_is_admin_true(self):
        """Admin user should return True for is_admin()."""
        from services.auth import authenticate, is_admin

        user = authenticate("admin", "admin2024")
        assert is_admin(user) is True

    def test_is_admin_false(self):
        """Demo user should return False for is_admin()."""
        from services.auth import authenticate, is_admin

        user = authenticate("demo", "demo2024")
        assert is_admin(user) is False

    def test_is_admin_empty_dict(self):
        """Empty dict should return False for is_admin()."""
        from services.auth import is_admin

        assert is_admin({}) is False


class TestLoginPageConfig:
    """Tests for login page configuration."""

    def test_config_has_required_fields(self):
        """Login config should have all required fields."""
        from services.auth import get_login_page_config

        config = get_login_page_config()
        assert "title" in config
        assert "subtitle" in config
        assert "hint" in config
        assert "demo_note" in config
        assert "available_users" in config

    def test_config_lists_demo_accounts(self):
        """Config should list available demo accounts."""
        from services.auth import get_login_page_config

        config = get_login_page_config()
        usernames = [u["username"] for u in config["available_users"]]
        assert "demo" in usernames
        assert "admin" in usernames
