"""
Authentication Service — Simple demo-grade auth for CT Orchestrator.

This is intentionally simple to demonstrate auth patterns:
- Hardcoded demo users (passwords hashed with bcrypt)
- Session-based login state
- Role-based access (admin vs viewer)

For production, replace with OAuth (Google, HF) or a proper identity provider.
"""

import hashlib
import secrets
from datetime import datetime
from typing import Optional


# Demo users — passwords are SHA-256 hashed for minimal security
# In production, use bcrypt + a real user database
DEMO_USERS = {
    "demo": {
        "password_hash": hashlib.sha256("demo2024".encode()).hexdigest(),
        "role": "viewer",
        "display_name": "Demo User",
    },
    "admin": {
        "password_hash": hashlib.sha256("admin2024".encode()).hexdigest(),
        "role": "admin",
        "display_name": "Admin",
    },
}

# Password hint shown on login page
PASSWORD_HINT = 'Hint: username + "2024"'


def authenticate(username: str, password: str) -> Optional[dict]:
    """
    Authenticate a user with username and password.
    
    Returns user info dict on success, None on failure.
    """
    username = username.strip().lower()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    user = DEMO_USERS.get(username)
    if user and user["password_hash"] == password_hash:
        return {
            "username": username,
            "role": user["role"],
            "display_name": user["display_name"],
            "session_id": _generate_session_id(),
            "login_time": datetime.now().isoformat(),
        }
    return None


def _generate_session_id() -> str:
    """Generate a unique session ID."""
    return secrets.token_hex(16)


def is_admin(user_info: dict) -> bool:
    """Check if the logged-in user has admin role."""
    return user_info.get("role") == "admin"


def get_login_page_config() -> dict:
    """Return configuration for the login page display."""
    return {
        "title": "🎬 CT Orchestrator",
        "subtitle": "AI-powered creative testing for media agencies",
        "hint": PASSWORD_HINT,
        "demo_note": (
            "This login demonstrates an authentication component. "
            "In production, this would connect to OAuth (Google, SSO) or an identity provider."
        ),
        "available_users": [
            {"username": "demo", "role": "Viewer — browse & chat"},
            {"username": "admin", "role": "Admin — full access + settings"},
        ],
    }
