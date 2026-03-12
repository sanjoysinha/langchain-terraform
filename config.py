"""Centralized configuration for guardrails, roles, timeouts, and limits."""

# --- Role-Based Access Control ---
ROLES = {
    "admin": {
        "label": "Admin",
        "allowed_tools": ["document_search", "document_summarize", "web_search"],
    },
    "user": {
        "label": "User",
        "allowed_tools": ["document_search", "document_summarize"],
    },
    "viewer": {
        "label": "Viewer",
        "allowed_tools": ["document_search"],
    },
}

DEFAULT_ROLE = "user"

# --- Input Validation Limits ---
USER_INPUT_MAX_LENGTH = 2000
USER_INPUT_MIN_LENGTH = 1
TOOL_QUERY_MAX_LENGTH = 500
TOOL_QUERY_MIN_LENGTH = 1
TOOL_TOPIC_MAX_LENGTH = 500
TOOL_TOPIC_MIN_LENGTH = 1

# --- Agent Iteration Limits ---
MAX_AGENT_ITERATIONS = 5

# --- Timeout Configuration (seconds) ---
TOOL_TIMEOUTS = {
    "document_search": 30,
    "document_summarize": 60,
    "web_search": 30,
}
DEFAULT_TOOL_TIMEOUT = 30

# --- Output Limits ---
MAX_OUTPUT_LENGTH = 50000
MAX_TOOL_RESULT_LENGTH = 20000
