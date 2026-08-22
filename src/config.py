"""Application configuration loaded from environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# --- Paths ---
KNOWLEDGE_BASE_DIR = _PROJECT_ROOT / "knowledge-base"
ORDERS_JSON_PATH = _PROJECT_ROOT / "data" / "orders.json"
ORDERS_DICT_PATH = _PROJECT_ROOT / "data" / "orders-data-dictionary.md"
INDEX_CACHE_DIR = _PROJECT_ROOT / "src" / "index_cache"
EVAL_CASES_PATH = _PROJECT_ROOT / "evaluation" / "visible-cases.json"

# --- LLM Provider ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
LLM_API_KEY = os.getenv("GEMINI_API_KEY") if LLM_PROVIDER == "gemini" else os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite") if LLM_PROVIDER == "gemini" else os.getenv("OPENAI_MODEL", "gpt-4o")

# --- Embedding Provider ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001") if EMBEDDING_PROVIDER == "gemini" else os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# --- Retrieval ---
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))

# --- Observability ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEBUG_TRACE = os.getenv("DEBUG_TRACE", "false").lower() == "true"

# --- Snapshot time (from orders.json, used for cancellation window) ---
SNAPSHOT_AT = "2026-08-15T12:00:00Z"


def validate_config():
    """Validate that required configuration is present.
    Returns list of errors (empty = valid)."""
    errors = []
    if not KNOWLEDGE_BASE_DIR.exists():
        errors.append(f"Knowledge base directory not found: {KNOWLEDGE_BASE_DIR}")
    if not ORDERS_JSON_PATH.exists():
        errors.append(f"Orders file not found: {ORDERS_JSON_PATH}")
    # API key is only required when actually making LLM calls
    return errors
