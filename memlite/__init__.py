"""memlite - a lean, single-file SQLite + vector semantic memory for AI agents.

Five-method API (add / search / get_all / update / delete) with a radically
simpler backend: one SQLite DB + the `sqlite-vec` extension for semantic search
+ FTS5 for keyword fallback. No vector server, no separate vector process, no single-client lock.
"""
from .core import Memory

__all__ = ["Memory"]
__version__ = "1.0.1"
