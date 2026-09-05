"""memlite - a lean, single-file SQLite + vector semantic memory for AI agents.

Fork of mem0's design (add / search / get_all / update / delete) with a radically
simpler backend: one SQLite DB + the `sqlite-vec` extension for semantic search
+ FTS5 for keyword fallback. No Qdrant, no separate vector process, no single-client lock.
"""
from .core import Memory

__all__ = ["Memory"]
__version__ = "0.1.0"
