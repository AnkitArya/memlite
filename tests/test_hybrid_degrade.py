"""A failing keyword leg must degrade hybrid recall, not void it.

Production symptom: one unparseable keyword query aborted ``Memory.search(
strategy="hybrid")`` *after* the semantic results had already been computed, so
prefetch returned an empty block and the turn silently lost all recalled
memories. Semantic-only is a perfectly good answer; throwing it away is not.

Fully offline: dummy embeddings, real sqlite-vec store, keyword leg patched to
raise the exact error seen in production.

Run: python -m pytest tests/test_hybrid_degrade.py -q
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Memory() builds an OpenAI client eagerly; a dummy key keeps the test offline
# (the embedder itself is stubbed below, so no request is ever made).
os.environ.setdefault("OPENAI_API_KEY", "test-not-a-real-key")

from memlite import Memory  # noqa: E402

DIMS = 8
FACT = "Deployment uses a dedicated git worktree off the playbook branch."


def _memory(tmp_path) -> Memory:
    mem = Memory(config={}, db_path=str(tmp_path / "hybrid.db"))
    # Stub the embedder: no network, deterministic 8-dim vectors.
    mem.embedder.embed = lambda text: [0.1] * DIMS
    mem.dims = DIMS
    mem._store_get().insert(FACT, [0.1] * DIMS, user_id="u")
    return mem


def test_hybrid_returns_semantic_hits_when_keyword_leg_raises(tmp_path):
    mem = _memory(tmp_path)
    store = mem._store_get()

    def _boom(*_args, **_kwargs):
        raise sqlite3.OperationalError('fts5: syntax error near "/"')

    store.keyword_search = _boom
    try:
        hits = mem.search("api/backtest", filters={"user_id": "u"},
                          top_k=3, strategy="hybrid")
    finally:
        mem.close()

    assert hits, "hybrid must fall back to the semantic results"
    assert any(FACT in h["memory"] for h in hits), hits


def test_hybrid_quoted_path_query_still_recalls(tmp_path):
    """End-to-end: the production query shape returns hits instead of erroring."""
    mem = _memory(tmp_path)
    try:
        hits = mem.search('the "api/backtest" endpoint', filters={"user_id": "u"},
                          top_k=3, strategy="hybrid")
    finally:
        mem.close()
    assert hits, "quoted-path recall returned nothing"
