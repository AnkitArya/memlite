"""FTS5 keyword leg must survive arbitrary user text.

Regression for the live production failure:

    MemLite prefetch failed: fts5: syntax error near "/"

A recall query that contains a double-quoted path (a normal thing for a user to
type: ``the "api/backtest" endpoint``) was split into terms and each term was
wrapped in double quotes *without* stripping embedded quotes, producing
``""api/backtest""``. FTS5 then closes the string literal early, parses
``api/backtest`` as bare query syntax and raises. Because the hybrid path calls
the keyword leg synchronously, that exception also discarded the semantic hits.

These tests are fully offline (dummy embeddings, no LLM, no network).

Run: python -m pytest tests/test_fts_quote_escape.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memlite.store import Store  # noqa: E402

DIMS = 8

BACKTEST_DOC = "Backtest endpoint POST /api/backtest/run rejects unsupported strategies."
WORKTREE_DOC = "Deployment uses a dedicated git worktree off the playbook branch."


def _store(tmp_path) -> Store:
    store = Store(str(tmp_path / "fts.db"), dims=DIMS)
    store.insert(BACKTEST_DOC, [0.1] * DIMS, user_id="u")
    store.insert(WORKTREE_DOC, [0.2] * DIMS, user_id="u")
    return store


# Every shape below used to raise sqlite3.OperationalError in keyword_search.
HOSTILE_QUERIES = [
    'the "api/backtest" endpoint',   # the exact production failure
    '"api/backtest"',
    '""',                            # doubled quote -> empty literal
    '"unbalanced',
    'balanced"',
    'a/b',
    r"C:\Users\x",
    "NIFTY50.NS",
    "agent/multi_agent.py",
    "--flag/with/slash",
    "NEAR(",
    "*",
    "(",
    "^",
    "-",
    "_",
    "a AND OR b",
    "```python/x```",
]


def test_keyword_search_never_raises_on_quoted_or_pathological_text(tmp_path):
    store = _store(tmp_path)
    try:
        for query in HOSTILE_QUERIES:
            rows = store.keyword_search(query, top_k=5)
            assert isinstance(rows, list), query
    finally:
        store.close()


def test_quoted_path_query_still_returns_the_matching_memory(tmp_path):
    """Escaping must not throw recall away — the tokens are still searchable."""
    store = _store(tmp_path)
    try:
        rows = store.keyword_search('the "api/backtest" endpoint', top_k=5)
        assert any("api/backtest" in r["memory"] for r in rows), rows
    finally:
        store.close()


def test_hyphen_and_underscore_still_split_into_separate_tokens(tmp_path):
    """Documented behaviour: hyphenated identifiers are searched token-by-token."""
    store = Store(str(tmp_path / "fts2.db"), dims=DIMS)
    try:
        store.insert("The india-trade-cli repo pushes to its own origin.",
                     [0.1] * DIMS, user_id="u")
        rows = store.keyword_search("india-trade-cli", top_k=5)
        assert any("india-trade-cli" in r["memory"] for r in rows), rows
    finally:
        store.close()
