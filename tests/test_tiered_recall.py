"""Tests for memlite tiered recall (L0 abstract / L1 overview) + retention policy,
ported as concepts from OpenViking's context layers (reimplemented, not copied).

- Tiered summaries: memories carry a one-line `abstract` (L0) and a 1-2 sentence
  `overview` (L1); `get_full()` returns the full L2 record.
- Retention policy: `retention="selective"` SKIPs near-duplicate paraphrases so
  the store doesn't bloat with repeated facts (OpenViking "merging or skipping").

Store-level tests are hermetic (pure SQLite, no network); `_decide` tests pin the
pure decision rule.
"""
import os
import shutil
import sqlite3
import sys

# Memory({}) constructs an OpenAI client; construction doesn't hit the network,
# but it needs a present (dummy) key in hermetic tests.
os.environ.setdefault("OPENAI_API_KEY", "test-dummy")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memlite.store import Store, _SCHEMA_MEMORIES, _SCHEMA_VEC, _SCHEMA_FTS, _SCHEMA_HISTORY
from memlite.core import Memory

DB = "/tmp/memlite_tier_test.db"


def build_store() -> Store:
    if os.path.exists(DB):
        os.remove(DB)
    return Store(DB, dims=4)


def _cand(text, cos, mid="id-1"):
    return {"id": mid, "memory": text, "score": cos}


D = Memory._decide


# ---- tiered summaries: write + read path (hermetic) ----

def test_insert_roundtrips_abstract_and_overview():
    s = build_store()
    mid = s.insert("User prefers dark mode in the editor", [0.1, 0.2, 0.3, 0.4],
                   user_id="alice", abstract="User likes dark editor theme",
                   overview="User prefers dark mode in their code editor")
    row = s.get_memory(mid)
    assert row is not None
    assert row["abstract"] == "User likes dark editor theme"
    assert row["overview"] == "User prefers dark mode in their code editor"
    assert row["memory"] == "User prefers dark mode in the editor"


def test_update_preserves_or_replaces_summaries():
    s = build_store()
    mid = s.insert("User codes in Python", [0.1, 0.1, 0.1, 0.1],
                   user_id="alice", abstract="User's language is Python", overview="User uses Python")
    # update without summaries -> keep existing
    s.update_memory(mid, "User codes in Python daily", [0.2, 0.2, 0.2, 0.2])
    row = s.get_memory(mid)
    assert row["abstract"] == "User's language is Python"
    assert row["overview"] == "User uses Python"
    assert row["memory"] == "User codes in Python daily"


def test_schema_migration_adds_columns_to_pre_existing_db():
    """A DB that predates the abstract/overview columns must be migrated on open."""
    if os.path.exists(DB):
        os.remove(DB)
    # build a 1.1-era store: memory/vector/fts/history tables WITHOUT abstract/overview
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA_MEMORIES.replace("    abstract   TEXT,            -- L0: one-line relevance summary (tiered recall)\n", "").replace("    overview   TEXT,            -- L1: 1-2 sentence core summary (tiered recall)\n", ""))
    conn.execute("INSERT INTO memories (mem_id, memory, metadata, embed, created_at, updated_at)"
                 " VALUES ('old1','legacy fact','{}','[0,0,0,0]','t','t')")
    conn.commit()
    conn.close()
    # open via Store -> migration must add the columns
    s = Store(DB, dims=4)
    cols = [r["name"] for r in s.conn.execute("PRAGMA table_info(memories)")]
    assert "abstract" in cols and "overview" in cols
    row = s.get_memory("old1")
    assert row is not None and row["memory"] == "legacy fact"
    assert row["abstract"] is None  # legacy rows get NULL summaries


# ---- get_full API ----

def test_get_full_returns_l2_record():
    if os.path.exists(DB):
        os.remove(DB)
    m = Memory({}, db_path=DB)
    m.dims = 4  # avoid embed probe (no network in hermetic tests)
    # store directly (no reconcile -> no embed network call)
    mid = m._store_get().insert(
        "User runs free-tier Oracle", [0.0, 0.0, 0.0, 0.0], user_id="bob",
        abstract="Oracle free tier", overview="User uses Oracle free tier ARM")
    full = m.get_full(mid)
    assert full is not None
    assert full["abstract"] == "Oracle free tier"
    assert full["overview"].startswith("User uses Oracle")
    assert full["memory"] == "User runs free-tier Oracle"


# ---- retention policy: _decide merge gate (pure) ----

def test_selective_merges_high_cosine_paraphrase():
    # selective: high-cosine aliasless near-duplicate (cos 0.85) merges via UPDATE.
    # Default "all" guard is 0.90 -> ADD at 0.85; selective lowers guard to 0.82 -> UPDATE.
    op = D("memory", None,
           [_cand("existing memory text", 0.85)],
           aliases=None, retention="selective")
    assert op["event"] == "UPDATE", op


def test_selective_guard_below_same_entity_different_fact():
    # selective still ADDs same-entity-different-fact (cos 0.76 < 0.82) — never lose info
    op = D("User went hiking with Max", None,
           [_cand("User has a dog named Max", 0.76)],
           aliases=None, retention="selective")
    assert op["event"] == "ADD", op


def test_retention_all_does_not_merge_marginal_near_dup():
    # default "all" guard is 0.90: a 0.85 near-dup (no shared bigram) stays ADD
    op = D("entirely reworded but topically near sentence", None,
           [_cand("a stored fact with no shared content words", 0.85)],
           aliases=None, retention="all")
    assert op["event"] == "ADD", op


def test_selective_keeps_delete_and_update_behavior():
    # DELETE (retraction) is unaffected by retention
    op = D("Forget that I love pineapple on pizza", None,
           [_cand("User loves pineapple on pizza", 0.769)],
           aliases=None, retention="selective")
    assert op["event"] == "DELETE", op
    # UPDATE (shared bigram) is unaffected
    op2 = D("my favorite color is teal", None,
            [_cand("user's favorite color is teal", 0.878)],
            aliases=None, retention="selective")
    assert op2["event"] == "UPDATE", op2


def test_memory_exposes_retention_and_get_full():
    assert hasattr(Memory, "get_full")
    m = Memory({}, db_path="/tmp/memlite_noop_tier.db")
    assert m.retention == "all"
    m2 = Memory({"retention": "selective"}, db_path="/tmp/memlite_noop_tier2.db")
    assert m2.retention == "selective"
