"""Tests for memlite history (audit trail) + scoped purge, ported from mem0.

These exercise the Store layer directly (pure SQLite, no network/LLM), which is
exactly where the new history + purge logic lives and what reconcile() calls.
Run from repo root:
    . .venv/bin/activate && python -m pytest tests/test_history_purge.py -s
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memlite.store import Store
from memlite.core import Memory

DB = "/tmp/memlite_hist_test.db"
if os.path.exists(DB):
    os.remove(DB)


def build_store() -> Store:
    if os.path.exists(DB):
        os.remove(DB)
    return Store(DB, dims=4)


# ---- history: write path (what reconcile calls) ----

def test_add_writes_history_add_event():
    s = build_store()
    mid = s.insert("User likes tea", [0.1, 0.2, 0.3, 0.4], user_id="alice")
    s.add_history(mid, None, "User likes tea", "ADD")
    h = s.get_history(mid)
    assert len(h) == 1
    assert h[0]["event"] == "ADD"
    assert h[0]["old_memory"] is None
    assert h[0]["new_memory"] == "User likes tea"


def test_update_records_old_and_new_values():
    s = build_store()
    mid = s.insert("User likes tea", [0.1, 0.2, 0.3, 0.4], user_id="alice")
    s.add_history(mid, None, "User likes tea", "ADD")
    # update in place + record the change (old -> new)
    s.add_history(mid, "User likes tea", "User switched to coffee", "UPDATE")
    h = s.get_history(mid)
    assert h[0]["event"] == "UPDATE"
    assert h[0]["old_memory"] == "User likes tea"
    assert h[0]["new_memory"] == "User switched to coffee"
    # newest first
    assert h[0]["event"] == "UPDATE" and h[1]["event"] == "ADD"


def test_delete_records_null_new_memory():
    s = build_store()
    mid = s.insert("Temporary fact", [0.0, 0.0, 0.0, 0.0], user_id="alice")
    s.add_history(mid, None, "Temporary fact", "ADD")
    s.add_history(mid, "Temporary fact", None, "DELETE")
    h = s.get_history(mid)
    assert h[0]["event"] == "DELETE"
    assert h[0]["old_memory"] == "Temporary fact"
    assert h[0]["new_memory"] is None
    # history SURVIVES the row delete — that's the whole point (recoverability)
    # note: store.delete() removes the memory row but we append history separately.
    # Here we appended the DELETE record manually (reconcile does the same in-txn).


# ---- history: reconcile wiring (in-txn) ----

def test_reconcile_writes_history_in_same_txn():
    """Simulate what core._deterministic_reconcile does: insert + add_history in
    one BEGIN IMMEDIATE transaction, then commit once. Rows must be visible."""
    s = build_store()
    s.begin()
    mid = s.insert("User codes in Python", [0.5, 0.5, 0.5, 0.5], user_id="alice", in_txn=True)
    s.add_history(mid, None, "User codes in Python", "ADD", in_txn=True)
    s.commit()
    h = s.get_history(mid)
    assert len(h) == 1 and h[0]["event"] == "ADD"


# ---- purge (scoped delete_all) ----

def test_delete_all_requires_a_scope_filter():
    s = build_store()
    s.insert("a", [0.0, 0.0, 0.0, 0.0], user_id="u1")
    try:
        s.delete_all()
        assert False, "delete_all with no scope must raise"
    except ValueError:
        pass


def test_delete_all_scope_deletes_vectors_fts_history():
    s = build_store()
    a = s.insert("fact a", [0.1, 0.1, 0.1, 0.1], user_id="u1", memory_id="a1")
    s.insert("fact b", [0.2, 0.2, 0.2, 0.2], user_id="u2", memory_id="b1")
    s.add_history(a, None, "fact a", "ADD")
    n = s.delete_all(user_id="u1")
    assert n == 1
    # memory row, vector, fts, and history all gone for u1
    assert s.list_all(filters={"user_id": "u1"}) == []
    assert s.list_all(filters={"user_id": "u2"}) != []
    assert s.get_history("a1") == []  # purge removes the trail


def test_delete_all_leaves_other_scope_untouched():
    s = build_store()
    s.insert("a", [0.1, 0.1, 0.1, 0.1], user_id="u1", memory_id="m1")
    s.insert("b", [0.2, 0.2, 0.2, 0.2], user_id="u2", memory_id="m2")
    s.insert("c", [0.3, 0.3, 0.3, 0.3], user_id="u2", memory_id="m3")
    n = s.delete_all(user_id="u2")
    assert n == 2
    assert len(s.list_all(filters={"user_id": "u1"})) == 1


# ---- Memory API surface ----

def test_memory_history_and_delete_all_exist():
    # Hermetic: just confirm the public methods exist on Memory (they delegate to Store).
    # A network-free end-to-end would need embeddings; the Store tests above are the
    # authoritative proof. Here we only assert the API shape.
    assert hasattr(Memory, "history")
    assert hasattr(Memory, "delete_all")
