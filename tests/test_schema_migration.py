import sqlite3

from memlite.store import Store


def test_legacy_schema_migration_preserves_memories(tmp_path):
    db = tmp_path / "legacy.db"

    store = Store(str(db), dims=3)
    kept_id = store.insert(
        "User keeps existing facts",
        [1.0, 0.0, 0.0],
        user_id="default",
        aliases=["retain"],
    )
    store.close()

    # Reproduce the schema from stores created before memory_type/history were cut.
    conn = sqlite3.connect(db)
    conn.execute("ALTER TABLE memories ADD COLUMN memory_type TEXT DEFAULT 'world_fact'")
    conn.execute("DROP INDEX idx_memories_scope")
    conn.execute(
        """CREATE INDEX idx_memories_scope
           ON memories(user_id, agent_id, run_id, memory_type)"""
    )
    conn.execute(
        """CREATE TABLE history (
            id TEXT PRIMARY KEY,
            memory_id TEXT,
            old_memory TEXT,
            new_memory TEXT,
            event TEXT,
            created_at TEXT
        )"""
    )
    conn.execute(
        "INSERT INTO history VALUES ('h1', ?, NULL, NULL, 'ADD', 'now')",
        (kept_id,),
    )
    conn.commit()
    conn.close()

    migrated = Store(str(db), dims=3)
    columns = {
        row[1] for row in migrated.conn.execute("PRAGMA table_info(memories)")
    }
    assert "memory_type" not in columns
    assert migrated.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='history'"
    ).fetchone() is None

    memories = migrated.list_all(filters={"user_id": "default"})
    assert len(memories) == 1
    assert memories[0]["id"] == kept_id
    assert memories[0]["memory"] == "User keeps existing facts"
    assert migrated.conn.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0] == 1
    assert migrated.conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0] == 1

    new_id = migrated.insert("New fact still writes", [0.0, 1.0, 0.0])
    assert new_id != kept_id
    assert migrated.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2
    migrated.close()
