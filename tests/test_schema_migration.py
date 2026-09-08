import sqlite3

from memlite.store import Store


def test_legacy_schema_migration_preserves_memories(tmp_path):
    db = tmp_path / "legacy.db"

    store = Store(str(db), dims=3)
    first_id = store.insert(
        "User keeps existing facts",
        [1.0, 0.0, 0.0],
        user_id="default",
        aliases=["retain"],
    )
    deleted_id = store.insert("This fact will be deleted", [0.0, 0.0, 1.0])
    last_id = store.insert(
        "User keeps another existing fact",
        [0.0, 1.0, 0.0],
        user_id="default",
    )
    store.delete(deleted_id)
    store.close()

    # Reproduce the schema from stores created before memory_type/history were cut.
    conn = sqlite3.connect(db)
    legacy_rows = conn.execute(
        "SELECT id, mem_id, memory FROM memories ORDER BY id"
    ).fetchall()
    assert [row[0] for row in legacy_rows] == [1, 3]
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
        (first_id,),
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

    rows = migrated.conn.execute(
        "SELECT id, mem_id, memory FROM memories ORDER BY id"
    ).fetchall()
    assert [(row[0], row[1]) for row in rows] == [
        (legacy_rows[0][0], first_id),
        (legacy_rows[1][0], last_id),
    ]
    assert [row[2] for row in rows] == [
        "User keeps existing facts",
        "User keeps another existing fact",
    ]

    # Every surviving canonical row still has its vector and FTS companion.
    for row_id, mem_id, memory in rows:
        assert migrated.conn.execute(
            "SELECT 1 FROM memory_vectors WHERE rowid=?", (row_id,)
        ).fetchone() is not None
        assert migrated.conn.execute(
            "SELECT memory FROM memories_fts WHERE id=?", (mem_id,)
        ).fetchone()[0] == memory

    new_id = migrated.insert("New fact still writes", [0.0, 1.0, 0.0])
    assert new_id not in {first_id, last_id}
    assert migrated.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 3
    migrated.close()

    # Reopening must be idempotent and must not rebuild or renumber anything.
    reopened = Store(str(db), dims=3)
    reopened_rows = reopened.conn.execute(
        "SELECT id, mem_id, memory FROM memories ORDER BY id"
    ).fetchall()
    assert [(row[0], row[1], row[2]) for row in reopened_rows] == [
        (row[0], row[1], row[2]) for row in rows
    ] + [(reopened_rows[-1][0], new_id, "New fact still writes")]
    assert reopened.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_memories_scope'"
    ).fetchone()[0].replace(" ", "").endswith("user_id,agent_id,run_id)")
    reopened.close()
