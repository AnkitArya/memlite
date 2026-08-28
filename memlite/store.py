"""Single-file SQLite store with sqlite-vec semantic search + FTS5 keyword fallback.

Tables (all in ONE db file, share a connection/WAL):
  memories          canonical rows
  memory_vectors    sqlite-vec vec0 virtual table (same DB, transactional)
  memories_fts      FTS5 keyword index (fallback when vector recall misses)
  history           audit log (like mem0's SQLiteManager)
"""
import json
import math
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

import sqlite_vec

_SCHEMA_MEMORIES = """
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    mem_id     TEXT NOT NULL UNIQUE,
    user_id    TEXT,
    agent_id   TEXT,
    run_id     TEXT,
    memory     TEXT NOT NULL,
    memory_type TEXT DEFAULT 'world_fact',   -- 'world_fact' | 'experience' (hindsight-inspired)
    metadata   TEXT,            -- JSON dict
    embed      TEXT NOT NULL,   -- JSON list of floats (kept inline for self-containment)
    created_at TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_scope
    ON memories(user_id, agent_id, run_id, memory_type);
"""

_SCHEMA_VEC = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors
USING vec0(
    embedding float[{dims}] distance_metric={distance}
);
"""

_SCHEMA_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
USING fts5(id UNINDEXED, memory);
"""

_SCHEMA_HISTORY = """
CREATE TABLE IF NOT EXISTS history (
    id         TEXT PRIMARY KEY,
    memory_id  TEXT,
    old_memory TEXT,
    new_memory TEXT,
    event      TEXT,
    created_at TEXT
);
"""

_DISTANCE = "cosine"  # vec0 supports cosine


class Store:
    def __init__(self, db_path: str = "memlite.db", dims: int = 768):
        self.db_path = db_path
        self.dims = dims
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        # Load sqlite-vec into this connection
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._init_schema()

    # ---------- schema ----------
    def _init_schema(self):
        with self._lock:
            cur = self.conn.cursor()
            cur.executescript(_SCHEMA_MEMORIES)
            self._ensure_vec_table(cur)
            cur.executescript(_SCHEMA_FTS)
            cur.executescript(_SCHEMA_HISTORY)
            self.conn.commit()

    def _ensure_vec_table(self, cur):
        """Create the vec0 table, or rebuild it if a legacy copy used L2.

        Databases created before distance_metric was wired up silently scored
        with (1 - L2), which is not cosine similarity and broke reconcile
        thresholds. Embeddings live inline in memories.embed, so the rebuild is
        a cheap one-time copy — no re-embedding, no data loss.
        """
        row = cur.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_vectors'"
        ).fetchone()
        if row is not None and "distance_metric=cosine" in (row["sql"] or ""):
            return  # current schema, nothing to do
        if row is not None:
            rows = cur.execute("SELECT id, embed FROM memories").fetchall()
            cur.execute("DROP TABLE memory_vectors")
            cur.executescript(_SCHEMA_VEC.format(dims=self.dims, distance=_DISTANCE))
            for r in rows:
                cur.execute(
                    "INSERT INTO memory_vectors(rowid, embedding) VALUES (?, ?)",
                    (r["id"], sqlite_vec.serialize_float32(json.loads(r["embed"]))),
                )
        else:
            cur.executescript(_SCHEMA_VEC.format(dims=self.dims, distance=_DISTANCE))

    # ---------- writes ----------
    def insert(
        self,
        memory: str,
        embedding: list[float],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict | None = None,
        memory_id: str | None = None,
        memory_type: str = "world_fact",
    ) -> str:
        mid = memory_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        vec_blob = self._vector_to_blob(embedding)
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """INSERT INTO memories
                   (mem_id, user_id, agent_id, run_id, memory, memory_type,
                    metadata, embed, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    mid, user_id, agent_id, run_id, memory, memory_type,
                    json.dumps(metadata or {}), json.dumps(embedding),
                    now, now,
                ),
            )
            row_id = cur.lastrowid
            # vector row + FTS row
            cur.execute(
                "INSERT INTO memory_vectors(rowid, embedding) VALUES (?, ?)",
                (row_id, vec_blob),
            )
            cur.execute(
                "INSERT INTO memories_fts(id, memory) VALUES (?, ?)",
                (mid, memory),
            )
            cur.execute(
                "INSERT INTO history(id, memory_id, old_memory, new_memory, event, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (str(uuid.uuid4()), mid, None, memory, "ADD", now),
            )
            self.conn.commit()
        return mid

    def update_memory(
        self, memory_id: str, memory: str, embedding: list[float]
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self.conn.cursor()
            row = cur.execute("SELECT id, memory FROM memories WHERE mem_id=?", (memory_id,)).fetchone()
            if row is None:
                return False
            row_id, old = row["id"], row["memory"]
            cur.execute(
                "UPDATE memories SET memory=?, embed=?, updated_at=? WHERE mem_id=?",
                (memory, json.dumps(embedding), now, memory_id),
            )
            cur.execute(
                "UPDATE memory_vectors SET embedding=? WHERE rowid=?",
                (self._vector_to_blob(embedding), row_id),
            )
            # FTS: delete + reinsert by id (standalone fts5 table)
            cur.execute("DELETE FROM memories_fts WHERE id=?", (memory_id,))
            cur.execute(
                "INSERT INTO memories_fts(id, memory) VALUES (?,?)",
                (memory_id, memory),
            )
            cur.execute(
                "INSERT INTO history(id, memory_id, old_memory, new_memory, event, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (str(uuid.uuid4()), memory_id, old, memory, "UPDATE", now),
            )
            self.conn.commit()
        return True

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            cur = self.conn.cursor()
            r = cur.execute("SELECT id FROM memories WHERE mem_id=?", (memory_id,)).fetchone()
            if r is None:
                return False
            row_id = r["id"]
            cur.execute("DELETE FROM memories WHERE mem_id=?", (memory_id,))
            cur.execute("DELETE FROM memory_vectors WHERE rowid=?", (row_id,))
            cur.execute("DELETE FROM memories_fts WHERE id=?", (memory_id,))
            now = datetime.now(timezone.utc).isoformat()
            cur.execute(
                "INSERT INTO history(id, memory_id, event, created_at) VALUES (?,?,?,?)",
                (str(uuid.uuid4()), memory_id, "DELETE", now),
            )
            self.conn.commit()
        return True

    # ---------- reads ----------
    def get(self, memory_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE mem_id=?", (memory_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_all(self, filters: dict | None = None, limit: int | None = None) -> list[dict]:
        sql = """SELECT m.mem_id AS id, m.memory, m.memory_type, m.user_id, m.agent_id,
                        m.run_id, m.metadata, m.created_at, m.updated_at
                 FROM memories m WHERE 1=1"""
        args = []
        for key in ("user_id", "agent_id", "run_id", "memory_type"):
            v = filters.get(key) if filters else None
            if v is not None:
                sql += f" AND {key}=?"
                args.append(v)
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        rows = self.conn.execute(sql, args).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def semantic_search(
        self, query_embedding: list[float], top_k: int = 5, filters: dict | None = None
    ) -> list[dict]:
        """Cosine-similarity search over the vec0 table, joined to rows + filters."""
        blob = self._vector_to_blob(query_embedding)
        # vec0 requires MATCH + LIMIT as the LAST clauses of the query it scans,
        # so run the knn in a subquery, then join+filter rows in the outer query.
        sql = """
            SELECT m.mem_id AS id,
                   (1 - knn.distance) AS score,
                   m.memory, m.memory_type,
                   m.user_id, m.agent_id, m.run_id, m.metadata,
                   m.created_at, m.updated_at
            FROM (
                SELECT rowid, distance
                FROM memory_vectors
                WHERE embedding MATCH ?
                LIMIT ?
            ) knn
            JOIN memories m ON m.id = knn.rowid
            WHERE 1=1
        """
        # over-fetch so post-filtering still yields top_k per scope
        args = [blob, top_k * 10]
        for key in ("user_id", "agent_id", "run_id", "memory_type"):
            v = filters.get(key) if filters else None
            if v is not None:
                sql += f" AND m.{key}=?"
                args.append(v)
        sql += " ORDER BY knn.distance ASC LIMIT ?"
        args.append(top_k)
        rows = self.conn.execute(sql, args).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def keyword_search(self, query: str, top_k: int = 5, filters: dict | None = None) -> list[dict]:
        """FTS5 (BM25-ish) fallback. Query terms are split on non-alphanumerics."""
        terms = [t for t in query.replace("-", " ").replace("_", " ").split() if t]
        if not terms:
            return []
        match_expr = " AND ".join(f'"{t}"' for t in terms)
        sql = """
            SELECT m.mem_id AS id, memories_fts.rank AS score,
                   m.memory, m.memory_type, m.user_id, m.agent_id, m.run_id, m.metadata,
                   m.created_at, m.updated_at
            FROM memories_fts
            JOIN memories m ON m.mem_id = memories_fts.id
            WHERE memories_fts MATCH ?
        """
        args = [match_expr]
        for key in ("user_id", "agent_id", "run_id", "memory_type"):
            v = filters.get(key) if filters else None
            if v is not None:
                sql += f" AND m.{key}=?"
                args.append(v)
        sql += " ORDER BY memories_fts.rank LIMIT ?"
        args.append(top_k)
        rows = self.conn.execute(sql, args).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def reset(self):
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DROP TABLE IF EXISTS memories")
            cur.execute("DROP TABLE IF EXISTS memory_vectors")
            cur.execute("DROP TABLE IF EXISTS memories_fts")
            cur.execute("DROP TABLE IF EXISTS history")
            self.conn.commit()
            self._init_schema()

    def close(self):
        self.conn.close()

    # ---------- helpers ----------
    @staticmethod
    def _vector_to_blob(vec: list[float]) -> bytes:
        return sqlite_vec.serialize_float32(vec)

    @staticmethod
    def _row_to_dict(row) -> dict:
        d = dict(row)
        if d.get("mem_id") is not None:
            d["id"] = d.pop("mem_id")
        if d.get("metadata"):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        if d.get("embed"):
            try:
                d["embedding"] = json.loads(d["embed"])
            except (json.JSONDecodeError, TypeError):
                pass
        # Normalize FTS bm25 (lower=better, negative) vs vector score (higher=better)
        # We keep both; callers pick a strategy.
        return d
