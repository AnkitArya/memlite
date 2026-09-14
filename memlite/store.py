"""Single-file SQLite store with sqlite-vec semantic search + FTS5 keyword fallback.

Tables (all in ONE db file, share a connection/WAL):
  memories          canonical rows
  memory_vectors    sqlite-vec vec0 virtual table (same DB, transactional)
  memories_fts      FTS5 keyword index (fallback when vector recall misses)
"""
import json
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
    metadata   TEXT,            -- JSON dict
    embed      TEXT NOT NULL,   -- JSON list of floats (kept inline for self-containment)
    aliases    TEXT,            -- JSON list of retrieval aliases (associated vocabulary)
    abstract   TEXT,            -- L0: one-line relevance summary (tiered recall)
    overview   TEXT,            -- L1: 1-2 sentence core summary (tiered recall)
    created_at TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_scope
    ON memories(user_id, agent_id, run_id);
"""

# Tiered-recall columns added after 1.0.x shipped with the base schema; on a
# pre-existing DB the CREATE TABLE IF NOT EXISTS above is a no-op, so ALTER
# ADD COLUMN brings old stores forward. Each ADD is wrapped in try/except
# (COLUMNNAME already exists) in _init_schema so re-runs stay idempotent.
_SCHEMA_MEMORIES_MIGRATE = [
    "ALTER TABLE memories ADD COLUMN abstract TEXT",
    "ALTER TABLE memories ADD COLUMN overview TEXT",
]

_SCHEMA_VEC = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors
USING vec0(
    embedding float[{dims}] distance_metric={distance}
);
"""

_SCHEMA_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
USING fts5(id UNINDEXED, memory, aliases);
"""

# Append-only audit trail ported from mem0's history (memory/storage.py). Every
# ADD / UPDATE / DELETE on a memory writes one row here, so a value that was
# overwritten or deleted in place is always recoverable. Lives in the SAME file
# as memories (unlike mem0's separate history.db) so it stays single-file and
# rides the existing WAL transaction.
_SCHEMA_HISTORY = """
CREATE TABLE IF NOT EXISTS memories_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    mem_id     TEXT NOT NULL,
    old_memory TEXT,
    new_memory TEXT,
    event      TEXT NOT NULL,        -- ADD | UPDATE | DELETE
    actor_id   TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_history_mem_id ON memories_history(mem_id);
"""

_DISTANCE = "cosine"  # vec0 supports cosine

# Cap on the vec0 kNN scan widening: beyond this the brute-force scan cost
# dominates; instead of rescanning the whole table, accept fewer results.
# kiss-cut: filtered scans silently return fewer than top_k once a scope's
# table exceeds this many rows; ceiling ~50k rows/scope. Upgrade: partition
# memory_vectors per scope when stores approach the cap.
_MAX_KNN_SCAN = 50000
# sqlite-vec hard limit: "k value in knn query too large" above 4096.
_VEC0_K_MAX = 4096


class Store:
    def __init__(self, db_path: str = "memlite.db", dims: int = 768):
        self.db_path = db_path
        self.dims = dims
        # kiss-cut: one global write lock serializes all mutations; SQLite is
        # single-writer anyway so this is free today, but it also serializes
        # the (read-side) _init_schema path. Ceiling ~1k writes/sec. Upgrade:
        # per-connection/per-user locks only if real contention ever shows.
        # RLock (not Lock): reset() calls _init_schema() while holding the
        # lock — a plain Lock would deadlock itself here.
        self._lock = threading.RLock()
        self.conn = self._connect()
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        # WAL + NORMAL: no fsync per commit, durability preserved vs power loss
        # only at OS level — the standard safe trade-off for WAL databases.
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        # Load sqlite-vec into this connection
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=5.0)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    # ---------- schema ----------
    def _init_schema(self):
        with self._lock:
            cur = self.conn.cursor()
            cur.executescript(_SCHEMA_MEMORIES)
            cur.executescript(_SCHEMA_VEC.format(dims=self.dims, distance=_DISTANCE))
            cur.executescript(_SCHEMA_FTS)
            cur.executescript(_SCHEMA_HISTORY)
            # bring pre-1.2 DBs forward: abstract/overview columns
            for ddl in _SCHEMA_MEMORIES_MIGRATE:
                try:
                    cur.execute(ddl)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
            self.conn.commit()

    # ---------- transaction control ----------
    def begin(self):
        """Explicit BEGIN so multiple mutations share one commit (one fsync)."""
        self.conn.execute("BEGIN IMMEDIATE")

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

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
        aliases: list[str] | None = None,
        abstract: str | None = None,
        overview: str | None = None,
        in_txn: bool = False,
    ) -> str:
        """Insert one memory. in_txn=True skips commit — the caller owns the
        transaction (single fsync for N mutations). *aliases* are associated
        retrieval terms (synonyms/related vocabulary); they are indexed into
        the FTS corpus alongside the main text so queries using related terms
        (e.g. "horoscope" for a stored "zodiac" fact) still recall.
        *abstract*/overview are the L0/L1 tiered-recall summaries."""
        mid = memory_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        vec_blob = sqlite_vec.serialize_float32(embedding)
        aliases_json = json.dumps(aliases) if aliases else None
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """INSERT INTO memories
                  (mem_id, user_id, agent_id, run_id, memory,
                   metadata, embed, aliases, abstract, overview, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    mid, user_id, agent_id, run_id, memory,
                    json.dumps(metadata or {}), json.dumps(embedding),
                    aliases_json, abstract, overview, now, now,
                ),
            )
            row_id = cur.lastrowid
            # vector row + FTS row (aliases in the corpus for BM25 recall)
            cur.execute(
                "INSERT INTO memory_vectors(rowid, embedding) VALUES (?, ?)",
                (row_id, vec_blob),
            )
            cur.execute(
                "INSERT INTO memories_fts(id, memory, aliases) VALUES (?, ?, ?)",
                (mid, memory, aliases_json or ""),
            )
            if not in_txn:
                self.conn.commit()
        return mid

    def update_memory(
        self, memory_id: str, memory: str, embedding: list[float],
        aliases: list[str] | None = None,
        abstract: str | None = None,
        overview: str | None = None,
        in_txn: bool = False,
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self.conn.cursor()
            row = cur.execute("SELECT id, aliases FROM memories WHERE mem_id=?", (memory_id,)).fetchone()
            if row is None:
                return False
            row_id = row["id"]
            # aliases: explicit replacement, else keep existing
            merged_aliases = row["aliases"]
            aliases_json = json.dumps(aliases) if aliases else merged_aliases
            # tiered summaries: explicit replacement, else keep existing (fall back to the new text)
            merged_abstract = cur.execute(
                "SELECT abstract FROM memories WHERE mem_id=?", (memory_id,)).fetchone()["abstract"]
            merged_overview = cur.execute(
                "SELECT overview FROM memories WHERE mem_id=?", (memory_id,)).fetchone()["overview"]
            abstract_final = abstract if abstract is not None else (merged_abstract or memory)
            overview_final = overview if overview is not None else (merged_overview or memory)
            cur.execute(
                "UPDATE memories SET memory=?, embed=?, aliases=?, abstract=?, overview=?, updated_at=? WHERE mem_id=?",
                (memory, json.dumps(embedding), aliases_json, abstract_final, overview_final, now, memory_id),
            )
            cur.execute(
                "UPDATE memory_vectors SET embedding=? WHERE rowid=?",
                (sqlite_vec.serialize_float32(embedding), row_id),
            )
            # FTS: delete + reinsert by id (standalone fts5 table)
            cur.execute("DELETE FROM memories_fts WHERE id=?", (memory_id,))
            cur.execute(
                "INSERT INTO memories_fts(id, memory, aliases) VALUES (?,?,?)",
                (memory_id, memory, aliases_json or ""),
            )
            if not in_txn:
                self.conn.commit()
        return True

    def delete(self, memory_id: str, in_txn: bool = False) -> bool:
        with self._lock:
            cur = self.conn.cursor()
            r = cur.execute("SELECT id FROM memories WHERE mem_id=?", (memory_id,)).fetchone()
            if r is None:
                return False
            row_id = r["id"]
            cur.execute("DELETE FROM memories WHERE mem_id=?", (memory_id,))
            cur.execute("DELETE FROM memory_vectors WHERE rowid=?", (row_id,))
            cur.execute("DELETE FROM memories_fts WHERE id=?", (memory_id,))
            if not in_txn:
                self.conn.commit()
        return True

    # ---------- history (audit trail) ----------
    def add_history(self, memory_id, old_memory, new_memory, event, *, actor_id=None,
                    created_at=None, in_txn=False) -> None:
        """Append one audit record. in_txn=True shares the caller's transaction
        (no extra fsync) — reconcile() writes its per-fact history rows inside the
        single BEGIN IMMEDIATE block."""
        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO memories_history (mem_id, old_memory, new_memory, event, actor_id, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (memory_id, old_memory, new_memory, event, actor_id, created_at),
            )
            if not in_txn:
                self.conn.commit()

    def get_history(self, memory_id: str, limit: int = 50) -> list[dict]:
        """Revisions for a memory, newest first. old_memory is None for ADD,
        new_memory is None for DELETE."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT event, old_memory, new_memory, actor_id, created_at"
                " FROM memories_history WHERE mem_id=? ORDER BY id DESC LIMIT ?",
                (memory_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_all(self, *, user_id=None, agent_id=None, run_id=None) -> int:
        """Scoped bulk delete (mem0.delete_all port). At least one scope filter is
        required — a bare wipe goes through reset(). Deletes the memory rows, their
        vector index rows, FTS rows, and their history trail atomically; returns the
        number of memories purged."""
        scope = [(k, v) for k, v in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id))
                 if v is not None]
        if not scope:
            raise ValueError("delete_all requires at least one of user_id, agent_id, run_id; use reset() to wipe everything")
        with self._lock:
            cur = self.conn.cursor()
            where = " AND ".join(f"{k}=?" for k, _ in scope)
            args = [v for _, v in scope]
            ids = cur.execute(f"SELECT id, mem_id FROM memories WHERE {where}", args).fetchall()
            if not ids:
                return 0
            row_ids = [r["id"] for r in ids]
            mem_ids = [r["mem_id"] for r in ids]
            marks = ",".join("?" * len(row_ids))
            cur.execute(f"DELETE FROM memories WHERE id IN ({marks})", row_ids)
            cur.execute(f"DELETE FROM memory_vectors WHERE rowid IN ({marks})", row_ids)
            cur.execute(f"DELETE FROM memories_fts WHERE id IN ({marks})", mem_ids)
            cur.execute(f"DELETE FROM memories_history WHERE mem_id IN ({marks})", mem_ids)
            self.conn.commit()
        return len(ids)

    # ---------- reads ----------
    def get_memory(self, memory_id: str) -> dict | None:
        """Fetch one memory row by mem_id (full detail, incl. L0/L1 summaries)."""
        with self._lock:
            row = self.conn.execute(
                "SELECT mem_id AS id, memory, abstract, overview, metadata, user_id, agent_id, run_id,"
                " created_at, updated_at FROM memories WHERE mem_id=?",
                (memory_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_all(self, filters: dict | None = None, limit: int | None = None) -> list[dict]:
        sql = """SELECT m.mem_id AS id, m.memory, m.user_id, m.agent_id,
                        m.run_id, m.metadata, m.created_at, m.updated_at
                 FROM memories m WHERE 1=1"""
        args = []
        for key in ("user_id", "agent_id", "run_id"):
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
        """Cosine-similarity search over the vec0 table, joined to rows + filters.

        Over-fetch trap: vec0's kNN scans the WHOLE table, so with filters the
        join can discard every hit (tenant has 10 rows in a 50k table; the
        global top-100 may contain none of them). Fix: progressively widen the
        kNN limit until enough scoped rows survive or the table is exhausted.
        """
        blob = sqlite_vec.serialize_float32(query_embedding)
        # vec0 requires MATCH + LIMIT as the LAST clauses of the query it scans,
        # so run the knn in a subquery, then join+filter rows in the outer query.
        filter_keys = [k for k in ("user_id", "agent_id", "run_id")
                       if (filters or {}).get(k) is not None]
        widen = 0
        knn_limit = min(top_k * 10, _VEC0_K_MAX)
        while True:
            sql = """
                SELECT m.mem_id AS id,
                       (1 - knn.distance) AS score,
                       m.memory,
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
            args = [blob, knn_limit]
            for key in filter_keys:
                sql += f" AND m.{key}=?"
                args.append(filters[key])
            sql += " ORDER BY knn.distance ASC LIMIT ?"
            args.append(top_k)
            rows = self.conn.execute(sql, args).fetchall()
            if len(rows) >= top_k or widen >= 2 or knn_limit >= min(_MAX_KNN_SCAN, _VEC0_K_MAX):
                return [self._row_to_dict(r) for r in rows]
            widen += 1
            knn_limit = min(knn_limit * 8, _VEC0_K_MAX)
        return [self._row_to_dict(r) for r in rows]

    def keyword_search(self, query: str, top_k: int = 5, filters: dict | None = None) -> list[dict]:
        """FTS5 (BM25-ish) fallback over text + aliases corpus."""
        terms = [t for t in query.replace("-", " ").replace("_", " ").split() if t]
        if not terms:
            return []
        match_expr = " OR ".join(f'"{t}"' for t in terms)
        sql = """
            SELECT m.mem_id AS id, fts.rank AS score,
                   m.memory, m.user_id, m.agent_id, m.run_id, m.metadata,
                   m.created_at, m.updated_at
            FROM (
                SELECT rowid, rank, id FROM memories_fts
                WHERE memories_fts MATCH ?
                ORDER BY memories_fts.rank
                LIMIT ?
            ) fts
            JOIN memories m ON m.mem_id = fts.id
        """
        args = [match_expr, top_k * 4]
        for key in ("user_id", "agent_id", "run_id"):
            v = filters.get(key) if filters else None
            if v is not None:
                sql += f" AND m.{key}=?"
                args.append(v)
        sql += " ORDER BY fts.rank LIMIT ?"
        args.append(top_k)
        rows = self.conn.execute(sql, args).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def reset(self):
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DROP TABLE IF EXISTS memories")
            cur.execute("DROP TABLE IF EXISTS memory_vectors")
            cur.execute("DROP TABLE IF EXISTS memories_fts")
            self.conn.commit()
            self._init_schema()

    def close(self):
        self.conn.close()

    # ---------- helpers ----------
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
        # Normalize FTS bm25 (lower=better, negative) vs vector score (higher=better)
        # We keep both; callers pick a strategy.
        return d
