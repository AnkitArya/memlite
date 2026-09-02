"""Memory - the public API, mirroring mem0.Memory (add/search/get_all/update/delete).

Design notes (lean on purpose):
  - Everything lives in one SQLite file via Store.
  - `add` always runs the LLM extraction pass (mem0-style), then deterministically
    reconciles each extracted fact (ADD / UPDATE / DELETE).
  - `search` uses semantic (sqlite-vec) by default; pass strategy="hybrid" to fuse
    vector + FTS5 keyword scores.
"""
import os
import re
import time
import uuid

from .embedder import Embedder
from .store import Store

# Recency weighting for hybrid search (mirrors mem0's recency weighting idea):
# score = rrf_score * (RECENCY_BASE + (1-RECENCY_BASE) * decay), where decay
# halves every DECAY_HALF_LIFE_DAYS since last update.
DECAY_HALF_LIFE_DAYS = 30.0
RECENCY_BASE = 0.5


def _age_days(iso_ts: str | None, now: float) -> float | None:
    if not iso_ts:
        return None
    try:
        if iso_ts.endswith("Z"):
            iso_ts = iso_ts[:-1] + "+00:00"
        import datetime as _dt
        t = _dt.datetime.fromisoformat(iso_ts)
        if t.tzinfo is None:
            t = t.replace(tzinfo=_dt.timezone.utc)
        return max(0.0, now - t.timestamp()) / 86400.0
    except (ValueError, TypeError):
        return None


_EXTRACT_PROMPT = """You extract durable, long-term memories from a conversation.

Conversation:
\"\"\"
{conversation}
\"\"\"

Rules:
- Extract ONLY facts worth remembering long-term (preferences, personal
  details, decisions, skills, important events). Ignore small talk, questions,
  transient states, and assistant filler.
- Each memory must be a short, self-contained statement. Rewrite "I/my" as
  "User". Never invent facts not present in the conversation.
- If nothing is worth remembering, return an empty list.

Return STRICTLY this JSON, no prose, no markdown fences:
{{"memories": ["fact 1", "fact 2"]}}
"""

# ---------------------------------------------------------------------------
# Deterministic (no-LLM) reconcile
# ADD / UPDATE / DELETE is decided with arithmetic, not a model:
#   - DELETE: the new text carries an explicit retraction intent (lexical
#     pattern) AND matches an existing memory closely.
#   - UPDATE: the new text is semantically near an existing memory AND shares
#     enough content-word tokens (same claim/topic, reworded or changed).
#   - ADD: otherwise — conservative default, so information is never dropped.
#
# Cosine thresholds were calibrated against live bge-base-en-v1.5 embeddings:
#   exact reword ~0.88, topic-update ~0.73, same-entity-different-fact ~0.76,
#   unrelated ~0.38-0.52. Word Jaccard CANNOT disambiguate topic-update from
#   same-entity-different-fact (both ~0.33), so the UPDATE gate uses SHARED
#   CONTENT-BIGRAMS instead: a value change ("favorite color is teal" ->
#   "changed to magenta") preserves the attribute key phrase ("favorite color"),
#   while a different fact about the same entity ("has a dog named Max" vs
#   "went hiking with Max") shares no content bigram. Known conservative miss:
#   fully-reworded value changes with no lexical residue ("lives in Hyderabad"
#   -> "moved to Bangalore") fall through to ADD — a duplicate, never data loss.
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "at", "that", "this", "i", "me", "my", "we", "our", "you", "your",
    "it", "its", "is", "are", "was", "were", "be", "been", "being", "has",
    "have", "had", "do", "does", "did", "will", "would", "can", "could", "now",
    "actually", "just", "really", "said", "say", "about", "their", "there",
}

# Phrases that signal an explicit retraction (DELETE intent). Anchored to word
# boundaries so "not my favorite" still matches but "don't forget" (positive)
# or "forgettable" do not.
_RETRACT_RE = re.compile(
    r"\b(forget|forgot|remove|delete|erase|drop|retract|scrap|never (?:liked|said|did|meant)|"
    r"don'?t (?:like|want|need|use)|no longer|i (?:was )?wrong about|change my mind about|"
    r"ignore what i (?:said|wrote))\b",
    re.IGNORECASE,
)


def _content_tokens(text: str) -> set[str]:
    """Set of meaningful lowercase tokens, stopwords and punctuation removed."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _content_token_list(text: str) -> list[str]:
    """ORDERED content tokens (set order is arbitrary — bigrams need order)."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _bigrams(text: str) -> set[str]:
    """Adjacent content-word pairs ("favorite color", "works as")."""
    ts = _content_token_list(text)
    return {" ".join(p) for p in zip(ts, ts[1:])}


def _shared_bigrams(a: str, b: str) -> set[str]:
    """Content-word bigrams present in BOTH strings.

    The UPDATE signal: a value change preserves the attribute key phrase
    ("favorite color", "works as"), a different fact about the same entity
    shares nothing. Word-set Jaccard cannot make this distinction (both score
    ~0.33); shared bigrams can.
    """
    return _bigrams(a) & _bigrams(b)


def _is_retraction(text: str) -> bool:
    m = _RETRACT_RE.search(text)
    if m is None:
        return False
    # "don't/didn't forget ..." is a reminder (positive intent), not a retraction:
    # if the matched token is forget/forgot and it is negated, keep it.
    if m.group(0).lower() in ("forget", "forgot"):
        prefix = text[: m.start()].lower()
        if re.search(r"\b(?:don'?t|do not|didn'?t|did not|never)\s+$", prefix):
            return False
    return True


class Memory:
    def __init__(self, config: dict | None = None, db_path: str = "memlite.db"):
        config = config or {}
        llm_cfg = config.get("llm", {}).get("config", {})
        emb_cfg = config.get("embedder", {}).get("config", {})
        vec_cfg = config.get("vector_store", {}).get("config", {})

        self.db_path = vec_cfg.get("path", db_path)
        self.embedder = Embedder(emb_cfg)
        self.dims = emb_cfg.get("embedding_dims")  # may be None; resolved on first embed

        # LLM for extraction (REQUIRED for add(); search/get_all/update/delete
        # work without it).
        self._llm_client = None
        self._llm_model = None
        self._llm_base = None
        if llm_cfg:
            try:
                import openai

                base = llm_cfg.get("openai_base_url", "https://api.deepinfra.com/v1/openai")
                key = (
                    llm_cfg.get("api_key")
                    or os.environ.get("OPENAI_API_KEY")
                    or os.environ.get("DEEPINFRA_API_KEY")
                )
                self._llm_client = openai.OpenAI(api_key=key, base_url=base)
                self._llm_model = llm_cfg.get("model", "deepseek-ai/DeepSeek-V3")
                self._llm_base = base
            except Exception:
                self._llm_client = None

        self._store: Store | None = None

    # lazy store so dims are known after first embed
    def _store_get(self) -> Store:
        if self._store is None:
            dims = self.dims or self._resolve_dims()
            self._store = Store(self.db_path, dims=dims)
        return self._store

    def _resolve_dims(self) -> int:
        v = self.embedder.embed(".")  # cheap probe
        self.dims = len(v)
        return len(v)

    # ---------- add ----------
    def add(
        self,
        messages,
        *,
        user_id=None,
        agent_id=None,
        run_id=None,
        metadata=None,
        memory_type="world_fact",
    ):
        """Create memory/memories. mirrors mem0 (incl. UPDATE/DELETE reconciliation).

        messages: str | dict | list[dict].
        Single path (always on):
          1) LLM extraction pass distills the conversation into discrete durable
             facts; retraction statements ("Forget X") bypass extraction so the
             DELETE intent survives. Requires an LLM in config.
          2) each extracted fact is reconciled against existing memories
             deterministically (no LLM):
              ADD (new fact) | UPDATE (reworded/changed same claim) | DELETE (retraction).
        memory_type: 'world_fact' (default) or 'experience' (hindsight-inspired).

        Returns: {"results": [{"id", "memory", "event"}, ...]} with event in
          ADD | UPDATE | DELETE.
        """
        texts = self._to_texts(messages)
        if not texts:
            return {"results": []}

        if self._llm_client is None:
            raise ValueError(
                "add() requires an LLM (config key 'llm' or OPENAI_API_KEY / "
                "DEEPINFRA_API_KEY) for the extraction pass."
            )

        # Retraction statements bypass extraction: the extractor would rephrase
        # "Forget X" into a plain fact and the DELETE intent would be lost.
        reclaimed = [t for t in texts if _is_retraction(t)]
        normal = [t for t in texts if t not in reclaimed]
        extracted = self._extract(normal) if normal else []
        extracted.extend(reclaimed)
        if not extracted:
            # nothing extracted and nothing was a retraction: treat raw text as
            # the fact rather than silently dropping the add
            extracted = list(texts)

        return self._reconcile_many(
            extracted, user_id=user_id, agent_id=agent_id, run_id=run_id,
            metadata=metadata, memory_type=memory_type,
        )

    def _reconcile_many(self, texts, *, user_id, agent_id, run_id, metadata,
                        memory_type):
        """Reconcile is ALWAYS deterministic (no LLM): extraction may use the
        LLM, the ADD/UPDATE/DELETE decision does not."""
        return self._deterministic_reconcile(
            texts, user_id=user_id, agent_id=agent_id, run_id=run_id,
            metadata=metadata, memory_type=memory_type,
        )

    def _extract(self, texts: list[str]) -> list[str]:
        """LLM extraction pass: distill a conversation into discrete memories.

        Mirrors mem0's fact extraction. Returns [] if no LLM or the call fails
        (callers fall back to raw dedupe-by-heuristic — never drop data).
        """
        if self._llm_client is None:
            return []
        convo = "\n".join(f"user says: {t}" for t in texts)
        prompt = _EXTRACT_PROMPT.format(conversation=convo)
        try:
            import json
            r = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            # tolerate markdown fences / prose around the JSON block
            m = re.search(r"\{.*\}", r.choices[0].message.content, re.S)
            if not m:
                return []
            val = json.loads(m.group(0))
            return [
                str(f).strip()
                for f in val.get("memories", [])
                if isinstance(f, (str, dict)) and str(f).strip()
            ]
        except Exception:
            return []

    def _deterministic_reconcile(self, texts, *, user_id, agent_id, run_id, metadata, memory_type):
        """No-LLM reconcile. Arithmetic decision over semantic + token overlap.

        For each new text: retrieve top similar existing memories in scope, then:
          - DELETE  if the text is an explicit retraction AND closely matches one (cos>=0.72).
          - UPDATE  if semantic cosine >= 0.65 AND shared content-bigram (or cos>=0.90 duplicate).
          - ADD     otherwise (conservative: never drop new information).

        Performance: embeddings are batched (one API call). I/O boundaries are
        strictly ordered: network I/O (embedding, LLM) FIRST, then read-only
        kNN searches, then ONE write transaction (BEGIN IMMEDIATE -> batch
        mutations -> COMMIT). The SQLite lock is never held across network
        calls.
        """
        store = self._store_get()
        scope = {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}

        cleaned = [t.strip() for t in texts if t.strip()]
        if not cleaned:
            return {"results": []}

        # 1) network I/O: batch-embed everything in one API call
        embs = self.embedder.embed_many(cleaned)

        # 2) read-only phase (NO transaction held): retrieve candidates and
        #    decide for each fact; also collect prior texts for history rows.
        plan: list[tuple[str, str, list]] = []  # (text, emb, existing)
        for text, emb in zip(cleaned, embs):
            existing = []
            try:
                existing = store.semantic_search(emb, top_k=5, filters=scope)
            except Exception:
                existing = []
            plan.append((text, emb, existing))

        # 3) single write transaction: all mutations, one commit (one fsync)
        store.begin()
        try:
            results = []
            for text, emb, existing in plan:
                op = self._decide(text, emb, existing)

                if op["event"] == "ADD":
                    mid = store.insert(
                        text, emb, user_id=user_id, agent_id=agent_id, run_id=run_id,
                        metadata=metadata, memory_type=memory_type, in_txn=True,
                    )
                    results.append({"id": mid, "memory": text, "event": "ADD"})
                elif op["event"] == "UPDATE":
                    mid = op["id"]
                    store.update_memory(mid, text, emb, in_txn=True)
                    results.append({"id": mid, "memory": text, "event": "UPDATE"})
                elif op["event"] == "DELETE":
                    mid = op["id"]
                    store.delete(mid, in_txn=True)
                    results.append({"id": mid, "memory": None, "event": "DELETE"})
            store.commit()
        except Exception:
            store.rollback()
            raise
        return {"results": results}

    @staticmethod
    def _decide(text: str, emb, existing: list, cos_update=0.65, cos_delete=0.72):
        """Pure decision function (unit-testable, no store/io dependence).

        UPDATE gate = shared content-bigrams (attribute key phrase survives a
        value change; a different fact about the same entity shares none).
        ADD is the conservative default — a wrong ADD is a duplicate, a wrong
        UPDATE/DELETE loses information.

        cos_update (0.65) is the UPDATE + duplicate-guard gate: a mistake only
        rephrases a memory, recoverable. cos_delete (0.72) is higher: DELETE is
        destructive. Calibrated on live bge-base-en-v1.5 embeddings: genuine
        retraction-of-stated-fact pairs score 0.77+, unrelated 0.38-0.52, mild
        topical overlap 0.65-0.70. Crucially the threshold only gates texts
        that ALREADY passed _is_retraction (explicit "forget X" intent), so a
        low-ball side risk is a missed DELETE (a duplicate), never data loss.
        """
        if not existing:
            return {"event": "ADD"}

        best = max(existing, key=lambda m: m.get("score", 0.0))
        cos_ = float(best.get("score", 0.0))  # cosine similarity (1 - distance)
        best_text = best["memory"]

        # DELETE: explicit retraction intent (already gate-checked) AND a close
        # match (0.72 — higher than unrelated/topical pairs at 0.38-0.70, and
        # below the 0.769 live cosine of retraction-vs-its-target-fact).
        if _is_retraction(text):
            if cos_ >= cos_delete:
                return {"event": "DELETE", "id": best["id"]}
            # retraction intent but no close match: nothing to retract -> ADD no-op
            return {"event": "ADD"}

        # UPDATE: same claim, changed value/rewording — cosine near AND the
        # attribute key phrase (shared content-bigram) survives the edit.
        if cos_ >= cos_update and _shared_bigrams(text, best_text):
            return {"event": "UPDATE", "id": best["id"], "cos": cos_,
                    "shared": sorted(_shared_bigrams(text, best_text))}

        # Near-duplicate guard: a new text that is almost identical to an
        # existing memory (cos >= 0.90) is treated as an UPDATE (in-place
        # refresh) rather than a duplicate ADD — the store does not bloat.
        if cos_ >= 0.90:
            return {"event": "UPDATE", "id": best["id"], "cos": cos_}

        # ADD: everything else — including high-cosine + no shared bigram
        # (same entity, different fact) which must be kept as a separate memory.
        return {"event": "ADD", "cos": cos_}

    @staticmethod
    def _to_texts(messages) -> list[str]:
        if isinstance(messages, str):
            return [messages]
        if isinstance(messages, dict):
            messages = [messages]
        texts = []
        for m in messages:
            content = m.get("content")
            if m.get("role") == "system" or content is None:
                continue
            texts.append(str(content))
        return texts

    # ---------- search ----------
    def search(self, query: str, *, filters: dict | None = None, top_k: int = 5,
               strategy: str = "semantic"):
        """Search memories. strategy: 'semantic' (sqlite-vec), 'keyword' (FTS5),
        or 'hybrid' (fused)."""
        store = self._store_get()
        qv = self.embedder.embed(query)
        if strategy == "semantic":
            rows = store.semantic_search(qv, top_k=top_k, filters=filters)
            return [self._shape(r, "semantic") for r in rows]
        if strategy == "keyword":
            rows = store.keyword_search(query, top_k=top_k, filters=filters)
            return [self._shape(r, "keyword") for r in rows]
        # hybrid: RRF (reciprocal rank fusion) — rank-based, scale-free, the
        # standard way to merge top-k lists from different scorers (bm25 rank
        # and cosine similarity are not directly comparable; raw positions are).
        sem = store.semantic_search(qv, top_k=top_k * 2, filters=filters)
        kw = store.keyword_search(query, top_k=top_k * 2, filters=filters)
        K = 60
        rrf: dict[str, dict] = {}  # id -> shaped row with rrf_score
        for rank, r in enumerate(sem):
            d = self._shape(r, "semantic")
            d["rrf_score"] = 1.0 / (K + rank + 1)
            rrf[d["id"]] = d
        for rank, r in enumerate(kw):
            d = self._shape(r, "keyword")
            contribution = 1.0 / (K + rank + 1)
            if d["id"] in rrf:
                rrf[d["id"]]["rrf_score"] += contribution
            else:
                d["rrf_score"] = contribution
                rrf[d["id"]] = d

        # recency tie-breaker: newer memories beat older ones at equal RRF rank.
        # score = rrf_score * (BASE + (1-BASE) * decay), decay halves every 30 days.
        now = time.time()
        items = list(rrf.values())
        for d in items:
            age = _age_days(d.get("updated_at"), now)
            if age is None:
                decay = 1.0
            else:
                decay = DECAY_HALF_LIFE_DAYS / (DECAY_HALF_LIFE_DAYS + age)
            d["score"] = d.pop("rrf_score") * (
                RECENCY_BASE + (1.0 - RECENCY_BASE) * decay
            )

        items.sort(key=lambda x: x["score"], reverse=True)
        return items[:top_k]

    # ---------- reflect (hindsight-inspired synthesis) ----------
    def reflect(self, query: str, *, filters: dict | None = None, top_k: int = 5):
        """Synthesize an answer from the recalled memories (Hindsight's 'reflect').

        Retrieves the top memories, then has the LLM compose a grounded, disposition-
        aware answer from them. Falls back to returning the raw hits if no LLM is
        configured or the call fails — reflect never blocks recall.
        """
        hits = self.search(query, filters=filters, top_k=top_k, strategy="hybrid")
        if not hits or self._llm_client is None:
            return {"answer": None, "memories": hits, "synthesized": False}

        evidence = "\n".join(f"- {h['memory']}" for h in hits)
        prompt = (
            "You are an assistant answering a question from the user's stored memories.\n"
            "Use ONLY the evidence below; if it does not answer the question, say so.\n\n"
            f"Question: {query}\n\nRelevant memories:\n{evidence}\n\n"
            "Answer concisely in 1-3 sentences, grounded strictly in the evidence."
        )
        try:
            r = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            answer = r.choices[0].message.content.strip()
            return {"answer": answer, "memories": hits, "synthesized": True}
        except Exception:
            return {"answer": None, "memories": hits, "synthesized": False}

    @staticmethod
    def _shape(row: dict, source: str) -> dict:
        score = row.get("score")
        # keyword bm25 is negative/lower-better; normalize to 0..1 higher-better
        if source == "keyword":
            score = 1.0 / (1.0 + abs(float(score or 0)))
        return {
            "id": row["id"],
            "memory": row["memory"],
            "memory_type": row.get("memory_type", "world_fact"),
            "score": float(score) if score is not None else 0.0,
            "metadata": row.get("metadata") or {},
            "user_id": row.get("user_id"),
            "agent_id": row.get("agent_id"),
            "run_id": row.get("run_id"),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "source": source,
        }

    # ---------- get_all ----------
    def get_all(self, *, filters: dict | None = None, limit: int | None = None) -> dict:
        rows = self._store_get().list_all(filters=filters, limit=limit)
        return {"results": [self._shape(r, "list") for r in rows]}

    # ---------- update / delete ----------
    def update(self, memory: str, memory_id: str) -> dict:
        emb = self.embedder.embed(memory)
        ok = self._store_get().update_memory(memory_id, memory, emb)
        if not ok:
            return {"results": []}
        return {"results": [{"id": memory_id, "memory": memory, "event": "UPDATE"}]}

    def delete(self, memory_id: str) -> dict:
        ok = self._store_get().delete(memory_id)
        return {"results": [{"id": memory_id, "event": "DELETE"}] if ok else []}

    def delete_all(self, *, user_id=None, agent_id=None, run_id=None) -> dict:
        """Single-transaction batch delete (no fetch-then-loop N commits)."""
        store = self._store_get()
        rows = store.list_all(filters={"user_id": user_id, "agent_id": agent_id, "run_id": run_id})
        store.begin()
        try:
            for r in rows:
                store.delete(r["id"], in_txn=True)
            store.commit()
        except Exception:
            store.rollback()
            raise
        return {"results": [{"id": r["id"], "event": "DELETE"} for r in rows]}

    def reset(self):
        if self._store:
            self._store.reset()

    def close(self):
        if self._store:
            self._store.close()
