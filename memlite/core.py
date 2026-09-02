"""Memory - the public API, mirroring mem0.Memory (add/search/get_all/update/delete).

Design notes (lean on purpose):
  - Everything lives in one SQLite file via Store.
  - `infer=True` (default) uses an LLM to extract discrete memories, like mem0.
  - `infer=False` stores raw text chunks directly.
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

_RECONCILE_PROMPT = """You maintain a user's memory store. The user just made a statement.
Your ONLY job is to reflect that statement into the memory store as exactly ONE
operation. NEVER invent facts, names, or topics that are not present in the
user's statement.

The user's statement is:
"{conversation}"

Decide the single operation to apply:
- DELETE: the statement explicitly retracts something that is in the Existing
  Memories (e.g. "forget X", "I never liked X", "remove what I said about X").
  Reference that existing memory's id; omit text.
- UPDATE: the statement clearly replaces or refines an Existing Memory with the
  SAME topic (e.g. favorite color changed from teal to magenta). Reference that
  existing memory's id and provide the new full text, phrased as a durable fact.
- ADD: otherwise, the statement is new information. Provide the full text as a
  durable, self-contained fact (rewrite "I/my" -> "User"). Keep it faithful to
  the statement — change only pronoun and tense, never the content.

ALWAYS echo the user's actual content. If the example statement mentions a color
change, your output must mention that exact color change — never replace it with
an unrelated topic.

Return STRICTLY this JSON, no prose, no markdown fences:
{{"memory": [{{"event": "ADD", "text": "..."}}]}}
or for update/delete:
{{"memory": [{{"event": "UPDATE", "id": "<existing id>", "text": "..."}}]}}
{{"memory": [{{"event": "DELETE", "id": "<existing id>"}}]}}

Existing Memories (only these ids are valid; [] means none):
{existing_memories}
"""


def _parse_ops(text: str) -> list[dict]:
    """Parse the reconcile LLM output into a list of op dicts. Tolerant of fences."""
    import json

    # cut to the first {...} block if there's prose around it
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return []
    try:
        val = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    ops = val.get("memory", []) if isinstance(val, dict) else []
    out = []
    for o in ops:
        if not isinstance(o, dict):
            continue
        ev = str(o.get("event", "")).upper()
        if ev in ("ADD", "UPDATE", "DELETE"):
            out.append({"event": ev, "id": o.get("id"), "text": o.get("text")})
    return out


# ---------------------------------------------------------------------------
# Deterministic (no-LLM) reconcile
#
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


def _jaccard(a: str, b: str) -> float:
    """Token-set Jaccard overlap in [0,1]; 1 = identical content words.

    Kept as a diagnostic helper — the reconcile decision itself uses
    _shared_bigrams, which separates value-change from same-entity-different-fact
    (Jaccard scores both ~0.33 and cannot).
    """
    sa, sb = _content_tokens(a), _content_tokens(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


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

        # LLM for extraction (optional). If no api key / config, infer falls back to raw.
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
        infer=True,
        memory_type="world_fact",
        reconcile_with_llm=False,
    ):
        """Create memory/memories. mirrors mem0 (incl. UPDATE/DELETE reconciliation).

        messages: str | dict | list[dict].
        infer=True: reconcile each new piece of information against existing
          memories — deterministically (no LLM) by default:
            ADD (new fact) | UPDATE (reworded/changed same claim) | DELETE (retraction).
          Pass reconcile_with_llm=True (and an LLM in config) to use an LLM for the
          same decision instead of the arithmetic rule.
        infer=False: raw text chunks stored directly as ADD (no LLM).
        memory_type: 'world_fact' (default) or 'experience' (hindsight-inspired).

        Returns: {"results": [{"id", "memory", "event"}, ...]} with event in
          ADD | UPDATE | DELETE.
        """
        texts = self._to_texts(messages)
        if not texts:
            return {"results": []}

        # Raw path (infer=False): plain ADD, one transaction, batch embeddings.
        if not infer:
            return self._raw_add(
                texts, user_id=user_id, agent_id=agent_id, run_id=run_id,
                metadata=metadata, memory_type=memory_type,
            )

        # infer=True: 1) extract discrete facts from the conversation (LLM if
        # available, otherwise the raw chunks are treated as already-extracted),
        # 2) reconcile each extracted fact against the store (ADD/UPDATE/DELETE).
        has_llm = self._llm_client is not None
        # Retraction statements bypass extraction: the extractor would rephrase
        # "Forget X" into a plain fact and the DELETE intent would be lost.
        reclaimed = [t for t in texts if _is_retraction(t)]
        normal = [t for t in texts if t not in reclaimed]
        extracted = self._extract(normal) if (has_llm and normal) else []
        if not extracted:
            extracted = list(normal)  # never silently drop data
        extracted.extend(reclaimed)

        return self._reconcile_many(
            extracted, user_id=user_id, agent_id=agent_id, run_id=run_id,
            metadata=metadata, memory_type=memory_type,
            use_llm=(has_llm and reconcile_with_llm),
        )

    def _raw_add(self, texts, *, user_id, agent_id, run_id, metadata, memory_type):
        """Plain ADD path with batched embedding (one API call for N texts)."""
        cleaned = [t.strip() for t in texts if t.strip()]
        if not cleaned:
            return {"results": []}
        embs = self.embedder.embed_many(cleaned)
        results = []
        for mem, emb in zip(cleaned, embs):
            mid = self._store_get().insert(
                mem, emb, user_id=user_id, agent_id=agent_id,
                run_id=run_id, metadata=metadata, memory_type=memory_type,
            )
            results.append({"id": mid, "memory": mem, "event": "ADD"})
        return {"results": results}

    def _reconcile_many(self, texts, *, user_id, agent_id, run_id, metadata,
                        memory_type, use_llm=False):
        if use_llm:
            return self._reconcile_add(
                texts, user_id=user_id, agent_id=agent_id, run_id=run_id,
                metadata=metadata, memory_type=memory_type,
            )
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
          - DELETE  if the text is an explicit retraction AND closely matches one.
          - UPDATE  if semantic cosine >= COS_UPDATE and token jaccard >= JAC_UPDATE
                    (same claim reworded or changed).
          - ADD     otherwise (conservative: never drop new information).
        """
        store = self._store_get()
        scope = {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}

        results = []
        for text in texts:
            text = text.strip()
            if not text:
                continue
            emb = self.embedder.embed(text)

            # retrieve top similar existing memories in the same scope
            existing = []
            try:
                existing = store.semantic_search(emb, top_k=5, filters=scope)
            except Exception:
                existing = []

            op = self._decide(text, emb, existing)

            if op["event"] == "ADD":
                mid = store.insert(
                    text, emb, user_id=user_id, agent_id=agent_id, run_id=run_id,
                    metadata=metadata, memory_type=memory_type,
                )
                results.append({"id": mid, "memory": text, "event": "ADD"})
            elif op["event"] == "UPDATE":
                mid = op["id"]
                # normalize pronouns to a durable fact
                new_text = text
                store.update_memory(mid, new_text, emb)
                results.append({"id": mid, "memory": new_text, "event": "UPDATE"})
            elif op["event"] == "DELETE":
                mid = op["id"]
                store.delete(mid)
                results.append({"id": mid, "memory": None, "event": "DELETE"})
        return {"results": results}

    @staticmethod
    def _decide(text: str, emb, existing: list, cos_update=0.65):
        """Pure decision function (unit-testable, no store/io dependence).

        UPDATE gate = shared content-bigrams (attribute key phrase survives a
        value change; a different fact about the same entity shares none).
        ADD is the conservative default — a wrong ADD is a duplicate, a wrong
        UPDATE/DELETE loses information.
        """
        if not existing:
            return {"event": "ADD"}

        best = max(existing, key=lambda m: m.get("score", 0.0))
        cos_ = float(best.get("score", 0.0))  # cosine similarity (1 - distance)
        best_text = best["memory"]

        # DELETE: explicit retraction intent AND close match to an existing fact
        if _is_retraction(text):
            if cos_ >= cos_update:
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

    def _reconcile_add(self, texts, *, user_id, agent_id, run_id, metadata, memory_type):
        """Single-call reconcile: semantic-retrieve existing, LLM decides ops, apply."""
        store = self._store_get()
        conversation = "\n".join(f"- {t}" for t in texts)
        scope = {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}

        # 1) retrieve existing memories in same scope (best-effort, never fatal)
        existing = []
        try:
            qv = self.embedder.embed(conversation)
            existing = store.semantic_search(qv, top_k=8, filters=scope)
        except Exception:
            existing = []

        existing_block = "\n".join(
            f'{{"id": "{m["id"]}", "text": "{m["memory"]}"}}' for m in existing
        ) or "[]"

        prompt = _RECONCILE_PROMPT.format(
            conversation=conversation, existing_memories=existing_block,
        )

        # 2) LLM decides ops
        ops = []
        try:
            r = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            ops = _parse_ops(r.choices[0].message.content)
        except Exception:
            ops = []  # fall through to raw ADD below; never drop data

        # 3) apply ops (defensive: validate ids before any write)
        results = []
        valid_ids = {m["id"] for m in existing}
        applied_ops = []
        for op in ops:
            event = op.get("event")
            text = (op.get("text") or "").strip()
            if event == "ADD" and text:
                applied_ops.append(("ADD", text, None))
            elif event == "UPDATE" and text and op.get("id") in valid_ids:
                applied_ops.append(("UPDATE", text, op["id"]))
            elif event == "DELETE" and op.get("id") in valid_ids:
                applied_ops.append(("DELETE", None, op["id"]))

        # If the LLM produced nothing usable, fall back to adding the raw chunks.
        if not applied_ops:
            applied_ops = [("ADD", t.strip(), None) for t in texts if t.strip()]

        for event, text, mid in applied_ops:
            if event == "ADD":
                emb = self.embedder.embed(text)
                new_id = store.insert(
                    text, emb, user_id=user_id, agent_id=agent_id, run_id=run_id,
                    metadata=metadata, memory_type=memory_type,
                )
                results.append({"id": new_id, "memory": text, "event": "ADD"})
            elif event == "UPDATE":
                emb = self.embedder.embed(text)
                store.update_memory(mid, text, emb)
                results.append({"id": mid, "memory": text, "event": "UPDATE"})
            elif event == "DELETE":
                store.delete(mid)
                results.append({"id": mid, "memory": None, "event": "DELETE"})
        return {"results": results}

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
        store = self._store_get()
        rows = store.list_all(filters={"user_id": user_id, "agent_id": agent_id, "run_id": run_id})
        for r in rows:
            store.delete(r["id"])
        return {"results": [{"id": r["id"], "event": "DELETE"} for r in rows]}

    def reset(self):
        if self._store:
            self._store.reset()

    def close(self):
        if self._store:
            self._store.close()
