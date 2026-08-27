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
import uuid

from .embedder import Embedder
from .store import Store

_ADDITIVE_PROMPT = """You are a Memory Extractor. Extract every piece of memorable
information from the conversation as self-contained factual statements. Cover BOTH
user and assistant messages. Capture each distinct topic as a separate memory.

Return ONLY a JSON array of strings, e.g. ["Memory one", "Memory two"]. No prose,
no markdown fences.

Conversation:
{conversation}
"""


def _extract_json_array(text: str) -> list[str]:
    """Pull the first JSON array out of LLM output, tolerating fences/prose."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        import json

        val = json.loads(m.group(0))
        return [str(x) for x in val] if isinstance(val, list) else []
    except json.JSONDecodeError:
        return []


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
    ):
        """Create memory/memories. mirrors mem0.
        messages: str | dict | list[dict]. infer=True uses the LLM to extract.
        memory_type: 'world_fact' (default) or 'experience' (hindsight-inspired)."""
        texts = self._to_texts(messages)
        if not texts:
            return {"results": []}

        memories = self._extract(texts) if infer and self._llm_client else texts
        results = []
        for mem in memories:
            mem = mem.strip()
            if not mem:
                continue
            emb = self.embedder.embed(mem)
            mid = self._store_get().insert(
                mem, emb, user_id=user_id, agent_id=agent_id,
                run_id=run_id, metadata=metadata, memory_type=memory_type,
            )
            results.append({"id": mid, "memory": mem, "event": "ADD"})
        return {"results": results}

    def _extract(self, texts: list[str]) -> list[str]:
        conversation = "\n".join(texts)
        prompt = _ADDITIVE_PROMPT.format(conversation=conversation)
        try:
            r = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return _extract_json_array(r.choices[0].message.content)
        except Exception:
            # LLM failed; fall back to raw chunks so add never silently drops data
            return texts

    @staticmethod
    def _to_texts(messages) -> list[str]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}].__str__() and [messages]
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
        # hybrid: semantic primary, keyword fills gaps, merged by score
        sem = store.semantic_search(qv, top_k=top_k * 2, filters=filters)
        kw = store.keyword_search(query, top_k=top_k * 2, filters=filters)
        merged: dict[str, dict] = {}
        for r in sem:
            d = self._shape(r, "semantic")
            merged[d["id"]] = d
        for r in kw:
            i = r["id"]
            if i in merged:
                merged[i]["score"] = 0.5 * merged[i]["score"] + 0.5 * (1.0 / (1.0 + abs(float(r.get("score") or 0))))
            else:
                d = self._shape(r, "keyword")
                merged[i] = d
        sorted_items = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return sorted_items[:top_k]

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
