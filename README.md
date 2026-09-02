# MemLite

A lean, dependency-light reimplementation of [mem0](https://github.com/mem0ai/mem0)'s
memory API — `add / search / get_all / update / delete` — backed by **a single
SQLite file with real semantic (vector) search**.

Built because mem0 OSS drags in a separate vector database (Qdrant), which on
constrained / headless hosts causes a whole class of failures:

| Problem in mem0 OSS | How MemLite fixes it |
|---|---|
| Needs a Qdrant server or embedded store | **One SQLite file** — no process, no service |
| Embedded Qdrant is single-client (one process locks it) | **WAL mode** — multiple processes can read safely, writes serialize via sqlite's own locking |
| EOL embedder models (e.g. `nv-embed-v1` → 410 Gone) | Point at any OpenAI-compatible embeddings API; nothing baked in |
| Vector store and row store are two systems (eventual-consistency headaches) | Vectors, rows, FTS index and history all live in **the same transactional DB** |
| Heavy deps (pydantic, qdrant-client, grpc, …) | Just `sqlite-vec` + `openai` |

## How semantic search works

`sqlite-vec` adds a `vec0` virtual table inside the same SQLite file. On `add()`
the text is embedded (default `BAAI/bge-base-en-v1.5`, 768-dim, via DeepInfra —
cheap and free-tier friendly) and stored alongside the row. On `search()` the
query is embedded and a **cosine k-NN** runs over the vector table, joined back
to the row data and scoped by `user_id / agent_id / run_id`. A parallel **FTS5**
index powers `keyword` mode and `hybrid` fusion (semantic + BM25).

```
what does alice like to eat?  →  [0.23] User loves spicy Indian food   ← top hit
                                [-0.01] User's favorite color is teal
```

## Quickstart

```bash
pip install -e .          # needs sqlite-vec + openai
export OPENAI_API_KEY=... # or DEEPINFRA_API_KEY (DeepInfra embeddings endpoint)
```

```python
from memlite import Memory

m = Memory({
    "embedder": {"config": {
        "model": "BAAI/bge-base-en-v1.5",
        "openai_base_url": "https://api.deepinfra.com/v1/openai",
    }},
}, db_path="memlite.db")

m.add("User loves spicy Indian food", user_id="alice")
m.add("User's favorite color is teal", user_id="alice")

hits = m.search("what does alice like to eat?", filters={"user_id": "alice"})

all_mem = m.get_all(filters={"user_id": "alice"})          # {"results": [...]}
m.update("User loves spicy Thai food", memory_id=hits[0]["id"])
m.delete(memory_id=hits[0]["id"])
```

`add()` always runs mem0's two-pass pipeline (there is no `infer` flag — LLM
extraction is the only path). **Pass 1 (extraction)**: the conversation is
distilled into discrete, durable facts — small talk, questions and assistant
filler are dropped. Statements with retraction intent ("Forget X") bypass
extraction so the DELETE signal survives. **Pass 2 (reconcile)**: each
extracted fact is reconciled against existing memories
**deterministically — with arithmetic, no LLM call**: retrieve the
top semantically similar memories in scope, then decide

- **DELETE** — the text carries explicit retraction intent ("forget X", "no
  longer ...") *and* closely matches an existing memory (cosine ≥ 0.82 —
  destructive actions need a high bar)
- **UPDATE** — cosine ≥ 0.65 **and** the pair shares a content-word bigram
  (the attribute key phrase — "favorite color", "works as" — survives a value
  change, while a *different* fact about the same entity shares none), or
  cosine ≥ 0.90 (near-duplicate guard: reworded re-adds refresh in place
  instead of bloating the store)
- **ADD** — everything else. Conservative default: a wrong ADD is a duplicate,
  a wrong UPDATE/DELETE loses information. Known conservative miss: a value
  change with *no* lexical residue ("lives in Hyderabad" → "moved to
  Bangalore") stores as a duplicate ADD rather than an in-place update.

```python
m.add("My favorite color is teal", user_id="alice")
m.add("Actually, my favorite color changed to magenta now", user_id="alice")
# → [{"id": "<teal-id>", "memory": "...magenta...", "event": "UPDATE"}]

m.add("Forget that I said I love pineapple on pizza", user_id="alice")
# → [{"id": "<pineapple-id>", "memory": None, "event": "DELETE"}]
```

The LLM's only role is **extraction** (pass 1). The ADD/UPDATE/DELETE decision
(pass 2) is always the deterministic arithmetic rule — no LLM call, no prompt
injection surface, no provider dependency in the reconcile path. An LLM is
**required** for `add()` (it raises `ValueError` without one); `search`,
`get_all`, `update`, `delete` work without it.

### Hindsight-inspired additions

Two light features borrowed from [vectorize-io/hindsight](https://github.com/vectorize-io/hindsight)'s
retain / recall / reflect model (mem0 doesn't have these):

- **`memory_type`** — tag a memory as `"world_fact"` (default) or `"experience"`,
  and filter on it in `search` / `get_all`:
  ```python
  m.add("The sky is blue on clear days", user_id="sam", memory_type="world_fact")
  m.add("I visited the Taj Mahal last Tuesday", user_id="sam", memory_type="experience")
  m.get_all(filters={"user_id": "sam", "memory_type": "experience"})
  ```
- **`reflect(query)`** — recall the top memories, then have the LLM compose a
  grounded, disposition-aware answer (a synthesis pass beyond raw recall):
  ```python
  r = m.reflect("where did the user travel recently?", filters={"user_id": "sam"})
  # {"answer": "The user recently traveled to the Taj Mahal last Tuesday.",
  #  "memories": [...hits...], "synthesized": True}
  ```
  `reflect` never blocks recall — if no LLM is configured or the call fails it
  returns `{"synthesized": False, "memories": hits}`.

> Why not port more of hindsight? Its embedded mode uses a closed `pg0://` binary
> engine (not sqlite-vec) and its core value (biomimetic memory types, entity
> graphs, mental models, consolidation) is exactly the heavyweight machinery this
> lean store deliberately omits. `memory_type` + `reflect` capture most of the
> user-facing value at negligible complexity.

Hybrid mode fuses the two lists with **reciprocal rank fusion (RRF, K=60)** —
rank-based and scale-free, unlike a raw bm25/cosine blend — then applies a
**recency decay** tie-breaker (`score *= 0.5 + 0.5 * 30/(30 + age_days)`), so a
newer, contradicting memory outranks the stale one it replaced. See
[docs/flows.md](docs/flows.md) for sequence diagrams of every flow.

## Config knobs

- `embedder.config.model` — any OpenAI-compatible embeddings model
- `embedder.config.openai_base_url` — DeepInfra / NVIDIA / OpenAI / Groq, etc.
- `embedder.config.embedding_dims` — set if you know it (else probed on first use)
- `vector_store.config.path` — where the `.db` file lives (default `memlite.db`)
- `search(strategy=...)` — `"semantic"` (default) | `"keyword"` | `"hybrid"`

## Storage layout (one file)

| Table | Purpose |
|---|---|
| `memories` | canonical rows: id, scope (user/agent/run), text, metadata, timestamps |
| `memory_vectors` | `sqlite-vec` `vec0` cosine index (same DB, transactional) |
| `memories_fts` | FTS5 keyword index (backup retrieval path) |
| `history` | ADD / UPDATE / DELETE audit log |

## Scope & honesty

This is an independent, ground-up implementation of mem0's public API surface,
not a fork of mem0's internals. It now covers mem0's core **reconcile** behavior
(ADD / UPDATE / DELETE on `add`), plus semantic + keyword + hybrid recall. It
still deliberately omits mem0's **entity graph / cross-memory relations,
temporal reasoning, procedural-memory pathway, vision, and cross-encoder
reranking** — those are the heavy parts. If your recall needs the full
production-grade ranking, use mem0. If you need cheap, single-file,
no-vector-server semantic memory with real conflict resolution, this is it.

## License

Apache 2.0.
