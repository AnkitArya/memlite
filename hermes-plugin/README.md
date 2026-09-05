# MemLite — Hermes Memory Provider Plugin

Single-file SQLite semantic memory for Hermes Agent: `sqlite-vec` + FTS5 in
one `.db` file, LLM fact extraction on write, deterministic
ADD/UPDATE/DELETE reconciliation, hybrid RRF + recency recall.

## Setup

```bash
pip install "memlite @ git+https://github.com/AnkitArya/memlite.git"   # into the hermes venv
hermes config set memory.provider memlite
echo "DEEPINFRA_API_KEY=..." >> ~/.hermes/.env
hermes gateway restart   # or restart the CLI
hermes memory status     # should report memlite as active
```

Optional tuning (defaults in parentheses):

```yaml
plugins:
  memlite:
    embedding_base_url: null  # (https://api.deepinfra.com/v1/openai)
    embedding_model: null     # (BAAI/bge-base-en-v1.5)
    llm_base_url: null        # (https://api.deepinfra.com/v1/openai)
    llm_model: null           # (deepseek-ai/DeepSeek-V3)
    db_path: null             # ($HERMES_HOME/memlite.db)
    user_scope: null          # (per-session user_id)
    top_k: 5
```

## What you get per turn

- **prefetch** — cached recall from the previous turn's background search,
  injected as `<relevant_user_memories>` (zero prompt-latency)
- **sync_turn** — the user+assistant exchange is distilled by the extraction
  LLM into durable facts, then deterministically reconciled (ADD / UPDATE /
  DELETE) against the store in one SQLite transaction — all non-blocking
- **tools** — `memlite_search`, `memlite_add`, `memlite_forget`

## CLI

```
hermes memlite status
hermes memlite list
hermes memlite search "favorite color" --strategy hybrid
hermes memlite stats
hermes memlite forget <memory_id>
```

## Failure behavior

- No `DEEPINFRA_API_KEY`/`OPENAI_API_KEY` → provider unavailable with an
  actionable reason; agent still starts.
- Dead endpoint during background sync → circuit breaker pauses for 2 min
  after 5 consecutive failures; conversations never block on memory.
- sync_turn runs only in the `primary` agent context (cron/subagent turns
  never write user memory).
