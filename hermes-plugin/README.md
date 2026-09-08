# MemLite — Hermes Memory Provider Plugin

Single-file SQLite semantic memory for Hermes Agent: `sqlite-vec` + FTS5 in
one `.db` file, LLM fact extraction on write, deterministic
ADD/UPDATE/DELETE reconciliation, hybrid RRF + recency recall.

MemLite implements the Hermes `MemoryProvider` ABC (`agent/memory_provider.py`)
and is integrated via `memory.provider: memlite` — exactly like the in-tree
providers (honcho, hindsight, …). It ships as a **standalone repo** (per
Hermes `plugins/AGENTS.md`: `plugins/memory/` is closed to new providers; new
backends are standalone repos discovered through the same path).

> **Version:** this guide targets **memlite ≥ v1.0.1** (adds `post_setup`, so
> `hermes memory setup memlite` works).

---

## 1. Install the engine into the Hermes venv

Hermes uses ONE shared venv (`~/.hermes/hermes-agent/venv`) for **all** profiles.
That venv ships **without `pip`** (stripped for install size), so install with
`uv` targeting it directly — **not** `pip`:

```bash
uv pip install --python ~/.hermes/hermes-agent/venv/bin/python \
  "memlite @ git+https://github.com/AnkitArya/memlite.git@v1.0.1"
```

Verify it resolves (must print the repo path — the editable/venv copy the
plugin imports):

```bash
~/.hermes/hermes-agent/venv/bin/python -c "import memlite; print(memlite.__version__, memlite.__file__)"
```

> Pin the version you want (`@v1.0.1`) or omit `@…` for latest `main`. The
> engine's `memlite/` package is what the provider imports
> (`from memlite import Memory`).

---

## 2. Deploy the provider plugin (flat under `$HERMES_HOME/plugins/`)

Hermes discovers memory providers from `$HERMES_HOME/plugins/<name>/` — a
**flat** layout rooted directly under the profile's `plugins/` dir (NOT
`plugins/memory/`). The provider dir is this `hermes-plugin/` folder:

```bash
PLUGIN_SRC="$(pwd)/hermes-plugin"        # this repo
mkdir -p ~/.hermes/plugins
cp -r "$PLUGIN_SRC" ~/.hermes/plugins/memlite
# or symlink so edits are live (symlinks work for plugin discovery):
#   ln -s "$PLUGIN_SRC" ~/.hermes/plugins/memlite
```

**Per named profile:** each profile has its own `HERMES_HOME`, so it needs its
own copy/symlink and does NOT inherit the top-level dir:

```bash
for prof in marketing trading-debby; do
  mkdir -p ~/.hermes/profiles/$prof/plugins
  ln -s ~/.hermes/plugins/memlite ~/.hermes/profiles/$prof/plugins/memlite
done
```

---

## 3. Activate via `hermes memory setup` (recommended) or config

### Recommended: `hermes memory setup`

`hermes memory setup` → pick `memlite`, or run non-interactively:

```bash
hermes memory setup memlite
```

Because memlite implements `post_setup(hermes_home, config)` (v1.0.1+), this
normalizes `memory.provider: memlite`, persists an explicit `plugins.memlite`
block with defaults, and saves config. Secrets are prompted separately and
written to `.env` (see below).

### Or set config directly

```bash
hermes config set memory.provider memlite
```

---

## 4. Provide credentials (`.env` — secrets only)

MemLite needs an embeddings + extraction-LLM endpoint (DeepInfra by default).
The API key goes in `~/.hermes/.env` (or the active profile's `.env`); nothing
secret is written to `config.yaml`:

```bash
echo "DEEPINFRA_API_KEY=..." >> ~/.hermes/.env
```

`is_available()` checks for `DEEPINFRA_API_KEY` or `OPENAI_API_KEY` (plus the
`openai` and `sqlite-vec` imports). Without one, the provider reports
`unavailable_reason` with an actionable message and the agent still starts.

---

## 5. Optional tuning (defaults shown)

```yaml
plugins:
  memlite:
    embedding_base_url: https://api.deepinfra.com/v1/openai   # any OpenAI-compatible embedding endpoint
    embedding_model:    BAAI/bge-base-en-v1.5
    llm_base_url:       https://api.deepinfra.com/v1/openai
    llm_model:          deepseek-ai/DeepSeek-V3
    db_path: ""         # empty -> $HERMES_HOME/memlite.db
    user_scope: ""      # empty -> per-session user_id
    top_k: 5
```

---

## 6. Restart & verify

CLI/gateway: restart the process. **Desktop app:** the agent backend is the
`hermes-dashboard` systemd unit, which caches the code it loaded at boot — a
fresh `pip install` or plugin edit requires a restart before desktop chats see
it:

```bash
sudo systemctl restart hermes-dashboard.service
```

Verify:

```bash
hermes memory status          # should report memlite as active
hermes memlite status         # provider CLI (if cli.py wired)
~/.hermes/hermes-agent/venv/bin/python -c "import sqlite_vec, openai"  # deps present
```

> memlite activation is **lazy per agent turn** — the "Memory provider
> 'memlite' activated" log and a fresh `memlite.db` appear on the profile's
> next conversation turn, not at process boot.

---

## What you get per turn

- **prefetch** — cached recall from the previous turn's background search,
  injected as `<relevant_user_memories>` (zero prompt-latency)
- **sync_turn** — the user+assistant exchange is distilled by the extraction
  LLM into durable facts, then deterministically reconciled (ADD / UPDATE /
  DELETE) against the store in one SQLite transaction — all non-blocking
- **tools** — `memlite_search`, `memlite_add`, `memlite_forget`
- **never blocks** — background failures are logged, never raised into the
  turn; `sync_turn` runs only in the `primary` agent context (cron/subagent
  turns never write user memory)

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
- Dead endpoint during background sync → logged; conversations never block
  on memory.
- sync_turn runs only in the `primary` agent context.
- `404 invalid_api_key` on embeddings → you passed a non-DeepInfra key to the
  DeepInfra endpoint; memlite reads `OPENAI_API_KEY` before
  `DEEPINFRA_API_KEY`, so if `.env` holds an `nvapi-` key, set
  `embedding_config.api_key` explicitly to `DEEPINFRA_API_KEY` (or set the base
  URL to match the key's provider).

## Migrating from another provider

Switching `memory.provider` does **not** auto-migrate old facts. To carry
existing facts / `memories/*.md` into memlite, `add_raw` each fact (already
extracted, no second LLM pass) via the library:

```python
from memlite import Memory
m = Memory({"llm": {"config": llm_cfg}, "embedder": {"config": emb_cfg}}, db_path="memlite.db")
for fact in old_facts:
    m.add_raw(fact, user_id="default")
```
