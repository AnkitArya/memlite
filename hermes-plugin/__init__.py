"""MemLite memory provider for Hermes Agent.

Implements the MemoryProvider ABC backed by the memlite engine: a single
SQLite file (memories + sqlite-vec vec0 + FTS5 + history) with an LLM
fact-extraction pass on add(), deterministic ADD/UPDATE/DELETE reconcile,
and hybrid (RRF + recency) recall.

Config in $HERMES_HOME/config.yaml (profile-scoped):
  plugins:
    memlite:
      db_path: null             # omit -> $HERMES_HOME/memlite.db
      embedding_base_url: null  # default: DeepInfra endpoint
      embedding_model: null     # default: BAAI/bge-base-en-v1.5
      embedding_api_key: null   # template like ${DEEPINFRA_API_KEY}
      llm_base_url: null        # default: DeepInfra endpoint
      llm_model: null           # default: deepseek-ai/DeepSeek-V3
      llm_api_key: null         # template like ${DEEPINFRA_API_KEY}
      user_scope: null          # user_id filter; empty -> per-session id
      top_k: 5
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from agent.memory_provider import MemoryProvider, RecallStatus
    _HOST_AVAILABLE = True
except ImportError:  # tests outside the hermes-agent cwd
    MemoryProvider = object
    RecallStatus = None

logger = logging.getLogger(__name__)

_CIRCUIT_THRESHOLD = 5
_CIRCUIT_OPEN_SECONDS = 120.0

_SEARCH_SCHEMA = {
    "name": "memlite_search",
    "description": (
        "Search MemLite long-term memory for durable user facts, preferences, "
        "and project context. strategy: 'hybrid' (recommended), 'semantic', "
        "or 'keyword'. Returns [{id, memory, score, ...}]."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Semantic/keyword query"},
            "strategy": {"type": "string", "enum": ["hybrid", "semantic", "keyword"],
                         "default": "hybrid"},
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    },
}

_ADD_SCHEMA = {
    "name": "memlite_add",
    "description": (
        "Copy an already-durable fact statement into MemLite long-term memory "
        "(no LLM extraction pass — the fact should be a clean, self-contained "
        "statement like 'User prefers snake_case in Python')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "fact": {"type": "string", "description": "Discrete durable fact statement"},
        },
        "required": ["fact"],
    },
}

_FORGET_SCHEMA = {
    "name": "memlite_forget",
    "description": "Delete one memory by its id (from memlite_search results).",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Exact memory id"},
        },
        "required": ["memory_id"],
    },
}


def _expand(value: str) -> str:
    """Expand ${VAR} templates in config values (env-based secrets)."""
    if not value or "${" not in value:
        return value
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), value)


def _load_plugin_config() -> dict:
    try:
        from hermes_cli.config import load_config_readonly, cfg_get
        all_config = load_config_readonly()
        return cfg_get(all_config, "plugins", "memlite", default={}) or {}
    except Exception:
        return {}


def _build_memory(config: dict, db_path: str):
    """Construct the memlite Memory engine from provider config + env."""
    from memlite import Memory

    emb_key = (_expand(config.get("embedding_api_key"))
               or _expand(config.get("llm_api_key"))
               or os.environ.get("DEEPINFRA_API_KEY")
               or os.environ.get("OPENAI_API_KEY"))
    llm_key = _expand(config.get("llm_api_key")) or emb_key

    emb_cfg = {
        "model": config.get("embedding_model") or "BAAI/bge-base-en-v1.5",
        "openai_base_url": (config.get("embedding_base_url")
                            or "https://api.deepinfra.com/v1/openai"),
        "api_key": emb_key,
    }
    llm_cfg = {
        "model": config.get("llm_model") or "deepseek-ai/DeepSeek-V3",
        "openai_base_url": (config.get("llm_base_url")
                            or "https://api.deepinfra.com/v1/openai"),
        "api_key": llm_key,
    }
    return Memory(
        {"llm": {"config": llm_cfg}, "embedder": {"config": emb_cfg}},
        db_path=db_path,
    )


class MemLiteProvider(MemoryProvider):  # type: ignore[misc,valid-type]
    """MemoryProvider backed by the memlite engine."""

    def __init__(self, config: dict | None = None):
        self._config = config if config is not None else _load_plugin_config()
        self._mem = None
        self._session_id = ""
        self._agent_context = "primary"
        self._hermes_home = None
        self._sync_thread: Optional[threading.Thread] = None
        self._prefetch_cache: List[Dict[str, Any]] = []
        self._last_prefetch_count: Optional[int] = None
        self._lock = threading.Lock()
        self._closed = False
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    # -- ABC surface -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "memlite"

    @property
    def mem(self):
        return self._mem

    def is_available(self) -> bool:
        """Config/deps check only — no network calls."""
        try:
            import openai  # noqa: F401
            import sqlite_vec  # noqa: F401
        except ImportError:
            return False
        import os
        return bool(os.environ.get("DEEPINFRA_API_KEY") or os.environ.get("OPENAI_API_KEY"))

    @property
    def unavailable_reason(self) -> str:
        try:
            import openai  # noqa: F401
            import sqlite_vec  # noqa: F401
        except ImportError:
            return ("install: pip install 'memlite @ "
                    "git+https://github.com/AnkitArya/memlite.git' (openai + sqlite-vec)")
        import os
        if not (os.environ.get("DEEPINFRA_API_KEY") or os.environ.get("OPENAI_API_KEY")):
            return ("set DEEPINFRA_API_KEY (or OPENAI_API_KEY) in ~/.hermes/.env — "
                    "required for the extraction LLM and embeddings")
        return ""

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._agent_context = kwargs.get("agent_context") or "primary"
        self._hermes_home = Path(kwargs.get("hermes_home") or Path.home() / ".hermes")
        db_path = self._config.get("db_path") or str(self._hermes_home / "memlite.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._mem = _build_memory(self._config, db_path)
        logger.info("MemLite provider initialized (db=%s session=%s context=%s)",
                    db_path, session_id, self._agent_context)

    # -- prefetch / sync -------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return the CACHED recall from the previous turn's queue_prefetch.

        Strictly zero-latency: no network I/O here; insert after extraction.
        """
        with self._lock:
            hits = self._prefetch_cache
            self._prefetch_cache = []
        lines = [f"- {h.get('memory', '')}" for h in hits if h.get("memory")]
        return ("<relevant_user_memories>\n" + "\n".join(lines)
                + "\n</relevant_user_memories>\n") if lines else ""

    def recall_status(self) -> Optional["RecallStatus"]:
        with self._lock:
            count = self._last_prefetch_count
            self._last_prefetch_count = None
        if RecallStatus is not None and count:
            return RecallStatus(count=count, source=self.name)
        return None

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Background hybrid search to pre-warm the NEXT turn."""
        if self._closed or self._is_circuit_open() or self._mem is None:
            return

        def _worker():
            try:
                hits = self._mem.search(
                    query,
                    filters={"user_id": self._user_scope()},
                    top_k=int(self._config.get("top_k", 5)),
                    strategy="hybrid",
                )
                with self._lock:
                    self._prefetch_cache = hits
                    self._last_prefetch_count = len(hits)
                self._record_success()
            except Exception as e:
                logger.error("MemLite prefetch failed: %s", e)
                logger.debug("%s", traceback.format_exc())
                self._record_failure()

        self._spawn(_worker)

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages: Optional[List[Dict[str, Any]]] = None):
        """Non-blocking background turn sync (host threading contract)."""
        if self._closed or self._agent_context != "primary":
            return  # cron/subagent prompts must not corrupt user memory
        if self._is_circuit_open():
            logger.warning("MemLite circuit open; skipping background sync.")
            return

        def _worker():
            try:
                if messages:
                    # use the real conversation context Hermes hands us
                    convo = [
                        {"role": m.get("role", "user"),
                         "content": m.get("content") or ""}
                        for m in messages
                        if m.get("role") in ("user", "assistant") and m.get("content")
                    ]
                else:
                    convo = [
                        {"role": "user", "content": user_content or ""},
                        {"role": "assistant", "content": assistant_content or ""},
                    ]
                self._mem.add(convo, user_id=self._user_scope())  # LLM extraction + reconcile
                # speculative prefetch for the next turn
                hits = self._mem.search(
                    user_content or assistant_content or "",
                    filters={"user_id": self._user_scope()},
                    top_k=int(self._config.get("top_k", 5)),
                    strategy="hybrid",
                )
                with self._lock:
                    self._prefetch_cache = hits
                    self._last_prefetch_count = len(hits)
                self._record_success()
            except Exception as e:
                logger.error("MemLite background sync failed: %s", e)
                logger.debug("%s", traceback.format_exc())
                self._record_failure()

        prev = self._sync_thread
        if prev and prev.is_alive():
            prev.join(timeout=5.0)
        self._sync_thread = threading.Thread(target=_worker, name="memlite-sync", daemon=True)
        self._sync_thread.start()

    # -- tools -----------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [_SEARCH_SCHEMA, _ADD_SCHEMA, _FORGET_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        scope = {"user_id": self._user_scope()}
        try:
            if tool_name == "memlite_search":
                hits = self._mem.search(
                    args["query"],
                    filters=scope,
                    strategy=args.get("strategy", "hybrid"),
                    top_k=int(args.get("top_k", 5)),
                )
                # strip embeddings (bloat) before returning to the model
                slim = [{k: v for k, v in h.items() if k != "embedding"} for h in hits]
                return json.dumps({"ok": True, "results": slim})
            if tool_name == "memlite_add":
                fact = (args.get("fact") or "").strip()
                if not fact:
                    return json.dumps({"error": "empty fact"})
                r = self._mem.add_raw(fact)
                out = [{k: v for k, v in x.items() if k != "embedding"}
                       for x in r.get("results", [])]
                return json.dumps({"ok": True, "results": out})
            if tool_name == "memlite_forget":
                mid = (args.get("memory_id") or "").strip()
                if not mid:
                    return json.dumps({"error": "memory_id is required (see memlite_search)"})
                r = self._mem.delete(mid)
                ok = bool(r and r.get("results"))
                return json.dumps({"ok": ok})
            return json.dumps({"error": f"unknown tool {tool_name}"})
        except Exception as e:
            logger.error("MemLite tool call failed: %s", e)
            logger.debug("%s", traceback.format_exc())
            return json.dumps({"error": str(e)})

    # -- session lifecycle ------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        self.shutdown()

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self._session_id = new_session_id

    def shutdown(self) -> None:
        self._closed = True
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)
        if self._mem:
            try:
                self._mem.close()
            except Exception:
                pass
            self._mem = None

    # -- config ------------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "api_key",
             "description": "API key for embeddings + extraction LLM (DeepInfra by default)",
             "secret": True, "required": True, "env_var": "DEEPINFRA_API_KEY",
             "url": "https://deepinfra.com/dash"},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        # secrets go to .env via env_var; nothing to persist natively
        pass

    def backup_paths(self) -> List[str]:
        db_path = self._config.get("db_path")
        return [str(db_path)] if db_path else []  # default db lives inside hermes_home

    # -- internal ------------------------------------------------------------------

    def _user_scope(self) -> str:
        return self._config.get("user_scope") or self._session_id or "default"

    def _is_circuit_open(self) -> bool:
        return time.time() < self._circuit_open_until

    def _record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= _CIRCUIT_THRESHOLD:
                self._circuit_open_until = time.time() + _CIRCUIT_OPEN_SECONDS
                logger.error("MemLite circuit breaker open after %d failures; backing off %ss",
                             _CIRCUIT_THRESHOLD, _CIRCUIT_OPEN_SECONDS)
                self._consecutive_failures = 0

    def _record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0

    def _spawn(self, fn) -> threading.Thread:
        t = threading.Thread(target=fn, name="memlite-worker", daemon=True)
        t.start()
        return t


def register(ctx) -> None:
    """Register the memlite memory provider with the Hermes plugin system."""
    ctx.register_memory_provider(MemLiteProvider())
