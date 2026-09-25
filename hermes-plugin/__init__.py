"""MemLite memory provider for Hermes Agent.

Implements the MemoryProvider ABC backed by the memlite engine: a single
SQLite file (memories + sqlite-vec vec0 + FTS5) with an LLM
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
      llm_model: null           # default: deepseek-ai/DeepSeek-V4-Flash-0731
      llm_api_key: null         # template like ${DEEPINFRA_API_KEY}
      user_scope: null          # user_id filter; empty -> per-session id
      top_k: 5
"""

from __future__ import annotations

import json
import logging
import os
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from agent.memory_provider import MemoryProvider, RecallStatus
    _HOST_AVAILABLE = True
except ImportError:  # tests outside the hermes-agent cwd
    MemoryProvider = object
    RecallStatus = None

try:
    from .config_schema import DEFAULTS
except ImportError:  # loaded as a flat package shell without config_schema
    DEFAULTS = {
        "embedding_base_url": "https://api.deepinfra.com/v1/openai",
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "llm_base_url": "https://api.deepinfra.com/v1/openai",
        "llm_model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "db_path": "",
        "user_scope": "",
        "top_k": 5,
        # retention: "all" stores every reconciled fact; "selective" merges
        # near-duplicate paraphrases (OpenViking-style) to curb store bloat.
        "retention": "all",
        # Background per-turn capture (mirrors mem0 sync_turn). Enabled by default;
        # truncation + durable-fact extraction guard against the working-note leak.
        "sync_turn_enabled": True,
        "sync_max_chars": 450,
    }

logger = logging.getLogger(__name__)

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
        "Store a durable fact about the user (verbatim — no LLM extraction pass). "
        "Call this the moment the user states a lasting preference, correction, decision, "
        "or personal detail worth recalling on future turns — don't wait to be asked to "
        "remember. The fact should be a clean, self-contained statement like "
        "'User prefers snake_case in Python'. Skip transient chit-chat and facts you've "
        "already stored. Optionally pass aliases: 2-4 closely-related retrieval terms "
        "(synonyms/super-categories, e.g. horoscope↔zodiac) so differently-phrased future "
        "queries still recall this fact."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "fact": {"type": "string", "description": "Discrete durable fact statement"},
            "aliases": {"type": "array", "items": {"type": "string"},
                         "description": "2-4 related retrieval terms (synonyms/associated vocabulary)"},
        },
        "required": ["fact"],
    },
}

_PROMPT_BODY = (
    "You have persistent memory of this user from past conversations. You should call "
    "memlite_search before answering anything that could depend on prior context (the user's "
    "preferences, facts, history, people, projects, or earlier decisions) — do not rely on the "
    "chat window alone, and do not assume you have no memory.\n"
    "When the user states a durable preference, correction, decision, or personal detail worth "
    "recalling later, call memlite_add immediately to store it — don't wait to be asked.\n"
    "Tools: memlite_search to recall facts, memlite_add to store facts, memlite_forget to "
    "remove by id, memlite_history to audit a fact's revisions."
)

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

_HISTORY_SCHEMA = {
    "name": "memlite_history",
    "description": "Get the audit trail for one memory by id (from memlite_search / memlite_add results). "
                   "Returns that memory's revisions newest-first, each with event (ADD|UPDATE|DELETE), "
                   "old_memory, new_memory, actor_id, created_at — so an overwritten or deleted value is recoverable.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Exact memory id"},
            "limit": {"type": "integer", "description": "Max revisions to return (default 50)"},
        },
        "required": ["memory_id"],
    },
}

_PURGE_SCHEMA = {
    "name": "memlite_purge",
    "description": "Delete ALL memories in a scope (this user unless overridden), including the audit trail. "
                   "At least one of user_id / agent_id / run_id is required; use with care — this is not undoable. "
                   "Returns how many memories were deleted.",
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "Scope: user (defaults to current user)"},
            "agent_id": {"type": "string", "description": "Scope: agent"},
            "run_id": {"type": "string", "description": "Scope: run"},
        },
        "required": [],
    },
}


def _expand(value: str) -> str:
    """Expand ${VAR} templates in config values (env-based secrets)."""
    return os.path.expandvars(value) if value else value


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
        "model": config.get("embedding_model") or DEFAULTS["embedding_model"],
        "openai_base_url": (config.get("embedding_base_url")
                            or DEFAULTS["embedding_base_url"]),
        "api_key": emb_key,
    }
    llm_cfg = {
        "model": config.get("llm_model") or DEFAULTS["llm_model"],
        "openai_base_url": (config.get("llm_base_url")
                            or DEFAULTS["llm_base_url"]),
        "api_key": llm_key,
    }
    return Memory(
        {
            "llm": {"config": llm_cfg},
            "embedder": {"config": emb_cfg},
            "retention": config.get("retention") or "all",
        },
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
        self._prefetch_cache: List[Dict[str, Any]] = []
        self._last_prefetch_count: Optional[int] = None
        self._lock = threading.Lock()
        self._sync_lock = threading.Lock()
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_turn_enabled = bool(self._config.get("sync_turn_enabled", True))
        self._sync_max_chars = int(self._config.get("sync_max_chars", 450) or 450)
        self._closed = False

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
        home = (kwargs.get("hermes_home")
                or getattr(self, "_hermes_home", None)
                or os.environ.get("HERMES_HOME"))
        if home:
            self._hermes_home = Path(home)
        else:
            # mirror hermes_constants resolution: profile-aware default
            try:
                from hermes_constants import get_hermes_home
                self._hermes_home = get_hermes_home()
            except Exception:
                self._hermes_home = Path.home() / ".hermes"
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
        if self._closed or self._mem is None:
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
            except Exception as e:
                logger.error("MemLite prefetch failed: %s", e)
                logger.debug("%s", traceback.format_exc())

        threading.Thread(target=_worker, name="memlite-worker", daemon=True).start()

    # -- background per-turn capture (sync_turn) -----------------------------
    #
    # `sync_turn` is the MemoryProvider ABC hook MemoryManager.sync_all() calls
    # after every turn. It was originally disabled (memlite issue #1) because
    # feeding raw transcripts to the extractor leaked working-notes as durable
    # facts. Re-enabled with mem0-derived safeguards that fix that leak:
    #   1) each message is truncated at a sentence boundary to `sync_max_chars`
    #      (default 450), so long scratch/analysis dumps never reach the LLM;
    #   2) extraction still goes through the engine's durable-fact pass
    #      (`add()` → `_extract`), which distills discrete facts and runs
    #      deterministic ADD/UPDATE/DELETE reconcile (no duplicated rows);
    #   3) it runs on a background thread with a join-guard so it never blocks
    #      the turn and never double-ingests a slow previous sync.
    # Gated on config `plugins.memlite.sync_turn_enabled` (default true).

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "") -> None:
        if self._closed or self._mem is None or not self._sync_turn_enabled:
            return

        def _truncate(text: str, max_len: int) -> str:
            if not text or len(text) <= max_len:
                return text or ""
            window = text[:max_len]
            cut = max(window.rfind(sep) for sep in ("。", "！", "？", ".", "!", "?"))
            if cut > max_len // 3:
                return window[:cut + 1]
            return window

        def _worker() -> None:
            try:
                uid = self._user_scope()
                messages = [
                    _truncate(user_content, self._sync_max_chars),
                    _truncate(assistant_content, self._sync_max_chars),
                ]
                messages = [m for m in messages if m.strip()]
                if not messages:
                    return
                result = self._mem.add(
                    messages, user_id=uid,
                    metadata={"write_origin": "sync_turn", "session_id": session_id or self._session_id},
                )
                results = result.get("results", []) if isinstance(result, dict) else []
                if results:
                    # Leak guard: if the extractor returned [] and the engine fell
                    # back to storing a raw input message verbatim (issue #1 path),
                    # drop it — the whole point of sync_turn is distilled FACTS, not
                    # transcripts. Only delete memories whose text is byte-identical
                    # to a truncated input (i.e. the fallback), never real extractions.
                    raw_set = {m for m in messages if m.strip()}
                    for r in results:
                        stored = (r.get("memory") or "").strip()
                        if stored in raw_set:
                            try:
                                self._mem.delete(r.get("id")) \
                                    if hasattr(self._mem, "delete") else None
                                logger.warning("MemLite sync_turn: dropped raw-fallback memory")
                            except Exception:
                                pass
                    logger.info("MemLite sync_turn: %s (%d facts)",
                                ", ".join(r.get("event", "?") for r in results), len(results))
            except Exception as e:
                # never let a background sync crash the turn or trip a breaker
                logger.error("MemLite sync_turn failed: %s", e)
                logger.debug("%s", traceback.format_exc())

        with self._sync_lock:
            prev = self._sync_thread
            if prev and prev.is_alive():
                prev.join(timeout=5.0)
                if prev.is_alive():  # still busy: skip to avoid duplicate ingestion
                    return
            self._sync_thread = threading.Thread(
                target=_worker, name="memlite-sync", daemon=True)
            self._sync_thread.start()

    # -- tools -----------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [_SEARCH_SCHEMA, _ADD_SCHEMA, _FORGET_SCHEMA, _HISTORY_SCHEMA, _PURGE_SCHEMA]

    def system_prompt_block(self) -> str:
        """STATIC system-prompt text telling the agent to proactively recall and
        store memories (mirrors mem0's _PROMPT_BODY). The model drives all writes
        explicitly via memlite_add (no background sync_turn)."""
        return f"# MemLite Memory\nActive.\n{_PROMPT_BODY}"

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        import json
        if self._mem is None:
            # initialize() not run in this process (or failed) — tool calls
            # would crash with NoneType; report it structurally instead.
            try:
                self.initialize(getattr(self, "_session_id", "") or "tool-fallback",
                                platform="tool-call")
            except Exception as e:
                return json.dumps({"error": f"memlite not initialized: {e}"})
            if self._mem is None:
                return json.dumps({"error": "memlite not initialized"})
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
                r = self._mem.add_raw(fact, user_id=self._user_scope(),
                                      aliases=args.get("aliases"))
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
            if tool_name == "memlite_history":
                mid = (args.get("memory_id") or "").strip()
                if not mid:
                    return json.dumps({"error": "memory_id is required (see memlite_search)"})
                hist = self._mem.history(mid, limit=int(args.get("limit", 50)))
                return json.dumps({"ok": True, "memory_id": mid, "history": hist})
            if tool_name == "memlite_purge":
                kwargs = {}
                for k in ("user_id", "agent_id", "run_id"):
                    v = args.get(k) or (self._user_scope() if k == "user_id" else None)
                    if v and str(v).strip():
                        kwargs[k] = str(v).strip()
                if "user_id" not in kwargs and "agent_id" not in kwargs and "run_id" not in kwargs:
                    return json.dumps({"error": "memlite_purge needs at least one of user_id/agent_id/run_id"})
                r = self._mem.delete_all(**kwargs)
                return json.dumps({"ok": True, "deleted": r.get("deleted", 0), "scope": kwargs})
            return json.dumps({"error": f"unknown tool {tool_name}"})
        except Exception as e:
            logger.error("MemLite tool call failed: %s", e)
            logger.debug("%s", traceback.format_exc())
            return json.dumps({"error": str(e)})

    # -- built-in memory tool bridge ----------------------------------------------

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mirror Hermes' built-in in-turn ``memory`` tool writes into the memlite
        store (the MemoryProvider ``on_memory_write`` hook).

        Hermes calls this whenever the model commits a ``memory(action=add|replace|remove,
        target=memory|user, ...)`` call (``MemoryManager.notify_memory_tool_write`` → here).
        ``metadata`` carries ``old_text`` (for replace/remove), ``session_id``, ``tool_name``,
        ``task_id``. This is the primary long-term-memory write path now that the background
        ``sync_turn`` is disabled (issue #1).
        """
        if self._closed or self._mem is None:
            return  # not initialized / shutting down — fail soft
        if not (action or "").strip():
            return
        uid = self._user_scope()
        if not isinstance(metadata, dict):
            metadata = {}
        meta = dict(metadata)
        meta.setdefault("write_origin", "memory_tool")
        meta.setdefault("target", target or "memory")
        # target is informational for now; the store is single-user "default".
        try:
            action = action.strip().lower()
            text = (content or "").strip()
            if action == "add":
                if text:
                    self._mem.add_raw(text, user_id=uid, metadata=meta)
                return
            old_text = (meta.get("old_text") or "").strip()
            mid = self._find_by_text(old_text, uid)
            if action == "remove":
                if mid:
                    self._mem.delete(mid)
                return
            if action == "replace":
                if mid and text:
                    self._mem.update(text, mid)
                elif text:
                    # no exact old_text match: fall back to deterministic reconcile (may UPDATE a near-dup)
                    self._mem.add_raw(text, user_id=uid, metadata=meta)
                return
            logger.warning("MemLite on_memory_write: ignoring unknown action %r", action)
        except Exception as e:
            logger.error("MemLite on_memory_write failed (%s): %s", action, e)
            logger.debug("%s", traceback.format_exc())

    def _find_by_text(self, text: str, user_id: str) -> Optional[str]:
        """Return the id of a memory whose text contains *text* (used for replace/remove
        old_text matching). Best-effort exact/substring match; None if ambiguous or absent."""
        if not text:
            return None
        try:
            all_rows = self._mem.get_all(filters={"user_id": user_id}).get("results", [])
        except Exception:
            return None
        matched = [r for r in all_rows if text and text in (r.get("memory") or "")]
        if len({r["memory"] for r in matched}) != 1:
            # ambiguous or none (distinct entries) — don't guess
            return None
        return matched[0]["id"] if matched else None

    # -- session lifecycle ------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        self.shutdown()

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self._session_id = new_session_id

    def shutdown(self) -> None:
        self._closed = True
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

    def post_setup(self, hermes_home: str, config: dict) -> None:
        """Non-interactive post-setup: activate memlite and persist an explicit
        ``plugins.memlite`` block with schema defaults, so ``hermes memory setup
        memlite`` leaves config.yaml self-consistent. Secrets are handled by the
        schema's ``env_var`` (DEEPINFRA_API_KEY) via the generic setup path, so
        nothing secret is written here.

        Mirrors the in-tree providers' contract (memory_setup.py: ``_post_setup_hook``
        normalizes the ``memory`` block; a provider with ``post_setup`` owns config
        persistence). memlite is fully self-configuring — every option has a usable
        default — so this is a normalization, not an interactive wizard.
        """
        config.setdefault("memory", {})["provider"] = self.name
        mem_cfg = config.setdefault("plugins", {}).setdefault(self.name, {})
        for k, v in DEFAULTS.items():
            mem_cfg.setdefault(k, v)
        try:
            from hermes_cli.config import save_config
            save_config(config)
        except Exception:
            logger.exception("MemLite post_setup could not persist config")

    def backup_paths(self) -> List[str]:
        db_path = self._config.get("db_path")
        return [str(db_path)] if db_path else []  # default db lives inside hermes_home

    # -- internal ------------------------------------------------------------------

    def _user_scope(self) -> str:
        return self._config.get("user_scope") or self._session_id or "default"


def register(ctx) -> None:
    """Register the memlite memory provider with the Hermes plugin system."""
    ctx.register_memory_provider(MemLiteProvider())
