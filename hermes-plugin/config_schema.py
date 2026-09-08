"""Dashboard config panel for the memlite memory provider.

Found next to the plugin's __init__.py by the memory plugin discovery
system — no import needed. Follows the bundled config_schema.py shape.
"""

# Single source of truth for provider defaults (imported by __init__.py
# for _build_memory/post_setup so the three copies can't drift).
DEFAULTS = {
    "embedding_base_url": "https://api.deepinfra.com/v1/openai",
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "llm_base_url": "https://api.deepinfra.com/v1/openai",
    "llm_model": "deepseek-ai/DeepSeek-V3",
    "db_path": "",
    "user_scope": "",
    "top_k": 5,
}

SCHEMA = {
    "name": "memlite",
    "description": "MemLite — single-file SQLite semantic memory "
                   "(sqlite-vec + FTS5, RRF hybrid recall, LLM fact extraction).",
    "fields": [
        {
            "key": "embedding_base_url",
            "label": "Embedding base URL",
            "type": "string",
            "default": DEFAULTS["embedding_base_url"],
            "help": "Any OpenAI-compatible embeddings endpoint.",
        },
        {
            "key": "embedding_model",
            "label": "Embedding model",
            "type": "string",
            "default": DEFAULTS["embedding_model"],
        },
        {
            "key": "llm_base_url",
            "label": "Extraction LLM base URL",
            "type": "string",
            "default": DEFAULTS["llm_base_url"],
        },
        {
            "key": "llm_model",
            "label": "Extraction LLM model",
            "type": "string",
            "default": DEFAULTS["llm_model"],
        },
        {
            "key": "db_path",
            "label": "Database path",
            "type": "string",
            "default": DEFAULTS["db_path"],
            "help": "Leave empty to use $HERMES_HOME/memlite.db.",
        },
        {
            "key": "user_scope",
            "label": "User scope (user_id filter)",
            "type": "string",
            "default": DEFAULTS["user_scope"],
            "help": "Empty = scope memories per session id.",
        },
        {
            "key": "top_k",
            "label": "Memories to recall per turn",
            "type": "integer",
            "default": DEFAULTS["top_k"],
            "minimum": 1,
            "maximum": 50,
        },
    ],
}
