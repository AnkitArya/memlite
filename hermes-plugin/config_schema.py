"""Dashboard config panel for the memlite memory provider.

Found next to the plugin's __init__.py by the memory plugin discovery
system — no import needed. Follows the bundled config_schema.py shape.
"""

SCHEMA = {
    "name": "memlite",
    "description": "MemLite — single-file SQLite semantic memory "
                   "(sqlite-vec + FTS5, RRF hybrid recall, LLM fact extraction).",
    "fields": [
        {
            "key": "embedding_base_url",
            "label": "Embedding base URL",
            "type": "string",
            "default": "https://api.deepinfra.com/v1/openai",
            "help": "Any OpenAI-compatible embeddings endpoint.",
        },
        {
            "key": "embedding_model",
            "label": "Embedding model",
            "type": "string",
            "default": "BAAI/bge-base-en-v1.5",
        },
        {
            "key": "llm_base_url",
            "label": "Extraction LLM base URL",
            "type": "string",
            "default": "https://api.deepinfra.com/v1/openai",
        },
        {
            "key": "llm_model",
            "label": "Extraction LLM model",
            "type": "string",
            "default": "deepseek-ai/DeepSeek-V3",
        },
        {
            "key": "db_path",
            "label": "Database path",
            "type": "string",
            "default": "",  # empty -> $HERMES_HOME/memlite.db
            "help": "Leave empty to use $HERMES_HOME/memlite.db.",
        },
        {
            "key": "user_scope",
            "label": "User scope (user_id filter)",
            "type": "string",
            "default": "",
            "help": "Empty = scope memories per session id.",
        },
        {
            "key": "top_k",
            "label": "Memories to recall per turn",
            "type": "integer",
            "default": 5,
            "minimum": 1,
            "maximum": 50,
        },
    ],
}
