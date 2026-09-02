"""Regression: scoped semantic search must not return 0 rows when the global
kNN window is flooded with other tenants' memories (over-fetch trap).

db has 200 rows for 'noise' users and 3 rows for 'target'; a query closest to
target's memories must still find them in scope.
"""
import os, sys
import sqlite_vec
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB = "/tmp/memlite_overfetch.db"
if os.path.exists(DB):
    os.remove(DB)

import os as _os
def _env(key):
    for line in open(_os.path.expanduser("~/.hermes/.env")):
        line = line.strip()
        if line.startswith(key + "="):
            return line.partition("=")[2].strip().strip('"').strip("'")
    return None

k = _env("DEEPINFRA_API_KEY") or _env("OPENAI_API_KEY")
_os.environ["OPENAI_API_KEY"] = k
_os.environ.setdefault("DEEPINFRA_API_KEY", k)

from memlite.store import Store
from memlite.embedder import Embedder

emb = Embedder({"model": "BAAI/bge-base-en-v1.5",
                "openai_base_url": "https://api.deepinfra.com/v1/openai"})
store = Store(DB, dims=emb.dims or len(emb.embed(".")))

# target's memories: cooking
q = "what does target like to cook?"
target_texts = [
    "User loves cooking italian pasta on weekends",
    "User's favorite dish is homemade lasagna",
    "User enjoys baking sourdough bread",
]
# noise: 200 unrelated rows for other users (random-ish distinct topics)
noise_bases = [
    "fiscal quarterly report numbers spreadsheet",
    "gaming keyboard rgb lighting setup",
    "yoga stretching routine morning",
    "crypto wallet hardware security",
    "houseplant watering schedule fern",
]
# simpler: build 200 distinct-ish noise rows
noise = []
for i in range(200):
    base = noise_bases[i % 5]
    noise.append(f"User note {i}: {base} reference {i*7}")

texts = target_texts + noise
labels = ["target"] * 3 + ["noise"] * 200
vecs = emb.embed_many(texts)
store.begin()
for t, v in zip(texts, vecs):
    store.insert(t, v, user_id="noise" if "note" in t else "target", in_txn=True)
store.commit()

qv = emb.embed("what does target like to cook on weekends?")
rows = store.semantic_search(qv, top_k=3, filters={"user_id": "target"})
print("scoped hits:", [r["memory"] for r in rows])
assert rows, "over-fetch trap reproduced: 0 scoped rows"
assert all(r["user_id"] == "target" for r in rows)
assert any("cook" in r["memory"] for r in rows), rows
print("PASS: scoped search finds target memories despite 200 noise rows")
