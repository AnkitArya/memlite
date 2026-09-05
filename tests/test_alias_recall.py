"""Live regression for related-term recall (horoscope<->zodiac case).

Simulates the user's real failure: fact stored mentioning "zodiac",
query phrased with "horoscope". Both retrieval legs must hit.

Run from repo root with the venv active:
    . .venv/bin/activate && python tests/test_alias_recall.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _env(key):
    for line in open(os.path.expanduser("~/.hermes/.env")):
        line = line.strip()
        if line.startswith(key + "="):
            return line.partition("=")[2].strip().strip('"').strip("'")
    return None


key = _env("DEEPINFRA_API_KEY") or _env("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = key
os.environ.setdefault("DEEPINFRA_API_KEY", key)

from memlite import Memory

db = "/tmp/memlite_alias.db"
if os.path.exists(db):
    os.remove(db)

cfg = {
    "llm": {"config": {
        "model": "deepseek-ai/DeepSeek-V3",
        "openai_base_url": "https://api.deepinfra.com/v1/openai"}},
    "embedder": {"config": {
        "model": "BAAI/bge-base-en-v1.5",
        "openai_base_url": "https://api.deepinfra.com/v1/openai"}},
}
m = Memory(cfg, db_path=db)

# ---- 1. write-side aliasing via add_raw (tool path) ----
r = m.add_raw("User's zodiac sign is Gemini and they prefer orange tones",
              user_id="u1",
              aliases=["horoscope", "sun sign", "astrology"])
assert r["results"], r
mid = r["results"][0]["id"]

import sqlite3
c = sqlite3.connect(db); c.row_factory = sqlite3.Row
row = c.execute("SELECT aliases, memory, user_id FROM memories WHERE mem_id=?", (mid,)).fetchone()
print("1. row stored:", dict(row) | {"memory": row["memory"][:50]})
assert row["user_id"] == "u1"
aliases_stored = json.loads(row["aliases"])
assert "horoscope" in aliases_stored, aliases_stored

# ---- 2. FTS5 alias column populated ----
fts = c.execute("SELECT aliases FROM memories_fts WHERE id=?", (mid,)).fetchone()
assert fts and "horoscope" in (fts["aliases"] or ""), fts
print("2. FTS corpus contains aliases: OK")

# ---- 3. KEYWORD leg recovery: 'horoscope' query must recall the zodiac fact ----
hits = m.search("whats my horoscope", filters={"user_id": "u1"}, strategy="keyword")
print("3. keyword leg hits:", [(round(h["score"], 4), h["memory"]) for h in hits])
assert any(h["id"] == mid for h in hits), "keyword leg missed!"

# ---- 4. SEMANTIC leg recovery (alias-boosted centroid) ----
sem = m.search("whats my horoscope", filters={"user_id": "u1"}, strategy="semantic")
print("4. semantic leg hits:", [(round(h["score"], 4), h["memory"]) for h in sem])
assert any(h["id"] == mid for h in sem), "semantic leg missed"

# ---- 5. HYBRID (what the agent actually uses) ----
hyb = m.search("whats my horoscope", filters={"user_id": "u1"}, strategy="hybrid")
assert any(h["id"] == mid for h in hyb), hyb
print("5. hybrid top hit:", hyb[0]["memory"])

# ---- 6. extraction pass generates aliases (v2 format) ----
facts = m._extract(["I just adopted a beagle named Rover and he loves the beach"])
print("6. extraction output:", facts)
assert facts and isinstance(facts[0], dict) and facts[0].get("text")
al = facts[0].get("aliases")
assert isinstance(al, list) and al, "extractor produced no aliases"
print("6. extractor aliases:", al)
stored = m.add([f["text"] for f in facts] + ["ignore this"], user_id="u2")
assert any("beagle" in (x["memory"] or "") for x in stored["results"] if x["event"] != "DELETE")

# ---- 7. no regression: unrelated keyword leg does NOT recall via OR-expansion ----
# "quantum mechanics homework" shares no vocabulary with the zodiac fact and
# no synonyms — only the OR-union keyword leg could wrongly surface it.
unrel_kw = m.search("quantum mechanics homework", filters={"user_id": "u1"}, strategy="keyword")
assert not any(h["id"] == mid for h in unrel_kw), unrel_kw
print("7. unrelated query stays clean (keyword leg)")

print("\nALL ALIAS-RECALL CHECKS PASSED")
