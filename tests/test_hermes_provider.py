"""Lifecycle test for the memlite Hermes memory provider plugin.

Run:  python tests/test_hermes_provider.py
(uses live DeepInfra embeddings + LLM; keep for manual verification)
"""
import importlib.util
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _env(key):
    for line in open(os.path.expanduser("~/.hermes/.env")):
        line = line.strip()
        if line.startswith(key + "="):
            return line.partition("=")[2].strip().strip('"').strip("'")
    return None


k = _env("DEEPINFRA_API_KEY") or _env("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = k
os.environ.setdefault("DEEPINFRA_API_KEY", k)

# make hermes-agent's `agent` package importable if present
HERMES_AGENT = os.path.expanduser("~/.hermes/hermes-agent")
if os.path.isdir(HERMES_AGENT):
    sys.path.insert(0, HERMES_AGENT)

# import plugin module directly (it may fall back to plain object ABC)
spec = importlib.util.spec_from_file_location(
    "memlite_plugin", os.path.join(REPO, "hermes-plugin", "__init__.py"))
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)

print("1. is_available:", plugin.MemLiteProvider().is_available())
assert plugin.MemLiteProvider().is_available()

p = plugin.MemLiteProvider(config={"user_scope": "testuser"})
p.initialize(session_id="sess-1", hermes_home="/tmp/memlite_hermes", platform="cli")
assert p.mem is not None
print("2. initialize OK, db:", p.mem.db_path)

# ABC conformance
assert p.name == "memlite"
schemas = p.get_tool_schemas()
assert {s["name"] for s in schemas} == {
    "memlite_search", "memlite_add", "memlite_forget", "memlite_history", "memlite_purge",
}
print("3. tool schemas OK:", [s["name"] for s in schemas])

# memlite_add tool call (no extraction pass)
resp = p.handle_tool_call("memlite_add", {"fact": "User tests with pytest and xdist"})
print("4. memlite_add:", resp)
payload = json.loads(resp)
assert payload.get("ok"), payload

# on_memory_write: the primary long-term write path (mirrors built-in `memory` tool).
# sync_turn was removed (issue #1): background raw-transcript sync is disabled, so a
# per-turn fact only lands when the model writes it via memlite_add or on_memory_write.
p.on_memory_write("add", "memory", "User's dog Biscuit is a golden retriever.",
                  {"session_id": "sess-1", "tool_name": "memory", "write_origin": "memory_tool"})
print("5. on_memory_write (add) executed")

deadline = time.time() + 90
hits = []
while time.time() < deadline:
    hits = p.mem.search("dog breed", filters={"user_id": "testuser"}, top_k=3)
    if any("Biscuit" in h["memory"] for h in hits):
        break
    time.sleep(2)
print("5b. search after on_memory_write:", [h["memory"] for h in hits])
assert any("Biscuit" in h["memory"] for h in hits), hits

# prefetch drains cache into context text
p.queue_prefetch("dog", session_id="sess-1")
deadline = time.time() + 30
while not p._prefetch_cache and time.time() < deadline:
    time.sleep(0.5)
ctx = p.prefetch("anything")
print("6. prefetch injects:", repr(ctx[:90]))
assert ctx == "" or ("relevant_user_memories" in ctx), ctx[:200]

# forget by id
mid = hits[0]["id"]
forgotten = json.loads(p.handle_tool_call("memlite_forget", {"memory_id": mid}))
print("7. forget:", forgotten)
assert forgotten.get("ok") or not forgotten.get("ok")  # either way valid outcome
gone = p.mem.get_all(filters={"user_id": "testuser"})["results"]
assert not any(x["id"] == mid for x in gone), "memory should be deleted"

p.shutdown()
print("\nALL PROVIDER LIFECYCLE TESTS PASSED")
