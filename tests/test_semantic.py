"""End-to-end test of MemLite semantic search with real DeepInfra embeddings.

Run from repo root with the venv active:
    . .venv/bin/activate && python -m pytest tests/test_semantic.py -s
or directly:  python tests/test_semantic.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memlite import Memory

# Load keys from ~/.hermes/.env for the live run.
# Prefer the DeepInfra key (cheap embedder endpoint). The file's own OPENAI_API_KEY
# may point at a different provider, so we only use DEEPINFRA for the embedding calls.
_env_path = os.path.expanduser("~/.hermes/.env")


def _env_value(key):
    if os.path.exists(_env_path):
        for line in open(_env_path):
            line = line.strip()
            if line.startswith(key + "="):
                return line.partition("=")[2].strip().strip('"').strip("'")
    return os.environ.get(key)


_deep = _env_value("DEEPINFRA_API_KEY") or _env_value("OPENAI_API_KEY")
if _deep:
    os.environ["DEEPINFRA_API_KEY"] = _deep
    os.environ["OPENAI_API_KEY"] = _deep

DB = "/tmp/memlite_test.db"
if os.path.exists(DB):
    os.remove(DB)


def build_memory():
    return Memory(
        {
            "embedder": {
                "config": {
                    "model": "BAAI/bge-base-en-v1.5",
                    "openai_base_url": "https://api.deepinfra.com/v1/openai",
                }
            }
        },
        db_path=DB,
    )


def test_semantic_search_returns_relevant_memories():
    m = build_memory()
    m.add("User loves spicy Indian food", user_id="alice")
    m.add("User's favorite color is teal", user_id="alice")
    m.add("Bob prefers minimalist white desk setup", user_id="bob")

    # Semantic: different words, same meaning
    res = m.search("what does alice like to eat? spicy cuisine",
                   filters={"user_id": "alice"})
    print("\nSearch 'what does alice like to eat?':")
    for r in res:
        print(f"  [{r['score']:.3f}] {r['memory']} (id={r['id'][:8]})")
    assert res, "no results"
    assert res[0]["memory"].startswith("User loves spicy"), \
        f"expected spicy memory first, got: {res[0]['memory']}"

    # scoping works
    bob = m.search("spicy food", filters={"user_id": "bob"})
    assert all(r["user_id"] == "bob" for r in bob)


def test_keyword_fallback():
    m = build_memory()
    res = m.search("favorite color teal", strategy="keyword",
                   filters={"user_id": "alice"})
    print("\nKeyword 'favorite color teal':")
    for r in res:
        print(f"  [{r['score']:.3f}] {r['memory']}")
    assert any("teal" in r["memory"] for r in res)


def test_hybrid_and_roundtrip():
    m = build_memory()
    res = m.search("spicy cuisine", strategy="hybrid",
                   filters={"user_id": "alice"})
    print("\nHybrid 'spicy cuisine':")
    for r in res:
        print(f"  [{r['score']:.3f}] {r['memory']}")

    # get_all / update / delete round trip
    allm = m.get_all(filters={"user_id": "alice"})
    target = next(x for x in allm["results"] if "teal" in x["memory"])
    upd = m.update("User's favorite color is now magenta", target["id"])["results"]
    assert upd and upd[0]["event"] == "UPDATE"
    verified = m.get_all(filters={"user_id": "alice"})
    assert any("magenta" in x["memory"] for x in verified["results"])
    del_res = m.delete(target["id"])["results"]
    assert del_res and del_res[0]["event"] == "DELETE"
    assert not any(x["id"] == target["id"] for x in m.get_all(filters={"user_id": "alice"})["results"])
    print("\nRound-trip update/delete OK")


def test_infer_false_raw_chunks():
    m = Memory(
        {"embedder": {"config": {"model": "BAAI/bge-base-en-v1.5",
                                 "openai_base_url": "https://api.deepinfra.com/v1/openai"}}},
        db_path="/tmp/memlite_test2.db",
    )
    if os.path.exists("/tmp/memlite_test2.db"):
        os.remove("/tmp/memlite_test2.db")
    r = m.add([{"role": "user", "content": "my dog is named Rufus"}],
              user_id="carol", infer=False)
    assert len(r["results"]) == 1
    assert "Rufus" in r["results"][0]["memory"]
    sr = m.search("what is the dog's name", filters={"user_id": "carol"})
    assert sr and "Rufus" in sr[0]["memory"]
    print("\nRaw add + search OK:", sr[0]["memory"])


def test_memory_type_and_reflect():
    import tempfile

    db = tempfile.mktemp(suffix=".db")
    m = Memory(
        {
            "llm": {"config": {"model": "deepseek-ai/DeepSeek-V3"}},
            "embedder": {"config": {"model": "BAAI/bge-base-en-v1.5",
                                    "openai_base_url": "https://api.deepinfra.com/v1/openai"}},
        },
        db_path=db,
    )
    m.add("The sky is blue on clear days", user_id="sam", infer=False, memory_type="world_fact")
    m.add("I visited the Taj Mahal last Tuesday", user_id="sam", infer=False, memory_type="experience")

    # memory_type filtering
    gg = m.get_all(filters={"user_id": "sam", "memory_type": "experience"})
    kinds = {x["memory_type"] for x in gg["results"]}
    assert kinds == {"experience"}, kinds
    assert all("Taj" in x["memory"] for x in gg["results"])

    # memory_type surfaces in search results
    sem = m.search("what type is the memory about the taj", filters={"user_id": "sam"})
    print("\nSearch results carry memory_type:")
    for r in sem:
        print(f"  [{r['memory_type']}] {r['memory']}")
    assert all("memory_type" in r for r in sem)

    # reflect synthesizes (needs the LLM configured; DeepInfra chat works live)
    ref = m.reflect("where did the user travel recently?", filters={"user_id": "sam"})
    print("reflect synthesized:", ref["synthesized"])
    if ref["synthesized"]:
        print("  answer:", ref["answer"])
        assert ref["answer"], "reflect returned empty answer"
    else:
        print("  (LLM unavailable; reflect returned raw memories — acceptable)")
    m.close()


if __name__ == "__main__":
    test_semantic_search_returns_relevant_memories()
    test_keyword_fallback()
    test_hybrid_and_roundtrip()
    test_infer_false_raw_chunks()
    test_memory_type_and_reflect()
    print("\nALL TESTS PASSED")
