"""Live extraction-quality check: status reports must not become memories.

Run: .venv/bin/python tests/test_extract_quality.py   (live: DeepInfra LLM)

Guards the `_EXTRACT_PROMPT` "still true in six months" rule. Two layers keep
stale pseudo-facts out of the store — sync_turn only feeds the user side of a
turn (see hermes-plugin/__init__.py), and the extractor itself refuses status
narration. This script checks the second layer directly, so it also covers the
public `Memory.add()` path.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _env(key):
    try:
        with open(os.path.expanduser("~/.hermes/.env")) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(key + "="):
                    return line.partition("=")[2].strip().strip('"').strip("'")
    except OSError:
        pass
    return None


key = _env("DEEPINFRA_API_KEY") or _env("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = key or ""
os.environ.setdefault("DEEPINFRA_API_KEY", key or "")

from memlite import Memory  # noqa: E402

passed, failed = [], []


def check(name, ok, detail=""):
    (passed if ok else failed).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))


def fresh_memory():
    tmp = tempfile.mkdtemp(prefix="memlite-extract-")
    return Memory(
        {
            "llm": {"config": {"model": "deepseek-ai/DeepSeek-V4-Flash-0731"}},
            "embedder": {"config": {"model": "BAAI/bge-base-en-v1.5",
                                    "openai_base_url": "https://api.deepinfra.com/v1/openai"}},
        },
        db_path=os.path.join(tmp, "extract.db"),
    )


# 1. A durable preference must still be captured.
mem = fresh_memory()
res = mem.add(["I always want the PR description to list the tests I ran."],
              user_id="u", metadata={"write_origin": "test"})
facts = [r["memory"] for r in res.get("results", [])]
mem.close()
check("durable preference is stored", len(facts) == 1, str(facts))

# 2. Status / progress narration must be skipped.
mem = fresh_memory()
res = mem.add(["Done — I opened PR #35 (4 commits, 13 files); the 14-config audit "
               "run completed with exit code 0 after 42 minutes."],
              user_id="u", metadata={"write_origin": "test"})
facts = [r["memory"] for r in res.get("results", [])]
mem.close()
check("PR/exit-code narration is skipped", facts == [], str(facts))

# 3. Deploy/commit chatter must be skipped.
mem = fresh_memory()
res = mem.add(["Committed and pushed 3d10e39 to origin; restarted the gateway and "
               "deployed the build."],
              user_id="u", metadata={"write_origin": "test"})
facts = [r["memory"] for r in res.get("results", [])]
mem.close()
check("commit/deploy narration is skipped", facts == [], str(facts))

# 4. "Active." smoke fact (the probe residue pattern) has no durable content.
mem = fresh_memory()
res = mem.add(["memlite-e2e-probe-20261003 works"], user_id="u",
              metadata={"write_origin": "test"})
facts = [r["memory"] for r in res.get("results", [])]
mem.close()
check("probe artifact is skipped", facts == [], str(facts))

print(f"\nRESULTS: {len(passed)} passed, {len(failed)} failed")
if failed:
    for name in failed:
        print(f"  FAIL: {name}")
    raise SystemExit(1)
print("ALL EXTRACTION-QUALITY CHECKS PASSED")
