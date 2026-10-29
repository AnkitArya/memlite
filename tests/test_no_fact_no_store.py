"""Empty extraction result must mean "nothing to remember", not "store the input".

Before this change ``Memory.add`` treated an empty extraction list as a failure
and stored the raw conversation verbatim. With the extractor now correctly
rejecting status narration ("opened PR #35", "run completed with exit code 0")
that fallback re-inserted exactly the text the prompt had just rejected — the
second half of the stale-fact leak.

The fallback is still required when extraction cannot run at all (no LLM, API
error, unparseable reply): a transient failure must never lose a write.

Fully offline: stub LLM client + stubbed embedder.

Run: python -m pytest tests/test_no_fact_no_store.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OPENAI_API_KEY", "test-not-a-real-key")

from memlite import Memory  # noqa: E402

DIMS = 8
STATUS_TEXT = ("Done — I opened PR #35 (4 commits, 13 files) and the audit run "
               "completed with exit code 0 after 42 minutes.")


class _Reply:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]


class _Completions:
    def __init__(self, content=None, raises=None):
        self._content = content
        self._raises = raises

    def create(self, **_kwargs):
        if self._raises:
            raise self._raises
        return _Reply(self._content)


class _StubLLM:
    def __init__(self, content=None, raises=None):
        self.chat = type("Chat", (), {"completions": _Completions(content, raises)})()


def _memory(tmp_path, llm=None) -> Memory:
    mem = Memory(config={}, db_path=str(tmp_path / "nofact.db"))
    mem.embedder.embed = lambda text: [0.1] * DIMS
    mem.embedder.embed_many = lambda texts: [[0.1] * DIMS for _ in texts]
    mem.dims = DIMS
    mem._llm_client = llm
    mem._llm_model = "stub"
    return mem


def test_empty_extraction_stores_nothing(tmp_path):
    mem = _memory(tmp_path, _StubLLM('{"memories": []}'))
    try:
        result = mem.add([STATUS_TEXT], user_id="u")
        assert result["results"] == [], result
        assert mem.get_all(filters={"user_id": "u"})["results"] == []
    finally:
        mem.close()


def test_real_extraction_is_stored(tmp_path):
    mem = _memory(tmp_path, _StubLLM(
        '{"memories": [{"text": "User always wants the tests listed in the PR body."}]}'))
    try:
        result = mem.add(["whatever"], user_id="u")
        assert len(result["results"]) == 1, result
        stored = mem.get_all(filters={"user_id": "u"})["results"]
        assert len(stored) == 1 and "PR body" in stored[0]["memory"]
    finally:
        mem.close()


def test_missing_llm_is_an_error_not_a_silent_raw_store(tmp_path):
    """No configured extractor: ``add`` refuses up front (documented contract)."""
    mem = _memory(tmp_path, llm=None)
    try:
        try:
            mem.add(["Note for later: deploy script requires sudo access."], user_id="u")
        except ValueError as exc:
            assert "requires an LLM" in str(exc)
        else:  # pragma: no cover - contract guard
            raise AssertionError("add() without an LLM must raise ValueError")
    finally:
        mem.close()


def test_llm_failure_still_stores_raw_text(tmp_path):
    """API error / unparseable reply is a failure, not an editorial decision."""
    mem = _memory(tmp_path, _StubLLM(raises=RuntimeError("upstream 400")))
    try:
        result = mem.add(["Note for later: the staging DB lives on port 5433."], user_id="u")
        facts = [r["memory"] for r in result["results"]]
        assert facts == ["Note for later: the staging DB lives on port 5433."], facts
    finally:
        mem.close()
