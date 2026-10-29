"""sync_turn must capture the user's side of the turn, not the assistant's.

Root cause of the "facts about the user that are really status reports" class
(memlite issue #1 follow-up): sync_turn fed BOTH user and assistant text to the
extractor, and the extraction prompt rewrites "I/my" as "User". Assistant
narration therefore became third-person facts, e.g.

    User opened PR #35 ... 4 commits and 13 files ...
    User's audit run (proc_...) completed with exit code 0, taking 42 minutes

Those are stale within a day. User text is where durable preferences live, so
that is what we extract by default. ``sync_include_assistant`` exists for anyone
who wants the old behaviour back.

Fully offline: the engine is replaced by a recording stub.

Run: python -m pytest tests/test_sync_turn_scope.py -q
"""
import importlib.util
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

USER_TEXT = "I always want the PR description to list the tests I ran."
ASSISTANT_TEXT = ("Done — I opened PR #35 (4 commits, 13 files) and the audit run "
                  "completed with exit code 0, taking 42 minutes.")


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "memlite_plugin_sync", os.path.join(REPO, "hermes-plugin", "__init__.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Recorded:
    def __init__(self):
        self.calls = []
        self.deleted = []

    def add(self, messages, **kwargs):
        self.calls.append({"messages": list(messages), "kwargs": kwargs})
        return {"results": []}

    def delete(self, memory_id):
        self.deleted.append(memory_id)


class _InlineThread:
    """Run the worker synchronously so the test is deterministic."""

    def __init__(self, fn):
        self._fn = fn

    def start(self):
        self._fn()

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


def _provider(plugin, config=None):
    provider = plugin.MemLiteProvider(config=config or {})
    plugin.spawn_context_thread = lambda fn, name=None: _InlineThread(fn)
    stub = _Recorded()
    provider._mem = stub
    provider._sync_turn_enabled = True
    return provider, stub


def test_assistant_text_is_not_extracted_by_default():
    plugin = _load_plugin()
    provider, stub = _provider(plugin)

    provider.sync_turn(USER_TEXT, ASSISTANT_TEXT, session_id="s1")

    assert len(stub.calls) == 1
    assert stub.calls[0]["messages"] == [USER_TEXT]


def test_assistant_text_can_be_opted_back_in():
    plugin = _load_plugin()
    provider, stub = _provider(plugin, {"sync_include_assistant": True})

    provider.sync_turn(USER_TEXT, ASSISTANT_TEXT, session_id="s1")

    assert len(stub.calls) == 1
    assert stub.calls[0]["messages"] == [USER_TEXT, ASSISTANT_TEXT]


def test_empty_user_side_does_not_call_the_extractor():
    plugin = _load_plugin()
    provider, stub = _provider(plugin)

    provider.sync_turn("   ", ASSISTANT_TEXT, session_id="s1")

    assert stub.calls == []


def test_capture_still_works_when_user_side_is_the_only_content():
    plugin = _load_plugin()
    provider, stub = _provider(plugin)

    provider.sync_turn(USER_TEXT, "", session_id="s1")

    assert len(stub.calls) == 1
    assert stub.calls[0]["kwargs"]["user_id"]
    assert stub.calls[0]["kwargs"]["metadata"]["write_origin"] == "sync_turn"
