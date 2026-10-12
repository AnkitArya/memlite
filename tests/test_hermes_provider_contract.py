"""Hermetic regression tests for the Hermes MemLite provider contract."""

import importlib.util
import sys
import types
import unittest


PLUGIN = "/home/ubuntu/Git Repos/memlite/hermes-plugin/__init__.py"


class FakeMemory:
    def __init__(self, *args, **kwargs):
        self.closed = False
        self.added = []
        self.rows = [{"id": "owned", "memory": "owned fact", "user_id": "u1"}]

    def search(self, query, filters=None, top_k=5, strategy="hybrid"):
        return [{
            "id": "owned",
            "memory": f"{query} result",
            "user_id": (filters or {}).get("user_id"),
        }]

    def add_raw(self, fact, user_id=None, aliases=None, metadata=None):
        self.added.append((fact, user_id, metadata))
        return {"results": [{"id": "new", "memory": fact, "event": "ADD"}]}

    def add(self, messages, user_id=None, metadata=None):
        self.added.append((messages, user_id, metadata))
        return {"results": [{"id": "sync", "memory": "extracted fact", "event": "ADD"}]}

    def get_all(self, filters=None):
        user_id = (filters or {}).get("user_id")
        return {"results": [row for row in self.rows if row["user_id"] == user_id]}

    def delete(self, memory_id):
        return {"results": [{"id": memory_id, "event": "DELETE"}]}

    def history(self, memory_id, limit=50):
        return [{"id": memory_id, "limit": limit}]

    def close(self):
        self.closed = True


def load_provider():
    fake_memlite = types.ModuleType("memlite")
    setattr(fake_memlite, "Memory", FakeMemory)
    sys.modules["memlite"] = fake_memlite
    spec = importlib.util.spec_from_file_location("memlite_provider_contract", PLUGIN)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load provider module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HermesProviderContractTests(unittest.TestCase):
    def setUp(self):
        self.module = load_provider()
        self.provider = self.module.MemLiteProvider({
            "db_path": "contract-test.db",
            "embedding_api_key": "test-only",
            "llm_api_key": "test-only",
            "user_scope": "u1",
            "sync_turn_enabled": True,
        })
        self.provider.initialize("session-1", hermes_home="contract-home")

    def tearDown(self):
        self.provider.shutdown()

    def test_session_boundary_keeps_provider_alive_and_scopes_prefetch(self):
        memory = self.provider.mem
        self.provider.on_session_end([])
        self.assertIs(self.provider.mem, memory)
        self.assertFalse(self.provider._closed)

        self.provider.on_session_switch("session-2")
        self.provider.queue_prefetch("hello", session_id="session-2")
        self.provider._prefetch_thread.join(timeout=2)

        self.assertIn("hello result", self.provider.prefetch("hello", session_id="session-2"))
        self.assertEqual("", self.provider.prefetch("hello", session_id="session-1"))
        status = self.provider.recall_status()
        self.assertEqual("memlite", status.provider_label)
        self.assertEqual(1, status.count)

    def test_mutations_are_scope_checked_and_arguments_are_bounded(self):
        result = self.provider.handle_tool_call("memlite_search", {
            "query": "facts", "top_k": 999, "strategy": "invalid",
        })
        self.assertIn('"ok": true', result)

        denied = self.provider.handle_tool_call("memlite_forget", {"memory_id": "missing"})
        self.assertIn("not owned", denied)
        denied = self.provider.handle_tool_call("memlite_history", {"memory_id": "missing"})
        self.assertIn("not owned", denied)

        allowed = self.provider.handle_tool_call("memlite_forget", {"memory_id": "owned"})
        self.assertIn('"ok": true', allowed)

    def test_context_thread_sync_and_shutdown(self):
        memory = self.provider.mem
        self.provider.sync_turn("user text", "assistant text", session_id="session-1")
        self.provider._sync_thread.join(timeout=2)
        self.assertTrue(memory.added)

        self.provider.shutdown()
        self.assertIsNone(self.provider.mem)
        self.assertTrue(self.provider._closed)
        self.assertTrue(memory.closed)


if __name__ == "__main__":
    unittest.main()
