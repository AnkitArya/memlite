"""Tests for memlite's optional in-chat 'saved to memory' indicator.

Mirrors the hindsight `_status_callback` pattern: the host injects a
callable(show_save_indicator) into the provider. Hermetic parts verify state
capture; the live test exercises the actual callback on a background sync.
"""
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-dummy")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
HERMES = os.path.expanduser("~/.hermes/hermes-agent")
if os.path.isdir(HERMES):
    sys.path.insert(0, HERMES)

spec = importlib.util.spec_from_file_location(
    "memlite_plugin_saveind", os.path.join(REPO, "hermes-plugin", "__init__.py"))
plug = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plug)

TMP = "/tmp/memlite_saveind.db"
Path(TMP).unlink(missing_ok=True)


def make(config=None, show=True):
    p = plug.MemLiteProvider(config=config or {})
    p._config.update({"user_scope": "sv", "db_path": TMP, "show_save_indicator": show})
    return p


# ---- hermetic: state capture without network ----

def test_initialize_captures_status_callback():
    p = make()
    calls = []
    p.initialize("s", hermes_home="/tmp", status_callback=calls.append)
    assert p._status_callback is not None
    # invoke it; should append to calls
    p._status_callback("test")
    assert calls == ["test"]
    assert p._show_save_indicator is True


def test_indicator_disabled_by_default():
    p = make(show=False)
    p.initialize("s", hermes_home="/tmp", status_callback=lambda msg: None)
    assert p._show_save_indicator is False


def test_config_schema_exposes_knobs():
    from pathlib import Path as P
    # confirm DEFAULTS source includes the new knobs (used by post_setup + schema)
    import importlib.util as ilu
    sch = os.path.join(REPO, "hermes-plugin", "config_schema.py")
    s = ilu.spec_from_file_location("cs_saveind", sch)
    m = ilu.module_from_spec(s); s.loader.exec_module(m)
    assert m.DEFAULTS["retention"] == "all"
    assert m.DEFAULTS["show_save_indicator"] is False


# ---- live: sync_turn fires the callback when a save happens ----

def test_sync_turn_fires_save_indicator():
    # needs real DeepInfra creds + network; skip if unavailable
    if not (os.environ.get("DEEPINFRA_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        return  # silent skip in hermetic runs
    for line in open(os.path.expanduser("~/.hermes/.env")):
        line = line.strip()
        if line.startswith("DEEPINFRA_API_KEY=") or line.startswith("OPENAI_API_KEY="):
            k, v = line.split("=", 1); os.environ.setdefault(k, v.strip().strip('"').strip("'"))
    p = make()
    msgs = []
    p.initialize("live_save", hermes_home="/tmp", status_callback=msgs.append)
    p.sync_turn("User now prefers hiking on weekends", "Noted, I have that.", session_id="live_save")
    for _ in range(40):
        if msgs:
            break
        time.sleep(0.5)
    assert msgs, "save indicator callback never fired"
    assert any("saved" in m.lower() for m in msgs)
