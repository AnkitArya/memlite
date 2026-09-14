# MemLite — Backlog / Known Items

## TODO: hermes memory setup — pre-select the ACTIVE provider in the picker

**Symptom (reported):** Running `hermes memory setup` when memlite is already the
configured provider does NOT have "memlite" pre-selected/highlighted in the picker —
it lands on "Built-in only" instead.

**Root cause (in Hermes, not memlite):**
`hermes_cli/memory_setup.py`, `cmd_setup()`, ~line 307:

```python
items = [(name, f"— {desc}") for name, desc, _ in providers]
items.append(("Built-in only", "— MEMORY.md / USER.md (default)"))
builtin_idx = len(items) - 1
selected = _curses_select("Memory provider setup", items, default=builtin_idx, ...)
```

The picker `default=builtin_idx` is **hardcoded** to the last item ("Built-in only"),
regardless of what `memory.provider` is currently set to. So even a fully-integrated,
active provider is never pre-selected — the operator must manually arrow down to it.

**Fix (later):** compute the default index from the current active provider
(`memory.provider` in config), e.g.:

```python
idx = next((i for i,(name,_,_) in enumerate(providers) if name == current_provider), builtin_idx)
selected = _curses_select("Memory provider setup", items, default=idx, ...)
```

That's an upstream Hermes change (`hermes_cli/memory_setup.py`), not a memlite one —
memlite already implements `post_setup()` and is discoverable via `discover_memory_providers()`
(this repo's `hermes-plugin/` is correctly wired; `hermes memory status` lists it as active).
