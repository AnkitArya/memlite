"""`hermes memlite` subcommands for the memlite memory provider.

Convention-based discovery: found next to __init__.py when the provider
is the ACTIVE memory provider (memory.provider: memlite).
"""


def register_cli(p) -> None:
    """Build the ``hermes memlite`` argparse subcommand tree.

    Called with *p* = the already-created ``hermes memlite`` parser
    (not the root subparsers object — the host adds the command itself).
    """
    sub = p.add_subparsers(dest="memlite_cmd", required=False)

    sub.add_parser("status", help="Show provider/store status")

    p_list = sub.add_parser("list", help="List all memories in scope")
    p_list.add_argument("--user", default=None, help="user_id scope (default: provider scope)")
    p_list.add_argument("--limit", type=int, default=50)

    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query")
    p_search.add_argument("--strategy", default="hybrid",
                          choices=["hybrid", "semantic", "keyword"])
    p_search.add_argument("--top-k", type=int, default=5)

    p_stats = sub.add_parser("stats", help="Store statistics")

    p_forget = sub.add_parser("forget", help="Delete one memory by id (or uuid prefix)")
    p_forget.add_argument("memory_id")


def memlite_command(args) -> None:
    """Route memlite subcommands (host handler convention: <name>_command)."""
    raise SystemExit(handle(args))


def handle(args) -> int:
    p = _ensure_initialized()
    cmd = getattr(args, "memlite_cmd", "status")

    if cmd == "status":
        print("provider: memlite")
        print(f"available: {p.is_available()}")
        if p.mem:
            print(f"db: {p.mem.db_path}")
        return 0

    if cmd == "list":
        r = p.mem.get_all(filters={"user_id": args.user or p._user_scope()})
        for x in r["results"]:
            print(f"{x['id']}  {x['memory']}")
        print(f"-- {len(r['results'])} memories --")
        return 0

    if cmd == "search":
        hits = p.mem.search(args.query, strategy=args.strategy, top_k=args.top_k)
        for h in hits:
            print(f"[{h['score']:.3f}] {h['memory']}  (id={h['id'][:8]})")
        return 0

    if cmd == "stats":
        all_rows = p.mem.get_all()["results"]
        print(f"total memories: {len(all_rows)}")
        return 0

    if cmd == "forget":
        ok = p.mem.delete(args.memory_id.strip())
        print("deleted" if ok else "not found")
        return 0 if ok else 1

    print(f"unknown memlite command: {cmd}")
    return 2


def _ensure_initialized():
    p = _plugin_provider()
    if p.mem is None:
        p.initialize(session_id="cli", hermes_home=str(_hermes_home()))
    return p


def _plugin_provider():
    try:
        from . import MemLiteProvider, _load_plugin_config
        return MemLiteProvider(config=_load_plugin_config())
    except ImportError:
        # The host loads cli.py as a standalone module
        # (_hermes_user_memory.memlite__source_...), with no package context,
        # so the relative import above fails. Load the sibling __init__.py
        # by path instead. (This fallback is load-bearing — do not inline.)
        import importlib.util
        from pathlib import Path
        here = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location(
            "memlite_provider", here / "__init__.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.MemLiteProvider(config=mod._load_plugin_config())


def _hermes_home():
    from hermes_constants import get_hermes_home
    return get_hermes_home()


if __name__ == "__main__":
    raise SystemExit("use via `hermes memlite ...`")
