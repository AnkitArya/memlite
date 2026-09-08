#!/usr/bin/env python3
"""Seed a fresh memlite store from Hermes MEMORY.md / USER.md facts.

Converts an existing profile's persistent memory (facts separated by the
`§` delimiter in memories/MEMORY.md and USER.md) into memlite-format facts
via `Memory.add_raw` — the no-LLM path, since these are already durable fact
statements. Deterministic reconcile de-dupes against any existing rows.

Usage:
    python scripts/seed_facts.py \
        --memory-md ~/.hermes/memories/MEMORY.md \
        --user-md    ~/.hermes/memories/USER.md \
        --db         ~/.hermes/memlite.db \
        --user-id    default

    (omit --user-md / --memory-md to skip that source; run twice against the
     same db to verify idempotency / dedupe.)
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SEP = "\n§\n"  # Hermes memory block separator in MEMORY.md / USER.md


def _facts_from(text: str) -> list[str]:
    if not text:
        return []
    blocks = [b.strip() for b in text.split(SEP)]
    return [b for b in blocks if b and b.lower() not in ("", "none")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--memory-md", default=None, help="path to MEMORY.md")
    ap.add_argument("--user-md", default=None, help="path to USER.md")
    ap.add_argument("--db", required=True, help="memlite db path (will be reset/created)")
    ap.add_argument("--user-id", default="default", help="memlite user_id scope")
    ap.add_argument("--reset", action="store_true",
                    help="drop all tables first (clean re-seed)")
    args = ap.parse_args()

    if not (os.environ.get("DEEPINFRA_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("FATAL: set DEEPINFRA_API_KEY / OPENAI_API_KEY in the environment")
        sys.exit(2)

    from memlite import Memory

    m = Memory({
        "embedder": {"config": {
            "model": "BAAI/bge-base-en-v1.5",
            "openai_base_url": "https://api.deepinfra.com/v1/openai",
        }},
    }, db_path=args.db)

    if args.reset:
        m.reset()

    inserted = 0
    for label, path in (("MEMORY", args.memory_md), ("USER", args.user_md)):
        if not path or not Path(path).exists():
            print(f"[{label}] source missing/skipped")
            continue
        facts = _facts_from(Path(path).read_text())
        print(f"[{label}] {len(facts)} fact block(s) from {path}")
        for i, fact in enumerate(facts, 1):
            res = m.add_raw(fact, user_id=args.user_id)
            events = [r.get("event", "?") for r in res.get("results", [])]
            print(f"  {i:2}. [{','.join(events) or 'ADD'}] {fact[:70]}{'...' if len(fact) > 70 else ''}")
            inserted += 1

    total = m.get_all(filters={"user_id": args.user_id})["results"]
    print(f"\nDONE: {len(total)} memories in db for user_id={args.user_id!r} "
          f"(db={args.db})")
    m.close()


if __name__ == "__main__":
    main()
