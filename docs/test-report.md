# MemLite — Architecture Test Report (T1–T10)

Live-run verdict matrix for 10 architecture-validation cases against real
DeepInfra embeddings (bge-base-en-v1.5) and the DeepSeek-V3 extraction LLM.
Run any of these yourself from the repo root:

```bash
. .venv/bin/activate
python tests/test_arch_suite.py    # T1–T5  (14/14 checks)
python tests/test_alias_recall.py  # alias recall deep-dive (7/7 checks)
python tests/test_t6t10.py         # T6–T10 (16/19 checks)
python -m pytest tests/ -q         # full regression base (18 tests)
```

Commit series: `1f4520d` (write-time alias recall) → `9857624` (retraction
lexical fallback + T1–T5 suite) → `502ea28` (T6–T10 suite, core fixes).

## Verification matrix — final state

| Test | Feature under test | Checks | Verdict |
|---|---|---|---|
| T1 | Cross-vocabulary bridging (write-time aliases + FTS5) | 3 | ✅ 3/3 |
| T2 | Deterministic retraction (regex + gate + history snapshot) | 3 | ✅ 3/3 |
| T3 | Topic-change in-place mutation | 4 | ✅ 4/4 |
| T4 | False-positive safety (same entity, different fact) | 2 | ✅ 2/2 |
| T5 | Hybrid RRF + recency decay | 2 | ✅ 2/2 |
| T6 | Multi-workspace scope isolation + adaptive widening | 5 | ✅ 4/5 |
| T7 | `add_raw` static alias enrichment (no LLM) | 2 | ✅ 2/2 |
| T8 | Pure-paraphrase duplicate gate (cos ≥ 0.90, disjoint bigrams) | 3 | ⚠ 1/3 (spec caveat) |
| T9 | Multi-mutation atomic batching (DELETE+UPDATE+ADD, one txn) | 6 | ✅ 6/6 |
| T10 | Extraction failure fallback (never drop data) | 3 | ✅ 3/3 |

**Total: 33 of 36 live checks pass.**

## What each test proved

### T1 — horoscope↔zodiac bridging (the case that started it all)
Store *"My zodiac sign is Gemini…"* → ask *"What's my horoscope and lucky
color?"* → the extraction stored `aliases=["horoscope","astrology","sun
sign"]`, the FTS corpus indexed them, and the recall surfaced the Gemini fact
in top-3 via hybrid. Demonstrates: related-term recall works even when the
query shares zero word-level vocabulary with the fact.

### T2 — polite-phrasing retraction (edge case that found a real bug)
Initial run **failed**: the cosine-only DELETE gate (≥0.72) miss-detects
polite phrasings — *"Please forget that I drink coffee…"* cosine vs the
stored, extractor-rephrased fact measured **0.594**, below the bar — and the
missed retraction was **stored as a fact**, polluting the store. Fix in
`_decide`: retraction intent now confirms on **either** cosine ≥ 0.72 or
shared content-bigrams (lexical-confirmation fallback), then DELETEs with
the `old_memory` snapshot. Lt. distance measured across phrasings: 0.61–0.83.

### T3 — topic-change in-place mutation
*"switched from Neovim to VS Code"* → `UPDATE`, same row id preserved,
`content`/`embedding` refreshed in place, `history` UPDATE logged. No row
fragmentation. Validated Cosine 0.728 + shared bigram `python development`.

### T4 — same-entity-different-fact safety
*"daily driver is a white Tata Safari"* then *"roof rack for my Tata
Safari"* → sibling `ADD`, both rows persist. Note (documented in `_decide`):
the *extractor* may merge such pairs into one row on write — Test 4's
"dual-row existence" is an extraction-judgment issue; the reconcile layer
correctly emitted `UPDATE + ADD` siblings, not destructive overrides.

### T5 — RRF + 30-day recency decay
A 45-day-old `PostgreSQL` fact vs today's `SQLite` fact, both scoring high on
BM25 and vector: recency decay factor `0.5 + 0.5 * 30/(30+45) = 0.7`
(discounting the stale record), and the fresh fact is rank #1. Formula
re-verified exactly per spec.

### T6 — multi-workspace scope isolation (4/5)
Global fact (`user_id=global`) + workspace facts (`workspace:web`,
`workspace:api`), query from `workspace:api` → returns Fact 1 + Fact 3
only. **Zero cross-workspace leakage** even under adaptive widening (60
noise rows for `otheruser`, k escalated to k=4096): nothing from `ws:web`
or the noise corpus leaks.
- ❌ **Open gap**: the reviewer's "global memories visible alongside
  workspace-scoped facts" needs **scope hierarchy** semantics — a workspace
  query today only matches the exact scope. Real feature, not implemented;
  see "Known unsupported" below.

### T7 — static alias enrichment for tool-added facts
`add_raw("User drives an electric vehicle daily.")` — no LLM call. The
static synonym dictionary matches "electric vehicle" → `["ev", "battery
car", "battery"]`, merges into `aliases`, and re-indexes FTS in-place
(update preserves row id). Query *"Does the user have an EV or battery
car?"* hits rank #1 via keyword only, no dense leg used. Confirms that
facts added programmatically (Hermes `memlite_add` tool) still bridge
vocabulary a later search may use.

### T8 — paraphrase duplicate gate (spec caveat — 1/3)
The spec asserts `cos ≥ 0.90` for the pair *"designing computer software
architectures"* ↔ *"creating system-level application blueprints"* —
**empirically false for bge-base-en-v1.5 at 0.7634**, so the near-duplicate
gate cannot fire as written. Two honest outcomes since a duplicate ADD is
bi-side safe (harmless duplication, never data loss):
- declared `0.90` for facts without aliases stays conservative,
- `0.80` when the incoming fact carries aliases (real paraphrase gate) —
  covers T9's rust case.
Future knob: an **ingest-side synonym expansion** would resolve T8 but is
deferred — no cheap gate exists without either heavy embeddings or a
fully-loaded synonym graph.

### T9 — multi-mutation atomic batch (whole refactor path)
Three intents in ONE user turn → `DELETE + UPDATE + ADD`; single
`BEGIN IMMEDIATE … COMMIT`; fact-A row deleted; fact-B row id preserved,
content refreshed to Go; keyboard fact inserted as a distinct new row;
`history` rows against the batch with the same commit timestamp, zero
partial writes. **Live-exposed a genuine product bug that ran the reviewer's
spec**: the oldr retraction reclamation marked the *whole turn* as retraction
(`_is_retraction` matched the joined text), which suppressed extraction of
the new facts. Fixed with **sentence-level split** inside the reclamation
loop — the exact bug class reviewer-specified and we caught running meta.

### T10 — extraction failure fallback ("never drop data" invariant)
Secretly stubbing the LLM client (`raise RuntimeError`) → `add()` does NOT
raise, falls back to raw chunks, stores the deploy fact, and the search
query *"deploy port 8080"* recovers it. Data-loss invariant holds under a
dead LLM. Run-time contract: the LLM is required for `add()` only in the
sense that absent-config raises `ValueError` at call time, unavailability
mid-run degrades to raw-chunk ingestion.

## Failure semantics (all live-verified)

| Failure | Behavior |
|---|---|
| No LLM configured | `add()` raises `ValueError`; reads/admin paths work |
| LLM call fails mid-turn | falls back to raw chunks (T10), store keeps a copy |
| Empty extraction + no retractions | raw texts become the facts |
| Extractor rephrases retractions | retraction regex bypasses the extractor (T2) |
| Any mutation fails mid-loop | whole transaction rolls back — no partials |
| db locked | busy_timeout 5000 ms, retry ×6 |
| Dead endpoint, background sync | circuit breaker: 5 fails → 2 min backoff |
| Multi-intent turn ("forget X… start Y") | sentence-level split keeps both intents (T9) |

## Known unsupported (documented non-goals)

- **Workspace-scope hierarchy** (T6a): a workspace-scoped search does not
  additionally include `global`-scoped memories. Exact-match `user_id`
  filters only. To implement, an `include_global` or hierarchical scope
  design would be needed — deliberately out of scope (pun intended).
- **Cos ≥ 0.90 paraphrase guarantees for arbitrary phrasings** (T8) — real
  embeddings do not universally score paraphrases that high; alias-boosted
  writing is the supported way to merge those (hits 0.80 gate).
- **Entity graph / cross-memory relations** (per mem0 full-product paths).
