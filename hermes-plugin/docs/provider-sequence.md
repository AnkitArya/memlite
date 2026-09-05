# MemLite — Hermes Memory Provider (lifecycle)

```mermaid
sequenceDiagram
    autonumber
    actor C as Caller / Agent loop
    participant HV as Hermes MemLiteProvider
    participant L as Extraction LLM
    participant E as Embedder (batched)
    participant S as Store (memlite.db: vec0 + FTS5 + history)

    note over C,HV: agent startup
    C->>HV: initialize(session_id, hermes_home=...)
    HV->>S: open/create memlite.db

    note over C,HV: every turn
    HV-->>C: prefetch() -> <relevant_user_memories> (cache only, zero latency)
    Note over E,S: begin turn sync (daemon thread)
    HV->>L: extract(user + assistant turn)
    L-->>HV: discrete durable facts (filler dropped)
    HV->>E: embed_many(facts)  [one API call]
    HV->>S: plan-then-mutate: kNN reads, BEGIN IMMEDIATE, batch writes, COMMIT
    HV->>S: queue_prefetch: hybrid search for next turn (speculative)

    alt background failure
        HV->>HV: circuit breaker trip (5 fails -> 2 min backoff)
    end

    HV->>HV: shutdown(): join worker, close store
```

### Activation

```bash
hermes config set memory.provider memlite
```

Hooks used: `prefetch` (before each API call), `sync_turn` (after each turn),
`queue_prefetch` (next-turn pre-warm), `on_session_end`. Tools registered:
`memlite_search`, `memlite_add`, `memlite_forget`.
