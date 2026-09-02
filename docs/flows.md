# MemLite — Flows

> Sequence diagrams for every operation. Render with
> `python docs/render_diagrams.py` (mermaid-cli, optional) or paste a block
> into any mermaid renderer.

## 1. add() — LLM extraction + deterministic reconcile (the only path)

```mermaid
sequenceDiagram
    participant C as Caller
    participant M as Memory
    participant L as LLM
    participant E as Embedder
    participant S as Store
    C->>M: add(messages)
    M->>M: _to_texts(messages)
    opt no LLM configured
        M-->>C: ValueError (LLM is required for extraction)
    end
    M->>M: _is_retraction() per text
    Note over M: retraction statements bypass extraction (keep DELETE intent)
    M->>L: extract(normal texts)
    L-->>M: discrete facts[] (small talk dropped)
    Note over M: extraction empty and no retractions -> raw texts become the facts
    Note over M: ONE txn for all mutations below (begin -> ... -> commit)
    M->>E: embed_many(facts)  %% batched, one API call
    loop per extracted fact
        M->>S: semantic_search(emb, top_k=5, scope)
        S-->>M: existing candidates
        M->>M: _decide(fact, emb, existing)
        alt DELETE (retraction + cos>=0.82)
            M->>S: delete(id, in_txn) + history(DELETE, old_memory snapshot)
            M-->>M: results += DELETE
        else UPDATE (cos>=0.65 + shared bigram) OR (cos>=0.90 dup guard)
            M->>S: update_memory(id, text, emb, in_txn) + history(UPDATE)
            M-->>M: results += UPDATE
        else ADD (everything else)
            M->>S: insert(fact, emb, in_txn) + history(ADD)
            M-->>M: results += ADD
        end
    end
    M->>S: commit (single fsync)
    M-->>C: {"results": [...]}
```

> The former raw-add (no LLM) and LLM-reconcile variants were removed:
> exactly one add() path exists — LLM extraction + deterministic reconcile.

## 2. search — semantic / keyword / hybrid (RRF + recency)

```mermaid
sequenceDiagram
    participant C as Caller
    participant M as Memory
    participant E as Embedder
    participant S as Store
    C->>M: search(query, strategy, filters)
    M->>E: embed(query)
    alt strategy = semantic
        M->>S: semantic_search (vec0 cosine kNN + scope filter)
        S-->>C: ranked hits (score = cosine)
    else strategy = keyword
        M->>S: keyword_search (FTS5 BM25)
        S-->>C: ranked hits
    else strategy = hybrid
        M->>S: semantic_search(top_k*2)
        M->>S: keyword_search(top_k*2)
        Note over M: RRF fusion K=60 (rank-based, scale-free)
        Note over M: recency decay: score * (0.5 + 0.5*30/(30+age_days))
        M-->>C: top_k fused, recency-weighted hits
    end
```

## 3. update / delete / delete_all / get_all (admin surface)

```mermaid
sequenceDiagram
    participant C as Caller
    participant M as Memory
    participant E as Embedder
    participant S as Store
    C->>M: update(text, memory_id)
    M->>E: embed(text)
    M->>S: update_memory(id, text, emb) -> vectors + FTS + history(UPDATE)
    M-->>C: {"results": [UPDATE]}
    C->>M: delete(memory_id)
    M->>S: delete(id) -> all indexes + history(DELETE, old_memory=<text>)
    M-->>C: {"results": [DELETE]}
    C->>M: delete_all(user_id=...)
    M->>S: list_all(scope) then delete each
    C->>M: get_all(filters)
    M->>S: list_all -> shaped rows
```

## 4. reflect(query) — recall + synthesis

```mermaid
sequenceDiagram
    participant C as Caller
    participant M as Memory
    participant L as LLM
    participant S as Store
    C->>M: reflect(query, filters)
    M->>S: hybrid search (RRF)
    S-->>M: top_k memories
    alt no hits OR no LLM configured
        M-->>C: {answer: null, memories, synthesized: false}
    else LLM available
        M->>L: grounded synthesis (evidence only)
        L-->>M: answer text
        M-->>C: {answer, memories, synthesized: true}
    end
```

## 5. Concurrency / locking

```mermaid
sequenceDiagram
    participant P1 as Process A (writer)
    participant P2 as Process B (reader/writer)
    participant DB as SQLite (WAL)
    P1->>DB: open ctx: busy_timeout=5000 (+ connect retry x6)
    P2->>DB: open ctx: busy_timeout=5000 (+ connect retry x6)
    P1->>DB: BEGIN IMMEDIATE (write txn)
    Note over DB: WAL: readers keep reading while writer active
    P2->>DB: read queries -> immediate
    P2->>DB: write -> SQLITE_BUSY -> busy_timeout wait -> retry
    P1->>DB: COMMIT
    P2->>DB: write proceeds
```
