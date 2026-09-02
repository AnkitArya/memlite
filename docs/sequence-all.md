
# MemLite — Single Unified Sequence Diagram (all cases)

All paths in one diagram: raw add, extraction, deterministic & LLM reconcile,
the three search strategies, reflect, admin surface, and multi-process locking.

```mermaid
sequenceDiagram
    autonumber
    actor C as Caller / App
    participant M as Memory (core.py)
    participant L as LLM (extraction only)
    participant E as Embedder (OpenAI-compatible)
    participant S as Store (SQLite: memories + vec0 + FTS5 + history)
    participant P2 as Other Process

    %% ============ WRITE: add() ============
    rect rgb(235,244,255)
    C->>M: add(messages)
    M->>M: _to_texts(messages) -> texts[]
    alt no LLM configured
        M-->>C: ValueError (LLM required for extraction pass)
    else extraction + deterministic reconcile (the only path)
        M->>M: split: retractions (_is_retraction) vs normal
        M->>L: extract(normal texts)   [extraction only — reconcile stays deterministic]
        L-->>M: discrete durable facts (filler dropped)
        Note over M: empty extraction and no retractions -> raw texts become the facts
        Note over M: retraction statements kept verbatim (DELETE intent preserved)
        Note over M: ONE txn for all mutations (begin -> ... -> commit, single fsync)
        M->>E: embed_many(facts)   [one batched API call]
        loop per fact / statement
            M->>S: semantic_search(emb, top_k, scope)
            S-->>M: existing candidates
            M->>M: _decide(text, emb, existing)
            alt DELETE: retraction intent AND cos>=0.82 match
                M->>S: delete(id, in_txn) + history[DELETE, old_memory snapshot]
                M-->>C: event DELETE
            else UPDATE: cos>=0.65 AND shared content-bigram, OR cos>=0.90 duplicate guard
                M->>S: update_memory(id, text, emb, in_txn) + history[UPDATE]
                M-->>C: event UPDATE (same id)
            else ADD: everything else (conservative)
                M->>S: insert(text, emb, in_txn) + history[ADD]
                M-->>C: event ADD
            end
        end
        M->>S: commit
        M-->>C: {results: [...]}
    end
    end

    %% ============ READ: search() ============
    rect rgb(240,255,240)
    C->>M: search(query, strategy, filters)
    M->>E: embed(query)
    alt strategy="semantic" (default)
        M->>S: vec0 cosine kNN (MATCH...LIMIT + adaptive widening) + scope join/filter
        S-->>M: raw rows
        Note over M: shape scores/fields
        M-->>C: scored hits
    else strategy="keyword"
        M->>S: FTS5 MATCH (BM25 rank)
        S-->>M: raw rows
        M-->>C: shaped hits (score = 1/(1+abs(rank)), rank from fts)
    else strategy="hybrid"
        M->>S: semantic_search(top_k*2)
        M->>S: keyword_search(top_k*2)
        S-->>M: both raw lists
        M->>M: RRF fuse: score = Σ 1/(60+rank) across both lists
        M->>M: recency: score *= 0.5 + 0.5*30/(30+age_days)  [newer wins ties]
        M-->>C: top_k fused, recency-weighted
    end
    end

    %% ============ SYNTHESIS: reflect() ============
    rect rgb(255,250,235)
    C->>M: reflect(query, filters)
    M->>S: hybrid search top_k
    S-->>M: memories
    alt no hits OR no LLM configured
        M-->>C: {answer: null, memories, synthesized: false}
    else LLM available
        M->>L: grounded synthesis (evidence only)
        L-->>M: answer
        M-->>C: {answer, memories, synthesized: true}
    end
    end

    %% ============ ADMIN ============
    rect rgb(246,240,255)
    opt update(text, memory_id)
        C->>M: update()
        M->>E: embed(text)
        M->>S: update row + vector + FTS + history[UPDATE]
        M-->>C: {results: [UPDATE]}
    end
    opt delete(memory_id)
        C->>M: delete()
        M->>S: delete from memories + vectors + FTS; history[DELETE, old_memory]
        M-->>C: {results: [DELETE]}
    end
    opt delete_all(scope)
        C->>M: delete_all()
        M->>S: list_all(scope) then delete each
    end
    opt get_all(filters)
        C->>M: get_all()
        M->>S: list_all (scope + memory_type)
        M-->>C: {results: [...]}
    end
    end

    %% ============ CONCURRENCY ============
    rect rgb(255,240,240)
    P2->>S: connect (busy_timeout=5000 + retry x6)
    alt P2 writes while S holds write lock
        P2-->>S: SQLITE_BUSY -> busy_timeout wait -> proceeds after COMMIT
    else P2 reads while S writes
        Note over P2: WAL: readers never blocked
    end
    end
```
