# MemLite — Flows

> Sequence diagrams for every operation. Render with
> `python docs/render_diagrams.py` (mermaid-cli, optional) or paste a block
> into any mermaid renderer.

## 1. add(infer=False) — Raw ADD

```mermaid
sequenceDiagram
    participant C as Caller
    participant M as Memory
    participant E as Embedder
    participant S as Store (SQLite)
    C->>M: add(messages, infer=False)
    M->>M: _to_texts(messages)
    M->>E: embed_many(texts)  %% batched, one API call
    E-->>M: embeddings[]
    M->>S: insert(mem, emb) x N
    Note over S: one txn: memories + memory_vectors + memories_fts + history(ADD)
    M-->>C: {"results": [{id, memory, event: ADD}]}
```

## 2. add(infer=True) — with LLM extraction + deterministic reconcile

```mermaid
sequenceDiagram
    participant C as Caller
    participant M as Memory
    participant L as LLM
    participant E as Embedder
    participant S as Store
    C->>M: add(messages, infer=True)
    M->>M: _to_texts() + _is_retraction() per text
    Note over M: retraction statements bypass extraction (keep DELETE intent)
    M->>L: extract(normal texts)
    L-->>M: discrete facts[] (small talk dropped)
    Note over M: extraction empty -> fall back to raw chunks (never drop data)
    loop per extracted fact
        M->>E: embed(fact)
        M->>S: semantic_search(emb, top_k=5, scope)
        S-->>M: existing candidates
        M->>M: _decide(fact, emb, existing)
        alt DELETE (retraction + cos>=0.65)
            M->>S: delete(id) + history(DELETE, old_memory snapshot)
            M-->>M: results += DELETE
        else UPDATE (cos>=0.65 + shared bigram) OR (cos>=0.90 dup guard)
            M->>S: update_memory(id, text, emb) + history(UPDATE)
            M-->>M: results += UPDATE
        else ADD (everything else)
            M->>S: insert(fact, emb) + history(ADD)
            M-->>M: results += ADD
        end
    end
    M-->>C: {"results": [...]}
```

## 3. add(infer=True, reconcile_with_llm=True) — LLM reconcile variant

```mermaid
sequenceDiagram
    participant C as Caller
    participant M as Memory
    participant L as LLM
    participant S as Store
    C->>M: add(messages, infer=True, reconcile_with_llm=True)
    M->>L: extraction pass(facts only)
    M->>E: embed(extracted)
    M->>S: semantic_search top_k=8 in scope
    S-->>M: existing memories with ids
    M->>L: reconcile prompt (statement + existing ids)
    L-->>M: {"memory": [event: ADD|UPDATE|DELETE, id?, text?]}
    M->>M: validate: id must exist in retrieved set, else ignored
    opt LLM returned nothing usable
        M->>S: raw ADD fallback (never drop data)
    end
    M->>S: apply ops transactionally
    M-->>C: {"results": [...]}
```

## 4. search — semantic / keyword / hybrid (RRF + recency)

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

## 5. update / delete / delete_all / get_all (admin surface)

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

## 6. reflect(query) — recall + synthesis

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

## 7. Concurrency / locking

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
