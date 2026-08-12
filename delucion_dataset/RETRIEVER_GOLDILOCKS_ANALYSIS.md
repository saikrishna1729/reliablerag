# Retriever Goldilocks Analysis (DelucionQA)

Source: `12_retriever_goldilocks.ipynb`. A three-phase search (method → k → method-specific
parameter), scored on a **fixed 20-question evaluation set reused across every config**
(same set-up as `11_chunking_goldilocks.ipynb`), with judge annotation retries instead of
silently dropping failed samples. Every config is built on top of `11`'s own winning
chunking configuration — sentence-level chunking + metadata heading, 64-token target, 10%
overlap (456 chunks, avg 68 tokens/chunk) — so this notebook's only independent variable is
the retriever. Composite score = 0.35·Relevance + 0.15·Utilization + 0.15·Completeness +
0.35·Adherence, identical weighting to `11` for direct comparability.

## Winning configuration

**Hybrid search (BM25 + dense), k=4, alpha=0.3** (30% BM25 / 70% dense, min-max normalized
and linearly combined).

| Metric | Value |
|---|---|
| Context Relevance | 0.453 |
| Utilization | 0.258 |
| Completeness | 0.616 |
| Adherence | 0.950 |
| Composite | **0.622** |
| Avg docs retrieved | 4.0 |
| Eval sample | 20/20 succeeded |

## Why DelucionQA specifically needs this configuration

This dataset's retriever "goldilocks zone" is a direct continuation of what `11` found
about the chunks it's retrieving over: DelucionQA's chunks are already small (68 tokens
average) and each one is built to hold a single self-contained fact from the Jeep Gladiator
owner's manual. That structure explains the shape of every phase below.

- **Fewer, higher-precision retrievals beat broad recall.** Context Relevance falls almost
  monotonically as `k` grows in Phase 2 — 0.354 at k=4 down to 0.172 at k=16 — the same
  "small footprint beats wide net" pattern `11` found for chunk size (0.284 at 64t → 0.108
  at 224t). With single-fact 68-token chunks, the correct chunk for a single-hop factual
  question is almost always in the top handful of results; pulling 12–16 chunks just
  dilutes the relevant-sentence ratio with unrelated manual sections (torque specs next to
  warning-light meanings next to maintenance steps) that Context Relevance directly
  penalizes. k=4 is a clear standout in Phase 2, not a marginal win (composite 0.598 vs.
  0.557 at k=8 vs. 0.501 at k=12).
- **Lexical and dense retrieval catch different failure modes, so combining them beats
  either alone.** In Phase 1, hybrid (composite 0.531) edges out MMR (0.529), which beats
  plain similarity (0.485), which beats BM25 alone (0.468) — but the gap between hybrid/MMR
  and similarity/BM25 matters more than the ordering within each pair. Manual text contains
  a lot of exact, low-frequency tokens (part numbers, button labels, warning-light names)
  that BM25's lexical overlap is well-suited to match verbatim, while dense embeddings are
  better at matching a paraphrased question ("how do I turn off the alarm" vs. a chunk that
  says "press and hold the panic button") to a chunk that never repeats the question's
  wording. Neither alone covers both cases; combining scores does.
- **The combination should lean dense, not lexical.** Phase 3's alpha sweep (weight on the
  BM25 side) peaks at alpha=0.3 (composite 0.622) and alpha=0.1 is nearly as strong (0.593),
  while alpha=0.9 — almost pure BM25 — is the worst config tested in that phase (0.481),
  consistent with BM25 alone being Phase 1's weakest standalone method. DelucionQA questions
  are typically phrased conversationally ("what should I do if...") against manual prose
  that answers procedurally, so the vocabulary overlap BM25 depends on is often thin; a
  small BM25 weight still adds value by rescuing exact-term matches dense retrieval
  under-ranks, but the majority of the signal should come from semantic similarity.
- **DelucionQA questions are single-hop factual lookups**, same as `11` found for
  chunking — the answer lives in one small manual passage, not assembled across several
  retrieved chunks. That's consistent with Adherence staying high (0.95–1.00) across almost
  every k and alpha value tested: once the right chunk is retrieved at all, the generator
  reliably stays grounded in it. The metric that moves is Context Relevance (precision of
  *which* chunks come back), not Adherence (whether the generator hallucinates beyond
  them) — which is exactly why narrowing k and tuning alpha toward dense had far more
  leverage than any change would have had on Adherence alone.

In short: this dataset rewards a retriever that mirrors its chunking — pull a small,
high-precision set of chunks (k=4) using a retrieval signal that's mostly semantic but
lexically assisted (hybrid, alpha≈0.3) rather than a wide net (k=12–16) or a single-signal
retriever (pure BM25 or pure dense/MMR).

## Phase 1 — retrieval method comparison (all at k=8)

Isolates retrieval *method* from depth. MMR and hybrid use their own default tuning value
(`lambda_mult=0.5`, `alpha=0.5`) here — each gets its own knob swept in Phase 3.

| Method | Avg retrieved | CR | Util | Completeness | Adherence | Composite |
|---|---|---|---|---|---|---|
| **hybrid (alpha=0.5)** | 8.0 | 0.235 | 0.129 | 0.649 | 0.950 | **0.531** |
| mmr (lambda=0.5) | 8.0 | 0.200 | 0.121 | 0.723 | 0.950 | 0.529 |
| similarity | 8.0 | 0.288 | 0.112 | 0.466 | 0.850 | 0.485 |
| bm25 | 8.0 | 0.164 | 0.094 | 0.544 | 0.900 | 0.468 |

**Takeaway:** hybrid and MMR (both of which blend/diversify beyond raw top-k similarity)
edge out plain similarity and BM25 alone, but the margin at k=8 is narrow (0.531 vs. 0.529)
— it's k and alpha, swept next, that produce the larger gains.

## Phase 2 — k sweep (hybrid, alpha=0.5)

Isolates retrieval *depth* from method.

| k | Avg retrieved | CR | Adherence | Composite |
|---|---|---|---|---|
| **4** | 4.0 | 0.354 | 1.000 | **0.598** |
| 6 | 6.0 | 0.305 | 0.950 | 0.565 |
| 8 | 8.0 | 0.233 | 1.000 | 0.557 |
| 16 | 16.0 | 0.172 | 1.000 | 0.519 |
| 10 | 10.0 | 0.202 | 0.950 | 0.515 |
| 12 | 12.0 | 0.189 | 0.950 | 0.501 |

**Takeaway:** Context Relevance falls off almost monotonically as k grows (0.354 at k=4 →
0.172 at k=16), mirroring `11`'s chunk-size finding exactly. k=4 is a clear standout, not a
marginal win over k=6.

## Phase 3 — alpha sweep (hybrid, k=4)

Isolates the hybrid retriever's own BM25/dense weighting from k.

| Alpha (BM25 weight) | CR | Adherence | Composite |
|---|---|---|---|
| **0.3** | 0.453 | 0.950 | **0.622** |
| 0.5 | 0.373 | 1.000 | 0.617 |
| 0.1 | 0.442 | 0.950 | 0.593 |
| 0.7 | 0.352 | 0.947 | 0.570 |
| 0.9 | 0.260 | 0.800 | 0.481 |

**Takeaway:** composite peaks in the dense-leaning range (alpha 0.1–0.5, composite
0.593–0.622) and drops sharply once BM25 dominates (alpha=0.9, composite 0.481, the worst
config in this phase). Unlike `11`'s overlap sweep (which was comparatively flat), this
phase has real spread — alpha matters, and it should stay low.

## Recommendation

Use **hybrid retrieval (BM25 + dense, min-max normalized, linearly combined) with k=4 and
alpha=0.3** as the default retriever configuration for DelucionQA, built on top of `11`'s
winning chunking config (sentence-level + metadata heading, 64 tokens, 10% overlap).

## Caveats — do not treat these exact numbers as final

- **Single run, one fixed 20-question sample.** Fixing the sample and retrying failed judge
  annotations removes some noise, but this is one dataset draw. Phase 3's alpha=0.1 vs.
  alpha=0.3 gap (0.593 vs. 0.622) is narrow enough that a different fixed sample could
  plausibly swap their order.
- **Chunking held constant** at `11`'s winning config. Chunk size and retriever choice
  interact — e.g. BM25 tends to reward larger, keyword-denser chunks differently than dense
  similarity does — not swept jointly with retriever config in this analysis.
- **Embedding model held constant** (`openai/text-embedding-3-small`). Not explored as a
  variable here.
- **Query transformation (rewriting, decomposition, HyDE) is out of scope.** `05` touched on
  these but never scored them on a fixed sample; a natural follow-up notebook would apply
  them on top of this winning retriever config.
- **Composite weighting (0.35/0.15/0.15/0.35) is a judgment call**, kept identical to `11`
  so the two analyses stay comparable, documented here so it can be argued with. Re-ranking
  by Relevance alone would still favor alpha=0.3 (its own strongest metric, 0.453); re-
  ranking by Adherence alone would favor alpha=0.5 (perfect 1.000) over alpha=0.3 (0.950).

This analysis is documentation only — no notebook or library code has been changed to
adopt this configuration yet.
