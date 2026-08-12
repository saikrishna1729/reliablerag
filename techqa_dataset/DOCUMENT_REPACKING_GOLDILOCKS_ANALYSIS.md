# Document Repacking Goldilocks Analysis (TechQA)

Source: `15_document_repacking_goldilocks.ipynb`. Same three-phase search (repacking
strategy → context-width sweep → wide-and-repacked vs. `14`'s narrow winner) as
`delucion_dataset`'s document repacking goldilocks notebook, scored on a **fixed
20-question evaluation set reused across every config**, with judge annotation retries
instead of silently dropping failed samples. Built on TechQA's full real winning stack:
`11`'s chunking (recursive_char, 128t nominal, 10% overlap), `12`'s retriever (hybrid,
alpha=0.5), `13`'s query transformation (HyDE, 3 documents + query, retrieval-only), and
`14`'s reranker (Cohere 4-Pro, candidate pool of 30). Composite score = 0.35·Relevance +
0.15·Utilization + 0.15·Completeness + 0.35·Adherence, identical weighting throughout.

## Winning configuration

**No repacking needed — final_k=2, the reranker's own (forward) order.**

| Metric | Value |
|---|---|
| Context Relevance | 0.594 |
| Utilization | 0.263 |
| Completeness | 0.521 |
| Adherence | 0.700 |
| Composite | **0.570** |
| Eval sample | 20/20 succeeded |
| vs. best wide+repacked config (forward, final_k=4) | +0.026 composite |

Same qualitative conclusion as DelucionQA — narrow context wins outright, repacking doesn't
close the gap — but the margin here (+0.026) is small enough to fall **within** this
series' established TechQA noise floor (~0.05–0.07 composite, per `12` and `13`'s
demonstrations), unlike DelucionQA's clearer +0.066 gap. Treat "narrow wins on TechQA" as
the same-direction result it is on DelucionQA, but a less decisive one.

## The headline finding: reverse ordering, DelucionQA's repacking winner, is TechQA's worst

- **Phase 1's ranking flips relative to DelucionQA.** On DelucionQA, `reverse` (most
  relevant chunk placed immediately before the question) clearly beat `forward` and `sides`.
  On TechQA, **`forward`** (the reranker's own default order, i.e. effectively no repacking
  transformation) wins outright (composite 0.469), `sides` is close behind (0.457), and
  **`reverse` is the worst of the three** (0.410) — the exact opposite ranking on the one
  strategy that mattered most for DelucionQA. This is another instance of the pattern `11`
  already established for chunking method: a specific technique that clearly helps one
  RAGBench dataset can clearly hurt another, and there is no universal repacking default
  that works across both.
- **A plausible mechanism, consistent with what `14` already found about TechQA.** `14`'s
  TechQA analysis showed Adherence rising with wider context (0.65→0.85 across final_k), the
  opposite of DelucionQA — suggesting TechQA's technical answers benefit from having more
  supporting material available, not just the single most relevant fact placed closest to
  the question. If that's right, `reverse`'s whole premise (front-load the *one* most
  relevant chunk right before the question, treating everything else as lower priority)
  works against a dataset where breadth of supporting context matters more than DelucionQA's
  single-fact lookups — while `forward`'s natural descending-relevance order at least keeps
  the reranker's own best-first judgment intact rather than actively deprioritizing it.
- **The context-width relationship is otherwise the same shape as DelucionQA's.** Phase 2
  (forward repacking): composite falls cleanly from 0.539 (final_k=4) → 0.522 (final_k=6) →
  0.413 (final_k=8) → 0.398 (final_k=12) — the same "narrower is better" pattern found for
  chunk size (`11`), retrieval depth (`12`), and rerank depth (`14`) on both datasets. Unlike
  Phase 1's method ranking, this axis generalizes across datasets even though the specific
  numbers and optimal final_k differ.
- **Phase 3's re-measurement this time landed close to its Phase 2 original** (forward @
  final_k=4: 0.539 in Phase 2, 0.545 in Phase 3 — only a 0.006 gap), a useful reminder that
  this series' noise floor is real but not *always* large; the 0.070 gap `13` observed for a
  different config was a notably bad case, not a universal property of every TechQA
  measurement.

## Phase 1 — repacking strategy comparison (final_k=6)

Isolates repacking strategy with context width fixed at 6 — wide enough for the three
strategies to actually differ, unlike `14`'s own final_k=2 winner.

| Strategy | CR | Util | Completeness | Adherence | Composite |
|---|---|---|---|---|---|
| **forward** | 0.332 | 0.239 | 0.593 | 0.65 | **0.469** |
| sides | 0.367 | 0.203 | 0.470 | 0.65 | 0.457 |
| reverse | 0.283 | 0.140 | 0.416 | 0.65 | 0.410 |

**Takeaway:** the reranker's own default order wins; actively reordering toward
"most-relevant-last" (reverse) is the worst option here, unlike on DelucionQA.

## Phase 2 — context-width sweep (forward repacking)

Isolates context width with the Phase 1 winning strategy fixed.

| final_k | CR | Adherence | Composite |
|---|---|---|---|
| **4** | 0.465 | 0.75 | **0.539** |
| 6 | 0.351 | 0.80 | 0.522 |
| 8 | 0.206 | 0.65 | 0.413 |
| 12 | 0.177 | 0.75 | 0.398 |

**Takeaway:** same clean "narrower wins" shape found throughout this whole two-dataset
series, on the one axis that has generalized consistently everywhere.

## Phase 3 — wide-and-repacked vs. `14`'s narrow winner

The question this notebook exists to answer, measured fresh on this notebook's own
pipeline rather than cited from `14`.

| Config | CR | Adherence | Composite |
|---|---|---|---|
| **narrow, final_k=2, no repacking needed** | 0.594 | 0.70 | **0.570** |
| forward, final_k=4 (Phase 2 winner) | 0.456 | 0.75 | 0.545 |

**Takeaway:** narrow still wins, but by a margin (+0.026) small enough to be within this
series' own established noise floor — a real but much less confident result than
DelucionQA's equivalent finding.

## Recommendation

**Skip document repacking for TechQA** and stick with `14`'s final_k=2, no-reordering
configuration — the same directional recommendation as DelucionQA, though held with less
confidence given the narrow margin here. If a future, larger-sample validation shows the
narrow-vs-wide gap collapses further or reverses, it would be worth revisiting given the
Adherence tradeoff `14` already documented for this dataset (wider context does
measurably improve faithfulness on TechQA, even if it doesn't win on composite here).

If repacking is used at all on TechQA (e.g. a deployment that needs a wider context for
other reasons), **use `forward` (the reranker's own order), not `reverse`** — the opposite
of DelucionQA's own repacking recommendation.

## Caveats — do not treat these exact numbers as final

- **Single run, one fixed 20-question sample**, and this notebook's own Phase 3 margin
  (+0.026) sits inside the noise floor `12` and `13` already established for this dataset
  (0.044–0.070 composite spreads on nominally identical configs). Don't treat "narrow beats
  wide+repacked" as a confident conclusion on TechQA the way it was on DelucionQA.
- **Chunking, retriever, query transformation, and reranker held constant** at `11`–`14`'s
  winning TechQA configs. Whether a different upstream pipeline changes which repacking
  strategy or final_k wins is not explored here.
- **`query_llm` runs at temperature 0.7** for the 3-document HyDE retrieval string feeding
  every config's candidate pool — part of the same noise source flagged above.
- **The `sides` implementation is one reasonable interpretation** (alternating head/tail
  placement by rank), same as DelucionQA's `15` — a different implementation could rank
  differently, especially given how close `sides` (0.457) came to `forward` (0.469) here.
- **The OpenRouter rerank endpoint is a third-party dependency** with its own latency, cost,
  and availability characteristics not modeled by the composite score.
- **Composite score weighting (0.35/0.15/0.15/0.35) is a judgment call**, kept identical
  throughout this series for comparability. Given TechQA's documented Adherence-vs-width
  tradeoff (`14`), a deployment weighting faithfulness more heavily might prefer the wider,
  repacked config despite its slightly lower composite here.

This analysis is documentation only — no notebook or library code has been changed to
adopt this configuration yet.
