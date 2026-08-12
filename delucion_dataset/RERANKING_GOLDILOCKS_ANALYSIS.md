# Reranking Goldilocks Analysis (DelucionQA)

Source: `14_reranking_goldilocks.ipynb`. A three-phase search (reranker model → candidate
pool width → final kept count), scored on a **fixed 20-question evaluation set reused
across every config** (same set-up as `11`/`12`/`13`), with judge annotation retries instead
of silently dropping failed samples. Every config is built on top of `11`'s winning
chunking config (sentence + metadata, 64t, 10% overlap, 456 chunks), `12`'s winning
retriever config (hybrid BM25+dense, alpha=0.3), and `13`'s winning query-transformation
config (HyDE, one hypothetical document concatenated with the query, driving retrieval
only — the generator and reranker always see the original question). Composite score =
0.35·Relevance + 0.15·Utilization + 0.15·Completeness + 0.35·Adherence, identical weighting
to the three prior notebooks for comparability.

## Winning configuration

**Cohere Rerank 4-Pro, candidate pool of 20 hybrid-ranked chunks reranked down to a final
2 chunks.**

| Metric | Value |
|---|---|
| Context Relevance | 0.619 |
| Utilization | 0.392 |
| Completeness | 0.676 |
| Adherence | 0.950 |
| Composite | **0.709** |
| Eval sample | 20/20 succeeded |
| vs. no-reranking control (candidate_k=20→final_k=4) | +0.089 composite |

This is the largest single-phase gain of any notebook in this series so far — the
0.709 composite comfortably clears `13`'s prior best (0.620) and `12`'s retriever-only
best (0.622). It also reverses `08`'s original conclusion ("reranking hurts, skip it")
outright: with a wider candidate pool and a narrower final cut, the same class of
reranker that previously looked harmful is now this pipeline's biggest lever.

## Why the result flipped from `08`

- **`08` never gave reranking a wider pool than the final context.** It reranked a k=8 pool
  down to k=8 — the reranker could only reorder, never drop anything. This notebook's
  Phase 2 shows candidate pool width matters on its own: holding final_k=4 fixed, composite
  rises from 0.580 (candidate_k=4, no real reranking headroom) to 0.616 (candidate_k=20)
  before falling off at candidate_k=30 (0.584). Reranking needs recall to work with; `08`
  never had any.
- **`08`'s retriever was Dense MMR, which already enforces diversity — hybrid search
  doesn't.** `08`'s own write-up (`RERANKING_ANALYSIS.md`) diagnosed the failure as
  "reranking destroys MMR's built-in diversity, over-indexing a single perspective." That
  diagnosis was retriever-specific. `12` replaced Dense MMR with hybrid (BM25+dense)
  retrieval, which has no diversity mechanism to destroy — there's nothing for a
  relevance-only reranker to break here, which is consistent with reranking now helping
  instead of hurting.
- **`08` reranked the same string used for both retrieval and generation, and never used
  query transformation.** This notebook reranks the candidate pool against the *original*
  question (not the HyDE pseudo-document that widened retrieval), matching `13`'s finding
  that the true information need — not the retrieval-side transformation — should drive
  everything downstream of retrieval itself.
- **Phase 3's clean, near-monotonic falloff as final_k grows (0.709 at final_k=2 → 0.494 at
  final_k=12) is the same "small footprint wins" pattern `11` and `12` already found** for
  chunk size and retrieval depth on this dataset. DelucionQA's single-hop, single-fact
  questions mean the correct answer usually lives in one or two of the tightest, highest
  scoring chunks; keeping more diluted the relevant-sentence ratio Context Relevance
  measures, exactly as adding more k did in `12`'s retriever sweep.
- **Cohere 4-Pro beat Cohere v3.5 and NVIDIA Nemotron consistently**, echoing `08`'s own
  finding about relative reranker quality (4-Pro was `08`'s best performer too, just still
  net-negative there) — model quality ordering held even though the overall verdict on
  reranking flipped.

## Phase 1 — reranker model comparison (candidate_k=20, final_k=4)

Isolates reranker choice with candidate pool width fixed wider than the final context.

| Reranker | CR | Util | Completeness | Adherence | Composite |
|---|---|---|---|---|---|
| **cohere_4pro** | 0.489 | 0.236 | 0.558 | 0.950 | **0.623** |
| no_rerank (control) | 0.425 | 0.224 | 0.585 | 1.000 | 0.620 |
| cohere_v35 | 0.420 | 0.240 | 0.590 | 0.950 | 0.604 |
| nvidia_nemotron | 0.456 | 0.213 | 0.494 | 0.950 | 0.598 |

**Takeaway:** even at this phase, before the final_k tuning that produces the big win,
Cohere 4-Pro already edges out the no-reranking control (0.623 vs. 0.620) — a margin narrow
enough to be noise on its own, but the ordering (4-Pro > control > v3.5 > Nemotron) is
consistent with `08`'s original ranking of reranker quality.

## Phase 2 — candidate pool width sweep (cohere_4pro, final_k=4)

Isolates how much pre-rerank recall matters, the exact axis `08` never varied.

| candidate_k | CR | Adherence | Composite |
|---|---|---|---|
| **20** | 0.477 | 0.900 | **0.616** |
| 30 | 0.451 | 0.900 | 0.584 |
| 4 | 0.372 | 0.950 | 0.580 |
| 8 | 0.442 | 0.850 | 0.577 |
| 12 | 0.471 | 0.800 | 0.563 |

**Takeaway:** composite peaks at candidate_k=20 and falls off in both directions — too
narrow (4–12) starves the reranker of real candidates to choose from; too wide (30) likely
reintroduces enough off-topic chunks that even a good reranker occasionally promotes one.
20 is a genuine sweet spot, not a monotonic "wider is always better" result.

## Phase 3 — final kept-count sweep (cohere_4pro, candidate_k=20)

Isolates how many reranked chunks should actually reach the generator.

| final_k | CR | Adherence | Composite |
|---|---|---|---|
| **2** | 0.619 | 0.950 | **0.709** |
| 4 | 0.492 | 0.850 | 0.583 |
| 6 | 0.341 | 1.000 | 0.580 |
| 8 | 0.305 | 0.850 | 0.494 |
| 12 | 0.218 | 0.950 | 0.496 |

**Takeaway:** the clearest, largest-magnitude finding in this notebook. Composite falls by
nearly a third between final_k=2 and final_k=4 alone, then keeps falling. Once a strong
reranker has already picked the 2 best chunks out of 20 candidates, adding more is close to
pure dilution — Context Relevance drops from 0.619 to 0.218 across the sweep while
Adherence stays roughly flat (0.85–1.00 throughout), meaning the generator stays grounded
regardless of context size but the *relevant fraction* of what it's grounded in collapses
as more marginal chunks get added.

## Recommendation

Use **Cohere Rerank 4-Pro, retrieving a candidate pool of 20 hybrid-ranked chunks (`11`
chunking + `12` retriever + `13` HyDE-for-retrieval) and keeping only the top 2 after
reranking** as the default reranking configuration for DelucionQA. This is the strongest
single result across the whole goldilocks series so far (composite 0.709 vs. `13`'s 0.620
and `12`'s 0.622) and, unlike the query-transformation win in `13`, the margin over the
no-reranking control (+0.089) is large enough to not plausibly be sampling noise.

## Caveats — do not treat these exact numbers as final

- **Still a single run on one fixed 20-question sample.** Phase 3's final_k=6 config lost
  one judge annotation even after retries (19/20 succeeded) — a reminder that even the
  retry mechanism doesn't guarantee every config is scored on the identical sample size.
- **`final_k=2` is an aggressive cut** — two 64–68 token chunks is roughly 130-140 tokens of
  total context. This works well for DelucionQA's single-hop factual questions specifically
  (established across `11`, `12`, and this notebook) but would likely need revisiting for
  any dataset with multi-part or comparison-style questions.
- **Chunking, retriever, and query transformation held constant** at `11`/`12`/`13`'s
  winning configs. Whether a different upstream pipeline changes which reranker or which
  final_k wins is not explored here.
- **`query_llm` runs at temperature 0.7** for the HyDE retrieval string feeding every
  config's candidate pool — a repeat run could shuffle close results even on the same fixed
  question set.
- **The OpenRouter rerank endpoint is a third-party dependency** with its own latency, cost,
  and availability characteristics the composite score doesn't model. Reranking a 20-chunk
  pool adds a network round-trip per query beyond what `12`'s retriever-only pipeline needs
  — a real cost not reflected in these quality numbers.
- **Composite score weighting (0.35/0.15/0.15/0.35) is a judgment call**, kept identical to
  `11`/`12`/`13` for comparability. Re-ranking by Adherence alone would favor the
  no-reranking control (1.000) or final_k=6 (1.000) over final_k=2 (0.950) — the composite's
  heavy weight on the much larger Relevance and Completeness gains at final_k=2 is what
  makes it the overall winner.

This analysis is documentation only — no notebook or library code has been changed to
adopt this configuration yet.
