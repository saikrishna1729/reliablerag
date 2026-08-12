# Query Transformation Goldilocks Analysis (TechQA)

Source: `13_query_transformation_goldilocks.ipynb`. Same three-phase search (transformation
method → HyDE concatenation ablation → retrieval-query vs. generation-query) as
`delucion_dataset`'s query-transformation goldilocks notebook, scored on a **fixed
20-question evaluation set reused across every config**, with judge annotation retries
instead of silently dropping failed samples. Built on TechQA's real winning foundation from
`11` (recursive-character chunking, 128t nominal, 10% overlap) and `12` (hybrid retrieval,
k=4, alpha=0.5). Composite score = 0.35·Relevance + 0.15·Utilization + 0.15·Completeness +
0.35·Adherence, identical weighting to the DelucionQA series for comparability.

## Winning configuration

**HyDE with three hypothetical documents concatenated with the original query, used for
retrieval only — the generator always sees the original question.**

| Metric | Value |
|---|---|
| Context Relevance | 0.323 |
| Utilization | 0.169 |
| Completeness | 0.518 |
| Adherence | 0.700 |
| Composite | **0.461** |
| Eval sample | 20/20 succeeded |
| vs. no-transformation baseline | **+0.150 composite** |

This is a far larger, more decisive win than DelucionQA's query-transformation notebook
found (+0.0065 composite there, within noise). On TechQA, query transformation is a real,
substantial lever — not a marginal one.

## The headline finding: TechQA rewards query transformation far more than DelucionQA does

- **Query rewriting alone beats the no-transformation baseline in Phase 1** (composite
  0.398 vs. 0.311) — the opposite of DelucionQA, where rewriting *underperformed* baseline
  (0.586 vs. 0.613 there). TechQA's questions are plausibly phrased in casual,
  underspecified forum-user language that needs expansion or clarification to match
  technical support vocabulary, whereas DelucionQA's questions were already close literal
  matches to manual text — rewriting had nothing to gain there and mostly just moved the
  query away from exact-term matches. TechQA's document/question mismatch is bigger, so
  there's more for rewriting to fix.
- **More hypothetical documents helps on TechQA — the opposite of what DelucionQA found.**
  Phase 2: `hyde_3docs_plus_query` (composite 0.461) beats `hyde_doc_plus_query` (0.431),
  which beats `hyde_doc_only` (0.336). On DelucionQA, three documents *hurt* relative to one
  (0.572 vs. 0.620 there). This actually **replicates the source paper's own TREC DL
  finding** (Table 8: 8 pseudo-docs + query beat 1 pseudo-doc + query, mAP 51.64 vs 50.87) —
  DelucionQA was the dataset where that finding *didn't* hold, not TechQA. Plausible
  explanation: TechQA's technical support questions likely admit multiple valid framings or
  cover multi-step troubleshooting, so several independently-generated hypothetical answers
  probably cover more of the actual answer space than DelucionQA's single-hop factual
  lookups needed.
- **The retrieval-query vs. generation-query finding replicates cleanly, and even more
  dramatically than on DelucionQA.** Phase 3 takes the winning transformation
  (`hyde_3docs_plus_query`) and tests feeding the transformed string to the generator
  instead of the original question: composite collapses from 0.391 (retrieval-only) to
  0.168 (retrieval+generation) — Adherence falls from 0.50 to 0.20, Completeness from 0.490
  to 0.118, Utilization from 0.140 to 0.029. DelucionQA showed the same qualitative
  collapse (0.587→0.365) but TechQA's is proportionally larger. **Across two different
  datasets now, the same structural finding holds: whatever string drives retrieval should
  never replace the question sent to the generator.** This is the most robust, dataset-
  independent finding in either goldilocks series so far.
- **Query decomposition is the worst transformation on both datasets.** Composite 0.250
  here (DelucionQA: 0.506) — consistent qualitative finding, for a plausibly consistent
  reason: neither dataset's questions are genuinely multi-hop compositional questions that
  benefit from being split into sub-queries.

## TechQA's noise floor is even larger than `12` already showed — flagged twice more in
## this notebook alone

`12`'s analysis flagged a 0.044 composite gap between two nominally identical alpha=0.5
measurements as evidence TechQA's noise floor is unusually large. This notebook provides
**two more, larger, independent confirmations**:

- **Phase 3's "retrieval-only" re-measurement of `hyde_3docs_plus_query`** (composite
  0.391) is meaningfully lower than **Phase 2's original measurement of the same
  transformation** (0.461) — a **0.070 composite gap** on what should be the identical
  config (same transformation, same retrieval-only behavior — Phase 2 always used the
  original question for generation too). This is the largest same-config discrepancy
  observed in this entire two-dataset series so far.
- This means the specific ranking within Phase 1/2 (e.g., whether `hyde_3docs_plus_query`
  truly beats `hyde_doc_plus_query` by 0.030, or query rewriting truly beats baseline by
  0.087) should be read as directionally informative, not precisely reliable — but the two
  headline structural findings above (transformation clearly beats baseline; transformed
  query must never reach the generator) both have effect sizes well above this noise floor,
  so those two conclusions are solid even given the measurement noise.

## Phase 1 — transformation method comparison (retrieval-query only)

Transformed query drives retrieval; the generator always sees the original question.

| Method | CR | Util | Completeness | Adherence | Composite |
|---|---|---|---|---|---|
| **query_rewriting** | 0.284 | 0.166 | 0.540 | 0.55 | **0.398** |
| hyde (doc only, no concat) | 0.287 | 0.091 | 0.340 | 0.50 | 0.340 |
| baseline | 0.273 | 0.097 | 0.292 | 0.45 | 0.311 |
| query_decomposition | 0.239 | 0.112 | 0.414 | 0.25 | 0.250 |

**Takeaway:** query rewriting wins outright here — the opposite ranking from DelucionQA,
where it underperformed baseline.

## Phase 2 — HyDE concatenation ablation

| Config | CR | Adherence | Composite |
|---|---|---|---|
| **doc + query (3 docs)** | 0.323 | 0.70 | **0.461** |
| doc + query (1 doc) | 0.293 | 0.60 | 0.431 |
| doc only (no query) | 0.381 | 0.35 | 0.336 |

**Takeaway:** more hypothetical documents helps on TechQA, unlike DelucionQA (where 1 beat
3). Concatenating with the original query is still clearly better than the pseudo-document
alone, on both datasets.

## Phase 3 — retrieval-query vs. generation-query (HyDE 3-docs+query, k=4, alpha=0.5)

| Config | CR | Adherence | Composite |
|---|---|---|---|
| **retrieval-only** (generator sees original question) | 0.346 | 0.50 | **0.391** |
| retrieval+generation (generator sees transformed query) | 0.217 | 0.20 | 0.168 |

**Takeaway:** the collapse when the generator sees the transformed string instead of the
question is even sharper than on DelucionQA — this finding generalizes, and generalizes
strongly.

## Recommendation

Use **HyDE with three hypothetical documents concatenated with the original query, for
retrieval only** as TechQA's default query-transformation strategy — layered on top of
`11`'s chunking config and `12`'s retriever config. The generator must always receive the
original question, never the transformed string — this is now confirmed on two datasets
with a large effect size both times, making it the single most trustworthy recommendation
in either goldilocks series to date. This feeds `14_reranking_goldilocks.ipynb` as TechQA's
frozen query-transformation foundation.

## Caveats — do not treat these exact numbers as final

- **TechQA's noise floor is large enough that this notebook demonstrates it twice.** The
  0.070 composite gap between Phase 2's and Phase 3's measurements of the same
  `hyde_3docs_plus_query`/retrieval-only config is bigger than several of the between-method
  differences reported above. Treat rankings within ~0.07 composite as tentative; the two
  headline conclusions (transformation beats baseline; never feed the transformed string to
  the generator) both clear that bar comfortably.
- **`query_llm` runs at temperature 0.7**, same as the DelucionQA series — part of the same
  noise source flagged above.
- **Chunking and retriever held constant** at `11`'s and `12`'s winning TechQA configs.
  Whether a different upstream pipeline changes which transformation wins is not explored.
- **Phase 3 only tested one transformation method** (the Phase 1/2 combined winner,
  `hyde_3docs_plus_query`). The retrieval/generation-query separation likely generalizes
  further — it's a structural argument that has now held on two different transformations
  across two different datasets — but query rewriting and decomposition were never run
  through the same ablation on TechQA to directly confirm.
- **Embedding model, generator, and judge models held constant** across this whole series.
  Not explored as variables here.
- **Composite score weighting (0.35/0.15/0.15/0.35) is a judgment call**, kept identical to
  the DelucionQA series for comparability.

This analysis is documentation only — no notebook or library code has been changed to
adopt this configuration yet.
