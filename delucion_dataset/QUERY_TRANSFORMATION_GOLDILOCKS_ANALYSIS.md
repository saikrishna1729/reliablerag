# Query Transformation Goldilocks Analysis (DelucionQA)

Source: `13_query_transformation_goldilocks.ipynb`. A three-phase search (transformation
method → HyDE concatenation ablation → retrieval-query vs. generation-query), scored on a
**fixed 20-question evaluation set reused across every config** (same set-up as
`11_chunking_goldilocks.ipynb` / `12_retriever_goldilocks.ipynb`), with judge annotation
retries instead of silently dropping failed samples. Every config is built on top of `11`'s
winning chunking config (sentence-level + metadata heading, 64t, 10% overlap, 456 chunks)
and `12`'s winning retriever config (hybrid BM25+dense, k=4, alpha=0.3) — unlike `06`'s
original run, which froze an arbitrary 192t-semantic/Dense-MMR pairing this repo no longer
believes is its best pipeline. Composite score = 0.35·Relevance + 0.15·Utilization +
0.15·Completeness + 0.35·Adherence, identical weighting to `11`/`12` for comparability.

## Winning configuration

**HyDE, one hypothetical document concatenated with the original query, used for retrieval
only — the original question always goes to the generator.**

| Metric | Value |
|---|---|
| Context Relevance | 0.424 |
| Utilization | 0.219 |
| Completeness | 0.588 |
| Adherence | 1.000 |
| Composite | **0.620** |
| Eval sample | 20/20 succeeded |
| vs. no-transformation baseline | +0.0065 composite |

This edges out the no-transformation baseline (composite 0.613) by a margin narrow enough
to be within run-to-run noise (see caveats) — a meaningfully different conclusion from `06`,
which found every transformation *hurt* by 20–40% relative to baseline. Both notebooks
tested DelucionQA; what changed is the foundation they were tested on top of, and how
transformation touched the generation prompt.

## Why the result flipped from `06`

- **`06` conflated "does this help retrieval" with "does this confuse the generator."** Its
  worst-performing method (HyDE, −21.6% vs. baseline in `06`) fed the LLM's own generated
  hypothetical *document* — a full invented paragraph, not the user's question — directly
  into the generation prompt as if it were the question. Phase 3 here isolates that: the
  identical HyDE-doc-plus-query transformation scores composite 0.587 when the generator
  sees the original question, and composite 0.365 — a catastrophic Adherence collapse from
  0.95 to 0.50 — when the generator instead sees the transformed string. HyDE was never
  actually bad for *retrieval* in `06`; it was bad for the generator's prompt, which is a
  different failure entirely and one `06`'s single "transformed_question" variable
  couldn't separate.
- **`06` also compared transformations on top of a foundation this repo has since found is
  worse than achievable.** `11` showed 64-token sentence+metadata chunks beat 192-token
  semantic chunks by a wide margin (composite 0.556 vs. 0.383 in Phase 1 of that search),
  and `12` showed hybrid retrieval at k=4 beats Dense MMR at k=8 (0.622 vs. 0.529). Compound
  small effects across two frozen layers plus the conflation above were enough to flip HyDE
  from `06`'s worst-scoring method to this notebook's best.
- **The paper's own Table 8 finding (concatenate the hypothetical document with the
  original query, don't use it alone) replicates on DelucionQA.** Phase 2:
  `hyde_doc_plus_query` (composite 0.620) clearly beats `hyde_doc_only` (composite 0.531).
  The mechanism the paper describes for TREC DL still applies here: the pseudo-document
  captures topical/semantic signal the short original question lacks, but the original
  question's own exact wording still carries retrieval-relevant terms (part names, manual
  section vocabulary) that get diluted or lost if the pseudo-document replaces it outright.
- **The paper's other lever — more hypothetical documents — does *not* replicate here.**
  `hyde_3docs_plus_query` (composite 0.572) underperforms the single-document version
  (0.620), the opposite of the paper's TREC DL finding (Table 8: 8 pseudo-docs + query beat
  1 pseudo-doc + query, mAP 51.64 vs 50.87). DelucionQA's single-hop, single-fact answers
  likely don't benefit from covering multiple angles the way TREC DL's more open-ended
  passage-ranking queries do — three independently-sampled 256-token hypothetical
  paragraphs (at temperature 0.7) probably introduce topic drift a single, more targeted
  document doesn't, diluting the 64-token chunks' already-tight relevance signal.
- **Query rewriting and decomposition still underperform baseline**, same qualitative
  finding as `06` (rewriting: composite 0.586 vs. baseline 0.613; decomposition: 0.506).
  Consistent with the "single-hop factual lookup" character of DelucionQA established in
  `11`'s and `12`'s analyses — there's no multi-part question to decompose, and the original
  question is usually already a clean, literal match for manual vocabulary, so rewriting it
  mostly just moves it away from exact-term matches the hybrid retriever's BM25 component
  depends on.

## Phase 1 — transformation method comparison (retrieval-query only)

Transformed query drives retrieval; the generator always sees the original question.

| Method | CR | Util | Completeness | Adherence | Composite |
|---|---|---|---|---|---|
| **baseline** | 0.439 | 0.245 | 0.601 | 0.950 | **0.613** |
| query_rewriting | 0.382 | 0.185 | 0.614 | 0.950 | 0.586 |
| hyde (doc only, no concat) | 0.461 | 0.202 | 0.489 | 0.850 | 0.562 |
| query_decomposition | 0.361 | 0.183 | 0.484 | 0.800 | 0.506 |

**Takeaway:** with the generator isolated from the transformed string, baseline wins this
phase outright — plain HyDE (no query concatenation) has the *highest* raw Context
Relevance (0.461) but the lowest Adherence among non-decomposition methods (0.850), pulling
its composite below baseline. This phase alone reproduces `06`'s qualitative conclusion; the
result only flips once Phase 2's concatenation fix is applied.

## Phase 2 — HyDE concatenation ablation (replicating the paper's Table 8)

| Config | CR | Adherence | Composite |
|---|---|---|---|
| **doc + query (1 doc)** | 0.424 | 1.000 | **0.620** |
| doc + query (3 docs) | 0.349 | 0.950 | 0.572 |
| doc only (no query) | 0.381 | 0.850 | 0.531 |

**Takeaway:** concatenating the hypothetical document with the original query — not using
either alone — is the single largest lever in this notebook, worth +0.089 composite over
doc-only. Adding more documents doesn't help on this dataset; one is enough and three
actively hurts.

## Phase 3 — retrieval-query vs. generation-query (HyDE doc+query, k=4, alpha=0.3)

| Config | CR | Adherence | Composite |
|---|---|---|---|
| **retrieval-only** (generator sees original question) | 0.395 | 0.950 | **0.587** |
| retrieval+generation (generator sees transformed query) | 0.389 | 0.500 | 0.365 |

**Takeaway:** Context Relevance is nearly identical between the two (0.395 vs. 0.389) —
confirming the transformation's effect on retrieval quality is real and separable — but
Adherence collapses from 0.95 to 0.50 the moment the generator is handed the hypothetical
document instead of the actual question. This is the single clearest finding in the
notebook: **query transformation should be a retrieval-only concern; whatever string drives
retrieval should never replace the question sent to the generator.**

## Recommendation

Use **HyDE with a single hypothetical document concatenated with the original query, for
retrieval only** as the default query-transformation strategy for DelucionQA — layered on
top of `11`'s chunking config (sentence + metadata, 64t) and `12`'s retriever config (hybrid,
k=4, alpha=0.3). The generator must always receive the original question, never the
transformed string. Given the narrow margin over baseline (+0.0065 composite, see
caveats), treat this as "worth adopting when free" rather than a decisive win — the
much larger and more robust finding is the retrieval/generation-query separation in Phase 3.

## Caveats — do not treat these exact numbers as final

- **Still a single run on one fixed 20-question sample.** The margin between the winning
  config (0.620) and baseline (0.613) is well within what a different fixed sample or a
  second run could plausibly reorder — this is not a strong win the way Phase 3's
  retrieval/generation-query gap (0.587 vs. 0.365) is.
- **`query_llm` runs at temperature 0.7**, unlike the deterministic chunking/retriever
  configs in `11`/`12`. HyDE and decomposition outputs vary run to run for the same
  question; a repeat pass could shuffle Phase 1/2's close rankings even with the same fixed
  question set.
- **Chunking and retriever held constant** at `11`'s and `12`'s winning configs. Whether a
  different chunking/retriever pairing changes which transformation wins is not explored.
- **Phase 3 only tested one transformation method** (HyDE doc+query, the best performer
  from Phases 1–2). The retrieval/generation-query separation likely generalizes — it's a
  structural argument, not specific to HyDE — but query rewriting and decomposition were
  never run through the same ablation to confirm.
- **Embedding model, generator, and judge models held constant** across this whole series.
  Not explored as variables here.
- **Composite score weighting (0.35/0.15/0.15/0.35) is a judgment call**, kept identical to
  `11`/`12` for comparability. Re-ranking by Context Relevance alone would favor plain HyDE
  without concatenation (0.461, Phase 1) over the composite's winner (0.424) — the
  composite's Adherence weighting is what pulls concatenation ahead.

This analysis is documentation only — no notebook or library code has been changed to
adopt this configuration yet.
