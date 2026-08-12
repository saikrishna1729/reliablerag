# Retriever Goldilocks Analysis (TechQA)

Source: `12_retriever_goldilocks.ipynb`. Same three-phase search (method → k → method-specific
parameter) as `delucion_dataset`'s retriever goldilocks notebook, scored on a **fixed
20-question evaluation set reused across every config**, with judge annotation retries
instead of silently dropping failed samples. Every config is built on top of `11`'s real
TechQA chunking winner — recursive-character chunking, 128-token nominal target (~90 actual
tokens/chunk), 10% overlap (1649 chunks) — not DelucionQA's sentence+metadata chunking.
Composite score = 0.35·Relevance + 0.15·Utilization + 0.15·Completeness + 0.35·Adherence,
identical weighting to the DelucionQA series for direct comparability.

This notebook was later extended with a **40-question re-run of the identical three-phase
search**, prompted by `16_holdout_validation.ipynb`'s finding that this exact search was one
of the two most likely to be overfit to its 20-question sample. That re-run is the most
important part of this document — it directly contradicts the original 20-question winner.

## Headline finding: the 20q and 40q re-runs disagree on both method and k

| | Method | k | Composite |
|---|---|---|---|
| 20-question (original) | **mmr**, lambda_mult=0.5 | **4** | 0.4419 |
| 40-question (re-run) | **similarity** | **8** | 0.3700 |

Neither the winning retrieval method nor the winning depth reproduced when the sample size
doubled. This is direct, within-notebook evidence — consistent with `16`'s holdout-validation
result (a −0.124 composite gap between disjoint 20-question samples) — that the original
20-question Phase 1–3 search was picking up sampling noise, not a real, generalizable
retriever preference. **Neither number in isolation should be treated as final; see
Recommendation below.**

## Original 20-question run

### Phase 1 — retrieval method comparison (all at k=8)

| Method | Avg retrieved | CR | Util | Completeness | Adherence | Composite |
|---|---|---|---|---|---|---|
| **mmr (lambda=0.5)** | 8.0 | 0.153 | 0.101 | 0.604 | 0.70 | **0.404** |
| similarity | 8.0 | 0.234 | 0.127 | 0.453 | 0.65 | 0.396 |
| hybrid (alpha=0.5) | 8.0 | 0.241 | 0.117 | 0.408 | 0.45 | 0.321 |
| bm25 | 8.0 | 0.250 | 0.077 | 0.347 | 0.45 | 0.309 |

**Takeaway:** mmr edges out similarity by only 0.008 composite — another near-tie, same
pattern of tight margins TechQA has shown in every phase of this notebook.

### Phase 2 — k sweep (mmr, lambda_mult=0.5)

| k | Avg retrieved | CR | Adherence | Composite |
|---|---|---|---|---|
| **4** | 4.0 | 0.319 | 0.55 | **0.420** |
| 6 | 6.0 | 0.208 | 0.65 | 0.403 |
| 8 | 8.0 | 0.178 | 0.65 | 0.392 |
| 12 | 12.0 | 0.138 | 0.65 | 0.363 |
| 10 | 10.0 | 0.157 | 0.60 | 0.356 |
| 16 | 16.0 | 0.087 | 0.50 | 0.280 |

**Takeaway:** k=4 wins, and this time the rest of the curve is much closer to the
monotonic decline DelucionQA showed (unlike the earlier wobble seen in the pre-revision
version of this notebook).

### Phase 3 — lambda_mult sweep (mmr, k=4)

| lambda_mult | CR | Adherence | Composite |
|---|---|---|---|
| **0.5** | 0.318 | 0.60 | **0.442** |
| 0.75 | 0.366 | 0.50 | 0.387 |
| 0.0 | 0.323 | 0.45 | 0.382 |
| 1.0 | 0.337 | 0.50 | 0.374 |
| 0.25 | 0.259 | 0.50 | 0.365 |

**20-question winner: `mmr, k=4, lambda_mult=0.5`** — CR 0.318, Util 0.192, Completeness
0.613, Adherence 0.60, **composite 0.442**, 20/20 eval succeeded.

## 40-question re-run

Rebuilt the chunking foundation from a larger (50-question) corpus and repeated the exact
same method → k → parameter search against a fresh, larger 40-question fixed sample.

### Phase 1 (40q) — retrieval method comparison (all at k=8)

| Method | Avg retrieved | CR | Util | Completeness | Adherence | Composite |
|---|---|---|---|---|---|---|
| **similarity** | 8.0 | 0.206 | 0.095 | 0.433 | 0.625 | **0.370** |
| hybrid (alpha=0.5) | 8.0 | 0.204 | 0.089 | 0.427 | 0.600 | 0.359 |
| mmr (lambda=0.5) | 8.0 | 0.150 | 0.082 | 0.461 | 0.525 | 0.318 |
| bm25 | 8.0 | 0.167 | 0.071 | 0.373 | 0.475 | 0.291 |

**mmr — the 20q winner — drops to third place.** similarity, which placed second at 20q,
takes over.

### Phase 2 (40q) — k sweep (similarity)

| k | Avg retrieved | CR | Adherence | Composite |
|---|---|---|---|---|
| **8** | 8.0 | 0.215 | 0.600 | **0.370** |
| 10 | 10.0 | 0.176 | 0.600 | 0.350 |
| 6 | 6.0 | 0.271 | 0.450 | 0.313 |
| 12 | 12.0 | 0.131 | 0.525 | 0.300 |
| 4 | 4.0 | 0.306 | 0.375 | 0.298 |
| 16 | 16.0 | 0.106 | 0.475 | 0.272 |

**k=8 wins here — not k=4**, the opposite depth from the 20-question run, and this curve is
non-monotonic again (k=4 is near the bottom, not the top).

### Phase 3 (40q) — parameter sweep fallback

`similarity` has no extra tunable knob beyond `k`, so — mirroring `11`'s fallback behavior
when its own Phase 1 winner wasn't sweepable — this phase falls back to sweeping `mmr`'s
`lambda_mult` at the winning k=8 instead:

| lambda_mult | CR | Adherence | Composite |
|---|---|---|---|
| **0.5** | 0.170 | 0.575 | **0.351** |
| 1.0 | 0.197 | 0.575 | 0.343 |
| 0.75 | 0.184 | 0.525 | 0.321 |
| 0.25 | 0.123 | 0.525 | 0.318 |
| 0.0 | 0.160 | 0.400 | 0.282 |

No lambda value beats Phase 1's untuned `similarity` baseline (0.370). **40-question
winner stays `p1_40_similarity` — the plain Phase 1 default, composite 0.370** — Phases 2
and 3 did not find anything better than what Phase 1 already had.

## Why the two runs disagree

- **Every phase's Phase 1 margin was already too tight to be decisive.** 20q's
  mmr-vs-similarity gap was 0.008; 40q's similarity-vs-hybrid gap was 0.011. Composite
  differences this small are smaller than TechQA's demonstrated noise floor (`16` found a
  0.044–0.070 spread from resampling alone), so treating either ranking as a real
  preference was never well-supported — the 40q run just makes the consequence visible.
- **k's "clear winner" also didn't survive.** k=4 looked like a clean standout at 20
  questions (0.420 vs. next-best 0.403); at 40 questions k=4 is the *worst* result in the
  similarity sweep (0.298). Depth preference and sample composition are confounded here in
  a way this notebook can't disentangle on its own.
- **This is consistent with, not independent from, `16`'s holdout finding.** `16` flagged
  this exact Phase 1–3 search (among others) as high-risk for overfitting to the fixed
  20-question sample. This re-run is the direct confirmation: not a different dataset draw
  scored against the same config, but the same search re-run at 2x the sample size,
  landing on a different method and a different k.
- **TechQA's per-metric ceiling remains low and noisy regardless of which config wins** —
  Adherence tops out around 0.60–0.70 across both runs, well below DelucionQA's 0.80–1.00,
  reinforcing `11`'s and the earlier retriever analysis's conclusion that this is a
  structural property of TechQA's forum-thread answers, not something retriever tuning
  fixes.

## Recommendation

**Do not lock in either single-run winner as TechQA's retriever configuration.** Both the
20-question (`mmr, k=4, lambda=0.5`) and 40-question (`similarity, k=8`) results are
plausible but neither has survived a genuinely independent replication — the 40q sample
itself is drawn from the same head-of-dataset ordering as the 20q sample (see caveats), so
even this "disagreement" isn't proof of which one (if either) is right, only proof that at
least one of them is noise.

**Next steps, in priority order:**
1. **Run a third search against a genuinely disjoint held-out sample** (same mechanism as
   `16_holdout_validation.ipynb`, but applied to this retriever search specifically, not
   reused from `16`'s own — different — sweep target). If similarity/k=8 and mmr/k=4 keep
   trading places, the honest conclusion is "TechQA retrieval choice is underdetermined at
   this sample size," and a much larger fixed eval set (60–100+ questions) is needed before
   any config is trusted for `13_query_transformation_goldilocks.ipynb`.
2. **If forced to pick one now**, prefer **similarity, k=8** over mmr — it won the
   larger of the two samples, has no extra tunable knob to overfit, and is the cheaper
   retriever to run (no BM25/hybrid scoring overhead). This is a pragmatic default, not a
   validated winner.
3. **Increase `EVAL_SAMPLE_SIZE` globally** for any future goldilocks notebook in this
   series before trusting a 20-question single-run result — this is now the second
   notebook (after `16`'s explicit audit) to demonstrate the 20-question sample is too
   small to resolve TechQA's real effect sizes from its own noise.

## Caveats — do not treat these exact numbers as final

- **The 40-question sample is not a genuine holdout.** It is drawn with the same `head(N)`
  convention from the same underlying dataset ordering, so it likely overlaps heavily with
  (probably as a superset of) the original 20. This re-run demonstrates instability under
  more data from the *same region* of the dataset — it does not demonstrate what happens on
  fresh, disjoint data the way `16`'s holdout check does.
- **The 40q chunking foundation was rebuilt from a larger corpus** (more source docs → more
  chunks: 2727 vs. 1649). `16` already flagged corpus-size-sensitive BM25 IDF weighting as a
  plausible secondary contributor to composite differences across runs of different sizes —
  keep that in mind when comparing raw composite magnitudes across the 20q and 40q tables,
  not just the winning method/k choices.
- **Single run each, no repeated sampling within either the 20q or 40q condition.** Each
  table above is one run, not an average — the true noise band around any single composite
  number is unknown beyond what `16`'s separate audit already established.
- **Embedding model held constant** (`openai/text-embedding-3-small`). Not explored as a
  variable in this notebook.
- **Composite weighting (0.35/0.15/0.15/0.35)** is a judgment call, kept identical to the
  rest of the series for comparability. Given how close every phase's margins are, a
  different weighting could plausibly flip several of the rankings above.

This analysis is documentation only — no notebook or library code has been changed to
adopt any configuration yet.
