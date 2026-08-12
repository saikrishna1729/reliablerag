# Reranking Goldilocks Analysis (TechQA)

Source: `14_reranking_goldilocks.ipynb`. Same three-phase search (reranker model → candidate
pool width → final kept count) as `delucion_dataset`'s reranking goldilocks notebook, scored
on a **fixed 20-question evaluation set reused across every config**, with judge annotation
retries instead of silently dropping failed samples. Built on TechQA's real winning
foundation: `11`'s chunking (recursive_char, 128t nominal, 10% overlap), `12`'s retriever
(hybrid, alpha=0.5), and `13`'s query transformation (HyDE, **3** hypothetical documents +
query, retrieval-only). Composite score = 0.35·Relevance + 0.15·Utilization +
0.15·Completeness + 0.35·Adherence, identical weighting to the DelucionQA series.

## Winning configuration

**Cohere Rerank 4-Pro, candidate pool of 30 hybrid-ranked chunks reranked down to a final 2
chunks.**

| Metric | Value |
|---|---|
| Context Relevance | 0.621 |
| Utilization | 0.401 |
| Completeness | 0.635 |
| Adherence | 0.650 |
| Composite | **0.600** |
| Eval sample | 20/20 succeeded |
| vs. no-reranking control (candidate_k=20→final_k=4) | **+0.220 composite** |

This mirrors DelucionQA's `14` almost exactly on structure — same winning reranker (Cohere
4-Pro), same "narrow to final_k=2" shape — but the **improvement over doing nothing is more
than double** DelucionQA's (+0.220 here vs. +0.089 there). Reranking matters even more on
TechQA than it did on DelucionQA.

## Why this result looks the way it does

- **Cohere 4-Pro wins Phase 1 clearly** (composite 0.498), ahead of NVIDIA Nemotron (0.431),
  Cohere v3.5 (0.417), and the no-reranking control (0.380, which also had TechQA's first
  judge-annotation failure seen in this notebook — 19/20 succeeded, a reminder that even the
  retry mechanism doesn't guarantee every config scores on an identical sample size). Same
  reranker quality ordering DelucionQA found (4-Pro > v3.5, Nemotron competitive), giving a
  second independent confirmation that Cohere 4-Pro is simply the strongest of the three
  tested rerankers, not a dataset-specific fluke.
- **TechQA needed a wider candidate pool to hit its peak (30, vs. DelucionQA's 20)** — Phase
  2's composite climbs from 0.426 (candidate_k=4) up to 0.514 (candidate_k=30), though not
  monotonically: candidate_k=12 dips to 0.426, below both its neighbors (ck=8 at 0.483 and
  ck=20 at 0.509). This is the same kind of local wobble `12`'s k-sweep and `11`'s
  chunk-size sweep both showed for TechQA — consistent with this dataset's larger noise
  floor (already flagged repeatedly in `12` and `13`'s analyses) rather than a genuine
  non-monotonic relationship.
- **Phase 3's final_k sweep, by contrast, is clean and close to monotonic** — composite
  falls smoothly from 0.600 (final_k=2) through 0.534, 0.483, 0.459, to 0.452 (final_k=12).
  This is the same "narrow final context wins" pattern found for retrieval depth (`12`) and
  now reranking depth (`14`) on both datasets — the most consistently reproduced structural
  finding across the whole two-dataset series, alongside the retrieval/generation-query
  separation from `13`.
- **A genuine, visible precision/adherence tradeoff on TechQA that DelucionQA never showed.**
  On DelucionQA, Adherence stayed roughly flat and already-high (0.85–1.00) across every
  final_k tested — narrowing context there was close to a free win on every metric at once.
  On TechQA, Adherence instead *rises* as final_k grows: 0.65 (fk=2) → 0.75 (fk=4) → 0.70
  (fk=6) → 0.75 (fk=8) → 0.85 (fk=12) — the widest context tested has the best grounding,
  not the worst. The winning fk=2 config still takes the composite because its Context
  Relevance (0.621) and Completeness (0.635) gains are large enough to outweigh the
  Adherence cost, but this is a real tradeoff being resolved by the composite weighting, not
  an all-metrics-agree win the way DelucionQA's narrow-context result was. This is
  consistent with the pattern established since `11`: TechQA's technical troubleshooting
  answers plausibly need more supporting context to be *fully* justified than DelucionQA's
  single-fact manual lookups do, even though a narrow, highly relevant context still wins on
  overall composite.

## Phase 1 — reranker model comparison (candidate_k=20, final_k=4)

| Reranker | CR | Util | Completeness | Adherence | Composite |
|---|---|---|---|---|---|
| **cohere_4pro** | 0.423 | 0.238 | 0.465 | 0.700 | **0.498** |
| nvidia_nemotron | 0.458 | 0.236 | 0.521 | 0.450 | 0.431 |
| cohere_v35 | 0.379 | 0.190 | 0.420 | 0.550 | 0.417 |
| no_rerank (control) | 0.268 | 0.113 | 0.324 | 0.632* | 0.380 |

\* no_rerank's Adherence is computed over 19/20 successful evaluations (1 judge-annotation
failure survived 3 retry attempts).

**Takeaway:** Cohere 4-Pro wins clearly, same ranking DelucionQA found.

## Phase 2 — candidate pool width sweep (cohere_4pro, final_k=4)

| candidate_k | CR | Adherence | Composite |
|---|---|---|---|
| **30** | 0.427 | 0.80 | **0.514** |
| 20 | 0.428 | 0.70 | 0.509 |
| 8 | 0.435 | 0.65 | 0.483 |
| 12 | 0.427 | 0.45 | 0.426 |
| 4 | 0.355 | 0.55 | 0.426 |

**Takeaway:** wider pools generally help (30 > 20 > 8 > {4,12}), but candidate_k=12's dip
breaks a clean monotonic trend — read this axis as "wider is generally better up to ~30,"
not as a precisely ordered relationship.

## Phase 3 — final kept-count sweep (cohere_4pro, candidate_k=30)

| final_k | CR | Adherence | Composite |
|---|---|---|---|
| **2** | 0.621 | 0.65 | **0.600** |
| 4 | 0.460 | 0.75 | 0.534 |
| 6 | 0.353 | 0.70 | 0.483 |
| 8 | 0.226 | 0.75 | 0.459 |
| 12 | 0.156 | 0.85 | 0.452 |

**Takeaway:** the cleanest, most reproducible relationship in this notebook — Context
Relevance falls steadily as final_k grows, Adherence rises, and composite still favors the
narrow end because the Relevance/Completeness swing is larger than the Adherence swing.

## Recommendation

Use **Cohere Rerank 4-Pro, retrieving a candidate pool of 30 hybrid-ranked chunks (`11`
chunking + `12` retriever + `13` HyDE-3-docs-for-retrieval) and keeping only the top 2 after
reranking** as TechQA's default reranking configuration. This is the strongest result in
the TechQA series so far (composite 0.600), and the margin over doing nothing (+0.220) is
TechQA's largest, most decisive single-phase improvement to date — well clear of the
~0.05–0.07 composite noise floor this series has repeatedly demonstrated. This feeds
`15_document_repacking_goldilocks.ipynb` as TechQA's frozen reranking foundation.

## Caveats — do not treat these exact numbers as final

- **Single run, one fixed 20-question sample.** Same noise-floor caveat as every prior
  notebook in this series — treat composite differences smaller than ~0.05–0.07 as
  inconclusive, per `12` and `13`'s own within-notebook demonstrations of that spread.
  Phase 3's final_k relationship and the overall reranking-vs-no-reranking gap both clear
  that bar comfortably; Phase 2's candidate-k ordering (30 vs. 20, and the ck=12 dip) does
  not.
- **Chunking, retriever, and query transformation held constant** at `11`, `12`, and `13`'s
  winning TechQA configs. Whether a different upstream pipeline changes which reranker or
  final_k wins is not explored here.
- **`query_llm` runs at temperature 0.7** for the 3-document HyDE retrieval string feeding
  every config's candidate pool — a repeat run could shuffle close results even on the same
  fixed question set, and 3 independent HyDE generations per query means more stochastic
  surface area than DelucionQA's single-document version had.
- **The TechQA-specific Adherence-vs-final_k tradeoff is a new pattern not seen on
  DelucionQA** — worth treating as a genuine dataset property rather than assuming it will
  reverse or disappear with more data, since it's consistent with the "TechQA needs more
  supporting context to fully ground answers" theme established across `11`, `12`, and `13`.
- **The OpenRouter rerank endpoint is a third-party dependency** with its own latency, cost,
  and availability characteristics not modeled by the composite score.
- **Composite score weighting (0.35/0.15/0.15/0.35) is a judgment call**, kept identical to
  the DelucionQA series for comparability. Given TechQA's visible Adherence/Relevance
  tradeoff at the final_k axis, a deployment that weights faithfulness more heavily might
  reasonably prefer a wider final_k (e.g. 4 or 6) despite the lower composite here.

This analysis is documentation only — no notebook or library code has been changed to
adopt this configuration yet.
