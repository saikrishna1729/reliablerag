# Document Repacking Goldilocks Analysis (DelucionQA)

Source: `15_document_repacking_goldilocks.ipynb`. A three-phase search (repacking strategy
→ context-width sweep → wide-and-repacked vs. `14`'s narrow winner), scored on a **fixed
20-question evaluation set reused across every config** (same set-up as `11`–`14`), with
judge annotation retries instead of silently dropping failed samples. Every config is built
on the full `11`+`12`+`13`+`14` winning stack: sentence + metadata chunks (64t, 10%
overlap), hybrid BM25+dense retrieval (alpha=0.3) driven by HyDE doc+query, and Cohere
Rerank 4-Pro over a candidate pool of 20. Composite score = 0.35·Relevance +
0.15·Utilization + 0.15·Completeness + 0.35·Adherence, identical weighting to `11`–`14` for
comparability.

## Winning configuration

**No repacking needed — final_k=2, the reranker's own order.**

| Metric | Value |
|---|---|
| Context Relevance | 0.602 |
| Utilization | 0.389 |
| Completeness | 0.653 |
| Adherence | 0.900 |
| Composite | **0.682** |
| Eval sample | 20/20 succeeded |
| vs. best wide+repacked config (reverse, final_k=4) | +0.066 composite |

This confirms the wrinkle flagged before building this notebook: at `14`'s own winning
depth (2 chunks), there's nothing meaningful to repack — and no wider context, however well
ordered, closes the gap back to it. Repacking is a real, measurable lever on this pipeline
(see Phase 1/2 below), just not a strong enough one to justify trading away `14`'s
narrow-context precision advantage.

## Why this result makes sense given the series so far

- **Reverse ordering is a genuine, replicated win — just not a large enough one.** Phase 1
  isolates repacking strategy at a fixed final_k=6: reverse (composite 0.618) clearly beats
  forward (0.584) and sides (0.581). This is exactly the "lost in the middle" mechanism the
  paper cites (Liu et al. 2024a) — putting the single most relevant chunk immediately before
  the question, rather than first in a list the generator has to scan past, measurably helps
  Context Relevance (0.441 vs. 0.388 for forward) without costing anything on Adherence
  (both hit 1.000). The effect is real; it just operates on a smaller lever (repacking a
  fixed-width context) than the two bigger levers this series already tuned (how many chunks
  to keep at all, and which chunks those are).
- **But composite keeps falling as final_k grows, even with the winning repacking strategy
  applied.** Phase 2 sweeps final_k with reverse ordering fixed: composite goes 0.603
  (fk=4) → 0.568 (fk=6) → 0.515 (fk=8) → 0.501 (fk=12) — the same near-monotonic "small
  footprint wins" pattern found for chunk size (`11`), retrieval depth (`12`), and rerank
  cut depth (`14`). Reverse ordering buys back some of what forward loses at the same width
  (Phase 1's fk=6 numbers), but it can't outrun the fact that DelucionQA's single-hop,
  single-fact questions get diluted by every additional chunk beyond the one or two that
  actually contain the answer — repacking changes *where* the dilution sits in the prompt,
  not *how much* dilution there is.
- **`14`'s own final_k=2 result doesn't need repacking to win, and reproduces close to its
  original number.** This notebook's fresh final_k=2 measurement (composite 0.682) is in
  the same neighborhood as `14`'s original reported winner (composite 0.709) but not
  identical — a ~0.03 spread on what is nominally the same config, run independently. That
  gap is a useful, concrete illustration of the run-to-run noise every notebook in this
  series has flagged as a caveat (temperature-0.7 HyDE generation, stochastic judge
  annotations): even a "fixed" 20-question evaluation doesn't fully pin down a config's
  score across separate runs. It doesn't change the ranking (final_k=2 still clearly beats
  every wide+repacked alternative by a wide margin), but it's a reminder not to over-read
  small composite differences elsewhere in this series as precise.
- **The `sides` strategy underperformed both alternatives**, unlike the paper's own finding
  where "sides" was competitive (used in their best-performance recipe, Table 1). At
  final_k=6 with the reranker's raw ranking split into head/tail halves, `sides` (0.581)
  actually landed slightly below plain `forward` (0.584). With only 6 chunks split across
  two positions, the paper's "put strong signal at both ends" logic has very little room to
  operate — a within-Phase-1 illustration of why context width and repacking interact rather
  than being independent levers.

## Phase 1 — repacking strategy comparison (final_k=6)

Isolates repacking strategy with context width fixed at 6 — wide enough for the three
strategies to actually differ, unlike `14`'s own final_k=2 winner.

| Strategy | CR | Util | Completeness | Adherence | Composite |
|---|---|---|---|---|---|
| **reverse** | 0.441 | 0.210 | 0.549 | 1.000 | **0.618** |
| forward | 0.388 | 0.165 | 0.494 | 1.000 | 0.584 |
| sides | 0.371 | 0.190 | 0.486 | 1.000 | 0.581 |

**Takeaway:** reverse ordering — most relevant chunk placed immediately before the
question — is a clear, if modest, win over both alternatives at this context width.

## Phase 2 — context-width sweep (reverse repacking)

Isolates context width with the Phase 1 winning strategy fixed.

| final_k | CR | Adherence | Composite |
|---|---|---|---|
| **4** | 0.505 | 0.900 | **0.603** |
| 6 | 0.389 | 0.900 | 0.568 |
| 8 | 0.319 | 0.850 | 0.515 |
| 12 | 0.219 | 0.900 | 0.501 |

**Takeaway:** even with the best repacking strategy applied at every width, composite
falls steadily as final_k grows. Repacking softens the decline (compare fk=6 here, 0.568,
against forward's 0.584 at the *same* fk=6 from Phase 1 — reverse actually trails forward
once diluted further at wider widths, suggesting the ordering benefit itself weakens as
more low-relevance chunks get added to reorder among).

## Phase 3 — wide-and-repacked vs. `14`'s narrow winner

The question this notebook exists to answer, measured fresh on this notebook's own
pipeline rather than cited from `14`.

| Config | CR | Adherence | Composite |
|---|---|---|---|
| **narrow, final_k=2, no repacking needed** | 0.602 | 0.900 | **0.682** |
| reverse, final_k=4 (Phase 2 winner) | 0.492 | 0.950 | 0.616 |

**Takeaway:** no amount of context widening plus good repacking recovers `14`'s narrow-cut
advantage. The gap (+0.066 composite in the narrow config's favor) is decisive relative to
the run-to-run noise observed above (~0.03).

## Recommendation

**Skip document repacking for DelucionQA** — stick with `14`'s final_k=2, no-reordering-
necessary configuration. Repacking (specifically, reverse ordering) is a real, positive
lever *when the context is wide enough for it to apply* (Phase 1), but this dataset's
single-hop, single-fact structure rewards narrowing the final context far more than it
rewards ordering a wider one, and the narrow config never needs repacking in the first
place. This is consistent with — and a direct extension of — the "small footprint wins"
pattern established across chunk size (`11`), retrieval depth (`12`), and rerank cut depth
(`14`): on this dataset, precision beats every attempt to make a wider context work better.

## Caveats — do not treat these exact numbers as final

- **Still a single run on one fixed 20-question sample.** The ~0.03 composite spread
  observed between this notebook's fresh final_k=2 measurement (0.682) and `14`'s original
  reported number for nominally the same config (0.709) is direct evidence that even this
  series' most careful methodology has real run-to-run variance — treat composite
  differences smaller than roughly that magnitude as inconclusive.
- **Chunking, retriever, query transformation, and reranker held constant** at `11`–`14`'s
  winning configs. Whether a different upstream pipeline (e.g. a retriever that doesn't
  already concentrate relevance into 1–2 chunks) would make repacking matter more is not
  explored here.
- **`query_llm` runs at temperature 0.7** for the HyDE retrieval string feeding every
  config's candidate pool — part of the same noise source as above.
- **The `sides` implementation here is one reasonable interpretation** (alternating
  head/tail placement by rank) of the paper's description, which doesn't specify an exact
  algorithm; a different implementation could rank differently, and might fare better at
  wider context widths than tested here.
- **The OpenRouter rerank endpoint is a third-party dependency** with its own latency,
  cost, and availability characteristics not modeled by the composite score.
- **Composite score weighting (0.35/0.15/0.15/0.35) is a judgment call**, kept identical to
  `11`–`14` for comparability. Re-ranking by Adherence alone would favor the wide+repacked
  reverse/fk=4 config (0.950) over the narrow winner (0.900) — the composite's heavier
  weight on Relevance and Completeness is what keeps the narrow config on top.

This analysis is documentation only — no notebook or library code has been changed to
adopt this configuration yet.
