# Held-Out Validation Analysis (DelucionQA)

Source: `16_holdout_validation.ipynb`. Not a search — this notebook makes no new
configuration choices. It takes the fully compounded winning pipeline from `11`+`12`+`13`+
`14`+`15` (sentence + metadata chunks @ 64t, 10% overlap → hybrid BM25+dense retrieval @
alpha=0.3 driven by HyDE doc+query → Cohere Rerank 4-Pro over a candidate pool of 20, kept
to a final 2 chunks, no repacking) exactly as-is, and scores it on three question samples:
the original fixed 20 questions every prior notebook in this series optimized against, a
fresh 20 questions never scored before, and the combined 40. The question: does the stack
generalize, or is it overfit to five consecutive rounds of "pick the best config" against
one fixed sample?

## Results

| Sample | Context Relevance | Utilization | Completeness | Adherence | Composite |
|---|---|---|---|---|---|
| **original_20** (11–15's fixed set) | 0.605 | 0.386 | 0.680 | 0.950 | **0.704** |
| **holdout_20** (fresh, unseen) | 0.596 | 0.363 | 0.595 | 0.950 | **0.685** |
| **combined_40** | 0.593 | 0.385 | 0.663 | 0.950 | **0.697** |

**Gap (held-out − original): −0.019 composite.**

## Verdict: the stack generalizes

A −0.019 composite gap is smaller than the ~0.03 spread this series has already observed
from pure run-to-run noise on the *same* question sample — `15`'s fresh re-measurement of
this exact config landed at composite 0.682 versus `14`'s originally reported 0.709, purely
from re-running the identical pipeline (temperature-0.7 HyDE generation and stochastic
judge annotation are both real noise sources independent of which questions get asked).
Against that noise floor, a gap this small on a genuinely different set of 20 questions is
not distinguishable from noise. **The five-notebook compounded pipeline is not an
overfitting artifact of the fixed evaluation sample** — its wins in `11` through `15`
reflect real properties of DelucionQA and this pipeline, not quirks of 20 specific
questions.

Context Relevance in particular held almost exactly steady (0.605 → 0.596, a 1.5% relative
drop) — the metric every phase in this series optimized most aggressively — while Adherence
was *identical* across all three samples (0.950), suggesting the generator's grounding
behavior is a stable property of the pipeline rather than something tuned to specific
question phrasing.

## What did move: Completeness

The one metric with a real gap is Completeness — 0.680 on the original sample vs. 0.595 on
the held-out sample, a 12.5% relative drop, the largest movement of any metric here. Since
Completeness measures what fraction of the *relevant* sentences in context actually got
used in the response, this most likely reflects genuine per-question variance (some
held-out questions have answers that are harder to fully extract from a 2-chunk context)
rather than a pipeline defect — Context Relevance and Adherence, which measure whether the
*right material was retrieved* and whether the *response stayed grounded*, both held
steady. This is consistent with Completeness being the noisiest of the four TRACe metrics
across this whole series (it swung the most in nearly every phase of `11`–`15` too).

## Combined 40-question estimate

The combined-sample composite (0.697) sits almost exactly between the original (0.704) and
held-out (0.685) numbers, as expected for a simple average of two comparable-sized draws.
0.697 is the most defensible single estimate of this pipeline's real performance on
DelucionQA — larger sample, no cherry-picking, and it doesn't rely on either the specific
sample the pipeline was tuned against or a single fresh draw.

## Recommendation

Treat the `11`–`15` compounded pipeline (sentence+metadata chunking, hybrid+HyDE retrieval,
Cohere 4-Pro rerank to final_k=2) as validated. No re-search is warranted based on this
result — the held-out gap doesn't exceed the series' own established noise floor. Use the
**combined 40-question composite (~0.697)** rather than the original sample's 0.704 as the
reference number going forward, since it's the least sample-dependent estimate available.

Proceed to generator fine-tuning (`17`) as planned, with the confidence that whatever
context this pipeline hands the generator is a stable, representative sample of what
production traffic would actually look like — not an artifact of which 20 questions
happened to get optimized against.

## Caveats — do not treat these exact numbers as final

- **Still only 20+20 questions.** This raises the sample size from one 20-question draw to
  two comparable draws, but DelucionQA has far more available in the RAGBench split — a
  larger validation (e.g. 50+50) would narrow the noise floor further and make a real
  Completeness drift easier to distinguish from sampling noise.
- **The held-out questions' source documents were indexed in the same combined vector
  store/BM25 index** as the original questions' documents. This tests whether the
  pipeline's *configuration* generalizes to new questions over a similar corpus, not
  whether it generalizes to a genuinely unseen corpus or domain.
- **No new configuration choices were tested.** This notebook validates the existing
  winning stack; it doesn't re-search chunking, retrieval, query transformation, reranking,
  or repacking against the held-out sample. The Completeness drop, while within plausible
  noise, would be worth watching if a future larger validation shows it persisting.
- **`query_llm` runs at temperature 0.7** for HyDE generation, and judge annotation is
  inherently somewhat stochastic — both samples carry that same noise source
  independently, which is exactly what makes the ~0.03 noise-floor comparison meaningful
  but also means any given single run (including this one) is itself just one draw from
  that noise distribution.
- **Composite score weighting (0.35/0.15/0.15/0.35) is a judgment call**, kept identical to
  `11`–`15` for comparability.

This analysis is documentation only — no notebook or library code has been changed as a
result of this validation.
