# Held-Out Validation Analysis (TechQA)

Source: `16_holdout_validation.ipynb`. Not a search — this notebook makes no new
configuration choices. It takes the fully compounded TechQA winning pipeline from `11`+`12`+
`13`+`14`+`15` (recursive-character chunking @ 128t nominal, 10% overlap → hybrid BM25+dense
retrieval @ alpha=0.5 driven by HyDE 3-docs+query → Cohere Rerank 4-Pro over a candidate
pool of 30, kept to a final 2 chunks, no repacking) exactly as-is, and scores it on three
question samples: the original fixed 20 questions every prior TechQA notebook optimized
against, a fresh 20 questions never scored before, and the combined 40.

## Results

| Sample | Context Relevance | Utilization | Completeness | Adherence | Composite |
|---|---|---|---|---|---|
| **original_20** (11–15's fixed set) | 0.577 | 0.343 | 0.623 | 0.750 | **0.609** |
| **holdout_20** (fresh, unseen) | 0.393 | 0.247 | 0.555 | 0.650 | **0.485** |
| **combined_40** | 0.518 | 0.320 | 0.605 | 0.650 | **0.548** |

**Gap (held-out − original): −0.124 composite.**

## Verdict: this does NOT generalize the way DelucionQA's stack did

This is a materially different outcome from `delucion_dataset/16_holdout_validation.ipynb`,
which found a −0.019 gap — comfortably inside DelucionQA's own noise floor. TechQA's
−0.124 gap is nearly **double** the 0.07 threshold this notebook was deliberately
recalibrated to (based on TechQA's own already-demonstrated 0.044–0.070 noise spreads from
`12` and `13`). **This is real evidence that some of `11`–`15`'s winning choices for TechQA
are overfit to the specific 20 questions they were tuned against, not just measurement
noise.**

Unlike DelucionQA's held-out check (where only Completeness moved meaningfully and Context
Relevance/Adherence held essentially flat), **every metric drops on TechQA's held-out
sample**: Context Relevance falls 32% relative (0.577→0.393, the largest mover by far),
Utilization falls 28% (0.343→0.247), Adherence falls 13% (0.75→0.65), and even
Completeness — TechQA's most volatile metric throughout this series — falls 11%
(0.623→0.555). A broad, all-metrics decline like this looks more like genuine overfitting to
the fixed sample than like noise concentrated in one already-known-volatile metric.

## Why this likely happened, and where to look first

- **Five sequential rounds of "pick the best of several close options" compound their
  optimism.** Each of `11`–`15` independently exhibited some degree of the "optimizer's
  curse" — picking whichever config scored highest on one fixed 20-question draw
  systematically overestimates that config's *true* performance, especially when margins
  between the top few options were narrow. `12`'s alpha sweep found *no* value that
  reliably beat the default (0.044 gap on a same-config rerun); `15`'s narrow-vs-repacked
  margin (+0.026) was explicitly flagged as sitting inside TechQA's noise floor. Stacking
  five such selections, several with narrow or noisy margins, compounds the inflation far
  more than DelucionQA's stack did — DelucionQA's phase margins were generally cleaner and
  larger relative to its own (smaller) noise floor.
- **`12` (retriever alpha) and `15` (repacking vs. narrow) are the most likely sources of an
  overfit choice**, based on how narrow their own margins already were relative to TechQA's
  demonstrated noise. `11` (chunking method: recursive_char beat every alternative by a wide
  margin, 0.382 vs. 0.319 runner-up) and `14` (reranking: +0.220 over no-reranking, this
  series' largest and most decisive single-phase gain) are more likely to be genuinely
  robust findings that would survive a re-search on a different sample.
- **Corpus growth is a plausible contributing (not sole) factor, worth naming even though it
  doesn't fully explain the gap.** This notebook's combined corpus (3248 chunks from 60
  questions) is roughly double `11`–`15`'s own (1649 chunks from 30 questions). BM25's IDF
  weighting is computed over the whole indexed corpus, so a larger corpus measurably shifts
  which chunks the hybrid retriever's sparse component favors — independent of any
  overfitting to specific questions. However, DelucionQA's own held-out notebook grew its
  corpus by a similar proportion (90→180 chunks) without producing a comparable gap, so
  corpus growth alone doesn't explain TechQA's much larger drop; it's more likely
  compounding with genuine per-phase overfitting than acting alone.
- **TechQA's held-out sample may simply be a harder draw**, consistent with `11`'s original
  finding that TechQA's achievable composite ceiling (0.401 even at the chunking stage
  alone) is structurally lower and noisier than DelucionQA's (0.556) — a dataset with a
  lower, noisier ceiling to begin with will show more sample-to-sample variation in which
  20-question draw looks best.

## Recommendation

**Don't treat `11`–`15`'s reported TechQA composite numbers as final performance
estimates.** Use the **combined 40-question composite (0.548)** as the more defensible
reference point going forward, rather than the original sample's 0.609 — it's still
optimistic relative to a fully independent validation, but it's the least
sample-dependent number available without collecting more data.

Given the size of this gap, a proportionate next step (before generator/embedding work in
`17`) would be re-running `12` (retriever) and `15` (repacking) specifically — the two
notebooks whose margins were already the narrowest — against a larger or rotating question
sample, to check whether their specific winning choices (alpha=0.5, no repacking) hold up or
were themselves artifacts of the fixed sample. `11` and `14`'s findings are less urgent to
re-verify given their much larger margins.

## Caveats — do not treat these exact numbers as final

- **Still only 20+20 questions**, and this notebook's own headline finding is that 20
  questions isn't enough to pin down TechQA's true pipeline performance reliably — a larger
  validation (e.g. 50+50) is the natural follow-up, not just a nice-to-have.
- **The held-out questions' source documents were indexed in the same combined vector
  store/BM25 index** as the original questions' documents (necessary for retrieval to work
  at all), which is also the mechanism behind the corpus-growth caveat above.
- **No new configuration choices were tested here** — this notebook only measures the
  existing stack's generalization gap, it doesn't diagnose which specific phase's choice is
  responsible. The "most likely candidates" identified above (`12`, `15`) are inferred from
  each notebook's own already-reported margin narrowness, not from a controlled ablation run
  in this notebook.
- **`query_llm` runs at temperature 0.7** for the 3-document HyDE generation, and judge
  annotation is inherently somewhat stochastic — both samples carry this same noise
  source independently, on top of whatever genuine overfitting this gap reflects.
- **Composite score weighting (0.35/0.15/0.15/0.35) is a judgment call**, kept identical to
  `11`–`15` for comparability.

This analysis is documentation only — no notebook or library code has been changed as a
result of this validation. Whether to re-run `12`/`15` against a larger sample, or proceed
to `17` with the caveat noted, is a scope decision for the next step.
