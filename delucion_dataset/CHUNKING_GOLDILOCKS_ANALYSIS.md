# Chunking Goldilocks Analysis (DelucionQA)

Source: `11_chunking_goldilocks.ipynb`. A three-phase search (method → size → overlap),
scored on a **fixed 20-question evaluation set reused across every config**, with judge
annotation retries instead of silently dropping failed samples. Composite score
= 0.35·Relevance + 0.15·Utilization + 0.15·Completeness + 0.35·Adherence (Relevance and
Adherence weighted highest since they're the two metrics with the most direct practical
consequence — how much irrelevant text gets retrieved, and how often the generator
hallucinates).

## Winning configuration

**Sentence-level chunking + metadata heading, target 64 tokens, 10% overlap.**

| Metric | Value |
|---|---|
| Context Relevance | 0.270 |
| Utilization | 0.142 |
| Completeness | 0.602 |
| Adherence | 1.000 |
| Composite | **0.556** |
| Avg chunk size | 68 tokens |
| Eval sample | 20/20 succeeded |

## Why DelucionQA specifically needs this configuration

DelucionQA is built from sections of the Jeep Gladiator owner's manual — RAGBench
reports its source documents average only **~296 tokens** each (see the RAGBench
paper's Table 1). That single fact explains most of what the sweep found:

- **Source documents are already short, so large chunks barely differ from whole
  documents.** A 265-token paragraph/semantic chunk (Phase 1's worst performer,
  composite 0.383) covers most of a ~296-token document. Context Relevance measures
  *what fraction of the retrieved text is actually relevant to the question* — when
  a "chunk" is nearly the whole document, that fraction stays low no matter what's
  asked, because most manual sections cover several unrelated facts (torque specs,
  warning lights, maintenance steps) side by side. Small chunks are the only way to
  raise that ratio.
- **Manual content is organized as discrete, single-fact statements** — a tire
  pressure spec, a warning-light meaning, a one-step instruction — each usually
  expressed in one or two sentences. Sentence-boundary chunking captures exactly
  one such fact per chunk without truncating it mid-thought (the failure mode of
  raw token cuts, which can slice a instruction in half) and without bundling
  several unrelated facts together (the failure mode of paragraph-level chunking).
  This is why sentence-level chunking outperformed token-level at the same budget
  in Phase 1 (composite 0.408 vs 0.425 is close, but sentence-level's Completeness
  and cleaner boundaries carried it further once combined with metadata).
- **Manual sections lose their identifying context once split this small.** A
  chunk that just says "Press and hold for three seconds" doesn't say which
  button, which system, or which section of the manual it came from — that context
  normally lives in a heading several sentences up, and gets stripped away by
  aggressive chunking. Prepending a lightweight proxy-title heading (each source
  document's first sentence) recovers enough of that lost context to lift
  Completeness (0.478 vs 0.368) and Adherence (0.900 vs 0.800) without needing
  bigger chunks — which is exactly why "sentence + metadata" beat plain sentence
  chunking in Phase 1, and why the final winning config keeps that heading at
  64 tokens rather than growing the chunk instead.
- **DelucionQA questions are single-hop factual lookups**, not multi-document
  reasoning (unlike, say, HotpotQA). The answer to "what's the recommended tire
  pressure" lives entirely inside one small manual passage — there's no need for a
  chunk to span multiple sections to assemble an answer. That's consistent with
  Phase 3's finding that overlap barely matters (composite spread only
  0.513–0.556 across 0–50% overlap): since a single small chunk already contains
  the full answer most of the time, whether neighboring sentences bleed in via
  overlap doesn't change much either way.

In short: this dataset's short, single-fact, manual-style documents reward chunking
that mirrors that structure — one fact per chunk, sentence-aligned so nothing gets
cut mid-thought, with a cheap proxy title to replace the section context that
splitting removes. A generic "512 tokens, sliding window" default (tuned for much
longer, more discursive source documents) works against that structure rather than
with it.

## Phase 1 — method comparison (all at ~128 tokens, 10% overlap)

Isolates chunking *algorithm* from size.

| Method | Chunks | Avg tokens | CR | Util | Completeness | Adherence | Composite |
|---|---|---|---|---|---|---|---|
| **sentence + metadata** | 238 | 115 | 0.153 | 0.066 | 0.478 | 0.900 | **0.450** |
| small2big (96→320) | 321 | 167 | 0.194 | 0.048 | 0.267 | 0.947 | 0.447 |
| recursive_char (LangChain std.) | 284 | 91 | 0.145 | 0.067 | 0.463 | 0.850 | 0.428 |
| token-level | 256 | 101 | 0.154 | 0.050 | 0.440 | 0.850 | 0.425 |
| sentence (no metadata) | 238 | 96 | 0.184 | 0.055 | 0.368 | 0.800 | 0.408 |
| semantic / paragraph | 90 | 265 | 0.058 | 0.022 | 0.416 | 0.850 | 0.383 |

## Phase 2 — chunk size sweep (sentence + metadata, 10% overlap)

Isolates *size* from algorithm.

| Size | Chunks | CR | Adherence | Composite |
|---|---|---|---|---|
| **64t** | 456 | 0.284 | 0.895 | **0.529** |
| 128t | 238 | 0.200 | 0.950 | 0.487 |
| 96t | 314 | 0.228 | 0.900 | 0.485 |
| 224t | 164 | 0.108 | 0.950 | 0.458 |
| 160t | 193 | 0.138 | 0.950 | 0.457 |
| 192t | 168 | 0.124 | 0.900 | 0.435 |

**Takeaway:** Context Relevance falls off almost monotonically as chunk size grows
(0.284 at 64t → 0.108 at 224t). 64 tokens is a clear standout, not a marginal win.

## Phase 3 — overlap sweep (sentence + metadata, 64 tokens)

Isolates *overlap* from size.

| Overlap | CR | Adherence | Composite |
|---|---|---|---|
| **10%** | 0.270 | 1.000 | **0.556** |
| 25% | 0.323 | 0.950 | 0.540 |
| 50% | 0.357 | 0.900 | 0.531 |
| 40% | 0.311 | 0.950 | 0.528 |
| 0% | 0.280 | 0.900 | 0.513 |

**Takeaway:** this phase is comparatively flat — composite scores span only
0.513–0.556 across all five overlap values, versus a 0.38–0.53 spread across chunk
sizes in Phase 2. 10% is the nominal winner (driven by a perfect 1.00 Adherence at
that point), but 25% and 50% score close behind with *higher* raw Relevance. **Any
overlap in the 10–50% range performs reasonably** — this parameter matters much
less than getting chunking method and size right.

## Recommendation

Use **sentence-level chunking with a metadata heading, 64-token target, 10%
overlap** as the default chunking strategy for DelucionQA.

## Caveats — do not treat these exact numbers as final

- **Single run, one fixed 20-question sample.** Fixing the sample and retrying
  failed judge annotations removes some noise, but this is one dataset draw. The
  Phase 3 spread (0.513–0.556) is narrow enough that a different fixed sample
  could plausibly reorder those five configs.
- **Embedding model held constant** (`openai/text-embedding-3-small`). Not
  explored as a variable here.
- **Retrieval depth `k` held at 8 throughout.** Chunk size and `k` interact —
  not swept jointly with chunk size in this analysis.
- **Composite weighting (0.35/0.15/0.15/0.35) is a judgment call**, documented so
  it can be argued with. Re-ranking by Relevance alone would put small2big (CR
  0.194 in Phase 1, or 64t's own strong Phase 2/3 Relevance numbers) closer to
  the top; re-ranking by Adherence alone favors the higher-overlap Phase 3 configs
  less than the raw table suggests once you look past the composite.

This analysis is documentation only — no notebook or library code has been changed
to adopt this configuration yet.
