# Chunking Goldilocks Analysis (TechQA)

Source: `11_chunking_goldilocks.ipynb`. Same three-phase search (method → size → overlap) as
`delucion_dataset`'s chunking goldilocks notebook, scored on a **fixed 20-question
evaluation set reused across every config**, with judge annotation retries instead of
silently dropping failed samples. Composite score = 0.35·Relevance + 0.15·Utilization +
0.15·Completeness + 0.35·Adherence, identical weighting to the DelucionQA series for
direct comparability across datasets.

## Winning configuration

**Recursive-character chunking (LangChain's `RecursiveCharacterTextSplitter`), 128-token
nominal target (~90 actual tokens/chunk), 10% overlap.**

| Metric | Value |
|---|---|
| Context Relevance | 0.265 |
| Utilization | 0.157 |
| Completeness | 0.497 |
| Adherence | 0.600 |
| Composite | **0.401** |
| Avg chunk size | 90 tokens (1649 chunks) |
| Eval sample | 20/20 succeeded |

## The headline finding: TechQA's winner is DelucionQA's worst performer

This is the result the whole point of doing a fresh goldilocks search per dataset was meant
to surface. DelucionQA's chunking winner was **sentence-level + metadata heading**
(composite 0.556 there). On TechQA, `sentence_metadata` finishes **dead last** of six
methods tested in Phase 1 (composite 0.244) — worse than every alternative, including plain
token-level splitting. Meanwhile `recursive_char` — which never placed better than 4th of 6
on DelucionQA's own Phase 1 (composite 0.428, per that notebook's results) — wins outright
here (0.382 in Phase 1, rising to 0.401 once size and overlap are tuned). **No chunking
method transfers as a default across datasets; the winner is a property of the source
documents, not a property of chunking theory in the abstract.**

## Why TechQA specifically favors character-based splitting over sentence-aware splitting

- **TechQA's source documents are tech-support forum threads, not manual prose.** Forum
  posts and replies routinely contain quoted fragments from earlier messages, inline code or
  log snippets, URLs, list-formatted troubleshooting steps, and abbreviated
  not-quite-grammatical sentences — text that doesn't segment cleanly into the well-formed
  sentences `sent_tokenize` (NLTK's Punkt tokenizer, what `sentence_level_chunks` and
  `metadata_enhanced_chunks` both depend on) was trained to detect. When sentence boundaries
  are unreliable, sentence-aware chunking either produces malformed chunks or silently
  degrades toward chunking on whatever punctuation happens to be present — losing the
  advantage that made it DelucionQA's best-boundary-respecting option.
- **`RecursiveCharacterTextSplitter` doesn't need clean sentence boundaries to work well.**
  It recursively tries paragraph breaks, then newlines, then sentence-ish punctuation, then
  raw character counts — falling back gracefully rather than failing outright when a forum
  post's structure doesn't match natural prose. That robustness to messy source formatting
  is exactly the property that matters for TechQA and doesn't matter (or even matters
  negatively, given DelucionQA's clean manual prose) for DelucionQA.
- **The metadata-heading addition hurt here instead of helping.** DelucionQA's chunks
  benefited from a proxy-title heading recovering context stripped away by small chunking
  (`11`'s DelucionQA analysis: Completeness 0.478 vs 0.368 with vs. without metadata). On
  TechQA, the same mechanism (prepending each source document's first sentence) finishes
  last — plausibly because a forum thread's first sentence is often a greeting, a restated
  problem summary, or otherwise low-information text (unlike a manual section's descriptive
  opening line), so the "cheap proxy title" trick's core assumption doesn't hold for this
  document type.
- **The overall composite ceiling is much lower than DelucionQA's** (0.401 here vs. 0.556
  there) — driven mainly by Adherence, which tops out at 0.60–0.65 across every config
  tested here vs. 0.80–1.00 typically seen on DelucionQA. This is likely a base-rate
  property of the dataset, not a chunking artifact: TechQA answers often require piecing
  together troubleshooting steps discussed across a multi-turn thread, which is inherently
  harder to fully ground a single response in than DelucionQA's single-hop factual lookups
  against a static manual. Worth watching whether retrieval/reranking phases can recover any
  of this gap, or whether it's a structural ceiling for this dataset.
- **Overlap prefers a small, non-zero value (10%)** — the same qualitative shape found in
  DelucionQA's own overlap phase (10% won there too), even though the winning method and
  size are completely different. Composite falls off sharply past 25% overlap (0.401 at 10%
  → 0.301 at 25% → 0.322–0.324 at 40–50%, a non-monotonic wobble suggesting overlap's effect
  is noisier/weaker here than method or size, consistent with `11`'s DelucionQA analysis
  flagging overlap as the least impactful of the three axes there too).

## Phase 1 — method comparison (all at ~128 tokens, 10% overlap)

Isolates chunking *algorithm* from size.

| Method | Chunks | Avg tokens | CR | Util | Completeness | Adherence | Composite |
|---|---|---|---|---|---|---|---|
| **recursive_char** | 1649 | 90 | 0.259 | 0.101 | 0.328 | 0.65 | **0.382** |
| token-level | 1342 | 120 | 0.183 | 0.089 | 0.441 | 0.50 | 0.319 |
| semantic / paragraph | 1150 | 126 | 0.186 | 0.067 | 0.380 | 0.50 | 0.307 |
| sentence (no metadata) | 1078 | 135 | 0.197 | 0.082 | 0.339 | 0.40 | 0.272 |
| small2big (96→320) | 1764 | 186 | 0.084 | 0.062 | 0.363 | 0.45 | 0.251 |
| sentence + metadata | 1078 | 163 | 0.185 | 0.076 | 0.305 | 0.35 | 0.244 |

**Takeaway:** recursive-character splitting wins by a clear margin (0.382 vs. 0.319 runner-up)
— and the method DelucionQA rated highest (sentence + metadata) finishes last here.

## Phase 2 — chunk size sweep (recursive_char, 10% overlap)

Isolates *size* from algorithm. (Actual avg tokens/chunk don't scale linearly with the
nominal target — `RecursiveCharacterTextSplitter`'s char-based `chunk_size` heuristic
interacts with TechQA's mix of prose, code, and short lines differently than DelucionQA's
uniform manual prose.)

| Target | Chunks | Avg actual tokens | CR | Adherence | Composite |
|---|---|---|---|---|---|
| **128t** | 1649 | 90 | 0.218 | 0.65 | **0.377** |
| 96t | 2247 | 66 | 0.250 | 0.50 | 0.340 |
| 192t | 1061 | 140 | 0.151 | 0.55 | 0.323 |
| 160t | 1294 | 115 | 0.201 | 0.50 | 0.317 |
| 64t | 3403 | 43 | 0.263 | 0.35 | 0.275 |
| 224t | 918 | 162 | 0.116 | 0.45 | 0.259 |

**Takeaway:** unlike DelucionQA (where smaller was consistently better, 64t won outright),
TechQA's composite peaks in the *middle* of the sweep (128t) — both smaller (64t: fragments
troubleshooting steps mid-thought, Adherence drops to 0.35) and larger (192–224t: dilutes
relevance, same pattern DelucionQA showed at its own large end) hurt here. This is a
genuinely different shape, not just a different optimal point.

## Phase 3 — overlap sweep (recursive_char, 128t)

Isolates *overlap* from size.

| Overlap | CR | Adherence | Composite |
|---|---|---|---|
| **10%** | 0.265 | 0.60 | **0.401** |
| 0% | 0.219 | 0.65 | 0.389 |
| 50% | 0.197 | 0.50 | 0.324 |
| 40% | 0.268 | 0.45 | 0.322 |
| 25% | 0.227 | 0.50 | 0.301 |

**Takeaway:** 10% is a modest but real winner over no overlap (0.401 vs. 0.389); anything
past 25% clearly hurts. Same qualitative shape as DelucionQA's overlap phase, despite every
other axis differing.

## Recommendation

Use **recursive-character chunking, 128-token nominal target (~90 actual tokens/chunk), 10%
overlap** as the default chunking strategy for TechQA. This feeds
`12_retriever_goldilocks.ipynb` as TechQA's frozen chunking foundation, exactly as
DelucionQA's winner fed its own `12`.

## Caveats — do not treat these exact numbers as final

- **Single run, one fixed 20-question sample.** Same caveat as every notebook in this
  series — this is one dataset draw, not a statistically robust estimate.
- **Embedding model held constant** (`openai/text-embedding-3-small`, `ragbench_lib`
  default). Not explored as a variable here — `17`'s embedding/generator sweep hasn't been
  run for TechQA yet.
- **Retrieval depth `k` held at 8 throughout**, matching the DelucionQA series' own Phase 1
  convention — not swept jointly with chunk size here.
- **The nominal-vs-actual token gap for `recursive_char`** (128t target → 90 actual tokens)
  reflects the same char-to-token heuristic (`target_tokens * 4` chars) DelucionQA's own
  notebook used; it's not recalibrated per dataset, so TechQA's different text composition
  (code/log snippets, URLs) changes the actual chunk sizes produced at a given nominal
  target differently than it would for DelucionQA.
- **Composite weighting (0.35/0.15/0.15/0.35) is a judgment call**, kept identical to the
  DelucionQA series so the two datasets' results stay comparable. Re-ranking by Adherence
  alone would still favor `recursive_char @ 0% overlap` (0.65) marginally over the 10%
  winner (0.60) — the composite's Relevance weighting is what tips it to 10%.
- **The much lower Adherence ceiling here (0.35–0.65) vs. DelucionQA (0.60–1.00)** may be a
  structural property of TechQA's multi-turn, harder-to-fully-ground answers rather than
  something any chunking config can fix — worth watching in later phases (retrieval,
  reranking) rather than assumed fixable at the chunking stage alone.

This analysis is documentation only — no notebook or library code has been changed to
adopt this configuration yet.
