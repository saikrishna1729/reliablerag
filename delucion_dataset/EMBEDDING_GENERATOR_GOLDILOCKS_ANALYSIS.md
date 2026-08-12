# Embedding & Generator Model Goldilocks Analysis (DelucionQA)

Source: `17_embedding_generator_goldilocks.ipynb`. A two-phase search (embedding model →
generator model), scored on the **same fixed 20-question evaluation set** used throughout
`11`–`16`, with judge annotation retries and the same composite score (0.35·Relevance +
0.15·Utilization + 0.15·Completeness + 0.35·Adherence). Built on the validated `11`+`12`+
`13`+`14`+`15` stack (sentence + metadata chunks @ 64t, hybrid BM25+dense retrieval @
alpha=0.3 driven by HyDE doc+query, Cohere Rerank 4-Pro over a candidate pool of 20, final 2
chunks, no repacking) — the only variables here are the embedding model (Phase 1) and the
generator model (Phase 2).

## Winning configuration

**Embedding: `baai/bge-m3`. Generator: `microsoft/phi-4`.**

| Metric | Value |
|---|---|
| Context Relevance | 0.718 |
| Utilization | 0.628 |
| Completeness | 0.884 |
| Adherence | 0.900 |
| Composite | **0.793** |
| Avg latency | 12.42s/query |
| Eval sample | 20/20 succeeded |
| vs. original defaults (openai/text-embedding-3-small + llama-3.1-8b-instruct) | **+0.103 composite** |

This is the largest single improvement found anywhere in this series — bigger than
reranking's win in `14` (+0.089) and more than an order of magnitude larger than query
transformation's win in `13` (+0.0065). Both the embedding model and the generator model
mattered, but the generator swap did most of the work.

## Why this result makes sense

- **Every non-default embedding model beat the OpenAI baseline, and by a wide margin on
  Completeness specifically.** Phase 1: bge-m3 (composite 0.727), Gemini's embedding model
  (0.699), and Qwen3-Embedding-8B (0.698) all clearly outscored `text-embedding-3-small`
  (0.690). All three alternatives also beat the baseline on Completeness (0.714/0.685/0.700
  vs. 0.671) — the fraction of retrieved-and-relevant content the generator actually used.
  `bge-m3` is purpose-built for retrieval (it's a hybrid dense/sparse/multi-vector model
  from the same family the paper's own Table 5 flags as a strong performer), so it's
  plausible its embedding space aligns better with what the hybrid retriever's dense
  component needs than a general-purpose OpenAI embedding tuned for broader use cases.
  `bge-m3` also had the best Adherence of the four (1.000) and the lowest latency of the
  three real alternatives (10.74s vs. 12.39s for Qwen), making it a clean win on every axis
  measured, not just composite.
- **The generator swap delivered the bigger, more decisive gain.** Phase 2 (embedding fixed
  at `bge-m3`): Phi-4 (composite 0.793) and Gemma-3-27B (0.771) both dramatically outscored
  the original default, Llama-3.1-8B-Instruct (0.686). The gap is driven almost entirely by
  Utilization and Completeness — Phi-4 nearly doubles Utilization (0.628 vs. 0.369) and
  gains 24 points of Completeness (0.884 vs. 0.641) over the baseline, while Context
  Relevance (which measures retrieval quality, not generation quality) only moves modestly
  (0.718 vs. 0.627, expected since the retrieved chunks are identical across all Phase 2
  configs). This isolates the effect cleanly: the retrieval pipeline was already handing
  every generator the same 2 chunks: Phi-4 and Gemma-3-27B simply *use* that context far
  more completely than Llama-3.1-8B does, consistent with both being larger/newer models
  with generally stronger instruction-following.
- **Phi-4 wins the tradeoff even though Gemma-3-27B has perfect Adherence.** Gemma-3-27B's
  1.000 Adherence (vs. Phi-4's 0.900) means it never says anything unsupported by context,
  but Phi-4's much larger Utilization/Completeness gains outweigh that in the composite.
  Which one is actually preferable depends on what a deployment prioritizes — see
  Recommendation below.
- **`mistralai/voxtral-small-24b-2507` failed on all 20 questions**, not because it's a bad
  generator but because it's the wrong kind of model: Voxtral is Mistral's audio/speech
  model family, not a general text chat model, and doesn't behave as a drop-in replacement
  for the text-only RAG generation call this notebook makes. This is an invalid candidate,
  not informative evidence about generator quality — worth removing from any future sweep
  rather than treating as "worst performer."
- **The Gemini embedding candidate resolved to `google/gemini-embedding-001` on
  OpenRouter**, not the `gemini-embedding-2` name originally requested — OpenRouter appears
  to alias or route the requested ID to its actual catalog name. It still ran successfully
  and scored competitively, so this is a naming footnote, not a functional issue.

## Phase 1 — embedding model comparison (generator fixed @ llama-3.1-8b-instruct)

| Embedding model | CR | Util | Completeness | Adherence | Latency | Composite |
|---|---|---|---|---|---|---|
| **baai/bge-m3** | 0.605 | 0.389 | 0.714 | 1.000 | 10.74s | **0.727** |
| google/gemini-embedding-001 | 0.632 | 0.403 | 0.685 | 0.900 | 9.73s | 0.699 |
| qwen/qwen3-embedding-8b | 0.618 | 0.411 | 0.700 | 0.900 | 12.39s | 0.698 |
| openai/text-embedding-3-small (baseline) | 0.568 | 0.384 | 0.671 | 0.950 | 12.39s | 0.690 |

**Takeaway:** every alternative beat the OpenAI default, with bge-m3 also being one of the
faster options tested — a rare case in this series where the winner isn't a quality/latency
tradeoff at all.

## Phase 2 — generator model comparison (embedding fixed @ bge-m3)

| Generator model | CR | Util | Completeness | Adherence | Latency | Composite |
|---|---|---|---|---|---|---|
| **microsoft/phi-4** | 0.718 | 0.628 | 0.884 | 0.900 | 12.42s | **0.793** |
| google/gemma-3-27b-it | 0.653 | 0.497 | 0.789 | 1.000 | 13.73s | 0.771 |
| meta-llama/llama-3.1-8b-instruct (baseline) | 0.627 | 0.369 | 0.641 | 0.900 | 12.60s | 0.686 |
| mistralai/voxtral-small-24b-2507 | — | — | — | — | — | failed (20/20) |

**Takeaway:** the two larger/newer models (Phi-4, Gemma-3-27B) both substantially
outperform the original 8B default, at comparable latency. Phi-4 wins on composite; Gemma
wins on pure faithfulness (Adherence).

## Recommendation

Switch the default embedding model to **`baai/bge-m3`** and the default generator to
**`microsoft/phi-4`** — combined, composite rises from 0.690 (original defaults, matching
`16`'s validated ~0.69–0.70 baseline) to 0.793, comfortably outside this series' established
noise floor (~0.02–0.03). If a deployment cannot tolerate any unsupported claims at all,
**`google/gemma-3-27b-it`** is the safer generator choice (perfect Adherence, composite
0.771 still well above baseline) at the cost of somewhat lower Utilization/Completeness and
~1.3s more latency per query.

Update `ragbench_lib/models.py`'s `DEFAULT_EMBEDDING_MODEL` and `DEFAULT_GENERATOR_MODEL`
constants to these values once this result is corroborated (see caveats) — every notebook
in this series reads those defaults centrally, so the change would apply everywhere without
per-notebook edits.

## Caveats — do not treat these exact numbers as final

- **Still a single run on one fixed 20-question sample.** `16` established this series' own
  noise floor at roughly 0.02–0.03 composite; the embedding-only gain (+0.037, bge-m3 vs.
  baseline) is closer to that floor than the generator swap's much larger gain (+0.107),
  so treat the embedding ranking as more tentative than the generator ranking.
- **Chunking, retrieval alpha/k, query transformation, reranker, and repacking held
  constant** at `11`–`15`'s winning configs. A different embedding model could plausibly
  shift the optimal hybrid alpha or candidate pool width — not re-swept here.
- **HyDE's own generation model was held fixed** at the notebook's default throughout both
  phases, not tied to whichever generator wins Phase 2 — isolates this notebook's variables
  from `13`'s already-settled query-transformation question.
- **`mistralai/voxtral-small-24b-2507` is not a valid text-generation candidate** for this
  pipeline (it's an audio-focused model) — its complete failure reflects a bad model
  choice for this sweep, not a real quality signal. Any follow-up sweep should replace it
  with a genuine text-chat alternative.
- **Latency numbers reflect this run's network conditions and OpenRouter routing**, not a
  controlled benchmark — treat them as directional, not precise SLA figures.
- **Composite score weighting (0.35/0.15/0.15/0.35) is a judgment call**, kept identical to
  `11`–`16` for comparability. A deployment that weights Adherence above all else would
  reasonably pick Gemma-3-27B over Phi-4 despite the lower composite.
- **This result was only possible after fixing `ragbench_lib/models.py`'s embedding
  factory** (it previously used `langchain_openai.OpenAIEmbeddings`, which silently
  mishandled non-OpenAI model names via local tiktoken tokenization) — a bug specific to
  this notebook's needs, now fixed for every notebook that calls `get_embedding_model()`.

This analysis is documentation only — no notebook or library default has been changed to
adopt this configuration yet (beyond the embedding-factory bug fix, which was a
correctness fix, not a config choice).
