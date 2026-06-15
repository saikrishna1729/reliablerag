# CUAD Retrieval Trials — RAGBench Evaluation Report

**Dataset:** galileo-ai/ragbench, split=`cuad`, `train[0:N]`  
**Judge:** TRACe (RAGBench, Friel et al.) — LLM-as-judge for Adherence; GPT-4-annotated labels for Relevance/Utilization/Completeness  
**Vector store:** ChromaDB  

---

## Metric Definitions

| Metric | What it measures |
|---|---|
| **Relevance** | Fraction of retrieved context that is relevant to the question |
| **Utilization** | Fraction of retrieved context that the LLM actually uses in its answer |
| **Completeness** | Fraction of *all* relevant content in the document that was captured (recall) |
| **Adherence** | Whether the LLM answer is fully grounded in the retrieved context (binary, LLM-judged) |

**Success criterion:** `our_completeness` within 0.05 of `ref_completeness` (0.717) while keeping `our_relevance ≥ ref_relevance` and adherence trending toward 90%.

---

## Trial Log

### Experiment A — Cosine similarity
**Full config:** embedder=`nomic-embed-text-v2-moe`, judge=`gemma4:12b-it-q4_K_M`, similarity=cosine, chunk_size=500, overlap=50, top_k=20, N=5  
**Result:** Large improvement — relevance ~0.09, completeness ~0.74 on 5-sample slice.  

---

### Experiment B — Eval stabilization
**Reason:** Scores were noisy run-to-run; needed a stable signal before iterating further.  
**Change:**  
- `evaluation.py`: length unit chars → whitespace tokens (closer to RAGBench paper).  
- Added `judge_llm` with `temperature=0`.  
- Added `n_runs` averaging; adherence by strict majority vote.  
**Full config:** embedder=`nomic-embed-text-v2-moe`, judge=`gemma4:12b-it-q4_K_M`, similarity=cosine, chunk_size=500, overlap=50, top_k=20, N=5  
**Result:** More stable scores across runs. No aggregate change to the numbers.  

---

### Experiment C — Cross-encoder reranker (BAAI/bge-reranker-base)
**Reason:** Cosine retrieval surfaces plausible-but-imprecise chunks; a cross-encoder reranker should push the most relevant ones to the top.  
**Change:** Over-fetch top-50 from Chroma, rerank with cross-encoder, keep top-N.  
**Full config:** embedder=`nomic-embed-text-v2-moe`, judge=`gemma4:12b-it-q4_K_M`, similarity=cosine, chunk_size=500, overlap=50, fetch_k=50, top_n=8 or 20, N=5  
**Result (top_n=8):** Mixed — precision improved on samples 3/4, but sample 2 collapsed (relevance 0.146→0.000, completeness 1.0→0.0). Reranker demoted context-establishing chunks below the cutoff.  
**Result (top_n=20):** Same context window as baseline, only reorders — within judge noise.  
**Verdict:** Inconclusive on N=5. Parked for later. Reranking targets precision, not recall.  

---

### Experiment D — Swap judge: Gemma4 12B → Llama 3.1 8B
**Reason:** Gemma4 12B judge was taking 3–4 min per TRACe call on Apple Silicon; iteration speed was blocking progress.  
**Change:** `JUDGE_MODEL=llama3.1:8b-instruct-q4_K_M`, `temperature=0`.  
**Full config:** embedder=`nomic-embed-text-v2-moe`, judge=`llama3.1:8b-instruct-q4_K_M`, similarity=cosine, chunk_size=500, overlap=50, top_k=20, N=5  
**5-sample numbers (post-swap):** Relevance 0.166, Completeness 0.674  
**Lesson:** Judge swaps move absolute numbers. Llama is stricter on adherence (40% vs 100%) and faster (~30s/call). Re-baseline after any judge change.  

---

### Experiment E — Scale to N=20 + fix judge parse errors (authoritative baseline)
**Reason:** N=5 was unrepresentative — first 5 CUAD samples are the easy ones; needed a real N before trusting any diagnosis. Parse errors in the Llama judge were silently zeroing out samples; fixed with corrective retry and exclusion of failed runs from the average.  
**Change:**  
- Bumped evaluation from 5 → 20 CUAD samples, `n_runs=3`.  
- `evaluation.py`: corrective retry on parse failure (shows the model its bad output, asks it to fix); failed runs excluded from average rather than counted as zeros.  
- Prompt strengthened: JSON-only instruction added at top and bottom of `_ANNOTATION_PROMPT`.  
**Full config:** embedder=`nomic-embed-text-v2-moe`, judge=`llama3.1:8b-instruct-q4_K_M`, similarity=cosine, chunk_size=500, overlap=50, top_k=20, N=20, n_runs=3  

| Metric | Ours | Ref (GPT-4) |
|---|---|---|
| Relevance | **0.173** | 0.069 |
| Utilization | **0.097** | 0.042 |
| Completeness | 0.564 | **0.717** |
| Adherence | **55%** (11/20) | 90% (18/20) |
| Parse errors | 2/20 | — |

**This is the authoritative baseline. All subsequent experiments are compared against it.**  
**Diagnosis:** Completeness gap (0.564 vs 0.717) is the remaining bottleneck. Dense retrieval misses clauses with vocabulary different from the query.  

---

### Experiment F — Chunk size sweep (500/50, 1000/150, 1500/200)
**Reason:** Completeness gap suggested small chunks were splitting legal clauses mid-sentence; larger chunks should capture full clause text.  
**Note:** All three configs run with the fixed evaluator (Exp E parse error fix applied).

#### F1 — 1500/200
**Full config:** embedder=`nomic-embed-text-v2-moe`, judge=`llama3.1:8b-instruct-q4_K_M`, similarity=cosine, chunk_size=1500, overlap=200, top_k=20, N=20, n_runs=3  

| Metric | Ours | Ref (GPT-4) | vs E baseline |
|---|---|---|---|
| Relevance | 0.071 | 0.069 | −0.102 |
| Utilization | 0.041 | 0.042 | −0.056 |
| Completeness | **0.592** | 0.717 | **+0.028** |
| Adherence | 10% (2/20) | 90% | −45pp |
| Parse errors | 2/20 | — | — |

**Verdict:** Best completeness (0.592) but adherence collapses to 10% — large chunks give the generator too much unfocused context, causing it to hedge. Net negative overall.

#### F2 — 1000/150
**Full config:** embedder=`nomic-embed-text-v2-moe`, judge=`llama3.1:8b-instruct-q4_K_M`, similarity=cosine, chunk_size=1000, overlap=150, top_k=20, N=20, n_runs=3  

| Metric | Ours | Ref (GPT-4) | vs E baseline |
|---|---|---|---|
| Relevance | 0.090 | 0.069 | −0.083 |
| Utilization | 0.043 | 0.042 | −0.054 |
| Completeness | 0.446 | 0.717 | −0.118 |
| Adherence | 35% (7/20) | 90% | −20pp |
| Parse errors | 5/20 | — | — |

**Verdict:** Worse than baseline on all metrics. Most parse errors of the three configs (5/20).

**Chunk sweep conclusion:** 500/50 wins on every metric except raw completeness (where 1500/200 edges ahead by 0.028 at the cost of 45pp adherence). Fixed-character chunking is not the lever for closing the completeness gap.

---

## Summary Table

All metrics are **averages across N=20 samples** with fixed evaluator. Ref metrics come from GPT-4 annotations in the RAGBench dataset and are fixed per sample.

| Exp | Embedder | Judge | Similarity | chunk/overlap | top_k | N | Our Rel. | Ref Rel. | Our Util. | Ref Util. | Our Comp. | Ref Comp. | Our Adh. | Ref Adh. | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E (baseline)** | nomic-embed-text-v2-moe | llama3.1:8b-q4_K_M | cosine | 500/50 | 20 | 20 | **0.173** | 0.069 | **0.097** | 0.042 | 0.564 | 0.717 | **55%** | 90% | Best overall |
| F1 | nomic-embed-text-v2-moe | llama3.1:8b-q4_K_M | cosine | 1500/200 | 20 | 20 | 0.071 | 0.069 | 0.041 | 0.042 | **0.592** | 0.717 | 10% | 90% | Best comp, adherence collapses |
| F2 | nomic-embed-text-v2-moe | llama3.1:8b-q4_K_M | cosine | 1000/150 | 20 | 20 | 0.090 | 0.069 | 0.043 | 0.042 | 0.446 | 0.717 | 35% | 90% | Worst overall |

**Current best config: E (500/50).** Proceeding to hybrid retrieval next.

---

## Open Diagnosis

- **Completeness gap (0.564 vs 0.717 ref):** Dense retrieval with a single query misses clauses that use different vocabulary than the query (e.g. "escrow" vs "source code deposit"). Chunk size does not fix this.
- **2 persistent parse errors per run:** Samples where Llama fails even after corrective retry. Marginal impact now that exclusion logic is in place.
- **Adherence 55% vs ref 90%:** Many "I do not know" answers are legitimate retrieval failures. Generator is over-hedging when relevant context is present.

---

## Next Steps (Priority Order)

1. **Hybrid retrieval (BM25 + dense)** — `EnsembleRetriever`. Exact-term matching covers vocabulary-gap queries that dense retrieval misses.
2. **Query transformation (HyDE)** — generate a hypothetical answer, embed that instead; catches paraphrase mismatches.
3. **Reranker revisited** — re-test on top of BM25+dense candidates.
4. **Stronger embedder** — `bge-large-en-v1.5` or a legal-tuned model if steps 1–3 plateau.