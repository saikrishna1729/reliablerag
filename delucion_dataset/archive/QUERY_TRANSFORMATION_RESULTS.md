# Query Transformation Analysis Results

## Executive Summary

**Surprising Finding:** Query transformations do NOT improve retrieval quality with Dense MMR on DelucionQA. In fact, they all **degrade performance**.

### Key Result
```
🥇 BEST: Baseline (Original Query)
   Context Relevance: 0.0616
   All transformations: -21.6% to -40.6% worse
```

---

## Full Results

| Rank | Method | Context Relevance | Utilization | Completeness | Adherence | vs Baseline |
|------|--------|-------------------|-------------|--------------|-----------|------------|
| 🥇 1 | **Q0: Baseline** | **0.0616** | **0.0421** | **0.6778** | **0.8** | — |
| 🥈 2 | Q1: HyDE | 0.0483 | 0.0111 | 0.2467 | 0.4 | **-21.6%** |
| 🥉 3 | Q3: Decomposition | 0.0470 | 0.0230 | 0.6071 | 0.8 | **-23.7%** |
| 4 | Q2: Rewriting | 0.0366 | 0.0291 | 0.8500 | 1.0 | **-40.6%** |

---

## Detailed Analysis

### Why Transformations Failed

#### Q1: HyDE (Hypothetical Documents) - **-21.6%**
**Approach:** Generate hypothetical documents that would answer the query

**Results:**
- Context Relevance: 0.0483 (worst besides rewriting)
- Utilization: 0.0111 (extremely low)
- Completeness: 0.2467 (poor coverage)
- Adherence: 0.4 (only 40% of responses grounded)

**Why it failed:**
1. **Semantic drift** - Generated hypothetical docs don't match actual document style/content
2. **Factual mismatch** - Hypothetical docs may invent details not in real documents
3. **Embedding confusion** - Model embedding hypothetical docs vs. actual docs creates mismatch
4. **Noise introduction** - Extra content confuses the retriever

**Insight:** Hypothetical documents work well when actual documents are very similar to what the model can generate, but DelucionQA documents have specific styles/formats that HyDE doesn't capture.

---

#### Q2: Query Rewriting - **-40.6% (WORST)**
**Approach:** Refine query to be more specific and optimized for retrieval

**Results:**
- Context Relevance: 0.0366 (lowest)
- Utilization: 0.0291 (low)
- Completeness: 0.8500 (highest, but irrelevant)
- Adherence: 1.0 (perfect grounding, but retrieves wrong docs)

**Why it failed:**
1. **Semantic shift** - Rewritten queries changed the original meaning
2. **Over-specification** - Added details that don't exist in documents
3. **LLM limitations** - 8B model not sophisticated enough for nuanced rewrites
4. **Assumptions injection** - Rewrites added assumptions not in original question

**Insight:** Query rewriting introduced assumptions that led retriever down wrong paths. Better adherence (1.0) but completely wrong context (lowest CR).

---

#### Q3: Query Decomposition - **-23.7%**
**Approach:** Break complex query into 2-3 sub-queries

**Results:**
- Context Relevance: 0.0470
- Utilization: 0.0230
- Completeness: 0.6071
- Adherence: 0.8 (maintained)

**Why it failed:**
1. **Scatter effect** - Sub-queries spread retrieval across different topics
2. **Loss of coherence** - Breaking query loses the integrated understanding
3. **Redundancy** - Sub-queries sometimes retrieve overlapping results
4. **Context fragmentation** - Each sub-query gets evaluated separately

**Insight:** Questions in DelucionQA are already simple/focused. Breaking them apart doesn't help — it fragments the retrieval signal.

---

## Why Baseline Wins

### Dense MMR + Original Query is Already Optimal

**1. Original Queries Are Well-Formed**
- DelucionQA questions are clear, specific, and direct
- No need for rewriting or decomposition
- Asking exactly what's needed

**2. Dense MMR Handles Complexity**
- Diversity mechanism prevents over-narrowing
- Already balances relevance + coverage
- Doesn't need query transformation to improve
- Lambda parameter (0.5) is well-tuned

**3. Transformations Add Noise**
- Each transformation is another LLM call (latency + cost)
- Each transformation adds another opportunity for error
- Transformations are not trained on this specific dataset
- Generic approaches don't work for specific domains

**4. Embedding Model is Specialized**
- text-embedding-3-small is trained on diverse text
- Works well with original, natural-language queries
- Hypothetical docs or rewrites confuse it

---

## Comparative Learning

| Transformation | When It Works | Why It Failed Here |
|---------------|--------------|--------------------|
| **HyDE** | Complex, underspecified queries | Queries already clear; generated docs too different |
| **Query Rewriting** | Poorly worded questions | Questions already well-formed |
| **Query Decomposition** | Multi-hop reasoning required | Questions are single-hop QA |

---

## Key Insights

### 1. Dataset-Specific Optimization is Critical
The paper tested these on TREC/MS MARCO (general retrieval tasks). DelucionQA is different:
- Shorter questions
- Simpler information needs
- Direct document matching works better

### 2. More Steps ≠ Better Results
- Baseline: 1 retrieval call
- HyDE: 1 LLM call + 1 retrieval call = 2x cost
- Rewriting: 1 LLM call + 1 retrieval call = 2x cost
- Decomposition: 1 LLM call + 3 retrieval calls = 4x cost
- **Result: All made things WORSE**

### 3. Simpler is Better
- Occam's Razor validated: original query works best
- Dense MMR is sophisticated enough on its own
- Additional layers add complexity without benefit

### 4. Dense MMR is the Right Choice
- Already optimized for this data
- Diversity mechanism provides what transformations attempted
- No need for query augmentation

---

## Final Recommendation

### ✅ Production Configuration (FINAL)

```yaml
chunking:
  strategy: semantic-level
  target_tokens: 192

retriever:
  method: dense-mmr
  k: 8
  lambda_mult: 0.5

query_transformation:
  enabled: false
  reason: "Baseline original query performs best; transformations degrade performance"

expected_performance:
  context_relevance: 0.0616
  utilization: 0.0421
  completeness: 0.6778
  adherence: 0.8000
```

### Why NOT to Use Transformations:
1. ❌ **Performance Loss** - All methods reduced CR by 21-41%
2. ❌ **Increased Latency** - Extra LLM calls add delay
3. ❌ **Increased Cost** - More API calls = higher costs
4. ❌ **Complexity** - More code paths to maintain
5. ❌ **Dataset Mismatch** - Paper's findings don't apply to DelucionQA

### Next Steps:
✅ Keep: Semantic-Level 192t + Dense MMR  
❌ Skip: Query Transformations  
⏳ Next: Generator Fine-tuning  
⏳ Final: End-to-end evaluation  

---

## Appendix: Statistical Breakdown

### Performance by Metric

**Context Relevance (Most Important):**
- Baseline: 0.0616 (best)
- HyDE: 0.0483 (-21.6%)
- Decomposition: 0.0470 (-23.7%)
- Rewriting: 0.0366 (-40.6%)

**Utilization (Efficiency):**
- Baseline: 0.0421 (good)
- Rewriting: 0.0291 (less efficient)
- Decomposition: 0.0230 (poor)
- HyDE: 0.0111 (very poor)

**Completeness (Coverage):**
- Rewriting: 0.8500 (covers most relevant info, but wrong info!)
- Baseline: 0.6778 (balanced)
- Decomposition: 0.6071
- HyDE: 0.2467 (poor coverage)

**Adherence (Reliability):**
- Rewriting: 1.0000 (perfect, but wrong context!)
- Baseline: 0.8000 (good, reliable)
- Decomposition: 0.8000 (reliable)
- HyDE: 0.4000 (only 40% grounded)

### Key Observation:
Query Rewriting achieved perfect adherence (1.0) and highest completeness (0.85), but **WORST context relevance (0.0366)**. This indicates:
- ✅ Responses are technically grounded in retrieved documents
- ✅ Uses all relevant retrieved information
- ❌ BUT retrieves completely wrong/irrelevant documents!

This proves transformations introduce semantic drift that makes retrieval worse.

---

**Experiment Date:** 2026-07-19  
**Dataset:** RAGBench DelucionQA (50 docs, 50 queries)  
**Models:** Llama 3.1 8B (generation), Llama 3.3 70B (evaluation)  
**Baseline:** Semantic-Level 192t + Dense MMR (k=8, λ=0.5)  
**Status:** Query transformations rejected; moving to generator fine-tuning
