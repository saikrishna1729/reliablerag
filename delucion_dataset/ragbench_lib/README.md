# ragbench_lib

Shared code for the DelucionQA / RAGBench experiment notebooks (`01`-`10` in
this folder). Every notebook imports from here instead of redefining its own
copy of data loading, model setup, chunking, or TRACe evaluation.

| Module | Contents |
|---|---|
| `data_loading.py` | `load_rag_bench_data(dataset_name, split, num_samples)` — loads + flattens any RAGBench component dataset |
| `models.py` | `get_embedding_model`, `get_generation_llm`, `get_judge_llm` — OpenRouter client factories, model names read from `.env` |
| `chunking.py` | `count_tokens`, `get_sentences`, `create_semantic_chunks`, `encoding` |
| `retrievers.py` | `DenseMMRRetriever` (shared verbatim across notebooks 06-08) |
| `trace_eval.py` | The TRACe annotation prompt + `annotate_response_for_metrics` + `compute_context_relevance` / `compute_utilization` / `compute_completeness` / `compute_adherence` |
| `generation_prompt.py` | `RAG_GENERATION_PROMPT` — the shared answer-generation prompt |
| `vector_store.py` | `get_persist_dir` / `named_persist_dir` — where Chroma collections live on disk |

## Loading a different RAGBench dataset

`load_rag_bench_data` defaults to DelucionQA/test (matching every existing
notebook call), but takes `dataset_name` and `split` too:

```python
from ragbench_lib.data_loading import load_rag_bench_data, RAGBENCH_CONFIGS

docs_df = load_rag_bench_data("pubmedqa", split="validation", num_samples=100)
docs_df = load_rag_bench_data(num_samples=None)  # full DelucionQA test split
```

`RAGBENCH_CONFIGS` lists the 12 component datasets from paper Table 1
(`pubmedqa`, `covidqa`, `hotpotqa`, `msmarco`, `hagrid`, `expertqa`, `cuad`,
`delucionqa`, `emanual`, `techqa`, `finqa`, `tatqa`).

`base-notebook-ragbench-delucionqa (1).ipynb` keeps its own local
`RAG_Bench_ds_loader` (it depends on `kaggle_secrets`, which only exists on
Kaggle, so it can't reach this repo's `ragbench_lib` at runtime anyway).

## Vector stores live in one place

Every notebook's Chroma collections are persisted under
`delucion_dataset/vector_stores/` instead of the system temp directory or
ad-hoc relative paths. Two helpers in `vector_store.py`:

- `get_persist_dir(prefix)` — drop-in replacement for
  `tempfile.mkdtemp(prefix=...)`. Returns a fresh, uniquely-named directory
  under `vector_stores/` each call (same semantics as before, including being
  safe to `shutil.rmtree` afterwards) — used by the chunk-size/retriever
  sweeps that build and tear down dozens of collections per run.
- `named_persist_dir(name)` — a **stable** path, `vector_stores/<name>`,
  reused across runs. Used by `01_base_rag_v01.ipynb`'s single long-lived
  `CHROMA_PATH` (previously a bare `"chromadb/"` relative path).
- `vector_store_names(dataset, strategy)` — returns a matching
  `(persist_dir_prefix, collection_name)` pair that encodes both which
  RAGBench dataset and which experiment strategy a database belongs to, e.g.
  `vector_store_names("delucionqa", "M3")` ->
  `("chroma_delucionqa_m3_", "ragbench_delucionqa_m3")`. Every notebook sets
  a `DATASET_NAME` variable next to its `load_rag_bench_data(...)` call and
  passes it here, so `ls vector_stores/` and Chroma's own collection names
  both make it obvious which dataset+strategy each database is for instead
  of a bare `chroma_mmr_xyz123/`.

`vector_stores/` is gitignored (added to the repo root `.gitignore`) since
it's generated data, not something to commit.

`base-notebook-ragbench-delucionqa (1).ipynb` keeps its own
`/kaggle/working/chroma_db` path untouched — it's a Kaggle-environment
notebook, and that path is meaningful only there.

## Models are configured centrally via `.env`

`delucion_dataset/.env` (see `.env.example`) sets which model each factory in
`models.py` defaults to:

```sh
RAGBENCH_EMBEDDING_MODEL=openai/text-embedding-3-small
RAGBENCH_GENERATOR_MODEL=meta-llama/llama-3.1-8b-instruct
RAGBENCH_JUDGE_MODEL=meta-llama/llama-3.3-70b-instruct
```

`get_embedding_model` / `get_generation_llm` / `get_judge_llm` read these at
call time via `os.environ.get(...)`, falling back to the same hardcoded
values if the var isn't set. Precedence: **explicit `model=` argument >
env var > hardcoded default**. Notebooks that deliberately test a different
model as their independent variable still pass `model=` explicitly and are
unaffected by the env config:

- `03_chunking_optimization.ipynb` — generator pinned to `openai/gpt-oss-20b`
- `06_query_transformation_optimization.ipynb`'s `query_llm` — same base
  generator model as everyone else, just a different `temperature`/`max_tokens`

Every other notebook's `llm`/`llm_base`/judge/embedding calls now read from
`.env`, so swapping models repo-wide is a one-line edit instead of hunting
through ten notebooks.

## Notebook-local by design

Not everything got pulled in here. Retriever implementations that are each
notebook's actual independent variable (`BM25Retriever` in 05,
`OptimizedRetriever` in 10_full, `LiveRetriever` in 10_live, the rerankers in
08, the chunking-method sweeps in 02/03/04) are intentionally left in their
notebooks — centralizing those would hide the thing each experiment is
testing. Likewise, `10_full_rag_pipeline.ipynb` / `10_live_rag_pipeline.ipynb`
keep their own `load_rag_bench_data`-equivalent because their flattened
dataframe schema (column names, dropped/added fields) differs from the
version other notebooks use.

## Notebook bootstrap

Each notebook adds the folder containing `ragbench_lib/` to `sys.path` before
importing from it:

```python
import sys, os

def _add_ragbench_lib_to_path():
    for candidate in (os.getcwd(), os.path.join(os.getcwd(), "delucion_dataset")):
        if os.path.isdir(os.path.join(candidate, "ragbench_lib")) and candidate not in sys.path:
            sys.path.insert(0, candidate)
            return

_add_ragbench_lib_to_path()
```

This works whether the notebook kernel's cwd is `delucion_dataset/` (the
common case) or the repo root.

## Bug fixes made during extraction

`trace_eval.py` fixes two correctness bugs that existed in the original
per-notebook copies:

1. **Sentence-key collision.** Keys were generated with
   `chr(97 + sent_idx % 26)`, which wraps every 26 sentences and silently
   collides two different sentences onto the same key (document 0's sentence
   0 and sentence 26 both became `"0a"`). Long documents (CUAD contracts,
   TechQA notes) routinely exceed 26 sentences. Fixed with an unbounded
   spreadsheet-style suffix (`a, b, ..., z, aa, ab, ...`).
2. **Unvalidated judge keys.** The judge LLM's returned relevant/utilized
   keys were never checked against the keys that actually exist in the
   documents, so a hallucinated or malformed key could silently inflate a
   score past 1.0. `annotate_response_for_metrics` now intersects returned
   keys with the real key set (normalizing a trailing `.`) before scoring,
   and reports any dropped keys under `dropped_relevant_keys` /
   `dropped_utilized_keys` for debugging.

Re-running any notebook's TRACe cells after this change may shift previously
recorded scores for long documents — that's the bug fix taking effect, not a
regression.
