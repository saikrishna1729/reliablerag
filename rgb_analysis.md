# RGB Evaluation — Analysis Log

Results from `notebooks/06_rgb_evaluation.ipynb`. `N_SAMPLES=10` smoke test.

**Config used:** models = `openai/gpt-oss-120b`, `meta-llama/llama-3.3-70b-instruct`,
`qwen/qwen-2.5-72b-instruct`, all via OpenRouter (OpenAI-compatible endpoint), `temperature=0`,
`docs_per_question=5` (default), `timeout=60`, `max_retries=2` (client-level).

---

## Noise Robustness (accuracy %, `en_refine.json`)

| Model | noise=0.0 | noise=0.2 | noise=0.4 | noise=0.6 | noise=0.8 |
|---|---|---|---|---|---|
| openai/gpt-oss-120b | 80.0 | 80.0 | 70.0 | 60.0 | 80.0 |
| meta-llama/llama-3.3-70b-instruct | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| qwen/qwen-2.5-72b-instruct | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

---

## Negative Rejection (Rej %, `en_refine.json` all-noise)

| Model | Rej % |
|---|---|
| openai/gpt-oss-120b | 70.0 |
| meta-llama/llama-3.3-70b-instruct | 80.0 |
| qwen/qwen-2.5-72b-instruct | 80.0 |

---

## Information Integration (accuracy %, `en_int.json`)

| Model | noise=0.0 | noise=0.2 | noise=0.4 |
|---|---|---|---|
| openai/gpt-oss-120b | 30.0 | 20.0 | 10.0 |
| meta-llama/llama-3.3-70b-instruct | 100.0 | 100.0 | 70.0 |
| qwen/qwen-2.5-72b-instruct | 100.0 | 100.0 | 80.0 |

---

## Counterfactual Robustness (`en_fact.json`)

| Model | ACC | ACC_doc | ED | CR |
|---|---|---|---|---|
| openai/gpt-oss-120b | 0.0 | 20.0 | 40.0 | 50.0 |
| meta-llama/llama-3.3-70b-instruct | 30.0 | 40.0 | 60.0 | 66.67 |
| qwen/qwen-2.5-72b-instruct | 0.0 | 60.0 | 70.0 | 85.71 |
