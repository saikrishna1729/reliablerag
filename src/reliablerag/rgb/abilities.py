"""The four RGB ability evaluations.

Each runner takes ``llms`` (a {name: chat model} dict) and a list of RGB records, and returns a
pandas DataFrame with models as rows — mirroring the paper's Tables 1, 3, 5, 7.

``ask()`` (see prompt.py) already retries transient API failures (rate limits, timeouts, 5xx).
``_safe_call`` below is the last line of defense: if a call still fails after those retries (or
raises for any other reason), we log it and count that record as a miss instead of crashing the
whole evaluation run.
"""

import random
import time

import pandas as pd
from langchain_core.language_models import BaseChatModel

from .metrics import detects_error, is_accurate, is_rejection
from .prompt import ask, ask_direct
from .sampling import get_sample_docs_at_noise_ratio, get_sample_int_docs_at_noise_ratio

DEFAULT_DOCS_PER_QUESTION = 5  # external docs per question (paper's setting)
DEFAULT_SAMPLING_SEED = 42
NOISE_RATIOS = (0.0, 0.2, 0.4, 0.6, 0.8)  # noise robustness
INT_RATIOS = (0.0, 0.2, 0.4)  # information integration
PROGRESS_EVERY = 25  # print a progress line every N records within a long-running batch


def _log_progress(tag: str, done: int, total: int, elapsed_seconds: float) -> None:
    print(f"{tag} ... {done}/{total} records ({elapsed_seconds:6.1f}s elapsed)")


def _safe_call(compute, context: str, default):
    """Run compute(); on failure, log context + the exception and return default instead of
    crashing the whole evaluation run. Use default=False for a miss on a boolean metric, or
    default=None when the caller needs to know the call failed (e.g. to skip dependent metrics)."""
    try:
        return compute()
    except Exception as e:
        print(f"[error] {context}: {e!r} — using default={default!r}")
        return default


def _require_records(records: list[dict], dataset_name: str) -> None:
    if not records:
        raise ValueError(f"{dataset_name} is empty — nothing to evaluate")


def run_noise_robustness(
    llms: dict[str, BaseChatModel],
    refine_records: list[dict],
    noise_ratios=NOISE_RATIOS,
    docs_per_question: int = DEFAULT_DOCS_PER_QUESTION,
    sampling_seed: int = DEFAULT_SAMPLING_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Table 1 — accuracy (%) per model across noise ratios (en_refine).

    refine_records    : records loaded from en_refine.json (query/answer/positive/negative).
    noise_ratios      : fraction of docs_per_question that are noise, swept per column.
    docs_per_question : total documents fed to the LLM as context per question.
    sampling_seed     : reset per (model, ratio) so every model sees the same sampled docs.
    """
    _require_records(refine_records, "refine_records")

    rows: dict[str, dict[str, float]] = {}
    for name, llm in llms.items():
        rows[name] = {}
        for ratio in noise_ratios:
            rng = random.Random(sampling_seed)
            correctly_answered_count = 0
            batch_start = time.perf_counter()
            for i, record in enumerate(refine_records, start=1):
                context = f"[noise] model={name} ratio={ratio} id={record.get('id')}"
                answered_correctly = _safe_call(
                    lambda: is_accurate(
                        ask(
                            llm,
                            record["query"],
                            get_sample_docs_at_noise_ratio(
                                record["positive"], record["negative"], ratio, docs_per_question, rng
                            ),
                        ),
                        record["answer"],
                    ),
                    context,
                    default=False,
                )
                correctly_answered_count += 1 if answered_correctly else 0
                if verbose and i % PROGRESS_EVERY == 0:
                    _log_progress(
                        f"[noise]  {name:<38} ratio={ratio:<4}",
                        i, len(refine_records), time.perf_counter() - batch_start,
                    )
            elapsed_seconds = time.perf_counter() - batch_start
            accuracy_pct = round(100 * correctly_answered_count / len(refine_records), 2)
            rows[name][f"noise={ratio}"] = accuracy_pct
            if verbose:
                print(f"[noise]  {name:<38} ratio={ratio:<4} acc={accuracy_pct:6.2f}%  time={elapsed_seconds:6.1f}s")
    return pd.DataFrame(rows).T


def run_negative_rejection(
    llms: dict[str, BaseChatModel],
    refine_records: list[dict],
    docs_per_question: int = DEFAULT_DOCS_PER_QUESTION,
    sampling_seed: int = DEFAULT_SAMPLING_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Table 3 — rejection rate Rej (%) per model, context = noise docs only (en_refine)."""
    _require_records(refine_records, "refine_records")

    rows: dict[str, float] = {}
    for name, llm in llms.items():
        rng = random.Random(sampling_seed)
        rejected_count = 0
        batch_start = time.perf_counter()
        for i, record in enumerate(refine_records, start=1):
            # Randomly (but reproducibly, via seeded rng) pick docs_per_question distinct noise
            # docs from record["negative"]; cap at len(negative) so sample() doesn't error if fewer exist.
            noise_docs = rng.sample(record["negative"], min(docs_per_question, len(record["negative"])))
            context = f"[reject] model={name} id={record.get('id')}"
            was_rejected = _safe_call(
                lambda: is_rejection(ask(llm, record["query"], noise_docs)),
                context,
                default=False,
            )
            rejected_count += 1 if was_rejected else 0
            if verbose and i % PROGRESS_EVERY == 0:
                _log_progress(f"[reject] {name:<38}", i, len(refine_records), time.perf_counter() - batch_start)
        elapsed_seconds = time.perf_counter() - batch_start
        rejection_pct = round(100 * rejected_count / len(refine_records), 2)
        rows[name] = rejection_pct
        if verbose:
            print(f"[reject] {name:<38} Rej={rejection_pct:6.2f}%  time={elapsed_seconds:6.1f}s")
    return pd.DataFrame({"Rej %": rows})


def run_information_integration(
    llms: dict[str, BaseChatModel],
    int_records: list[dict],
    noise_ratios=INT_RATIOS,
    docs_per_question: int = DEFAULT_DOCS_PER_QUESTION,
    sampling_seed: int = DEFAULT_SAMPLING_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Table 5 — accuracy (%) per model across noise ratios for compound questions (en_int)."""
    _require_records(int_records, "int_records")

    rows: dict[str, dict[str, float]] = {}
    for name, llm in llms.items():
        rows[name] = {}
        for ratio in noise_ratios:
            rng = random.Random(sampling_seed)
            correctly_answered_count = 0
            batch_start = time.perf_counter()
            for i, record in enumerate(int_records, start=1):
                context = f"[integ] model={name} ratio={ratio} id={record.get('id')}"
                answered_correctly = _safe_call(
                    lambda: is_accurate(
                        ask(
                            llm,
                            record["query"],
                            get_sample_int_docs_at_noise_ratio(
                                record["positive"], record["negative"], ratio, docs_per_question, rng
                            ),
                        ),
                        record["answer"],
                    ),
                    context,
                    default=False,
                )
                correctly_answered_count += 1 if answered_correctly else 0
                if verbose and i % PROGRESS_EVERY == 0:
                    _log_progress(
                        f"[integ]  {name:<38} ratio={ratio:<4}",
                        i, len(int_records), time.perf_counter() - batch_start,
                    )
            elapsed_seconds = time.perf_counter() - batch_start
            accuracy_pct = round(100 * correctly_answered_count / len(int_records), 2)
            rows[name][f"noise={ratio}"] = accuracy_pct
            if verbose:
                print(f"[integ]  {name:<38} ratio={ratio:<4} acc={accuracy_pct:6.2f}%  time={elapsed_seconds:6.1f}s")
    return pd.DataFrame(rows).T


def run_counterfactual(
    llms: dict[str, BaseChatModel],
    fact_records: list[dict],
    verbose: bool = True,
) -> pd.DataFrame:
    """Table 7 — ACC / ACC_doc / ED / CR (%) per model (en_fact).

    ACC     : accuracy with no documents at all, asked directly with no RAG framing (internal
              knowledge; paper trusts models with ACC > 70%) (correct_responses_count_without_docs).
    ACC_doc : accuracy with counterfactual (poisoned) documents (correct_responses_count_with_poisoned_docs).
    ED      : error-detection rate — response flags "factual errors ..." (wrong_data_detected_with_poisoned_docs_count).
    CR      : error-correction rate — of the ED-positive instances, fraction also stating the truth
              (wrong_data_detected_and_corrected_with_poisoned_docs_count).
    """
    _require_records(fact_records, "fact_records")

    rows: dict[str, dict[str, float]] = {}
    for name, llm in llms.items():
        fact_records_count = len(fact_records)
        correct_responses_count_without_docs = 0
        correct_responses_count_with_poisoned_docs = 0
        wrong_data_detected_with_poisoned_docs_count = 0
        wrong_data_detected_and_corrected_with_poisoned_docs_count = 0
        batch_start = time.perf_counter()
        for i, record in enumerate(fact_records, start=1):
            answer = record["answer"]  # correct answer (str)
            record_id = record.get("id")

            # ask_direct (not ask with docs=[]) -- ask()'s RAG system prompt still tells the model
            # to refuse when "the document" lacks the answer, so an empty doc list makes a
            # well-aligned model refuse regardless of what it actually knows. ask_direct asks the
            # bare question with no document-grounding framing at all, matching the paper's own
            # ACC methodology ("we assess their performance by directly asking them questions").
            knew_answer_unaided = _safe_call(
                lambda: is_accurate(ask_direct(llm, record["query"]), answer),
                f"[fact] model={name} id={record_id} (no-doc ACC)",
                default=False,
            )
            correct_responses_count_without_docs += 1 if knew_answer_unaided else 0

            # correct_responses_count_with_poisoned_docs / wrong_data_detected_with_poisoned_docs_count /
            # wrong_data_detected_and_corrected_with_poisoned_docs_count all depend on the poisoned-document
            # response, so if that call itself fails after retries, skip all three for this record rather
            # than guessing.
            resp = _safe_call(
                lambda: ask(llm, record["query"], record["positive_wrong"]),
                f"[fact] model={name} id={record_id} (poisoned-doc ACC_doc/ED/CR)",
                default=None,
            )
            if resp is not None:
                correct_responses_count_with_poisoned_docs += 1 if is_accurate(resp, answer) else 0
                if detects_error(resp):
                    wrong_data_detected_with_poisoned_docs_count += 1
                    wrong_data_detected_and_corrected_with_poisoned_docs_count += (
                        1 if is_accurate(resp, answer) else 0
                    )

            if verbose and i % PROGRESS_EVERY == 0:
                _log_progress(f"[fact]   {name:<38}", i, fact_records_count, time.perf_counter() - batch_start)

        elapsed_seconds = time.perf_counter() - batch_start
        rows[name] = {
            "ACC (correct_responses_count_without_docs)": round(
                100 * correct_responses_count_without_docs / fact_records_count, 2
            ),
            "ACC_doc (correct_responses_count_with_poisoned_docs)": round(
                100 * correct_responses_count_with_poisoned_docs / fact_records_count, 2
            ),
            "ED (wrong_data_detected_with_poisoned_docs_count)": round(
                100 * wrong_data_detected_with_poisoned_docs_count / fact_records_count, 2
            ),
            "CR (wrong_data_detected_and_corrected_with_poisoned_docs_count)": round(
                100
                * wrong_data_detected_and_corrected_with_poisoned_docs_count
                / wrong_data_detected_with_poisoned_docs_count,
                2,
            )
            if wrong_data_detected_with_poisoned_docs_count
            else 0.0,
        }
        if verbose:
            print(f"[fact]   {name:<38} {rows[name]}  time={elapsed_seconds:6.1f}s")
    return pd.DataFrame(rows).T
