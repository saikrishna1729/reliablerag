"""Build each question's document context at a target noise ratio.

The caller passes a seeded ``random.Random`` so sampling is reproducible and identical across models.
"""

import math
import random


def get_sample_docs_at_noise_ratio(
    positive_documents: list[str],
    negative_documents: list[str],
    noise_ratio: float,
    docs_per_question: int,
    rng: random.Random,
) -> list[str]:
    """Assemble the documents to show the model for one question, at a chosen noise level.

    The result is a shuffled mix of documents that actually answer the question and documents
    that don't (noise) — the proportion of noise is controlled by noise_ratio. A ratio of 0 means
    a clean, fully answerable context; a ratio of 1.0 means every document is noise, which is
    exactly the "negative rejection" scenario (the model should refuse to answer).

    positive_documents : documents that contain the answer (RGB record's "positive" field).
    negative_documents : documents relevant to the question but without the answer — the "noise"
                         docs (RGB record's "negative" field).
    """
    n_neg = round(noise_ratio * docs_per_question)
    n_pos = docs_per_question - n_neg

    pos = positive_documents[:n_pos]  # already answer-bearing; take the first n_pos
    neg = rng.sample(negative_documents, min(n_neg, len(negative_documents)))
    docs = pos + neg
    rng.shuffle(docs)
    return docs


def get_sample_int_docs_at_noise_ratio(
    positive_document_groups: list[list[str]],
    negative_documents: list[str],
    noise_ratio: float,
    docs_per_question: int,
    rng: random.Random,
) -> list[str]:
    """Information integration: mirrors the RGB paper's reference recipe (evalue.py::processdata,
    the "_int" branch) — same fixed docs_per_question as noise robustness, split into
    n_pos/n_neg by noise_ratio (n_neg rounded up so a nonzero ratio always adds >=1 noise doc).

    Positive docs are built by taking one doc per sub-question group first, so every sub-question
    stays answerable, then padding with further docs from the groups (round-robin) if there are
    fewer groups than n_pos calls for. If padding still falls short of n_pos, the remaining budget
    is filled with noise docs instead.

    positive_document_groups : one answer-bearing document list per sub-question (RGB en_int
                               record's "positive" field, grouped rather than flat).
    negative_documents       : noise docs shared across sub-questions (RGB record's "negative" field).
    """
    n_neg = math.ceil(docs_per_question * noise_ratio)
    n_pos = docs_per_question - n_neg

    shuffled_groups = [rng.sample(group, len(group)) for group in positive_document_groups if group]
    positive_docs = [group[0] for group in shuffled_groups]

    max_group_len = max((len(group) for group in shuffled_groups), default=0)
    for depth in range(1, max_group_len):
        if len(positive_docs) >= n_pos:
            break
        for group in shuffled_groups:
            if len(group) > depth:
                positive_docs.append(group[depth])
                if len(positive_docs) >= n_pos:
                    break

    n_neg = max(0, docs_per_question - len(positive_docs))
    noise_docs = rng.sample(negative_documents, min(n_neg, len(negative_documents)))

    docs = positive_docs + noise_docs
    rng.shuffle(docs)
    return docs
