"""RAGBench data loading -- any of the 12 component datasets, not just DelucionQA."""

import pandas as pd
from datasets import load_dataset

RAGBENCH_DATASET = "galileo-ai/ragbench"

# The domain/task configs available in RAGBench (paper Table 1: RAGBench
# component datasets). Pass any of these as `dataset_name`.
RAGBENCH_CONFIGS = (
    "pubmedqa",    # bio-medical research
    "covidqa",     # bio-medical research
    "hotpotqa",    # general knowledge
    "msmarco",     # general knowledge
    "hagrid",      # general knowledge
    "expertqa",    # general knowledge
    "cuad",        # legal contracts
    "delucionqa",  # customer support (Jeep manual)
    "emanual",     # customer support (TV manual)
    "techqa",      # customer support (tech forums)
    "finqa",       # finance
    "tatqa",       # finance
)


def load_rag_bench_data(
    dataset_name: str = "delucionqa",
    split: str = "test",
    num_samples: int | None = 50,
) -> pd.DataFrame:
    """Load a RAGBench component dataset and flatten to one row per (question, document).

    dataset_name: one of RAGBENCH_CONFIGS, e.g. "delucionqa", "pubmedqa", "cuad".
    split: "train", "validation", or "test".
    num_samples: cap on the number of source (question, documents) rows loaded
        before flattening, to keep local experiments fast. Pass None to load
        the full split.

    Each returned row corresponds to a single context document for a question
    (there are len(documents) rows per original question), matching the shape
    the chunking pipelines in these notebooks expect.
    """
    ds = load_dataset(RAGBENCH_DATASET, dataset_name, split=split)
    if num_samples is not None:
        ds = ds[:num_samples]
    df = pd.DataFrame(ds)

    all_docs = []
    for _, row in df.iterrows():
        for doc_pos, doc_text in enumerate(row["documents"]):
            all_docs.append({
                "doc_id": f"{row['id']}_d{doc_pos}",
                "row_id": str(row["id"]),
                "doc_pos": doc_pos,
                "text": doc_text.strip(),
                "question": row["question"],
                "response": row["response"],
            })

    return pd.DataFrame(all_docs)
