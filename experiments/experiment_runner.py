import os
import glob
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from config import ExperimentConfig, DATA_DIR
from src.chunking import chunk_documents
from src.embeddings import get_embedder
from src.retrieval import get_retriever
from src.generation import get_generator
from src.evaluation import get_evaluator
from src.tracking import log_experiment_run

def run_experiment(config: ExperimentConfig) -> str:
    """
    Orchestrates a single RAG benchmarking experiment from loading data to logging results.
    """
    print(f"\n========================================================")
    print(f"Starting experiment for user: {config.username}")
    print(f"Profile: {config.profile}")
    print(f"Dataset: {config.dataset} (n_records: {config.n_records})")
    print(f"Configuration: Chunker={config.chunker}, Embedder={config.embedder}, "
          f"Retriever={config.retriever}, Generator={config.generator}, Evaluator={config.evaluator}")
    print(f"========================================================")

    # 1. Locate and load dataset Parquet file
    dataset_dir = DATA_DIR / "raw" / config.dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            f"Please ensure it is downloaded or mapped correctly."
        )

    parquet_files = glob.glob(str(dataset_dir / "*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {dataset_dir}")

    # Prioritize split: test -> validation -> train -> any
    test_files = [f for f in parquet_files if "test" in os.path.basename(f)]
    val_files = [f for f in parquet_files if "validation" in os.path.basename(f) or "val" in os.path.basename(f)]
    train_files = [f for f in parquet_files if "train" in os.path.basename(f)]

    if test_files:
        file_to_load = test_files[0]
    elif val_files:
        file_to_load = val_files[0]
    elif train_files:
        file_to_load = train_files[0]
    else:
        file_to_load = parquet_files[0]

    print(f"Loading data from {os.path.basename(file_to_load)}...")
    df = pd.read_parquet(file_to_load)
    
    # Slice rows
    if config.n_records is not None:
        df = df.head(config.n_records)
    print(f"Loaded {len(df)} records for evaluation.")

    # 2. Initialize factory components
    embedder = get_embedder(config)
    generator = get_generator(config)
    evaluator = get_evaluator(config, generator)

    # 3. Extract unique corpus documents
    print("Extracting and deduplicating document corpus...")
    all_docs = []
    seen_docs = set()
    doc_metadata = []

    for idx, row in df.iterrows():
        row_docs = row.get("documents", [])
        # RAGBench documents could be strings or lists
        if isinstance(row_docs, str):
            row_docs = [row_docs]
            
        row_id = row.get("id", f"q_{idx}")
        for doc_idx, doc_text in enumerate(row_docs):
            if not doc_text:
                continue
            if doc_text not in seen_docs:
                seen_docs.add(doc_text)
                all_docs.append(doc_text)
                doc_metadata.append({
                    "source": f"{row_id}_doc_{doc_idx}",
                    "dataset": config.dataset
                })

    print(f"Found {len(all_docs)} unique raw documents in corpus.")

    # 4. Chunk corpus
    print(f"Chunking corpus using strategy: {config.chunker}...")
    chunks = chunk_documents(
        texts=all_docs,
        metadatas=doc_metadata,
        chunk_strategy=config.chunker,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        embedder=embedder,
        dataset_name=config.dataset
    )
    print(f"Created {len(chunks)} text chunks.")

    # 5. Build/load index and get retriever
    retriever = get_retriever(config, chunks, embedder)

    # 6. Run evaluation loop
    print("Running inference and evaluation loop...")
    run_rows = []
    
    if config.run_id_prefix:
        run_id = f"{config.run_id_prefix}_{config.chunker}_{config.embedder}_{config.retriever}"
    else:
        timestamp_str = time.strftime("%Y%m%d_%H%M")
        run_id = f"{timestamp_str}_{config.username}_{config.dataset}_{config.chunker}_{config.embedder}_{config.retriever}"

    for idx, row in df.iterrows():
        question = row["question"]
        question_id = row.get("id", f"q_{idx}")
        print(f"[{idx+1}/{len(df)}] Querying: '{question[:60]}...'")

        start_time = time.time()
        
        # Retrieve context
        try:
            retrieved_chunks = retriever.retrieve(question)
            context = "\n\n".join([c.page_content for c in retrieved_chunks])
            retrieved_sources = "|".join([c.metadata.get("source", "unknown") for c in retrieved_chunks])
        except Exception as e:
            print(f"Error during retrieval for query {question_id}: {e}")
            retrieved_chunks = []
            context = ""
            retrieved_sources = "error"

        # Generate response
        try:
            answer = generator.generate(question, context)
        except Exception as e:
            print(f"Error during generation for query {question_id}: {e}")
            answer = f"ERROR: Generation failed. {e}"

        latency_ms = int((time.time() - start_time) * 1000)

        # Evaluate response
        try:
            scores = evaluator.score(question, context, answer)
        except Exception as e:
            print(f"Error during evaluation for query {question_id}: {e}")
            scores = {
                "context_relevance": 0.0,
                "context_utilization": 0.0,
                "completeness": 0.0,
                "adherence": 0.0
            }

        # Build log row
        run_row = {
            "run_id": run_id,
            "username": config.username,
            "question_id": question_id,
            "question": question,
            "retrieved_docs": retrieved_sources,
            "answer": answer,
            "context_relevance": scores["context_relevance"],
            "context_utilization": scores["context_utilization"],
            "completeness": scores["completeness"],
            "adherence": scores["adherence"],
            "latency_ms": latency_ms,
            "embedder": config.embedder,
            "generator": config.generator,
            "evaluator": config.evaluator,
            "chunker": config.chunker,
            "retriever": config.retriever,
            "chunk_size": config.chunk_size,
            "top_k": config.top_k
        }
        run_rows.append(run_row)

    # 7. Compute aggregate metrics
    latencies = [row["latency_ms"] for row in run_rows]
    mean_metrics = {
        "mean_context_relevance": float(np.mean([row["context_relevance"] for row in run_rows])),
        "mean_context_utilization": float(np.mean([row["context_utilization"] for row in run_rows])),
        "mean_completeness": float(np.mean([row["completeness"] for row in run_rows])),
        "mean_adherence": float(np.mean([row["adherence"] for row in run_rows])),
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0
    }

    print("\n--- Summary Results ---")
    for k, v in mean_metrics.items():
        print(f"{k}: {v:.4f}")

    # 8. Log run to CSV and master leaderboard
    detailed_filename = log_experiment_run(config, run_rows, mean_metrics)
    print("Experiment run finished successfully.\n")
    return detailed_filename
