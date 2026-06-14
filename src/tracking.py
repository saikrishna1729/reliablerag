import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from config import ExperimentConfig, EVAL_DIR, BASE_DIR

def log_experiment_run(
    config: ExperimentConfig,
    run_rows: List[Dict[str, Any]],
    mean_metrics: Dict[str, float]
) -> str:
    """
    Saves the detailed run rows into a YYYYMMDD_HHMM_USER_DOMAIN_CHUNKER_EMBEDDER_RETRIEVER.csv
    and appends a summary row to master_tracking.csv.
    
    Returns:
        The filename of the detailed run CSV.
    """
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
    
    # 1. Create directory if not exists
    os.makedirs(EVAL_DIR, exist_ok=True)
    
    # 2. Save detailed run CSV
    detailed_filename = (
        f"{timestamp_str}_{config.username}_{config.dataset}_"
        f"{config.chunker}_{config.embedder}_{config.retriever}.csv"
    )
    detailed_path = EVAL_DIR / detailed_filename
    
    detailed_df = pd.DataFrame(run_rows)
    detailed_df.to_csv(detailed_path, index=False)
    print(f"Detailed run logged to {detailed_path}")
    
    # 3. Create run_id
    run_id = (
        f"{timestamp_str}_{config.username}_{config.dataset}_"
        f"{config.chunker}_{config.embedder}_{config.retriever}"
    )
    
    # 4. Append to master_tracking.csv
    master_path = BASE_DIR / "master_tracking.csv"
    
    summary_row = {
        "run_id": run_id,
        "username": config.username,
        "dataset": config.dataset,
        "n_records": config.n_records if config.n_records is not None else "all",
        "chunker": config.chunker,
        "embedder": config.embedder,
        "retriever": config.retriever,
        "generator": config.generator,
        "evaluator": config.evaluator,
        "mean_context_relevance": mean_metrics["mean_context_relevance"],
        "mean_context_utilization": mean_metrics["mean_context_utilization"],
        "mean_completeness": mean_metrics["mean_completeness"],
        "mean_adherence": mean_metrics["mean_adherence"],
        "mean_latency_ms": mean_metrics["mean_latency_ms"],
        "timestamp": datetime.now().isoformat()
    }
    
    summary_df = pd.DataFrame([summary_row])
    
    if os.path.exists(master_path):
        # Read existing and append
        try:
            existing_df = pd.read_csv(master_path)
            # Ensure columns align
            combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
            combined_df.to_csv(master_path, index=False)
        except Exception as e:
            print(f"Error reading master_tracking.csv: {e}. Rewriting...")
            summary_df.to_csv(master_path, index=False)
    else:
        summary_df.to_csv(master_path, index=False)
        
    print(f"Leaderboard updated in {master_path}")
    return detailed_filename
