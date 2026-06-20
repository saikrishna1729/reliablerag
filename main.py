import argparse
import os
import sys
import time
from config import ExperimentConfig, ProfileType, EmbedderType, GeneratorType, EvaluatorType
from experiments.experiment_runner import run_experiment

def get_default_username() -> str:
    try:
        return os.getlogin()
    except Exception:
        return os.environ.get("USER", "anonymous")

def main():
    parser = argparse.ArgumentParser(description="RAGStack Benchmarking Pipeline")
    
    # Environment profile
    parser.add_argument(
        "--profile",
        choices=["local-mock", "local-real", "colab-gpu"],
        default="local-mock",
        help="Primary execution profile (default: local-mock)"
    )
    
    # Overrides
    parser.add_argument(
        "--embedder",
        choices=["mock", "minilm", "bge-small", "bge-large", "ollama"],
        default=None,
        help="Override profile default embedder"
    )
    parser.add_argument(
        "--generator",
        choices=["mock", "ollama", "hf-small", "hf-large"],
        default=None,
        help="Override profile default generator"
    )
    parser.add_argument(
        "--evaluator",
        choices=["mock", "heuristic", "llm"],
        default=None,
        help="Override profile default evaluator"
    )
    
    # User attribution
    parser.add_argument(
        "--username",
        default=get_default_username(),
        help="Username for logging and attribution"
    )
    
    # Dataset and slicing
    parser.add_argument(
        "--dataset",
        default="covidqa",
        help="RAGBench dataset name (default: covidqa)"
    )
    parser.add_argument(
        "-n", "--n_records",
        type=int,
        default=None,
        help="Number of records to evaluate (None for full dataset)"
    )
    
    # Chunker settings
    parser.add_argument(
        "--chunker",
        choices=["recursive", "semantic", "metadata"],
        default="recursive",
        help="Text splitting strategy (default: recursive)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Chunk size in characters (default: 512)"
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=50,
        help="Chunk overlap in characters (default: 50)"
    )
    
    # Retrieval settings
    parser.add_argument(
        "--retriever",
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="Retrieval architecture (default: hybrid)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of retrieved documents to inject in prompt (default: 5)"
    )
    
    # Sweep execution mode
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run batch sweep across chunkers, embedders, and retrievers"
    )
    parser.add_argument(
        "--run_id_prefix",
        default=None,
        help="Custom run ID prefix to group runs under (default: YYYYMMDD_HHMM_username_dataset)"
    )

    args = parser.parse_args()

    # Determine run_id_prefix once before starting any runs
    if args.run_id_prefix:
        run_id_prefix = args.run_id_prefix
    else:
        timestamp_str = time.strftime("%Y%m%d_%H%M")
        run_id_prefix = f"{timestamp_str}_{args.username}_{args.dataset}"

    # Determine sweep configurations
    latest_run_ids = []
    if args.sweep:
        print("Starting batch sweep mode...")
        chunkers = ["recursive", "semantic"]
        retrievers = ["dense", "hybrid"]
        
        # Guard against heavy downloads in local mock/real profiles during sweeps
        if args.profile == "local-mock":
            embedders = ["mock"]
        elif args.profile == "local-real":
            embedders = ["minilm"]
        else: # colab-gpu
            embedders = ["bge-small", "bge-large"]
            
        total_runs = len(chunkers) * len(embedders) * len(retrievers)
        print(f"Total sweep runs to execute: {total_runs}")
        
        run_count = 0
        for chunker in chunkers:
            for embedder in embedders:
                for retriever in retrievers:
                    run_count += 1
                    print(f"\n[Sweep Run {run_count}/{total_runs}]")
                    config = ExperimentConfig(
                        profile=args.profile,
                        embedder=embedder,
                        generator=args.generator,
                        evaluator=args.evaluator,
                        username=args.username,
                        dataset=args.dataset,
                        n_records=args.n_records,
                        chunker=chunker,
                        chunk_size=args.chunk_size,
                        chunk_overlap=args.chunk_overlap,
                        retriever=retriever,
                        top_k=args.top_k,
                        run_id_prefix=run_id_prefix
                    )
                    try:
                        detailed_filename = run_experiment(config)
                        # Extract run_id from filename (remove .csv extension)
                        run_id = os.path.splitext(detailed_filename)[0]
                        latest_run_ids.append(run_id)
                    except Exception as e:
                        print(f"Sweep run failed for config {config}: {e}")
                        
        print("\nSweep mode finished.")
    else:
        # Single run configuration
        config = ExperimentConfig(
            profile=args.profile,
            embedder=args.embedder,
            generator=args.generator,
            evaluator=args.evaluator,
            username=args.username,
            dataset=args.dataset,
            n_records=args.n_records,
            chunker=args.chunker,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            retriever=args.retriever,
            top_k=args.top_k,
            run_id_prefix=run_id_prefix
        )
        try:
            detailed_filename = run_experiment(config)
            run_id = os.path.splitext(detailed_filename)[0]
            latest_run_ids.append(run_id)
        except Exception as e:
            print(f"Run failed: {e}")

    # Write latest run IDs to a marker file
    if latest_run_ids:
        try:
            os.makedirs("eval", exist_ok=True)
            with open("eval/latest_run.txt", "w") as f:
                f.write("\n".join(latest_run_ids))
            print(f"\nSaved {len(latest_run_ids)} executed run IDs to eval/latest_run.txt for tracking.")
        except Exception as e:
            print(f"Warning: Failed to save latest run IDs to marker file: {e}")

if __name__ == "__main__":
    main()
