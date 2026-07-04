import os
import glob
import shutil
import tempfile
import zipfile
import argparse
from pathlib import Path
import pandas as pd

def find_latest_zip():
    downloads_dir = Path.home() / "Downloads"
    zip_files = glob.glob(str(downloads_dir / "eval_results*.zip"))
    if not zip_files:
        return None
    # Return the most recently modified zip file
    return max(zip_files, key=os.path.getmtime)

def df_to_markdown(df):
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        row_str = [str(x) if pd.notnull(x) else "" for x in row]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)

def generate_markdown_summary(df, output_path, local_repo):
    # Filter out mock runs (where embedder is 'mock')
    df = df[df["embedder"] != "mock"].copy()

    # Base layout
    md_content = [
        "# 🏆 RAGStack Evaluation Leaderboard",
        "",
        "This is an automatically generated summary of the RAG benchmarking runs, grouped by run session (Sweep ID).",
        "This file is tracked under Git version control to keep a record of your experimental history.",
        "",
    ]

    # Helper function to extract sweep_id prefix from run_id
    def get_sweep_prefix(run_id, dataset):
        if not isinstance(dataset, str) or not isinstance(run_id, str):
            return run_id
        idx = run_id.find(dataset)
        if idx != -1:
            return run_id[:idx + len(dataset)]
        return run_id

    # Create sweep_id column
    df['sweep_id'] = df.apply(lambda r: get_sweep_prefix(r['run_id'], r['dataset']), axis=1)

    # Group leaderboard by sweep_id, sorted descending (latest sweeps first)
    unique_sweeps = df["sweep_id"].dropna().unique()
    
    for sweep_id in sorted(unique_sweeps, reverse=True):
        sweep_df = df[df["sweep_id"] == sweep_id]
        if sweep_df.empty:
            continue
            
        # Get dataset and timestamp info from the sweep
        first_row = sweep_df.iloc[0]
        dataset = first_row.get("dataset", "unknown")
        
        # Sort this sweep's runs by mean_adherence descending, then mean_context_relevance
        sorted_df = sweep_df.sort_values(by=["mean_adherence", "mean_context_relevance"], ascending=False)
        
        cols = [
            "run_id", "chunker", "embedder", "retriever",
            "mean_adherence", "mean_context_relevance", "mean_context_utilization",
            "mean_latency_ms", "timestamp"
        ]
        cols = [c for c in cols if c in df.columns]
        display_df = sorted_df[cols].copy()
        
        # Format floats
        for col in ["mean_adherence", "mean_context_relevance", "mean_context_utilization"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
                
        if "mean_latency_ms" in display_df.columns:
            display_df["mean_latency_ms"] = display_df["mean_latency_ms"].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
            
        # Format sweep ID prefix to a user-friendly timestamp display
        timestamp_display = sweep_id
        parts = sweep_id.split('_')
        if len(parts) >= 2 and len(parts[0]) == 8 and len(parts[1]) == 4:
            date_part = f"{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:]}"
            time_part = f"{parts[1][:2]}:{parts[1][2:]}"
            user_part = "_".join(parts[2:-1]) if len(parts) > 3 else parts[2]
            timestamp_display = f"{date_part} {time_part} (User: {user_part})"
            
        md_content.extend([
            f"## 📊 Run Group: {dataset} - {timestamp_display}",
            f"**Sweep ID:** `{sweep_id}`",
            "",
            df_to_markdown(display_df),
            ""
        ])
        
        # Check if plot exists for this specific sweep
        eval_dir = local_repo / "eval"
        plot_filename = f"benchmark_plot_{sweep_id}.png"
        plot_path = eval_dir / plot_filename
        
        # Backward compatibility check for older plot filename formats
        if not plot_path.exists():
            # Check for format: benchmark_plot_{dataset}_{sweep_id}.png
            plot_filename = f"benchmark_plot_{dataset}_{sweep_id}.png"
            plot_path = eval_dir / plot_filename
            
        if plot_path.exists():
            md_content.extend([
                "### Performance Visualization",
                "",
                f"![RAG Performance - {sweep_id}]({plot_filename})",
                ""
            ])
            
        md_content.append("---")
        md_content.append("")
        
        # Generate individual report for this sweep
        ind_report_content = [
            f"# 🏆 RAGStack Evaluation Leaderboard - {sweep_id}",
            "",
            f"This is an automatically generated summary of the RAG benchmarking runs for sweep session `{sweep_id}`.",
            "",
            f"## 📊 Run Group: {dataset} - {timestamp_display}",
            f"**Sweep ID:** `{sweep_id}`",
            "",
            df_to_markdown(display_df),
            ""
        ]
        
        if plot_path.exists():
            ind_report_content.extend([
                "### Performance Visualization",
                "",
                f"![RAG Performance - {sweep_id}]({plot_filename})",
                ""
            ])
            
        ind_report_content.append(f"*Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        ind_report_path = eval_dir / f"evaluation_summary_{sweep_id}.md"
        with open(ind_report_path, "w") as f:
            f.write("\n".join(ind_report_content))
        print(f"Generated individual evaluation report: {ind_report_path}")

    md_content.append(f"*Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Clean up any old individual reports that are not in the active filtered sweep list
    eval_dir = local_repo / "eval"
    if eval_dir.exists():
        for f in eval_dir.glob("evaluation_summary_*.md"):
            stem = f.stem
            if stem.startswith("evaluation_summary_"):
                sweep_id_from_file = stem[len("evaluation_summary_"):]
                if sweep_id_from_file not in unique_sweeps:
                    try:
                        f.unlink()
                        print(f"Deleted old/mock report file: {f.name}")
                    except Exception as e:
                        print(f"Warning: Failed to delete old report file {f.name}: {e}")

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(md_content))
    print(f"Concise markdown summary updated: {output_path}")

def merge_results(zip_path):
    local_repo = Path(__file__).parent.parent.resolve()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Extracting {zip_path} to temp directory...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_path)
            
        # 1. Copy dataset-specific benchmark plots
        # Find all files matching benchmark_plot_*.png in extracted dir (could be at root or under eval/)
        copied_plots = 0
        local_plot_dir = local_repo / "eval"
        os.makedirs(local_plot_dir, exist_ok=True)
        
        # Search in temp root and temp/eval
        for search_path in [temp_path, temp_path / "eval"]:
            if search_path.exists():
                for plot_file in search_path.glob("benchmark_plot_*.png"):
                    shutil.copy2(plot_file, local_plot_dir / plot_file.name)
                    print(f"Benchmark plot copied to {local_plot_dir / plot_file.name}")
                    copied_plots += 1
                    
        # 2. Merge master_tracking.csv
        temp_master = temp_path / "master_tracking.csv"
        local_master = local_repo / "master_tracking.csv"
        
        combined_df = None
        if temp_master.exists():
            print("Merging master_tracking.csv...")
            new_df = pd.read_csv(temp_master)
            
            if local_master.exists():
                try:
                    existing_df = pd.read_csv(local_master)
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df.drop_duplicates(subset=["run_id"], keep="last", inplace=True)
                except Exception as e:
                    print(f"Warning: Failed to read local master_tracking.csv ({e}). Overwriting.")
                    combined_df = new_df
            else:
                combined_df = new_df
                
            combined_df.to_csv(local_master, index=False)
            print(f"Leaderboard successfully updated: {local_master}")
            
            # Generate concise markdown summary grouped by dataset
            summary_path = local_repo / "eval" / "evaluation_summary.md"
            generate_markdown_summary(combined_df, summary_path, local_repo)
        else:
            print("No master_tracking.csv found in the zip file.")
            
        # 3. Merge eval/results/runs/ directory
        temp_runs_dir = temp_path / "eval" / "results" / "runs"
        local_runs_dir = local_repo / "eval" / "results" / "runs"
        
        if temp_runs_dir.exists():
            os.makedirs(local_runs_dir, exist_ok=True)
            copied_count = 0
            for run_file in temp_runs_dir.glob("*.csv"):
                dest_file = local_runs_dir / run_file.name
                shutil.copy2(run_file, dest_file)
                copied_count += 1
            print(f"Copied {copied_count} detailed run logs to {local_runs_dir}")
        else:
            print("No detailed run files found in the zip file.")

def main():
    parser = argparse.ArgumentParser(description="Merge Colab benchmark results into local repository.")
    parser.add_argument(
        "--zip",
        type=str,
        default=None,
        help="Path to the downloaded eval_results.zip file."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Only refresh the markdown reports from local master_tracking.csv without merging."
    )
    args = parser.parse_args()
    
    local_repo = Path(__file__).parent.parent.resolve()
    local_master = local_repo / "master_tracking.csv"
    
    if args.refresh:
        if local_master.exists():
            print("Refreshing markdown reports from local master_tracking.csv...")
            df = pd.read_csv(local_master)
            summary_path = local_repo / "eval" / "evaluation_summary.md"
            generate_markdown_summary(df, summary_path, local_repo)
        else:
            print(f"Error: Local master_tracking.csv not found at {local_master}")
        return

    zip_path = args.zip
    if not zip_path:
        zip_path = find_latest_zip()
        if not zip_path:
            print("Error: Could not find any 'eval_results*.zip' file in your Downloads folder.")
            print("Please specify the path manually using: python scripts/merge_results.py --zip /path/to/file.zip")
            return
            
    print(f"Using zip file: {zip_path}")
    merge_results(zip_path)

if __name__ == "__main__":
    main()
