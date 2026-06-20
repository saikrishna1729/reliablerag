import os
import zipfile
from pathlib import Path

def zip_workspace(output_filename="ragstack_colab.zip"):
    workspace_dir = Path("/Users/aishwaryashilpi/workspace/ragstack")
    output_path = workspace_dir / "scripts" / output_filename

    # Files/directories to exclude
    exclude_dirs = {".git", ".venv", "__pycache__", "chroma_db"}
    exclude_files = {".DS_Store", "master_tracking.csv", "zip_workspace.py", output_filename}
    
    print(f"Creating zip archive at {output_path}...")
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(workspace_dir):
            root_path = Path(root)
            
            # Check if any part of the path is in excluded directories
            relative_parts = root_path.relative_to(workspace_dir).parts
            if any(p in exclude_dirs for p in relative_parts):
                continue
                
            # If under data/raw, exclude all subfolders except covidqa
            if "data" in relative_parts and "raw" in relative_parts:
                raw_idx = relative_parts.index("raw")
                if len(relative_parts) > raw_idx + 1:
                    dataset_subfolder = relative_parts[raw_idx + 1]
                    if dataset_subfolder != "covidqa":
                        continue
            
            # Exclude eval run csv files
            if "eval" in relative_parts and "results" in relative_parts and "runs" in relative_parts:
                pass

            for file in files:
                if file in exclude_files:
                    continue
                
                # If under eval/results/runs, exclude csv files
                if "eval" in relative_parts and "results" in relative_parts and "runs" in relative_parts:
                    if file.endswith(".csv"):
                        continue
                        
                file_path = root_path / file
                archive_name = file_path.relative_to(workspace_dir)
                zipf.write(file_path, archive_name)
                print(f"  Added: {archive_name}")

    print(f"\nSuccessfully created {output_filename} ({os.path.getsize(output_path) / 1024:.2f} KB)")

if __name__ == "__main__":
    zip_workspace()
