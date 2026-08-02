import os
import glob
import json

downloads_dir = "/Users/aishwaryashilpi/Downloads/drive-download-20260802T031937Z-1-001/"
workspace_dir = "/Users/aishwaryashilpi/workspace/ragstack/"

print("--- DOWNLOADS ---")
d_files = sorted(glob.glob(os.path.join(downloads_dir, "*_summary.json")))
for f in d_files:
    print(os.path.basename(f))

print("\n--- WORKSPACE eval/results/rgb/ ---")
w_files_rgb = sorted(glob.glob(os.path.join(workspace_dir, "eval/results/rgb/*_summary.json")))
for f in w_files_rgb:
    print(os.path.basename(f))

print("\n--- WORKSPACE eval/results/runs/ ---")
# Let's see if there are any summary.json or similar files here
w_files_runs = sorted(glob.glob(os.path.join(workspace_dir, "eval/results/runs/*_summary.json")))
for f in w_files_runs:
    print(os.path.basename(f))
