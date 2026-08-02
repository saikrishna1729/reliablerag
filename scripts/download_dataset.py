import os
import sys

def download_dataset(dataset_name="cuad"):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: The 'datasets' library is required to download RAGBench datasets.")
        print("Please install it first: pip install datasets")
        return

    print(f"Downloading RAGBench '{dataset_name}' dataset from Hugging Face...")
    try:
        # Load the dataset from Hugging Face RAGBench repository
        dataset = load_dataset("rungalileo/ragbench", dataset_name)
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}' from Hugging Face: {e}")
        print("Valid RAGBench datasets include: 'covidqa', 'cuad', 'delucionqa', 'emanual', 'expertqa', 'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa', 'tatqa', 'techqa'")
        return

    # Output directory
    output_dir = os.path.join("data", "raw", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save each split as a Parquet file in the expected format
    print(f"Saving parquet splits to {output_dir}...")
    for split_name, split_data in dataset.items():
        parquet_path = os.path.join(output_dir, f"{split_name}-00000-of-00001.parquet")
        try:
            split_data.to_parquet(parquet_path)
            print(f"  ✓ Saved {split_name} split to: {parquet_path}")
        except Exception as e:
            print(f"  ✗ Failed to save {split_name} split: {e}")

    print(f"\nSuccessfully downloaded '{dataset_name}' dataset. You can now use it in your benchmark runs!")

if __name__ == "__main__":
    target_dataset = sys.argv[1] if len(sys.argv) > 1 else "cuad"
    download_dataset(target_dataset)
