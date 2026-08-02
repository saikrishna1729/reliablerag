import sys
import chromadb
from pathlib import Path

# Path to database (assumes it runs from the scripts directory)
db_path = Path(__file__).parent.parent / "chroma_db"

def inspect():
    if not db_path.exists():
        print(f"Chroma DB directory not found at: {db_path.resolve()}")
        print("Please ensure you have run a benchmarking experiment locally first (which populates chroma_db/).")
        print("If you are running in Google Colab, you can run this script directly in a notebook cell with the Colab db_path.")
        return

    client = chromadb.PersistentClient(path=str(db_path))
    collections = client.list_collections()

    if not collections:
        print("No collections found in Chroma DB.")
        return

    print("\n=== Available ChromaDB Collections ===")
    for idx, col in enumerate(collections, 1):
        print(f"[{idx}] Name: {col.name} | Total Chunks: {col.count()}")

    # Select collection to inspect
    if len(sys.argv) > 1:
        col_name = sys.argv[1]
    else:
        col_name = collections[0].name
        print(f"\nDefaulting to inspecting: {col_name}")

    try:
        collection = client.get_collection(col_name)
        count = collection.count()
        if count == 0:
            print("Collection is empty.")
            return

        # Fetch first 5 chunks
        results = collection.get(limit=5)
        
        print(f"\n=== Showing first 5 chunks of '{col_name}' ===")
        for i in range(len(results['ids'])):
            print(f"\n--- [Chunk {i+1}] ID: {results['ids'][i]} ---")
            if results['metadatas'] and results['metadatas'][i]:
                print(f"Metadata: {results['metadatas'][i]}")
            print("Raw Text:")
            print("-" * 40)
            print(results['documents'][i])
            print("-" * 40)

    except Exception as e:
        print(f"Error accessing collection '{col_name}': {e}")

if __name__ == "__main__":
    inspect()
