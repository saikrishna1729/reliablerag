"""Download and load the RGB English testbeds.

RGB ships each question with its own documents (no retrieval needed). We pull the three English
files from the RGB repo — used for data + schema reference only, no implementation code copied.

    en_refine.json  -> noise robustness + negative rejection
    en_int.json     -> information integration
    en_fact.json    -> counterfactual robustness
"""

import json
import urllib.request
from pathlib import Path

RGB_RAW = "https://raw.githubusercontent.com/chen700564/RGB/master/data"
_FILES = ("en_refine.json", "en_int.json", "en_fact.json")

# Resolve data dir relative to the repo root so it works regardless of cwd (e.g. notebooks/).
# data.py -> rgb -> reliablerag -> src -> <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "rgb"


def download_rgb_data(data_dir: Path | str = DEFAULT_DATA_DIR) -> Path:
    """Download the three English RGB files into data_dir (skips files already present)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in _FILES:
        dest = data_dir / fname
        if dest.exists():
            print(f"[skip] {fname} already present")
            continue
        print(f"[download] {fname} ...")
        urllib.request.urlretrieve(f"{RGB_RAW}/{fname}", dest)
        print(f"[ok] {dest}")
    return data_dir


def _load_jsonl(path: Path) -> list[dict]:
    """RGB files are JSON-lines: one record per line."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rgb(data_dir: Path | str = DEFAULT_DATA_DIR) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (refine, integ, fact) — the three English testbeds as lists of records."""
    data_dir = Path(data_dir)
    refine = _load_jsonl(data_dir / "en_refine.json")
    integ = _load_jsonl(data_dir / "en_int.json")
    fact = _load_jsonl(data_dir / "en_fact.json")
    return refine, integ, fact
