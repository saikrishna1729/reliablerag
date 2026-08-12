"""Central location for on-disk Chroma vector stores.

Every notebook used to persist its Chroma collections into the system temp
directory (via bare `tempfile.mkdtemp()`) or an ad-hoc relative path like
"chromadb/". That scatters generated DB files outside the project, makes them
easy to lose track of, and gives every notebook its own idea of where things
live. All vector stores now go under `delucion_dataset/vector_stores/` instead.
"""

import os
import re
import tempfile

VECTOR_STORE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vector_stores"
)


def get_persist_dir(prefix: str = "chroma_") -> str:
    """Create and return a fresh, uniquely-named persist directory under vector_stores/.

    Drop-in replacement for `tempfile.mkdtemp(prefix=...)` -- same one-collection-
    per-call semantics (including being safe to `shutil.rmtree` afterwards), just
    rooted inside the project instead of the system temp directory.
    """
    os.makedirs(VECTOR_STORE_ROOT, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=VECTOR_STORE_ROOT)


def named_persist_dir(name: str) -> str:
    """A stable directory (reused across runs, not unique per call) under vector_stores/<name>."""
    path = os.path.join(VECTOR_STORE_ROOT, name)
    os.makedirs(path, exist_ok=True)
    return path


def _slug(text: str) -> str:
    """Chroma collection names must be alphanumeric/underscore/hyphen, 3-63 chars."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def vector_store_names(dataset: str, strategy: str) -> tuple[str, str]:
    """Build a matching (persist-dir prefix, chroma collection_name) for a dataset+strategy combo.

    e.g. vector_store_names("delucionqa", "M3") ->
        ("chroma_delucionqa_m3_", "ragbench_delucionqa_m3")

    So `ls vector_stores/` and Chroma's own collection listing both show which
    dataset and which experiment strategy each database belongs to.
    """
    slug = f"{_slug(dataset)}_{_slug(strategy)}"
    return f"chroma_{slug}_", f"ragbench_{slug}"
