from dataclasses import dataclass, field
from typing import Optional, Dict
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
EVAL_DIR  = BASE_DIR / "eval" / "results" / "runs"

# ── Type aliases ───────────────────────────────────────────────────────────
EmbedderType  = Literal["mock", "minilm", "bge-small", "bge-large", "ollama"]
GeneratorType = Literal["mock", "ollama", "hf-small", "hf-large"]
EvaluatorType = Literal["mock", "heuristic", "llm"]
ProfileType   = Literal["local-mock", "local-real", "colab-gpu"]

# ── Environment profiles ───────────────────────────────────────────────────
PROFILES: Dict[ProfileType, dict] = {
    "local-mock": {
        "embedder":  "mock",
        "generator": "mock",
        "evaluator": "mock",
    },
    "local-real": {
        "embedder":  "minilm",       # CPU, ~90MB, no GPU needed
        "generator": "ollama",       # requires Ollama running locally
        "evaluator": "heuristic",    # no LLM judge needed
    },
    "colab-gpu": {
        "embedder":  "bge-small",    # upgrade to bge-large for best quality
        "generator": "hf-small",     # google/flan-t5-base
        "evaluator": "llm",          # LLM-as-a-judge via HuggingFace
    },
}

# ── Main config dataclass ──────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    # Primary knob — swap this to reconfigure everything
    profile: ProfileType = "local-mock"

    # Component overrides — set any of these to override the profile default
    embedder:  Optional[EmbedderType]  = None
    generator: Optional[GeneratorType] = None
    evaluator: Optional[EvaluatorType] = None

    # User
    username:  str           = "anonymous"

    # Dataset params
    dataset:   str           = "covidqa"
    n_records: Optional[int] = None      # None = full dataset

    # Chunking params
    chunker:       Literal["recursive", "semantic", "metadata"] = "recursive"
    chunk_size:    int = 512
    chunk_overlap: int = 50

    # Retrieval params
    retriever: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k:     int = 5

    # Unique prefix for run session / sweep grouping
    run_id_prefix: Optional[str] = None

    def __post_init__(self):
        defaults = PROFILES[self.profile]
        if self.embedder  is None: self.embedder  = defaults["embedder"]
        if self.generator is None: self.generator = defaults["generator"]
        if self.evaluator is None: self.evaluator = defaults["evaluator"]

    @property
    def chroma_collection_name(self) -> str:
        """Unique collection per embedder — prevents dimension mismatch on Drive."""
        return f"{self.dataset}_{self.embedder}"
