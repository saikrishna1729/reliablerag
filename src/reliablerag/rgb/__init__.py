"""RGB (Retrieval-Augmented Generation Benchmark) evaluation.

Reproduces the four RAG abilities from Chen et al., 2023 (arXiv:2309.01431), English-only.
See RG_plan.md and notebooks/06_rgb_evaluation.ipynb.
"""

from .abilities import (
    INT_RATIOS,
    NOISE_RATIOS,
    run_counterfactual,
    run_information_integration,
    run_negative_rejection,
    run_noise_robustness,
)
from .data import download_rgb_data, load_rgb
from .metrics import find_answer_matches, detects_error, is_accurate, is_rejection
from .plotting import MODEL_COLORS, plot_bar, plot_grouped_bar, plot_line_by_ratio
from .prompt import SYSTEM_PROMPT, USER_TEMPLATE, ask, ask_direct

__all__ = [
    "download_rgb_data",
    "load_rgb",
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
    "ask",
    "ask_direct",
    "is_accurate",
    "is_rejection",
    "detects_error",
    "find_answer_matches",
    "run_noise_robustness",
    "run_negative_rejection",
    "run_information_integration",
    "run_counterfactual",
    "NOISE_RATIOS",
    "INT_RATIOS",
    "MODEL_COLORS",
    "plot_line_by_ratio",
    "plot_bar",
    "plot_grouped_bar",
]