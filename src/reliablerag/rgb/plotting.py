"""Comparison plots for the four RGB abilities.

Every chart uses the same fixed color per model (never re-cycled per chart) so a model's
identity reads consistently across every plot in the notebook.
"""

import matplotlib.pyplot as plt

MODEL_COLORS = {
    "openai/gpt-oss-120b": "#2a78d6",  # blue
    "meta-llama/llama-3.3-70b-instruct": "#008300",  # green
    "qwen/qwen-2.5-72b-instruct": "#e87ba4",  # magenta
}

_GRIDLINE_COLOR = "#e1e0d9"


def _short_label(model: str) -> str:
    return model.split("/")[-1]


def _style_axes(ax) -> None:
    ax.grid(axis="y", color=_GRIDLINE_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_line_by_ratio(
    results: dict[str, list[float]],
    ratios: list[float],
    title: str,
    ylabel: str = "Accuracy (%)",
    xlabel: str = "Noise Ratio",
) -> None:
    """Line chart: one line per model, accuracy (%) across a swept ratio, with every point
    labeled with its value. Used for noise robustness and information integration.

    All labels at a given x are anchored to that x's HIGHEST marker (not each label's own
    marker) and stacked upward from there by rank -- so a label never collides with another
    model's nearby marker, only with other labels, which the rank spacing already prevents.
    """
    models = list(results.keys())
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for model in models:
        color = MODEL_COLORS[model]
        ax.plot(ratios, results[model], marker="o", markersize=8, linewidth=2, color=color, label=_short_label(model))

    label_step = 18  # points between stacked labels at the same x position (must clear font height)
    for x_idx, ratio in enumerate(ratios):
        ranked = sorted(((results[m][x_idx], m) for m in models), key=lambda p: p[0])
        top_value = ranked[-1][0]
        for rank, (value, model) in enumerate(ranked):
            ax.annotate(
                f"{value:.1f}", (ratio, top_value),
                textcoords="offset points", xytext=(0, 10 + rank * label_step),
                fontsize=8, fontweight="bold", color=MODEL_COLORS[model], ha="center",
            )

    all_values = [v for values in results.values() for v in values]
    # Default floor is 60 (5-unit gridlines from 60-100) for these accuracy-% charts; only go
    # lower if the data itself would otherwise be clipped.
    y_min = min(60, 5 * (int(min(all_values)) // 5))
    y_max = max(all_values) + 8 + (len(models) - 1) * label_step * 0.12  # headroom for stacked labels
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(range(y_min, 101, 5))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(ratios)
    ax.set_xlim(ratios[0] - 0.05, ratios[-1] + 0.05)
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower left")
    _style_axes(ax)
    plt.tight_layout()
    plt.show()


def plot_bar(results: dict[str, float], title: str, ylabel: str) -> None:
    """Single bar per model. Used for negative rejection (one Rej% value per model)."""
    models = list(results.keys())
    values = list(results.values())
    labels = [_short_label(m) for m in models]
    colors = [MODEL_COLORS[m] for m in models]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.set_title(title)
    _style_axes(ax)
    plt.tight_layout()
    plt.show()


def plot_grouped_bar(
    results: dict[str, list[float]],
    categories: list[str],
    title: str,
    ylabel: str = "Score (%)",
) -> None:
    """Grouped bar chart: one group per category (e.g. ACC/ACC_doc/ED/CR), one bar per model
    within each group. Used for counterfactual robustness."""
    import numpy as np

    x = np.arange(len(categories))
    n_models = len(results)
    bar_width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (model, values) in enumerate(results.items()):
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, width=bar_width * 0.9, color=MODEL_COLORS[model], label=_short_label(model))
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    _style_axes(ax)
    plt.tight_layout()
    plt.show()