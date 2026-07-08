"""Shared helpers used by all chart modules."""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRIC_FIELDS = [
    ("precision_at_k", "Precision"),
    ("recall_at_k", "Recall"),
    ("f1_at_k", "F1"),
    ("map_at_k", "MAP"),
    ("mrr_at_k", "MRR"),
]

EVALUATION_DISPLAY_NAMES = {
    "actions_all": "Transitions (linked + orphan)",
    "actions_with_linked_widget": "Transitions with linked widget",
    "actions_orphan": "Orphan transitions",
    "widgets_linked_resolved": "Linked resolved widgets",
    "widgets_functional_resolved": "Functional resolved widgets",
    "widgets_all_resolved": "All resolved widgets",
    "combined_actions_and_linked_widgets": "Linked widgets + transitions",
    "combined_actions_and_widgets": "Widget and transition candidates",
}

METHOD_ORDER = {
    "VSM": 0,
    "LSI": 1,
    "JSM": 2,
    "Qwen3 0.6B": 3,
    "Qwen3 4B": 4,
    "Jina v3": 5,
    "Stella 1.5B": 6,
}
FILTER_ORDER = {"Strict": 0, "Relaxed": 1}
DATASET_ORDER = {"Focus": 0, "LLMReq": 1}
SOURCE_ORDER = {"GPT5": 0, "Gemma3": 1}


def chart_sort_key(row_or_label: Any) -> Tuple[int, int, int, int, str]:
    if isinstance(row_or_label, dict):
        method = row_or_label.get("method", "")
        filter_variant = row_or_label.get("filter_variant", "")
        dataset = row_or_label.get("dataset", "")
        source_model = row_or_label.get("source_model", "")
        label = row_or_label.get("method_filter", method)
    else:
        label = str(row_or_label)
        parts = label.rsplit(" ", 1)
        method = parts[0] if parts else label
        filter_variant = parts[1] if len(parts) == 2 else ""
        dataset = ""
        source_model = ""
    return (
        METHOD_ORDER.get(method, 99),
        FILTER_ORDER.get(filter_variant, 99),
        DATASET_ORDER.get(dataset, 99),
        SOURCE_ORDER.get(source_model, 99),
        label,
    )


def scenario_sort_key(label: str) -> Tuple[int, int, str]:
    dataset, _, source_model = label.partition("-")
    return (DATASET_ORDER.get(dataset, 99), SOURCE_ORDER.get(source_model, 99), label)


def rows_for(summary_rows: Sequence[Dict[str, Any]], evaluation: str, k: int) -> List[Dict[str, Any]]:
    return [
        row
        for row in summary_rows
        if row["evaluation"] == evaluation and int(row["k"]) == k
    ]


def display_name(value: str) -> str:
    return EVALUATION_DISPLAY_NAMES.get(value, value.replace("_", " "))


def metric_display_name(metric: str) -> str:
    return dict(METRIC_FIELDS).get(metric, display_name(metric))


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_placeholder_chart(path: Path, title: str, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=11)
    return save_figure(fig, path)


def apply_chart_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8").to_dict("records")
