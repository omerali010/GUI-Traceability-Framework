"""Bar chart helpers used by the final thesis figure workflow."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from chart_common import (
    chart_sort_key,
    save_figure,
    save_placeholder_chart,
)
from metrics_common import clean, mean, safe_float


CANDIDATE_CHART_COLUMNS = {
    "model": ("Model", "Model variant"),
    "filter": ("Filter",),
    "widgets": ("Widgets", "Widget candidates"),
    "transitions": ("Transitions", "Transition candidates"),
    "total": ("Total cand.", "Total candidates"),
    "percentage": ("Relevant cand. (%)", "Relevant candidate %"),
}

CANDIDATE_CHART_ORDER: Sequence[Tuple[str, str, str]] = (
    ("Focus-Gemma3", "Strict", "Gemma3 Focus\nStrict"),
    ("Focus-Gemma3", "Relaxed", "Gemma3 Focus\nRelaxed"),
    ("Focus-GPT5", "Strict", "GPT5 Focus\nStrict"),
    ("Focus-GPT5", "Relaxed", "GPT5 Focus\nRelaxed"),
    ("LLMReq-Gemma3", "Strict", "Gemma3 LLMReq\nStrict"),
    ("LLMReq-Gemma3", "Relaxed", "Gemma3 LLMReq\nRelaxed"),
    ("LLMReq-GPT5", "Strict", "GPT5 LLMReq\nStrict"),
    ("LLMReq-GPT5", "Relaxed", "GPT5 LLMReq\nRelaxed"),
)


def read_candidate_space_chart_rows(path: Path) -> List[Dict[str, Any]]:
    """Read candidate-space CSV rows in the order used by the final bar chart."""
    with path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        columns: Dict[str, str] = {}
        for logical_name, aliases in CANDIDATE_CHART_COLUMNS.items():
            column = next((alias for alias in aliases if alias in reader.fieldnames), None)
            if column is None:
                raise KeyError(f"Could not find column for {logical_name!r}. Tried: {', '.join(aliases)}")
            columns[logical_name] = column

        rows = [
            {
                "model": row[columns["model"]],
                "filter": row[columns["filter"]],
                "widgets": int(row[columns["widgets"]]),
                "transitions": int(row[columns["transitions"]]),
                "total": int(row[columns["total"]]),
                "percentage": float(row[columns["percentage"]]),
            }
            for row in reader
        ]

    rows_by_key = {(row["model"], row["filter"]): row for row in rows}
    ordered_rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for model, filter_variant, label in CANDIDATE_CHART_ORDER:
        row = rows_by_key.get((model, filter_variant))
        if row is None:
            missing.append(f"{model} / {filter_variant}")
            continue
        ordered_rows.append({**row, "label": label})
    if missing:
        raise ValueError(f"Missing expected summary rows: {', '.join(missing)}")
    return ordered_rows


def plot_candidate_space(
    rows: Sequence[Dict[str, Any]],
    output_pdf: Path,
    output_png: Path,
) -> List[Path]:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    labels = [row["label"] for row in rows]
    widgets = np.array([row["widgets"] for row in rows], dtype=float)
    transitions = np.array([row["transitions"] for row in rows], dtype=float)
    totals = np.array([row["total"] for row in rows], dtype=float)
    percentages = [row["percentage"] for row in rows]
    # Add extra horizontal space between model variants while keeping each
    # strict/relaxed pair visually close together.
    x = np.array([group * 2.15 + offset for group in range(4) for offset in (0.0, 1.1)])
    bar_width = 0.52

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ax.bar(x, widgets, width=bar_width, label="Widget candidates", color="#4C78A8")
    ax.bar(
        x,
        transitions,
        width=bar_width,
        bottom=widgets,
        label="Transition candidates",
        color="#F58518",
    )

    annotation_offset = max(totals) * 0.025
    for position, total, percentage in zip(x, totals, percentages):
        ax.text(
            position,
            total + annotation_offset,
            f"{percentage:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.set_ylabel("Number of candidates")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.margins(x=0.025)
    ax.set_ylim(0, max(totals) * 1.16)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncol=2)

    fig.tight_layout()
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    return [output_pdf, output_png]


def _scalability_labels(rows: Sequence[Dict[str, Any]], group_key: str) -> List[str]:
    if group_key == "method_family":
        family_order = {"IR": 0, "CE": 1}
        return sorted(
            {clean(row.get(group_key)) for row in rows if clean(row.get(group_key)) in family_order},
            key=lambda label: family_order[label],
        )
    return sorted({clean(row.get(group_key)) for row in rows}, key=chart_sort_key)


def _mean_for_group(rows: Sequence[Dict[str, Any]], group_key: str, label: str, field: str) -> float:
    return mean([safe_float(row.get(field)) for row in rows if clean(row.get(group_key)) == label])


def _plot_scalability_runtime(
    rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    group_key: str,
    filename: str,
    figsize: Tuple[float, float],
    rotate_labels: bool,
) -> Path:
    path = output_dir / filename
    if not rows:
        return save_placeholder_chart(path, "Scalability runtime", "No runtime metadata was found in result JSON files.")

    labels = _scalability_labels(rows, group_key)
    if not labels:
        return save_placeholder_chart(
            path,
            "Scalability runtime by method family",
            "No IR or CE method-family labels were available.",
        )
    values = [_mean_for_group(rows, group_key, label, "total_seconds") for label in labels]
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_ylabel("Mean runtime (seconds)")
    ax.set_xticks(x)
    if rotate_labels:
        ax.set_xticklabels(labels, rotation=35, ha="right")
    else:
        ax.set_xticklabels(labels)
    return save_figure(fig, path)


def _plot_scalability_memory(
    rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    group_key: str,
    filename: str,
    figsize: Tuple[float, float],
    rotate_labels: bool,
) -> Path:
    path = output_dir / filename
    if not rows:
        return save_placeholder_chart(path, "Scalability memory", "No memory metadata was found in result JSON files.")

    labels = _scalability_labels(rows, group_key)
    if not labels:
        return save_placeholder_chart(
            path,
            "Scalability memory by method family",
            "No IR or CE method-family labels were available.",
        )
    x = np.arange(len(labels))
    rss_values = [_mean_for_group(rows, group_key, label, "process_rss_peak_mb") for label in labels]
    cuda_values = [_mean_for_group(rows, group_key, label, "cuda_peak_allocated_mb") for label in labels]
    width = 0.36
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, rss_values, width, label="System RAM peak")
    ax.bar(x + width / 2, cuda_values, width, label="GPU memory peak")
    ax.set_ylabel("Mean peak memory (MB)")
    ax.set_xticks(x)
    if rotate_labels:
        ax.set_xticklabels(labels, rotation=35, ha="right")
    else:
        ax.set_xticklabels(labels)
    ax.legend()
    return save_figure(fig, path)


def plot_scalability_runtime(scalability_rows: Sequence[Dict[str, Any]], output_dir: Path) -> Path:
    labels = _scalability_labels(scalability_rows, "method_filter")
    return _plot_scalability_runtime(
        scalability_rows,
        output_dir,
        "method_filter",
        "scalability_runtime_total_seconds.png",
        (max(9, len(labels) * 1.15), 4.8),
        True,
    )


def plot_scalability_memory(scalability_rows: Sequence[Dict[str, Any]], output_dir: Path) -> Path:
    labels = _scalability_labels(scalability_rows, "method_filter")
    return _plot_scalability_memory(
        scalability_rows,
        output_dir,
        "method_filter",
        "scalability_peak_memory_mb.png",
        (max(9, len(labels) * 1.15), 4.8),
        True,
    )
