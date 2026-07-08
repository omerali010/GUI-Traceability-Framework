"""Heatmaps for requirement-source and guiding-model comparisons."""

import math
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from chart_common import (
    chart_sort_key,
    metric_display_name,
    rows_for,
    save_figure,
    save_placeholder_chart,
    scenario_sort_key,
)
from metrics_common import clean, mean


def plot_heatmap(
    summary_rows: Sequence[Dict[str, Any]],
    main_k: int,
    evaluation: str,
    metric: str,
    output_dir: Path,
    filename_prefix: str = "",
    custom_title: str = "",
    average_scope: str = "",
    row_group: str = "method_filter",
) -> Path:
    selected = [
        row
        for row in rows_for(summary_rows, evaluation, main_k)
        if not average_scope or clean(row.get("average_scope")) == average_scope
    ]
    if filename_prefix:
        metric_token = "f1" if metric == "f1_at_k" else metric
        path = output_dir / f"{filename_prefix}_heatmap_{metric_token}_at_k{main_k}_{evaluation}.png"
    else:
        path = output_dir / f"heatmap_{metric}_at_k{main_k}_{evaluation}.png"
    if not selected:
        return save_placeholder_chart(path, f"{metric} heatmap", f"No rows found for {evaluation}.")

    if row_group == "method_family":
        family_order = {"IR": 0, "CE": 1}
        row_labels = sorted(
            {clean(row.get("method_family")) for row in selected if clean(row.get("method_family")) in family_order},
            key=lambda label: family_order[label],
        )
    else:
        row_labels = sorted({row["method_filter"] for row in selected}, key=chart_sort_key)
    col_labels = sorted({row["scenario"] for row in selected}, key=scenario_sort_key)
    matrix = np.full((len(row_labels), len(col_labels)), math.nan)
    for row_index, row_label in enumerate(row_labels):
        for col_index, col_label in enumerate(col_labels):
            values = [
                float(row[metric])
                for row in selected
                if (
                    (clean(row.get("method_family")) if row_group == "method_family" else row["method_filter"])
                    == row_label
                )
                and row["scenario"] == col_label
            ]
            if values:
                matrix[row_index, col_index] = mean(values)

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.6), max(5, len(row_labels) * 0.55)))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#f2f2f2")
    finite_values = matrix[np.isfinite(matrix)]
    vmax = max(0.01, float(np.max(finite_values))) if finite_values.size else 1.0
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for row_index in range(len(row_labels)):
        for col_index in range(len(col_labels)):
            value = matrix[row_index, col_index]
            if not math.isnan(value):
                red, green, blue, _alpha = image.cmap(image.norm(value))
                luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                color = "white" if luminance < 0.5 else "black"
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=12,
                    fontweight="bold",
                )
    fig.colorbar(image, ax=ax, label=metric_display_name(metric))
    return save_figure(fig, path)
