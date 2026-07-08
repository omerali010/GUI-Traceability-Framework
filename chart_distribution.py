"""Distribution and histogram charts."""

from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from chart_common import save_figure, save_placeholder_chart


def plot_backward_top_requirement_distribution(length_rows: Sequence[Dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "backward_top_requirement_distribution.png"
    if not length_rows:
        return save_placeholder_chart(path, "Backward TopRequirements distribution", "No backward candidate rows were found.")

    values = [int(row["top_requirements_count"]) for row in length_rows]
    max_value = max(values) if values else 0
    bins = np.arange(0, max_value + 2) - 0.5 if max_value > 0 else np.arange(0, 2) - 0.5
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(values, bins=bins, edgecolor="white")
    ax.set_xlabel("Number of returned requirements")
    ax.set_ylabel("GUI candidate count")
    ax.set_xticks(range(0, max_value + 1))
    return save_figure(fig, path)

