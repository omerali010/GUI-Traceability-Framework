"""Generate the final thesis figures directly from metric CSV files.

The figures in ``Thesis Figures`` are the replication-package outputs. This
module recreates them from the CSV files in ``Thesis Figures/data`` and does
not copy images from ``OLD FIGURES``.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_THESIS_FIGURES_DIR = ROOT / "Thesis Figures"
OUTPUT_DIR = DEFAULT_THESIS_FIGURES_DIR
_OUTPUT_TARGETS: Dict[str, Tuple[Path, Path]] = {}

METHOD_ORDER = [
    "VSM",
    "LSI",
    "JSM",
    "Jina v3",
    "Qwen3 0.6B",
    "Qwen3 4B",
    "Stella 1.5B",
]

SCENARIO_ORDER = ["Focus-GPT5", "Focus-Gemma3", "LLMReq-GPT5", "LLMReq-Gemma3"]

FILTER_ORDER = ["Strict", "Relaxed"]
FILTER_COLORS = {
    "Strict": "#1f77b4",
    "Relaxed": "#d55e00",
}

FORWARD_CANDIDATE_SETS = {
    "actions_all": "Transitions",
    "widgets_all_resolved": "Widgets",
    "combined_actions_and_widgets": "Combined",
}

LINE_METRIC_FIELDS = [
    ("precision_at_k", "Precision"),
    ("recall_at_k", "Recall"),
    ("hit_at_k", "Hit@k"),
    ("mrr_at_k", "MRR"),
    ("f1_at_k", "F1"),
]

FAMILY_LINE_STYLES = {
    "IR": "-",
    "CE": "--",
}

def setup_matplotlib() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_explicit_figure(fig: plt.Figure, output_pdf: Path, output_png: Path) -> List[Path]:
    """Save a figure to the exact PDF and PNG paths requested by the caller."""
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    return [output_pdf, output_png]


def read_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / filename)


def sorted_methods_present(df: pd.DataFrame) -> List[str]:
    present = set(df["method"].dropna().astype(str))
    return [method for method in METHOD_ORDER if method in present]


def apply_filters(
    df: pd.DataFrame,
    filters: Sequence[Tuple[str, Callable[[pd.DataFrame], pd.Series], str]],
) -> Tuple[pd.DataFrame, List[str]]:
    filtered = df.copy()
    notes: List[str] = []
    for label, condition, reason in filters:
        before = len(filtered)
        filtered = filtered.loc[condition(filtered)].copy()
        notes.append(f"{label}: kept {len(filtered)} of {before} rows")
        if before != len(filtered):
            notes.append(f"dropped {before - len(filtered)} rows because {reason}")
    return filtered, notes


def clean_plot_rows(
    df: pd.DataFrame,
    value_columns: Sequence[str],
    optional_columns: Sequence[str] = (),
) -> Tuple[pd.DataFrame, List[str]]:
    cleaned = df.copy()
    notes: List[str] = []

    before = len(cleaned)
    cleaned = cleaned.loc[cleaned["method"].isin(METHOD_ORDER)].copy()
    dropped = before - len(cleaned)
    if dropped:
        notes.append(f"dropped {dropped} rows because method was not in the requested order")

    for column in value_columns:
        before = len(cleaned)
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned = cleaned.loc[cleaned[column].notna()].copy()
        dropped = before - len(cleaned)
        if dropped:
            notes.append(f"dropped {dropped} rows because {column} was missing or not numeric")

    for column in optional_columns:
        if column in cleaned.columns:
            before = len(cleaned)
            cleaned = cleaned.loc[cleaned[column].notna()].copy()
            dropped = before - len(cleaned)
            if dropped:
                notes.append(f"dropped {dropped} rows because {column} was missing")

    cleaned["method"] = pd.Categorical(cleaned["method"], METHOD_ORDER, ordered=True)
    cleaned = cleaned.sort_values(["method"]).copy()
    return cleaned, notes


def save_figure(fig: plt.Figure, output_stem: str) -> None:
    if output_stem in _OUTPUT_TARGETS:
        pdf_path, png_path = _OUTPUT_TARGETS[output_stem]
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        png_path = OUTPUT_DIR / f"{output_stem}.png"
        pdf_path = OUTPUT_DIR / f"{output_stem}.pdf"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def add_filter_legend(ax: plt.Axes, include_filter_variant: bool) -> None:
    if not include_filter_variant:
        return
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=FILTER_COLORS[variant],
            markeredgecolor="none",
            markersize=6,
            label=variant,
            alpha=0.8,
        )
        for variant in FILTER_ORDER
    ]
    ax.legend(handles=handles, title="Filter", frameon=True)


def plot_method_boxpoints(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    y_label: str,
    rng: np.random.Generator,
    include_filter_variant: bool = True,
) -> None:
    methods = sorted_methods_present(df)
    positions = np.arange(1, len(methods) + 1)
    values_by_method = [
        df.loc[df["method"].astype(str) == method, value_col].to_numpy(dtype=float)
        for method in methods
    ]

    ax.boxplot(
        values_by_method,
        positions=positions,
        widths=0.58,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#e6edf3", "edgecolor": "#4a5560", "linewidth": 1.0},
        medianprops={"color": "#111111", "linewidth": 1.5},
        whiskerprops={"color": "#4a5560", "linewidth": 1.0},
        capprops={"color": "#4a5560", "linewidth": 1.0},
    )

    if include_filter_variant and "filter_variant" in df.columns:
        offsets = {"Strict": -0.12, "Relaxed": 0.12}
        for index, method in enumerate(methods, start=1):
            method_rows = df.loc[df["method"].astype(str) == method]
            for variant in FILTER_ORDER:
                values = method_rows.loc[
                    method_rows["filter_variant"].astype(str) == variant, value_col
                ].to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                jitter = rng.normal(0, 0.035, len(values))
                alpha = 0.22 if len(df) > 10000 else 0.42
                ax.scatter(
                    np.full(len(values), index + offsets[variant]) + jitter,
                    values,
                    s=8,
                    color=FILTER_COLORS[variant],
                    alpha=alpha,
                    linewidths=0,
                    zorder=3,
                )
    else:
        for index, method in enumerate(methods, start=1):
            values = df.loc[df["method"].astype(str) == method, value_col].to_numpy(dtype=float)
            jitter = rng.normal(0, 0.06, len(values))
            alpha = 0.18 if len(df) > 10000 else 0.38
            ax.scatter(
                np.full(len(values), index) + jitter,
                values,
                s=8,
                color="#333333",
                alpha=alpha,
                linewidths=0,
                zorder=3,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_xlabel("Similarity method")
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    ax.margins(x=0.02)


def figure_a(data_dir: Path, rng: np.random.Generator) -> None:
    csv_file = "forward_per_requirement.csv"
    df = read_csv(data_dir, csv_file)
    filtered, _ = apply_filters(
        df,
        [
            (
                'evaluation == "combined_actions_and_widgets"',
                lambda frame: frame["evaluation"] == "combined_actions_and_widgets",
                "the figure uses the combined widget-and-transition candidate set",
            ),
            ("k == 10", lambda frame: frame["k"] == 10, "the figure uses F1@10"),
        ],
    )
    filtered, _ = clean_plot_rows(filtered, ["f1_at_k"])

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    plot_method_boxpoints(ax, filtered, "f1_at_k", "Forward F1@10", rng)
    ax.set_ylim(-0.02, 1.02)
    add_filter_legend(ax, True)
    fig.tight_layout()
    save_figure(fig, "fig_7_3_forward_f1_method_distribution")


def figure_b(data_dir: Path, rng: np.random.Generator) -> None:
    csv_file = "forward_per_requirement.csv"
    df = read_csv(data_dir, csv_file)
    wanted = list(FORWARD_CANDIDATE_SETS)
    filtered, _ = apply_filters(
        df,
        [
            ("k == 10", lambda frame: frame["k"] == 10, "the figure uses F1@10"),
            (
                "evaluation in actions_all, widgets_all_resolved, combined_actions_and_widgets",
                lambda frame: frame["evaluation"].isin(wanted),
                "only the three candidate-set views are plotted",
            ),
        ],
    )
    filtered, _ = clean_plot_rows(filtered, ["f1_at_k"], ["evaluation"])
    filtered["candidate_set"] = filtered["evaluation"].map(FORWARD_CANDIDATE_SETS)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3), sharey=True)
    for ax, evaluation in zip(axes, wanted):
        panel = filtered.loc[filtered["evaluation"] == evaluation].copy()
        plot_method_boxpoints(ax, panel, "f1_at_k", "F1@10", rng)
        ax.set_title(FORWARD_CANDIDATE_SETS[evaluation])
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("")
    add_filter_legend(axes[-1], True)
    fig.supylabel("Forward F1@10", x=0.01, fontsize=10)
    fig.tight_layout()
    save_figure(fig, "fig_7_3_forward_candidate_type_f1_by_method")

def method_filter_sort_key(label: str, df: pd.DataFrame) -> Tuple[int, int, str]:
    rows = df.loc[df["method_filter"].astype(str) == label]
    if rows.empty:
        return (99, 99, label)
    row = rows.iloc[0]
    method = str(row.get("method", ""))
    filter_variant = str(row.get("filter_variant", ""))
    return (
        METHOD_ORDER.index(method) if method in METHOD_ORDER else 99,
        FILTER_ORDER.index(filter_variant) if filter_variant in FILTER_ORDER else 99,
        label,
    )


def clean_line_chart_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    cleaned = df.copy()
    notes: List[str] = []

    if "method_filter" not in cleaned.columns:
        cleaned["method_filter"] = cleaned["method"].astype(str) + " " + cleaned["filter_variant"].astype(str)

    before = len(cleaned)
    cleaned = cleaned.loc[cleaned["method"].isin(METHOD_ORDER)].copy()
    if before != len(cleaned):
        notes.append(f"dropped {before - len(cleaned)} rows because method was not in the requested order")

    before = len(cleaned)
    cleaned["k"] = pd.to_numeric(cleaned["k"], errors="coerce")
    cleaned = cleaned.loc[cleaned["k"].notna()].copy()
    if before != len(cleaned):
        notes.append(f"dropped {before - len(cleaned)} rows because k was missing or not numeric")
    cleaned["k"] = cleaned["k"].astype(int)

    for metric, _ in LINE_METRIC_FIELDS:
        before = len(cleaned)
        cleaned[metric] = pd.to_numeric(cleaned[metric], errors="coerce")
        cleaned = cleaned.loc[cleaned[metric].notna()].copy()
        if before != len(cleaned):
            notes.append(f"dropped {before - len(cleaned)} rows because {metric} was missing or not numeric")

    return cleaned, notes


def figure_forward_line_family_style(data_dir: Path) -> None:
    csv_file = "forward_metrics_summary_by_k.csv"
    df = read_csv(data_dir, csv_file)
    filtered, _ = apply_filters(
        df,
        [
            (
                'direction == "forward"',
                lambda frame: frame["direction"] == "forward"
                if "direction" in frame.columns
                else pd.Series(True, index=frame.index),
                "the improved figure reproduces the forward metrics-over-k chart",
            ),
            (
                'average_scope == "gold"',
                lambda frame: frame["average_scope"] == "gold"
                if "average_scope" in frame.columns
                else pd.Series(True, index=frame.index),
                "the original forward summary uses the gold average scope",
            ),
            (
                'evaluation == "combined_actions_and_widgets"',
                lambda frame: frame["evaluation"] == "combined_actions_and_widgets",
                "the source chart uses the combined widget-and-transition candidate set",
            ),
        ],
    )
    filtered, _ = clean_line_chart_rows(filtered)

    labels = sorted(
        filtered["method_filter"].dropna().astype(str).unique().tolist(),
        key=lambda label: method_filter_sort_key(label, filtered),
    )
    k_values = sorted(filtered["k"].dropna().astype(int).unique().tolist())
    family_by_label = {
        label: str(filtered.loc[filtered["method_filter"].astype(str) == label, "method_family"].iloc[0])
        for label in labels
    }

    color_values = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(labels), 1)))
    color_by_label = {label: color_values[index] for index, label in enumerate(labels)}

    fig, axes = plt.subplots(1, len(LINE_METRIC_FIELDS), figsize=(19, 4.8), sharex=True)
    for ax, (metric, title) in zip(axes, LINE_METRIC_FIELDS):
        for label in labels:
            label_rows = filtered.loc[filtered["method_filter"].astype(str) == label]
            means = label_rows.groupby("k")[metric].mean()
            x_values = [k for k in k_values if k in means.index]
            y_values = [float(means.loc[k]) for k in x_values]
            if not x_values:
                continue
            family = family_by_label.get(label, "")
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.6,
                linestyle=FAMILY_LINE_STYLES.get(family, "-"),
                color=color_by_label[label],
                label=label,
            )
        ax.set_xlabel("k")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1)
        ax.set_xticks(k_values)
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_axisbelow(True)

    method_handles, method_labels = axes[0].get_legend_handles_labels()
    if method_handles:
        fig.legend(
            method_handles,
            method_labels,
            loc="lower center",
            ncol=min(4, len(method_labels)),
            bbox_to_anchor=(0.5, -0.04),
            frameon=True,
        )
    style_handles = [
        plt.Line2D([0], [0], color="#222222", linewidth=2.0, linestyle="-", label="IR method"),
        plt.Line2D([0], [0], color="#222222", linewidth=2.0, linestyle="--", label="CE method"),
    ]
    fig.legend(
        handles=style_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    save_figure(fig, "fig_7_3_forward_line_metrics_over_k_combined_actions_and_widgets_family_style")


def figure_e(data_dir: Path, rng: np.random.Generator) -> None:
    csv_file = "backward_per_gui_candidate.csv"
    df = read_csv(data_dir, csv_file)
    filtered, _ = apply_filters(
        df,
        [
            (
                'evaluation == "combined_actions_and_widgets"',
                lambda frame: frame["evaluation"] == "combined_actions_and_widgets",
                "the figure uses the combined widget-and-transition candidate set",
            ),
            ("k == 3", lambda frame: frame["k"] == 3, "the figure uses F1@3"),
        ],
    )
    filtered, _ = clean_plot_rows(filtered, ["f1_at_k"], ["candidate_type"])
    panel_specs = [
        ("Widget", filtered.loc[filtered["candidate_type"] == "Widget"].copy()),
        ("Transition", filtered.loc[filtered["candidate_type"] == "Transition"].copy()),
        ("Combined", filtered.copy()),
    ]
    panel_specs = [(label, panel) for label, panel in panel_specs if not panel.empty]

    fig, axes = plt.subplots(1, len(panel_specs), figsize=(15.5, 5.3), sharey=True)
    if len(panel_specs) == 1:
        axes = [axes]
    for ax, (panel_label, panel) in zip(axes, panel_specs):
        plot_method_boxpoints(ax, panel, "f1_at_k", "F1@3", rng)
        ax.set_title(panel_label)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("")
    add_filter_legend(axes[-1], True)
    fig.supylabel("Backward all-candidate F1@3", x=0.01, fontsize=10)
    fig.tight_layout()
    save_figure(fig, "fig_7_4_backward_candidate_type_f1_by_method")

def save_png_as_pdf(source_png: Path, destination_pdf: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to create PDF companions for PNG-only charts.") from exc

    destination_pdf.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_png) as image:
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            image_to_save = background
        else:
            image_to_save = image.convert("RGB")
        image_to_save.save(destination_pdf, "PDF", resolution=300.0)


def generated_png_with_pdf(png_path: Path, pdf_path: Path) -> List[Path]:
    save_png_as_pdf(png_path, pdf_path)
    return [pdf_path, png_path]


def plot_backward_coverage_hit_f1_by_method(csv_path: Path, output_pdf: Path, output_png: Path) -> List[Path]:
    from chart_common import apply_chart_style as _apply_chart_style, read_csv_rows
    from metrics_common import clean, safe_float

    rows = [
        row
        for row in read_csv_rows(csv_path)
        if clean(row.get("average_scope")) == "all"
        and clean(row.get("evaluation")) == "combined_actions_and_widgets"
        and int(row.get("k", 0)) == 3
    ]
    if not rows:
        raise ValueError(f"No backward all-candidate k=3 rows found in {csv_path}.")

    method_order = ["VSM", "LSI", "JSM", "Jina v3", "Qwen3 0.6B", "Qwen3 4B", "Stella 1.5B"]
    filter_styles = {
        "Strict": {"color": "#4C78A8", "marker": "o"},
        "Relaxed": {"color": "#F58518", "marker": "s"},
    }
    scenario_offsets = {
        "Focus-GPT5": -0.06,
        "Focus-Gemma3": -0.02,
        "LLMReq-GPT5": 0.02,
        "LLMReq-Gemma3": 0.06,
    }
    filter_offsets = {"Strict": -0.09, "Relaxed": 0.09}

    def coverage_ratio(row: Dict[str, Any]) -> float:
        total = safe_float(row.get("candidate_queries_total"))
        with_gold = safe_float(row.get("candidate_queries_with_gold"))
        return with_gold / total if total else 0.0

    metrics = [
        ("coverage", "Coverage", coverage_ratio),
        ("hit_at_k", "Hit@3", lambda row: safe_float(row.get("hit_at_k"))),
        ("f1_at_k", "F1@3", lambda row: safe_float(row.get("f1_at_k"))),
    ]

    _apply_chart_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.2), sharey=True)
    positions = np.arange(1, len(method_order) + 1)
    for ax, (_metric, title, value_func) in zip(axes, metrics):
        grouped_values = [
            [value_func(row) for row in rows if clean(row.get("method")) == method]
            for method in method_order
        ]
        box = ax.boxplot(
            grouped_values,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
        )
        for patch in box["boxes"]:
            patch.set_facecolor("#E6E6E6")
            patch.set_edgecolor("#555555")
            patch.set_alpha(0.55)
        for element in ["whiskers", "caps", "medians"]:
            for artist in box[element]:
                artist.set_color("#555555")
                artist.set_linewidth(0.9)

        for method_index, method in enumerate(method_order, start=1):
            for filter_variant, style in filter_styles.items():
                selected_rows = [
                    row
                    for row in rows
                    if clean(row.get("method")) == method and clean(row.get("filter_variant")) == filter_variant
                ]
                x_values = [
                    method_index
                    + filter_offsets[filter_variant]
                    + scenario_offsets.get(clean(row.get("scenario")), 0.0)
                    for row in selected_rows
                ]
                y_values = [value_func(row) for row in selected_rows]
                ax.scatter(
                    x_values,
                    y_values,
                    s=18,
                    alpha=0.78,
                    color=style["color"],
                    marker=style["marker"],
                    edgecolors="white",
                    linewidths=0.35,
                    label=filter_variant,
                    zorder=3,
                )

        ax.set_title(title)
        ax.set_xticks(positions)
        ax.set_xticklabels(method_order, rotation=35, ha="right")
        ax.set_xlabel("Similarity method")
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis="y", alpha=0.35)

    axes[0].set_ylabel("Ratio / score")
    handles, labels = axes[-1].get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)
    fig.legend(unique_handles, unique_labels, title="Filter", frameon=False, loc="lower center", ncol=2)
    fig.subplots_adjust(bottom=0.25, wspace=0.08)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [output_pdf, output_png]


def plot_candidate_space_composition(data_dir: Path, output_pdf: Path, output_png: Path) -> List[Path]:
    from chart_bar import (
        plot_candidate_space as _plot_candidate_space,
        read_candidate_space_chart_rows,
    )

    _plot_candidate_space(
        read_candidate_space_chart_rows(data_dir / "candidate_space_summary.csv"),
        output_pdf,
        output_png,
    )
    save_png_as_pdf(output_png, output_pdf)
    return [output_pdf, output_png]


def plot_backward_top_requirement_distribution_figure(data_dir: Path, output_pdf: Path, output_png: Path) -> List[Path]:
    from chart_common import apply_chart_style as _apply_chart_style, read_csv_rows
    from chart_distribution import plot_backward_top_requirement_distribution

    _apply_chart_style()
    png_path = plot_backward_top_requirement_distribution(
        read_csv_rows(data_dir / "backward_output_candidate_lengths.csv"),
        output_png.parent,
    )
    if png_path != output_png:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        png_path.replace(output_png)
    return generated_png_with_pdf(output_png, output_pdf)


def plot_f1_heatmap_figure(data_dir: Path, summary_csv: str, output_pdf: Path, output_png: Path, prefix: str, scope: str = "") -> List[Path]:
    from chart_common import apply_chart_style as _apply_chart_style, read_csv_rows
    from chart_heatmap import plot_heatmap

    _apply_chart_style()
    png_path = plot_heatmap(
        read_csv_rows(data_dir / summary_csv),
        10 if prefix == "forward" else 3,
        "combined_actions_and_widgets",
        "f1_at_k",
        output_png.parent,
        filename_prefix=prefix,
        average_scope=scope,
        row_group="method_filter",
    )
    if png_path != output_png:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        png_path.replace(output_png)
    return generated_png_with_pdf(output_png, output_pdf)


def plot_scalability_figure(data_dir: Path, output_pdf: Path, output_png: Path, kind: str) -> List[Path]:
    from chart_bar import plot_scalability_memory, plot_scalability_runtime
    from chart_common import apply_chart_style as _apply_chart_style, read_csv_rows

    _apply_chart_style()
    rows = read_csv_rows(data_dir / "scalability_runtime_memory.csv")
    if kind == "memory":
        png_path = plot_scalability_memory(rows, output_png.parent)
    else:
        png_path = plot_scalability_runtime(rows, output_png.parent)
    if png_path != output_png:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        png_path.replace(output_png)
    return generated_png_with_pdf(output_png, output_pdf)

def plot_backward_gt_candidate_type_f1(data_dir: Path, output_pdf: Path, output_png: Path) -> List[Path]:
    setup_matplotlib()
    data_path = data_dir / "backward_per_gui_candidate.csv"
    PANEL_SPECS = [
        ("Widget", "widgets_all_resolved", "Widget"),
        ("Transition", "actions_all", "Transition"),
        ("Combined", "combined_actions_and_widgets", None),
    ]

    def plot_panel(ax, panel_df, panel_label, rng):
        methods = [method for method in METHOD_ORDER if method in set(panel_df["method"])]
        positions = np.arange(1, len(methods) + 1)
        values_by_method = [
            panel_df.loc[panel_df["method"] == method, "f1_at_k"].to_numpy(dtype=float)
            for method in methods
        ]

        ax.boxplot(
            values_by_method,
            positions=positions,
            widths=0.58,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#e6edf3", "edgecolor": "#4a5560", "linewidth": 1.0},
            medianprops={"color": "#111111", "linewidth": 1.5},
            whiskerprops={"color": "#4a5560", "linewidth": 1.0},
            capprops={"color": "#4a5560", "linewidth": 1.0},
        )

        offsets = {"Strict": -0.12, "Relaxed": 0.12}
        for index, method in enumerate(methods, start=1):
            method_rows = panel_df.loc[panel_df["method"] == method]
            for variant in FILTER_ORDER:
                values = method_rows.loc[method_rows["filter_variant"] == variant, "f1_at_k"].to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                jitter = rng.normal(0, 0.035, len(values))
                ax.scatter(
                    np.full(len(values), index + offsets[variant]) + jitter,
                    values,
                    s=8,
                    color=FILTER_COLORS[variant],
                    alpha=0.24,
                    linewidths=0,
                    zorder=3,
                )

        ax.set_title(panel_label)
        ax.set_xticks(positions)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_xlabel("Similarity method")
        ax.set_ylabel("")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_axisbelow(True)
        ax.margins(x=0.02)


    df = pd.read_csv(
        data_path,
        usecols=[
            "method",
            "filter_variant",
            "scenario",
            "candidate_type",
            "evaluation",
            "k",
            "ground_truth_requirements",
            "f1_at_k",
        ],
    )
    df = df.loc[
        (df["k"] == 3)
        & (df["ground_truth_requirements"] > 0)
        & (df["method"].isin(METHOD_ORDER))
    ].copy()
    df["f1_at_k"] = pd.to_numeric(df["f1_at_k"], errors="coerce")
    df = df.loc[df["f1_at_k"].notna()].copy()

    plot_frames = []
    for panel_label, evaluation, candidate_type in PANEL_SPECS:
        panel = df.loc[df["evaluation"] == evaluation].copy()
        if candidate_type is not None:
            panel = panel.loc[panel["candidate_type"] == candidate_type].copy()
        panel["candidate_set"] = panel_label
        plot_frames.append(panel)
    plot_df = pd.concat(plot_frames, ignore_index=True)
    plot_df["method"] = pd.Categorical(plot_df["method"], METHOD_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["candidate_set", "method", "filter_variant", "scenario"]).copy()

    rng = np.random.default_rng(202607)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3), sharey=True)
    for ax, (panel_label, _, _) in zip(axes, PANEL_SPECS):
        panel_df = plot_df.loc[plot_df["candidate_set"] == panel_label].copy()
        plot_panel(ax, panel_df, panel_label, rng)

    add_filter_legend(axes[-1], True)
    fig.supylabel("Backward GT-only F1@3", x=0.01, fontsize=10)
    fig.tight_layout()

    pdf_path, png_path = save_explicit_figure(fig, output_pdf, output_png)

    print(f"CSV file used: {data_path}")
    print("One point means: one raw GUI-candidate-level backward ranking result at k=3 for one scenario, filter variant, and candidate set, restricted to candidates with ground_truth_requirements > 0.")
    print("No scenario/filter aggregation is applied before plotting; zero-valued F1 rows among GT-linked candidates are retained.")
    print("Panel filters: Widget=widgets_all_resolved + Widget rows; Transition=actions_all + Transition rows; Combined=combined_actions_and_widgets with Widget and Transition rows pooled.")
    print(f"Plotted rows: {len(plot_df)}")
    print("Counts per method and panel:")
    counts = (
        plot_df.groupby(["candidate_set", "method"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=[label for label, _, _ in PANEL_SPECS], columns=METHOD_ORDER, fill_value=0)
    )
    print(counts.to_string())
    print(f"Value range: {plot_df['f1_at_k'].min()} to {plot_df['f1_at_k'].max()}")
    print(f"Zero-valued GT-linked rows retained: {int((plot_df['f1_at_k'] == 0).sum())}")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    return [pdf_path, png_path]


def plot_forward_backward_summary(
    data_dir: Path,
    output_pdf: Path,
    output_png: Path,
    backward_csv: str,
    backward_label: str,
    backward_filter_text: str,
) -> List[Path]:
    setup_matplotlib()
    directions = [
        ("Forward F1@10", "#1f77b4", -0.18),
        (backward_label, "#d55e00", 0.18),
    ]

    forward = pd.read_csv(data_dir / "forward_summary.csv")
    forward = forward.loc[
        (forward["evaluation"] == "combined_actions_and_widgets")
        & (forward["k"] == 10)
        & (forward["method"].isin(METHOD_ORDER))
    ].copy()
    forward["direction"] = "Forward F1@10"

    backward = pd.read_csv(data_dir / backward_csv)
    backward = backward.loc[
        (backward["evaluation"] == "combined_actions_and_widgets")
        & (backward["k"] == 3)
        & (backward["method"].isin(METHOD_ORDER))
    ].copy()
    backward["direction"] = backward_label

    plot_df = pd.concat(
        [
            forward[["method", "filter_variant", "scenario", "direction", "f1_at_k"]],
            backward[["method", "filter_variant", "scenario", "direction", "f1_at_k"]],
        ],
        ignore_index=True,
    )
    plot_df["f1_at_k"] = pd.to_numeric(plot_df["f1_at_k"], errors="coerce")
    plot_df = plot_df.loc[plot_df["f1_at_k"].notna()].copy()
    plot_df["method"] = pd.Categorical(plot_df["method"], METHOD_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["method", "direction", "filter_variant", "scenario"]).copy()

    fig, ax = plt.subplots(figsize=(10.8, 5.3))
    rng = np.random.default_rng(202607)
    centers = np.arange(1, len(METHOD_ORDER) + 1)

    for direction_label, color, offset in directions:
        direction_rows = plot_df.loc[plot_df["direction"] == direction_label]
        values_by_method = [
            direction_rows.loc[direction_rows["method"] == method, "f1_at_k"].to_numpy(dtype=float)
            for method in METHOD_ORDER
        ]
        positions = centers + offset
        ax.boxplot(
            values_by_method,
            positions=positions,
            widths=0.26,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": color, "edgecolor": "#4a5560", "linewidth": 1.0, "alpha": 0.22},
            medianprops={"color": "#111111", "linewidth": 1.5},
            whiskerprops={"color": "#4a5560", "linewidth": 1.0},
            capprops={"color": "#4a5560", "linewidth": 1.0},
        )
        for index, method in enumerate(METHOD_ORDER):
            values = values_by_method[index]
            if len(values) == 0:
                continue
            jitter = rng.normal(0, 0.025, len(values))
            ax.scatter(
                np.full(len(values), positions[index]) + jitter,
                values,
                s=22,
                color=color,
                alpha=0.82,
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=7,
            label=label,
        )
        for label, color, _ in directions
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True)
    ax.set_xticks(centers)
    ax.set_xticklabels(METHOD_ORDER, rotation=30, ha="right")
    ax.set_xlabel("Similarity method")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 0.30)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()

    pdf_path, png_path = save_explicit_figure(fig, output_pdf, output_png)

    print("CSV files used:")
    print(f"- {data_dir / 'forward_summary.csv'}")
    print(f"- {data_dir / backward_csv}")
    print("Filters:")
    print("- Forward: evaluation == combined_actions_and_widgets, k == 10")
    print(f"- {backward_filter_text}: evaluation == combined_actions_and_widgets, k == 3")
    print("One point = one generated model scenario and filtering variant for one method and direction.")
    print("No method-family aggregation is used.")
    print("Counts per method and direction:")
    counts = (
        plot_df.groupby(["direction", "method"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=[label for label, _, _ in directions], columns=METHOD_ORDER, fill_value=0)
    )
    print(counts.to_string())
    print(f"F1 range: {plot_df['f1_at_k'].min()} to {plot_df['f1_at_k'].max()}")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    return [pdf_path, png_path]


def plot_forward_backward_summary_all(data_dir: Path, output_pdf: Path, output_png: Path) -> List[Path]:
    return plot_forward_backward_summary(
        data_dir=data_dir,
        output_pdf=output_pdf,
        output_png=output_png,
        backward_csv="backward_all_candidate_summary.csv",
        backward_label="Backward all-candidate F1@3",
        backward_filter_text="Backward",
    )


def plot_forward_backward_summary_gt_only(data_dir: Path, output_pdf: Path, output_png: Path) -> List[Path]:
    return plot_forward_backward_summary(
        data_dir=data_dir,
        output_pdf=output_pdf,
        output_png=output_png,
        backward_csv="backward_gold_only_diagnostic_summary.csv",
        backward_label="Backward GT-only F1@3",
        backward_filter_text="Backward GT-only",
    )


def plot_filtering_delta_f1(data_dir: Path, output_pdf: Path, output_png: Path) -> List[Path]:
    setup_matplotlib()
    data_path = data_dir / "filtering_delta_summary.csv"
    PANEL_SPECS = [
        ("Forward@10", "forward", 10, "#1f77b4"),
        ("Backward@3", "backward_all_candidate", 3, "#d55e00"),
    ]
    SCENARIO_MARKERS = {
        "Focus-GPT5": "o",
        "Focus-Gemma3": "s",
        "LLMReq-GPT5": "^",
        "LLMReq-Gemma3": "D",
    }

    raw = pd.read_csv(data_path)
    raw["computed_delta"] = raw["relaxed_f1_at_k"] - raw["strict_f1_at_k"]
    raw["delta_f1"] = raw["delta_f1_relaxed_minus_strict"].fillna(raw["computed_delta"])
    frames = []
    for panel_label, direction, k_value, color in PANEL_SPECS:
        panel = raw.loc[
            (raw["direction"] == direction)
            & (raw["k"] == k_value)
            & (raw["evaluation"] == "combined_actions_and_widgets")
            & (raw["method"].isin(METHOD_ORDER))
        ].copy()
        panel["panel"] = panel_label
        panel["panel_color"] = color
        frames.append(panel)
    plot_df = pd.concat(frames, ignore_index=True)
    plot_df["delta_f1"] = pd.to_numeric(plot_df["delta_f1"], errors="coerce")
    plot_df = plot_df.loc[plot_df["delta_f1"].notna()].copy()
    plot_df["method"] = pd.Categorical(plot_df["method"], METHOD_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["panel", "method", "scenario"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True)
    rng = np.random.default_rng(202607)
    positions = np.arange(1, len(METHOD_ORDER) + 1)
    for ax, (panel_label, _, _, color) in zip(axes, PANEL_SPECS):
        panel_df = plot_df.loc[plot_df["panel"] == panel_label].copy()
        ax.axhline(0, color="#222222", linewidth=1.0, linestyle="--", alpha=0.9, zorder=1)
        for index, method in enumerate(METHOD_ORDER, start=1):
            method_rows = panel_df.loc[panel_df["method"] == method]
            if method_rows.empty:
                continue
            for scenario in sorted(method_rows["scenario"].unique()):
                scenario_rows = method_rows.loc[method_rows["scenario"] == scenario]
                jitter = rng.normal(0, 0.035, len(scenario_rows))
                ax.scatter(
                    np.full(len(scenario_rows), index) + jitter,
                    scenario_rows["delta_f1"].to_numpy(dtype=float),
                    s=28,
                    marker=SCENARIO_MARKERS.get(scenario, "o"),
                    color=color,
                    alpha=0.82,
                    edgecolors="white",
                    linewidths=0.35,
                    zorder=3,
                )
            ax.scatter(
                [index],
                [method_rows["delta_f1"].mean()],
                s=45,
                marker="D",
                color="#111111",
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
            )
        ax.set_title(panel_label)
        ax.set_xticks(positions)
        ax.set_xticklabels(METHOD_ORDER, rotation=30, ha="right")
        ax.set_xlabel("Similarity method")
        ax.set_ylabel("" if ax is not axes[0] else r"$\Delta$F1 = Relaxed F1 - Strict F1")
        ax.set_ylim(-0.10, 0.04)
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_axisbelow(True)

    scenario_handles = [
        plt.Line2D([0], [0], marker=marker, color="none", markerfacecolor="#666666", markeredgecolor="white", markeredgewidth=0.35, markersize=6, label=scenario)
        for scenario, marker in SCENARIO_MARKERS.items()
    ]
    mean_handle = plt.Line2D([0], [0], marker="D", color="none", markerfacecolor="#111111", markeredgecolor="white", markeredgewidth=0.5, markersize=6, label="Method mean")
    fig.legend(handles=scenario_handles + [mean_handle], loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03), frameon=True)
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    pdf_path, png_path = save_explicit_figure(fig, output_pdf, output_png)

    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print("Counts per method and panel:")
    print(plot_df.groupby(["panel", "method"], observed=False).size().unstack(fill_value=0).reindex(index=[label for label, _, _, _ in PANEL_SPECS], columns=METHOD_ORDER, fill_value=0).to_string())
    print(f"Delta F1 range: {plot_df['delta_f1'].min()} to {plot_df['delta_f1'].max()}")
    return [pdf_path, png_path]


def plot_query_level_filtering_delta_f1(data_dir: Path, output_pdf: Path, output_png: Path) -> List[Path]:
    setup_matplotlib()
    PANEL_SPECS = [
        ("forward", r"Forward $\Delta$F1@10", "Forward", data_dir / "forward_metrics_per_requirement_by_k.csv", 10, "requirement"),
        ("backward", r"Backward $\Delta$F1@3", "Backward", data_dir / "backward_metrics_per_gui_candidate_by_k.csv", 3, "gui_candidate"),
    ]
    USECOLS_BY_DIRECTION = {
        "Forward": ["method", "filter_variant", "scenario", "evaluation", "query_id", "requirement_id", "k", "f1_at_k"],
        "Backward": ["method", "filter_variant", "scenario", "evaluation", "query_id", "candidate_key", "candidate_type", "k", "f1_at_k"],
    }

    def make_delta_rows(panel_key, panel_title, direction_label, path, k_value, query_unit):
        df = pd.read_csv(path, usecols=USECOLS_BY_DIRECTION[direction_label])
        df = df.loc[
            (df["evaluation"] == "combined_actions_and_widgets")
            & (df["k"] == k_value)
            & (df["method"].isin(METHOD_ORDER))
        ].copy()
        df["f1_at_k"] = pd.to_numeric(df["f1_at_k"], errors="coerce")
        df = df.loc[df["f1_at_k"].notna()].copy()

        pair_cols = ["method", "scenario", "evaluation", "k", "query_id"]
        strict = df.loc[df["filter_variant"] == "Strict", pair_cols + ["f1_at_k"]].copy()
        relaxed = df.loc[df["filter_variant"] == "Relaxed", pair_cols + ["f1_at_k"]].copy()

        strict_keys = set(map(tuple, strict[pair_cols].astype(str).values.tolist()))
        relaxed_keys = set(map(tuple, relaxed[pair_cols].astype(str).values.tolist()))

        merged = strict.merge(relaxed, on=pair_cols, suffixes=("_strict", "_relaxed"))
        merged["delta_f1"] = merged["f1_at_k_relaxed"] - merged["f1_at_k_strict"]
        merged["panel_key"] = panel_key
        merged["panel_title"] = panel_title
        merged["direction"] = direction_label
        merged["query_unit"] = query_unit

        stats = {
            "panel": panel_key,
            "path": path,
            "input_rows": len(df),
            "strict_rows": len(strict),
            "relaxed_rows": len(relaxed),
            "matched_pairs": len(merged),
            "strict_only": len(strict_keys - relaxed_keys),
            "relaxed_only": len(relaxed_keys - strict_keys),
        }
        return merged, stats

    frames = []
    stats_rows = []
    for spec in PANEL_SPECS:
        merged, stats = make_delta_rows(*spec)
        frames.append(merged)
        stats_rows.append(stats)
    plot_df = pd.concat(frames, ignore_index=True)
    plot_df["method"] = pd.Categorical(plot_df["method"], METHOD_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["panel_key", "method", "scenario", "query_id"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9), sharey=True)
    rng = np.random.default_rng(202607)
    positions = np.arange(1, len(METHOD_ORDER) + 1)
    for ax, (panel_key, panel_title, *_rest) in zip(axes, PANEL_SPECS):
        panel_df = plot_df.loc[plot_df["panel_key"] == panel_key].copy()
        values_by_method = [panel_df.loc[panel_df["method"] == method, "delta_f1"].to_numpy(dtype=float) for method in METHOD_ORDER]
        ax.boxplot(
            values_by_method,
            positions=positions,
            widths=0.58,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#e6edf3", "edgecolor": "#4a5560", "linewidth": 1.0, "alpha": 0.72},
            medianprops={"color": "#111111", "linewidth": 1.4},
            whiskerprops={"color": "#4a5560", "linewidth": 0.9},
            capprops={"color": "#4a5560", "linewidth": 0.9},
        )
        ax.axhline(0, color="#222222", linewidth=1.0, linestyle="--", alpha=0.9, zorder=1)
        for index, method in enumerate(METHOD_ORDER, start=1):
            values = panel_df.loc[panel_df["method"] == method, "delta_f1"].to_numpy(dtype=float)
            if len(values) == 0:
                continue
            jitter = rng.normal(0, 0.055, len(values))
            alpha = 0.12 if len(values) > 1000 else 0.24
            ax.scatter(np.full(len(values), index) + jitter, values, s=7, color="#2f4b7c", alpha=alpha, linewidths=0, zorder=3)
        ax.set_title(panel_title)
        ax.set_xticks(positions)
        ax.set_xticklabels(METHOD_ORDER, rotation=30, ha="right")
        ax.set_xlabel("Similarity method")
        ax.set_ylabel("" if ax is not axes[0] else r"$\Delta$F1 = relaxed - strict")
        ax.set_ylim(-0.9, 0.9)
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_axisbelow(True)
        ax.margins(x=0.02)

    fig.tight_layout()
    pdf_path, png_path = save_explicit_figure(fig, output_pdf, output_png)

    print("Generated query-level filtering-effect figure")
    for stats in stats_rows:
        print(f"{stats['panel']}: matched={stats['matched_pairs']}, strict_only={stats['strict_only']}, relaxed_only={stats['relaxed_only']}")
    counts = plot_df.groupby(["panel_key", "method"], observed=False).size().unstack(fill_value=0).reindex(index=[key for key, *_ in PANEL_SPECS], columns=METHOD_ORDER, fill_value=0)
    print(counts.to_string())
    print(f"Delta F1 range: {plot_df['delta_f1'].min()} to {plot_df['delta_f1'].max()}")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    return [pdf_path, png_path]


def plot_model_variant_f1_distribution_detailed(data_dir: Path, output_pdf: Path, output_png: Path) -> List[Path]:
    setup_matplotlib()
    FILTER_MARKERS = {"Strict": "o", "Relaxed": "s"}
    METHOD_COLORS = {
        "VSM": "#1f77b4",
        "LSI": "#ff7f0e",
        "JSM": "#2ca02c",
        "Jina v3": "#d62728",
        "Qwen3 0.6B": "#9467bd",
        "Qwen3 4B": "#8c564b",
        "Stella 1.5B": "#e377c2",
    }
    PANEL_SPECS = [
        ("Forward F1@10", data_dir / "forward_summary.csv", 10),
        ("Backward all-candidate F1@3", data_dir / "backward_all_candidate_summary.csv", 3),
    ]

    frames = []
    for panel_label, csv_path, k_value in PANEL_SPECS:
        df = pd.read_csv(csv_path)
        df = df.loc[
            (df["evaluation"] == "combined_actions_and_widgets")
            & (df["k"] == k_value)
            & (df["method"].isin(METHOD_ORDER))
            & (df["scenario"].isin(SCENARIO_ORDER))
        ].copy()
        df["panel"] = panel_label
        df["f1_at_k"] = pd.to_numeric(df["f1_at_k"], errors="coerce")
        df = df.loc[df["f1_at_k"].notna()].copy()
        frames.append(df[["panel", "scenario", "method", "filter_variant", "f1_at_k"]])
    plot_df = pd.concat(frames, ignore_index=True)
    plot_df["scenario"] = pd.Categorical(plot_df["scenario"], SCENARIO_ORDER, ordered=True)
    plot_df["method"] = pd.Categorical(plot_df["method"], METHOD_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["panel", "scenario", "method", "filter_variant"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.6), sharey=True)
    rng = np.random.default_rng(202607)
    positions = np.arange(1, len(SCENARIO_ORDER) + 1)

    for ax, (panel_label, _csv_path, _k_value) in zip(axes, PANEL_SPECS):
        panel_df = plot_df.loc[plot_df["panel"] == panel_label].copy()
        values_by_scenario = [
            panel_df.loc[panel_df["scenario"] == scenario, "f1_at_k"].to_numpy(dtype=float)
            for scenario in SCENARIO_ORDER
        ]
        ax.boxplot(
            values_by_scenario,
            positions=positions,
            widths=0.56,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#e6edf3", "edgecolor": "#4a5560", "linewidth": 1.0, "alpha": 0.78},
            medianprops={"color": "#111111", "linewidth": 1.5},
            whiskerprops={"color": "#4a5560", "linewidth": 1.0},
            capprops={"color": "#4a5560", "linewidth": 1.0},
        )
        for index, scenario in enumerate(SCENARIO_ORDER, start=1):
            scenario_rows = panel_df.loc[panel_df["scenario"] == scenario]
            for method in METHOD_ORDER:
                method_rows = scenario_rows.loc[scenario_rows["method"] == method]
                for filter_variant, marker in FILTER_MARKERS.items():
                    row_values = method_rows.loc[method_rows["filter_variant"] == filter_variant, "f1_at_k"].to_numpy(dtype=float)
                    if len(row_values) == 0:
                        continue
                    jitter = rng.normal(0, 0.045, len(row_values))
                    ax.scatter(
                        np.full(len(row_values), index) + jitter,
                        row_values,
                        s=34,
                        marker=marker,
                        color=METHOD_COLORS[method],
                        alpha=0.86,
                        edgecolors="white",
                        linewidths=0.35,
                        zorder=3,
                    )
        ax.set_title(panel_label)
        ax.set_xticks(positions)
        ax.set_xticklabels(SCENARIO_ORDER, rotation=25, ha="right")
        ax.set_xlabel("Generated model variant")
        ax.set_ylabel("F1 score" if ax is axes[0] else "")
        ax.set_ylim(0, 0.30)
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_axisbelow(True)

    method_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=7,
            label=method,
        )
        for method in METHOD_ORDER
    ]
    filter_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="#333333",
            markerfacecolor="#333333",
            markeredgecolor="white",
            markeredgewidth=0.35,
            linestyle="None",
            markersize=7,
            label=filter_variant,
        )
        for filter_variant, marker in FILTER_MARKERS.items()
    ]
    legend1 = fig.legend(
        handles=method_handles,
        title="Similarity method",
        loc="lower center",
        ncol=7,
        bbox_to_anchor=(0.5, 0.035),
        frameon=True,
    )
    fig.add_artist(legend1)
    fig.legend(
        handles=filter_handles,
        title="Filtering variant",
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.035),
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1))

    pdf_path, png_path = save_explicit_figure(fig, output_pdf, output_png)

    print("Generated detailed model-variant F1 distribution figure")
    print("One point = one individual similarity method and filtering-variant combination inside one generated model variant.")
    print("No method-family aggregation; Strict and Relaxed remain separate points.")
    print("Counts per generated model variant and panel:")
    counts = plot_df.groupby(["panel", "scenario"], observed=False).size().unstack(fill_value=0).reindex(index=[label for label, *_ in PANEL_SPECS], columns=SCENARIO_ORDER)
    print(counts.to_string())
    print("Counts per method and filtering variant:")
    print(plot_df.groupby(["method", "filter_variant"], observed=False).size().unstack(fill_value=0).reindex(index=METHOD_ORDER).to_string())
    print(f"F1 range: {plot_df['f1_at_k'].min()} to {plot_df['f1_at_k'].max()}")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    return [pdf_path, png_path]


def plot_model_variant_f1_distribution_detailed_gt_only(data_dir: Path, output_pdf: Path, output_png: Path) -> List[Path]:
    setup_matplotlib()
    FILTER_MARKERS = {"Strict": "o", "Relaxed": "s"}
    METHOD_COLORS = {
        "VSM": "#1f77b4",
        "LSI": "#ff7f0e",
        "JSM": "#2ca02c",
        "Jina v3": "#d62728",
        "Qwen3 0.6B": "#9467bd",
        "Qwen3 4B": "#8c564b",
        "Stella 1.5B": "#e377c2",
    }
    PANEL_SPECS = [
        ("Forward F1@10", data_dir / "forward_summary.csv", 10),
        ("Backward GT-only F1@3", data_dir / "backward_gold_only_diagnostic_summary.csv", 3),
    ]

    frames = []
    for panel_label, csv_path, k_value in PANEL_SPECS:
        df = pd.read_csv(csv_path)
        df = df.loc[
            (df["evaluation"] == "combined_actions_and_widgets")
            & (df["k"] == k_value)
            & (df["method"].isin(METHOD_ORDER))
            & (df["scenario"].isin(SCENARIO_ORDER))
        ].copy()
        df["panel"] = panel_label
        df["f1_at_k"] = pd.to_numeric(df["f1_at_k"], errors="coerce")
        df = df.loc[df["f1_at_k"].notna()].copy()
        frames.append(df[["panel", "scenario", "method", "filter_variant", "f1_at_k"]])
    plot_df = pd.concat(frames, ignore_index=True)
    plot_df["scenario"] = pd.Categorical(plot_df["scenario"], SCENARIO_ORDER, ordered=True)
    plot_df["method"] = pd.Categorical(plot_df["method"], METHOD_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["panel", "scenario", "method", "filter_variant"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.6), sharey=True)
    rng = np.random.default_rng(202607)
    positions = np.arange(1, len(SCENARIO_ORDER) + 1)

    for ax, (panel_label, _csv_path, _k_value) in zip(axes, PANEL_SPECS):
        panel_df = plot_df.loc[plot_df["panel"] == panel_label].copy()
        values_by_scenario = [
            panel_df.loc[panel_df["scenario"] == scenario, "f1_at_k"].to_numpy(dtype=float)
            for scenario in SCENARIO_ORDER
        ]
        ax.boxplot(
            values_by_scenario,
            positions=positions,
            widths=0.56,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#e6edf3", "edgecolor": "#4a5560", "linewidth": 1.0, "alpha": 0.78},
            medianprops={"color": "#111111", "linewidth": 1.5},
            whiskerprops={"color": "#4a5560", "linewidth": 1.0},
            capprops={"color": "#4a5560", "linewidth": 1.0},
        )
        for index, scenario in enumerate(SCENARIO_ORDER, start=1):
            scenario_rows = panel_df.loc[panel_df["scenario"] == scenario]
            for method in METHOD_ORDER:
                method_rows = scenario_rows.loc[scenario_rows["method"] == method]
                for filter_variant, marker in FILTER_MARKERS.items():
                    row_values = method_rows.loc[method_rows["filter_variant"] == filter_variant, "f1_at_k"].to_numpy(dtype=float)
                    if len(row_values) == 0:
                        continue
                    jitter = rng.normal(0, 0.045, len(row_values))
                    ax.scatter(
                        np.full(len(row_values), index) + jitter,
                        row_values,
                        s=34,
                        marker=marker,
                        color=METHOD_COLORS[method],
                        alpha=0.86,
                        edgecolors="white",
                        linewidths=0.35,
                        zorder=3,
                    )
        ax.set_title(panel_label)
        ax.set_xticks(positions)
        ax.set_xticklabels(SCENARIO_ORDER, rotation=25, ha="right")
        ax.set_xlabel("Generated model variant")
        ax.set_ylabel("F1 score" if ax is axes[0] else "")
        ax.set_ylim(0, 0.30)
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_axisbelow(True)

    method_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=7,
            label=method,
        )
        for method in METHOD_ORDER
    ]
    filter_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="#333333",
            markerfacecolor="#333333",
            markeredgecolor="white",
            markeredgewidth=0.35,
            linestyle="None",
            markersize=7,
            label=filter_variant,
        )
        for filter_variant, marker in FILTER_MARKERS.items()
    ]
    legend1 = fig.legend(
        handles=method_handles,
        title="Similarity method",
        loc="lower center",
        ncol=7,
        bbox_to_anchor=(0.5, 0.035),
        frameon=True,
    )
    fig.add_artist(legend1)
    fig.legend(
        handles=filter_handles,
        title="Filtering variant",
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.035),
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1))

    pdf_path, png_path = save_explicit_figure(fig, output_pdf, output_png)

    print("Generated detailed ground-truth-scope model-variant F1 distribution figure")
    print("Forward uses forward_summary.csv at k=10; backward uses backward_gold_only_diagnostic_summary.csv at k=3.")
    print("One point = one individual similarity method and filtering-variant combination inside one generated model variant.")
    print("No method-family aggregation; Strict and Relaxed remain separate points.")
    print("Counts per generated model variant and panel:")
    counts = plot_df.groupby(["panel", "scenario"], observed=False).size().unstack(fill_value=0).reindex(index=[label for label, *_ in PANEL_SPECS], columns=SCENARIO_ORDER)
    print(counts.to_string())
    print(f"F1 range: {plot_df['f1_at_k'].min()} to {plot_df['f1_at_k'].max()}")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    return [pdf_path, png_path]


def plot_model_variant_query_mean_f1_distribution(data_dir: Path, output_pdf: Path, output_png: Path) -> List[Path]:
    setup_matplotlib()
    PANEL_SPECS = [
        (
            "Forward query-mean F1@10",
            data_dir / "forward_metrics_per_requirement_by_k.csv",
            10,
            "requirement",
            "#1f77b4",
        ),
        (
            "Backward query-mean F1@3",
            data_dir / "backward_metrics_per_gui_candidate_by_k.csv",
            3,
            "gui_candidate",
            "#d55e00",
        ),
    ]

    frames = []
    stats = []
    for panel_label, csv_path, k_value, query_unit, color in PANEL_SPECS:
        raw = pd.read_csv(
            csv_path,
            usecols=["method", "filter_variant", "scenario", "evaluation", "query_id", "k", "f1_at_k"],
        )
        raw = raw.loc[
            (raw["evaluation"] == "combined_actions_and_widgets")
            & (raw["k"] == k_value)
            & (raw["scenario"].isin(SCENARIO_ORDER))
            & (raw["method"].isin(METHOD_ORDER))
        ].copy()
        raw["f1_at_k"] = pd.to_numeric(raw["f1_at_k"], errors="coerce")
        raw = raw.loc[raw["f1_at_k"].notna()].copy()

        grouped = (
            raw.groupby(["scenario", "query_id"], as_index=False)
            .agg(mean_f1=("f1_at_k", "mean"), contributing_rows=("f1_at_k", "size"))
        )
        grouped["panel"] = panel_label
        grouped["query_unit"] = query_unit
        grouped["color"] = color
        frames.append(grouped)
        stats.append((panel_label, csv_path, len(raw), grouped.copy()))

    plot_df = pd.concat(frames, ignore_index=True)
    plot_df["scenario"] = pd.Categorical(plot_df["scenario"], SCENARIO_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["panel", "scenario", "query_id"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.1), sharey=True)
    rng = np.random.default_rng(202607)
    positions = np.arange(1, len(SCENARIO_ORDER) + 1)

    for ax, (panel_label, _csv_path, _k_value, _query_unit, color) in zip(axes, PANEL_SPECS):
        panel_df = plot_df.loc[plot_df["panel"] == panel_label].copy()
        values_by_scenario = [
            panel_df.loc[panel_df["scenario"] == scenario, "mean_f1"].to_numpy(dtype=float)
            for scenario in SCENARIO_ORDER
        ]
        ax.boxplot(
            values_by_scenario,
            positions=positions,
            widths=0.56,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#e6edf3", "edgecolor": "#4a5560", "linewidth": 1.0, "alpha": 0.78},
            medianprops={"color": "#111111", "linewidth": 1.5},
            whiskerprops={"color": "#4a5560", "linewidth": 1.0},
            capprops={"color": "#4a5560", "linewidth": 1.0},
        )
        for index, scenario in enumerate(SCENARIO_ORDER, start=1):
            values = panel_df.loc[panel_df["scenario"] == scenario, "mean_f1"].to_numpy(dtype=float)
            if len(values) == 0:
                continue
            jitter = rng.normal(0, 0.055, len(values))
            alpha = 0.12 if len(values) > 500 else 0.34
            ax.scatter(
                np.full(len(values), index) + jitter,
                values,
                s=8,
                color=color,
                alpha=alpha,
                linewidths=0,
                zorder=3,
            )
        ax.set_title(panel_label)
        ax.set_xticks(positions)
        ax.set_xticklabels(SCENARIO_ORDER, rotation=25, ha="right")
        ax.set_xlabel("Generated model variant")
        ax.set_ylabel("Mean F1 score" if ax is axes[0] else "")
        ax.set_ylim(0, 0.80)
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_axisbelow(True)

    fig.tight_layout()
    pdf_path, png_path = save_explicit_figure(fig, output_pdf, output_png)

    print("Generated model-variant query-mean F1 distribution chart")
    print("No legend included, as requested.")
    print("Forward point = mean F1@10 for one requirement query within one generated model variant, averaged over available method/filter rows.")
    print("Backward point = mean F1@3 for one GUI model element query within one generated model variant, averaged over available method/filter rows.")
    for panel_label, csv_path, raw_count, grouped in stats:
        print(f"\n{panel_label}")
        print(f"CSV: {csv_path}")
        print(f"raw rows used: {raw_count}")
        print(f"query-mean points: {len(grouped)}")
        print("points per generated model variant:")
        print(grouped.groupby("scenario").size().reindex(SCENARIO_ORDER).to_string())
        print(f"mean F1 range: {grouped['mean_f1'].min()} to {grouped['mean_f1'].max()}")
        print(f"contributing rows per point: min={grouped['contributing_rows'].min()}, max={grouped['contributing_rows'].max()}, mean={grouped['contributing_rows'].mean():.2f}")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    return [pdf_path, png_path]


def plot_model_variant_query_mean_f1_distribution_gt_only(data_dir: Path, output_pdf: Path, output_png: Path) -> List[Path]:
    setup_matplotlib()
    PANEL_SPECS = [
        (
            "Forward query-mean F1@10",
            data_dir / "forward_metrics_per_requirement_by_k.csv",
            10,
            "requirement",
            "#1f77b4",
        ),
        (
            "Backward GT-only query-mean F1@3",
            data_dir / "backward_metrics_per_gui_candidate_by_k.csv",
            3,
            "gui_candidate",
            "#d55e00",
        ),
    ]

    frames = []
    stats = []
    for panel_label, csv_path, k_value, query_unit, color in PANEL_SPECS:
        raw = pd.read_csv(
            csv_path,
            usecols=["method", "filter_variant", "scenario", "evaluation", "query_id", "k", "gold_count", "f1_at_k"],
        )
        raw = raw.loc[
            (raw["evaluation"] == "combined_actions_and_widgets")
            & (raw["k"] == k_value)
            & (raw["scenario"].isin(SCENARIO_ORDER))
            & (raw["method"].isin(METHOD_ORDER))
            & (raw["gold_count"] > 0)
        ].copy()
        raw["f1_at_k"] = pd.to_numeric(raw["f1_at_k"], errors="coerce")
        raw = raw.loc[raw["f1_at_k"].notna()].copy()

        grouped = (
            raw.groupby(["scenario", "query_id"], as_index=False)
            .agg(mean_f1=("f1_at_k", "mean"), contributing_rows=("f1_at_k", "size"))
        )
        grouped["panel"] = panel_label
        grouped["query_unit"] = query_unit
        grouped["color"] = color
        frames.append(grouped)
        stats.append((panel_label, csv_path, len(raw), grouped.copy()))

    plot_df = pd.concat(frames, ignore_index=True)
    plot_df["scenario"] = pd.Categorical(plot_df["scenario"], SCENARIO_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["panel", "scenario", "query_id"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.1), sharey=True)
    rng = np.random.default_rng(202607)
    positions = np.arange(1, len(SCENARIO_ORDER) + 1)

    for ax, (panel_label, _csv_path, _k_value, _query_unit, color) in zip(axes, PANEL_SPECS):
        panel_df = plot_df.loc[plot_df["panel"] == panel_label].copy()
        values_by_scenario = [
            panel_df.loc[panel_df["scenario"] == scenario, "mean_f1"].to_numpy(dtype=float)
            for scenario in SCENARIO_ORDER
        ]
        ax.boxplot(
            values_by_scenario,
            positions=positions,
            widths=0.56,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#e6edf3", "edgecolor": "#4a5560", "linewidth": 1.0, "alpha": 0.78},
            medianprops={"color": "#111111", "linewidth": 1.5},
            whiskerprops={"color": "#4a5560", "linewidth": 1.0},
            capprops={"color": "#4a5560", "linewidth": 1.0},
        )
        for index, scenario in enumerate(SCENARIO_ORDER, start=1):
            values = panel_df.loc[panel_df["scenario"] == scenario, "mean_f1"].to_numpy(dtype=float)
            if len(values) == 0:
                continue
            jitter = rng.normal(0, 0.055, len(values))
            alpha = 0.14 if len(values) > 500 else 0.34
            ax.scatter(
                np.full(len(values), index) + jitter,
                values,
                s=8,
                color=color,
                alpha=alpha,
                linewidths=0,
                zorder=3,
            )
        ax.set_title(panel_label)
        ax.set_xticks(positions)
        ax.set_xticklabels(SCENARIO_ORDER, rotation=25, ha="right")
        ax.set_xlabel("Generated model variant")
        ax.set_ylabel("Mean F1 score" if ax is axes[0] else "")
        ax.set_ylim(0, 0.80)
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_axisbelow(True)

    fig.tight_layout()
    pdf_path, png_path = save_explicit_figure(fig, output_pdf, output_png)

    print("Generated GT-only model-variant query-mean F1 distribution chart")
    print("No legend included, matching the previous query-mean chart.")
    print("GT-only filter: gold_count > 0 before grouping into query-level means.")
    for panel_label, csv_path, raw_count, grouped in stats:
        print(f"\n{panel_label}")
        print(f"CSV: {csv_path}")
        print(f"GT-filtered raw rows used: {raw_count}")
        print(f"query-mean points: {len(grouped)}")
        print("points per generated model variant:")
        print(grouped.groupby("scenario").size().reindex(SCENARIO_ORDER).to_string())
        print(f"mean F1 range: {grouped['mean_f1'].min()} to {grouped['mean_f1'].max()}")
        print(f"contributing rows per point: min={grouped['contributing_rows'].min()}, max={grouped['contributing_rows'].max()}, mean={grouped['contributing_rows'].mean():.2f}")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    return [pdf_path, png_path]


def plot_backward_line_family_styles(data_dir: Path, output_dir: Path) -> List[Path]:
    setup_matplotlib()
    output_root = output_dir
    LINE_METRIC_FIELDS = [
        ("precision_at_k", "Precision@k"),
        ("recall_at_k", "Recall@k"),
        ("hit_at_k", "Hit@k"),
        ("mrr_at_k", "MRR@k"),
        ("f1_at_k", "F1@k"),
    ]


    def method_filter_sort_key(label: str, frame: pd.DataFrame):
        rows = frame.loc[frame["method_filter"].astype(str) == label]
        if rows.empty:
            return (99, 99, label)
        row = rows.iloc[0]
        method = str(row.get("method", ""))
        filter_variant = str(row.get("filter_variant", ""))
        return (
            METHOD_ORDER.index(method) if method in METHOD_ORDER else 99,
            FILTER_ORDER.index(filter_variant) if filter_variant in FILTER_ORDER else 99,
            label,
        )


    def clean_line_chart_rows(frame: pd.DataFrame) -> pd.DataFrame:
        cleaned = frame.copy()
        if "method_filter" not in cleaned.columns:
            cleaned["method_filter"] = cleaned["method"].astype(str) + " " + cleaned["filter_variant"].astype(str)
        cleaned = cleaned.loc[cleaned["method"].isin(METHOD_ORDER)].copy()
        cleaned["k"] = pd.to_numeric(cleaned["k"], errors="coerce")
        cleaned = cleaned.loc[cleaned["k"].notna()].copy()
        cleaned["k"] = cleaned["k"].astype(int)
        for metric, _ in LINE_METRIC_FIELDS:
            cleaned[metric] = pd.to_numeric(cleaned[metric], errors="coerce")
            cleaned = cleaned.loc[cleaned[metric].notna()].copy()
        return cleaned


    def generate_chart(csv_name: str, average_scope: str, output_subdir: str, output_stem: str) -> None:
        csv_path = data_dir / csv_name
        df = pd.read_csv(csv_path)
        filtered = df.loc[
            (df["direction"] == "backward")
            & (df["average_scope"] == average_scope)
            & (df["evaluation"] == "combined_actions_and_widgets")
        ].copy()
        filtered = clean_line_chart_rows(filtered)

        labels = sorted(
            filtered["method_filter"].dropna().astype(str).unique().tolist(),
            key=lambda label: method_filter_sort_key(label, filtered),
        )
        k_values = sorted(filtered["k"].dropna().astype(int).unique().tolist())
        family_by_label = {
            label: str(filtered.loc[filtered["method_filter"].astype(str) == label, "method_family"].iloc[0])
            for label in labels
        }
        color_values = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(labels), 1)))
        color_by_label = {label: color_values[index] for index, label in enumerate(labels)}

        fig, axes = plt.subplots(1, len(LINE_METRIC_FIELDS), figsize=(19, 4.8), sharex=True)
        for ax, (metric, label_text) in zip(axes, LINE_METRIC_FIELDS):
            for label in labels:
                label_rows = filtered.loc[filtered["method_filter"].astype(str) == label]
                means = label_rows.groupby("k")[metric].mean()
                x_values = [k for k in k_values if k in means.index]
                y_values = [float(means.loc[k]) for k in x_values]
                if not x_values:
                    continue
                family = family_by_label.get(label, "")
                ax.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=1.6,
                    linestyle=FAMILY_LINE_STYLES.get(family, "-"),
                    color=color_by_label[label],
                    label=label,
                )
            ax.set_xlabel("k")
            ax.set_ylabel(label_text)
            ax.set_ylim(0, 1)
            ax.set_xticks(k_values)
            ax.grid(True, axis="y", alpha=0.35)
            ax.set_axisbelow(True)

        method_handles, method_labels = axes[0].get_legend_handles_labels()
        if method_handles:
            fig.legend(
                method_handles,
                method_labels,
                loc="lower center",
                ncol=min(4, len(method_labels)),
                bbox_to_anchor=(0.5, -0.04),
                frameon=True,
            )
        style_handles = [
            plt.Line2D([0], [0], color="#222222", linewidth=2.0, linestyle="-", label="IR method"),
            plt.Line2D([0], [0], color="#222222", linewidth=2.0, linestyle="--", label="CE method"),
        ]
        fig.legend(
            handles=style_handles,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.02),
            frameon=True,
        )
        fig.tight_layout(rect=(0, 0.14, 1, 0.95))

        output_dir = output_root / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        png_path = output_dir / f"{output_stem}.png"
        pdf_path = output_dir / f"{output_stem}.pdf"
        save_explicit_figure(fig, pdf_path, png_path)

        print(f"Updated: {png_path}")
        print(f"Updated: {pdf_path}")
        print(f"Y-axis labels: {[label for _, label in LINE_METRIC_FIELDS]}")


    generate_chart(
        "backward_metrics_summary_by_k.csv",
        "all",
        "Backward_Traceability_Figures",
        "backward_line_metrics_over_k_combined_actions_and_widgets_family_style",
    )
    generate_chart(
        "backward_metrics_summary_gold_only_by_k.csv",
        "gold",
        "Backward_GT_Only_Figures",
        "backward_gt_only_line_metrics_over_k_combined_actions_and_widgets_family_style",
    )
    return [
        output_dir / "Backward_Traceability_Figures" / "backward_line_metrics_over_k_combined_actions_and_widgets_family_style.pdf",
        output_dir / "Backward_Traceability_Figures" / "backward_line_metrics_over_k_combined_actions_and_widgets_family_style.png",
        output_dir / "Backward_GT_Only_Figures" / "backward_gt_only_line_metrics_over_k_combined_actions_and_widgets_family_style.pdf",
        output_dir / "Backward_GT_Only_Figures" / "backward_gt_only_line_metrics_over_k_combined_actions_and_widgets_family_style.png",
    ]


def plot_forward_backward_query_level_figures(data_dir: Path, output_dir: Path) -> List[Path]:
    setup_matplotlib()
    output_root = output_dir / "Forward_VS_Backward_Figures"


    def load_query_rows(gt_only: bool) -> pd.DataFrame:
        forward = pd.read_csv(
            data_dir / "forward_metrics_per_requirement_by_k.csv",
            usecols=["method", "filter_variant", "scenario", "evaluation", "query_id", "k", "gold_count", "f1_at_k"],
        )
        forward = forward.loc[
            (forward["evaluation"] == "combined_actions_and_widgets")
            & (forward["k"] == 10)
            & (forward["method"].isin(METHOD_ORDER))
        ].copy()
        if gt_only:
            forward = forward.loc[forward["gold_count"] > 0].copy()
        forward["direction"] = "Forward F1@10"
        forward["query_unit"] = "requirement"

        backward = pd.read_csv(
            data_dir / "backward_metrics_per_gui_candidate_by_k.csv",
            usecols=["method", "filter_variant", "scenario", "evaluation", "query_id", "k", "gold_count", "f1_at_k"],
        )
        backward = backward.loc[
            (backward["evaluation"] == "combined_actions_and_widgets")
            & (backward["k"] == 3)
            & (backward["method"].isin(METHOD_ORDER))
        ].copy()
        if gt_only:
            backward = backward.loc[backward["gold_count"] > 0].copy()
        backward["direction"] = "Backward F1@3"
        backward["query_unit"] = "gui_candidate"

        plot_df = pd.concat(
            [
                forward[["method", "filter_variant", "scenario", "query_id", "query_unit", "direction", "f1_at_k"]],
                backward[["method", "filter_variant", "scenario", "query_id", "query_unit", "direction", "f1_at_k"]],
            ],
            ignore_index=True,
        )
        plot_df["f1_at_k"] = pd.to_numeric(plot_df["f1_at_k"], errors="coerce")
        plot_df = plot_df.loc[plot_df["f1_at_k"].notna()].copy()
        plot_df["method"] = pd.Categorical(plot_df["method"], METHOD_ORDER, ordered=True)
        return plot_df.sort_values(["method", "direction", "scenario", "filter_variant", "query_id"]).copy()


    def draw_chart(plot_df: pd.DataFrame, output_stem: str, backward_label: str) -> None:
        direction_specs = [("Forward F1@10", "#1f77b4", -0.18), (backward_label, "#d55e00", 0.18)]
        plot_df = plot_df.copy()
        plot_df.loc[plot_df["direction"] == "Backward F1@3", "direction"] = backward_label

        fig, ax = plt.subplots(figsize=(10.8, 5.3))
        rng = np.random.default_rng(202607)
        centers = np.arange(1, len(METHOD_ORDER) + 1)

        for direction_label, color, offset in direction_specs:
            direction_rows = plot_df.loc[plot_df["direction"] == direction_label]
            values_by_method = [
                direction_rows.loc[direction_rows["method"] == method, "f1_at_k"].to_numpy(dtype=float)
                for method in METHOD_ORDER
            ]
            positions = centers + offset
            ax.boxplot(
                values_by_method,
                positions=positions,
                widths=0.26,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": color, "edgecolor": "#4a5560", "linewidth": 1.0, "alpha": 0.20},
                medianprops={"color": "#111111", "linewidth": 1.4},
                whiskerprops={"color": "#4a5560", "linewidth": 0.9},
                capprops={"color": "#4a5560", "linewidth": 0.9},
            )
            for index, method in enumerate(METHOD_ORDER):
                values = values_by_method[index]
                if len(values) == 0:
                    continue
                jitter = rng.normal(0, 0.035, len(values))
                alpha = 0.10 if len(values) > 1000 else 0.28
                ax.scatter(
                    np.full(len(values), positions[index]) + jitter,
                    values,
                    s=7,
                    color=color,
                    alpha=alpha,
                    linewidths=0,
                    zorder=3,
                )

        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="none", markersize=7, label=label)
            for label, color, _ in direction_specs
        ]
        ax.legend(handles=handles, loc="upper right", frameon=True)
        ax.set_xticks(centers)
        ax.set_xticklabels(METHOD_ORDER, rotation=30, ha="right")
        ax.set_xlabel("Similarity method")
        ax.set_ylabel("F1 score")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_axisbelow(True)
        fig.tight_layout()

        png_path = output_root / f"{output_stem}.png"
        pdf_path = output_root / f"{output_stem}.pdf"
        save_explicit_figure(fig, pdf_path, png_path)

        print(f"\n{output_stem}")
        print("One point = one query-level F1 row for one method, scenario, and filtering variant.")
        print("Forward query unit = requirement; backward query unit = GUI candidate.")
        counts = (
            plot_df.groupby(["direction", "method"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reindex(index=[label for label, _, _ in direction_specs], columns=METHOD_ORDER, fill_value=0)
        )
        print(counts.to_string())
        print(f"F1 range: {plot_df['f1_at_k'].min()} to {plot_df['f1_at_k'].max()}")
        print(f"PNG: {png_path}")
        print(f"PDF: {pdf_path}")

    all_candidate = load_query_rows(gt_only=False)
    draw_chart(
        all_candidate,
        "forward_backward_query_level_f1_by_method_combined_actions_and_widgets",
        "Backward all-candidate F1@3",
    )

    gt_only = load_query_rows(gt_only=True)
    draw_chart(
        gt_only,
        "forward_backward_ground_truth_only_query_level_f1_by_method_combined_actions_and_widgets",
        "Backward GT-only F1@3",
    )
    return [
        output_root / "forward_backward_query_level_f1_by_method_combined_actions_and_widgets.pdf",
        output_root / "forward_backward_query_level_f1_by_method_combined_actions_and_widgets.png",
        output_root / "forward_backward_ground_truth_only_query_level_f1_by_method_combined_actions_and_widgets.pdf",
        output_root / "forward_backward_ground_truth_only_query_level_f1_by_method_combined_actions_and_widgets.png",
    ]




def postprocess_candidate_type_axis_labels(forward_pdf: Path, backward_pdf: Path, gt_pdf: Path) -> None:
    """Replace repeated candidate-type x-axis labels with one centered label in PDFs."""
    try:
        import os
        import tempfile
        import fitz
    except ImportError:
        return

    label = "Similarity method"
    default_label_y0 = 358.06
    default_label_y1 = 369.24

    for pdf_path in [forward_pdf, backward_pdf, gt_pdf]:
        doc = fitz.open(pdf_path)
        page = doc[0]
        rects = page.search_for(label)
        if rects:
            y0 = min(rect.y0 for rect in rects)
            y1 = max(rect.y1 for rect in rects)
            for rect in rects:
                cover = fitz.Rect(rect.x0 - 7, rect.y0 - 2, rect.x1 + 7, rect.y1 + 3)
                page.draw_rect(cover, color=(1, 1, 1), fill=(1, 1, 1), width=0, overlay=True)
        else:
            y0 = default_label_y0
            y1 = default_label_y1
        label_box = fitz.Rect(0, y0 - 2, page.rect.width, y1 + 3)
        page.insert_textbox(
            label_box,
            label,
            fontsize=10,
            fontname="helv",
            color=(0, 0, 0),
            align=fitz.TEXT_ALIGN_CENTER,
            overlay=True,
        )
        fd, tmp_name = tempfile.mkstemp(suffix=".pdf", dir=str(pdf_path.parent))
        os.close(fd)
        tmp_path = Path(tmp_name)
        doc.save(tmp_path, garbage=3, deflate=True)
        doc.close()
        tmp_path.replace(pdf_path)

    font = "helv"
    fontsize = 10
    baseline_y = 366.2
    for pdf_path in [backward_pdf, gt_pdf]:
        doc = fitz.open(pdf_path)
        page = doc[0]
        for rect in page.search_for(label):
            if rect.y0 > 350:
                cover = fitz.Rect(rect.x0 - 7, rect.y0 - 2, rect.x1 + 7, rect.y1 + 3)
                page.draw_rect(cover, color=(1, 1, 1), fill=(1, 1, 1), width=0, overlay=True)
        text_width = fitz.get_text_length(label, fontname=font, fontsize=fontsize)
        x = (page.rect.width - text_width) / 2
        page.insert_text(
            fitz.Point(x, baseline_y),
            label,
            fontsize=fontsize,
            fontname=font,
            color=(0, 0, 0),
            overlay=True,
        )
        fd, tmp_name = tempfile.mkstemp(suffix=".pdf", dir=str(pdf_path.parent))
        os.close(fd)
        tmp_path = Path(tmp_name)
        doc.save(tmp_path, garbage=3, deflate=True)
        doc.close()
        tmp_path.replace(pdf_path)

def generate_all_thesis_figures(data_dir: Path, output_dir: Path = DEFAULT_THESIS_FIGURES_DIR) -> List[Path]:
    """Regenerate the curated final thesis figures from CSV inputs."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    created: List[Path] = []

    targets = {
        "fig_7_3_forward_f1_method_distribution": (
            output_dir / "Forward_Traceability_Figures" / "forward_f1_method_distribution.pdf",
            output_dir / "Forward_Traceability_Figures" / "forward_f1_method_distribution.png",
        ),
        "fig_7_3_forward_candidate_type_f1_by_method": (
            output_dir / "Forward_Traceability_Figures" / "forward_candidate_type_f1_by_method.pdf",
            output_dir / "Forward_Traceability_Figures" / "forward_candidate_type_f1_by_method.png",
        ),
        "fig_7_3_forward_line_metrics_over_k_combined_actions_and_widgets_family_style": (
            output_dir / "Forward_Traceability_Figures" / "forward_line_metrics_over_k_combined_actions_and_widgets_family_style.pdf",
            output_dir / "Forward_Traceability_Figures" / "forward_line_metrics_over_k_combined_actions_and_widgets_family_style.png",
        ),
        "fig_7_4_backward_candidate_type_f1_by_method": (
            output_dir / "Backward_Traceability_Figures" / "backward_candidate_type_f1_by_method.pdf",
            output_dir / "Backward_Traceability_Figures" / "backward_candidate_type_f1_by_method.png",
        ),
    }
    _OUTPUT_TARGETS.clear()
    _OUTPUT_TARGETS.update(targets)

    setup_matplotlib()
    rng = np.random.default_rng(202607)
    figure_a(data_dir, rng)
    figure_b(data_dir, rng)
    figure_forward_line_family_style(data_dir)
    figure_e(data_dir, rng)
    for pdf_path, png_path in targets.values():
        created.extend([pdf_path, png_path])

    created.extend(plot_backward_coverage_hit_f1_by_method(
        data_dir / "backward_metrics_summary_by_k.csv",
        output_dir / "Backward_Traceability_Figures" / "backward_coverage_hit_f1_by_method.pdf",
        output_dir / "Backward_Traceability_Figures" / "backward_coverage_hit_f1_by_method.png",
    ))
    created.extend(plot_backward_line_family_styles(data_dir, output_dir))
    created.extend(plot_backward_top_requirement_distribution_figure(
        data_dir,
        output_dir / "Backward_Traceability_Figures" / "backward_top_requirement_distribution.pdf",
        output_dir / "Backward_Traceability_Figures" / "backward_top_requirement_distribution.png",
    ))
    created.extend(plot_backward_gt_candidate_type_f1(
        data_dir,
        output_dir / "Backward_GT_Only_Figures" / "backward_gt_only_candidate_type_f1_by_method.pdf",
        output_dir / "Backward_GT_Only_Figures" / "backward_gt_only_candidate_type_f1_by_method.png",
    ))
    postprocess_candidate_type_axis_labels(
        output_dir / "Forward_Traceability_Figures" / "forward_candidate_type_f1_by_method.pdf",
        output_dir / "Backward_Traceability_Figures" / "backward_candidate_type_f1_by_method.pdf",
        output_dir / "Backward_GT_Only_Figures" / "backward_gt_only_candidate_type_f1_by_method.pdf",
    )
    created.extend(plot_candidate_space_composition(
        data_dir,
        output_dir / "Descriptive_Statistics" / "candidate_space_composition.pdf",
        output_dir / "Descriptive_Statistics" / "candidate_space_composition.png",
    ))
    created.extend(plot_filtering_delta_f1(
        data_dir,
        output_dir / "Filtering_Figures" / "filtering_delta_f1_by_method_direction.pdf",
        output_dir / "Filtering_Figures" / "filtering_delta_f1_by_method_direction.png",
    ))
    created.extend(plot_query_level_filtering_delta_f1(
        data_dir,
        output_dir / "Filtering_Figures" / "query_level_delta_f1_by_method_direction.pdf",
        output_dir / "Filtering_Figures" / "query_level_delta_f1_by_method_direction.png",
    ))
    created.extend(plot_forward_backward_summary_all(
        data_dir,
        output_dir / "Forward_VS_Backward_Figures" / "forward_backward_f1_by_method_combined_actions_and_widgets.pdf",
        output_dir / "Forward_VS_Backward_Figures" / "forward_backward_f1_by_method_combined_actions_and_widgets.png",
    ))
    created.extend(plot_forward_backward_summary_gt_only(
        data_dir,
        output_dir / "Forward_VS_Backward_Figures" / "forward_backward_ground_truth_only_f1_by_method_combined_actions_and_widgets.pdf",
        output_dir / "Forward_VS_Backward_Figures" / "forward_backward_ground_truth_only_f1_by_method_combined_actions_and_widgets.png",
    ))
    created.extend(plot_forward_backward_query_level_figures(data_dir, output_dir))
    created.extend(plot_f1_heatmap_figure(
        data_dir,
        "backward_metrics_summary_by_k.csv",
        output_dir / "GuidingLLM_Req_Source_Figures" / "backward_all_candidate_heatmap_f1_at_k3_combined_actions_and_widgets.pdf",
        output_dir / "GuidingLLM_Req_Source_Figures" / "backward_all_candidate_heatmap_f1_at_k3_combined_actions_and_widgets.png",
        "backward_all_candidate",
        "all",
    ))
    created.extend(plot_f1_heatmap_figure(
        data_dir,
        "backward_metrics_summary_gold_only_by_k.csv",
        output_dir / "GuidingLLM_Req_Source_Figures" / "backward_gold_only_heatmap_f1_at_k3_combined_actions_and_widgets.pdf",
        output_dir / "GuidingLLM_Req_Source_Figures" / "backward_gold_only_heatmap_f1_at_k3_combined_actions_and_widgets.png",
        "backward_gold_only",
        "gold",
    ))
    created.extend(plot_f1_heatmap_figure(
        data_dir,
        "forward_metrics_summary_by_k.csv",
        output_dir / "GuidingLLM_Req_Source_Figures" / "forward_heatmap_f1_at_k10_combined_actions_and_widgets.pdf",
        output_dir / "GuidingLLM_Req_Source_Figures" / "forward_heatmap_f1_at_k10_combined_actions_and_widgets.png",
        "forward",
        "",
    ))
    created.extend(plot_model_variant_f1_distribution_detailed(
        data_dir,
        output_dir / "GuidingLLM_Req_Source_Figures" / "model_variant_f1_distribution_detailed.pdf",
        output_dir / "GuidingLLM_Req_Source_Figures" / "model_variant_f1_distribution_detailed.png",
    ))
    created.extend(plot_model_variant_f1_distribution_detailed_gt_only(
        data_dir,
        output_dir / "GuidingLLM_Req_Source_Figures" / "model_variant_f1_distribution_detailed_ground_truth_only.pdf",
        output_dir / "GuidingLLM_Req_Source_Figures" / "model_variant_f1_distribution_detailed_ground_truth_only.png",
    ))
    created.extend(plot_model_variant_query_mean_f1_distribution(
        data_dir,
        output_dir / "GuidingLLM_Req_Source_Figures" / "model_variant_query_mean_f1_distribution.pdf",
        output_dir / "GuidingLLM_Req_Source_Figures" / "model_variant_query_mean_f1_distribution.png",
    ))
    created.extend(plot_model_variant_query_mean_f1_distribution_gt_only(
        data_dir,
        output_dir / "GuidingLLM_Req_Source_Figures" / "model_variant_query_mean_f1_distribution_ground_truth_only.pdf",
        output_dir / "GuidingLLM_Req_Source_Figures" / "model_variant_query_mean_f1_distribution_ground_truth_only.png",
    ))
    created.extend(plot_scalability_figure(
        data_dir,
        output_dir / "Scalability_Figures" / "scalability_peak_memory_mb.pdf",
        output_dir / "Scalability_Figures" / "scalability_peak_memory_mb.png",
        "memory",
    ))
    created.extend(plot_scalability_figure(
        data_dir,
        output_dir / "Scalability_Figures" / "scalability_runtime_total_seconds.pdf",
        output_dir / "Scalability_Figures" / "scalability_runtime_total_seconds.png",
        "runtime",
    ))
    return created


if __name__ == "__main__":
    generated = generate_all_thesis_figures(DEFAULT_THESIS_FIGURES_DIR / "data", DEFAULT_THESIS_FIGURES_DIR)
    print(f"Generated {len(generated)} thesis figure files.")
