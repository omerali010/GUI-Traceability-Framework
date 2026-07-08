"""Internal orchestration for generating final thesis charts from metric CSV files."""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

from chart_thesis_figures import generate_all_thesis_figures


ROOT = Path(__file__).resolve().parent
DEFAULT_THESIS_FIGURES_DIR = ROOT / "Thesis Figures"


def generate_thesis_figure_set(output_dir: Path = DEFAULT_THESIS_FIGURES_DIR) -> List[Path]:
    """Generate the curated final thesis figure set from ``output_dir / "data"``."""
    output_dir = Path(output_dir)
    return generate_all_thesis_figures(output_dir / "data", output_dir)


def generate_section_charts(output_dir: Path) -> List[Path]:
    """Compatibility entry point used after metric CSVs are written.

    The final thesis figures are regenerated from the CSV files in
    ``output_dir / "data"`` so the same path is used for full metric runs and
    for plots-only runs.
    """
    return generate_thesis_figure_set(Path(output_dir))


def plot_charts_from_csv_outputs(output_dir: Path) -> List[Path]:
    """Generate final thesis figures from existing CSV files."""
    return generate_thesis_figure_set(Path(output_dir))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the curated final thesis figure PDF and PNG set from Thesis Figures/data."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_THESIS_FIGURES_DIR))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    paths = generate_thesis_figure_set(output_dir=output_dir)
    print(f"Generated {len(paths)} thesis figure files in {output_dir}.")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
