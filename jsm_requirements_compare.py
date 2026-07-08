"""Run the Jensen-Shannon baseline and write traceability rankings."""

import argparse
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import CountVectorizer
from typing import Any, Dict, List

from candidate_construction import (
    DEFAULT_DATA_PATH,
    DEFAULT_REQ_PATH,
    load_widget_and_transition_candidates,
    read_requirements,
)
from results_common import (
    DEFAULT_TOP_K,
    RuntimeMemoryTracker,
    build_result_environment_metadata,
    default_backward_output_path,
    invert_ranked_results,
    write_ranked_results,
)
from text_preprocessing import tokenize

DEFAULT_OUT_PATH = "Results/IR/Strict/JSM/jsm_matches_Focus_GPT5.json"
DEFAULT_ALPHA = 1e-6


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the Jensen-Shannon run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", default=DEFAULT_REQ_PATH)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", default=DEFAULT_OUT_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--backward-output",
        default=None,
        help="Path for inverted GUI-candidate-to-requirement ranking JSON. Defaults to a backward filename next to --output.",
    )
    parser.add_argument(
        "--backward-field",
        default="TopMatches",
        choices=["TopMatches", "TopWidgets", "TopTransitions"],
        help="Forward result field to invert for backward traceability.",
    )
    parser.add_argument(
        "--backward-top-k",
        type=int,
        default=None,
        help="Optional number of requirements to keep per inverted GUI candidate.",
    )
    parser.add_argument(
        "--no-backward-output",
        action="store_true",
        help="Disable writing the inverted GUI-candidate-to-requirement ranking JSON.",
    )
    return parser.parse_args()


def empty_jsm_space(document_count: int, alpha: float) -> Dict[str, Any]:
    return {
        "vectorizer": None,
        "alpha": alpha,
        "distributions": np.zeros((document_count, 0), dtype=np.float64),
    }


def normalize_count_matrix(count_matrix: Any, alpha: float) -> np.ndarray:
    """Convert token counts into smoothed probability distributions."""
    distributions = count_matrix.toarray().astype(np.float64) + alpha
    totals = distributions.sum(axis=1, keepdims=True)
    return np.divide(distributions, totals, out=np.zeros_like(distributions), where=totals != 0.0)


def jensen_shannon_similarity(distribution_a: np.ndarray, distribution_b: np.ndarray) -> float:
    """Convert Jensen-Shannon distance into a similarity score."""
    if distribution_a.size == 0 or distribution_b.size == 0:
        return 0.0
    distance = float(jensenshannon(distribution_a, distribution_b, base=2.0))
    similarity = 1.0 - (distance * distance)
    return max(0.0, similarity)


def build_jsm_space(candidate_docs: List[str], alpha: float) -> Dict[str, Any]:
    """Build the smoothed term-distribution space for GUI candidates."""
    vectorizer = CountVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
    )
    try:
        counts = vectorizer.fit_transform(candidate_docs)
    except ValueError:
        return empty_jsm_space(len(candidate_docs), alpha)

    return {
        "vectorizer": vectorizer,
        "alpha": alpha,
        "distributions": normalize_count_matrix(counts, alpha),
    }


def rank(
    requirement_text: str,
    jsm_space: Dict[str, Any],
    metadata: List[Dict[str, Any]],
    top_k: int,
) -> List[List[Any]]:
    """Rank candidate documents for one requirement using Jensen-Shannon similarity."""
    vectorizer = jsm_space["vectorizer"]
    if vectorizer is None:
        return []
    query_distribution = normalize_count_matrix(
        vectorizer.transform([requirement_text]),
        jsm_space["alpha"],
    )[0]
    scored = []
    for index, document_distribution in enumerate(jsm_space["distributions"]):
        score = jensen_shannon_similarity(query_distribution, document_distribution)
        if score > 0.0:
            scored.append([round(score, 6), metadata[index]])
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def main() -> None:
    """Run Jensen-Shannon ranking over widget and transition candidates."""
    args = parse_args()
    tracker = RuntimeMemoryTracker()

    requirements = read_requirements(args.requirements)
    candidates = load_widget_and_transition_candidates(args.data)
    widget_docs = candidates["widget_docs"]
    widget_meta = candidates["widget_meta"]
    transition_docs = candidates["transition_docs"]
    transition_meta = candidates["transition_meta"]

    widget_jsm_space = build_jsm_space(widget_docs, args.alpha)
    transition_jsm_space = build_jsm_space(transition_docs, args.alpha)

    results = []
    for requirement in requirements:
        top_widgets = rank(requirement["text"], widget_jsm_space, widget_meta, args.top_k)
        top_transitions = rank(requirement["text"], transition_jsm_space, transition_meta, args.top_k)
        combined = sorted(top_widgets + top_transitions, key=lambda item: item[0], reverse=True)[: args.top_k]
        results.append(
            {
                "ReqID": requirement["id"],
                "Requirement": requirement["text"],
                "TopWidgets": top_widgets,
                "TopTransitions": top_transitions,
                "TopMatches": combined,
            }
        )

    backward_output = None if args.no_backward_output else args.backward_output or default_backward_output_path(
        args.output,
        "jsm_matches",
        "jsm_backward_matches",
    )
    backward_results = []
    if backward_output:
        backward_results = invert_ranked_results(
            results,
            source_field=args.backward_field,
            top_k=args.backward_top_k,
        )

    runtime, memory = tracker.stop()

    environment = build_result_environment_metadata(
        requirements_path=args.requirements,
        data_path=args.data,
        output_path=args.output,
        runtime=runtime,
        memory=memory,
    )
    write_ranked_results(args.output, results, environment)

    if backward_output:
        backward_environment = build_result_environment_metadata(
            requirements_path=args.requirements,
            data_path=args.data,
            output_path=backward_output,
            runtime=runtime,
            memory=memory,
        )
        write_ranked_results(backward_output, backward_results, backward_environment)

    print(
        f"Wrote {args.output} "
        f"({len(requirements)} requirements, {len(widget_docs)} widget candidates, {len(transition_docs)} transition candidates)"
    )
    if backward_output:
        print(
            f"Wrote {backward_output} "
            f"({len(backward_results)} GUI candidates with at least one requirement match)"
        )


if __name__ == "__main__":
    main()
