"""Run the VSM baseline and write forward and backward traceability rankings."""

import argparse
from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

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

DEFAULT_OUT_PATH = "Results/IR/Strict/VSM/vsm_matches_Focus_GPT5.json"


def build_tfidf_space(requirement_texts: List[str], candidate_docs: List[str]) -> Dict[str, Any]:
    """Fit TF-IDF on GUI candidates and project requirements into the same space."""
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        norm="l2",
        smooth_idf=True,
        use_idf=True,
    )
    candidate_matrix = vectorizer.fit_transform(candidate_docs)
    requirement_matrix = vectorizer.transform(requirement_texts)
    return {
        "requirement_matrix": requirement_matrix,
        "candidate_matrix": candidate_matrix,
    }


def rank_scores(
    scores: List[float],
    document_metadata: List[Dict[str, Any]],
    top_k: int,
) -> List[List[Any]]:
    scored = [
        [round(float(score), 6), document_metadata[index]]
        for index, score in enumerate(scores)
        if score > 0.0
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def rank(
    requirement_index: int,
    requirement_matrix: Any,
    document_matrix: Any,
    document_metadata: List[Dict[str, Any]],
    top_k: int,
) -> List[List[Any]]:
    scores = linear_kernel(requirement_matrix[requirement_index], document_matrix).ravel()
    return rank_scores(scores.tolist(), document_metadata, top_k)

def parse_args() -> argparse.Namespace:
    """Parse command-line options for the VSM run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", default=DEFAULT_REQ_PATH, help="Path to the requirements file.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to the filtered GUI model JSON.")
    parser.add_argument("--output", default=DEFAULT_OUT_PATH, help="Path for the ranking output JSON.")
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
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of matches to keep per ranking.")
    return parser.parse_args()

def main() -> None:
    """Run VSM ranking over widget and transition candidates."""
    args = parse_args()
    tracker = RuntimeMemoryTracker()

    requirements = read_requirements(args.requirements)
    candidates = load_widget_and_transition_candidates(args.data)
    widget_docs = candidates["widget_docs"]
    widget_meta = candidates["widget_meta"]
    transition_docs = candidates["transition_docs"]
    transition_meta = candidates["transition_meta"]
    # Fit one TF-IDF vectorizer on GUI candidates so requirement queries are projected into the same space.
    candidate_docs = widget_docs + transition_docs
    tfidf_space = build_tfidf_space([req["text"] for req in requirements], candidate_docs)
    requirement_matrix = tfidf_space["requirement_matrix"]
    widget_matrix = tfidf_space["candidate_matrix"][: len(widget_docs)]
    transition_matrix = tfidf_space["candidate_matrix"][len(widget_docs) :]
    results = []
    for index, requirement in enumerate(requirements):
        top_widgets = rank(index, requirement_matrix, widget_matrix, widget_meta, args.top_k)
        top_transitions = rank(index, requirement_matrix, transition_matrix, transition_meta, args.top_k)

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
        "vsm_matches",
        "vsm_backward_matches",
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
