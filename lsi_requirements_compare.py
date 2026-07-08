"""Run the LSI baseline and write forward and backward traceability rankings."""

import argparse
import numpy as np
from scipy.linalg import svd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

DEFAULT_OUT_PATH = "Results/IR/Strict/LSI/lsi_matches_Focus_GPT5.json"
DEFAULT_COMPONENTS = 100


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the LSI run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", default=DEFAULT_REQ_PATH)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", default=DEFAULT_OUT_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--components", type=int, default=DEFAULT_COMPONENTS)
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

def empty_lsi_space(document_count: int) -> Dict[str, Any]:
    return {
        "vectorizer": None,
        "doc_latent": np.zeros((document_count, 0)),
        "components": np.zeros((0, 0)),
        "sigma_inv": np.zeros(0),
    }


def build_lsi_space(candidate_docs: List[str], max_components: int) -> Dict[str, Any]:
    """Build the reduced TF-IDF space used for LSI ranking."""
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        norm=None,
        smooth_idf=True,
        use_idf=True,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(candidate_docs)
    except ValueError:
        return empty_lsi_space(len(candidate_docs))

    if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
        return empty_lsi_space(len(candidate_docs))

    # SVD reduces sparse term counts into a smaller semantic space.
    u_matrix, singular_values, vt_matrix = svd(tfidf_matrix.toarray(), full_matrices=False)
    non_zero = int(np.sum(singular_values > 1e-12))
    component_count = min(max_components, non_zero)
    if component_count == 0:
        return empty_lsi_space(len(candidate_docs))

    u_k = u_matrix[:, :component_count]
    sigma_k = singular_values[:component_count]
    vt_k = vt_matrix[:component_count, :]
    sigma_inv = np.divide(1.0, sigma_k, out=np.zeros_like(sigma_k), where=sigma_k > 1e-12)
    doc_latent = u_k * sigma_k
    return {
        "vectorizer": vectorizer,
        "doc_latent": doc_latent,
        "components": vt_k,
        "sigma_inv": sigma_inv,
    }


def project_query(text: str, lsi_space: Dict[str, Any]) -> np.ndarray:
    """Project a requirement into the LSI space built from GUI candidates."""
    vectorizer = lsi_space["vectorizer"]
    if vectorizer is None:
        return np.zeros(0)

    if lsi_space["components"].size == 0:
        return np.zeros(0, dtype=np.float64)
    query_tfidf = vectorizer.transform([text])
    return np.asarray((query_tfidf @ lsi_space["components"].T) * lsi_space["sigma_inv"]).ravel()


def rank(requirement_text: str, lsi_space: Dict[str, Any], metadata: List[Dict[str, Any]], top_k: int) -> List[List[Any]]:
    """Rank candidate documents for one requirement in the LSI space."""
    query_latent = project_query(requirement_text, lsi_space)
    if query_latent.size == 0 or lsi_space["doc_latent"].shape[1] == 0:
        return []
    scores = cosine_similarity(query_latent.reshape(1, -1), lsi_space["doc_latent"]).ravel()
    scored = [
        [round(float(score), 6), metadata[index]]
        for index, score in enumerate(scores)
        if score > 0.0
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def main() -> None:
    """Run LSI ranking over widget and transition candidates."""
    args = parse_args()
    tracker = RuntimeMemoryTracker()

    requirements = read_requirements(args.requirements)
    candidates = load_widget_and_transition_candidates(args.data)
    widget_docs = candidates["widget_docs"]
    widget_meta = candidates["widget_meta"]
    transition_docs = candidates["transition_docs"]
    transition_meta = candidates["transition_meta"]

    widget_lsi_space = build_lsi_space(widget_docs, args.components)
    transition_lsi_space = build_lsi_space(transition_docs, args.components)

    results = []
    for requirement in requirements:
        top_widgets = rank(requirement["text"], widget_lsi_space, widget_meta, args.top_k)
        top_transitions = rank(requirement["text"], transition_lsi_space, transition_meta, args.top_k)
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
        "lsi_matches",
        "lsi_backward_matches",
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
