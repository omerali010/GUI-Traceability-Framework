"""Run Qwen3-Embedding-4B traceability ranking."""

import argparse

from embedding_requirements_compare_common import (
    DEFAULT_DATA_PATH,
    DEFAULT_REQ_PATH,
    DEFAULT_TOP_K,
    run_embedding_pipeline,
)

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"

# Pin the Hugging Face model snapshot used for replication.
MODEL_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"

DEFAULT_OUT_PATH = "Results/CE/Strict/Qwen4B/qwen3_embedding_4b_matches_Focus_GPT5.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the Qwen3 4B run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", default=DEFAULT_REQ_PATH)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", default=DEFAULT_OUT_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--query-instruction", default=None)
    parser.add_argument("--document-instruction", default=None)
    parser.add_argument(
        "--backward-output",
        default=None,
        help="Path for inverted GUI-candidate-to-requirement ranking JSON. Defaults to '<output>_backward.json'.",
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


def main() -> None:
    """Run the shared embedding pipeline with Qwen3 4B."""
    args = parse_args()
    stats = run_embedding_pipeline(
        model_name=MODEL_NAME,
        output_path=args.output,
        requirements_path=args.requirements,
        data_path=args.data,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device=args.device,
        normalize_embeddings=not args.no_normalize,
        query_instruction=args.query_instruction,
        document_instruction=args.document_instruction,
        query_prompt_name="query",
        model_revision=MODEL_REVISION,
        backward_output_path=args.backward_output,
        backward_field=args.backward_field,
        backward_top_k=args.backward_top_k,
        write_backward_output=not args.no_backward_output,
    )
    print(
        f"Wrote {args.output} "
        f"({stats['requirements']} requirements, {stats['widget_candidates']} widget candidates, {stats['transition_candidates']} transition candidates)"
    )
    if not args.no_backward_output:
        print(f"Wrote backward output ({stats['backward_candidates']} GUI candidates with at least one requirement match)")


if __name__ == "__main__":
    main()
