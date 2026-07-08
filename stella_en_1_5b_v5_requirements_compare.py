"""Run Stella 1.5B traceability ranking."""

import argparse
import gc
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from embedding_requirements_compare_common import (
    build_environment_metadata,
    clear_cuda_memory,
    cosine_rank,
    default_backward_output_path,
    DEFAULT_DATA_PATH,
    DEFAULT_REQ_PATH,
    DEFAULT_TOP_K,
    is_cuda_oom_error,
    load_candidates,
    read_requirements,
    RuntimeMemoryTracker,
    select_execution_device,
    write_ce_results,
)
from results_common import invert_ranked_results

MODEL_NAME = "it-just-works/stella_en_1.5B_v5_bf16"

# Pin the Hugging Face model snapshot used for replication.
MODEL_REVISION = "b6f39e45892c6edd44f1e602d84b6adf8891a1e3"

DEFAULT_OUT_PATH = "Results/CE/Strict/Stella1.5B/stella_en_1_5b_v5_matches_Focus_GPT5.json"

# Keep Stella inference practical on limited hardware.
MAX_LENGTH = 512

QUERY_PROMPT_NAME = "s2p_query"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the Stella run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", default=DEFAULT_REQ_PATH)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", default=DEFAULT_OUT_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-normalize", action="store_true")
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


def load_model(device: str) -> Any:
    """Load Stella and apply the sequence length used for replication."""
    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    model_kwargs["revision"] = MODEL_REVISION
    if device:
        model_kwargs["device"] = device
    model = SentenceTransformer(MODEL_NAME, **model_kwargs)
    model.max_seq_length = MAX_LENGTH
    return model


def encode_texts(
    model: Any,
    texts: List[str],
    batch_size: int,
    normalize_embeddings: bool,
    prompt_name: Optional[str] = None,
) -> np.ndarray:
    """Encode texts with Stella and return float32 NumPy embeddings."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
        prompt_name=prompt_name,
    )
    return embeddings.detach().to(torch.float32).cpu().numpy()


def main() -> None:
    """Run Stella ranking over widget and transition candidates."""
    args = parse_args()
    tracker = RuntimeMemoryTracker()

    requirements = read_requirements(args.requirements)
    candidates = load_candidates(args.data)
    requirement_texts = [requirement["text"] for requirement in requirements]

    requested_device = select_execution_device(MODEL_NAME, args.device)
    allow_cpu_fallback = args.device is None or str(args.device).lower().startswith("cuda")
    normalize_embeddings = not args.no_normalize

    try:
        model = load_model(requested_device)
        requirement_embeddings = encode_texts(
            model,
            requirement_texts,
            batch_size=args.batch_size,
            normalize_embeddings=normalize_embeddings,
            prompt_name=QUERY_PROMPT_NAME,
        )
        widget_embeddings = encode_texts(
            model,
            candidates["widget_docs"],
            batch_size=args.batch_size,
            normalize_embeddings=normalize_embeddings,
        )
        transition_embeddings = encode_texts(
            model,
            candidates["transition_docs"],
            batch_size=args.batch_size,
            normalize_embeddings=normalize_embeddings,
        )
    except Exception as error:
        if not (allow_cpu_fallback and is_cuda_oom_error(error)):
            raise

        print("CUDA out of memory detected. Falling back to CPU inference.")
        if "model" in locals():
            del model
        gc.collect()
        clear_cuda_memory()

        model = load_model("cpu")
        requirement_embeddings = encode_texts(
            model,
            requirement_texts,
            batch_size=args.batch_size,
            normalize_embeddings=normalize_embeddings,
            prompt_name=QUERY_PROMPT_NAME,
        )
        widget_embeddings = encode_texts(
            model,
            candidates["widget_docs"],
            batch_size=args.batch_size,
            normalize_embeddings=normalize_embeddings,
        )
        transition_embeddings = encode_texts(
            model,
            candidates["transition_docs"],
            batch_size=args.batch_size,
            normalize_embeddings=normalize_embeddings,
        )

    results = []
    for index, requirement in enumerate(requirements):
        query_embedding = requirement_embeddings[index]
        top_widgets = cosine_rank(query_embedding, widget_embeddings, candidates["widget_meta"], args.top_k)
        top_transitions = cosine_rank(query_embedding, transition_embeddings, candidates["transition_meta"], args.top_k)
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

    backward_results = []
    backward_output = None
    if not args.no_backward_output:
        backward_output = args.backward_output or default_backward_output_path(args.output)
        backward_results = invert_ranked_results(
            results,
            source_field=args.backward_field,
            top_k=args.backward_top_k,
        )

    runtime, memory = tracker.stop()

    environment = build_environment_metadata(
        requirements_path=args.requirements,
        data_path=args.data,
        output_path=args.output,
        runtime=runtime,
        memory=memory,
    )
    write_ce_results(args.output, results, environment)

    if backward_output:
        backward_environment = build_environment_metadata(
            requirements_path=args.requirements,
            data_path=args.data,
            output_path=backward_output,
            runtime=runtime,
            memory=memory,
        )
        write_ce_results(backward_output, backward_results, backward_environment)

    print(
        f"Wrote {args.output} "
        f"({len(requirements)} requirements, {len(candidates['widget_docs'])} widget candidates, {len(candidates['transition_docs'])} transition candidates)"
    )
    if not args.no_backward_output:
        print(f"Wrote backward output ({len(backward_results)} GUI candidates with at least one requirement match)")


if __name__ == "__main__":
    main()
