"""Run Jina embeddings v3 traceability ranking."""

import argparse
import gc
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

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

# Use the native Transformers checkpoint for stable adapter handling.
MODEL_NAME = "jinaai/jina-embeddings-v3-hf"

# Pin the Hugging Face model snapshot used for replication.
MODEL_REVISION = "d18862d9a48706220815554fac3ebb4dfa46fc28"

DEFAULT_OUT_PATH = "Results/CE/Strict/Jina/jina_embeddings_v3_matches_Focus_Gemma3.json"

# Keep inference practical while preserving the most important requirement and GUI terms.
MAX_LENGTH = 512

TASK_TO_ADAPTER = {
    "query": "retrieval_query",
    "passage": "retrieval_passage",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the Jina run."""
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


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool token embeddings into one vector while ignoring padding tokens."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1e-9)
    return pooled / counts


def load_model(device: str) -> Tuple[Any, Any]:
    """Load the Jina tokenizer and model on the selected device."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    model_kwargs: Dict[str, Any] = {}
    if str(device).lower().startswith("cuda"):
        model_kwargs["dtype"] = torch.float32
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        **model_kwargs,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def activate_adapter(model: Any, role: str) -> bool:
    """Activate the Jina retrieval adapter for query or passage encoding."""
    adapter_name = TASK_TO_ADAPTER[role]
    existing_adapters = getattr(model, "peft_config", {})
    if adapter_name in existing_adapters:
        model.set_adapter(adapter_name)
        print(f"Loaded jina adapter: {adapter_name}")
        return True

    try:
        model.load_adapter(
            MODEL_NAME,
            adapter_name=adapter_name,
            adapter_kwargs={"subfolder": adapter_name, "revision": MODEL_REVISION},
        )
        model.set_adapter(adapter_name)
        print(f"Loaded jina adapter: {adapter_name}")
        return True
    except Exception as error:
        print(f"Warning: could not load jina adapter '{adapter_name}'. Continuing without it. ({error})")
        return False


def encode_texts(
    tokenizer: Any,
    model: Any,
    texts: List[str],
    role: str,
    batch_size: int,
    normalize_embeddings: bool,
    device: str,
) -> np.ndarray:
    """Encode texts with the Jina query or passage adapter."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    activate_adapter(model, role)
    all_embeddings: List[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            embeddings = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            if normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.detach().to(torch.float32).cpu().numpy())

    return np.vstack(all_embeddings) if all_embeddings else np.zeros((0, 0), dtype=np.float32)


def main() -> None:
    """Run Jina ranking over widget and transition candidates."""
    args = parse_args()
    tracker = RuntimeMemoryTracker()

    requirements = read_requirements(args.requirements)
    candidates = load_candidates(args.data)
    requirement_texts = [requirement["text"] for requirement in requirements]

    requested_device = select_execution_device(MODEL_NAME, args.device)
    allow_cpu_fallback = args.device is None or str(args.device).lower().startswith("cuda")
    normalize_embeddings = not args.no_normalize

    try:
        tokenizer, model = load_model(requested_device)
        requirement_embeddings = encode_texts(
            tokenizer,
            model,
            requirement_texts,
            role="query",
            batch_size=args.batch_size,
            normalize_embeddings=normalize_embeddings,
            device=requested_device,
        )
        widget_embeddings = encode_texts(
            tokenizer,
            model,
            candidates["widget_docs"],
            role="passage",
            batch_size=args.batch_size,
            normalize_embeddings=normalize_embeddings,
            device=requested_device,
        )
        transition_embeddings = encode_texts(
            tokenizer,
            model,
            candidates["transition_docs"],
            role="passage",
            batch_size=args.batch_size,
            normalize_embeddings=normalize_embeddings,
            device=requested_device,
        )
    except Exception as error:
        if not (allow_cpu_fallback and is_cuda_oom_error(error)):
            raise

        print("CUDA out of memory detected. Falling back to CPU inference.")
        if "model" in locals():
            del model
        gc.collect()
        clear_cuda_memory()

        tokenizer, model = load_model("cpu")
        requirement_embeddings = encode_texts(
            tokenizer,
            model,
            requirement_texts,
            role="query",
            batch_size=args.batch_size,
            normalize_embeddings=normalize_embeddings,
            device="cpu",
        )
        widget_embeddings = encode_texts(
            tokenizer,
            model,
            candidates["widget_docs"],
            role="passage",
            batch_size=args.batch_size,
            normalize_embeddings=normalize_embeddings,
            device="cpu",
        )
        transition_embeddings = encode_texts(
            tokenizer,
            model,
            candidates["transition_docs"],
            role="passage",
            batch_size=args.batch_size,
            normalize_embeddings=normalize_embeddings,
            device="cpu",
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
