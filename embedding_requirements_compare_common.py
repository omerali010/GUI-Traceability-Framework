"""Shared pipeline for contextual embedding traceability methods."""

import json
import gc
import os
import platform
import threading
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from candidate_construction import (
    DEFAULT_DATA_PATH,
    DEFAULT_REQ_PATH,
    load_widget_and_transition_candidates,
    read_requirements,
)
from results_common import DEFAULT_TOP_K, invert_ranked_results

# Conservative free-memory checks used before trying CUDA inference.
MODEL_MIN_FREE_GPU_GB = {
    "Qwen/Qwen3-Embedding-4B": 10.0,
    "Qwen/Qwen3-Embedding-0.6B": 4.0,
    "jinaai/jina-embeddings-v3-hf": 4.0,
    "it-just-works/stella_en_1.5B_v5_bf16": 9.0,
}

PACKAGE_VERSION_NAMES = [
    "numpy",
    "torch",
    "transformers",
    "sentence-transformers",
    "huggingface-hub",
    "tokenizers",
    "peft",
]


def load_candidates(data_path: str) -> Dict[str, Any]:
    """Load the candidate text fields needed by embedding methods."""
    candidates = load_widget_and_transition_candidates(data_path)
    return {
        "widget_docs": candidates["widget_docs"],
        "widget_meta": candidates["widget_meta"],
        "transition_docs": candidates["transition_docs"],
        "transition_meta": candidates["transition_meta"],
    }


def package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package_name in PACKAGE_VERSION_NAMES:
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = "not_installed"
    return versions


def cuda_environment() -> Dict[str, Any]:
    cuda_info: Dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        cuda_info["devices"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return cuda_info


class RuntimeMemoryTracker:
    """Track runtime, process memory, and CUDA peak memory for CE results."""

    def __init__(self, sample_interval_seconds: float = 0.1) -> None:
        self.sample_interval_seconds = sample_interval_seconds
        self.started_at_utc = datetime.now(timezone.utc).isoformat()
        self.start_time = time.perf_counter()
        self.process = self._load_process()
        self.rss_start_bytes = self._rss_bytes()
        self.rss_peak_bytes = self.rss_start_bytes
        self.stop_event = threading.Event()
        self.sampler: Optional[threading.Thread] = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        if self.process is not None:
            self.sampler = threading.Thread(target=self._sample_rss, daemon=True)
            self.sampler.start()

    def _load_process(self) -> Any:
        try:
            import psutil
        except Exception:
            return None
        return psutil.Process(os.getpid())

    def _rss_bytes(self) -> Optional[int]:
        if self.process is None:
            return None
        try:
            return int(self.process.memory_info().rss)
        except Exception:
            return None

    def _sample_rss(self) -> None:
        while not self.stop_event.wait(self.sample_interval_seconds):
            rss_bytes = self._rss_bytes()
            if rss_bytes is None:
                continue
            if self.rss_peak_bytes is None or rss_bytes > self.rss_peak_bytes:
                self.rss_peak_bytes = rss_bytes

    def stop(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.stop_event.set()
        if self.sampler is not None:
            self.sampler.join(timeout=1.0)

        ended_at_utc = datetime.now(timezone.utc).isoformat()
        total_seconds = round(time.perf_counter() - self.start_time, 6)
        rss_end_bytes = self._rss_bytes()
        if rss_end_bytes is not None and (
            self.rss_peak_bytes is None or rss_end_bytes > self.rss_peak_bytes
        ):
            self.rss_peak_bytes = rss_end_bytes

        cuda_peak_allocated_bytes: Optional[int] = None
        cuda_peak_reserved_bytes: Optional[int] = None
        if torch.cuda.is_available():
            cuda_peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
            cuda_peak_reserved_bytes = int(torch.cuda.max_memory_reserved())

        runtime = {
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": ended_at_utc,
            "total_seconds": total_seconds,
        }
        memory = {
            "process_rss_start_bytes": self.rss_start_bytes,
            "process_rss_end_bytes": rss_end_bytes,
            "process_rss_peak_bytes": self.rss_peak_bytes,
            "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
            "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
        }
        return runtime, memory


def build_environment_metadata(
    requirements_path: str,
    data_path: str,
    output_path: str,
    runtime: Optional[Dict[str, Any]] = None,
    memory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build environment metadata stored in CE result JSON files."""
    environment = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
        },
        "packages": package_versions(),
        "cuda": cuda_environment(),
        "run": {
            "requirements_path": requirements_path,
            "data_path": data_path,
            "output_path": output_path,
        },
    }
    if runtime is not None:
        environment["runtime"] = runtime
    if memory is not None:
        environment["memory"] = memory
    return environment


def write_ce_results(output_path: str, results: List[Dict[str, Any]], environment: Dict[str, Any]) -> None:
    """Write CE rankings to JSON and create the output folder if needed."""
    payload = {
        "Environment": environment,
        "Results": results,
    }
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)


def default_backward_output_path(output_path: str) -> str:
    if output_path.lower().endswith(".json"):
        return f"{output_path[:-5]}_backward.json"
    return f"{output_path}_backward.json"

def is_cuda_oom_error(error: BaseException) -> bool:
    """Return whether an exception looks like a CUDA out-of-memory failure."""
    message = str(error).lower()
    return (
        "cuda out of memory" in message
        or "cuda error: out of memory" in message
        or "torch.outofmemoryerror" in message
    )


def clear_cuda_memory() -> None:
    """Clear cached CUDA memory after a failed GPU attempt."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def required_free_gpu_gb(model_name: str) -> float:
    """Return the heuristic free GPU memory needed for an embedding model."""
    return MODEL_MIN_FREE_GPU_GB.get(model_name, 8.0)


def select_execution_device(model_name: str, requested_device: Optional[str]) -> str:
    """Choose CPU or GPU before loading the embedding model."""
    if requested_device and str(requested_device).lower() == "cpu":
        return "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    candidate_device = requested_device or "cuda"
    if not str(candidate_device).lower().startswith("cuda"):
        return str(candidate_device)

    device = torch.device(candidate_device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    required_gb = required_free_gpu_gb(model_name)

    if free_gb < required_gb:
        print(
            f"Only {free_gb:.2f} GB free on {device} ({total_gb:.2f} GB total); "
            f"{model_name} needs about {required_gb:.1f}+ GB free. Using CPU instead."
        )
        return "cpu"

    return candidate_device

def apply_instruction(texts: List[str], instruction: Optional[str]) -> List[str]:
    """Add an instruction prefix to texts when a model needs one."""
    if not instruction:
        return texts
    return [f"{instruction}\n{text}" for text in texts]


def encode_texts(
    model: Any,
    texts: List[str],
    batch_size: int,
    normalize_embeddings: bool,
    instruction: Optional[str] = None,
    prompt_name: Optional[str] = None,
    task: Optional[str] = None,
) -> np.ndarray:
    """Encode texts as dense vectors for cosine-similarity ranking."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    prepared = apply_instruction(texts, instruction)
    encode_kwargs = {
        "batch_size": batch_size,
        "convert_to_numpy": True,
        "normalize_embeddings": normalize_embeddings,
        "show_progress_bar": True,
    }
    if prompt_name is not None:
        encode_kwargs["prompt_name"] = prompt_name
    if task is not None:
        encode_kwargs["task"] = task
    embeddings = model.encode(prepared, **encode_kwargs)
    return np.asarray(embeddings, dtype=np.float32)


def load_model(model_name: str, model_revision: Optional[str], device: Optional[str]) -> Any:
    """Load a SentenceTransformer model, optionally pinned to a Hugging Face revision."""
    from sentence_transformers import SentenceTransformer

    model_kwargs = {"trust_remote_code": True}
    if model_revision:
        model_kwargs["revision"] = model_revision
    if device:
        model_kwargs["device"] = device
    return SentenceTransformer(model_name, **model_kwargs)


def cosine_rank(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    top_k: int,
) -> List[List[Any]]:
    """Rank embedded candidates against one embedded requirement."""
    if query_embedding.size == 0 or document_embeddings.size == 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    scores = query_embedding @ document_embeddings.T
    scored = []
    for index, score in enumerate(scores[0]):
        if float(score) > 0.0:
            scored.append([round(float(score), 6), metadata[index]])
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def run_embedding_pipeline(
    model_name: str,
    output_path: str,
    requirements_path: str = DEFAULT_REQ_PATH,
    data_path: str = DEFAULT_DATA_PATH,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = 8,
    device: Optional[str] = None,
    normalize_embeddings: bool = True,
    query_instruction: Optional[str] = None,
    document_instruction: Optional[str] = None,
    query_prompt_name: Optional[str] = None,
    document_prompt_name: Optional[str] = None,
    query_task: Optional[str] = None,
    document_task: Optional[str] = None,
    model_revision: Optional[str] = None,
    backward_output_path: Optional[str] = None,
    backward_field: str = "TopMatches",
    backward_top_k: Optional[int] = None,
    write_backward_output: bool = False,
) -> Dict[str, int]:
    """Run a CE method from requirements and GUI candidates to ranked JSON outputs."""
    tracker = RuntimeMemoryTracker()
    requirements = read_requirements(requirements_path)
    candidates = load_candidates(data_path)
    requirement_texts = [requirement["text"] for requirement in requirements]
    
    requested_device = select_execution_device(model_name, device)
    allow_cpu_fallback = device is None or str(device).lower().startswith("cuda")
    # GPU runs can fail late during encoding, so the whole load/encode block is retried on CPU if needed.
    try:
        model = load_model(model_name, model_revision, requested_device)
        requirement_embeddings = encode_texts(
            model,
            requirement_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            instruction=query_instruction,
            prompt_name=query_prompt_name,
            task=query_task,
        )
        widget_embeddings = encode_texts(
            model,
            candidates["widget_docs"],
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            instruction=document_instruction,
            prompt_name=document_prompt_name,
            task=document_task,
        )
        transition_embeddings = encode_texts(
            model,
            candidates["transition_docs"],
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            instruction=document_instruction,
            prompt_name=document_prompt_name,
            task=document_task,
        )
    except Exception as error:
        if not (allow_cpu_fallback and is_cuda_oom_error(error)):
            raise

        print("CUDA out of memory detected. Falling back to CPU inference.")
        if "model" in locals():
            del model
        gc.collect()
        clear_cuda_memory()

        model = load_model(model_name, model_revision, "cpu")
        requirement_embeddings = encode_texts(
            model,
            requirement_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            instruction=query_instruction,
            prompt_name=query_prompt_name,
            task=query_task,
        )
        widget_embeddings = encode_texts(
            model,
            candidates["widget_docs"],
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            instruction=document_instruction,
            prompt_name=document_prompt_name,
            task=document_task,
        )
        transition_embeddings = encode_texts(
            model,
            candidates["transition_docs"],
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            instruction=document_instruction,
            prompt_name=document_prompt_name,
            task=document_task,
        )

    results = []
    for index, requirement in enumerate(requirements):
        query_embedding = requirement_embeddings[index]
        top_widgets = cosine_rank(query_embedding, widget_embeddings, candidates["widget_meta"], top_k)
        top_transitions = cosine_rank(query_embedding, transition_embeddings, candidates["transition_meta"], top_k)
        combined = sorted(top_widgets + top_transitions, key=lambda item: item[0], reverse=True)[:top_k]
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
    resolved_backward_output_path = None
    if write_backward_output:
        resolved_backward_output_path = backward_output_path or default_backward_output_path(output_path)
        backward_results = invert_ranked_results(
            results,
            source_field=backward_field,
            top_k=backward_top_k,
        )

    runtime, memory = tracker.stop()

    environment = build_environment_metadata(
        requirements_path=requirements_path,
        data_path=data_path,
        output_path=output_path,
        runtime=runtime,
        memory=memory,
    )
    write_ce_results(output_path, results, environment)

    if resolved_backward_output_path:
        backward_environment = build_environment_metadata(
            requirements_path=requirements_path,
            data_path=data_path,
            output_path=resolved_backward_output_path,
            runtime=runtime,
            memory=memory,
        )
        write_ce_results(resolved_backward_output_path, backward_results, backward_environment)

    return {
        "requirements": len(requirements),
        "widget_candidates": len(candidates["widget_docs"]),
        "transition_candidates": len(candidates["transition_docs"]),
        "backward_candidates": len(backward_results) if write_backward_output else 0,
        "total_seconds": runtime["total_seconds"],
        "process_rss_peak_bytes": memory["process_rss_peak_bytes"],
        "cuda_peak_allocated_bytes": memory["cuda_peak_allocated_bytes"],
    }
