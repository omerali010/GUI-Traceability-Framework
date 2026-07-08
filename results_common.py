"""Shared result-writing, runtime tracking, and backward-ranking helpers."""

import json
import os
import platform
import threading
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from text_preprocessing import text_value

DEFAULT_TOP_K = 50

IR_PACKAGE_VERSION_NAMES = ["numpy", "scikit-learn", "scipy", "psutil"]


def package_versions(package_names: Iterable[str] = IR_PACKAGE_VERSION_NAMES) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = "not_installed"
    return versions


class RuntimeMemoryTracker:
    """Track wall-clock runtime and process memory for result metadata."""

    def __init__(self, sample_interval_seconds: float = 0.1) -> None:
        self.sample_interval_seconds = sample_interval_seconds
        self.started_at_utc = datetime.now(timezone.utc).isoformat()
        self.start_time = time.perf_counter()
        self.process = self._load_process()
        self.rss_start_bytes = self._rss_bytes()
        self.rss_peak_bytes = self.rss_start_bytes
        self.stop_event = threading.Event()
        self.sampler: Optional[threading.Thread] = None
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

        runtime = {
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": ended_at_utc,
            "total_seconds": total_seconds,
        }
        memory = {
            "process_rss_start_bytes": self.rss_start_bytes,
            "process_rss_end_bytes": rss_end_bytes,
            "process_rss_peak_bytes": self.rss_peak_bytes,
        }
        return runtime, memory


def build_result_environment_metadata(
    requirements_path: str,
    data_path: str,
    output_path: str,
    runtime: Optional[Dict[str, Any]] = None,
    memory: Optional[Dict[str, Any]] = None,
    package_names: Iterable[str] = IR_PACKAGE_VERSION_NAMES,
) -> Dict[str, Any]:
    """Build environment metadata stored next to each ranking output."""
    environment = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
        },
        "packages": package_versions(package_names),
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


def write_ranked_results(
    output_path: str,
    results: List[Dict[str, Any]],
    environment: Dict[str, Any],
) -> None:
    """Write ranked matches to JSON using the repository result schema."""
    payload = {
        "Environment": environment,
        "Results": results,
    }
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)


def default_backward_output_path(
    output_path: str,
    forward_prefix: str,
    backward_prefix: str,
) -> str:
    """Build the default backward filename for an IR result file."""
    path = Path(output_path)
    stem = path.stem
    if stem.startswith(forward_prefix):
        backward_name = stem.replace(forward_prefix, backward_prefix, 1)
    else:
        backward_name = f"{stem}_backward"
    return str(path.with_name(f"{backward_name}{path.suffix or '.json'}"))


def candidate_backward_key(metadata: Dict[str, Any]) -> str:
    item_type = text_value(metadata, "Type")
    if item_type == "Widget":
        widget_signature = text_value(metadata, "WidgetSignature")
        return f"W:{widget_signature}" if widget_signature else ""
    if item_type == "Transition":
        action_id = text_value(metadata, "ActionID")
        return f"T:{action_id}" if action_id else ""
    return ""


def invert_ranked_results(
    results: List[Dict[str, Any]],
    source_field: str = "TopMatches",
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Turn requirement-to-candidate rankings into candidate-to-requirement rankings."""
    grouped: Dict[str, Dict[str, Any]] = {}

    for entry in results:
        req_id = text_value(entry, "ReqID")
        requirement_text = text_value(entry, "Requirement")
        ranked_items = entry.get(source_field, [])
        if not isinstance(ranked_items, list):
            continue

        for item in ranked_items:
            if not (
                isinstance(item, list)
                and len(item) >= 2
                and isinstance(item[0], (int, float))
                and isinstance(item[1], dict)
            ):
                continue

            score = round(float(item[0]), 6)
            metadata = item[1]
            candidate_key = candidate_backward_key(metadata)
            if not candidate_key:
                continue

            if candidate_key not in grouped:
                grouped[candidate_key] = {
                    "Type": text_value(metadata, "Type"),
                    "CandidateKey": candidate_key,
                    "Candidate": metadata,
                    "TopRequirements": [],
                }

            grouped[candidate_key]["TopRequirements"].append(
                {
                    "Score": score,
                    "ReqID": req_id,
                    "Requirement": requirement_text,
                }
            )

    inverted = list(grouped.values())
    for candidate_entry in inverted:
        candidate_entry["TopRequirements"].sort(
            key=lambda requirement: requirement["Score"],
            reverse=True,
        )
        if top_k is not None:
            candidate_entry["TopRequirements"] = candidate_entry["TopRequirements"][:top_k]

    inverted.sort(
        key=lambda candidate_entry: (
            -candidate_entry["TopRequirements"][0]["Score"] if candidate_entry["TopRequirements"] else 0.0,
            candidate_entry["Type"],
            candidate_entry["CandidateKey"],
        )
    )
    return inverted
