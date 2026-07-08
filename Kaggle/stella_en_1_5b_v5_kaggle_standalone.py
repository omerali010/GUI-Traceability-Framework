"""Standalone Kaggle version of the Stella 1.5B traceability run."""

import argparse
import gc
import json
import os
import platform
import re
import threading
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "it-just-works/stella_en_1.5B_v5_bf16"

# Pin the Hugging Face model snapshot used for replication.
MODEL_REVISION = "b6f39e45892c6edd44f1e602d84b6adf8891a1e3"

DEFAULT_OUT_PATH = "stella_en_1_5b_v5_matches_Gemma3_Focus.json"

DEFAULT_TOP_K = 50

# Conservative free-memory check before trying CUDA inference.
MODEL_MIN_FREE_GPU_GB = 9.0

PACKAGE_VERSION_NAMES = [
    "numpy",
    "torch",
    "transformers",
    "sentence-transformers",
    "huggingface-hub",
    "tokenizers",
    "peft",
]

# Keep Stella inference practical on limited hardware.
MAX_LENGTH = 512

QUERY_PROMPT_NAME = "s2p_query"

TRACEABILITY_STOPWORDS = {
    "a",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "can",
    "current",
    "do",
    "does",
    "for",
    "from",
    "given",
    "have",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "loaded",
    "my",
    "of",
    "on",
    "or",
    "press",
    "that",
    "the",
    "their",
    "them",
    "then",
    "they",
    "this",
    "to",
    "up",
    "user",
    "viewing",
    "visits",
    "website",
    "when",
    "will",
    "with",
}


def normalize_text(text: str) -> str:
    """Normalize requirement and GUI text into a simple searchable form."""
    if not isinstance(text, str):
        return ""
    replacements = {
        r"\bascending\b": "ascending low high",
        r"\bdescending\b": "descending high low",
        r"\bcatalogue\b": "catalogue catalog",
        r"\bfavourites\b": "favourites favorites wishlist",
        r"\bfavorites\b": "favorites favourites wishlist",
        r"\beco-friendly\b": "eco friendly sustainability sustainable",
        r"\beco friendly\b": "eco friendly sustainability sustainable",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = "".join(ch if ord(ch) < 128 else " " for ch in text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Split normalized text into tokens and remove low-information words."""
    return [
        token
        for token in normalize_text(text).split()
        if token and token not in TRACEABILITY_STOPWORDS and (len(token) > 1 or token.isdigit())
    ]


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    """Keep unique non-empty strings in their original order."""
    seen = set()
    ordered = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


def read_requirements(path: str) -> List[Dict[str, str]]:
    """Read a requirement text file into ID/text pairs."""
    with open(path, "r", encoding="utf-8") as file_handle:
        raw_text = file_handle.read()

    blocks = [block.strip() for block in re.split(r"\n\s*\n", raw_text) if block.strip()]
    requirements = []
    for index, block in enumerate(blocks, 1):
        first_line, *rest = block.splitlines()
        match = re.match(r"^(RQ\d+)\.\s*(.*)$", first_line.strip(), flags=re.IGNORECASE)
        if match:
            req_id = match.group(1).upper()
            first_text = match.group(2).strip()
            text = "\n".join([first_text] + rest).strip()
        else:
            req_id = f"R{index}"
            text = block
        requirements.append({"id": req_id, "text": text})
    return requirements


def text_value(node: Dict[str, Any], key: str) -> str:
    """Return a stripped string value from a dictionary field."""
    value = node.get(key)
    return value.strip() if isinstance(value, str) else ""


def first_nonempty_text(node: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = text_value(node, key)
        if value:
            return value
    return ""


def selector_tokens(selector: str) -> str:
    """Lightly normalize escaped CSS selector text from TESTAR JSON."""
    if not selector:
        return ""
    selector = selector.replace("\\/", "/")
    selector = selector.replace("\\:", ":")
    return selector


def is_widget_candidate(node: Dict[str, Any]) -> bool:
    """Return whether a widget was marked as a traceability candidate."""
    return bool(node.get("TraceabilityCandidate"))


def action_lookup_keys(action: Dict[str, Any]) -> List[str]:
    return unique_preserve_order(
        [
            text_value(action, "actionId"),
            text_value(action, "AbstractID"),
        ]
    )


def transition_action_eval_id(transition: Dict[str, Any]) -> str:
    return first_nonempty_text(transition, ("ActionId", "Action"))


def transition_action_abstract_id(transition: Dict[str, Any]) -> str:
    return first_nonempty_text(transition, ("ActionWA", "Action"))


def build_widget_doc(
    state: Dict[str, Any],
    node: Dict[str, Any],
    parent: Optional[Dict[str, Any]],
    sibling_nodes: List[Dict[str, Any]],
    child_nodes: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """Build searchable text and metadata for one widget occurrence."""
    widget_id = text_value(node, "AbstractID")
    widget_concrete = text_value(node, "ConcreteID")
    state_id = text_value(state, "AbstractID")
    state_title = text_value(state, "WebTitle")
    state_href = text_value(state, "WebHref")
    widget_text = text_value(node, "WebTextContent")
    widget_tag = text_value(node, "WebTagName")
    widget_selector = selector_tokens(text_value(node, "WebCssSelector"))
    widget_href = text_value(node, "WebHref")

    parent_text = text_value(parent or {}, "WebTextContent")
    parent_tag = text_value(parent or {}, "WebTagName")
    parent_selector = selector_tokens(text_value(parent or {}, "WebCssSelector"))

    sibling_texts = unique_preserve_order(
        text_value(sibling, "WebTextContent")
        for sibling in sibling_nodes
        if sibling is not node
    )[:5]
    child_texts = unique_preserve_order(
        text_value(child, "WebTextContent")
        for child in child_nodes
    )[:5]

    parts = []
    parts.extend([widget_text] * 3 if widget_text else [])
    parts.extend([widget_href] * 2 if widget_href else [])
    parts.extend([widget_tag, widget_selector])
    parts.extend([state_title, state_href])
    parts.extend([parent_text, parent_tag, parent_selector])
    parts.extend(sibling_texts)
    parts.extend(child_texts)

    document = " ".join(part for part in parts if part)
    metadata = {
        "Type": "Widget",
        "WidgetOccurrenceID": f"{state_id}::{widget_concrete or widget_id}",
        "WidgetID": widget_id,
        "WidgetAbstractID": widget_id,
        "WidgetConcreteID": widget_concrete,
        "WidgetSignature": "||".join(
            [
                normalize_text(widget_tag),
                normalize_text(widget_selector),
                normalize_text(widget_text),
                normalize_text(widget_href),
            ]
        ),
        "StateID": state_id,
        "StateTitle": state_title,
        "StateHref": state_href,
        "WidgetText": widget_text,
        "WidgetTag": widget_tag,
        "WidgetSelector": widget_selector,
        "WidgetHref": widget_href,
        "ParentText": parent_text,
        "SiblingTexts": sibling_texts,
        "ChildTexts": child_texts,
    }
    return document, metadata


def collect_widget_documents(states: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Walk all widget trees and collect candidate widget documents."""
    documents: List[Tuple[str, Dict[str, Any]]] = []

    def visit(
        state: Dict[str, Any],
        node: Dict[str, Any],
        parent: Optional[Dict[str, Any]],
        siblings: List[Dict[str, Any]],
    ) -> None:
        children = node.get("children") if isinstance(node.get("children"), list) else []
        if is_widget_candidate(node):
            documents.append(build_widget_doc(state, node, parent, siblings, children))
        for child in children:
            if isinstance(child, dict):
                visit(state, child, node, children)

    for state in states:
        widget_tree = state.get("WidgetTree")
        if isinstance(widget_tree, list):
            for root in widget_tree:
                if isinstance(root, dict):
                    visit(state, root, None, widget_tree)
        elif isinstance(widget_tree, dict):
            visit(state, widget_tree, None, [widget_tree])
    return documents


def aggregate_widget_documents(
    widget_documents: List[Tuple[str, Dict[str, Any]]]
) -> List[Tuple[str, Dict[str, Any]]]:
    """Merge repeated widget occurrences into one logical widget candidate."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for _, meta in widget_documents:
        grouped.setdefault(meta["WidgetSignature"], []).append(meta)

    aggregated = []
    for widget_signature, occurrences in grouped.items():
        widget_texts = unique_preserve_order(meta["WidgetText"] for meta in occurrences)[:8]
        widget_tags = unique_preserve_order(meta["WidgetTag"] for meta in occurrences)[:4]
        widget_selectors = unique_preserve_order(meta["WidgetSelector"] for meta in occurrences)[:8]
        widget_hrefs = unique_preserve_order(meta["WidgetHref"] for meta in occurrences)[:8]
        widget_ids = unique_preserve_order(meta["WidgetID"] for meta in occurrences)[:10]
        widget_abstract_ids = unique_preserve_order(meta["WidgetAbstractID"] for meta in occurrences)[:10]
        widget_concrete_ids = unique_preserve_order(meta["WidgetConcreteID"] for meta in occurrences)[:20]
        parent_texts = unique_preserve_order(meta["ParentText"] for meta in occurrences)[:8]
        sibling_texts = unique_preserve_order(
            text
            for meta in occurrences
            for text in meta["SiblingTexts"]
        )[:10]
        child_texts = unique_preserve_order(
            text
            for meta in occurrences
            for text in meta["ChildTexts"]
        )[:10]
        state_ids = unique_preserve_order(meta["StateID"] for meta in occurrences)[:10]
        state_titles = unique_preserve_order(meta["StateTitle"] for meta in occurrences)[:10]
        state_hrefs = unique_preserve_order(meta["StateHref"] for meta in occurrences)[:10]

        parts = []
        for text in widget_texts:
            parts.extend([text] * 3)
        for href in widget_hrefs:
            parts.extend([href] * 2)
        parts.extend(widget_tags)
        parts.extend(widget_selectors)
        parts.extend(state_titles)
        parts.extend(state_hrefs)
        parts.extend(parent_texts)
        parts.extend(sibling_texts)
        parts.extend(child_texts)

        aggregated.append(
            (
                " ".join(part for part in parts if part),
                {
                    "Type": "Widget",
                    "WidgetSignature": widget_signature,
                    "WidgetID": widget_ids[0] if widget_ids else "",
                    "WidgetIDs": widget_ids,
                    "WidgetAbstractID": widget_abstract_ids[0] if widget_abstract_ids else "",
                    "WidgetAbstractIDs": widget_abstract_ids,
                    "WidgetConcreteID": widget_concrete_ids[0] if widget_concrete_ids else "",
                    "WidgetConcreteIDs": widget_concrete_ids,
                    "OccurrenceCount": len(occurrences),
                    "WidgetTexts": widget_texts,
                    "WidgetTags": widget_tags,
                    "WidgetSelectors": widget_selectors,
                    "WidgetHrefs": widget_hrefs,
                    "StateIDs": state_ids,
                    "StateTitles": state_titles,
                    "StateHrefs": state_hrefs,
                    "ParentTexts": parent_texts,
                    "SiblingTexts": sibling_texts,
                    "ChildTexts": child_texts,
                    "SampleOccurrenceIDs": [meta["WidgetOccurrenceID"] for meta in occurrences[:5]],
                },
            )
        )
    return aggregated


def build_source_widget_lookup(widget_metadata: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Build a lookup from state/widget IDs to widget metadata."""
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for meta in widget_metadata:
        widget_keys = unique_preserve_order(
            [
                meta.get("WidgetConcreteID", ""),
                meta.get("WidgetID", ""),
                meta.get("WidgetAbstractID", ""),
            ]
        )
        for widget_key in widget_keys:
            key = (meta["StateID"], widget_key)
            if widget_key and key not in lookup:
                lookup[key] = meta
    return lookup


def build_transition_doc(
    transition: Dict[str, Any],
    action: Dict[str, Any],
    states_by_id: Dict[str, Dict[str, Any]],
    source_widget_lookup: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """Build searchable text and metadata for one transition occurrence."""
    source_id = text_value(transition, "Source")
    target_id = text_value(transition, "Target")
    action_id = transition_action_eval_id(transition)
    action_abstract_id = transition_action_abstract_id(transition) or text_value(action, "AbstractID")
    action_concrete_id = first_nonempty_text(transition, ("ActionId",)) or text_value(action, "actionId")
    action_widget_concrete_id = first_nonempty_text(transition, ("ActionWC",)) or text_value(action, "ConcreteID")

    source_state = states_by_id.get(source_id, {})
    target_state = states_by_id.get(target_id, {})
    source_widget = {}
    for widget_key in unique_preserve_order(
        [
            action_widget_concrete_id,
            action_abstract_id,
            action_id,
        ]
    ):
        source_widget = source_widget_lookup.get((source_id, widget_key), {})
        if source_widget:
            break

    action_desc = text_value(action, "Desc")
    action_tag = text_value(action, "WebTagName")
    action_selector = selector_tokens(text_value(action, "WebCssSelector"))
    action_href = text_value(action, "WebHref")

    parts = []
    parts.extend([action_desc] * 3 if action_desc else [])
    parts.extend([action_href] * 2 if action_href else [])
    parts.extend(
        [
            action_tag,
            action_selector,
            text_value(source_state, "WebTitle"),
            text_value(source_state, "WebHref"),
            text_value(target_state, "WebTitle"),
            text_value(target_state, "WebHref"),
            source_widget.get("WidgetText", ""),
            source_widget.get("ParentText", ""),
        ]
    )
    parts.extend(source_widget.get("SiblingTexts", [])[:3])
    document = " ".join(part for part in parts if part)

    metadata = {
        "Type": "Transition",
        "TransitionID": f"{source_id}->{action_id}->{target_id}",
        "TransitionIDAbstract": f"{source_id}->{action_abstract_id}->{target_id}" if action_abstract_id else "",
        "Source": source_id,
        "Target": target_id,
        "ActionID": action_id,
        "ActionConcreteID": action_concrete_id,
        "ActionAbstractID": action_abstract_id,
        "ActionWidgetConcreteID": action_widget_concrete_id,
        "ActionDesc": action_desc,
        "ActionTag": action_tag,
        "ActionSelector": action_selector,
        "ActionHref": action_href,
        "SourceTitle": text_value(source_state, "WebTitle"),
        "SourceHref": text_value(source_state, "WebHref"),
        "TargetTitle": text_value(target_state, "WebTitle"),
        "TargetHref": text_value(target_state, "WebHref"),
        "ActionWidgetText": source_widget.get("WidgetText", ""),
    }
    return document, metadata


def collect_transition_documents(
    states: List[Dict[str, Any]],
    actions: List[Dict[str, Any]],
    transitions: List[Dict[str, Any]],
    widget_metadata: List[Dict[str, Any]],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Collect transition documents and attach nearby widget context when available."""
    action_map: Dict[str, Dict[str, Any]] = {}
    for action in actions:
        for key in action_lookup_keys(action):
            if key not in action_map:
                action_map[key] = action
    states_by_id = {text_value(state, "AbstractID"): state for state in states}
    source_widget_lookup = build_source_widget_lookup(widget_metadata)

    documents = []
    for transition in transitions:
        action_id = transition_action_eval_id(transition) or transition_action_abstract_id(transition)
        action = action_map.get(action_id, {})
        if not action:
            action = action_map.get(transition_action_abstract_id(transition), {})
        documents.append(
            build_transition_doc(transition, action, states_by_id, source_widget_lookup)
        )
    return documents


def aggregate_transition_documents(
    transition_documents: List[Tuple[str, Dict[str, Any]]]
) -> List[Tuple[str, Dict[str, Any]]]:
    """Merge repeated transition occurrences into one logical transition candidate."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for _, meta in transition_documents:
        grouped.setdefault(meta["ActionID"], []).append(meta)

    aggregated = []
    for action_id, transitions in grouped.items():
        action_descs = unique_preserve_order(meta["ActionDesc"] for meta in transitions)[:5]
        action_concrete_ids = unique_preserve_order(meta["ActionConcreteID"] for meta in transitions)[:5]
        action_abstract_ids = unique_preserve_order(meta["ActionAbstractID"] for meta in transitions)[:5]
        action_widget_concrete_ids = unique_preserve_order(meta["ActionWidgetConcreteID"] for meta in transitions)[:10]
        action_tags = unique_preserve_order(meta["ActionTag"] for meta in transitions)[:4]
        action_selectors = unique_preserve_order(meta["ActionSelector"] for meta in transitions)[:5]
        action_hrefs = unique_preserve_order(meta["ActionHref"] for meta in transitions)[:5]
        source_titles = unique_preserve_order(meta["SourceTitle"] for meta in transitions)[:8]
        source_hrefs = unique_preserve_order(meta["SourceHref"] for meta in transitions)[:8]
        target_titles = unique_preserve_order(meta["TargetTitle"] for meta in transitions)[:8]
        target_hrefs = unique_preserve_order(meta["TargetHref"] for meta in transitions)[:8]
        action_widget_texts = unique_preserve_order(meta["ActionWidgetText"] for meta in transitions)[:8]

        parts = []
        for desc in action_descs:
            parts.extend([desc] * 3)
        for href in action_hrefs:
            parts.extend([href] * 2)
        parts.extend(action_tags)
        parts.extend(action_selectors)
        parts.extend(source_titles)
        parts.extend(source_hrefs)
        parts.extend(target_titles)
        parts.extend(target_hrefs)
        parts.extend(action_widget_texts)

        aggregated.append(
            (
                " ".join(part for part in parts if part),
                {
                    "Type": "Transition",
                    "ActionID": action_id,
                    "ActionConcreteID": action_concrete_ids[0] if action_concrete_ids else "",
                    "ActionConcreteIDs": action_concrete_ids,
                    "ActionAbstractID": action_abstract_ids[0] if action_abstract_ids else "",
                    "ActionAbstractIDs": action_abstract_ids,
                    "ActionWidgetConcreteIDs": action_widget_concrete_ids,
                    "TransitionCount": len(transitions),
                    "ActionDesc": action_descs[0] if action_descs else "",
                    "ActionDescs": action_descs,
                    "ActionTags": action_tags,
                    "ActionSelectors": action_selectors,
                    "ActionHrefs": action_hrefs,
                    "SourceTitles": source_titles,
                    "SourceHrefs": source_hrefs,
                    "TargetTitles": target_titles,
                    "TargetHrefs": target_hrefs,
                    "ActionWidgetTexts": action_widget_texts,
                    "SampleTransitionIDs": [meta["TransitionID"] for meta in transitions[:5]],
                },
            )
        )
    return aggregated


def load_widget_and_transition_candidates(data_path: str) -> Dict[str, Any]:
    """Load a filtered GUI model and return aggregated widget and transition candidates."""
    with open(data_path, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    states = data.get("ConcreteState", [])
    actions = data.get("ConcreteAction", [])
    transitions = data.get("ConcreteTransitions", [])

    widget_occurrences = collect_widget_documents(states)
    widget_docs_with_meta = aggregate_widget_documents(widget_occurrences)
    widget_docs = [doc for doc, _ in widget_docs_with_meta]
    widget_meta = [meta for _, meta in widget_docs_with_meta]

    occurrence_meta = [meta for _, meta in widget_occurrences]
    transition_occurrences = collect_transition_documents(states, actions, transitions, occurrence_meta)
    transition_docs_with_meta = aggregate_transition_documents(transition_occurrences)
    transition_docs = [doc for doc, _ in transition_docs_with_meta]
    transition_meta = [meta for _, meta in transition_docs_with_meta]

    return {
        "states": states,
        "actions": actions,
        "transitions": transitions,
        "widget_occurrences": widget_occurrences,
        "widget_docs": widget_docs,
        "widget_meta": widget_meta,
        "transition_occurrences": transition_occurrences,
        "transition_docs": transition_docs,
        "transition_meta": transition_meta,
    }


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


def select_execution_device(requested_device: Optional[str]) -> str:
    """Choose CPU or GPU before loading Stella."""
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

    if free_gb < MODEL_MIN_FREE_GPU_GB:
        print(
            f"Only {free_gb:.2f} GB free on {device} ({total_gb:.2f} GB total); "
            f"{MODEL_NAME} needs about {MODEL_MIN_FREE_GPU_GB:.1f}+ GB free. Using CPU instead."
        )
        return "cpu"

    return candidate_device


def apply_instruction(texts: List[str], instruction: Optional[str]) -> List[str]:
    """Add an instruction prefix to texts when one is provided."""
    if not instruction:
        return texts
    return [f"{instruction}\n{text}" for text in texts]


def load_model(device: Optional[str]) -> Any:
    """Load the pinned Stella SentenceTransformer model."""
    model_kwargs = {"trust_remote_code": True}
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
    instruction: Optional[str] = None,
    prompt_name: Optional[str] = None,
) -> np.ndarray:
    """Encode texts with Stella and return float32 NumPy embeddings."""
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
    embeddings = model.encode(prepared, **encode_kwargs)
    return np.asarray(embeddings, dtype=np.float32)


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
    """Track runtime, process memory, and CUDA peak memory for result metadata."""

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
    requirements_path: Optional[str],
    data_path: Optional[str],
    output_path: str,
    runtime: Optional[Dict[str, Any]] = None,
    memory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
    """Write CE rankings to JSON using the replication result schema."""
    payload = {
        "Environment": environment,
        "Results": results,
    }
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)


def default_backward_output_path(output_path: str) -> str:
    if output_path.lower().endswith(".json"):
        return f"{output_path[:-5]}_backward.json"
    return f"{output_path}_backward.json"


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


def run_pipeline(
    requirements_path: Optional[str],
    data_path: Optional[str],
    output_path: str,
    top_k: int,
    batch_size: int,
    device: Optional[str],
    normalize_embeddings: bool,
    query_instruction: Optional[str],
    document_instruction: Optional[str],
    backward_output_path: Optional[str],
    backward_field: str,
    backward_top_k: Optional[int],
    write_backward_output: bool,
) -> Dict[str, Any]:
    """Run the complete standalone Stella pipeline and write JSON outputs."""
    tracker = RuntimeMemoryTracker()
    requirements = read_requirements(requirements_path)
    candidates = load_widget_and_transition_candidates(data_path)
    requirement_texts = [requirement["text"] for requirement in requirements]

    requested_device = select_execution_device(device)
    allow_cpu_fallback = device is None or str(device).lower().startswith("cuda")

    try:
        model = load_model(requested_device)
        requirement_embeddings = encode_texts(
            model,
            requirement_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            instruction=query_instruction,
            prompt_name=QUERY_PROMPT_NAME,
        )
        widget_embeddings = encode_texts(
            model,
            candidates["widget_docs"],
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            instruction=document_instruction,
        )
        transition_embeddings = encode_texts(
            model,
            candidates["transition_docs"],
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            instruction=document_instruction,
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
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            instruction=query_instruction,
            prompt_name=QUERY_PROMPT_NAME,
        )
        widget_embeddings = encode_texts(
            model,
            candidates["widget_docs"],
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            instruction=document_instruction,
        )
        transition_embeddings = encode_texts(
            model,
            candidates["transition_docs"],
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            instruction=document_instruction,
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
        "requirements_path": requirements_path,
        "data_path": data_path,
        "requirements": len(requirements),
        "widget_candidates": len(candidates["widget_docs"]),
        "transition_candidates": len(candidates["transition_docs"]),
        "backward_candidates": len(backward_results),
        "total_seconds": runtime["total_seconds"],
        "process_rss_peak_bytes": memory["process_rss_peak_bytes"],
        "cuda_peak_allocated_bytes": memory["cuda_peak_allocated_bytes"],
    }


def parse_args() -> argparse.Namespace:
    """Parse Kaggle-friendly command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", default="/kaggle/input/datasets/omeralicelikci/kaggle-data/Kaggle Data2/Requirements focus group.txt")
    parser.add_argument("--data", default="/kaggle/input/datasets/omeralicelikci/kaggle-data/Kaggle Data2/Filtered models/filtered_elements_gpt5_focus.json")
    parser.add_argument("--output", default="/kaggle/working/stella_en_1_5b_v5_matches_Focus_Gpt5.json")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--query-instruction", default=None)
    parser.add_argument("--document-instruction", default=None)
    parser.add_argument("--backward-output", default=None)
    parser.add_argument("--backward-field", default="TopMatches", choices=["TopMatches", "TopWidgets", "TopTransitions"])
    parser.add_argument("--backward-top-k", type=int, default=None)
    parser.add_argument("--no-backward-output", action="store_true")
    return parser.parse_known_args()[0]


def main() -> None:
    """Run the standalone Stella script and print a short summary."""
    args = parse_args()
    stats = run_pipeline(
        requirements_path=args.requirements,
        data_path=args.data,
        output_path=args.output,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device=args.device,
        normalize_embeddings=not args.no_normalize,
        query_instruction=args.query_instruction,
        document_instruction=args.document_instruction,
        backward_output_path=args.backward_output,
        backward_field=args.backward_field,
        backward_top_k=args.backward_top_k,
        write_backward_output=not args.no_backward_output,
    )
    print(
        f"Wrote {args.output} "
        f"({stats['requirements']} requirements, {stats['widget_candidates']} widget candidates, "
        f"{stats['transition_candidates']} transition candidates)"
    )
    if not args.no_backward_output:
        print(f"Wrote backward output ({stats['backward_candidates']} GUI candidates with at least one requirement match)")
    print(f"Requirements: {stats['requirements_path']}")
    print(f"Data: {stats['data_path']}")


if __name__ == "__main__":
    main()
