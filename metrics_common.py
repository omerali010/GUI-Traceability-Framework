"""Compute traceability metrics and write the CSVs used by the thesis figures."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from candidate_construction import load_widget_and_transition_candidates
from text_preprocessing import normalize_text, selector_tokens


DEFAULT_GROUND_TRUTH = "Ground Truth"

REQUIRED_GT_HEADERS = [
    "Requirement_ID",
    "Requirement_Text",
    "Relevant_Transition_ActionId_Concrete",
    "Relevant_Transition_ActionId_Abstract",
    "Transition_Notes",
    "Relevant_Widget_AbstractID",
    "Relevant_Widget_ConcreteID",
    "Relevant_Widget_Selector",
    "Relevant_Widget_Text",
    "Relevant_Widget_Tag",
    "Relevant_Widget_Href",
    "Widget_Notes",
    "Relevance_Type",
]
OPTIONAL_GT_HEADERS = {"Relevant_Widget_StateID"}


def clean(value: Any) -> str:
    return str(value or "").strip()


def requirement_sort_key(req_id: str) -> Tuple[int, str]:
    match = re.search(r"\d+", req_id)
    return (int(match.group(0)) if match else 10**9, req_id)


def read_ground_truth_csv(path: Path) -> Tuple[List[str], List[Tuple[int, Dict[str, str]]]]:
    """Read an exported ground-truth CSV and preserve CSV line numbers for warnings."""
    dataframe = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    headers = [clean(header) for header in dataframe.columns]
    dataframe.columns = headers
    rows: List[Tuple[int, Dict[str, str]]] = []
    for index, raw_row in dataframe.iterrows():
        row = {header: clean(raw_row.get(header, "")) for header in headers}
        if any(row.values()):
            rows.append((int(index) + 2, row))
    return headers, rows


def normalize_selector(selector: str) -> str:
    """Normalize JSON/CSV escaping while preserving CSS escape semantics."""
    selector = clean(selector)
    while "\\\\" in selector:
        selector = selector.replace("\\\\", "\\")
    return selector_tokens(selector)


def selector_matches(left: str, right: str) -> bool:
    return normalize_selector(left) == normalize_selector(right)


def widget_signature(tag: str, selector: str, text: str, href: str) -> str:
    return "||".join(
        [
            normalize_text(tag),
            normalize_text(normalize_selector(selector)),
            normalize_text(text),
            normalize_text(href),
        ]
    )


def iter_widget_nodes(data: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    def walk(node: Any, state_id: str) -> Iterable[Dict[str, str]]:
        if isinstance(node, dict):
            yield {
                "StateID": state_id,
                "AbstractID": clean(node.get("AbstractID")),
                "ConcreteID": clean(node.get("ConcreteID")),
                "WebCssSelector": clean(node.get("WebCssSelector")),
                "WebTextContent": clean(node.get("WebTextContent")),
                "WebTagName": clean(node.get("WebTagName")),
                "WebHref": clean(node.get("WebHref")),
                "TraceabilityCandidate": str(bool(node.get("TraceabilityCandidate"))),
            }
            children = node.get("children")
            if isinstance(children, list):
                for child in children:
                    yield from walk(child, state_id)
        elif isinstance(node, list):
            for child in node:
                yield from walk(child, state_id)

    for state in data.get("ConcreteState", []):
        state_id = clean(state.get("AbstractID"))
        yield from walk(state.get("WidgetTree"), state_id)


def build_widget_occurrence_index(model_path: Path) -> Dict[str, List[Dict[str, str]]]:
    with model_path.open("r", encoding="utf-8") as file_handle:
        model = json.load(file_handle)

    by_concrete_id: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for occurrence in iter_widget_nodes(model):
        concrete_id = occurrence["ConcreteID"]
        if concrete_id:
            by_concrete_id[concrete_id].append(occurrence)
    return by_concrete_id


def widget_row_matches_occurrence(row: Dict[str, str], occurrence: Dict[str, str]) -> bool:
    if row["Relevant_Widget_AbstractID"] and row["Relevant_Widget_AbstractID"] != occurrence["AbstractID"]:
        return False
    if row["Relevant_Widget_Selector"] and not selector_matches(
        row["Relevant_Widget_Selector"], occurrence["WebCssSelector"]
    ):
        return False
    if row["Relevant_Widget_Text"] and row["Relevant_Widget_Text"] != occurrence["WebTextContent"]:
        return False
    if row["Relevant_Widget_Tag"] and row["Relevant_Widget_Tag"] != occurrence["WebTagName"]:
        return False
    if row["Relevant_Widget_Href"] and row["Relevant_Widget_Href"] != occurrence["WebHref"]:
        return False
    if row.get("Relevant_Widget_StateID", "") and row["Relevant_Widget_StateID"] != occurrence["StateID"]:
        return False
    return True


def derive_widget_gold_key(
    row: Dict[str, str],
    widget_index: Dict[str, List[Dict[str, str]]],
) -> Tuple[str, str]:
    concrete_id = row["Relevant_Widget_ConcreteID"]
    occurrences = widget_index.get(concrete_id, [])
    matching_occurrences = [
        occurrence for occurrence in occurrences if widget_row_matches_occurrence(row, occurrence)
    ]

    if matching_occurrences:
        occurrence = matching_occurrences[0]
        return (
            widget_signature(
                occurrence["WebTagName"],
                occurrence["WebCssSelector"],
                occurrence["WebTextContent"],
                occurrence["WebHref"],
            ),
            "model",
        )

    if occurrences:
        # Use the model value if the ID exists but the optional disambiguating fields differ.
        occurrence = occurrences[0]
        return (
            widget_signature(
                occurrence["WebTagName"],
                occurrence["WebCssSelector"],
                occurrence["WebTextContent"],
                occurrence["WebHref"],
            ),
            "model_fallback",
        )

    # Last fallback: keep the row evaluable, but record that it could not be tied back to the model.
    return (
        widget_signature(
            row["Relevant_Widget_Tag"],
            row["Relevant_Widget_Selector"],
            row["Relevant_Widget_Text"],
            row["Relevant_Widget_Href"],
        ),
        "row_fallback",
    )


def build_ground_truth(
    rows: Sequence[Tuple[int, Dict[str, str]]],
    widget_index: Dict[str, List[Dict[str, str]]],
) -> Tuple[
    Dict[str, Dict[str, Set[str]]],
    Dict[str, Dict[str, Dict[str, Set[str]]]],
    List[Dict[str, str]],
]:
    """Build the forward ground truth sets used for each evaluation view."""
    truth: Dict[str, Dict[str, Set[str]]] = defaultdict(
        lambda: {
            "actions_all": set(),
            "actions_with_linked_widget": set(),
            "actions_orphan": set(),
            "widgets_linked_resolved": set(),
            "widgets_functional_resolved": set(),
            "widgets_all_resolved": set(),
            "combined_actions_and_linked_widgets": set(),
            "combined_actions_and_widgets": set(),
        }
    )
    aliases: Dict[str, Dict[str, Dict[str, Set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    warnings: List[Dict[str, str]] = []

    def add_alias(req_id: str, truth_key: str, alias: str, canonical: str) -> None:
        alias = clean(alias)
        canonical = clean(canonical)
        if alias and canonical:
            aliases[req_id][truth_key][alias].add(canonical)

    for row_number, row in rows:
        req_id = row["Requirement_ID"]
        relevance_type = row["Relevance_Type"]
        transition_id = row["Relevant_Transition_ActionId_Concrete"]
        transition_abstract_id = row["Relevant_Transition_ActionId_Abstract"]
        widget_concrete_id = row["Relevant_Widget_ConcreteID"]

        if transition_id or transition_abstract_id:
            canonical_transition_id = transition_id or transition_abstract_id
            transition_key = f"T:{canonical_transition_id}"
            truth[req_id]["actions_all"].add(canonical_transition_id)
            truth[req_id]["combined_actions_and_linked_widgets"].add(transition_key)
            truth[req_id]["combined_actions_and_widgets"].add(transition_key)
            if relevance_type == "Linked" and widget_concrete_id:
                truth[req_id]["actions_with_linked_widget"].add(canonical_transition_id)
            if relevance_type == "Orphan-Action":
                truth[req_id]["actions_orphan"].add(canonical_transition_id)

            for action_id in {transition_id, transition_abstract_id}:
                add_alias(req_id, "actions_all", action_id, canonical_transition_id)
                add_alias(req_id, "combined_actions_and_linked_widgets", f"T:{action_id}", transition_key)
                add_alias(req_id, "combined_actions_and_widgets", f"T:{action_id}", transition_key)
                if relevance_type == "Linked" and widget_concrete_id:
                    add_alias(req_id, "actions_with_linked_widget", action_id, canonical_transition_id)
                if relevance_type == "Orphan-Action":
                    add_alias(req_id, "actions_orphan", action_id, canonical_transition_id)

        if relevance_type in {"Linked", "Functional"} and widget_concrete_id:
            widget_key, source = derive_widget_gold_key(row, widget_index)
            prefixed_widget_key = f"W:{widget_key}"
            truth[req_id]["widgets_all_resolved"].add(widget_key)
            truth[req_id]["combined_actions_and_widgets"].add(prefixed_widget_key)

            if relevance_type == "Linked":
                truth[req_id]["widgets_linked_resolved"].add(widget_key)
                truth[req_id]["combined_actions_and_linked_widgets"].add(prefixed_widget_key)
            if relevance_type == "Functional":
                truth[req_id]["widgets_functional_resolved"].add(widget_key)

            if source != "model":
                warnings.append(
                    {
                        "row": str(row_number),
                        "requirement_id": req_id,
                        "warning": source,
                        "widget_concrete_id": widget_concrete_id,
                        "widget_selector": row["Relevant_Widget_Selector"],
                    }
                )

    return truth, aliases, warnings


def transition_prediction_keys(meta: Dict[str, Any]) -> Set[str]:
    keys = {
        clean(meta.get("ActionConcreteID")),
        clean(meta.get("ActionID")),
        clean(meta.get("ActionAbstractID")),
    }
    keys.update(clean(value) for value in meta.get("ActionConcreteIDs", []) if clean(value))
    keys.update(clean(value) for value in meta.get("ActionAbstractIDs", []) if clean(value))
    return {key for key in keys if key}


def widget_prediction_keys(meta: Dict[str, Any]) -> Set[str]:
    signature = clean(meta.get("WidgetSignature"))
    if signature:
        return {signature}
    return set()


def combined_prediction_keys(meta: Dict[str, Any]) -> Set[str]:
    item_type = clean(meta.get("Type"))
    if item_type == "Transition":
        return {f"T:{key}" for key in transition_prediction_keys(meta)}
    if item_type == "Widget":
        return {f"W:{key}" for key in widget_prediction_keys(meta)}
    return set()


def ranked_meta(result_entry: Dict[str, Any], field_name: str, k: int) -> List[Dict[str, Any]]:
    ranked_items = result_entry.get(field_name, [])
    metadata: List[Dict[str, Any]] = []
    for item in ranked_items[:k]:
        if isinstance(item, list) and len(item) >= 2 and isinstance(item[1], dict):
            metadata.append(item[1])
        elif isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], dict):
            metadata.append(item[1])
    return metadata


def apply_aliases(prediction_keys: Set[str], alias_map: Dict[str, Set[str]]) -> Set[str]:
    normalized_keys: Set[str] = set()
    for key in prediction_keys:
        normalized_keys.update(alias_map.get(key, {key}))
    return normalized_keys


def calculate_ranking_metrics(ranked_key_sets: List[Set[str]], gold_keys: Set[str], k: int) -> Dict[str, float]:
    """Calculate precision, recall, F1, average precision, and reciprocal rank at k."""
    if not gold_keys:
        return {
            "gold_count": 0,
            "retrieved_count": len(ranked_key_sets[:k]),
            "hits": 0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "f1_at_k": 0.0,
            "ap_at_k": 0.0,
            "rr_at_k": 0.0,
        }

    hits = 0
    first_relevant_rank = 0
    ap_sum = 0.0
    matched_gold: Set[str] = set()

    for rank, prediction_keys in enumerate(ranked_key_sets[:k], start=1):
        new_matches = (prediction_keys & gold_keys) - matched_gold
        if not new_matches:
            continue

        hits += 1
        matched_gold.update(new_matches)
        ap_sum += hits / rank
        if first_relevant_rank == 0:
            first_relevant_rank = rank

    precision = hits / k
    recall = len(matched_gold) / len(gold_keys)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    ap = ap_sum / min(len(gold_keys), k)
    rr = 0.0 if first_relevant_rank == 0 else 1 / first_relevant_rank

    return {
        "gold_count": len(gold_keys),
        "retrieved_count": len(ranked_key_sets[:k]),
        "hits": hits,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
        "ap_at_k": ap,
        "rr_at_k": rr,
    }


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_float(value: Any) -> float:
    try:
        if value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: Any) -> int:
    try:
        if value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def project_rows(
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str],
    rename_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    projected_rows: List[Dict[str, Any]] = []
    rename_map = rename_map or {}
    for row in rows:
        projected_row: Dict[str, Any] = {}
        for column in columns:
            projected_row[rename_map.get(column, column)] = row.get(column, "")
        projected_rows.append(projected_row)
    return projected_rows


def summarize_per_query(per_query_rows: Sequence[Dict[str, Any]], average_scope: str = "gold") -> List[Dict[str, Any]]:
    if not per_query_rows:
        return []
    if average_scope not in {"gold", "all"}:
        raise ValueError(f"Unsupported average_scope: {average_scope}")

    dataframe = pd.DataFrame(per_query_rows)
    dataframe["direction"] = dataframe.get("direction", "forward")
    dataframe["k"] = dataframe["k"].astype(int)
    dataframe["gold_count"] = dataframe["gold_count"].astype(int)
    dataframe["hits"] = dataframe["hits"].astype(int)
    for metric in ["precision_at_k", "recall_at_k", "f1_at_k", "ap_at_k", "rr_at_k"]:
        dataframe[metric] = dataframe[metric].astype(float)

    group_columns = ["result_file", "method", "direction", "evaluation", "k"]

    summary_rows: List[Dict[str, Any]] = []
    for (result_file, method, direction, evaluation, k), group in dataframe.groupby(group_columns, sort=True):
        rows_with_gold = group[group["gold_count"] > 0]
        averaging_rows = rows_with_gold if average_scope == "gold" else group
        denominator = len(averaging_rows)
        hit_count = int((averaging_rows["hits"] > 0).sum()) if denominator else 0
        query_unit = str(group["query_unit"].iloc[0]) if "query_unit" in group.columns and not group.empty else ""
        summary_rows.append(
            {
                "average_scope": average_scope,
                "direction": direction,
                "result_file": result_file,
                "method": method,
                "evaluation": evaluation,
                "k": int(k),
                "query_unit": query_unit or "requirement",
                "queries_total": int(len(group)),
                "queries_with_gold": int(len(rows_with_gold)),
                "queries_without_gold": int(len(group) - len(rows_with_gold)),
                "candidate_queries_total": int(len(group)),
                "candidate_queries_with_gold": int(len(rows_with_gold)),
                "candidate_queries_without_gold": int(len(group) - len(rows_with_gold)),
                "requirements_total": int(len(group)),
                "requirements_with_gold": int(len(rows_with_gold)),
                "gold_links_total": int(rows_with_gold["gold_count"].sum()),
                "gold_candidates_total": int(rows_with_gold["gold_count"].sum()),
                "hits_total": int(rows_with_gold["hits"].sum()),
                "hit_at_k": hit_count / denominator if denominator else 0.0,
                "precision_at_k": mean(averaging_rows["precision_at_k"].tolist()) if denominator else 0.0,
                "recall_at_k": mean(averaging_rows["recall_at_k"].tolist()) if denominator else 0.0,
                "f1_at_k": mean(averaging_rows["f1_at_k"].tolist()) if denominator else 0.0,
                "map_at_k": mean(averaging_rows["ap_at_k"].tolist()) if denominator else 0.0,
                "mrr_at_k": mean(averaging_rows["rr_at_k"].tolist()) if denominator else 0.0,
            }
        )
    return summary_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write rows to CSV while preserving the first-seen column order."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    pd.DataFrame(rows, columns=fieldnames).to_csv(path, index=False, encoding="utf-8")


def evaluate_result_file(
    result_file: Path,
    truth: Dict[str, Dict[str, Set[str]]],
    aliases: Dict[str, Dict[str, Dict[str, Set[str]]]],
    k: int,
) -> List[Dict[str, Any]]:
    """Evaluate one forward result JSON against the ground truth."""
    with result_file.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if isinstance(payload, dict) and isinstance(payload.get("Results"), list):
        results = payload["Results"]
    elif isinstance(payload, list):
        results = payload
    else:
        raise ValueError(f"Unsupported result JSON schema in {result_file}")

    method = result_file.parent.name
    per_requirement_rows: List[Dict[str, Any]] = []

    evaluations = [
        ("actions_all", "actions_all", "TopTransitions", transition_prediction_keys),
        ("actions_with_linked_widget", "actions_with_linked_widget", "TopTransitions", transition_prediction_keys),
        ("actions_orphan", "actions_orphan", "TopTransitions", transition_prediction_keys),
        ("widgets_linked_resolved", "widgets_linked_resolved", "TopWidgets", widget_prediction_keys),
        ("widgets_functional_resolved", "widgets_functional_resolved", "TopWidgets", widget_prediction_keys),
        ("widgets_all_resolved", "widgets_all_resolved", "TopWidgets", widget_prediction_keys),
        (
            "combined_actions_and_linked_widgets",
            "combined_actions_and_linked_widgets",
            "TopMatches",
            combined_prediction_keys,
        ),
        ("combined_actions_and_widgets", "combined_actions_and_widgets", "TopMatches", combined_prediction_keys),
    ]

    for entry in results:
        req_id = clean(entry.get("ReqID"))
        for evaluation_name, truth_key, result_field, key_function in evaluations:
            alias_map = aliases[req_id][truth_key]
            ranked_keys: List[Set[str]] = []
            for meta in ranked_meta(entry, result_field, k):
                prediction_keys = key_function(meta)
                if prediction_keys:
                    ranked_keys.append(apply_aliases(prediction_keys, alias_map))
            metrics = calculate_ranking_metrics(ranked_keys, truth[req_id][truth_key], k)
            per_requirement_rows.append(
                {
                    "direction": "forward",
                    "result_file": str(result_file),
                    "method": method,
                    "evaluation": evaluation_name,
                    "query_unit": "requirement",
                    "query_id": req_id,
                    "requirement_id": req_id,
                    "k": k,
                    **metrics,
                }
            )

    return per_requirement_rows


DEFAULT_CHART_OUTPUT_DIR = "Thesis Figures"
DEFAULT_RESULT_ROOTS = [
    "Results/IR/Strict",
    "Results/IR/Relaxed",
    "Results/CE/Strict",
    "Results/CE/Relaxed",
]
DEFAULT_FORWARD_K_VALUES = "1,5,10,20,30,40,50"
DEFAULT_BACKWARD_K_VALUES = "1,3,5,10"
DEFAULT_MAIN_K = 10
DEFAULT_BACKWARD_MAIN_K = 3
DEFAULT_EVALUATION = "combined_actions_and_widgets"

RESULT_METHOD_ORDER = {
    "VSM": 0,
    "LSI": 1,
    "JSM": 2,
    "Qwen3 0.6B": 3,
    "Qwen3 4B": 4,
    "Jina v3": 5,
    "Stella 1.5B": 6,
}
RESULT_FILTER_ORDER = {"Strict": 0, "Relaxed": 1}
RESULT_DATASET_ORDER = {"Focus": 0, "LLMReq": 1}
RESULT_SOURCE_ORDER = {"GPT5": 0, "Gemma3": 1}

RESULT_CSV_ROOTS = {
    ("CE", "Strict"): "Results CE New",
    ("IR", "Strict"): "Results IR New",
    ("CE", "Relaxed"): "Results Relaxed CE New",
    ("IR", "Relaxed"): "Results Relaxed IR New",
}


def result_sort_key(row: Dict[str, Any]) -> Tuple[int, int, int, int, str]:
    method = clean(row.get("method"))
    filter_variant = clean(row.get("filter_variant"))
    dataset = clean(row.get("dataset"))
    source_model = clean(row.get("source_model"))
    label = clean(row.get("method_filter")) or method
    return (
        RESULT_METHOD_ORDER.get(method, 99),
        RESULT_FILTER_ORDER.get(filter_variant, 99),
        RESULT_DATASET_ORDER.get(dataset, 99),
        RESULT_SOURCE_ORDER.get(source_model, 99),
        label,
    )

GROUND_TRUTH_CSV_BY_SCENARIO = {
    ("Focus", "GPT5"): "Ground truth Focus_GPT5.csv",
    ("Focus", "Gemma3"): "Ground truth Focus_Gemma3.csv",
    ("LLMReq", "GPT5"): "Ground truth LLM_GPT5.csv",
    ("LLMReq", "Gemma3"): "Ground truth LLM_Gemma3.csv",
}

MODEL_BY_SCENARIO = {
    ("Strict", "Focus", "GPT5"): "Filtered models/filtered_model_GPT5_Focus.json",
    ("Strict", "Focus", "Gemma3"): "Filtered models/filtered_model_Gemma3_Focus.json",
    ("Strict", "LLMReq", "GPT5"): "Filtered models/filtered_model_GPT5_Llmreq.json",
    ("Strict", "LLMReq", "Gemma3"): "Filtered models/filtered_model_Gemma3_Llmreq.json",
    ("Relaxed", "Focus", "GPT5"): "Filtered relaxed models/filtered_model_relaxed_GPT5_Focus.json",
    ("Relaxed", "Focus", "Gemma3"): "Filtered relaxed models/filtered_model_relaxed_Gemma3_Focus.json",
    ("Relaxed", "LLMReq", "GPT5"): "Filtered relaxed models/filtered_model_relaxed_GPT5_Llmreq.json",
    ("Relaxed", "LLMReq", "Gemma3"): "Filtered relaxed models/filtered_model_relaxed_Gemma3_Llmreq.json",
}

MODEL_CSV_PATH_BY_SCENARIO = {
    ("Strict", "Focus", "GPT5"): "New filtered models/filtered_model_GPT5_focus.json",
    ("Strict", "Focus", "Gemma3"): "New filtered models/filtered_model_Gemma3_focus.json",
    ("Strict", "LLMReq", "GPT5"): "New filtered models/filtered_model_GPT5_Llmreq.json",
    ("Strict", "LLMReq", "Gemma3"): "New filtered models/filtered_model_Gemma3_Llmreq.json",
}


@dataclass(frozen=True)
class CandidateSpaceScenario:
    model_variant: str
    ground_truth_file: str
    requirement_count: int
    covered_requirement_count: int
    strict_model_file: str
    relaxed_model_file: str


CANDIDATE_SPACE_SCENARIOS: Sequence[CandidateSpaceScenario] = (
    CandidateSpaceScenario(
        "Focus-GPT5",
        "Ground truth Focus_GPT5.csv",
        30,
        30,
        "filtered_model_GPT5_Focus.json",
        "filtered_model_relaxed_GPT5_Focus.json",
    ),
    CandidateSpaceScenario(
        "Focus-Gemma3",
        "Ground truth Focus_Gemma3.csv",
        30,
        27,
        "filtered_model_Gemma3_Focus.json",
        "filtered_model_relaxed_Gemma3_Focus.json",
    ),
    CandidateSpaceScenario(
        "LLMReq-GPT5",
        "Ground truth LLM_GPT5.csv",
        36,
        36,
        "filtered_model_GPT5_Llmreq.json",
        "filtered_model_relaxed_GPT5_Llmreq.json",
    ),
    CandidateSpaceScenario(
        "LLMReq-Gemma3",
        "Ground truth LLM_Gemma3.csv",
        36,
        30,
        "filtered_model_Gemma3_Llmreq.json",
        "filtered_model_relaxed_Gemma3_Llmreq.json",
    ),
)

CANDIDATE_SPACE_COLUMNS = [
    "Model variant",
    "Filter",
    "Requirements",
    "Covered requirements",
    "Widget candidates",
    "Transition candidates",
    "Total candidates",
    "GT-relevant widgets",
    "GT-relevant transitions",
    "Relevant candidate %",
]

EVALUATION_VIEW_ROWS = [
    {
        "evaluation": "actions_all",
        "ranked_field": "TopTransitions",
        "gold_scope": "All relevant transition links, including orphan transitions.",
        "includes_linked_transitions": "yes",
        "includes_orphan_transitions": "yes",
        "includes_resolved_widgets": "no",
    },
    {
        "evaluation": "actions_with_linked_widget",
        "ranked_field": "TopTransitions",
        "gold_scope": "Only transition links from rows with Relevance_Type = Linked and a widget identifier.",
        "includes_linked_transitions": "yes",
        "includes_orphan_transitions": "no",
        "includes_resolved_widgets": "no",
    },
    {
        "evaluation": "actions_orphan",
        "ranked_field": "TopTransitions",
        "gold_scope": "Only relevant orphan transitions whose concrete widget ID is not resolved as a widget-tree candidate.",
        "includes_linked_transitions": "no",
        "includes_orphan_transitions": "yes",
        "includes_resolved_widgets": "no",
    },
    {
        "evaluation": "widgets_linked_resolved",
        "ranked_field": "TopWidgets",
        "gold_scope": "Only resolved widget-tree candidates with Relevance_Type = Linked.",
        "includes_linked_transitions": "no",
        "includes_orphan_transitions": "no",
        "includes_resolved_widgets": "linked only",
    },
    {
        "evaluation": "widgets_functional_resolved",
        "ranked_field": "TopWidgets",
        "gold_scope": "Only resolved widget-tree candidates with Relevance_Type = Functional.",
        "includes_linked_transitions": "no",
        "includes_orphan_transitions": "no",
        "includes_resolved_widgets": "functional only",
    },
    {
        "evaluation": "widgets_all_resolved",
        "ranked_field": "TopWidgets",
        "gold_scope": "All resolved widget-tree candidates, both Linked and Functional.",
        "includes_linked_transitions": "no",
        "includes_orphan_transitions": "no",
        "includes_resolved_widgets": "yes",
    },
    {
        "evaluation": "combined_actions_and_linked_widgets",
        "ranked_field": "TopMatches",
        "gold_scope": "All relevant transitions, including orphan transitions, plus linked widget candidates only.",
        "includes_linked_transitions": "yes",
        "includes_orphan_transitions": "yes",
        "includes_resolved_widgets": "linked only",
    },
    {
        "evaluation": "combined_actions_and_widgets",
        "ranked_field": "TopMatches",
        "gold_scope": "All relevant transitions, including orphan transitions, plus all resolved widget-tree candidates.",
        "includes_linked_transitions": "yes",
        "includes_orphan_transitions": "yes",
        "includes_resolved_widgets": "yes",
    },
]

BACKWARD_EVALUATIONS = [
    ("actions_all", "actions_all"),
    ("actions_with_linked_widget", "actions_with_linked_widget"),
    ("actions_orphan", "actions_orphan"),
    ("widgets_linked_resolved", "widgets_linked_resolved"),
    ("widgets_functional_resolved", "widgets_functional_resolved"),
    ("widgets_all_resolved", "widgets_all_resolved"),
    ("combined_actions_and_linked_widgets", "combined_actions_and_linked_widgets"),
    ("combined_actions_and_widgets", "combined_actions_and_widgets"),
]


def parse_k_values(raw_values: str) -> List[int]:
    values: List[int] = []
    for value in raw_values.split(","):
        value = value.strip()
        if not value:
            continue
        k = int(value)
        if k <= 0:
            raise ValueError(f"k must be positive: {k}")
        values.append(k)
    return sorted(set(values))


def discover_forward_result_files(args: argparse.Namespace) -> List[Path]:
    if args.result_file:
        return sorted(Path(path) for path in args.result_file)

    result_files: List[Path] = []
    for root in args.results_root or DEFAULT_RESULT_ROOTS:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in root_path.rglob("*.json"):
            if "backward" in path.name.lower():
                continue
            result_files.append(path)

    return sorted(set(result_files), key=lambda path: str(path).lower())


def csv_result_path(result_file: Path, meta: Dict[str, str]) -> str:
    root = RESULT_CSV_ROOTS[(meta["method_family"], meta["filter_variant"])]
    return str(Path(root) / result_file.parent.name / result_file.name)


def infer_method_name(result_file: Path) -> str:
    text = str(result_file).replace("\\", "/").lower()
    parent = result_file.parent.name.lower()
    if "vsm" in result_file.name.lower() or parent.startswith("vsm"):
        return "VSM"
    if "lsi" in result_file.name.lower() or parent.startswith("lsi"):
        return "LSI"
    if "jsm" in result_file.name.lower() or parent.startswith("jsm"):
        return "JSM"
    if "qwen3_embedding_0.6b" in text or "qwen0.6b" in parent:
        return "Qwen3 0.6B"
    if "qwen3_embedding_4b" in text or "qwen4b" in parent or "qwen3 4b" in parent:
        return "Qwen3 4B"
    if "jina" in text:
        return "Jina v3"
    if "stella" in text:
        return "Stella 1.5B"
    return result_file.parent.name


def classify_result_file(result_file: Path) -> Dict[str, str]:
    text = str(result_file).replace("\\", "/").lower()
    filter_variant = "Relaxed" if "relaxed" in text else "Strict"
    dataset = "Focus" if "focus" in text else "LLMReq" if "llmreq" in text or "llm_req" in text else ""
    source_model = "Gemma3" if "gemma3" in text else "GPT5" if "gpt5" in text else ""
    if not dataset or not source_model:
        raise ValueError(f"Could not infer dataset/source model from {result_file}")

    method = infer_method_name(result_file)
    ce_methods = {"Qwen3 0.6B", "Qwen3 4B", "Jina v3", "Stella 1.5B"}
    method_family = "CE" if method in ce_methods else "IR"
    scenario = f"{dataset}-{source_model}"
    condition = f"{scenario}-{filter_variant}"
    return {
        "method": method,
        "method_family": method_family,
        "filter_variant": filter_variant,
        "dataset": dataset,
        "source_model": source_model,
        "scenario": scenario,
        "condition": condition,
        "method_filter": f"{method} {filter_variant}",
        "method_condition": f"{method} | {condition}",
    }


def scenario_key(meta: Dict[str, str]) -> Tuple[str, str, str]:
    return (meta["filter_variant"], meta["dataset"], meta["source_model"])


def read_ground_truth_for_scenario(
    ground_truth_path: Path,
    dataset: str,
    source_model: str,
) -> Tuple[str, Path, List[str], List[Tuple[int, Dict[str, str]]]]:
    scenario = (dataset, source_model)
    csv_name = GROUND_TRUTH_CSV_BY_SCENARIO.get(scenario)
    if ground_truth_path.is_dir():
        if not csv_name:
            raise ValueError(f"No CSV ground-truth mapping configured for {scenario}")
        csv_path = ground_truth_path / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Ground-truth CSV not found: {csv_path}")
        headers, rows = read_ground_truth_csv(csv_path)
        return csv_path.stem, csv_path, headers, rows

    if ground_truth_path.suffix.lower() == ".csv":
        headers, rows = read_ground_truth_csv(ground_truth_path)
        return ground_truth_path.stem, ground_truth_path, headers, rows

    raise ValueError(
        f"Ground truth must be the CSV directory or a CSV file, got: {ground_truth_path}"
    )


def get_truth_context(
    meta: Dict[str, str],
    ground_truth_path: Path,
    cache: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    key = scenario_key(meta)
    if key in cache:
        return cache[key]

    model_path_text = MODEL_BY_SCENARIO.get(key)
    if not model_path_text:
        raise ValueError(f"No ground-truth/model mapping configured for {key}")

    ground_truth_name, ground_truth_source, headers, rows = read_ground_truth_for_scenario(
        ground_truth_path,
        meta["dataset"],
        meta["source_model"],
    )
    missing_headers = [header for header in REQUIRED_GT_HEADERS if header not in headers]
    unexpected_headers = [
        header
        for header in headers
        if header not in REQUIRED_GT_HEADERS and header not in OPTIONAL_GT_HEADERS
    ]
    if missing_headers or unexpected_headers:
        raise ValueError(f"Unexpected ground-truth headers in {ground_truth_source}: {headers}")

    model_path = Path(model_path_text)
    widget_index = build_widget_occurrence_index(model_path)
    truth, aliases, warnings = build_ground_truth(rows, widget_index)
    cache[key] = {
        "sheet": ground_truth_name,
        "ground_truth_file": str(ground_truth_source),
        "model_path": str(Path(MODEL_CSV_PATH_BY_SCENARIO.get(key, model_path_text))),
        "truth": truth,
        "aliases": aliases,
        "backward_truth": build_backward_ground_truth(truth, aliases),
        "warnings": warnings,
    }
    return cache[key]


def backward_candidate_keys_for_truth(
    req_id: str,
    truth_key: str,
    truth: Dict[str, Dict[str, Set[str]]],
    aliases: Dict[str, Dict[str, Dict[str, Set[str]]]],
) -> Set[str]:
    if truth_key in {"actions_all", "actions_with_linked_widget", "actions_orphan"}:
        keys = {f"T:{key}" for key in truth[req_id][truth_key]}
        for alias, canonical_keys in aliases[req_id][truth_key].items():
            if canonical_keys & truth[req_id][truth_key]:
                keys.add(f"T:{alias}")
        return keys

    if truth_key in {"widgets_linked_resolved", "widgets_functional_resolved", "widgets_all_resolved"}:
        return {f"W:{key}" for key in truth[req_id][truth_key]}

    keys = set(truth[req_id][truth_key])
    for alias, canonical_keys in aliases[req_id][truth_key].items():
        if canonical_keys & truth[req_id][truth_key]:
            keys.add(alias)
    return keys


def build_backward_ground_truth(
    truth: Dict[str, Dict[str, Set[str]]],
    aliases: Dict[str, Dict[str, Dict[str, Set[str]]]],
) -> Dict[str, Dict[str, Set[str]]]:
    backward_truth: Dict[str, Dict[str, Set[str]]] = {
        evaluation_name: defaultdict(set) for evaluation_name, _truth_key in BACKWARD_EVALUATIONS
    }
    for req_id in truth:
        for evaluation_name, truth_key in BACKWARD_EVALUATIONS:
            for candidate_key in backward_candidate_keys_for_truth(req_id, truth_key, truth, aliases):
                backward_truth[evaluation_name][candidate_key].add(req_id)
    return backward_truth


def find_backward_result_file(forward_file: Path) -> Optional[Path]:
    candidates = [forward_file.with_name(f"{forward_file.stem}_backward{forward_file.suffix}")]
    stem = forward_file.stem
    for prefix in ("vsm", "lsi", "jsm"):
        token = f"{prefix}_matches"
        if stem.startswith(token):
            candidates.append(
                forward_file.with_name(stem.replace(token, f"{prefix}_backward_matches", 1) + forward_file.suffix)
            )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_backward_entries(backward_file: Path) -> List[Dict[str, Any]]:
    with backward_file.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if isinstance(payload, dict) and isinstance(payload.get("Results"), list):
        return payload["Results"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported backward result JSON schema in {backward_file}")


def backward_candidate_type(candidate_key: str) -> str:
    if candidate_key.startswith("W:"):
        return "Widget"
    if candidate_key.startswith("T:"):
        return "Transition"
    return "Unknown"


def backward_candidate_matches_evaluation(candidate_key: str, evaluation: str) -> bool:
    if evaluation in {"actions_all", "actions_with_linked_widget", "actions_orphan"}:
        return candidate_key.startswith("T:")
    if evaluation in {"widgets_linked_resolved", "widgets_functional_resolved", "widgets_all_resolved"}:
        return candidate_key.startswith("W:")
    return candidate_key.startswith(("W:", "T:"))


def ranked_requirement_key_sets(entry: Dict[str, Any], k: int) -> List[Set[str]]:
    top_requirements = entry.get("TopRequirements", [])
    ranked_keys: List[Set[str]] = []
    if not isinstance(top_requirements, list):
        return ranked_keys
    for item in top_requirements[:k]:
        if isinstance(item, dict):
            req_id = clean(item.get("ReqID"))
            if req_id:
                ranked_keys.append({req_id})
    return ranked_keys


def evaluate_backward_result_file(
    backward_file: Path,
    backward_truth: Dict[str, Dict[str, Set[str]]],
    k: int,
) -> List[Dict[str, Any]]:
    """Evaluate one backward result JSON for GUI-candidate-to-requirement ranking."""
    entries = load_backward_entries(backward_file)
    method = backward_file.parent.name
    rows: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()

    for entry in entries:
        candidate_key = clean(entry.get("CandidateKey"))
        if not candidate_key:
            continue
        candidate_type = clean(entry.get("Type")) or backward_candidate_type(candidate_key)
        for evaluation_name, candidate_truth in BACKWARD_EVALUATIONS:
            if not backward_candidate_matches_evaluation(candidate_key, evaluation_name):
                continue
            gold_req_ids = backward_truth[evaluation_name].get(candidate_key, set())
            metrics = calculate_ranking_metrics(ranked_requirement_key_sets(entry, k), gold_req_ids, k)
            rows.append(
                {
                    "direction": "backward",
                    "result_file": str(backward_file),
                    "method": method,
                    "evaluation": evaluation_name,
                    "query_unit": "gui_candidate",
                    "query_id": candidate_key,
                    "candidate_key": candidate_key,
                    "candidate_type": candidate_type,
                    "k": k,
                    **metrics,
                }
            )
            seen.add((evaluation_name, candidate_key))

    for evaluation_name, candidate_truth in backward_truth.items():
        for candidate_key, gold_req_ids in candidate_truth.items():
            if (evaluation_name, candidate_key) in seen:
                continue
            metrics = calculate_ranking_metrics([], gold_req_ids, k)
            rows.append(
                {
                    "direction": "backward",
                    "result_file": str(backward_file),
                    "method": method,
                    "evaluation": evaluation_name,
                    "query_unit": "gui_candidate",
                    "query_id": candidate_key,
                    "candidate_key": candidate_key,
                    "candidate_type": backward_candidate_type(candidate_key),
                    "k": k,
                    **metrics,
                }
            )

    return rows


def collect_backward_stats(
    backward_file: Path,
    meta: Dict[str, str],
    recorded_output_path: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    entries = load_backward_entries(backward_file)
    type_counts: Counter[str] = Counter()
    length_rows: List[Dict[str, Any]] = []

    for entry in entries:
        candidate_type = clean(entry.get("Type")) or "Unknown"
        top_requirements = entry.get("TopRequirements", [])
        top_count = len(top_requirements) if isinstance(top_requirements, list) else 0
        type_counts[candidate_type] += 1
        length_rows.append(
            {
                **meta,
                "backward_result_file": recorded_output_path,
                "candidate_type": candidate_type,
                "candidate_key": clean(entry.get("CandidateKey")),
                "top_requirements_count": top_count,
            }
        )

    counts = [int(row["top_requirements_count"]) for row in length_rows]
    summary = {
        **meta,
        "backward_result_file": recorded_output_path,
        "total_candidates": len(entries),
        "widget_candidates": type_counts.get("Widget", 0),
        "transition_candidates": type_counts.get("Transition", 0),
        "unknown_candidates": sum(count for key, count in type_counts.items() if key not in {"Widget", "Transition"}),
        "avg_top_requirements": mean(counts),
        "min_top_requirements": min(counts) if counts else 0,
        "max_top_requirements": max(counts) if counts else 0,
    }
    return summary, length_rows


def bytes_to_mb(value: Any) -> float:
    try:
        return float(value) / (1024 * 1024)
    except (TypeError, ValueError):
        return 0.0


def collect_scalability_row(
    result_file: Path,
    meta: Dict[str, str],
    recorded_output_path: str,
) -> Optional[Dict[str, Any]]:
    with result_file.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, dict):
        return None

    environment = payload.get("Environment")
    if not isinstance(environment, dict):
        return None

    runtime = environment.get("runtime") if isinstance(environment.get("runtime"), dict) else {}
    memory = environment.get("memory") if isinstance(environment.get("memory"), dict) else {}
    total_seconds = runtime.get("total_seconds")
    if total_seconds is None:
        return None

    return {
        **meta,
        "result_file": recorded_output_path,
        "total_seconds": float(total_seconds),
        "process_rss_peak_mb": bytes_to_mb(memory.get("process_rss_peak_bytes")),
        "cuda_peak_allocated_mb": bytes_to_mb(memory.get("cuda_peak_allocated_bytes")),
        "cuda_peak_reserved_mb": bytes_to_mb(memory.get("cuda_peak_reserved_bytes")),
    }


def add_meta(
    row: Dict[str, Any],
    meta: Dict[str, str],
    result_output_path: str,
    backward_output_path: str,
) -> Dict[str, Any]:
    enriched = {
        **meta,
        "backward_result_file": backward_output_path,
        **row,
    }
    enriched["method"] = meta["method"]
    enriched["result_file"] = result_output_path
    return enriched


def build_ground_truth_warning_summary(warning_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not warning_rows:
        return []
    dataframe = pd.DataFrame(warning_rows)
    group_columns = ["warning", "filter_variant", "dataset", "source_model"]
    for column in group_columns:
        if column not in dataframe.columns:
            dataframe[column] = ""
    grouped = dataframe.groupby(group_columns, dropna=False).size().reset_index(name="count")
    return grouped.sort_values(group_columns).to_dict("records")


def thesis_backward_all_candidate_rows(
    backward_summary_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in backward_summary_rows:
        if clean(row.get("average_scope")) != "all":
            continue
        candidate_queries_total = safe_int(row.get("candidate_queries_total"))
        candidate_queries_with_gold = safe_int(row.get("candidate_queries_with_gold"))
        enriched = {
            **row,
            "candidate_gold_coverage_ratio": (
                candidate_queries_with_gold / candidate_queries_total if candidate_queries_total else 0.0
            ),
        }
        rows.append(enriched)
    return project_rows(
        rows,
        [
            "method",
            "method_family",
            "filter_variant",
            "dataset",
            "source_model",
            "scenario",
            "evaluation",
            "k",
            "candidate_queries_total",
            "candidate_queries_with_gold",
            "candidate_queries_without_gold",
            "candidate_gold_coverage_ratio",
            "hit_at_k",
            "precision_at_k",
            "recall_at_k",
            "f1_at_k",
            "map_at_k",
            "mrr_at_k",
        ],
        {
            "candidate_queries_total": "gui_candidates_total",
            "candidate_queries_with_gold": "gui_candidates_with_ground_truth",
            "candidate_queries_without_gold": "gui_candidates_without_ground_truth",
            "candidate_gold_coverage_ratio": "gui_candidate_ground_truth_coverage_ratio",
            "hit_at_k": "gui_candidate_hit_at_k",
        },
    )


def thesis_backward_gold_only_rows(
    backward_gold_summary_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = [row for row in backward_gold_summary_rows if clean(row.get("average_scope")) == "gold"]
    return project_rows(
        rows,
        [
            "method",
            "method_family",
            "filter_variant",
            "dataset",
            "source_model",
            "scenario",
            "evaluation",
            "k",
            "candidate_queries_total",
            "candidate_queries_with_gold",
            "candidate_queries_without_gold",
            "hit_at_k",
            "precision_at_k",
            "recall_at_k",
            "f1_at_k",
            "map_at_k",
            "mrr_at_k",
        ],
        {
            "candidate_queries_total": "gui_candidates_total",
            "candidate_queries_with_gold": "gui_candidates_with_ground_truth",
            "candidate_queries_without_gold": "gui_candidates_without_ground_truth",
            "hit_at_k": "linked_gui_candidate_hit_at_k",
        },
    )


def build_thesis_filtering_delta_summary(
    forward_summary_rows: Sequence[Dict[str, Any]],
    backward_summary_rows: Sequence[Dict[str, Any]],
    forward_main_k: int,
    backward_main_k: int,
    evaluation: str,
) -> List[Dict[str, Any]]:
    source_rows = [
        ("forward", forward_summary_rows, forward_main_k),
        ("backward_all_candidate", backward_summary_rows, backward_main_k),
    ]
    delta_rows: List[Dict[str, Any]] = []
    for direction, rows, main_k in source_rows:
        selected = [
            row
            for row in rows
            if row.get("evaluation") == evaluation and safe_int(row.get("k")) == main_k
        ]
        grouped: Dict[Tuple[str, str, str, str, str, int], Dict[str, Dict[str, Any]]] = defaultdict(dict)
        for row in selected:
            key = (
                clean(row.get("method")),
                clean(row.get("dataset")),
                clean(row.get("source_model")),
                clean(row.get("scenario")),
                clean(row.get("evaluation")),
                safe_int(row.get("k")),
            )
            grouped[key][clean(row.get("filter_variant"))] = row
        for (method, dataset, source_model, scenario, row_evaluation, k), variants in sorted(grouped.items()):
            strict = variants.get("Strict")
            relaxed = variants.get("Relaxed")
            if strict is None or relaxed is None:
                continue
            delta_rows.append(
                {
                    "direction": direction,
                    "method": method,
                    "dataset": dataset,
                    "source_model": source_model,
                    "scenario": scenario,
                    "evaluation": row_evaluation,
                    "k": k,
                    "strict_f1_at_k": safe_float(strict.get("f1_at_k")),
                    "relaxed_f1_at_k": safe_float(relaxed.get("f1_at_k")),
                    "delta_f1_relaxed_minus_strict": safe_float(relaxed.get("f1_at_k"))
                    - safe_float(strict.get("f1_at_k")),
                    "strict_recall_at_k": safe_float(strict.get("recall_at_k")),
                    "relaxed_recall_at_k": safe_float(relaxed.get("recall_at_k")),
                    "delta_recall_relaxed_minus_strict": safe_float(relaxed.get("recall_at_k"))
                    - safe_float(strict.get("recall_at_k")),
                    "strict_precision_at_k": safe_float(strict.get("precision_at_k")),
                    "relaxed_precision_at_k": safe_float(relaxed.get("precision_at_k")),
                    "delta_precision_relaxed_minus_strict": safe_float(relaxed.get("precision_at_k"))
                    - safe_float(strict.get("precision_at_k")),
                }
            )
    return delta_rows


def build_thesis_model_variant_summary(
    forward_summary_rows: Sequence[Dict[str, Any]],
    backward_summary_rows: Sequence[Dict[str, Any]],
    forward_main_k: int,
    backward_main_k: int,
    evaluation: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for direction, source_rows, main_k in [
        ("forward", forward_summary_rows, forward_main_k),
        ("backward_all_candidate", backward_summary_rows, backward_main_k),
    ]:
        selected = [
            row
            for row in source_rows
            if row.get("evaluation") == evaluation and safe_int(row.get("k")) == main_k
        ]
        for row in selected:
            rows.append({"direction": direction, **row})
    return project_rows(
        rows,
        [
            "direction",
            "method",
            "method_family",
            "filter_variant",
            "dataset",
            "source_model",
            "scenario",
            "evaluation",
            "k",
            "precision_at_k",
            "recall_at_k",
            "f1_at_k",
            "map_at_k",
            "mrr_at_k",
            "hit_at_k",
        ],
    )


def build_thesis_warning_summary(warning_summary_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = project_rows(
        warning_summary_rows,
        ["warning", "filter_variant", "dataset", "source_model", "count"],
    )
    for row in rows:
        if row.get("warning") == "row_fallback":
            row["interpretation"] = (
                "Manually annotated widget was not present in the filtered model variant; "
                "this can occur when strict filtering removes that widget instance."
            )
        else:
            row["interpretation"] = (
                "Ground-truth widget could not be resolved directly against the filtered model representation."
            )
    return rows


def write_thesis_facing_outputs(
    data_dir: Path,
    forward_summary_rows: Sequence[Dict[str, Any]],
    forward_per_query_rows: Sequence[Dict[str, Any]],
    backward_summary_rows: Sequence[Dict[str, Any]],
    backward_gold_summary_rows: Sequence[Dict[str, Any]],
    backward_per_query_rows: Sequence[Dict[str, Any]],
    warning_summary_rows: Sequence[Dict[str, Any]],
    forward_main_k: int,
    backward_main_k: int,
    evaluation: str,
) -> List[Path]:
    """Write the thesis-facing CSV files consumed by the final plotting workflow."""
    outputs: List[Tuple[Path, List[Dict[str, Any]]]] = [
        (
            data_dir / "forward_summary.csv",
            project_rows(
                forward_summary_rows,
                [
                    "method",
                    "method_family",
                    "filter_variant",
                    "dataset",
                    "source_model",
                    "scenario",
                    "evaluation",
                    "k",
                    "queries_total",
                    "queries_with_gold",
                    "gold_links_total",
                    "hits_total",
                    "hit_at_k",
                    "precision_at_k",
                    "recall_at_k",
                    "f1_at_k",
                    "map_at_k",
                    "mrr_at_k",
                ],
                {
                    "queries_total": "requirements_total",
                    "queries_with_gold": "requirements_with_ground_truth",
                    "gold_links_total": "ground_truth_links_total",
                    "hits_total": "correctly_retrieved_links",
                    "hit_at_k": "requirement_hit_at_k",
                },
            ),
        ),
        (
            data_dir / "forward_per_requirement.csv",
            project_rows(
                forward_per_query_rows,
                [
                    "method",
                    "method_family",
                    "filter_variant",
                    "dataset",
                    "source_model",
                    "scenario",
                    "evaluation",
                    "requirement_id",
                    "k",
                    "gold_count",
                    "hits",
                    "precision_at_k",
                    "recall_at_k",
                    "f1_at_k",
                    "ap_at_k",
                    "rr_at_k",
                ],
                {
                    "gold_count": "ground_truth_links",
                    "hits": "correctly_retrieved_links",
                    "ap_at_k": "average_precision_at_k",
                    "rr_at_k": "reciprocal_rank_at_k",
                },
            ),
        ),
        (
            data_dir / "backward_all_candidate_summary.csv",
            thesis_backward_all_candidate_rows(backward_summary_rows),
        ),
        (
            data_dir / "backward_gold_only_diagnostic_summary.csv",
            thesis_backward_gold_only_rows(backward_gold_summary_rows),
        ),
        (
            data_dir / "backward_per_gui_candidate.csv",
            project_rows(
                backward_per_query_rows,
                [
                    "method",
                    "method_family",
                    "filter_variant",
                    "dataset",
                    "source_model",
                    "scenario",
                    "evaluation",
                    "candidate_type",
                    "k",
                    "gold_count",
                    "hits",
                    "precision_at_k",
                    "recall_at_k",
                    "f1_at_k",
                    "ap_at_k",
                    "rr_at_k",
                ],
                {
                    "gold_count": "ground_truth_requirements",
                    "hits": "correctly_retrieved_requirements",
                    "ap_at_k": "average_precision_at_k",
                    "rr_at_k": "reciprocal_rank_at_k",
                },
            ),
        ),
        (
            data_dir / "filtering_delta_summary.csv",
            build_thesis_filtering_delta_summary(
                forward_summary_rows,
                backward_summary_rows,
                forward_main_k,
                backward_main_k,
                evaluation,
            ),
        ),
        (
            data_dir / "model_variant_summary.csv",
            build_thesis_model_variant_summary(
                forward_summary_rows,
                backward_summary_rows,
                forward_main_k,
                backward_main_k,
                evaluation,
            ),
        ),
        (
            data_dir / "ground_truth_warning_summary.csv",
            build_thesis_warning_summary(warning_summary_rows),
        ),
    ]
    paths: List[Path] = []
    for path, rows in outputs:
        write_csv(path, rows)
        paths.append(path)
    return paths


def candidate_keys(metadata_rows: Sequence[Dict[str, Any]], prefix: str) -> Set[str]:
    keys: Set[str] = set()
    for metadata in metadata_rows:
        keys.update(combined_prediction_keys(metadata))
    return {key for key in keys if key.startswith(prefix)}


def combined_candidate_truth_keys(truth: Dict[str, Dict[str, Set[str]]]) -> Set[str]:
    keys: Set[str] = set()
    for requirement_truth in truth.values():
        keys.update(requirement_truth["combined_actions_and_widgets"])
    return keys


def count_candidates_with_truth(
    metadata_rows: Sequence[Dict[str, Any]],
    truth_keys: Set[str],
) -> int:
    return sum(bool(combined_prediction_keys(metadata) & truth_keys) for metadata in metadata_rows)


def add_unmatched_candidate_warning(
    diagnostics: List[str],
    scenario: str,
    filter_variant: str,
    candidate_type: str,
    unmatched_keys: Iterable[str],
) -> None:
    unmatched = sorted(unmatched_keys)
    if not unmatched:
        return
    preview = ", ".join(unmatched[:5])
    suffix = "" if len(unmatched) <= 5 else f", ... (+{len(unmatched) - 5} more)"
    diagnostics.append(
        f"[warning] {scenario} / {filter_variant}: {len(unmatched)} unmatched "
        f"GT {candidate_type} key(s): {preview}{suffix}"
    )


def add_widget_fallback_warning(
    diagnostics: List[str],
    scenario: str,
    filter_variant: str,
    warning_rows: Sequence[Dict[str, str]],
) -> None:
    if not warning_rows:
        return

    counts = Counter(row.get("warning", "unknown") for row in warning_rows)
    detail = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    diagnostics.append(
        f"[warning] {scenario} / {filter_variant}: {len(warning_rows)} GT widget "
        f"row(s) required fallback resolution ({detail})"
    )


def validate_candidate_space_row(row: Dict[str, Any]) -> None:
    if row["Total candidates"] != row["Widget candidates"] + row["Transition candidates"]:
        raise ValueError(f"Candidate total mismatch for {row['Model variant']} / {row['Filter']}")
    if row["GT-relevant widgets"] > row["Widget candidates"]:
        raise ValueError(f"GT-relevant widget count exceeds candidate count for {row['Model variant']}")
    if row["GT-relevant transitions"] > row["Transition candidates"]:
        raise ValueError(f"GT-relevant transition count exceeds candidate count for {row['Model variant']}")


def build_candidate_space_row(
    scenario: CandidateSpaceScenario,
    filter_variant: str,
    model_path: Path,
    ground_truth_dir: Path,
    diagnostics: List[str],
) -> Dict[str, Any]:
    """Count candidates and ground-truth-relevant candidates for one model variant."""
    candidates = load_widget_and_transition_candidates(str(model_path))
    widget_metadata = candidates["widget_meta"]
    transition_metadata = candidates["transition_meta"]
    widget_count = len(widget_metadata)
    transition_count = len(transition_metadata)
    total_count = widget_count + transition_count

    widget_keys = candidate_keys(widget_metadata, "W:")
    transition_keys = candidate_keys(transition_metadata, "T:")
    _, ground_truth_rows = read_ground_truth_csv(ground_truth_dir / scenario.ground_truth_file)
    widget_index = build_widget_occurrence_index(model_path)
    truth, _, ground_truth_warnings = build_ground_truth(ground_truth_rows, widget_index)
    truth_keys = combined_candidate_truth_keys(truth)
    truth_widget_keys = {key for key in truth_keys if key.startswith("W:")}
    truth_transition_keys = {key for key in truth_keys if key.startswith("T:")}

    relevant_widgets = count_candidates_with_truth(widget_metadata, truth_widget_keys)
    relevant_transitions = count_candidates_with_truth(transition_metadata, truth_transition_keys)
    add_unmatched_candidate_warning(
        diagnostics,
        scenario.model_variant,
        filter_variant,
        "widget",
        truth_widget_keys - widget_keys,
    )
    add_unmatched_candidate_warning(
        diagnostics,
        scenario.model_variant,
        filter_variant,
        "transition",
        truth_transition_keys - transition_keys,
    )
    add_widget_fallback_warning(
        diagnostics,
        scenario.model_variant,
        filter_variant,
        ground_truth_warnings,
    )

    relevant_count = relevant_widgets + relevant_transitions
    relevant_percentage = round((relevant_count / total_count) * 100, 2) if total_count else 0.0
    row = {
        "Model variant": scenario.model_variant,
        "Filter": filter_variant,
        "Requirements": scenario.requirement_count,
        "Covered requirements": scenario.covered_requirement_count,
        "Widget candidates": widget_count,
        "Transition candidates": transition_count,
        "Total candidates": total_count,
        "GT-relevant widgets": relevant_widgets,
        "GT-relevant transitions": relevant_transitions,
        "Relevant candidate %": f"{relevant_percentage:.2f}",
    }
    validate_candidate_space_row(row)
    return row


def write_candidate_space_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=CANDIDATE_SPACE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def generate_candidate_space_summary(
    output_dir: Path,
    ground_truth_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Path], List[str]]:
    """Create the candidate-space summary CSV and return diagnostics."""
    repo_root = Path(__file__).resolve().parent
    ground_truth_dir = ground_truth_path if ground_truth_path.is_dir() else ground_truth_path.parent
    diagnostics: List[str] = []
    rows: List[Dict[str, Any]] = []

    for scenario in CANDIDATE_SPACE_SCENARIOS:
        model_paths = (
            ("Strict", repo_root / "Filtered models" / scenario.strict_model_file),
            ("Relaxed", repo_root / "Filtered relaxed models" / scenario.relaxed_model_file),
        )
        for filter_variant, model_path in model_paths:
            rows.append(
                build_candidate_space_row(
                    scenario,
                    filter_variant,
                    model_path,
                    ground_truth_dir,
                    diagnostics,
                )
            )

    csv_path = output_dir / "data" / "candidate_space_summary.csv"
    write_candidate_space_csv(rows, csv_path)
    return rows, [csv_path], diagnostics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate traceability metrics across result files and generate thesis-oriented charts."
    )
    parser.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--results-root", action="append", help="Root folder to scan for forward result JSON files.")
    parser.add_argument("--result-file", action="append", help="Specific forward result JSON file to evaluate.")
    parser.add_argument("--output-dir", default=DEFAULT_CHART_OUTPUT_DIR)
    parser.add_argument("--k-values", default=None, help="Legacy alias for --forward-k-values.")
    parser.add_argument("--forward-k-values", default=DEFAULT_FORWARD_K_VALUES)
    parser.add_argument("--backward-k-values", default=DEFAULT_BACKWARD_K_VALUES)
    parser.add_argument("--main-k", type=int, default=None, help="Legacy alias for --forward-main-k.")
    parser.add_argument("--forward-main-k", type=int, default=DEFAULT_MAIN_K)
    parser.add_argument("--backward-main-k", type=int, default=DEFAULT_BACKWARD_MAIN_K)
    parser.add_argument("--evaluation", default=DEFAULT_EVALUATION)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--no-plots", action="store_true", help="Only compute metric CSV files; do not generate charts.")
    mode.add_argument("--plots-only", action="store_true", help="Only generate charts from existing metric CSV files.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the full metrics workflow, optionally followed by chart generation."""
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    ground_truth_path = Path(args.ground_truth)
    forward_k_values = parse_k_values(args.k_values or args.forward_k_values)
    backward_k_values = parse_k_values(args.backward_k_values)
    forward_main_k = args.main_k if args.main_k is not None else args.forward_main_k
    backward_main_k = args.backward_main_k
    if forward_main_k not in forward_k_values:
        forward_k_values = sorted([*forward_k_values, forward_main_k])
    if backward_main_k not in backward_k_values:
        backward_k_values = sorted([*backward_k_values, backward_main_k])

    if args.plots_only:
        from plot_metrics_charts import plot_charts_from_csv_outputs

        chart_paths = plot_charts_from_csv_outputs(
            output_dir=output_dir,
        )
        print(f"Generated charts from existing metric CSV files in {output_dir / 'data'}.")
        print("Charts:")
        for path in chart_paths:
            print(f"- {path}")
        return

    result_files = discover_forward_result_files(args)
    if not result_files:
        raise ValueError("No forward result JSON files found.")

    truth_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    warning_keys: Set[Tuple[str, str, str]] = set()
    warning_rows: List[Dict[str, Any]] = []
    forward_per_query_rows: List[Dict[str, Any]] = []
    forward_summary_rows: List[Dict[str, Any]] = []
    backward_per_query_rows: List[Dict[str, Any]] = []
    backward_summary_rows: List[Dict[str, Any]] = []
    backward_gold_summary_rows: List[Dict[str, Any]] = []
    backward_output_summary_rows: List[Dict[str, Any]] = []
    backward_length_rows: List[Dict[str, Any]] = []
    scalability_rows: List[Dict[str, Any]] = []

    for result_file in result_files:
        meta = classify_result_file(result_file)
        result_output_path = csv_result_path(result_file, meta)
        context = get_truth_context(meta, ground_truth_path, truth_cache)
        scalability_row = collect_scalability_row(result_file, meta, result_output_path)
        if scalability_row:
            scalability_rows.append(scalability_row)

        key = scenario_key(meta)
        if key not in warning_keys:
            # These warnings are diagnostic, not fatal. They can occur when a manually
            # annotated widget is absent from a filtered model variant, especially strict filtering.
            warning_rows.extend(
                {
                    **meta,
                    "sheet": context["sheet"],
                    "ground_truth_file": context["ground_truth_file"],
                    "model_path": context["model_path"],
                    **warning,
                }
                for warning in context["warnings"]
            )
            warning_keys.add(key)

        backward_file = find_backward_result_file(result_file)
        backward_output_path = ""
        if backward_file:
            backward_output_path = csv_result_path(backward_file, meta)
            backward_summary, backward_lengths = collect_backward_stats(
                backward_file,
                meta,
                backward_output_path,
            )
            backward_output_summary_rows.append(backward_summary)
            backward_length_rows.extend(backward_lengths)

            for k in backward_k_values:
                rows = evaluate_backward_result_file(backward_file, context["backward_truth"], k)
                for row in rows:
                    row["method"] = meta["method"]
                backward_per_query_rows.extend(
                    add_meta(row, meta, backward_output_path, backward_output_path) for row in rows
                )

                current_summary_rows = summarize_per_query(rows, average_scope="all")
                backward_summary_rows.extend(
                    add_meta(row, meta, backward_output_path, backward_output_path)
                    for row in current_summary_rows
                )
                current_gold_summary_rows = summarize_per_query(rows, average_scope="gold")
                backward_gold_summary_rows.extend(
                    add_meta(row, meta, backward_output_path, backward_output_path)
                    for row in current_gold_summary_rows
                )

        for k in forward_k_values:
            rows = evaluate_result_file(result_file, context["truth"], context["aliases"], k)
            for row in rows:
                row["method"] = meta["method"]
            forward_per_query_rows.extend(
                add_meta(row, meta, result_output_path, backward_output_path) for row in rows
            )

            current_summary_rows = summarize_per_query(rows, average_scope="gold")
            forward_summary_rows.extend(
                add_meta(row, meta, result_output_path, backward_output_path)
                for row in current_summary_rows
            )

    forward_per_query_rows.sort(
        key=lambda row: (
            result_sort_key(row),
            row["scenario"],
            int(row["k"]),
            row["evaluation"],
            requirement_sort_key(row["query_id"]),
        )
    )
    backward_per_query_rows.sort(
        key=lambda row: (
            result_sort_key(row),
            row["scenario"],
            int(row["k"]),
            row["evaluation"],
            row["candidate_type"],
            row["candidate_key"],
        )
    )
    forward_summary_rows.sort(
        key=lambda row: (
            result_sort_key(row),
            row["scenario"],
            int(row["k"]),
            row["evaluation"],
        )
    )
    backward_summary_rows.sort(
        key=lambda row: (
            result_sort_key(row),
            row["scenario"],
            int(row["k"]),
            row["evaluation"],
        )
    )
    backward_gold_summary_rows.sort(
        key=lambda row: (
            result_sort_key(row),
            row["scenario"],
            int(row["k"]),
            row["evaluation"],
        )
    )
    backward_output_summary_rows.sort(key=lambda row: clean(row["backward_result_file"]).lower())
    backward_length_rows.sort(key=lambda row: clean(row["backward_result_file"]).lower())
    scalability_rows.sort(key=lambda row: clean(row["result_file"]).lower())

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"

    warning_summary_rows = build_ground_truth_warning_summary(warning_rows)
    write_csv(data_dir / "forward_metrics_per_requirement_by_k.csv", forward_per_query_rows)
    write_csv(data_dir / "forward_metrics_summary_by_k.csv", forward_summary_rows)
    write_csv(data_dir / "backward_metrics_per_gui_candidate_by_k.csv", backward_per_query_rows)
    write_csv(data_dir / "backward_metrics_summary_by_k.csv", backward_summary_rows)
    write_csv(data_dir / "backward_metrics_summary_gold_only_by_k.csv", backward_gold_summary_rows)
    write_csv(data_dir / "backward_output_candidate_summary.csv", backward_output_summary_rows)
    write_csv(data_dir / "backward_output_candidate_lengths.csv", backward_length_rows)
    write_csv(data_dir / "scalability_runtime_memory.csv", scalability_rows)
    write_csv(data_dir / "chart_ground_truth_warnings.csv", warning_rows)
    write_csv(data_dir / "evaluation_views.csv", EVALUATION_VIEW_ROWS)

    thesis_csv_paths = write_thesis_facing_outputs(
        data_dir=data_dir,
        forward_summary_rows=forward_summary_rows,
        forward_per_query_rows=forward_per_query_rows,
        backward_summary_rows=backward_summary_rows,
        backward_gold_summary_rows=backward_gold_summary_rows,
        backward_per_query_rows=backward_per_query_rows,
        warning_summary_rows=warning_summary_rows,
        forward_main_k=forward_main_k,
        backward_main_k=backward_main_k,
        evaluation=args.evaluation,
    )
    _, candidate_space_paths, candidate_diagnostics = generate_candidate_space_summary(
        output_dir,
        ground_truth_path,
    )

    chart_paths: List[Path] = []
    if not args.no_plots:
        from plot_metrics_charts import generate_section_charts

        chart_paths = generate_section_charts(
            output_dir=output_dir,
        )

    print(
        f"Evaluated {len(result_files)} forward result files at "
        f"forward k={','.join(str(k) for k in forward_k_values)} and "
        f"backward k={','.join(str(k) for k in backward_k_values)}."
    )
    print("Backward summary averaging: all candidate GUI queries.")
    print(f"Forward summary metrics: {data_dir / 'forward_metrics_summary_by_k.csv'}")
    print(f"Backward summary metrics: {data_dir / 'backward_metrics_summary_by_k.csv'}")
    print(f"Backward gold-only summary metrics: {data_dir / 'backward_metrics_summary_gold_only_by_k.csv'}")
    print(f"Ground-truth warnings: {len(warning_rows)}")
    print(f"Ground-truth warning summary rows: {len(warning_summary_rows)}")
    if warning_rows:
        print(f"Ground-truth mapping warnings: {data_dir / 'chart_ground_truth_warnings.csv'}")
        print(dict(Counter(warning["warning"] for warning in warning_rows)))
    if thesis_csv_paths:
        print("Thesis-facing CSV files:")
        for path in thesis_csv_paths:
            print(f"- {path}")
    print("Candidate-space summary files:")
    for path in candidate_space_paths:
        print(f"- {path}")
    if candidate_diagnostics:
        print("Candidate-space diagnostics:")
        for diagnostic in candidate_diagnostics:
            print(diagnostic)
    if chart_paths:
        print("Charts:")
        for path in chart_paths:
            print(f"- {path}")


if __name__ == "__main__":
    main()

