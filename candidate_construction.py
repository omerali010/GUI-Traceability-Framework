"""Build requirement, widget, and transition candidates for ranking scripts."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from text_preprocessing import (
    first_nonempty_text,
    normalize_text,
    selector_tokens,
    text_value,
    unique_preserve_order,
)

DEFAULT_REQ_PATH = "Requirement Specifications/Requirements focus group.txt"
DEFAULT_DATA_PATH = "Filtered models/filtered_model_GPT5_Focus.json"

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


def is_widget_candidate(node: Dict[str, Any]) -> bool:
    return bool(node.get("TraceabilityCandidate"))


def widget_abstract_id(node: Dict[str, Any]) -> str:
    return text_value(node, "AbstractID")


def widget_concrete_id(node: Dict[str, Any]) -> str:
    return text_value(node, "ConcreteID")


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
    widget_id = widget_abstract_id(node)
    widget_concrete = widget_concrete_id(node)
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
