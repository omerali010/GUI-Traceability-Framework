"""Create the strict filtered GUI models used by the traceability pipeline."""

import argparse
import json
import re
from typing import Any, Dict, Set

MODEL_PATH = "Models/Gemma3_focus.json"
OUTPUT_PATH = "Filtered models/filtered_model_Gemma3_Focus.json"

# Fields retained from the raw TESTAR model.
STATE_KEYS = {"AbstractID", "WebHref", "WebTitle"}
ACTION_KEYS = {
    "AbstractID",
    "ConcreteID",
    "actionId", 
    "Desc",
    "WebCssSelector",
    "WebTagName",
    "WebHref",
}

WIDGET_KEYS = {
    "AbstractID",
    "ConcreteID",
    "WebCssSelector",
    "WebTextContent",
    "WebTagName",
    "WebHref",
    "TraceabilityCandidate",
    "children",
}

# Tags and selector patterns used to separate meaningful widgets from structural markup.
INTERACTIVE_TAGS = {"a", "button", "input", "select", "option", "textarea"}
GENERIC_WIDGET_TAGS = {
    "app-root",
    "body",
    "div",
    "li",
    "nav",
    "router-outlet",
    "span",
    "ul",
}

GENERIC_TAGS = {
    "div",
    "span",
    "li",
    "ul",
    "nav",
}

GENERIC_SELECTOR_PATTERNS = [
    r":nth-child\(",
    r"^\.btn",
    r"^\.nav",
    r"nav-",
    r"header",
    r"footer",
    r"lang-",
]

SELECTOR_REGEX = [re.compile(p) for p in GENERIC_SELECTOR_PATTERNS]

def filter_dict(obj: Dict[str, Any], allowed: set) -> Dict[str, Any]:
    return {k: obj.get(k) for k in allowed if k in obj}

def is_generic_selector(selector: str) -> bool:
    """Return whether a selector looks too generic for traceability."""
    if not selector:
        return False
    return any(r.search(selector) for r in SELECTOR_REGEX)

def norm_text(text: str) -> str:
    """Normalize short widget strings for filtering decisions."""
    return re.sub(r"\s+", " ", text.strip().lower())

def is_specific_selector(selector: str) -> bool:
    """Return whether a selector contains a useful identifying marker."""
    return any(marker in selector for marker in ("#", "[", "data-test", "name=", "href="))


def is_action_referenced_widget(
    node: Dict[str, Any],
    referenced_abstract_ids: Set[str],
    referenced_concrete_ids: Set[str],
) -> bool:
    abstract_id = node.get("AbstractID")
    concrete_id = node.get("ConcreteID")
    return abstract_id in referenced_abstract_ids or concrete_id in referenced_concrete_ids

def is_traceability_candidate(node: Dict[str, Any]) -> bool:
    """Return whether a widget is meaningful enough to rank as a candidate."""
    text = node.get("WebTextContent")
    selector = node.get("WebCssSelector")
    tag = norm_text(node.get("WebTagName", "") or "")
    href = node.get("WebHref")

    has_text = isinstance(text, str) and text.strip() != ""
    has_href = isinstance(href, str) and href.strip() != ""
    has_specific_selector = isinstance(selector, str) and is_specific_selector(selector)
    numeric_only_text = has_text and norm_text(text).isdigit()

    has_semantic_anchor = has_text or has_href or has_specific_selector or tag in INTERACTIVE_TAGS
    purely_structural = tag in GENERIC_WIDGET_TAGS and not has_text and not has_href and not has_specific_selector
    return has_semantic_anchor and not purely_structural and not numeric_only_text

def keep_widget(
    node: Dict[str, Any],
    referenced_abstract_ids: Set[str],
    referenced_concrete_ids: Set[str],
) -> bool:
    """Return whether a widget node should remain in the filtered tree."""
    text = node.get("WebTextContent")
    selector = node.get("WebCssSelector")
    tag = node.get("WebTagName")

    if is_action_referenced_widget(node, referenced_abstract_ids, referenced_concrete_ids):
        return True

    has_text = isinstance(text, str) and text.strip() != ""
    has_selector = isinstance(selector, str) and selector.strip() != ""

    if has_selector and is_generic_selector(selector):
        # keep if it has meaningful text
        if has_text:
            return True
        return False

    # drop purely structural nodes
    if (not has_text) and (not has_selector) and (tag in GENERIC_TAGS):
        return False

    return True

def subtree_contains_action_reference(
    node: Any,
    referenced_abstract_ids: Set[str],
    referenced_concrete_ids: Set[str],
) -> bool:
    if isinstance(node, dict):
        if is_action_referenced_widget(node, referenced_abstract_ids, referenced_concrete_ids):
            return True
        children = node.get("children")
        if isinstance(children, list):
            return any(
                subtree_contains_action_reference(child, referenced_abstract_ids, referenced_concrete_ids)
                for child in children
            )
    elif isinstance(node, list):
        return any(
            subtree_contains_action_reference(item, referenced_abstract_ids, referenced_concrete_ids)
            for item in node
        )
    return False


def filter_widget_node(
    node: Any,
    referenced_abstract_ids: Set[str],
    referenced_concrete_ids: Set[str],
) -> Any:
    """Filter a widget subtree and mark retained traceability candidates."""
    if isinstance(node, dict):
        out: Dict[str, Any] = {}
        raw_children = node.get("children")
        new_children = []
        if isinstance(raw_children, list):
            for child in raw_children:
                filtered = filter_widget_node(child, referenced_abstract_ids, referenced_concrete_ids)
                if isinstance(filtered, dict):
                    new_children.append(filtered)
                elif isinstance(filtered, list):
                    new_children.extend(filtered)

        for key, value in node.items():
            if key in WIDGET_KEYS:
                if key != "children":
                    out[key] = value
        out["children"] = new_children
        out["TraceabilityCandidate"] = is_traceability_candidate(out)

        if keep_widget(out, referenced_abstract_ids, referenced_concrete_ids):
            return out

        return [
            child
            for child in new_children
            if subtree_contains_action_reference(child, referenced_abstract_ids, referenced_concrete_ids)
        ]
    if isinstance(node, list):
        new_list = []
        for item in node:
            filtered = filter_widget_node(item, referenced_abstract_ids, referenced_concrete_ids)
            if isinstance(filtered, dict):
                new_list.append(filtered)
            elif isinstance(filtered, list):
                new_list.extend(filtered)
        return new_list
    return node

def dedupe_widgets(
    tree: Any,
    referenced_abstract_ids: Set[str],
    referenced_concrete_ids: Set[str],
) -> Any:
    """Remove duplicate widget nodes while preserving action-linked widgets."""
    seen = set()

    def dedupe(node: Any) -> Any:
        if isinstance(node, dict):
            if isinstance(node.get("children"), list):
                new_children = []
                for child in node["children"]:
                    kept = dedupe(child)
                    if kept is not None:
                        new_children.append(kept)
                node["children"] = new_children

            if subtree_contains_action_reference(node, referenced_abstract_ids, referenced_concrete_ids):
                return node

            tag = node.get("WebTagName", "") or ""
            text = norm_text(node.get("WebTextContent", "") or "")
            selector = node.get("WebCssSelector", "") or ""
            key = (tag, text, selector)

            if key in seen:
                return None
            seen.add(key)

            return node
        if isinstance(node, list):
            new_list = []
            for item in node:
                kept = dedupe(item)
                if kept is not None:
                    new_list.append(kept)
            return new_list

        return node
    return dedupe(tree)

def main() -> None:
    """Load a raw TESTAR model, filter it, and write the strict JSON output."""
    parser = argparse.ArgumentParser(
        description="Filter TESTAR model output to retain high-value traceability fields."
    )
    parser.add_argument("--input", default=MODEL_PATH, help="Path to the raw model JSON.")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Path for the filtered model JSON.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        model = json.load(f)
    referenced_abstract_ids = {
        action.get("AbstractID")
        for action in model.get("ConcreteAction", [])
        if action.get("AbstractID")
    }
    referenced_concrete_ids = {
        action.get("ConcreteID")
        for action in model.get("ConcreteAction", [])
        if action.get("ConcreteID")
    }
    out: Dict[str, Any] = {
        "ConcreteState": [],
        "ConcreteAction": [],
        "ConcreteTransitions": model.get("ConcreteTransitions", []),
    }

    # Action filtering
    for action in model.get("ConcreteAction", []):
        filtered_action = filter_dict(action, ACTION_KEYS)
        out["ConcreteAction"].append(filtered_action)

    # States + widget tree filtering
    for state in model.get("ConcreteState", []):
        filtered_state = filter_dict(state, STATE_KEYS)
        widget_tree = state.get("WidgetTree")
        if widget_tree is not None:
            filtered_tree = filter_widget_node(
                widget_tree,
                referenced_abstract_ids,
                referenced_concrete_ids,
            )
            filtered_tree = dedupe_widgets(
                filtered_tree,
                referenced_abstract_ids,
                referenced_concrete_ids,
            )
            filtered_state["WidgetTree"] = filtered_tree
        out["ConcreteState"].append(filtered_state)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
