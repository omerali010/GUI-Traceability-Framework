"""Create relaxed filtered GUI models that keep more traceability evidence."""

import argparse
import json
import re
from typing import Any, Dict, List

MODEL_PATH = "Models/Gemma3_Llmreq.json"
OUTPUT_PATH = "Filtered relaxed models/filtered_model_relaxed_Gemma3_Llmreq.json"

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
    "InputText",
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

# Relaxed filtering removes fewer tags and selector patterns than the strict filter.
INTERACTIVE_TAGS = {"a", "button", "input", "select", "option", "textarea"}
GENERIC_WIDGET_TAGS = {
    "app-root",
    "body",
    "nav",
    "router-outlet",
    "span",
}
GENERIC_TAGS = {
    "span",
    "nav",
}
GENERIC_SELECTOR_PATTERNS = [
    r":nth-child\(",
]
SELECTOR_REGEX = [re.compile(pattern) for pattern in GENERIC_SELECTOR_PATTERNS]

def filter_dict(obj: Dict[str, Any], allowed: set) -> Dict[str, Any]:
    return {key: obj.get(key) for key in allowed if key in obj}

def is_generic_selector(selector: str) -> bool:
    """Return whether a selector looks too generic for traceability."""
    if not selector:
        return False
    return any(regex.search(selector) for regex in SELECTOR_REGEX)

def norm_text(text: str) -> str:
    """Normalize short widget strings for filtering decisions."""
    return re.sub(r"\s+", " ", text.strip().lower())

def is_specific_selector(selector: str) -> bool:
    """Return whether a selector contains a useful identifying marker."""
    return any(marker in selector for marker in ("#", "[", "data-test", "name=", "href="))

def is_traceability_candidate(node: Dict[str, Any]) -> bool:
    """Return whether a widget is meaningful enough to rank as a candidate."""
    text = node.get("WebTextContent")
    selector = node.get("WebCssSelector")
    tag = norm_text(node.get("WebTagName", "") or "")
    href = node.get("WebHref")

    has_text = isinstance(text, str) and text.strip() != ""
    has_href = isinstance(href, str) and href.strip() != ""
    has_specific_selector = isinstance(selector, str) and is_specific_selector(selector)

    has_semantic_anchor = has_text or has_href or has_specific_selector or tag in INTERACTIVE_TAGS
    purely_structural = tag in GENERIC_WIDGET_TAGS and not has_text and not has_href and not has_specific_selector
    return has_semantic_anchor and not purely_structural

def keep_widget(node: Dict[str, Any]) -> bool:
    """Return whether a widget node should remain in the relaxed tree."""
    text = node.get("WebTextContent")
    selector = node.get("WebCssSelector")
    tag = norm_text(node.get("WebTagName", "") or "")
    href = node.get("WebHref")

    has_text = isinstance(text, str) and text.strip() != ""
    has_selector = isinstance(selector, str) and selector.strip() != ""
    has_href = isinstance(href, str) and href.strip() != ""

    if has_selector and is_generic_selector(selector):
        if has_text or has_href or is_specific_selector(selector) or tag in INTERACTIVE_TAGS:
            return True
        return False

    if (not has_text) and (not has_selector) and (not has_href) and (tag in GENERIC_TAGS):
        return False

    return True

def filter_widget_tree(node: Any) -> List[Dict[str, Any]]:
    """Filter a widget subtree and mark retained traceability candidates."""
    if isinstance(node, dict):
        out: Dict[str, Any] = {}
        raw_children = node.get("children")
        filtered_children: List[Dict[str, Any]] = []
        if isinstance(raw_children, list):
            for child in raw_children:
                filtered_children.extend(filter_widget_tree(child))

        for key, value in node.items():
            if key in WIDGET_KEYS and key != "children":
                out[key] = value

        out["children"] = filtered_children
        out["TraceabilityCandidate"] = is_traceability_candidate(out)

        if keep_widget(out):
            return [out]
        return filtered_children

    if isinstance(node, list):
        flattened: List[Dict[str, Any]] = []
        for item in node:
            flattened.extend(filter_widget_tree(item))
        return flattened

    return []

def main() -> None:
    """Load a raw TESTAR model, filter it, and write the relaxed JSON output."""
    parser = argparse.ArgumentParser(
        description="Create a relaxed filtered TESTAR model that preserves more potentially traceable GUI evidence."
    )
    parser.add_argument("--input", default=MODEL_PATH, help="Path to the raw model JSON.")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Path for the filtered model JSON.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as file_handle:
        model = json.load(file_handle)

    out: Dict[str, Any] = {
        "ConcreteState": [],
        "ConcreteAction": [],
        "ConcreteTransitions": model.get("ConcreteTransitions", []),
    }
    for action in model.get("ConcreteAction", []):
        out["ConcreteAction"].append(filter_dict(action, ACTION_KEYS))

    for state in model.get("ConcreteState", []):
        filtered_state = filter_dict(state, STATE_KEYS)
        widget_tree = state.get("WidgetTree")
        if widget_tree is not None:
            filtered_state["WidgetTree"] = filter_widget_tree(widget_tree)
        out["ConcreteState"].append(filtered_state)

    with open(args.output, "w", encoding="utf-8") as file_handle:
        json.dump(out, file_handle, ensure_ascii=False, indent=2)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
