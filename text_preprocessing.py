"""Shared text cleanup used before building IR and embedding inputs."""

import re
from typing import Any, Dict, Iterable, List

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


def text_value(node: Dict[str, Any], key: str) -> str:
    value = node.get(key)
    return value.strip() if isinstance(value, str) else ""


def first_nonempty_text(node: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = text_value(node, key)
        if value:
            return value
    return ""


def selector_tokens(selector: str) -> str:
    if not selector:
        return ""
    selector = selector.replace("\\/", "/")
    selector = selector.replace("\\:", ":")
    return selector
