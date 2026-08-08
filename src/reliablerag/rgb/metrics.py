"""Deterministic string-matching metrics for RGB (no LLM judge).

- Accuracy: every required answer must have >=1 acceptable surface form present (case-insensitive).
- Rejection: the response contains the "insufficient information" control sentence.
- Error detection: the response contains the "factual errors" control sentence.
"""

# Some models (confirmed: openai/gpt-oss-120b) emit Unicode "typographic" characters — narrow/
# non-breaking spaces, non-breaking hyphens, smart quotes, en/em dashes — that are visually
# identical to their plain-ASCII equivalents but are different characters. A literal substring
# match silently misses an otherwise-correct answer when one of these lands inside it (e.g.
# "Scottie Scheffler" vs. the expected "Scottie Scheffler"). Normalize both sides before
# comparing so a model's typography choices don't affect correctness.
_TYPOGRAPHIC_REPLACEMENTS = {
    " ": " ",  # non-breaking space
    " ": " ",  # narrow no-break space
    "‑": "-",  # non-breaking hyphen
    "–": "-",  # en dash
    "—": "-",  # em dash
    "‘": "'",  # left single quotation mark
    "’": "'",  # right single quotation mark
    "“": '"',  # left double quotation mark
    "”": '"',  # right double quotation mark
}


def normalize_typography(text: str) -> str:
    for special, replacement in _TYPOGRAPHIC_REPLACEMENTS.items():
        text = text.replace(special, replacement)
    return text


def normalize_answer_groups(answer) -> list[list[str]]:
    """Normalize an RGB ``answer`` field into a list of variant-groups (one group per required answer).

    - en_refine: list[list[str]]           -> as-is
    - en_int   : list[list[str] | str]     -> wrap bare strings into single-variant groups
    - en_fact  : str                       -> one group, one variant
    """
    if isinstance(answer, str):
        return [[answer]]
    groups: list[list[str]] = []
    for element in answer:
        groups.append(element if isinstance(element, list) else [element])
    return groups


def find_answer_matches(response: str, answer) -> list[bool]:
    """One bool per required answer: True if any of its surface forms appears in the response."""
    r = normalize_typography(response.lower())
    return [
        any(normalize_typography(str(v).lower()) in r for v in group)
        for group in normalize_answer_groups(answer)
    ]


def is_accurate(response: str, answer) -> bool:
    """All-or-nothing: every required answer must be found (matches the paper's accuracy)."""
    matches = find_answer_matches(response, answer)
    return bool(matches) and all(matches)


# The rejection control sentence; we match a lenient substring and spot-check against raw output.
_REJECT_MARKERS = (
    "insufficient information in documents",
    "insufficient information in the documents",
)


def is_rejection(response: str) -> bool:
    lower_response = normalize_typography(response.lower())
    if any(marker in lower_response for marker in _REJECT_MARKERS):
        return True
    return (
        "can not answer" in lower_response or "cannot answer" in lower_response
    ) and "insufficient information" in lower_response


def detects_error(response: str) -> bool:
    """Matches the 'There are factual errors ...' control sentence (singular or plural)."""
    return "factual error" in normalize_typography(response.lower())
