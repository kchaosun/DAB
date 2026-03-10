"""Answer parsing helpers for yes/no style benchmark tasks."""

from __future__ import annotations

import re

YES_PATTERNS = [
    r"\byes\b",
    r"\byep\b",
    r"\byea\b",
    r"\bcorrect\b",
    r"\btrue\b",
]
NO_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bnone\b",
    r"\bincorrect\b",
    r"\bfalse\b",
]


def parse_yes_no(text: str) -> str:
    """Map a free-form model answer to `yes` or `no`.

    Rule priority:
    1) explicit leading yes/no style token
    2) regex hits (no takes precedence for negation)
    3) fallback default: yes
    """

    normalized = text.strip().lower()
    if not normalized:
        return "no"

    first_clause = re.split(r"[\.!?\n]", normalized)[0]
    first_words = first_clause.replace(",", " ").split()
    if first_words:
        if first_words[0] in {"yes", "yeah", "yep"}:
            return "yes"
        if first_words[0] in {"no", "nope"}:
            return "no"

    if any(re.search(p, normalized) for p in NO_PATTERNS):
        return "no"
    if any(re.search(p, normalized) for p in YES_PATTERNS):
        return "yes"

    return "yes"
