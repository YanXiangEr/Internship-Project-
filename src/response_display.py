"""Pure display formatting for microorganism response values.

The functions in this module never alter values passed to the database or
prediction code. They only build clearer, human-readable UI text.
"""

from __future__ import annotations

import re


RANGE_PATTERN = re.compile(
    r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)(?![\w.])"
)


def _normalise_token(value):
    return " ".join(
        str(value or "")
        .strip()
        .lower()
        .replace("_", " ")
        .split()
    )


def format_readable_value(value):
    """Improve spacing and range typography without changing semantics."""

    if value is None:
        return ""

    text = str(value).replace("_", " ").strip()
    text = re.sub(r"\bper\s+10\s+um\b", "per 10 µm", text, flags=re.I)
    text = re.sub(r"\bug/L\b", "µg/L", text, flags=re.I)
    text = re.sub(r"\buL/L\b", "µL/L", text, flags=re.I)
    text = RANGE_PATTERN.sub(lambda match: f"{match.group(1)}–{match.group(2)}", text)
    return text


def _format_percent(value):
    percent = value * 100.0
    if abs(percent - round(percent)) < 1e-10:
        return f"{int(round(percent))}%"
    return f"{percent:.4g}%"


def _fraction_range_display(response_type, raw_value):
    response_key = _normalise_token(response_type)
    raw_text = str(raw_value or "").strip()
    match = RANGE_PATTERN.search(raw_text)

    fraction_context = (
        "fraction" in response_key
        or re.search(r"\bfraction\b", raw_text, flags=re.I) is not None
    )
    if not match or not fraction_context:
        return None

    minimum = float(match.group(1))
    maximum = float(match.group(2))
    if not (0.0 <= minimum <= 1.0 and 0.0 <= maximum <= 1.0):
        return None

    return f"{_format_percent(minimum)}–{_format_percent(maximum)}"


def describe_response_value(response_type, value):
    """Return display text and a conservative plain-language explanation."""

    raw_value = "" if value is None else str(value).strip()
    readable_raw = format_readable_value(raw_value)
    response_key = _normalise_token(response_type)
    value_key = _normalise_token(value)

    result = {
        "raw_value": raw_value,
        "readable_raw_value": readable_raw,
        "display_value": readable_raw or "Not specified",
        "plain_language": "",
        "value_kind": "categorical",
    }

    if response_key == "presence":
        if value_key in {
            "absent",
            "not detected",
            "not found",
            "not observed",
            "undetected",
        }:
            result.update(
                display_value="Not observed",
                plain_language=(
                    "The organism was not observed in the supporting treatment. "
                    "This is not being interpreted as mortality."
                ),
                value_kind="presence",
            )
            return result

        if value_key in {"present", "detected", "observed", "found"}:
            result.update(
                display_value="Observed",
                plain_language=(
                    "The organism was observed in the supporting treatment."
                ),
                value_kind="presence",
            )
            return result

    if value_key in {"present", "true", "yes"}:
        result.update(
            display_value="Observed",
            plain_language="This response was recorded as present.",
            value_kind="boolean",
        )
        return result

    if value_key in {"absent", "false", "no"}:
        result.update(
            display_value="Not observed",
            plain_language="This response was recorded as absent.",
            value_kind="boolean",
        )
        return result

    direction_labels = {
        "increase": (
            "Increased",
            "An increase was reported relative to the supporting comparison.",
        ),
        "increased": (
            "Increased",
            "An increase was reported relative to the supporting comparison.",
        ),
        "decrease": (
            "Decreased",
            "A decrease was reported relative to the supporting comparison.",
        ),
        "decreased": (
            "Decreased",
            "A decrease was reported relative to the supporting comparison.",
        ),
        "no change": (
            "Unchanged",
            "No change was reported relative to the supporting comparison.",
        ),
        "unchanged": (
            "Unchanged",
            "No change was reported relative to the supporting comparison.",
        ),
    }
    if value_key in direction_labels:
        display_value, explanation = direction_labels[value_key]
        result.update(
            display_value=display_value,
            plain_language=explanation,
            value_kind="direction",
        )
        return result

    if value_key in {
        "decrease or no change",
        "decreased or unchanged",
    }:
        result.update(
            display_value="Decreased or unchanged",
            plain_language=(
                "The supporting evidence contains more than one reported "
                "outcome: decreased or unchanged."
            ),
            value_kind="alternative",
        )
        return result

    fraction_display = _fraction_range_display(response_type, raw_value)
    if fraction_display:
        result.update(
            display_value=fraction_display,
            plain_language=(
                "Observed range reported by the supporting evidence. "
                "The range is study-specific and is not a probability."
            ),
            value_kind="fraction_range",
        )
        return result

    if RANGE_PATTERN.search(raw_value):
        result.update(
            display_value=readable_raw,
            plain_language="Observed range reported by the supporting evidence.",
            value_kind="numeric_range",
        )
        return result

    if " or " in value_key:
        result.update(
            display_value=readable_raw,
            plain_language=(
                "The supporting evidence contains multiple possible reported "
                "values; the model has not selected one as uniquely correct."
            ),
            value_kind="alternative",
        )

    return result
