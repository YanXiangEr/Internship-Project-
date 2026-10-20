"""Evidence-aware narrative helpers for predicted response summaries."""

from __future__ import annotations

from collections.abc import Mapping


OCCURRENCE_RESPONSE_TYPES = {
    "presence",
    "organism_presence",
    "occurrence",
    "detection_status",
}

OBSERVED_VALUES = {
    "present",
    "observed",
    "detected",
    "found",
    "true",
    "yes",
    "1",
}

NOT_OBSERVED_VALUES = {
    "absent",
    "not_detected",
    "not detected",
    "not_found",
    "not found",
    "not_observed",
    "not observed",
    "undetected",
    "false",
    "no",
    "0",
}


def normalise(value):
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def response_values(response):
    """Return unique stored and supported values without changing their order."""

    if isinstance(response, Mapping):
        supported_values = response.get("supported_values", []) or []
        values = []
        for supported in supported_values:
            if isinstance(supported, Mapping):
                value = str(supported.get("value", "")).strip()
            else:
                value = str(supported or "").strip()
            if value and value not in values:
                values.append(value)

        fallback = str(response.get("value", "")).strip()
        if fallback and fallback not in values:
            values.append(fallback)
        return values

    value = str(response or "").strip()
    return [value] if value else []


def occurrence_narrative(species, organism_state):
    """Describe occurrence evidence and return remaining response entries.

    Unknown occurrence values remain in the returned response mapping so the
    ordinary response formatter can still display them without information loss.
    """

    if not isinstance(organism_state, Mapping):
        return "", organism_state

    occurrence_entries = {
        key: value
        for key, value in organism_state.items()
        if normalise(key) in {normalise(item) for item in OCCURRENCE_RESPONSE_TYPES}
    }

    if not occurrence_entries:
        return "", dict(organism_state)

    classifications = []
    for response in occurrence_entries.values():
        for value in response_values(response):
            cleaned = normalise(value)
            if cleaned in {normalise(item) for item in OBSERVED_VALUES}:
                label = "observed"
            elif cleaned in {normalise(item) for item in NOT_OBSERVED_VALUES}:
                label = "not_observed"
            else:
                label = "unknown"
            if label not in classifications:
                classifications.append(label)

    # Retain unrecognised occurrence values in the ordinary formatter.
    if not classifications or "unknown" in classifications:
        return "", dict(organism_state)

    remaining = {
        key: value
        for key, value in organism_state.items()
        if key not in occurrence_entries
    }
    species_text = str(species or "The organism").strip()

    if classifications == ["observed"]:
        sentence = (
            f"The occurrence evidence indicates that {species_text} was "
            "observed in the supporting observation."
        )
    elif classifications == ["not_observed"]:
        sentence = (
            f"The occurrence evidence indicates that {species_text} was not "
            "observed in the supporting observation."
        )
    else:
        sentence = (
            f"The occurrence evidence for {species_text} contains both "
            "observed and not-observed outcomes across the supporting "
            "observations."
        )

    return sentence, remaining

