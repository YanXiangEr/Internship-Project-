"""
visual_state.py

Response-aware evidence-backed visual-state resolver for the
microorganism world model.

Purpose
-------
This is the lightweight v1 visual-simulation layer.

It does NOT train or run a generative image model.

Instead, it selects the most relevant existing microscopy image using:

    1. same microorganism species
    2. same predicted healthy/stressed state
    3. predicted biological-response similarity
    4. water/environmental-context similarity

This is more biologically useful than matching on water-factor text alone,
because the main world model may already predict responses such as:

    cell_death
    pore_percolation
    frustule_breakage
    reduced_viability

Those predicted responses can help choose a closer visual analogue.

The internet-derived images are not required for this resolver. It can work
with the internal spreadsheet/Word-document-derived microscopy records already
present in world_model_v1_dataset.csv.
"""

import csv
import re
from pathlib import Path

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from visual_reference_safety import (
    reference_policy,
    resolve_display_source,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

WORLD_MODEL_DATASET = (
    DATA_DIR
    / "world_model_v1_dataset.csv"
)


# ============================================================
# SCORING SETTINGS
# ============================================================

# When predicted biological responses are available, visual-response
# agreement should matter more than generic environmental text similarity.

RESPONSE_SIMILARITY_WEIGHT = 0.70
ENVIRONMENT_SIMILARITY_WEIGHT = 0.30

# Only an explicit shared water-factor name is allowed to contribute to the
# environmental part of a visual match. This prevents generic words such as
# "present" or "treatment" from turning a very small TF-IDF overlap into a
# claimed condition match.
ENVIRONMENT_FACTOR_FIELDS = (
    "primary_water_factor",
    "secondary_water_factor",
)

# ============================================================
# VISUAL STATE
# ============================================================

class VisualState:
    """
    Structured evidence-backed visual state.

    The selected image is a visual analogue from the available evidence.
    It is not claimed to be a newly generated future microscopy image.
    """

    def __init__(
        self,
        species,
        predicted_condition=None,
        phenotype=None,
        visual_features=None,
        image_path=None,
        source_image_path=None,
        display_crop_used=False,
        display_crop_label=None,
        display_crop_notes=None,
        evidence_record_id=None,
        evidence_similarity=None,
        environment_similarity=None,
        response_similarity=None,
        matching_response_types=None,
        matching_environment_factors=None,
        evidence_type=None,
        description=None,
    ):

        self.species = (
            species
        )

        self.predicted_condition = (
            predicted_condition
        )

        self.phenotype = (
            phenotype
        )

        self.visual_features = (
            visual_features
            or []
        )

        self.image_path = (
            image_path
        )

        self.source_image_path = source_image_path or image_path
        self.display_crop_used = bool(display_crop_used)
        self.display_crop_label = display_crop_label
        self.display_crop_notes = display_crop_notes

        self.evidence_record_id = (
            evidence_record_id
        )

        # Combined response/environment match score.
        self.evidence_similarity = (
            evidence_similarity
        )

        self.environment_similarity = (
            environment_similarity
        )

        self.response_similarity = (
            response_similarity
        )

        self.matching_response_types = (
            matching_response_types
            or []
        )

        self.matching_environment_factors = (
            matching_environment_factors
            or []
        )

        self.evidence_type = (
            evidence_type
        )

        self.description = (
            description
        )

    @property
    def has_image(self):

        return bool(
            self.image_path
        )

    def to_dict(self):

        return {
            "species":
                self.species,

            "predicted_condition":
                self.predicted_condition,

            "phenotype":
                self.phenotype,

            "visual_features":
                list(
                    self.visual_features
                ),

            "image_path":
                self.image_path,

            "source_image_path":
                self.source_image_path,

            "display_crop_used":
                self.display_crop_used,

            "display_crop_label":
                self.display_crop_label,

            "display_crop_notes":
                self.display_crop_notes,

            "evidence_record_id":
                self.evidence_record_id,

            "evidence_similarity":
                self.evidence_similarity,

            "environment_similarity":
                self.environment_similarity,

            "response_similarity":
                self.response_similarity,

            "matching_response_types":
                list(
                    self.matching_response_types
                ),

            "matching_environment_factors":
                list(
                    self.matching_environment_factors
                ),

            "evidence_type":
                self.evidence_type,

            "description":
                self.description,
        }


# ============================================================
# HELPERS
# ============================================================

def clean(
    value
):

    if value is None:
        return ""

    return str(
        value
    ).strip()


def as_bool(
    value
):

    return (
        clean(
            value
        ).lower()
        in {
            "true",
            "1",
            "yes",
            "y",
        }
    )


def normalise_token_text(
    value
):
    """
    Normalise text for tolerant response matching.

    Examples
    --------
    pore_percolation=present
        ->
    pore percolation present
    """

    text = (
        clean(
            value
        )
        .lower()
        .replace(
            "_",
            " "
        )
        .replace(
            "=",
            " "
        )
        .replace(
            "|",
            " "
        )
        .replace(
            ";",
            " "
        )
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LOAD DATA
# ============================================================

def load_visual_records():

    if not WORLD_MODEL_DATASET.exists():

        raise FileNotFoundError(
            f"World-model dataset not found:\n"
            f"{WORLD_MODEL_DATASET}"
        )

    with open(
        WORLD_MODEL_DATASET,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        rows = list(
            csv.DictReader(
                file
            )
        )

    return rows


# ============================================================
# WATER STATE -> TEXT
# ============================================================

def water_state_to_text(
    water_state
):
    """
    Convert WorldState water metadata into search text.

    Numeric concentrations are kept here because the visual resolver may use
    them as retrieval context even though the separate healthy/stressed
    classifier is qualitative-only.
    """

    parts = []

    for factor, value in (
        water_state.items()
    ):

        factor_text = (
            clean(
                factor
            )
        )

        if isinstance(
            value,
            dict
        ):

            numeric_value = (
                value.get(
                    "value"
                )
            )

            unit = (
                value.get(
                    "unit"
                )
            )

            if (
                numeric_value
                is not None
            ):

                text = (
                    f"{factor_text} "
                    f"{numeric_value}"
                )

                if unit:

                    text += (
                        f" {unit}"
                    )

            else:

                text = (
                    factor_text
                )

        else:

            value_text = (
                clean(
                    value
                )
            )

            if value_text:

                text = (
                    f"{factor_text} "
                    f"{value_text}"
                )

            else:

                text = (
                    factor_text
                )

        parts.append(
            text
        )

    return " | ".join(
        parts
    )


# ============================================================
# RECORD SEARCH TEXT
# ============================================================

def build_record_environment_text(
    row
):

    fields = [
        "qualitative_condition",
        "primary_water_factor",
        "secondary_water_factor",
        "primary_concentration_value",
        "primary_concentration_unit",
        "secondary_concentration_value",
        "secondary_concentration_unit",
        "exposure_time",
        "exposure_unit",
    ]

    parts = []

    for field in fields:

        value = clean(
            row.get(
                field
            )
        )

        if value:

            parts.append(
                value
            )

    return " | ".join(
        parts
    )


def matching_environment_factors(
    row,
    water_state
):
    """Return explicitly shared water-factor names.

    Free-text similarity remains useful for ranking otherwise equivalent
    references, but it is not enough to claim a context match. A supported
    context match requires the requested factor name to occur in one of the
    record's dedicated water-factor fields.
    """

    requested = {
        normalise_token_text(factor)
        for factor in (water_state or {})
        if normalise_token_text(factor)
    }

    documented = set()

    for field in ENVIRONMENT_FACTOR_FIELDS:

        value = clean(row.get(field))

        if not value:
            continue

        for part in re.split(r"[|;,]", value):

            factor = normalise_token_text(part)

            if factor:
                documented.add(factor)

    return sorted(
        requested & documented
    )


def build_record_response_text(
    row
):
    """
    Build the output-side visual/biological response representation used only
    for selecting an image after the world model has already predicted its
    biological responses.
    """

    fields = [
        "state_label",
        "visual_phenotype",
        "response_types",
        "response_labels",
        "visual_feature_1",
        "visual_feature_2",
    ]

    parts = []

    for field in fields:

        value = clean(
            row.get(
                field
            )
        )

        if value:

            parts.append(
                value
            )

    return " | ".join(
        parts
    )


# ============================================================
# PREDICTED RESPONSES -> TEXT
# ============================================================

def build_predicted_response_text(
    predicted_responses
):

    if not predicted_responses:

        return ""

    parts = []

    for prediction in (
        predicted_responses
    ):

        response_type = clean(
            prediction.get(
                "response_type"
            )
        )

        response_value = clean(
            prediction.get(
                "response_value"
            )
        )

        if response_type:

            if response_value:

                parts.append(
                    f"{response_type} "
                    f"{response_value}"
                )

            else:

                parts.append(
                    response_type
                )

    return " | ".join(
        parts
    )


def predicted_response_types(
    predicted_responses
):

    if not predicted_responses:
        return []

    response_types = []

    for prediction in predicted_responses:

        response_type = clean(
            prediction.get(
                "response_type"
            )
        )

        if (
            response_type
            and response_type not in response_types
        ):

            response_types.append(
                response_type
            )

    return response_types


# ============================================================
# VISUAL FEATURES
# ============================================================

def collect_visual_features(
    row
):
    """
    Collect evidence-backed visual/biological feature annotations without
    displaying the same semantic feature more than once.

    Dataset fields can repeat a feature using different capitalisation,
    spacing, underscore style, or pipe/semicolon grouping. De-duplication is
    performed on each individual feature=value segment after tolerant
    normalisation, while preserving the first original display form.
    """

    features = []
    seen_feature_keys = set()

    for field in [
        "visual_phenotype",
        "response_labels",
        "visual_feature_1",
        "visual_feature_2",
    ]:

        value = clean(
            row.get(
                field
            )
        )

        if not value:

            continue

        # A semicolon can either introduce another feature expression or
        # another supported value for the current feature. Preserve the latter
        # so "trichome_shape=slightly_curved; straight" is not rendered as an
        # orphan bullet named "Straight".
        feature_parts = []

        for pipe_group in value.split("|"):

            for semicolon_part in pipe_group.split(";"):

                part = clean(semicolon_part)

                if not part:
                    continue

                if (
                    "=" not in part
                    and feature_parts
                    and "=" in feature_parts[-1]
                ):
                    feature_parts[-1] += f"; {part}"

                else:
                    feature_parts.append(part)

        for feature_part in feature_parts:

            if "=" in feature_part:

                feature_name, feature_value = feature_part.split(
                    "=",
                    1,
                )

                feature_name = clean(feature_name)
                feature_value = clean(feature_value)

                normalised_name = normalise_token_text(feature_name)
                normalised_value = normalise_token_text(feature_value)

                # Normalise inverted database fields such as
                # calyptra_absent=present into the readable property
                # calyptra=absent without changing the stored dataset.
                if (
                    normalised_name.endswith(" absent")
                    and normalised_value in {"present", "true", "yes", "1"}
                ):
                    feature_name = re.sub(
                        r"(?:_|\s)?absent$",
                        "",
                        feature_name,
                        flags=re.IGNORECASE,
                    )
                    feature_value = "absent"

                feature_values = [
                    clean(item)
                    for item in feature_value.split(";")
                    if clean(item)
                ]

                if len(feature_values) > 1:
                    feature_value = " or ".join(feature_values)

                feature_part = f"{feature_name}={feature_value}"

            feature_key = normalise_token_text(
                feature_part
            )

            if (
                feature_key
                and feature_key not in seen_feature_keys
            ):

                seen_feature_keys.add(
                    feature_key
                )

                features.append(
                    feature_part
                )

    return features


# ============================================================
# SIMILARITY
# ============================================================

def compute_tfidf_similarities(
    candidate_texts,
    query_text
):
    """
    Calculate candidate-to-query text similarities.

    Empty text is converted into a harmless placeholder so that TF-IDF does
    not fail with an empty vocabulary.
    """

    safe_candidates = [
        (
            text
            if clean(
                text
            )
            else "unspecified"
        )
        for text in candidate_texts
    ]

    safe_query = (
        query_text
        if clean(
            query_text
        )
        else "unspecified"
    )

    vectorizer = (
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(
                1,
                2
            ),
        )
    )

    matrix = (
        vectorizer.fit_transform(
            safe_candidates
            + [
                safe_query
            ]
        )
    )

    candidate_vectors = (
        matrix[
            :-1
        ]
    )

    query_vector = (
        matrix[
            -1
        ]
    )

    return (
        cosine_similarity(
            candidate_vectors,
            query_vector,
        )
        .flatten()
    )


def find_exact_response_matches(
    row,
    response_types
):
    """
    Detect response types whose names are explicitly present in the
    candidate's biological/morphological annotation.
    """

    if not response_types:
        return []

    candidate_text = (
        normalise_token_text(
            build_record_response_text(
                row
            )
        )
    )

    matches = []

    for response_type in (
        response_types
    ):

        normalised_type = (
            normalise_token_text(
                response_type
            )
        )

        if (
            normalised_type
            and
            normalised_type
            in candidate_text
        ):

            matches.append(
                response_type
            )

    return matches


# ============================================================
# VALUE-AWARE RESPONSE MATCHING
# ============================================================

def parse_numeric_range(
    value
):
    """
    Parse a simple numeric response value/range.

    Examples
    --------
    "2.8-3.0 um" -> (2.8, 3.0, "um")
    "20.8 percent" -> (20.8, 20.8, "percent")

    Returns None for categorical values.
    """

    text = normalise_token_text(
        value
    )

    if not text:

        return None

    range_match = re.fullmatch(
        r"\s*([-+]?\d+(?:\.\d+)?)\s*-\s*"
        r"([-+]?\d+(?:\.\d+)?)\s*([a-z0-9/]+)?\s*",
        text
    )

    if range_match:

        low = float(
            range_match.group(1)
        )

        high = float(
            range_match.group(2)
        )

        unit = clean(
            range_match.group(3)
        ).lower()

        return (
            min(low, high),
            max(low, high),
            unit
        )

    exact_match = re.fullmatch(
        r"\s*([-+]?\d+(?:\.\d+)?)\s*([a-z0-9/]+)?\s*",
        text
    )

    if exact_match:

        number = float(
            exact_match.group(1)
        )

        unit = clean(
            exact_match.group(2)
        ).lower()

        return (
            number,
            number,
            unit
        )

    return None


def numeric_ranges_overlap(
    first,
    second
):
    """
    Return True when two parsed numeric intervals overlap and their
    units are compatible.
    """

    if (
        first is None
        or second is None
    ):

        return False

    first_low, first_high, first_unit = first
    second_low, second_high, second_unit = second

    if (
        first_unit
        and second_unit
        and first_unit != second_unit
    ):

        return False

    return (
        max(
            first_low,
            second_low
        )
        <=
        min(
            first_high,
            second_high
        )
    )


def predictions_indicate_absence(
    predicted_responses
):
    """Return True only for an explicit organism-presence absence result.

    This is deliberately narrow. It does not treat zero growth, mortality, or
    an arbitrary feature whose value is "absent" as organism absence.
    """

    absence_values = {
        "absent",
        "not detected",
        "not found",
        "not observed",
        "undetected",
    }

    for prediction in predicted_responses or []:

        response_type = normalise_token_text(
            prediction.get("response_type")
        )

        response_value = normalise_token_text(
            prediction.get("response_value")
        )

        if (
            response_type == "presence"
            and response_value in absence_values
        ):
            return True

    return False


def extract_candidate_response_values(
    row,
    response_type
):
    """
    Extract values associated with one response type from the candidate
    record's structured response labels/features.

    The dataset stores response annotations as text such as:
        cell_length=2.8-3 um
        trichome_width=13.9-14.1 um
    """

    normalised_type = normalise_token_text(
        response_type
    )

    if not normalised_type:

        return []

    type_pattern = re.escape(
        normalised_type
    ).replace(
        r"\ ",
        r"[\s_]+"
    )

    values = []

    for field in [
        "response_labels",
        "visual_feature_1",
        "visual_feature_2",
        "visual_phenotype",
    ]:

        raw_text = clean(
            row.get(
                field
            )
        )

        if not raw_text:

            continue

        # Work with separators used throughout the generated dataset.
        segments = re.split(
            r"[|;]",
            raw_text
        )

        for segment in segments:

            segment_normalised = (
                normalise_token_text(
                    segment
                )
            )

            match = re.match(
                rf"^\s*{type_pattern}\s+(.+?)\s*$",
                segment_normalised
            )

            if match:

                candidate_value = clean(
                    match.group(1)
                )

                if (
                    candidate_value
                    and candidate_value not in values
                ):

                    values.append(
                        candidate_value
                    )

    return values


def response_value_match_score(
    predicted_value,
    candidate_value
):
    """
    Compare two values for the same response type.

    Returns
    -------
    1.0
        Exact categorical match, exact numeric match, or overlapping
        numeric ranges.
    0.0
        Explicitly incompatible categorical values or non-overlapping
        numeric ranges.
    None
        Value-aware comparison is not possible from the available text.
    """

    predicted_text = normalise_token_text(
        predicted_value
    )

    candidate_text = normalise_token_text(
        candidate_value
    )

    if (
        not predicted_text
        or not candidate_text
    ):

        return None

    if predicted_text == candidate_text:

        return 1.0

    # Bracketed text records evidence qualifiers such as significance,
    # severity, exposure trend, or study context. Preserve those qualifiers in
    # display text, while allowing the underlying categorical value to agree.
    # For example, "present" agrees with "present [significant]" and
    # "darker" agrees with "darker [increased with exposure]".
    def base_qualified_value(value):
        base = re.sub(
            r"\s*\[[^\]]+\]\s*",
            " ",
            value,
        )
        return re.sub(r"\s+", " ", base).strip()

    predicted_base = base_qualified_value(predicted_text)
    candidate_base = base_qualified_value(candidate_text)

    if (
        predicted_base
        and candidate_base
        and predicted_base == candidate_base
    ):

        return 1.0

    predicted_numeric = parse_numeric_range(
        predicted_text
    )

    candidate_numeric = parse_numeric_range(
        candidate_text
    )

    if (
        predicted_numeric is not None
        and candidate_numeric is not None
    ):

        return (
            1.0
            if numeric_ranges_overlap(
                predicted_numeric,
                candidate_numeric
            )
            else 0.0
        )

    if (
        predicted_numeric is None
        and candidate_numeric is None
    ):

        return 0.0

    return None


def compute_structured_response_agreement(
    row,
    predicted_responses
):
    """
    Score candidate response agreement without treating response-type
    overlap alone as a full value match.

    Multiple evidence-supported predicted values for the same response
    type are preserved. A candidate receives credit when its annotated
    value agrees with at least one supported predicted value.

    Returns
    -------
    score : float
        Mean agreement across unique predicted response types.
    matching_types : list[str]
        Unique response types explicitly present in the candidate.
    value_matching_types : list[str]
        Response types with a defensible value/range match.
    """

    if not predicted_responses:

        return (
            0.0,
            [],
            []
        )

    predictions_by_type = {}

    for prediction in predicted_responses:

        response_type = clean(
            prediction.get(
                "response_type"
            )
        )

        response_value = clean(
            prediction.get(
                "response_value"
            )
        )

        if not response_type:

            continue

        predictions_by_type.setdefault(
            response_type,
            []
        )

        if (
            response_value
            and response_value
            not in predictions_by_type[
                response_type
            ]
        ):

            predictions_by_type[
                response_type
            ].append(
                response_value
            )

    if not predictions_by_type:

        return (
            0.0,
            [],
            []
        )

    type_matches = find_exact_response_matches(
        row,
        list(
            predictions_by_type.keys()
        )
    )

    type_matches = list(
        dict.fromkeys(
            type_matches
        )
    )

    value_matching_types = []
    per_type_scores = []

    for response_type in predictions_by_type:

        if response_type not in type_matches:

            per_type_scores.append(
                0.0
            )
            continue

        candidate_values = (
            extract_candidate_response_values(
                row,
                response_type
            )
        )

        predicted_values = (
            predictions_by_type[
                response_type
            ]
        )

        comparisons = []

        for predicted_value in predicted_values:

            for candidate_value in candidate_values:

                comparison = (
                    response_value_match_score(
                        predicted_value,
                        candidate_value
                    )
                )

                if comparison is not None:

                    comparisons.append(
                        comparison
                    )

        if comparisons:

            best_match = max(
                comparisons
            )

            per_type_scores.append(
                best_match
            )

            if best_match > 0:

                value_matching_types.append(
                    response_type
                )

        else:

            # The response type is explicitly present, but the dataset
            # does not expose a safely comparable value. Give partial
            # type-level credit rather than pretending it is an exact
            # value match.
            per_type_scores.append(
                0.5
            )

    return (
        float(
            np.mean(
                per_type_scores
            )
        ),
        type_matches,
        value_matching_types
    )


# ============================================================
# RESOLVE VISUAL STATE
# ============================================================

def resolve_visual_state(
    species,
    predicted_condition,
    water_state,
    predicted_responses=None,
):
    """
    Find the closest evidence-backed visual example.

    Selection policy
    ----------------
    1. same species
    2. same predicted healthy/stressed condition
    3. image must exist
    4. score response/morphology agreement
    5. score environmental-context agreement
    6. combine both scores conservatively

    When predicted biological responses are available, response agreement
    receives more weight than generic water-context text.
    """

    if predictions_indicate_absence(predicted_responses):

        return VisualState(
            species=species,
            predicted_condition=predicted_condition,
            evidence_type="absence_no_visual_applicable",
            description=(
                "The evidence-backed response indicates that the organism "
                "was not observed in the supporting treatment. A "
                "condition-specific organism appearance is therefore not "
                "applicable. This absence result is not being reinterpreted "
                "as mortality."
            ),
        )

    rows = (
        load_visual_records()
    )

    requested_condition = clean(
        predicted_condition
    ).lower()

    candidates = [
        row
        for row in rows
        if (
            clean(
                row.get(
                    "species"
                )
            )
            ==
            species

            and

            clean(
                row.get(
                    "condition_label"
                )
            ).lower()
            ==
            requested_condition

            and

            as_bool(
                row.get(
                    "image_available"
                )
            )

            and

            clean(
                row.get(
                    "local_path"
                )
            )
        )
    ]

    reference_scope = (
        "same_state"
    )

    # Exact-species external images often have no defensible overall
    # healthy/stressed label. Retain them as state-neutral references rather
    # than inventing a condition. They are considered only when a same-state
    # image is unavailable, including when the biological condition itself is
    # intentionally unresolved.
    if not candidates:

        state_neutral_labels = {
            "state_neutral",
            "unresolved",
            "unknown",
        }

        candidates = [
            row
            for row in rows
            if (
                clean(
                    row.get(
                        "species"
                    )
                )
                ==
                species

                and

                clean(
                    row.get(
                        "condition_label"
                    )
                ).lower()
                in
                state_neutral_labels

                and

                as_bool(
                    row.get(
                        "image_available"
                    )
                )

                and

                clean(
                    row.get(
                        "local_path"
                    )
                )
            )
        ]

        reference_scope = (
            "state_neutral"
        )

    # --------------------------------------------------------
    # NO SAME-SPECIES / SAME-STATE VISUAL EVIDENCE
    # --------------------------------------------------------

    if not candidates:

        return VisualState(
            species=(
                species
            ),

            predicted_condition=(
                predicted_condition
            ),

            description=(
                "No same-species, same-state microscopy "
                "reference is currently available."
            ),
        )

    # --------------------------------------------------------
    # ENVIRONMENT SIMILARITY
    # --------------------------------------------------------

    environment_query = (
        water_state_to_text(
            water_state
        )
    )

    candidate_environment_texts = [
        build_record_environment_text(
            row
        )
        for row in candidates
    ]

    environment_similarities = (
        compute_tfidf_similarities(
            candidate_environment_texts,
            environment_query,
        )
    )

    matching_environment_factors_by_candidate = [
        matching_environment_factors(
            row=row,
            water_state=water_state,
        )
        for row in candidates
    ]

    # Generic words may create a non-zero text similarity even when the
    # reference has no documented factor in common with the simulation. Gate
    # the environmental score on an explicit factor-name match.
    environment_similarities = np.asarray(
        [
            float(score)
            if matching_environment_factors_by_candidate[index]
            else 0.0
            for index, score in enumerate(environment_similarities)
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # BIOLOGICAL RESPONSE SIMILARITY
    # --------------------------------------------------------

    response_query = (
        build_predicted_response_text(
            predicted_responses
        )
    )

    response_types = (
        predicted_response_types(
            predicted_responses
        )
    )

    candidate_response_texts = [
        build_record_response_text(
            row
        )
        for row in candidates
    ]

    if response_query:

        response_similarities = (
            compute_tfidf_similarities(
                candidate_response_texts,
                response_query,
            )
        )

    else:

        response_similarities = (
            np.zeros(
                len(
                    candidates
                ),
                dtype=float,
            )
        )

    # --------------------------------------------------------
    # COMBINED SCORE
    # --------------------------------------------------------

    combined_scores = []

    exact_matches_by_candidate = []

    value_matches_by_candidate = []

    structured_response_scores = []

    for index, row in enumerate(
        candidates
    ):

        (
            structured_response_score,
            exact_matches,
            value_matches
        ) = compute_structured_response_agreement(
            row=row,
            predicted_responses=predicted_responses,
        )

        exact_matches_by_candidate.append(
            exact_matches
        )

        value_matches_by_candidate.append(
            value_matches
        )

        structured_response_scores.append(
            structured_response_score
        )

        if response_query:

            # Structured response/value agreement is authoritative for
            # the biological-response part of visual matching. Generic
            # TF-IDF remains useful for retrieval context but no longer
            # allows a shared response name with an incompatible numeric
            # value to receive full biological-match credit.
            response_component = (
                structured_response_score
            )

            score = (
                RESPONSE_SIMILARITY_WEIGHT
                *
                response_component

                +

                ENVIRONMENT_SIMILARITY_WEIGHT
                *
                float(
                    environment_similarities[
                        index
                    ]
                )
            )

        else:

            score = float(
                environment_similarities[
                    index
                ]
            )

        combined_scores.append(
            min(
                float(
                    score
                ),
                1.0
            )
        )

    best_index = int(
        np.argmax(
            combined_scores
        )
    )

    best_row = (
        candidates[
            best_index
        ]
    )

    combined_similarity = float(
        combined_scores[
            best_index
        ]
    )

    environment_similarity = float(
        environment_similarities[
            best_index
        ]
    )

    response_similarity = float(
        structured_response_scores[
            best_index
        ]
    )

    matching_response_types = list(
        dict.fromkeys(
            exact_matches_by_candidate[
                best_index
            ]
        )
    )

    matching_water_factors = list(
        dict.fromkeys(
            matching_environment_factors_by_candidate[
                best_index
            ]
        )
    )

    value_matching_response_types = list(
        dict.fromkeys(
            value_matches_by_candidate[
                best_index
            ]
        )
    )

    # --------------------------------------------------------
    # RESOLVE LOCAL PATH
    # --------------------------------------------------------

    local_path = clean(
        best_row.get(
            "local_path"
        )
    )

    if local_path:

        path = Path(
            local_path
        )

        if not path.is_absolute():

            path = (
                PROJECT_ROOT
                / path
            )

        if not path.exists():

            local_path = None

        else:

            local_path = str(
                path
            )

    source_local_path = local_path
    display_crop_used = False
    display_crop_label = None
    display_crop_notes = None
    record_id = clean(best_row.get("record_id"))

    if source_local_path:
        policy = reference_policy(record_id)
        display_path = resolve_display_source(source_local_path, policy)
        if display_path.is_file() and display_path != Path(source_local_path):
            local_path = str(display_path)
            display_crop_used = True
            display_crop_label = clean(policy.get("display_panel_label")) or None
            display_crop_notes = clean(policy.get("display_notes")) or None

    # --------------------------------------------------------
    # MORPHOLOGY
    # --------------------------------------------------------

    visual_features = (
        collect_visual_features(
            best_row
        )
    )

    phenotype = clean(
        best_row.get(
            "visual_phenotype"
        )
    )

    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    if reference_scope == "state_neutral":

        description_parts = [
            (
                f"A state-neutral exact-species visual reference for "
                f"{species} was selected. The source does not support "
                f"labelling this image as healthy or stressed."
            )
        ]

    else:

        description_parts = [
            (
                f"The closest currently available "
                f"{predicted_condition} visual analogue for "
                f"{species} was selected from existing microscopy evidence."
            )
        ]

    if matching_response_types:

        description_parts.append(
            (
                "It explicitly shares the following predicted "
                "response type(s): "
                +
                ", ".join(
                    matching_response_types
                )
                +
                "."
            )
        )

    if value_matching_response_types:

        description_parts.append(
            (
                "Response-value agreement is supported for: "
                +
                ", ".join(
                    value_matching_response_types
                )
                +
                "."
            )
        )

    if matching_water_factors:

        description_parts.append(
            (
                "It explicitly shares the following environmental factor "
                "name(s): "
                + ", ".join(matching_water_factors)
                + "."
            )
        )

    if visual_features:

        feature_text = "; ".join(
            visual_features[
                :5
            ]
        )

        description_parts.append(
            (
                "Features documented in the selected reference record "
                "include: "
                f"{feature_text}."
            )
        )

        description_parts.append(
            (
                "These descriptors belong to the reference record and are "
                "not additional effects predicted for the current water "
                "condition."
            )
        )

    if response_query:

        description_parts.append(
            (
                f"Response similarity={response_similarity:.3f}; "
                f"environment similarity={environment_similarity:.3f}; "
                f"combined visual-match score={combined_similarity:.3f}."
            )
        )

    else:

        description_parts.append(
            (
                f"Environment similarity="
                f"{environment_similarity:.3f}."
            )
        )

    description_parts.append(
        (
            "This reference should be interpreted as an evidence-backed "
            "visual analogue, not as a synthetic image showing the exact "
            "future appearance under the supplied condition."
        )
    )

    return VisualState(
        species=(
            species
        ),

        predicted_condition=(
            predicted_condition
        ),

        phenotype=(
            phenotype
            or None
        ),

        visual_features=(
            visual_features
        ),

        image_path=(
            local_path
        ),

        source_image_path=(
            source_local_path
        ),

        display_crop_used=(
            display_crop_used
        ),

        display_crop_label=(
            display_crop_label
        ),

        display_crop_notes=(
            display_crop_notes
        ),

        evidence_record_id=record_id,

        evidence_similarity=(
            combined_similarity
        ),

        environment_similarity=(
            environment_similarity
        ),

        response_similarity=(
            response_similarity
        ),

        matching_response_types=(
            matching_response_types
        ),

        matching_environment_factors=(
            matching_water_factors
        ),

        evidence_type=clean(
            best_row.get(
                "evidence_type"
            )
        ),

        description=" ".join(
            description_parts
        ),
    )


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    examples = [
        {
            "species":
                "Nitzschia palea",

            "condition":
                "stressed",

            "water_state": {
                "petrol": {
                    "value": 10,
                    "unit": "uL/L",
                }
            },

            "predicted_responses": [
                {
                    "response_type":
                        "cell_death",

                    "response_value":
                        "20.8 percent",
                },
                {
                    "response_type":
                        "oil_production",

                    "response_value":
                        "increased",
                },
                {
                    "response_type":
                        "pore_percolation",

                    "response_value":
                        "present",
                },
            ],
        },

        {
            "species":
                "Nitzschia palea",

            "condition":
                "stressed",

            "water_state": {
                "petrol": {
                    "value": 100,
                    "unit": "uL/L",
                }
            },

            "predicted_responses": [
                {
                    "response_type":
                        "cell_death",

                    "response_value":
                        "97.6 percent",
                },
                {
                    "response_type":
                        "reduced_viability",

                    "response_value":
                        "present",
                },
                {
                    "response_type":
                        "lipid_body_change",

                    "response_value":
                        "decrease_or_no_change",
                },
                {
                    "response_type":
                        "pore_percolation",

                    "response_value":
                        "present",
                },
            ],
        },
    ]

    for example in examples:

        print()
        print(
            "=" * 78
        )

        print(
            example[
                "species"
            ],
            example[
                "water_state"
            ]
        )

        print(
            "=" * 78
        )

        result = (
            resolve_visual_state(
                species=(
                    example[
                        "species"
                    ]
                ),

                predicted_condition=(
                    example[
                        "condition"
                    ]
                ),

                water_state=(
                    example[
                        "water_state"
                    ]
                ),

                predicted_responses=(
                    example[
                        "predicted_responses"
                    ]
                ),
            )
        )

        print(
            result.to_dict()
        )
