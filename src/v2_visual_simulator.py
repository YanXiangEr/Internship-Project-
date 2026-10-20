"""
v2_visual_simulator.py

Evidence-aware lightweight V2 visual simulator.

Purpose
-------
Create a NEW illustrative microscopy-style image from an existing
evidence-backed reference image.

Pipeline
--------
species + predicted biological state + predicted responses
        ↓
select best internal same-species/same-state reference
        ↓
apply conservative response-specific transformations
        ↓
simulated visual output

Important
---------
This is an illustrative V2 baseline, not a scientifically validated
generative model. The transformations are deliberately conservative
and only applied to response types for which a visual cue is reasonably
interpretable.

No new images are downloaded.
No neural network is trained.
"""

import csv
import hashlib
import re
from pathlib import Path

from PIL import (
    Image,
    ImageEnhance,
    ImageFilter,
    ImageChops,
    ImageOps,
    ImageStat,
)

from visual_state import (
    compute_structured_response_agreement,
    predictions_indicate_absence,
)
from visual_reference_safety import (
    analyse_layout_signals,
    policy_allows_transform,
    policy_reason,
    reference_policy,
    resolve_project_image_path,
    resolve_transform_source,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

V2_DATASET = (
    DATA_DIR
    / "v2_visual_conditioning_dataset.csv"
)

# The main V2 dataset remains the authoritative conditioning table.  A small
# reviewed overlay can supply metadata for panel records that were created by
# the Word-document crop workflow after the base table was generated.  Rows
# from this file are admitted only when the transform-safety policy explicitly
# approves the corresponding record.
TRANSFORM_CANDIDATE_METADATA = (
    DATA_DIR
    / "visual_transform_candidate_metadata.csv"
)

OUTPUT_DIR = (
    DATA_DIR
    / "v2_simulated_images"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EVIDENCE-AWARE VISUAL TRANSFORMATION POLICY
# ============================================================

# These responses remain important biological evidence and can help
# choose the closest reference image, but they should not directly
# alter a static microscopy image in V2.
NON_STATIC_TRANSFORM_RESPONSES = {
    "presence",
    "relative_abundance",
    "abundance",
    "cell_density",
    "growth_rate",
    "cell_death",
    "cell_mortality_fraction",
    "mortality",
    "reduced_viability",
    "oil_production",
    "growth_inhibition",
    "growth_reduction",
    "growth_increase",
    "lipid_body_change",
    "population_reduction",
    "photosynthesis_inhibition",
    "avoidance_behavior",
    "gliding_motility",
    "metaboly",
    "oscillatory_motility",
    "pore_percolation",
}

# These categories explain why a supported response is not converted into a
# single-organism static morphology. They do not change biological evidence.
POPULATION_RESPONSES = {
    "relative_abundance",
    "abundance",
    "cell_density",
    "cell_death",
    "cell_mortality_fraction",
    "mortality",
    "reduced_viability",
    "population_reduction",
}

# Occurrence records describe whether the organism was observed in the
# supporting treatment. They are neither single-cell morphology nor a direct
# measurement of population size, so expose them separately in the UI.
OCCURRENCE_RESPONSES = {
    "presence",
}

TEMPORAL_RESPONSES = {
    "growth_rate",
    "growth_inhibition",
    "growth_reduction",
    "growth_increase",
}

BEHAVIORAL_RESPONSES = {
    "avoidance_behavior",
    "gliding_motility",
    "metaboly",
    "oscillatory_motility",
    "pore_percolation",
}

PHYSIOLOGICAL_RESPONSES = {
    "oil_production",
    "lipid_body_change",
    "photosynthesis_inhibition",
}

# Current narrow set of responses with explicit static-image rules.
STRUCTURAL_RESPONSES = {
    "frustule_breakage",
    "frustule_dispersion",
    "valve_deformation",
    "deformation",
}

PIGMENT_RESPONSES = {
    "pigmentation_change",
}

CHLOROPLAST_RESPONSES = {
    "chloroplast_damage",
}

# These two rules are restricted to reviewed fluorescence-like references.
# They attenuate measured colour/brightness signals only; they do not redraw
# organelles or claim to reconstruct exact fluorescence intensity.
FLUORESCENCE_REDUCTION_RESPONSES = {
    "reduced_chlorophyll_fluorescence",
}

# A reported decrease in chlorophyll is not automatically visible in an
# arbitrary bright-field image.  This separate rule is therefore restricted to
# the same reviewed fluorescence-like modality as fluorescence attenuation. It
# provides only a restrained signal-loss cue, not a quantitative chlorophyll
# reconstruction.
CHLOROPHYLL_DECREASE_RESPONSES = {
    "chlorophyll_decrease",
}

CHLOROPLAST_LOSS_RESPONSES = {
    "chloroplast_absent",
}

STATIC_TRANSFORM_RESPONSES = (
    STRUCTURAL_RESPONSES
    | PIGMENT_RESPONSES
    | CHLOROPLAST_RESPONSES
    | FLUORESCENCE_REDUCTION_RESPONSES
    | CHLOROPHYLL_DECREASE_RESPONSES
    | CHLOROPLAST_LOSS_RESPONSES
)


# ============================================================
# HELPERS
# ============================================================

def clean(value):

    if value is None:
        return ""

    return str(value).strip()


def normalise_text(value):

    text = clean(value).lower()

    text = text.replace(
        "_",
        " "
    )

    text = text.replace(
        "=",
        " "
    )

    text = text.replace(
        "|",
        " "
    )

    text = text.replace(
        ";",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def load_transform_candidate_metadata():

    if not TRANSFORM_CANDIDATE_METADATA.exists():

        return []

    with open(
        TRANSFORM_CANDIDATE_METADATA,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        return list(
            csv.DictReader(
                file
            )
        )


def merge_approved_transform_candidates(
    dataset_rows,
    candidate_rows,
):
    """Add missing, explicitly approved panel metadata to V2 selection.

    Existing conditioning rows always win.  An overlay row is ignored unless
    it has the minimum metadata needed for value-aware matching and its record
    is approved by the existing visual-reference safety policy.  This makes
    the mechanism general for future reviewed crops without turning
    context-only or unreviewed figures into transformation candidates.
    """

    merged = list(dataset_rows or [])
    known_ids = {
        clean(row.get("record_id"))
        for row in merged
        if clean(row.get("record_id"))
    }

    required_fields = {
        "record_id",
        "species",
        "biological_state",
        "image_path",
        "response_types",
        "response_labels",
    }

    for source_row in candidate_rows or []:

        row = dict(source_row)
        record_id = clean(row.get("record_id"))

        if not record_id or record_id in known_ids:
            continue

        if any(not clean(row.get(field)) for field in required_fields):
            continue

        policy = reference_policy(record_id)
        if not policy_allows_transform(policy):
            continue

        row["candidate_source"] = (
            "approved_record_metadata_overlay"
        )
        merged.append(row)
        known_ids.add(record_id)

    return merged


def load_dataset():

    if not V2_DATASET.exists():

        raise FileNotFoundError(
            f"V2 dataset not found:\n"
            f"{V2_DATASET}"
        )

    with open(
        V2_DATASET,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        dataset_rows = list(
            csv.DictReader(
                file
            )
        )

    return merge_approved_transform_candidates(
        dataset_rows,
        load_transform_candidate_metadata(),
    )


def deterministic_fraction(text):

    digest = hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        int(
            digest[:8],
            16
        )
        /
        0xFFFFFFFF
    )


def first_numeric_value(text):

    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
        clean(text)
    )

    if not match:

        return None

    try:

        return float(
            match.group(0)
        )

    except ValueError:

        return None


# ============================================================
# REFERENCE SELECTION
# ============================================================
#
# Reference-response matching uses the same structured, value-aware
# agreement policy as V1 visual_state.py. This avoids the old token-overlap
# score, which could over-credit shared words and produce hard-to-interpret
# values such as 0.786 for an otherwise exact response match.

def build_candidate_text(row):

    fields = [
        "response_types",
        "response_labels",
        "visual_phenotype",
        "visual_feature_1",
        "visual_feature_2",
        "visual_description",
    ]

    return " ".join(
        normalise_text(
            row.get(
                field
            )
        )
        for field in fields
        if clean(
            row.get(
                field
            )
        )
    )


def build_response_query(
    predicted_responses
):

    if not predicted_responses:

        return ""

    parts = []

    for response in predicted_responses:

        response_type = clean(
            response.get(
                "response_type"
            )
        )

        response_value = clean(
            response.get(
                "response_value"
            )
        )

        if response_type:

            parts.append(
                response_type
            )

        if response_value:

            parts.append(
                response_value
            )

    return normalise_text(
        " ".join(
            parts
        )
    )


def response_overlap_score(
    candidate_text,
    query_text
):
    """
    Legacy token-overlap helper retained for backward compatibility.

    V2 reference selection no longer uses this score. Structured,
    value-aware response agreement from visual_state.py is used instead.
    """

    if not query_text:

        return 0.0

    candidate_tokens = set(
        candidate_text.split()
    )

    query_tokens = set(
        query_text.split()
    )

    if not query_tokens:

        return 0.0

    return (
        len(
            candidate_tokens
            &
            query_tokens
        )
        /
        len(
            query_tokens
        )
    )


def supporting_observation_ids(predicted_responses):
    """
    Collect observation IDs that directly support the retained predictions.

    These IDs are used only as a tie-break after structured response-value
    agreement. They do not override a better biological-response match.
    """

    observation_ids = set()

    for response in predicted_responses or []:

        for evidence in response.get(
            "supporting_evidence",
            []
        ) or []:

            observation_id = clean(
                evidence.get(
                    "observation_id"
                )
            )

            if observation_id:

                observation_ids.add(
                    observation_id
                )

    return observation_ids


def candidate_observation_id(row):
    """
    Resolve the source observation represented by a V2 conditioning row.

    Prefer an explicit observation_id column when available. Otherwise derive
    it conservatively from record_id values such as
    Nitzschia_palea_stressed_row20_image1.
    """

    explicit = clean(
        row.get(
            "observation_id"
        )
    )

    if explicit:

        return explicit

    record_id = clean(
        row.get(
            "record_id"
        )
    )

    if not record_id:

        return ""

    return re.sub(
        r"_image\d+$",
        "",
        record_id,
        flags=re.IGNORECASE,
    )


def _reference_agreement(row, predicted_responses):
    """Return structured agreement details for one visual reference."""

    if not predicted_responses:
        return 0.0, set(), set()

    score, matching_types, value_matching_types = (
        compute_structured_response_agreement(
            row=row,
            predicted_responses=predicted_responses,
        )
    )

    return (
        float(score),
        set(matching_types or []),
        set(value_matching_types or []),
    )


def _approved_transform_candidate(row, predicted_responses):
    """Check whether a reference is a safe, response-compatible V2 source.

    Safety approval alone is deliberately insufficient. The reference must
    also agree on the value of at least one predicted response that has an
    explicit static-image rule.
    """

    policy = reference_policy(row.get("record_id"))

    if not policy_allows_transform(policy):
        return None

    full_path = resolve_project_image_path(row.get("image_path"))
    if full_path is None or not full_path.is_file():
        return None

    transform_path = resolve_transform_source(full_path, policy)
    if transform_path is None:
        return None

    score, matching_types, value_matching_types = _reference_agreement(
        row,
        predicted_responses,
    )
    supported_static_types = (
        value_matching_types & STATIC_TRANSFORM_RESPONSES
    )

    if not supported_static_types:
        return None

    return {
        "row": row,
        "score": score,
        "matching_types": matching_types,
        "value_matching_types": value_matching_types,
        "supported_static_types": supported_static_types,
    }


def _best_transform_fallback(candidates, predicted_responses):
    """Select the strongest approved response-compatible V2 reference."""

    supporting_ids = supporting_observation_ids(predicted_responses)
    eligible = []

    for row in candidates:
        result = _approved_transform_candidate(
            row,
            predicted_responses,
        )
        if result is None:
            continue

        observation_id = candidate_observation_id(row)
        result["evidence_tiebreak"] = (
            1 if observation_id in supporting_ids else 0
        )
        eligible.append(result)

    eligible.sort(
        key=lambda item: (
            -len(item["supported_static_types"]),
            -item["score"],
            -item["evidence_tiebreak"],
            clean(item["row"].get("record_id")),
        )
    )

    return eligible[0] if eligible else None


def select_reference(
    species,
    biological_state,
    predicted_responses=None,
    preferred_reference_record_id=None,
):

    rows = load_dataset()

    requested_state = clean(
        biological_state
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
                    "biological_state"
                )
            ).lower()
            ==
            requested_state
        )
    ]

    if not candidates:

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
                        "biological_state"
                    )
                ).lower()
                in
                {
                    "state_neutral",
                    "unresolved",
                    "unknown",
                }
            )
        ]

    if not candidates:

        raise ValueError(
            (
                "No reference image exists for "
                f"{species} in state "
                f"'{biological_state or 'unresolved'}'."
            )
        )

    preferred_reference_record_id = clean(
        preferred_reference_record_id
    )

    # V1 has already selected the strongest overall visual analogue using
    # response and environmental evidence. V2 reuses it when safe. If it is
    # not transform-approved, V2 may use a separate reviewed crop only when
    # that crop has value-level agreement with a predicted static response.
    # This preserves V1 as the contextual analogue without weakening any
    # transformation-safety rule.
    if preferred_reference_record_id:

        for row in candidates:

            if clean(row.get("record_id")) != preferred_reference_record_id:
                continue

            preferred_score, _, _ = _reference_agreement(
                row,
                predicted_responses,
            )

            preferred_policy = reference_policy(
                preferred_reference_record_id
            )
            preferred_full_path = resolve_project_image_path(
                row.get("image_path")
            )
            preferred_transform_path = None
            if (
                preferred_full_path is not None
                and preferred_full_path.is_file()
            ):
                preferred_transform_path = resolve_transform_source(
                    preferred_full_path,
                    preferred_policy,
                )

            preferred_is_transform_safe = (
                preferred_transform_path is not None
            )
            requested_static_types = (
                response_types_from_predictions(predicted_responses)
                & STATIC_TRANSFORM_RESPONSES
            )

            if preferred_is_transform_safe or not requested_static_types:
                return row, preferred_score

            fallback = _best_transform_fallback(
                [
                    candidate
                    for candidate in candidates
                    if clean(candidate.get("record_id"))
                    != preferred_reference_record_id
                ],
                predicted_responses,
            )

            if fallback is not None:
                return fallback["row"], fallback["score"]

            return row, preferred_score

    query_text = (
        build_response_query(
            predicted_responses
        )
    )

    supporting_ids = (
        supporting_observation_ids(
            predicted_responses
        )
    )

    scored = []

    for row in candidates:

        score, _matching_types, _value_matching_types = (
            _reference_agreement(
                row,
                predicted_responses,
            )
        )

        observation_id = (
            candidate_observation_id(
                row
            )
        )

        evidence_tiebreak = (
            1
            if observation_id in supporting_ids
            else 0
        )

        scored.append(
            (
                float(
                    score
                ),
                evidence_tiebreak,
                clean(
                    row.get(
                        "record_id"
                    )
                ),
                row,
            )
        )

    # Primary criterion: structured response-value agreement.
    # Tie-break 1: candidate comes from an observation that directly supports
    # the current retained predictions.
    # Tie-break 2: record_id for deterministic behaviour across CSV order.
    scored.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            item[2],
        )
    )

    return (
        scored[0][3],
        scored[0][0],
    )


# ============================================================
# STRESS INTENSITY
# ============================================================

def response_types_from_predictions(predicted_responses):

    return {
        clean(
            response.get(
                "response_type"
            )
        )
        for response in (
            predicted_responses
            or []
        )
        if clean(
            response.get(
                "response_type"
            )
        )
    }


def classify_predicted_responses(predicted_responses):

    response_types = response_types_from_predictions(
        predicted_responses
    )

    return {
        "transformable":
            sorted(
                response_types
                &
                STATIC_TRANSFORM_RESPONSES
            ),

        "non_static":
            sorted(
                response_types
                &
                NON_STATIC_TRANSFORM_RESPONSES
            ),

        "unknown":
            sorted(
                response_types
                -
                STATIC_TRANSFORM_RESPONSES
                -
                NON_STATIC_TRANSFORM_RESPONSES
            ),
    }


def classify_visual_response_roles(predicted_responses):
    """Group responses by the kind of visual evidence they would require."""

    response_types = response_types_from_predictions(
        predicted_responses
    )

    static_morphology = response_types & STATIC_TRANSFORM_RESPONSES
    population = response_types & POPULATION_RESPONSES
    occurrence = response_types & OCCURRENCE_RESPONSES
    temporal = response_types & TEMPORAL_RESPONSES
    behavioral = response_types & BEHAVIORAL_RESPONSES
    physiological = response_types & PHYSIOLOGICAL_RESPONSES

    assigned = (
        static_morphology
        | occurrence
        | population
        | temporal
        | behavioral
        | physiological
    )

    return {
        "static_morphology": sorted(static_morphology),
        "occurrence_level": sorted(occurrence),
        "population_level": sorted(population),
        "temporal": sorted(temporal),
        "behavioral": sorted(behavioral),
        "physiological_non_static": sorted(physiological),
        "unclassified": sorted(response_types - assigned),
    }


def analyse_reference_image(image):
    """Return conservative image-suitability signals for visual rules.

    The project dataset does not yet contain authoritative microscopy-modality
    or organism-mask fields. These signals therefore gate transformations; they
    are not treated as scientific image classifications.
    """

    return analyse_layout_signals(image)


def evaluate_visual_rule_applicability(
    predicted_responses,
    reference_profile,
    reference_is_state_neutral=False,
    transform_policy=None,
):
    """Decide which requested rules are safe enough to apply to this image."""

    response_types = response_types_from_predictions(
        predicted_responses
    )
    requested = []

    if response_types & STRUCTURAL_RESPONSES:
        requested.append(
            (
                "structural",
                sorted(response_types & STRUCTURAL_RESPONSES),
                "No colour-channel requirement.",
            )
        )
    if response_types & PIGMENT_RESPONSES:
        requested.append(
            (
                "pigmentation",
                sorted(response_types & PIGMENT_RESPONSES),
                "Requires a reference with a measurable colour signal.",
            )
        )
    if response_types & CHLOROPLAST_RESPONSES:
        requested.append(
            (
                "chloroplast",
                sorted(response_types & CHLOROPLAST_RESPONSES),
                "Requires a reference with a measurable colour signal.",
            )
        )
    if response_types & FLUORESCENCE_REDUCTION_RESPONSES:
        requested.append(
            (
                "fluorescence_attenuation",
                sorted(response_types & FLUORESCENCE_REDUCTION_RESPONSES),
                (
                    "Requires a reviewed dark-background fluorescence-like "
                    "reference with a measurable coloured signal."
                ),
            )
        )
    if response_types & CHLOROPHYLL_DECREASE_RESPONSES:
        requested.append(
            (
                "chlorophyll_signal_attenuation",
                sorted(response_types & CHLOROPHYLL_DECREASE_RESPONSES),
                (
                    "Requires a reviewed dark-background fluorescence-like "
                    "reference with a measurable coloured signal."
                ),
            )
        )
    if response_types & CHLOROPLAST_LOSS_RESPONSES:
        requested.append(
            (
                "chloroplast_signal_loss",
                sorted(response_types & CHLOROPLAST_LOSS_RESPONSES),
                (
                    "Requires a reviewed dark-background fluorescence-like "
                    "reference with a measurable coloured signal."
                ),
            )
        )

    decisions = []
    applied_rules = []

    for rule, supporting_types, requirement in requested:
        applied = True
        reason = "The reference passed the current conservative safety checks."

        if transform_policy is not None and not policy_allows_transform(
            transform_policy
        ):
            applied = False
            reason = policy_reason(transform_policy)
        elif reference_is_state_neutral:
            applied = False
            reason = (
                "The reference is state-neutral and cannot justify a "
                "condition-specific transformation."
            )
        elif reference_profile.get("composite_layout_risk", False):
            applied = False
            reason = (
                "The reference has a montage or annotation-layout risk. "
                "Without an organism mask, transforming it could alter labels, "
                "scale bars, or unrelated panels."
            )
        elif (
            rule in {"pigmentation", "chloroplast"}
            and not reference_profile.get("color_signal_supported", False)
        ):
            applied = False
            reason = (
                "The reference does not contain enough colour signal for this "
                "colour-dependent rule."
            )
        elif (
            rule in {
                "fluorescence_attenuation",
                "chlorophyll_signal_attenuation",
                "chloroplast_signal_loss",
            }
            and not reference_profile.get("fluorescence_like_supported", False)
        ):
            applied = False
            reason = (
                "The reference does not pass the conservative fluorescence-"
                "like modality check (dark field plus a measurable saturated "
                "signal)."
            )

        if applied:
            applied_rules.append(rule)

        decisions.append(
            {
                "rule": rule,
                "response_types": supporting_types,
                "applied": applied,
                "reason": reason,
                "modality_requirement": requirement,
                "transformation_scope": (
                    "whole reference image"
                    if applied
                    else "not applied"
                ),
                "organism_mask_available": bool(
                    reference_profile.get("organism_mask_available", False)
                ),
            }
        )

    return {
        "applied_rules": applied_rules,
        "decisions": decisions,
    }


def calculate_difference_metrics(reference_image, simulated_image):
    """Measure rendering differences; these are not biological effect sizes."""

    reference = reference_image.convert("RGB")
    simulated = simulated_image.convert("RGB")

    if simulated.size != reference.size:
        simulated = simulated.resize(
            reference.size,
            Image.Resampling.BILINEAR,
        )

    difference = ImageChops.difference(reference, simulated)
    grayscale = ImageOps.grayscale(difference)
    histogram = grayscale.histogram()
    total_pixels = max(1, sum(histogram))
    absolute_total = sum(
        level * count
        for level, count in enumerate(histogram)
    )
    changed_pixels = sum(histogram[3:])

    return {
        "mean_absolute_difference": round(
            absolute_total / (total_pixels * 255.0),
            6,
        ),
        "changed_pixel_fraction": round(
            changed_pixels / total_pixels,
            6,
        ),
        "changed_pixel_threshold": "grayscale difference > 2/255",
    }


def save_difference_map(reference_image, simulated_image, output_path):
    """Save an amplified rendering-difference map for UI inspection."""

    reference = reference_image.convert("RGB")
    simulated = simulated_image.convert("RGB")
    if simulated.size != reference.size:
        simulated = simulated.resize(
            reference.size,
            Image.Resampling.BILINEAR,
        )

    difference = ImageChops.difference(reference, simulated)
    grayscale = ImageOps.grayscale(difference)
    amplified = grayscale.point(
        lambda value: min(255, value * 8)
    )
    heatmap = ImageOps.colorize(
        amplified,
        black=(0, 0, 0),
        white=(255, 90, 0),
    )
    heatmap.save(output_path, format="PNG")
    return str(output_path)


def estimate_stress_intensity(
    biological_state,
    predicted_responses
):
    """
    Return a conservative 0-1 rendering intensity.

    IMPORTANT:
    The intensity is now based only on responses with an explicit
    static-image transformation rule. A stressed state by itself no
    longer causes generic darkening, blur or geometric changes.
    """

    if (
        clean(
            biological_state
        ).lower()
        !=
        "stressed"
    ):

        return 0.0

    classifications = classify_predicted_responses(
        predicted_responses
    )

    transformable = classifications[
        "transformable"
    ]

    if not transformable:

        return 0.0

    evidence_scores = []

    for response in (
        predicted_responses
        or []
    ):

        response_type = clean(
            response.get(
                "response_type"
            )
        )

        if (
            response_type
            not in
            STATIC_TRANSFORM_RESPONSES
        ):

            continue

        score = response.get(
            "evidence_score"
        )

        try:

            if score is not None:

                evidence_scores.append(
                    float(
                        score
                    )
                )

        except (
            TypeError,
            ValueError
        ):

            pass

    if evidence_scores:

        evidence_component = (
            sum(
                evidence_scores
            )
            /
            len(
                evidence_scores
            )
        )

    else:

        evidence_component = 0.5

    intensity = (
        0.18
        +
        0.32
        *
        max(
            0.0,
            min(
                evidence_component,
                1.0
            )
        )
    )

    intensity += min(
        0.06
        *
        max(
            0,
            len(
                transformable
            )
            -
            1
        ),
        0.12
    )

    return min(
        intensity,
        0.62
    )


# ============================================================
# GENERIC STRESS TRANSFORM
# ============================================================

def apply_generic_stress_transform(
    image,
    intensity,
    seed_text,
):
    """
    Preserve the reference image.

    Generic stress styling has intentionally been removed. A predicted
    stressed state alone does not justify brightness, contrast, blur,
    colour or position changes.
    """

    return image.convert(
        "RGB"
    ).copy()


# ============================================================
# RESPONSE-SPECIFIC TRANSFORMS
# ============================================================

def apply_structural_adjustment(
    image,
    intensity,
    seed_text,
):
    """
    Very mild structural cue for explicitly structural morphology
    responses. This does not draw cracks or invent fragments.
    """

    if intensity <= 0:

        return image.copy()

    width, height = image.size

    fraction = deterministic_fraction(
        seed_text
        +
        "_structural"
    )

    signed = (
        fraction
        -
        0.5
    ) * 2.0

    shear = (
        signed
        *
        0.018
        *
        intensity
    )

    x_shift = (
        -shear
        *
        height
        /
        2
    )

    output = image.transform(
        image.size,
        Image.Transform.AFFINE,
        (
            1,
            shear,
            x_shift,
            0,
            1,
            0,
        ),
        resample=Image.Resampling.BILINEAR,
    )

    output = output.filter(
        ImageFilter.GaussianBlur(
            radius=(
                0.05
                +
                0.12
                *
                intensity
            )
        )
    )

    return output


def apply_pigment_adjustment(
    image,
    intensity,
):
    """
    Mild saturation cue for explicit pigmentation_change only.
    """

    return (
        ImageEnhance
        .Color(
            image
        )
        .enhance(
            max(
                0.78,
                1.0
                -
                0.20
                *
                intensity
            )
        )
    )


def apply_chloroplast_adjustment(
    image,
    intensity,
):
    """
    Conservative saturation cue for explicit chloroplast_damage.
    No chloroplast geometry is invented.
    """

    return (
        ImageEnhance
        .Color(
            image
        )
        .enhance(
            max(
                0.82,
                1.0
                -
                0.16
                *
                intensity
            )
        )
    )


def apply_fluorescence_attenuation(
    image,
    intensity,
):
    """Slightly dim an explicitly reduced fluorescence signal.

    The operation is deliberately global because no organism/channel mask is
    available. It is therefore allowed only for separately reviewed,
    fluorescence-like crops and must not be read as a quantitative intensity
    prediction.
    """

    output = ImageEnhance.Brightness(image).enhance(
        max(0.82, 1.0 - 0.18 * intensity)
    )
    return ImageEnhance.Color(output).enhance(
        max(0.88, 1.0 - 0.10 * intensity)
    )


def apply_chloroplast_signal_loss(
    image,
    intensity,
):
    """Apply a restrained signal-loss cue for chloroplast_absent.

    This does not erase, segment or redraw a chloroplast. It only reduces the
    visible coloured signal in an approved fluorescence-like crop.
    """

    output = ImageEnhance.Brightness(image).enhance(
        max(0.78, 1.0 - 0.22 * intensity)
    )
    return ImageEnhance.Color(output).enhance(
        max(0.74, 1.0 - 0.24 * intensity)
    )


def apply_response_specific_adjustments(
    image,
    predicted_responses,
    intensity,
    seed_text="",
    applied_rule_categories=None,
):
    """
    Apply only explicitly supported static-image rules.
    """

    if (
        not predicted_responses
        or
        intensity <= 0
    ):

        return image.convert(
            "RGB"
        ).copy()

    output = image.convert(
        "RGB"
    )

    response_types = (
        response_types_from_predictions(
            predicted_responses
        )
    )

    allowed_rules = set(
        applied_rule_categories
        if applied_rule_categories is not None
        else {
            "structural",
            "pigmentation",
            "chloroplast",
            "fluorescence_attenuation",
            "chlorophyll_signal_attenuation",
            "chloroplast_signal_loss",
        }
    )

    if (
        response_types
        &
        STRUCTURAL_RESPONSES
        and
        "structural" in allowed_rules
    ):

        output = (
            apply_structural_adjustment(
                output,
                intensity,
                seed_text
            )
        )

    if (
        response_types
        &
        PIGMENT_RESPONSES
        and
        "pigmentation" in allowed_rules
    ):

        output = (
            apply_pigment_adjustment(
                output,
                intensity
            )
        )

    if (
        response_types
        &
        CHLOROPLAST_RESPONSES
        and
        "chloroplast" in allowed_rules
    ):

        output = (
            apply_chloroplast_adjustment(
                output,
                intensity
            )
        )

    if (
        response_types
        &
        FLUORESCENCE_REDUCTION_RESPONSES
        and
        "fluorescence_attenuation" in allowed_rules
    ):

        output = apply_fluorescence_attenuation(
            output,
            intensity,
        )

    if (
        response_types
        &
        CHLOROPHYLL_DECREASE_RESPONSES
        and
        "chlorophyll_signal_attenuation" in allowed_rules
    ):

        output = apply_fluorescence_attenuation(
            output,
            intensity,
        )

    if (
        response_types
        &
        CHLOROPLAST_LOSS_RESPONSES
        and
        "chloroplast_signal_loss" in allowed_rules
    ):

        output = apply_chloroplast_signal_loss(
            output,
            intensity,
        )

    return output


# ============================================================
# SIMULATE
# ============================================================

def simulate_visual_appearance(
    species,
    biological_state,
    predicted_responses=None,
    output_name=None,
    preferred_reference_record_id=None,
):

    if predictions_indicate_absence(predicted_responses):

        response_types = response_types_from_predictions(
            predicted_responses
        )

        return {
            "species": species,
            "biological_state": biological_state,
            "organism_absent": True,
            "reference_is_state_neutral": False,
            "reference_record_id": None,
            "reference_image_path": None,
            "reference_response_match": 0.0,
            "reference_match_method": "absence_no_visual_applicable",
            "reference_observation_id": None,
            "reference_supported_by_current_evidence": False,
            "transformation_intensity": 0.0,
            "visual_change_supported": False,
            "visual_response_roles": classify_visual_response_roles(
                predicted_responses
            ),
            "transformable_response_types": [],
            "applied_transform_response_types": [],
            "non_static_response_types": [],
            "unknown_visual_response_types": sorted(response_types),
            "applied_response_rules": [],
            "visual_rule_decisions": [],
            "reference_visual_profile": None,
            "transformation_scope": "not applicable",
            "simulated_image_path": None,
            "difference_image_path": None,
            "difference_metrics": None,
            "note": (
                "The evidence-backed response indicates that the organism "
                "was not observed in the supporting treatment. A static "
                "organism image is not a condition-specific visual "
                "prediction for absence, and no transformation is generated. "
                "This absence result is not being reinterpreted as mortality."
            ),
        }

    requested_v1_reference_record_id = clean(
        preferred_reference_record_id
    )

    reference_row, reference_match = (
        select_reference(
            species=species,
            biological_state=biological_state,
            predicted_responses=predicted_responses,
            preferred_reference_record_id=(
                preferred_reference_record_id
            ),
        )
    )

    selected_reference_record_id = clean(
        reference_row.get("record_id")
    )
    reference_differs_from_v1 = bool(
        requested_v1_reference_record_id
        and selected_reference_record_id
        != requested_v1_reference_record_id
    )

    full_reference_path = resolve_project_image_path(
        reference_row["image_path"]
    )

    if full_reference_path is None or not full_reference_path.exists():

        raise FileNotFoundError(
            f"Reference image not found:\n"
            f"{full_reference_path}"
        )

    transform_policy = reference_policy(reference_row["record_id"])
    approved_transform_path = resolve_transform_source(
        full_reference_path,
        transform_policy,
    )
    if policy_allows_transform(transform_policy) and approved_transform_path is None:
        transform_policy = dict(transform_policy)
        transform_policy["transform_eligible"] = False
        transform_policy["review_notes"] = (
            "The approved crop is missing or unreadable, so transformation "
            "was withheld. Re-run the visual-reference audit/crop workflow."
        )

    reference_path = approved_transform_path or full_reference_path

    source_image = (
        Image.open(
            reference_path
        )
        .convert(
            "RGB"
        )
    )

    reference_profile = analyse_reference_image(
        source_image
    )

    classifications = (
        classify_predicted_responses(
            predicted_responses
        )
    )

    visual_response_roles = classify_visual_response_roles(
        predicted_responses
    )

    visual_rule_predictions = list(predicted_responses or [])
    fallback_supported_static_types = set()

    if reference_differs_from_v1:
        _, _, fallback_value_matches = _reference_agreement(
            reference_row,
            predicted_responses,
        )
        fallback_supported_static_types = (
            fallback_value_matches & STATIC_TRANSFORM_RESPONSES
        )
        visual_rule_predictions = [
            prediction
            for prediction in predicted_responses or []
            if clean(prediction.get("response_type"))
            in fallback_supported_static_types
        ]

    intensity = (
        estimate_stress_intensity(
            biological_state=biological_state,
            predicted_responses=visual_rule_predictions,
        )
    )

    state_text = (
        clean(
            biological_state
        )
        or
        "state_neutral"
    )

    reference_is_state_neutral = (
        clean(
            reference_row.get(
                "biological_state"
            )
        ).lower()
        in
        {
            "state_neutral",
            "unresolved",
            "unknown",
        }
    )

    seed_text = (
        species
        +
        state_text
        +
        build_response_query(
            predicted_responses
        )
    )

    rule_evaluation = evaluate_visual_rule_applicability(
        predicted_responses=visual_rule_predictions,
        reference_profile=reference_profile,
        reference_is_state_neutral=reference_is_state_neutral,
        transform_policy=transform_policy,
    )

    applied_response_rules = rule_evaluation[
        "applied_rules"
    ]

    visual_rule_decisions = rule_evaluation[
        "decisions"
    ]

    if clean(transform_policy.get("review_status")).lower() == "verified_crop":
        for decision in visual_rule_decisions:
            if decision.get("applied"):
                decision["transformation_scope"] = (
                    "reviewed organism-region crop"
                )

    visual_change_supported = (
        bool(applied_response_rules)
        and intensity > 0
        and not reference_is_state_neutral
    )

    if visual_change_supported:
        simulated = apply_generic_stress_transform(
            image=source_image,
            intensity=intensity,
            seed_text=seed_text,
        )
        simulated = apply_response_specific_adjustments(
            image=simulated,
            predicted_responses=visual_rule_predictions,
            intensity=intensity,
            seed_text=seed_text,
            applied_rule_categories=applied_response_rules,
        )
    else:
        # Keep the in-memory output pixel-equivalent to the selected reference
        # whenever no supported transformation is being claimed.
        simulated = source_image.copy()
        applied_response_rules = []

    applied_transform_response_types = sorted(
        {
            clean(response_type)
            for decision in visual_rule_decisions
            if decision.get("applied")
            for response_type in decision.get("response_types", [])
            if clean(response_type)
        }
    )

    if not output_name:

        safe_species = (
            species.replace(
                " ",
                "_"
            )
        )

        output_name = (
            f"{safe_species}_"
            f"{state_text}_"
            f"v2_simulated.jpg"
        )

    output_path = (
        OUTPUT_DIR
        /
        output_name
    )

    simulated.save(
        output_path,
        format="JPEG",
        quality=95,
    )

    difference_image_path = None
    difference_metrics = None

    if visual_change_supported:
        difference_metrics = calculate_difference_metrics(
            source_image,
            simulated,
        )
        difference_path = (
            OUTPUT_DIR
            / f"{Path(output_name).stem}_difference.png"
        )
        difference_image_path = save_difference_map(
            source_image,
            simulated,
            difference_path,
        )

    if visual_change_supported:

        note = (
            "Illustrative evidence-aware V2 simulation. "
            "Only responses with an explicit static-image rule were "
            "used to alter the reference. This is not a scientifically "
            "validated exact morphology prediction."
        )

    elif reference_is_state_neutral:

        note = (
            "The selected reference is state-neutral, so it is preserved "
            "without transformation even if a predicted response has a "
            "static-image rule. A reference without a supported biological-"
            "state label cannot justify a condition-specific visual change."
        )

    elif visual_rule_decisions:

        blocked_reasons = [
            decision["reason"]
            for decision in visual_rule_decisions
            if not decision["applied"]
        ]

        note = (
            "A static-image rule exists, but it was not applied because the "
            "selected reference did not pass the current visual-safety checks. "
            + " ".join(blocked_reasons)
        )

    else:

        note = (
            "The current predicted responses and evidence do not justify a "
            "static-image transformation. The output therefore preserves "
            "the selected reference rather than inventing a visible change."
        )

    return {
        "species":
            species,

        "organism_absent":
            False,

        "biological_state":
            biological_state,

        "reference_is_state_neutral":
            reference_is_state_neutral,

        "reference_record_id":
            reference_row[
                "record_id"
            ],

        "v1_reference_record_id":
            requested_v1_reference_record_id or None,

        "reference_differs_from_v1":
            reference_differs_from_v1,

        "reference_selection_method":
            (
                "transform_safe_response_fallback"
                if reference_differs_from_v1
                else "v1_preferred_reference"
                if requested_v1_reference_record_id
                else "response_ranked_reference"
            ),

        "reference_image_path":
            str(
                reference_path
            ),

        "full_reference_image_path":
            str(
                full_reference_path
            ),

        "reference_transform_policy":
            transform_policy,

        "reference_response_match":
            round(
                reference_match,
                4
            ),

        "reference_match_method":
            "structured_value_aware_response_agreement_with_evidence_tiebreak",

        "reference_observation_id":
            candidate_observation_id(
                reference_row
            ),

        "reference_supported_by_current_evidence":
            (
                candidate_observation_id(
                    reference_row
                )
                in
                supporting_observation_ids(
                    predicted_responses
                )
            ),

        "transformation_intensity":
            round(
                intensity if visual_change_supported else 0.0,
                4
            ),

        "requested_transformation_intensity":
            round(
                intensity,
                4
            ),

        "visual_change_supported":
            visual_change_supported,

        "visual_response_roles":
            visual_response_roles,

        "transformable_response_types":
            classifications[
                "transformable"
            ],

        "applied_transform_response_types":
            applied_transform_response_types,

        "non_static_response_types":
            classifications[
                "non_static"
            ],

        "unknown_visual_response_types":
            classifications[
                "unknown"
            ],

        "applied_response_rules":
            applied_response_rules,

        "visual_rule_decisions":
            visual_rule_decisions,

        "reference_visual_profile":
            reference_profile,

        "transformation_scope":
            (
                "reviewed organism-region crop"
                if (
                    visual_change_supported
                    and clean(transform_policy.get("review_status")).lower()
                    == "verified_crop"
                )
                else "whole reference image"
                if visual_change_supported
                else "not applied"
            ),

        "simulated_image_path":
            str(
                output_path
            ),

        "difference_image_path":
            difference_image_path,

        "difference_metrics":
            difference_metrics,

        "note":
            note,
    }



# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=" * 78
    )

    print(
        "V2 RESPONSE-SPECIFIC VISUAL SIMULATOR"
    )

    print(
        "=" * 78
    )

    example_responses = [
        {
            "response_type": "frustule_breakage",
            "response_value": "present",
            "evidence_score": 1.0,
        }
    ]

    # example_responses = [
    #     {
    #         "response_type":
    #             "cell_death",

    #         "response_value":
    #             "97.6 percent",
    #     },
    #     {
    #         "response_type":
    #             "reduced_viability",

    #         "response_value":
    #             "present",
    #     },
    #     {
    #         "response_type":
    #             "pore_percolation",

    #         "response_value":
    #             "present",
    #     },
    # ]

    result = (
        simulate_visual_appearance(
            species="Nitzschia palea",
            biological_state="stressed",
            predicted_responses=example_responses,
            output_name=(
                "nitzschia_frustule_breakage_test.jpg"
            ),
        )
    )

    # result = (
    #     simulate_visual_appearance(
    #         species="Nitzschia palea",
    #         biological_state="stressed",
    #         predicted_responses=example_responses,
    #         output_name=(
    #             "nitzschia_petrol_example_v2.jpg"
    #         ),
    #     )
    # )

    print()

    for key, value in (
        result.items()
    ):

        print(
            f"{key}: {value}"
        )

    print()

    print(
        "RESULT:"
    )

    print(
        "A response-specific illustrative simulated image "
        "was created."
    )

    print()

    print(
        "Next step: rerun the existing unseen-image "
        "evaluation and compare the mean similarity change."
    )
