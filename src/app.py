import sys
import hashlib
from pathlib import Path
import streamlit as st


# ============================================================
# MAKE src IMPORTS WORK
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(CURRENT_DIR)
    )


from world_model import (
    predict_response,
    infer_water_quality
)

from world_state import WorldState
from simulator import SimpleWorldSimulator
from v2_visual_simulator import simulate_visual_appearance
from response_display import (
    describe_response_value,
    format_readable_value,
)
from model_information import render_model_information
from response_narrative import occurrence_narrative

from config import DATABASE_FILE

from database_queries import (
    get_species,
    get_water_factors,
    get_response_types,
    get_water_factor_details,
    get_response_details,
    get_media_for_observation,
    get_general_media
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Microorganism World Model",
    page_icon="🔬",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = PROJECT_ROOT / "WSCA-Logo.jpg"

st.image(str(LOGO_PATH), width=700)

st.info(
    "This project was incubated by Water Sensitive Cities Australia "
    "through the Lab to Land initiative."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def humanize(text):
    """
    Convert database-style names into readable labels.

    Example:
        cell_death
            ->
        Cell Death
    """

    if text is None:
        return ""

    return (
        str(text)
        .replace("_", " ")
        .strip()
        .title()
    )


def humanize_value(value):
    """
    Convert stored categorical values into readable display text
    without changing the value passed to the model/database.

    Examples:
        decrease_or_no_change -> decrease or no change
        slightly_elevated     -> slightly elevated
        35.0-36.0 per_10_um   -> 35.0-36.0 per 10 um

    Numeric values and ordinary text are otherwise preserved.
    """

    if value is None:
        return ""

    return (
        str(value)
        .replace("_", " ")
        .strip()
    )


def show_wrapped_value(
    label,
    display_value,
    description=None,
    raw_value=None,
):
    """Render a long value without Streamlit metric truncation."""

    st.markdown(f"**{label}**")
    st.markdown(f"#### {display_value or 'Not specified'}")

    if description:
        st.caption(description)

    readable_raw = format_readable_value(raw_value)
    if (
        readable_raw
        and readable_raw != str(display_value or "").strip()
    ):
        st.caption(f"Evidence value: {readable_raw}")


def show_response_value(response_type, value):
    """Render a response value with a conservative plain-language gloss."""

    presentation = describe_response_value(
        response_type,
        value,
    )
    show_wrapped_value(
        "Response value",
        presentation["display_value"],
        description=presentation["plain_language"],
        raw_value=presentation["raw_value"],
    )
    return presentation


def organism_absence_is_predicted(
    organism_state
):
    """Detect only an explicit whole-organism presence=absent response."""

    response = (organism_state or {}).get("presence")

    if isinstance(response, dict):
        value = response.get("value")
    else:
        value = response

    normalised = (
        str(value or "")
        .strip()
        .lower()
        .replace("_", " ")
    )

    return normalised in {
        "absent",
        "not detected",
        "not found",
        "not observed",
        "undetected",
    }


def evidence_label(score):
    """
    Convert an evidence score into a qualitative label.

    IMPORTANT:
    This is not a biological probability.
    """

    if score >= 0.9:
        return "Strong"

    if score >= 0.6:
        return "Moderate"

    if score >= 0.4:
        return "Limited"

    return "Weak"


def format_known_concentration(
    minimum,
    maximum,
    unit
):
    """
    Format concentration information.

    Examples:
        5 - 15 uL/L
        > 15 uL/L
        100 ug/L
    """

    if minimum is None:
        return None

    unit_text = unit or ""

    if maximum is None:

        return (
            f"> {minimum} "
            f"{unit_text}"
        ).strip()

    if float(minimum) == float(maximum):

        return (
            f"{minimum} "
            f"{unit_text}"
        ).strip()

    return (
        f"{minimum} - {maximum} "
        f"{unit_text}"
    ).strip()


# ============================================================
# WATER-FACTOR DATABASE INFORMATION
# ============================================================

def show_water_factor_database_info(
    species,
    factor
):
    """
    Display known values, concentrations, and units for a
    selected water factor.
    """

    details = get_water_factor_details(
        species,
        factor
    )

    if not details:
        return

    known_text_values = sorted(
        {
            str(
                item["value_text"]
            )
            for item in details
            if item["value_text"] is not None
        }
    )

    known_units = sorted(
        {
            str(unit)

            for item in details

            for unit in [
                item["factor_unit"],
                item["concentration_unit"]
            ]

            if unit is not None
        }
    )

    concentration_records = []

    for item in details:

        minimum = item[
            "concentration_min"
        ]

        maximum = item[
            "concentration_max"
        ]

        unit = item[
            "concentration_unit"
        ]

        concentration = (
            format_known_concentration(
                minimum,
                maximum,
                unit
            )
        )

        if concentration is not None:

            concentration_records.append(
                (
                    float(minimum),
                    concentration
                )
            )

    # Remove duplicate concentration labels while retaining
    # numeric sorting.
    unique_concentrations = {}

    for minimum, text in concentration_records:

        if text not in unique_concentrations:

            unique_concentrations[
                text
            ] = minimum

    known_concentrations = sorted(
        unique_concentrations.keys(),
        key=lambda text:
            unique_concentrations[text]
    )

    if not (
        known_text_values
        or
        known_units
        or
        known_concentrations
    ):
        return

    st.caption(
        "Known values in the current evidence database"
    )

    if known_text_values:

        st.write(
            "**Text values:** "
            +
            ", ".join(
                humanize_value(value)
                for value in known_text_values
            )
        )

    if known_concentrations:

        st.write(
            "**Concentrations/ranges:** "
            +
            ", ".join(
                known_concentrations
            )
        )

    if known_units:

        st.write(
            "**Units:** "
            +
            ", ".join(
                known_units
            )
        )


def get_known_water_factor_value_options(
    species,
    factor
):
    """
    Return known water-factor values from the evidence database.

    Each option contains:
        label       - readable text shown in the Streamlit dropdown
        model_value - value passed to predict_response()

    Concentration ranges are converted to a representative numeric value
    inside the stored range so the existing forward-model concentration
    matcher can evaluate the relevant case-specific evidence.
    """

    details = get_water_factor_details(
        species,
        factor
    )

    if not details:
        return []

    options = []
    seen = set()

    for item in details:

        # --------------------------------------------------------
        # TEXT / CATEGORICAL VALUE
        # --------------------------------------------------------

        if item["value_text"] is not None:

            value_text = str(
                item["value_text"]
            ).strip()

            if value_text:

                label = value_text

                key = (
                    "text",
                    value_text.lower()
                )

                if key not in seen:

                    options.append(
                        {
                            "label": label,
                            "model_value": value_text,
                            "kind": "text",
                        }
                    )

                    seen.add(
                        key
                    )

        # --------------------------------------------------------
        # KNOWN CONCENTRATION / RANGE
        # --------------------------------------------------------

        minimum = item[
            "concentration_min"
        ]

        maximum = item[
            "concentration_max"
        ]

        unit = item[
            "concentration_unit"
        ]

        if minimum is None:
            continue

        minimum = float(
            minimum
        )

        if maximum is not None:
            maximum = float(
                maximum
            )

        label = format_known_concentration(
            minimum,
            maximum,
            unit
        )

        if not label:
            continue

        # The existing forward model accepts one numeric concentration.
        # For a stored range, use its midpoint; for an open-ended
        # "> minimum" record, use a value infinitesimally above the
        # threshold so it belongs to the stored interval.
        if maximum is None:

            epsilon = max(
                abs(minimum) * 1e-6,
                1e-6
            )

            representative_value = (
                minimum
                +
                epsilon
            )

        elif minimum == maximum:

            representative_value = minimum

        else:

            representative_value = (
                minimum
                +
                maximum
            ) / 2.0

        key = (
            "concentration",
            minimum,
            maximum,
            str(unit or "")
        )

        if key not in seen:

            options.append(
                {
                    "label": label,
                    "model_value": {
                        "value": representative_value,
                        "unit": (
                            str(unit).strip()
                            if unit
                            else None
                        ),
                    },
                    "kind": "concentration",
                }
            )

            seen.add(
                key
            )

    return options


# ============================================================
# RESPONSE DATABASE INFORMATION
# ============================================================

def show_response_database_info(
    species,
    response
):
    """
    Display known response values currently stored for the
    selected microorganism.
    """

    details = get_response_details(
        species,
        response
    )

    if not details:
        return

    known_values = []

    for item in details:

        # ----------------------------------------------------
        # TEXT VALUE
        # ----------------------------------------------------

        if item["value_text"] is not None:

            known_values.append(
                str(
                    item["value_text"]
                )
            )

        # ----------------------------------------------------
        # NUMERIC VALUE
        # ----------------------------------------------------

        elif item["value_numeric"] is not None:

            text = str(
                item[
                    "value_numeric"
                ]
            )

            if item["unit"]:

                text += (
                    f" {item['unit']}"
                )

            known_values.append(
                text
            )

        # ----------------------------------------------------
        # NUMERIC RANGE
        # ----------------------------------------------------

        elif (
            item["value_min"] is not None
            and
            item["value_max"] is not None
        ):

            text = (
                f"{item['value_min']}"
                f" - "
                f"{item['value_max']}"
            )

            if item["unit"]:

                text += (
                    f" {item['unit']}"
                )

            known_values.append(
                text
            )

    known_values = sorted(
        set(
            known_values
        )
    )

    if not known_values:
        return

    st.caption(
        "Known values in the current evidence database"
    )

    st.write(
        ", ".join(
            humanize_value(value)
            for value in known_values
        )
    )


def get_known_response_value_options(
    species,
    response
):
    """
    Return de-duplicated, human-readable response values stored
    in the evidence database for the selected species/response.
    """

    details = get_response_details(
        species,
        response
    )

    if not details:
        return []

    values = []

    for item in details:

        if item["value_text"] is not None:
            values.append(
                str(item["value_text"]).strip()
            )

        elif item["value_numeric"] is not None:
            value_text = str(item["value_numeric"])

            if item["unit"]:
                value_text += f" {item['unit']}"

            values.append(value_text.strip())

        elif (
            item["value_min"] is not None
            and
            item["value_max"] is not None
        ):
            value_text = (
                f"{item['value_min']}"
                f"-"
                f"{item['value_max']}"
            )

            if item["unit"]:
                value_text += f" {item['unit']}"

            values.append(value_text.strip())

    unique_values = []

    for value in values:
        if value and value not in unique_values:
            unique_values.append(value)

    return unique_values


# ============================================================
# OBSERVATION-SPECIFIC MEDIA
# ============================================================

def show_observation_media(
    observation_id,
    context_key
):
    """
    Display media tied directly to one spreadsheet observation.

    The context_key makes Streamlit widget keys unique when
    the same observation appears under multiple predictions
    or evidence records.
    """

    media_records = get_media_for_observation(
        observation_id
    )

    if not media_records:

        st.caption(
            "No observation-specific media is available."
        )

        return

    images = [
        media
        for media in media_records
        if media["media_type"] == "image"
    ]

    videos = [
        media
        for media in media_records
        if media["media_type"] == "video"
    ]

    # ========================================================
    # IMAGES
    # ========================================================

    if images:

        st.markdown(
            "**Supporting microscopy image(s)**"
        )

        # Several database media records can point to separate local files
        # containing exactly the same image bytes. Keep every database record
        # intact, but render each unique image only once in the UI.
        grouped_images = {}

        for image in images:

            local_path = image[
                "local_path"
            ]

            image_key = None

            # Prefer a content hash when the local file exists. This catches
            # duplicate images even when they have different filenames/paths.
            if (
                local_path
                and
                Path(
                    local_path
                ).exists()
            ):

                try:

                    digest = hashlib.sha256()

                    with open(
                        local_path,
                        "rb"
                    ) as image_file:

                        for chunk in iter(
                            lambda:
                                image_file.read(
                                    1024 * 1024
                                ),
                            b""
                        ):

                            digest.update(
                                chunk
                            )

                    image_key = (
                        "sha256",
                        digest.hexdigest()
                    )

                except Exception:

                    image_key = None

            # If hashing is unavailable, fall back to the normalized path.
            if (
                image_key is None
                and local_path
            ):

                try:

                    image_key = (
                        "local_path",
                        str(
                            Path(
                                local_path
                            ).resolve()
                        )
                    )

                except Exception:

                    image_key = (
                        "local_path",
                        str(
                            local_path
                        ).strip()
                    )

            # If no local file is available, use the source URL when possible.
            if (
                image_key is None
                and image["url"]
            ):

                image_key = (
                    "url",
                    str(
                        image["url"]
                    ).strip()
                )

            # With no usable path/hash/URL, retain the record separately rather
            # than incorrectly assuming that it duplicates another image.
            if image_key is None:

                image_key = (
                    "media_id",
                    str(
                        image["media_id"]
                    )
                )

            grouped_images.setdefault(
                image_key,
                []
            ).append(
                image
            )

        unique_image_groups = list(
            grouped_images.values()
        )

        for image_index, image_group in enumerate(
            unique_image_groups,
            start=1
        ):

            image = image_group[0]

            local_path = image[
                "local_path"
            ]

            media_ids = [
                str(
                    item["media_id"]
                )
                for item in image_group
            ]

            caption = (
                f"Image {image_index}: "
                f"{media_ids[0]}"
            )

            if len(
                image_group
            ) > 1:

                caption += (
                    f" (same image referenced by "
                    f"{len(image_group)} media records)"
                )

            if (
                local_path
                and
                Path(local_path).exists()
            ):

                st.image(
                    local_path,
                    caption=caption,
                    use_container_width=True
                )

            else:

                st.warning(
                    "The local microscopy image file "
                    "could not be found."
                )

                if local_path:

                    st.caption(
                        local_path
                    )

            # Preserve provenance even when duplicate visual renders are
            # collapsed into one image.
            if len(
                image_group
            ) > 1:

                st.caption(
                    "Media records represented: "
                    +
                    ", ".join(
                        f"`{media_id}`"
                        for media_id in media_ids
                    )
                )

            # Show each distinct source URL once for the grouped image.
            source_urls = []

            for grouped_image in image_group:

                source_url = grouped_image[
                    "url"
                ]

                if (
                    source_url
                    and source_url not in source_urls
                ):

                    source_urls.append(
                        source_url
                    )

            for source_index, source_url in enumerate(
                source_urls,
                start=1
            ):

                button_label = (
                    "Open image/source reference"
                    if len(source_urls) == 1
                    else
                    f"Open image/source reference {source_index}"
                )

                st.link_button(
                    button_label,
                    source_url,
                    key=(
                        f"image_source_"
                        f"{context_key}_"
                        f"{observation_id}_"
                        f"{image_index}_"
                        f"{source_index}"
                    )
                )

            # Show each distinct note once.
            notes = []

            for grouped_image in image_group:

                note = grouped_image[
                    "notes"
                ]

                if (
                    note
                    and note not in notes
                ):

                    notes.append(
                        note
                    )

            for note in notes:

                st.caption(
                    note
                )

    # ========================================================
    # OBSERVATION-SPECIFIC VIDEOS
    # ========================================================

    if videos:

        st.markdown(
            "**Supporting video(s)**"
        )

        for video_index, video in enumerate(
            videos,
            start=1
        ):

            st.markdown(
                f"**Video {video_index}**"
            )

            if video["url"]:

                try:

                    st.video(
                        video["url"]
                    )

                except Exception:

                    st.link_button(
                        "Open video",
                        video["url"],
                        key=(
                            f"observation_video_"
                            f"{context_key}_"
                            f"{observation_id}_"
                            f"{video['media_id']}_"
                            f"{video_index}"
                        )
                    )

            if video["notes"]:

                st.caption(
                    video["notes"]
                )


# ============================================================
# GENERAL REFERENCE VIDEOS
# ============================================================

def show_reference_videos(
    species
):
    """
    Display organism-level videos that are not tied to one
    specific evidence observation.

    These videos provide visual context only. They are not
    treated as direct evidence for an individual prediction.
    """

    reference_media = get_general_media(
        species
    )

    reference_videos = [
        media
        for media in reference_media
        if media["media_type"] == "video"
    ]

    if not reference_videos:
        return

    with st.expander(
        "Reference microorganism videos"
    ):

        st.info(
            """
            These videos are general reference media for the
            selected microorganism. They are **not direct
            evidence for a particular prediction or inferred
            water condition**.
            """
        )

        for index, video in enumerate(
            reference_videos,
            start=1
        ):

            condition = (
                humanize(
                    video["condition"]
                )
                if video["condition"]
                else
                "Reference"
            )

            st.markdown(
                f"### Video {index} — {condition}"
            )

            if video["url"]:

                try:

                    st.video(
                        video["url"]
                    )

                except Exception:

                    st.link_button(
                        "Open video",
                        video["url"],
                        key=(
                            f"general_video_"
                            f"{species}_"
                            f"{video['media_id']}_"
                            f"{index}"
                        )
                    )

            if video["notes"]:

                st.write(
                    video["notes"]
                )

            if index < len(
                reference_videos
            ):

                st.divider()



# ============================================================
# INTEGRATED WORLD-MODEL INTERPRETATION
# ============================================================

def format_simulation_water_context(
    water_state
):
    """
    Convert the selected water-quality state into concise readable text.
    """

    parts = []

    for factor, value in water_state.items():

        factor_name = humanize(
            factor
        )

        if isinstance(
            value,
            dict
        ):

            numeric_value = value.get(
                "value"
            )

            unit = value.get(
                "unit"
            )

            if numeric_value is not None:

                value_text = str(
                    numeric_value
                )

                if unit:

                    value_text += (
                        f" {unit}"
                    )

                parts.append(
                    f"{factor_name} at {value_text}"
                )

            else:

                parts.append(
                    factor_name
                )

        else:

            value_text = str(
                value
            ).strip()

            if (
                value_text
                and
                value_text.lower()
                not in {
                    "present",
                    "true",
                    "yes",
                    "1",
                }
            ):

                parts.append(
                    f"{factor_name} ({humanize_value(value_text)})"
                )

            else:

                parts.append(
                    factor_name
                )

    if not parts:

        return "the supplied water-quality conditions"

    if len(parts) == 1:

        return parts[0]

    if len(parts) == 2:

        return (
            f"{parts[0]} and {parts[1]}"
        )

    return (
        ", ".join(
            parts[:-1]
        )
        +
        f", and {parts[-1]}"
    )


def format_response_sentence(
    organism_state
):
    """
    Convert predicted microorganism responses into one concise clause.
    """

    if not organism_state:

        return ""

    parts = []

    for response_type, response in (
        organism_state.items()
    ):

        response_name = humanize(
            response_type
        ).lower()

        supported_values = response.get(
            "supported_values",
            []
        )

        response_values = []

        for supported in supported_values:

            supported_value = str(
                supported.get(
                    "value",
                    ""
                )
            ).strip()

            if (
                supported_value
                and
                supported_value not in response_values
            ):

                response_values.append(
                    supported_value
                )

        if not response_values:

            response_value = str(
                response.get(
                    "value",
                    ""
                )
            ).strip()

            if response_value:

                response_values.append(
                    response_value
                )

        if not response_values:

            continue

        if (
            len(response_values) == 1
            and
            response_values[0].lower() in {
                "present",
                "true",
                "yes",
            }
        ):

            parts.append(
                response_name
            )

        elif len(response_values) == 1:

            parts.append(
                f"{response_name} of "
                f"{humanize_value(response_values[0])}"
            )

        else:

            readable_values = [
                humanize_value(value)
                for value in response_values
            ]

            parts.append(
                f"{response_name} with evidence-supported values/ranges "
                f"{' and '.join(readable_values)}"
            )

    if not parts:

        return ""

    if len(parts) == 1:

        return parts[0]

    if len(parts) == 2:

        return (
            f"{parts[0]} and {parts[1]}"
        )

    return (
        ", ".join(
            parts[:-1]
        )
        +
        f", and {parts[-1]}"
    )


def collect_interpretation_visual_features(
    visual_state
):
    """
    Return a short de-duplicated list of readable visual features.
    """

    if (
        visual_state is None
        or
        not visual_state.visual_features
    ):

        return []

    features = []

    for feature_group in (
        visual_state.visual_features
    ):

        for feature in str(
            feature_group
        ).split(
            "|"
        ):

            feature = feature.strip()

            if not feature:
                continue

            readable = humanize(
                feature
            )

            if readable not in features:

                features.append(
                    readable
                )

    return features


def build_world_model_interpretation(
    simulation_result
):
    """
    Produce one concise, evidence-grounded paragraph joining:

        water condition
        -> overall biological state
        -> predicted biological responses
        -> closest visual analogue

    This is a presentation-layer summary only. It does not create
    new scientific labels or make any additional model prediction.
    """

    current_state = (
        simulation_result
        .current_state
    )

    next_state = (
        simulation_result
        .next_state
    )

    species = (
        current_state.species
    )

    water_context = (
        format_simulation_water_context(
            current_state.water_state
        )
    )

    predicted_condition = (
        simulation_result
        .predicted_condition
    )

    condition_source = (
        simulation_result
        .condition_source
    )

    condition_confidence = (
        simulation_result
        .condition_confidence
    )

    classifier_environment_terms = list(
        getattr(
            simulation_result,
            "classifier_environment_terms",
            [],
        )
        or []
    )

    occurrence_text, remaining_organism_state = occurrence_narrative(
        species,
        next_state.organism_state,
    )

    response_text = format_response_sentence(
        remaining_organism_state
    )

    visual_state = (
        simulation_result
        .visual_state
    )

    sentences = []

    organism_absent = organism_absence_is_predicted(
        next_state.organism_state
    )

    # --------------------------------------------------------
    # WATER CONDITION + OVERALL STATE
    # --------------------------------------------------------

    if predicted_condition:

        first_sentence = (
            f"Under {water_context}, "
            f"{species} is predicted to be in a "
            f"{predicted_condition.lower()} biological state"
        )

        if (
            condition_source
            ==
            "direct_response_evidence"
            and
            condition_confidence is not None
        ):

            first_sentence += (
                f", supported by direct response evidence "
                f"with an evidence score of "
                f"{condition_confidence:.3f}"
            )

        elif (
            condition_source
            ==
            "observation_condition_evidence"
            and
            condition_confidence is not None
        ):

            first_sentence += (
                f", supported by observation-condition evidence "
                f"with an evidence score of "
                f"{condition_confidence:.3f}"
            )

        elif (
            condition_source
            ==
            "qualitative_state_classifier"
            and
            condition_confidence is not None
        ):

            first_sentence += (
                f", with a classifier probability of "
                f"{condition_confidence:.1%}"
            )

        first_sentence += "."

        sentences.append(
            first_sentence
        )

    else:

        if (
            condition_source
            ==
            "qualitative_state_classifier_no_input_coverage"
        ):

            sentences.append(
                (
                    f"Under {water_context}, no overall healthy/stressed "
                    f"state is assigned for {species} because the "
                    "qualitative classifier does not recognise a supported "
                    "environmental factor term in the supplied input."
                )
            )

        else:

            sentences.append(
                (
                    f"Under {water_context}, the current evidence "
                    f"does not support a single overall healthy/stressed "
                    f"state prediction for {species}."
                )
            )

    # --------------------------------------------------------
    # PREDICTED RESPONSES
    # --------------------------------------------------------

    if occurrence_text:

        sentences.append(
            occurrence_text
        )

    if response_text:

        response_prefix = (
            "Other predicted biological responses include "
            if occurrence_text
            else "The predicted next state includes "
        )

        sentences.append(
            f"{response_prefix}{response_text}."
        )

    # --------------------------------------------------------
    # VISUAL ANALOGUE
    # --------------------------------------------------------

    if organism_absent:

        sentences.append(
            (
                "Because the evidence-backed response is organism absence, "
                "a condition-specific microscopy appearance is not "
                "applicable. Any separate species image should be treated "
                "only as an identification reference. Absence from the "
                "supporting observation is not being reinterpreted as "
                "mortality."
            )
        )

    elif (
        visual_state is not None
        and
        visual_state.has_image
    ):

        matching_responses = list(
            dict.fromkeys(
                visual_state.matching_response_types
                or []
            )
        )

        visual_features = (
            collect_interpretation_visual_features(
                visual_state
            )
        )

        state_neutral_reference = (
            visual_state.evidence_type
            ==
            "external_state_neutral_reference"
        )

        response_similarity = float(
            visual_state.response_similarity
            or
            0.0
        )

        matching_environment_factors = list(
            dict.fromkeys(
                visual_state.matching_environment_factors
                or []
            )
        )

        if state_neutral_reference:

            visual_sentence = (
                "The displayed microscopy reference matches the species, "
                "but its source does not establish a healthy/stressed state "
                "or matching environmental condition"
            )

        elif matching_responses and response_similarity > 0:

            readable_matches = [
                humanize(
                    item
                ).lower()
                for item in matching_responses
            ]

            if len(readable_matches) == 1:

                match_text = (
                    readable_matches[0]
                )

            elif len(readable_matches) == 2:

                match_text = (
                    f"{readable_matches[0]} "
                    f"and {readable_matches[1]}"
                )

            else:

                match_text = (
                    ", ".join(
                        readable_matches[:-1]
                    )
                    +
                    f", and {readable_matches[-1]}"
                )

            visual_sentence = (
                "The selected same-species, same-state microscopy reference "
                f"shares the predicted response evidence for {match_text}"
            )

            if not matching_environment_factors:

                visual_sentence += (
                    ", but it has no matching environmental context"
                )

        else:

            if matching_environment_factors:

                visual_sentence = (
                    "The selected microscopy reference matches the species "
                    "and biological-state label and has some environmental-"
                    "context similarity, but it does not share a supported "
                    "predicted biological response"
                )

            else:

                visual_sentence = (
                    "The selected microscopy reference matches the species "
                    "and biological-state label only; it does not share a "
                    "supported predicted response or matching environmental "
                    "context"
                )

        if visual_features:

            selected_features = (
                visual_features[:4]
            )

            feature_text = (
                ", ".join(
                    selected_features
                )
            )

            visual_sentence += (
                f". The reference record is annotated with {feature_text}"
            )

        visual_sentence += "."

        sentences.append(
            visual_sentence
        )

        if visual_features:

            sentences.append(
                (
                    "Those annotations describe the selected reference "
                    "record; they are not additional effects predicted for "
                    "the current water condition."
                )
            )

        sentences.append(
            (
                "The displayed image is an evidence-backed visual "
                "analogue from the current dataset, not an exact "
                "synthetic prediction of the organism's future appearance."
            )
        )

    elif visual_state is not None:

        sentences.append(
            (
                "No suitable same-species, same-state microscopy "
                "image is currently available, so no visual appearance "
                "is inferred beyond the supported biological responses."
            )
        )

    return " ".join(
        sentences
    )


# ============================================================
# SIMULATION BIOLOGICAL / VISUAL STATE
# ============================================================

def show_simulation_biological_visual_state(
    simulation_result
):
    """
    Display the overall biological condition and the evidence-backed
    visual analogue returned by SimpleWorldSimulator.

    Important distinctions:
    - direct_response_evidence means an explicitly adverse response
      from the evidence model determined the stressed state.
    - qualitative_state_classifier means the healthy/stressed ML model
      was used as a fallback because no direct adverse response was found.
    - the displayed microscopy image is currently an evidence-backed
      visual analogue, not a newly generated synthetic image.
    """

    # ========================================================
    # BIOLOGICAL CONDITION
    # ========================================================

    st.subheader(
        "Predicted biological condition"
    )

    predicted_condition = (
        simulation_result
        .predicted_condition
    )

    condition_confidence = (
        simulation_result
        .condition_confidence
    )

    condition_source = (
        simulation_result
        .condition_source
    )

    classifier_environment_terms = list(
        getattr(
            simulation_result,
            "classifier_environment_terms",
            [],
        )
        or []
    )

    organism_absent = organism_absence_is_predicted(
        simulation_result.next_state.organism_state
    )

    if predicted_condition:

        condition_col1, condition_col2 = (
            st.columns(
                2
            )
        )

        with condition_col1:

            st.metric(
                "Predicted state",
                humanize(
                    predicted_condition
                )
            )

        with condition_col2:

            if (
                condition_confidence
                is not None
            ):

                if (
                    condition_source
                    in {
                        "direct_response_evidence",
                        "observation_condition_evidence",
                    }
                ):

                    confidence_label = (
                        "Supporting evidence score"
                    )

                    confidence_value = (
                        f"{condition_confidence:.3f}"
                    )

                else:

                    confidence_label = (
                        "Classifier probability"
                    )

                    confidence_value = (
                        f"{condition_confidence:.1%}"
                    )

                st.metric(
                    confidence_label,
                    confidence_value
                )

        if (
            condition_source
            ==
            "direct_response_evidence"
        ):

            st.success(
                """
                The overall condition was determined from direct
                adverse-response evidence in the current simulation.
                This takes priority over the broader qualitative
                healthy/stressed classifier.
                """
            )

        elif (
            condition_source
            ==
            "observation_condition_evidence"
        ):

            if organism_absent:

                st.info(
                    "The supporting observation records the organism as "
                    "absent from the treatment and carries a stressed-state "
                    "label. The overall condition therefore comes from that "
                    "observation's provenance. Absence is not being "
                    "reinterpreted as mortality, and the displayed score is "
                    "an evidence score rather than a probability."
                )

            else:

                st.info(
                    """
                    No qualifying direct adverse-response relationship
                    determined the overall state. The condition was instead
                    supported by the healthy/stressed provenance of the
                    observations contributing the current response evidence.
                    The displayed score is an evidence score, not a classifier
                    probability.
                    """
                )

            st.caption(
                "Condition source: "
                f"`{condition_source}`"
            )

        elif (
            condition_source
            ==
            "qualitative_state_classifier"
        ):

            st.info(
                """
                No direct adverse response or unambiguous observation-condition
                evidence determined the overall state, so the qualitative
                healthy/stressed classifier was used as a fallback.
                """
            )

            if classifier_environment_terms:

                st.caption(
                    "Classifier-recognised environmental term(s): "
                    + ", ".join(
                        humanize_value(term)
                        for term in classifier_environment_terms
                    )
                )

        elif condition_source:

            st.caption(
                "Condition source: "
                f"`{condition_source}`"
            )

    else:

        if (
            condition_source
            ==
            "contextual_evidence_no_condition"
        ):

            st.info(
                "The predicted response is supported by contextual "
                "evidence, but that evidence does not justify assigning "
                "an overall healthy or stressed state."
            )

        elif (
            condition_source
            ==
            "qualitative_state_classifier_no_input_coverage"
        ):

            st.info(
                "No healthy/stressed state was assigned because the "
                "qualitative classifier did not recognise a supported "
                "environmental factor term in the supplied input. A "
                "species-only baseline prediction is not shown."
            )

            st.caption(
                "Condition source: "
                f"`{condition_source}`"
            )

        else:

            st.info(
                "No biological-condition prediction is currently available."
            )

    # ========================================================
    # VISUAL STATE
    # ========================================================

    st.subheader(
        "Predicted visual appearance"
    )

    visual_state = (
        simulation_result
        .visual_state
    )

    if visual_state is None:

        st.info(
            "No visual-state result is available."
        )

        st.markdown(
            "### World Model Interpretation"
        )

        st.write(
            build_world_model_interpretation(
                simulation_result
            )
        )

        return

    if not visual_state.has_image:

        st.info(
            visual_state.description
            or
            (
                "No suitable microscopy reference is currently "
                "available for this predicted state."
            )
        )

        st.markdown(
            "### World Model Interpretation"
        )

        st.write(
            build_world_model_interpretation(
                simulation_result
            )
        )

        return

    visual_similarity = (
        visual_state
        .evidence_similarity
    )

    response_similarity = float(
        visual_state.response_similarity
        or
        0.0
    )

    environment_similarity = float(
        visual_state.environment_similarity
        or
        0.0
    )

    state_neutral_reference = (
        visual_state.evidence_type
        ==
        "external_state_neutral_reference"
    )

    has_environment_match = bool(
        visual_state.matching_environment_factors
    )

    # --------------------------------------------------------
    # CLASSIFY THE VISUAL MATCH
    # --------------------------------------------------------

    if state_neutral_reference:

        visual_match_label = (
            "State-neutral species reference"
        )

        st.info(
            "An exact-species external visual reference is available, "
            "but its source does not support a healthy/stressed label. "
            "It is shown for species-level visual context only."
        )

    elif response_similarity <= 0 and not has_environment_match:

        visual_match_label = (
            "State-only reference"
        )

        st.warning(
            """
            A same-species, same-state microscopy image is available, but it
            does not share a supported predicted response or matching
            environmental context. It is a state-level reference only. Its
            documented features are not predictions for this simulation.
            """
        )

    elif response_similarity > 0 and has_environment_match:

        visual_match_label = (
            "Response-and-context analogue"
        )

        st.success(
            """
            The selected same-species, same-state reference shares supported
            biological-response evidence and some environmental-context
            similarity. It remains an analogue rather than an exact visual
            prediction for the supplied condition.
            """
        )

    elif response_similarity > 0:

        visual_match_label = (
            "Response-matched reference"
        )

        st.info(
            """
            The selected same-species, same-state reference shares supported
            biological-response evidence, but no matching environmental
            context was found. It is not a condition-specific prediction.
            """
        )

    else:

        visual_match_label = (
            "Context-matched reference"
        )

        st.info(
            """
            The selected same-species, same-state reference has some
            environmental-context similarity, but it does not share a
            supported predicted biological response. Its documented features
            are not additional predictions for this simulation.
            """
        )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image_path = Path(
        visual_state.image_path
    )

    if image_path.exists():

        st.image(
            str(
                image_path
            ),
            caption=(
                f"{visual_match_label} — "
                f"{visual_state.species}"
            ),
            use_container_width=True
        )

        if getattr(visual_state, "display_crop_used", False):
            panel_label = getattr(visual_state, "display_crop_label", None)
            panel_text = f" ({panel_label})" if panel_label else ""
            st.caption(
                "Reviewed record-specific panel crop"
                f"{panel_text}. The complete source figure remains available "
                "below for provenance and is not overwritten."
            )
            crop_notes = getattr(visual_state, "display_crop_notes", None)
            if crop_notes:
                st.caption(crop_notes)
            source_image_path = getattr(
                visual_state,
                "source_image_path",
                None,
            )
            if source_image_path:
                source_path = Path(source_image_path)
                if source_path.exists() and source_path != image_path:
                    with st.expander("View complete source evidence figure"):
                        st.image(
                            str(source_path),
                            caption="Complete source figure from the evidence record",
                            use_container_width=True,
                        )

    else:

        st.warning(
            "The selected visual-reference image could not be found "
            "on disk."
        )

        st.caption(
            str(
                image_path
            )
        )

    # --------------------------------------------------------
    # MATCH METADATA
    # --------------------------------------------------------

    visual_col1, visual_col2 = (
        st.columns(
            2
        )
    )

    with visual_col1:

        st.metric(
            "Visual match",
            visual_match_label
        )

    with visual_col2:

        if (
            visual_similarity
            is not None
        ):

            st.metric(
                "Combined visual match",
                f"{visual_similarity:.3f}"
            )

    if (
        visual_state.response_similarity
        is not None
        or
        visual_state.environment_similarity
        is not None
    ):

        score_col1, score_col2 = st.columns(
            2
        )

        with score_col1:

            if (
                visual_state.response_similarity
                is not None
            ):

                st.metric(
                    "Biological-response similarity",
                    f"{visual_state.response_similarity:.3f}"
                )

        with score_col2:

            if (
                visual_state.environment_similarity
                is not None
            ):

                st.metric(
                    "Environmental similarity",
                    f"{visual_state.environment_similarity:.3f}"
                )

    if visual_state.matching_response_types:

        st.markdown(
            "**Shared predicted/visual responses**"
        )

        unique_matching_response_types = list(
            dict.fromkeys(
                visual_state.matching_response_types
            )
        )

        for response_type in (
            unique_matching_response_types
        ):

            st.write(
                f"- {humanize(response_type)}"
            )
    
    # --------------------------------------------------------
    # PHENOTYPE / FEATURES
    # --------------------------------------------------------

    if visual_state.phenotype:

        st.write(
            "**Phenotype documented in the selected reference:** "
            f"{humanize(visual_state.phenotype)}"
        )

    if (
        visual_state
        .visual_features
    ):

        st.markdown(
            "**Features documented in the selected reference**"
        )

        for feature in (
            visual_state
            .visual_features
        ):

            # Some database records store several feature=value
            # expressions in one pipe-separated field.
            feature_parts = [
                item.strip()
                for item
                in str(feature).split("|")
                if item.strip()
            ]

            for feature_part in feature_parts:

                st.write(
                    f"- {humanize(feature_part)}"
                )

        st.caption(
            "These features describe the selected reference record. They "
            "are not additional morphological effects predicted for the "
            "current water condition."
        )

    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    if visual_state.description:

        st.markdown(
            "**Visual interpretation**"
        )

        st.write(
            visual_state.description
        )

    # --------------------------------------------------------
    # PROVENANCE / LIMITATION
    # --------------------------------------------------------

    if (
        visual_state
        .evidence_record_id
    ):

        st.caption(
            "Visual evidence record: "
            f"`{visual_state.evidence_record_id}`"
        )

    if visual_state.evidence_type:

        st.caption(
            "Visual evidence type: "
            f"{humanize(visual_state.evidence_type)}"
        )

    if state_neutral_reference:

        st.caption(
            "This V1 image is an exact-species, state-neutral reference for "
            "visual context only. Because it has no supported healthy/stressed "
            "label, V2 will not transform it."
        )

    else:

        st.caption(
            "This V1 image is a reference selected from the project's "
            "evidence dataset. V2 applies a conservative illustrative "
            "transformation only when a predicted response has an explicit "
            "static-image rule. Features already present in the reference "
            "are not automatically predictions for the current condition."
        )

    # ========================================================
    # FINAL INTEGRATED V1 DESCRIPTION
    # ========================================================

    st.markdown(
        "### World Model Interpretation"
    )

    st.write(
        build_world_model_interpretation(
            simulation_result
        )
    )



# ============================================================
# V2 ILLUSTRATIVE VISUAL SIMULATION
# ============================================================

def show_v2_visual_simulation(
    simulation_result,
    minimum_evidence_score
):
    """Display evidence-aware V2 without rerunning the biological model."""

    predicted_condition = simulation_result.predicted_condition

    raw_result = simulation_result.raw_result or {}
    raw_predictions = raw_result.get("predictions", [])

    retained_predictions = [
        prediction
        for prediction in raw_predictions
        if prediction.get("evidence_score", 0.0) >= minimum_evidence_score
    ]

    safe_species = simulation_result.current_state.species.replace(" ", "_")
    output_state = predicted_condition or "state_neutral"
    output_name = f"streamlit_{safe_species}_{output_state}_v2.jpg"

    try:
        preferred_reference_record_id = None

        if simulation_result.visual_state is not None:
            preferred_reference_record_id = (
                simulation_result.visual_state.evidence_record_id
            )

        v2_result = simulate_visual_appearance(
            species=simulation_result.current_state.species,
            biological_state=predicted_condition,
            predicted_responses=retained_predictions,
            output_name=output_name,
            preferred_reference_record_id=(
                preferred_reference_record_id
            ),
        )
    except Exception as error:
        st.warning("The V2 illustrative visual simulation could not be generated.")
        st.exception(error)
        return

    if v2_result.get("organism_absent", False):

        st.markdown("### V2 Illustrative Visual Simulation")

        st.caption(
            "V2 applies a visual transformation only when the current "
            "evidence supports a static-image rule and an organism "
            "appearance is applicable."
        )

        st.info(
            "No organism appearance is generated because the "
            "evidence-backed response is absence from the supporting "
            "treatment observation. A species image could be shown "
            "separately for identification, but it would not represent the "
            "predicted condition. Absence is not being interpreted as "
            "mortality."
        )

        st.metric(
            "Transformation intensity",
            "0.000",
        )

        st.caption(v2_result.get("note", ""))
        return

    simulated_path = Path(v2_result["simulated_image_path"])
    reference_path = Path(v2_result["reference_image_path"])

    visual_change_supported = v2_result.get("visual_change_supported", False)
    transformable_responses = v2_result.get("transformable_response_types", [])
    applied_transform_responses = v2_result.get(
        "applied_transform_response_types",
        [],
    )
    non_static_responses = v2_result.get("non_static_response_types", [])
    unknown_responses = v2_result.get("unknown_visual_response_types", [])
    applied_response_rules = v2_result.get("applied_response_rules", [])
    visual_rule_decisions = v2_result.get("visual_rule_decisions", [])
    visual_response_roles = v2_result.get("visual_response_roles", {}) or {}
    reference_profile = v2_result.get("reference_visual_profile", {}) or {}
    transform_policy = v2_result.get("reference_transform_policy", {}) or {}
    full_reference_value = v2_result.get("full_reference_image_path")
    full_reference_path = (
        Path(full_reference_value)
        if full_reference_value
        else reference_path
    )
    using_verified_crop = (
        transform_policy.get("review_status") == "verified_crop"
        and reference_path != full_reference_path
    )
    difference_metrics = v2_result.get("difference_metrics") or {}
    difference_image_value = v2_result.get("difference_image_path")
    difference_path = (
        Path(difference_image_value)
        if difference_image_value
        else None
    )
    blocked_rule_decisions = [
        decision
        for decision in visual_rule_decisions
        if not decision.get("applied", False)
    ]
    state_neutral_reference = v2_result.get(
        "reference_is_state_neutral",
        False,
    )
    reference_differs_from_v1 = v2_result.get(
        "reference_differs_from_v1",
        False,
    )
    v1_reference_record_id = v2_result.get(
        "v1_reference_record_id"
    )

    st.markdown("### V2 Illustrative Visual Simulation")

    st.caption(
        "V2 uses the predicted biological state and responses to select an "
        "evidence-backed microscopy reference. A visual transformation is "
        "applied only when the current evidence supports a static-image rule "
        "for the predicted response. Otherwise, the reference image is "
        "preserved rather than inventing unsupported morphology."
    )

    if reference_differs_from_v1:
        st.info(
            "The best V1 visual analogue is retained in the section above, "
            "but it is not approved for image transformation. V2 therefore "
            "uses a separate reviewed crop from the same species and "
            "biological state that has value-level agreement with at least "
            "one predicted response covered by a static visual rule."
        )
        st.caption(
            "V1 analogue record: "
            f"`{v1_reference_record_id}`  •  "
            "V2 transformation reference: "
            f"`{v2_result['reference_record_id']}`"
        )

    if state_neutral_reference:
        st.info(
            "The reference is state-neutral: it is exact-species visual "
            "evidence, but it is not labelled healthy or stressed. No "
            "visual transformation is applied."
        )
    elif visual_change_supported:
        st.success(
            "Evidence-supported illustrative transformation available. "
            "At least one predicted biological response has a supported "
            "static-image transformation rule."
        )
    elif blocked_rule_decisions:
        st.warning(
            "A static-image rule exists, but it was withheld because the "
            "selected reference did not pass the visual-safety checks. The "
            "reference is preserved without changing labels, scale bars, or "
            "unrelated montage panels."
        )
    else:
        st.info(
            "No supported static visual transformation is available for the "
            "predicted biological responses. The evidence-backed reference "
            "image is therefore preserved rather than inventing a visible "
            "morphological change."
        )

    if visual_change_supported:
        reference_col, simulated_col = st.columns(2)

        with reference_col:
            if reference_path.exists():
                st.image(
                    str(reference_path),
                    caption=(
                        "Separate reviewed organism-region crop used for V2 transformation"
                        if reference_differs_from_v1
                        else "Verified organism-region crop used for transformation"
                        if using_verified_crop
                        else "Evidence-backed reference image"
                    ),
                    use_container_width=True,
                )
            else:
                st.info("Reference image is unavailable.")

        with simulated_col:
            if simulated_path.exists():
                st.image(
                    str(simulated_path),
                    caption="V2 illustrative simulated appearance",
                    use_container_width=True,
                )
            else:
                st.info("V2 output image is unavailable.")

        if difference_path is not None and difference_path.exists():
            st.markdown("#### Rendering difference map")
            st.image(
                str(difference_path),
                caption=(
                    "Amplified pixel-level rendering difference. Brighter "
                    "orange regions changed more during image processing; "
                    "this is not a biological-effect or uncertainty map."
                ),
                use_container_width=True,
            )

    else:
        if reference_path.exists():
            st.image(
                str(reference_path),
                caption="Evidence-backed reference image",
                use_container_width=True,
            )
        else:
            st.info("Reference image is unavailable.")

        if state_neutral_reference:

            st.info(
                "**No transformed image generated.** The selected image is "
                "state-neutral and cannot justify a condition-specific visual "
                "change. It is retained for exact-species context only."
            )

        elif blocked_rule_decisions:

            st.info(
                "**No transformed image generated.** A static-image rule was "
                "available, but the selected reference did not pass the "
                "visual-safety checks. The original reference is retained."
            )

        else:

            st.info(
                "**No transformed image generated.** The current predicted "
                "responses and evidence do not justify a static visual "
                "transformation. The selected reference is retained without "
                "inventing additional morphology."
            )

    metric_count = 4 if difference_metrics else 2
    score_columns = st.columns(metric_count)

    with score_columns[0]:
        st.metric(
            "Reference response match",
            f"{v2_result['reference_response_match']:.3f}",
        )

    with score_columns[1]:
        st.metric(
            "Applied transformation intensity",
            f"{v2_result['transformation_intensity']:.3f}",
        )

    if difference_metrics:
        with score_columns[2]:
            st.metric(
                "Changed pixels",
                f"{difference_metrics['changed_pixel_fraction']:.1%}",
            )

        with score_columns[3]:
            st.metric(
                "Mean rendering difference",
                f"{difference_metrics['mean_absolute_difference']:.4f}",
            )

        st.caption(
            "The difference metrics describe image-processing changes only. "
            "They are not biological effect sizes, probabilities, or model "
            "confidence values."
        )

    if visual_rule_decisions:
        with st.expander("Visual-rule safety and modality checks"):
            if reference_profile:
                st.write(
                    "**Reference dimensions:** "
                    f"{reference_profile.get('width')} × "
                    f"{reference_profile.get('height')} pixels"
                )
                st.write(
                    "**Composite/annotation-layout risk:** "
                    + (
                        "Detected"
                        if reference_profile.get("composite_layout_risk")
                        else "Not detected by the current heuristic"
                    )
                )
                st.write(
                    "**Organism mask available:** "
                    + (
                        "Yes"
                        if reference_profile.get("organism_mask_available")
                        else "No"
                    )
                )
                st.write(
                    "**Fluorescence-like modality check:** "
                    + (
                        "Passed"
                        if reference_profile.get("fluorescence_like_supported")
                        else "Not passed"
                    )
                )
                if "dark_background_fraction" in reference_profile:
                    st.caption(
                        "Fluorescence heuristic: dark-background fraction="
                        f"{reference_profile.get('dark_background_fraction', 0):.3f}; "
                        "saturated-signal fraction="
                        f"{reference_profile.get('saturated_signal_fraction', 0):.3f}. "
                        "This is an image-modality safety signal, not a "
                        "biological measurement."
                    )

            if transform_policy:
                st.write(
                    "**Transform review status:** "
                    f"{humanize(transform_policy.get('review_status', 'unreviewed'))}"
                )
                st.write(
                    "**Explicitly approved for transformation:** "
                    + (
                        "Yes"
                        if transform_policy.get("transform_eligible")
                        else "No"
                    )
                )
                risk_reasons = transform_policy.get("automated_risk_reasons")
                if risk_reasons:
                    st.write(
                        "**Automated risk signals:** "
                        f"{str(risk_reasons).replace('|', ', ')}"
                    )
                review_notes = transform_policy.get("review_notes")
                if review_notes:
                    st.caption(f"Review note: {review_notes}")

            for decision in visual_rule_decisions:
                status = "Applied" if decision.get("applied") else "Withheld"
                st.markdown(
                    f"**{humanize(decision.get('rule'))}: {status}**"
                )
                st.write(decision.get("reason", ""))
                st.caption(
                    "Modality requirement: "
                    f"{decision.get('modality_requirement', 'Not specified')}"
                )
                st.caption(
                    "Transformation scope: "
                    f"{decision.get('transformation_scope', 'Not specified')}"
                )

    population_responses = visual_response_roles.get("population_level", [])
    occurrence_responses = visual_response_roles.get("occurrence_level", [])
    temporal_responses = visual_response_roles.get("temporal", [])
    behavioral_responses = visual_response_roles.get("behavioral", [])
    physiological_responses = visual_response_roles.get(
        "physiological_non_static",
        [],
    )

    if occurrence_responses:
        st.markdown("**Occurrence-level responses (observation status):**")
        for response in occurrence_responses:
            st.write(f"- {humanize(response)}")

    if population_responses:
        st.markdown("**Population-level responses (not single-cell morphology):**")
        for response in population_responses:
            st.write(f"- {humanize(response)}")

    if temporal_responses:
        st.markdown("**Time-dependent responses (not represented by one image):**")
        for response in temporal_responses:
            st.write(f"- {humanize(response)}")

    if behavioral_responses:
        st.markdown("**Behavioral responses (require video or time-series evidence):**")
        for response in behavioral_responses:
            st.write(f"- {humanize(response)}")

    if physiological_responses:
        st.markdown("**Physiological responses without a validated static rule:**")
        for response in physiological_responses:
            st.write(f"- {humanize(response)}")

    if applied_transform_responses and visual_change_supported:
        st.markdown("**Responses used for the illustrative transformation:**")
        for response in applied_transform_responses:
            st.write(f"- {humanize(response)}")

    elif transformable_responses:
        st.markdown("**Responses with a static-image rule not applied here:**")
        for response in transformable_responses:
            st.write(f"- {humanize(response)}")

        st.caption(
            "Having a static-image rule is not sufficient by itself. The "
            "selected reference and current evidence must also support using "
            "that rule in this simulation."
        )

    unused_transformable_responses = [
        response
        for response in transformable_responses
        if response not in applied_transform_responses
    ]

    if visual_change_supported and unused_transformable_responses:
        st.markdown(
            "**Static-rule responses not applied to this fallback reference:**"
        )
        for response in unused_transformable_responses:
            st.write(f"- {humanize(response)}")
        st.caption(
            "These responses remain biological predictions, but the separate "
            "V2 reference does not provide value-level visual agreement for "
            "using their rules in this rendering."
        )

    if applied_response_rules:
        st.markdown("**Applied visual rule categories:**")
        for rule in applied_response_rules:
            st.write(f"- {humanize(rule)}")

    categorised_non_static = set(
        occurrence_responses
        + population_responses
        + temporal_responses
        + behavioral_responses
        + physiological_responses
    )
    remaining_non_static_responses = [
        response
        for response in non_static_responses
        if response not in categorised_non_static
    ]

    if remaining_non_static_responses:
        st.markdown("**Predicted responses not converted into visual changes:**")
        for response in remaining_non_static_responses:
            st.write(f"- {humanize(response)}")

    if unknown_responses:
        st.markdown("**Responses without a current visual rule:**")
        for response in unknown_responses:
            st.write(f"- {humanize(response)}")

    note = v2_result.get("note")
    if note:
        st.caption(note)

    st.caption(
        "V2 reference record: "
        f"`{v2_result['reference_record_id']}`"
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "🔬 Microorganism World Model"
)

st.write(
    """
    This prototype explores relationships between water-quality
    conditions and the observed responses of **Nitzschia palea**,
    **Euglena viridis**, and **Oscillatoria limosa** using the
    evidence stored in the project database.
    """
)

st.info(
    """
    **Important:** The evidence score is a ranking score derived
    from the current evidence database. It is **not a biological
    probability** and should not be interpreted as a percentage
    chance that a biological response or environmental condition
    will occur.
    """
)


# ============================================================
# DATABASE CHECK
# ============================================================

if not DATABASE_FILE.exists():

    st.error(
        f"""
        The SQLite database could not be found:

        `{DATABASE_FILE}`

        Build the database first by running:

        `python src/build_database.py`
        """
    )

    st.stop()


# ============================================================
# LOAD SPECIES DYNAMICALLY
# ============================================================

try:

    SPECIES = get_species()

except Exception as error:

    st.error(
        "Could not read microorganism species "
        "from the database."
    )

    st.exception(
        error
    )

    st.stop()


if not SPECIES:

    st.error(
        """
        No microorganisms were found in the database.

        Rebuild the database and try again.
        """
    )

    st.stop()


# ============================================================
# MODE SELECTION
# ============================================================

mode = st.radio(
    "Choose model mode:",
    [
        "Water quality → Microorganism response",
        "Microorganism response → Water quality",
        "Simulate world state",
        "About this model",
    ],
    horizontal=True
)

st.divider()


# ============================================================
# FORWARD MODEL
# ============================================================

if (
    mode
    ==
    "Water quality → Microorganism response"
):

    st.header(
        "Forward Model"
    )

    st.write(
        """
        Select a microorganism and one or more water-quality
        conditions. The model will rank organism responses
        supported by the current evidence database.
        """
    )

    # --------------------------------------------------------
    # SPECIES
    # --------------------------------------------------------

    species = st.selectbox(
        "Microorganism",
        SPECIES,
        key="forward_species"
    )

    # --------------------------------------------------------
    # GENERAL REFERENCE VIDEOS
    # --------------------------------------------------------

    show_reference_videos(
        species
    )

    # --------------------------------------------------------
    # WATER FACTORS
    # --------------------------------------------------------

    available_factors = get_water_factors(
        species
    )

    if not available_factors:

        st.warning(
            """
            No water-quality factors with evidence relationships
            were found for this microorganism.
            """
        )

        st.stop()

    # --------------------------------------------------------
    # WATER-QUALITY INPUTS
    # --------------------------------------------------------

    st.subheader(
        "Water-quality inputs"
    )

    selected_factors = st.multiselect(
        "Choose one or more water-quality factors",
        available_factors,
        format_func=humanize,
        help=(
            "The available factors are read directly from "
            "SQLite. New factors will appear automatically "
            "after the database is rebuilt."
        )
    )

    water_state = {}

    # ========================================================
    # CONFIGURE EACH WATER FACTOR
    # ========================================================

    for factor in selected_factors:

        with st.expander(
            humanize(factor),
            expanded=True
        ):

            # ------------------------------------------------
            # EXISTING DATABASE VALUES
            # ------------------------------------------------

            show_water_factor_database_info(
                species,
                factor
            )

            st.divider()

            # ------------------------------------------------
            # INPUT TYPE
            # ------------------------------------------------

            value_mode = st.radio(
                "How would you like to specify this factor?",
                [
                    "Text / presence",
                    "Numeric concentration"
                ],
                key=(
                    f"forward_"
                    f"{species}_"
                    f"{factor}_mode"
                ),
                horizontal=True
            )

            # ------------------------------------------------
            # TEXT INPUT
            # ------------------------------------------------

            if (
                value_mode
                ==
                "Text / presence"
            ):

                known_factor_values = (
                    get_known_water_factor_value_options(
                        species,
                        factor
                    )
                )

                if known_factor_values:

                    option_labels = [
                        option["label"]
                        for option
                        in known_factor_values
                    ]

                    option_labels.append(
                        "Custom value..."
                    )

                    selected_label = st.selectbox(
                        "Known value / concentration",
                        option_labels,
                        key=(
                            f"forward_"
                            f"{species}_"
                            f"{factor}_known_value"
                        ),
                        help=(
                            "Known values and concentration ranges are "
                            "read directly from the evidence database. "
                            "Selecting a stored concentration range lets "
                            "the forward model match its concentration-"
                            "specific evidence without requiring you to "
                            "retype the internal numeric format."
                        ),
                        format_func=humanize_value
                    )

                    if (
                        selected_label
                        ==
                        "Custom value..."
                    ):

                        text_value = st.text_input(
                            "Custom value",
                            value="present",
                            key=(
                                f"forward_"
                                f"{species}_"
                                f"{factor}_custom_text"
                            ),
                            help=(
                                "Examples include present, high, elevated, "
                                "or another categorical database value. "
                                "For an exact numeric concentration, use "
                                "the Numeric concentration mode instead."
                            )
                        )

                        water_state[
                            factor
                        ] = (
                            text_value.strip()
                            or
                            "present"
                        )

                    else:

                        selected_option = next(
                            option
                            for option
                            in known_factor_values
                            if option["label"]
                            ==
                            selected_label
                        )

                        water_state[
                            factor
                        ] = selected_option[
                            "model_value"
                        ]

                        if (
                            selected_option["kind"]
                            ==
                            "concentration"
                        ):

                            st.caption(
                                "This stored concentration/range will be "
                                "matched using a representative numeric "
                                "value inside that range."
                            )

                else:

                    text_value = st.text_input(
                        "Value",
                        value="present",
                        key=(
                            f"forward_"
                            f"{species}_"
                            f"{factor}_text"
                        ),
                        help=(
                            "Examples include present, high, "
                            "elevated, or another known database "
                            "value."
                        )
                    )

                    water_state[
                        factor
                    ] = (
                        text_value.strip()
                        or
                        "present"
                    )

            # ------------------------------------------------
            # NUMERIC INPUT
            # ------------------------------------------------

            else:

                numeric_value = st.number_input(
                    "Numeric concentration/value",
                    min_value=0.0,
                    value=10.0,
                    step=1.0,
                    key=(
                        f"forward_"
                        f"{species}_"
                        f"{factor}_numeric"
                    )
                )

                unit = st.text_input(
                    "Unit",
                    value="",
                    placeholder=(
                        "Examples: uL/L, ug/L, mg/L"
                    ),
                    key=(
                        f"forward_"
                        f"{species}_"
                        f"{factor}_unit"
                    )
                )

                water_state[
                    factor
                ] = {
                    "value":
                        float(
                            numeric_value
                        ),

                    "unit":
                        unit.strip()
                        or
                        None
                }

    # ========================================================
    # FORWARD PREDICTION
    # ========================================================

    if st.button(
        "Predict microorganism response",
        type="primary"
    ):

        if not water_state:

            st.warning(
                """
                Please choose at least one water-quality factor.
                """
            )

        else:

            with st.spinner(
                "Querying the evidence database..."
            ):

                try:

                    result = predict_response(
                        species=species,
                        water_state=water_state,
                        top_k=10
                    )

                except Exception as error:

                    st.error(
                        "The forward model encountered an error."
                    )

                    st.exception(
                        error
                    )

                    st.stop()

            predictions = (
                result[
                    "predictions"
                ]
            )

            # ------------------------------------------------
            # NO RESULTS
            # ------------------------------------------------

            if not predictions:

                st.warning(
                    """
                    No matching evidence was found for the
                    selected microorganism and water state.

                    This does not necessarily mean that no
                    response exists. It means that the current
                    evidence database does not contain a
                    matching relationship.
                    """
                )

            # ------------------------------------------------
            # RESULTS
            # ------------------------------------------------

            else:

                st.success(
                    f"Found {len(predictions)} "
                    "supported response result(s)."
                )

                st.subheader(
                    "Predicted responses"
                )

                for (
                    index,
                    prediction
                ) in enumerate(
                    predictions,
                    start=1
                ):

                    with st.container(
                        border=True
                    ):

                        st.subheader(
                            f"{index}. "
                            f"{humanize(prediction['response_type'])}"
                        )

                        col1, col2, col3 = st.columns(
                            [2, 1, 1]
                        )

                        with col1:

                            show_response_value(
                                prediction[
                                    "response_type"
                                ],
                                prediction[
                                    "response_value"
                                ],
                            )

                        with col2:

                            st.metric(
                                "Evidence score",
                                prediction[
                                    "evidence_score"
                                ]
                            )

                        with col3:

                            st.metric(
                                "Independent observations",
                                prediction[
                                    "independent_observations"
                                ]
                            )

                        st.caption(
                            "Evidence strength: "
                            f"**{evidence_label(prediction['evidence_score'])}**"
                        )

                        # ====================================
                        # SUPPORTING EVIDENCE
                        # ====================================

                        with st.expander(
                            "View supporting evidence"
                        ):

                            for (
                                evidence_index,
                                evidence
                            ) in enumerate(
                                prediction[
                                    "supporting_evidence"
                                ],
                                start=1
                            ):

                                st.markdown(
                                    f"### Evidence "
                                    f"{evidence_index}"
                                )

                                st.write(
                                    "**Observation:** "
                                    f"`{evidence['observation_id']}`"
                                )

                                if (
                                    evidence[
                                        "case_id"
                                    ]
                                ):

                                    st.write(
                                        "**Case:** "
                                        f"`{evidence['case_id']}`"
                                    )

                                if (
                                    evidence[
                                        "case_concentration"
                                    ]
                                ):

                                    st.write(
                                        "**Case concentration:** "
                                        f"{evidence['case_concentration']}"
                                    )

                                st.write(
                                    "**Relationship type:** "
                                    f"{humanize(evidence['relationship_type'])}"
                                )

                                st.write(
                                    "**Evidence contribution:** "
                                    f"{round(evidence['evidence_strength'], 3)}"
                                )

                                concentration_specificity = (
                                    evidence[
                                        "concentration_specificity"
                                    ]
                                )

                                if (
                                    concentration_specificity
                                    !=
                                    "not_requested"
                                ):

                                    st.write(
                                        "**Concentration matching:** "
                                        f"{humanize(concentration_specificity)}"
                                    )

                                if (
                                    evidence[
                                        "source_url"
                                    ]
                                ):

                                    st.link_button(
                                        "Open scientific source",
                                        evidence[
                                            "source_url"
                                        ],
                                        key=(
                                            f"forward_source_"
                                            f"{index}_"
                                            f"{evidence_index}_"
                                            f"{evidence['observation_id']}"
                                        )
                                    )

                                # =================================
                                # OBSERVATION MEDIA
                                # =================================

                                st.markdown(
                                    "**Associated microscopy media:**"
                                )

                                show_observation_media(
                                    evidence[
                                        "observation_id"
                                    ],
                                    context_key=(
                                        f"forward_"
                                        f"{index}_"
                                        f"{evidence_index}"
                                    )
                                )

                                if (
                                    evidence_index
                                    <
                                    len(
                                        prediction[
                                            "supporting_evidence"
                                        ]
                                    )
                                ):

                                    st.divider()


# ============================================================
# INVERSE MODEL
# ============================================================

elif (
    mode
    ==
    "Microorganism response → Water quality"
):

    st.header(
        "Inverse Model"
    )

    st.write(
        """
        Select a microorganism and one or more observed
        biological responses. The model will rank water-quality
        conditions associated with those observations in the
        current evidence database.
        """
    )

    # --------------------------------------------------------
    # SPECIES
    # --------------------------------------------------------

    species = st.selectbox(
        "Microorganism",
        SPECIES,
        key="inverse_species"
    )

    # --------------------------------------------------------
    # GENERAL REFERENCE VIDEOS
    # --------------------------------------------------------

    show_reference_videos(
        species
    )

    # --------------------------------------------------------
    # AVAILABLE RESPONSES
    # --------------------------------------------------------

    available_responses = get_response_types(
        species
    )

    if not available_responses:

        st.warning(
            """
            No organism responses with evidence relationships
            were found for this microorganism.
            """
        )

        st.stop()

    # --------------------------------------------------------
    # RESPONSE SELECTION
    # --------------------------------------------------------

    selected_responses = st.multiselect(
        "Choose one or more observed responses",
        available_responses,
        format_func=humanize,
        help=(
            "These response types are read directly from the "
            "SQLite database."
        )
    )

    observed_response = {}

    # ========================================================
    # CONFIGURE EACH RESPONSE
    # ========================================================

    for response in selected_responses:

        with st.expander(
            humanize(response),
            expanded=True
        ):

            # ------------------------------------------------
            # KNOWN RESPONSE VALUES
            # ------------------------------------------------

            show_response_database_info(
                species,
                response
            )

            st.divider()

            # ------------------------------------------------
            # INPUT MODE
            # ------------------------------------------------

            value_mode = st.radio(
                "How would you like to specify this response?",
                [
                    "Text / presence",
                    "Numeric value"
                ],
                key=(
                    f"inverse_"
                    f"{species}_"
                    f"{response}_mode"
                ),
                horizontal=True
            )

            # ------------------------------------------------
            # TEXT RESPONSE
            # ------------------------------------------------

            if (
                value_mode
                ==
                "Text / presence"
            ):

                known_response_values = (
                    get_known_response_value_options(
                        species,
                        response
                    )
                )

                if known_response_values:

                    response_value_options = (
                        known_response_values
                        +
                        ["Custom value..."]
                    )

                    selected_value = st.selectbox(
                        "Observed value",
                        response_value_options,
                        key=(
                            f"inverse_"
                            f"{species}_"
                            f"{response}_known_value"
                        ),
                        help=(
                            "Known values are read directly from "
                            "the current evidence database. Choose "
                            "'Custom value...' only when you need "
                            "to enter a value that is not listed."
                        ),
                        format_func=humanize_value
                    )

                    if (
                        selected_value
                        ==
                        "Custom value..."
                    ):

                        text_value = st.text_input(
                            "Custom observed value",
                            value="",
                            key=(
                                f"inverse_"
                                f"{species}_"
                                f"{response}_custom_text"
                            ),
                            help=(
                                "Enter a categorical value, numeric "
                                "value with unit, or range exactly as "
                                "you want the inverse model to match."
                            )
                        )

                        observed_response[
                            response
                        ] = text_value.strip()

                    else:

                        observed_response[
                            response
                        ] = selected_value

                else:

                    text_value = st.text_input(
                        "Observed value",
                        value="present",
                        key=(
                            f"inverse_"
                            f"{species}_"
                            f"{response}_text"
                        )
                    )

                    observed_response[
                        response
                    ] = (
                        text_value.strip()
                        or
                        "present"
                    )

            # ------------------------------------------------
            # NUMERIC RESPONSE
            # ------------------------------------------------

            else:

                numeric_value = st.number_input(
                    "Numeric response value",
                    value=0.0,
                    key=(
                        f"inverse_"
                        f"{species}_"
                        f"{response}_numeric"
                    )
                )

                unit = st.text_input(
                    "Unit",
                    value="",
                    placeholder=(
                        "Examples: percent, um"
                    ),
                    key=(
                        f"inverse_"
                        f"{species}_"
                        f"{response}_unit"
                    )
                )

                observed_response[
                    response
                ] = {
                    "value":
                        float(
                            numeric_value
                        ),

                    "unit":
                        unit.strip()
                        or
                        None
                }

    # ========================================================
    # INVERSE PREDICTION
    # ========================================================

    if st.button(
        "Infer possible water quality",
        type="primary"
    ):

        if not observed_response:

            st.warning(
                """
                Please choose at least one observed
                microorganism response.
                """
            )

        else:

            with st.spinner(
                "Querying the evidence database..."
            ):

                try:

                    result = infer_water_quality(
                        species=species,
                        observed_response=observed_response,
                        top_k=10
                    )

                except Exception as error:

                    st.error(
                        "The inverse model encountered an error."
                    )

                    st.exception(
                        error
                    )

                    st.stop()

            conditions = (
                result[
                    "possible_water_conditions"
                ]
            )

            # ------------------------------------------------
            # NO RESULTS
            # ------------------------------------------------

            if not conditions:

                st.warning(
                    """
                    No matching water-quality evidence was found
                    for the selected response.

                    This means that the current evidence
                    database does not contain a matching
                    relationship.
                    """
                )

            # ------------------------------------------------
            # RESULTS
            # ------------------------------------------------

            else:

                st.success(
                    f"Found {len(conditions)} "
                    "possible water-quality condition(s)."
                )

                st.subheader(
                    "Possible water conditions"
                )

                for (
                    index,
                    condition
                ) in enumerate(
                    conditions,
                    start=1
                ):

                    with st.container(
                        border=True
                    ):

                        st.subheader(
                            f"{index}. "
                            f"{humanize(condition['factor_name'])}"
                        )

                        col1, col2, col3 = st.columns(
                            [2, 1, 1]
                        )

                        with col1:

                            show_wrapped_value(
                                "Water value",
                                format_readable_value(
                                    condition[
                                        "factor_value"
                                    ]
                                ),
                                raw_value=condition[
                                    "factor_value"
                                ],
                            )

                        with col2:

                            st.metric(
                                "Evidence score",
                                condition[
                                    "evidence_score"
                                ]
                            )

                        with col3:

                            st.metric(
                                "Independent observations",
                                condition[
                                    "independent_observations"
                                ]
                            )

                        if (
                            condition[
                                "concentration"
                            ]
                        ):

                            st.write(
                                "**Associated concentration:** "
                                f"{condition['concentration']}"
                            )

                        st.caption(
                            "Evidence strength: "
                            f"**{evidence_label(condition['evidence_score'])}**"
                        )

                        # ====================================
                        # SUPPORTING EVIDENCE
                        # ====================================

                        with st.expander(
                            "View supporting evidence"
                        ):

                            for (
                                evidence_index,
                                evidence
                            ) in enumerate(
                                condition[
                                    "supporting_evidence"
                                ],
                                start=1
                            ):

                                st.markdown(
                                    f"### Evidence "
                                    f"{evidence_index}"
                                )

                                st.write(
                                    "**Observation:** "
                                    f"`{evidence['observation_id']}`"
                                )

                                if (
                                    evidence[
                                        "case_id"
                                    ]
                                ):

                                    st.write(
                                        "**Case:** "
                                        f"`{evidence['case_id']}`"
                                    )

                                st.write(
                                    "**Relationship type:** "
                                    f"{humanize(evidence['relationship_type'])}"
                                )

                                st.write(
                                    "**Evidence contribution:** "
                                    f"{round(evidence['evidence_strength'], 3)}"
                                )

                                if (
                                    evidence[
                                        "source_url"
                                    ]
                                ):

                                    st.link_button(
                                        "Open scientific source",
                                        evidence[
                                            "source_url"
                                        ],
                                        key=(
                                            f"inverse_source_"
                                            f"{index}_"
                                            f"{evidence_index}_"
                                            f"{evidence['observation_id']}"
                                        )
                                    )

                                # =================================
                                # OBSERVATION MEDIA
                                # =================================

                                st.markdown(
                                    "**Associated microscopy media:**"
                                )

                                show_observation_media(
                                    evidence[
                                        "observation_id"
                                    ],
                                    context_key=(
                                        f"inverse_"
                                        f"{index}_"
                                        f"{evidence_index}"
                                    )
                                )

                                if (
                                    evidence_index
                                    <
                                    len(
                                        condition[
                                            "supporting_evidence"
                                        ]
                                    )
                                ):

                                    st.divider()


# ============================================================
# WORLD-STATE SIMULATOR
# ============================================================

elif mode == "Simulate world state":

    st.header(
        "World-State Simulator"
    )

    st.write(
        """
        Define the current environmental state of a
        microorganism and apply the evidence-based world model
        to generate a predicted next state, an overall biological
        condition, the closest evidence-backed visual analogue,
        and a lightweight V2 illustrative simulated appearance.
        """
    )

    st.info(
        """
        **How to interpret this simulation**

        `t → t+1` represents one application of the current
        evidence model. It does **not** currently represent a
        fixed amount of biological time such as one hour, one
        day, or one generation.
        """
    )

    # --------------------------------------------------------
    # SPECIES
    # --------------------------------------------------------

    species = st.selectbox(
        "Microorganism",
        SPECIES,
        key="simulation_species"
    )

    # --------------------------------------------------------
    # GENERAL REFERENCE VIDEOS
    # --------------------------------------------------------

    show_reference_videos(
        species
    )

    # --------------------------------------------------------
    # AVAILABLE WATER FACTORS
    # --------------------------------------------------------

    available_factors = get_water_factors(
        species
    )

    if not available_factors:

        st.warning(
            """
            No water-quality factors with evidence
            relationships were found for this microorganism.
            """
        )

        st.stop()

    # --------------------------------------------------------
    # WATER-QUALITY INPUT
    # --------------------------------------------------------

    st.subheader(
        "Current environmental state"
    )

    selected_factors = st.multiselect(
        "Choose one or more water-quality factors",
        available_factors,
        format_func=humanize,
        key="simulation_factors",
        help=(
            "These factors are read directly from the "
            "evidence database."
        )
    )

    simulation_water_state = {}

    # ========================================================
    # CONFIGURE EACH WATER FACTOR
    # ========================================================

    for factor in selected_factors:

        with st.expander(
            humanize(factor),
            expanded=True
        ):

            # ------------------------------------------------
            # SHOW KNOWN DATABASE VALUES
            # ------------------------------------------------

            show_water_factor_database_info(
                species,
                factor
            )

            st.divider()

            # ------------------------------------------------
            # INPUT MODE
            # ------------------------------------------------

            value_mode = st.radio(
                "How would you like to specify this factor?",
                [
                    "Text / presence",
                    "Numeric concentration"
                ],
                key=(
                    f"simulation_"
                    f"{species}_"
                    f"{factor}_mode"
                ),
                horizontal=True
            )

            # ------------------------------------------------
            # TEXT / PRESENCE
            # ------------------------------------------------

            if (
                value_mode
                ==
                "Text / presence"
            ):

                # Use the same evidence-backed known-value/concentration
                # selector as the forward model.
                known_factor_values = (
                    get_known_water_factor_value_options(
                        species,
                        factor
                    )
                )

                if known_factor_values:

                    option_labels = [
                        option["label"]
                        for option
                        in known_factor_values
                    ]

                    option_labels.append(
                        "Custom value..."
                    )

                    selected_label = st.selectbox(
                        "Known value / concentration",
                        option_labels,
                        key=(
                            f"simulation_"
                            f"{species}_"
                            f"{factor}_known_value"
                        ),
                        help=(
                            "Known values and concentration ranges are "
                            "read directly from the evidence database. "
                            "Selecting a stored concentration range lets "
                            "the simulator match concentration-specific "
                            "evidence using the same input format as the "
                            "forward model."
                        ),
                        format_func=humanize_value
                    )

                    if (
                        selected_label
                        ==
                        "Custom value..."
                    ):

                        text_value = st.text_input(
                            "Custom value",
                            value="present",
                            key=(
                                f"simulation_"
                                f"{species}_"
                                f"{factor}_custom_text"
                            ),
                            help=(
                                "Examples include present, high, elevated, "
                                "or another categorical database value. "
                                "For an exact numeric concentration, use "
                                "the Numeric concentration mode instead."
                            )
                        )

                        simulation_water_state[
                            factor
                        ] = (
                            text_value.strip()
                            or
                            "present"
                        )

                    else:

                        selected_option = next(
                            option
                            for option
                            in known_factor_values
                            if option["label"]
                            ==
                            selected_label
                        )

                        simulation_water_state[
                            factor
                        ] = selected_option[
                            "model_value"
                        ]

                        if (
                            selected_option["kind"]
                            ==
                            "concentration"
                        ):

                            st.caption(
                                "This stored concentration/range will be "
                                "matched using a representative numeric "
                                "value inside that range."
                            )

                else:

                    text_value = st.text_input(
                        "Value",
                        value="present",
                        key=(
                            f"simulation_"
                            f"{species}_"
                            f"{factor}_text"
                        )
                    )

                    simulation_water_state[
                        factor
                    ] = (
                        text_value.strip()
                        or
                        "present"
                    )

            # ------------------------------------------------
            # NUMERIC CONCENTRATION
            # ------------------------------------------------

            else:

                numeric_value = st.number_input(
                    "Numeric concentration/value",
                    min_value=0.0,
                    value=10.0,
                    step=1.0,
                    key=(
                        f"simulation_"
                        f"{species}_"
                        f"{factor}_numeric"
                    )
                )

                unit = st.text_input(
                    "Unit",
                    value="",
                    placeholder=(
                        "Examples: uL/L, ug/L, mg/L"
                    ),
                    key=(
                        f"simulation_"
                        f"{species}_"
                        f"{factor}_unit"
                    )
                )

                simulation_water_state[
                    factor
                ] = {
                    "value":
                        float(
                            numeric_value
                        ),

                    "unit":
                        unit.strip()
                        or
                        None
                }

    # ========================================================
    # SIMULATION SETTINGS
    # ========================================================

    st.subheader(
        "Simulation settings"
    )

    minimum_evidence_score = st.slider(
        "Minimum evidence score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        key="simulation_evidence_threshold",
        help=(
            "Predicted responses below this evidence score "
            "will not be included in the predicted next state. "
            "The score is an evidence-ranking score, not a "
            "biological probability."
        )
    )

    st.caption(
        "For example, setting the threshold to 0.9 keeps only "
        "strongly supported predictions in the simulated state."
    )

    # ========================================================
    # SIMULATE
    # ========================================================

    if st.button(
        "Simulate next state",
        type="primary",
        key="simulate_next_state"
    ):

        if not simulation_water_state:

            st.warning(
                """
                Please choose at least one water-quality factor
                before running the simulation.
                """
            )

        else:

            # ------------------------------------------------
            # CREATE CURRENT WORLD STATE
            # ------------------------------------------------

            current_state = WorldState(
                species=species,
                water_state=simulation_water_state
            )

            # ------------------------------------------------
            # CREATE SIMULATOR
            # ------------------------------------------------

            simulator = SimpleWorldSimulator(
                top_k=10,
                minimum_evidence_score=(
                    minimum_evidence_score
                )
            )

            # ------------------------------------------------
            # RUN STATE TRANSITION
            # ------------------------------------------------

            with st.spinner(
                "Simulating the next world state..."
            ):

                try:

                    simulation_result = (
                        simulator.simulate_step(
                            current_state
                        )
                    )

                except Exception as error:

                    st.error(
                        "The world-state simulator encountered "
                        "an error."
                    )

                    st.exception(
                        error
                    )

                    st.stop()

            next_state = (
                simulation_result.next_state
            )

            # =================================================
            # STATE TRANSITION
            # =================================================

            st.success(
                "World-state transition completed."
            )

            st.subheader(
                "State transition"
            )

            current_column, arrow_column, next_column = (
                st.columns(
                    [5, 1, 5]
                )
            )

            # =================================================
            # STATE AT t
            # =================================================

            with current_column:

                st.markdown(
                    "### State at t"
                )

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"**Species:** {current_state.species}"
                    )

                    st.markdown(
                        "#### Water quality"
                    )

                    for (
                        factor,
                        value
                    ) in (
                        current_state
                        .water_state
                        .items()
                    ):

                        if isinstance(
                            value,
                            dict
                        ):

                            factor_value = value.get(
                                "value"
                            )

                            unit = value.get(
                                "unit"
                            )

                            if unit:

                                display_value = (
                                    f"{factor_value} "
                                    f"{unit}"
                                )

                            else:

                                display_value = str(
                                    factor_value
                                )

                        else:

                            display_value = str(
                                value
                            )

                        st.write(
                            f"**{humanize(factor)}:** "
                            f"{humanize_value(display_value)}"
                        )

                    st.markdown(
                        "#### Microorganism responses"
                    )

                    if (
                        current_state
                        .organism_state
                    ):

                        for (
                            response_type,
                            response
                        ) in (
                            current_state
                            .organism_state
                            .items()
                        ):

                            st.markdown(
                                f"**{humanize(response_type)}**"
                            )
                            show_response_value(
                                response_type,
                                response.get("value"),
                            )

                    else:

                        st.caption(
                            "No microorganism responses are "
                            "specified in the initial state."
                        )

            # =================================================
            # ARROW
            # =================================================

            with arrow_column:

                st.markdown(
                    "<div style='text-align:center;"
                    "font-size:42px;"
                    "padding-top:120px;'>"
                    "→"
                    "</div>",
                    unsafe_allow_html=True
                )

            # =================================================
            # STATE AT t+1
            # =================================================

            with next_column:

                st.markdown(
                    "### Predicted state at t+1"
                )

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"**Species:** {next_state.species}"
                    )

                    st.markdown(
                        "#### Water quality"
                    )

                    for (
                        factor,
                        value
                    ) in (
                        next_state
                        .water_state
                        .items()
                    ):

                        if isinstance(
                            value,
                            dict
                        ):

                            factor_value = value.get(
                                "value"
                            )

                            unit = value.get(
                                "unit"
                            )

                            if unit:

                                display_value = (
                                    f"{factor_value} "
                                    f"{unit}"
                                )

                            else:

                                display_value = str(
                                    factor_value
                                )

                        else:

                            display_value = str(
                                value
                            )

                        st.write(
                            f"**{humanize(factor)}:** "
                            f"{humanize_value(display_value)}"
                        )

                    st.markdown(
                        "#### Predicted microorganism responses"
                    )

                    if (
                        next_state
                        .organism_state
                    ):

                        for (
                            response_type,
                            response
                        ) in (
                            next_state
                            .organism_state
                            .items()
                        ):

                            st.markdown(
                                f"**{humanize(response_type)}**"
                            )

                            supported_values = response.get(
                                "supported_values",
                                []
                            )

                            display_values = []

                            for supported in supported_values:

                                supported_value = str(
                                    supported.get(
                                        "value",
                                        ""
                                    )
                                ).strip()

                                if (
                                    supported_value
                                    and
                                    supported_value not in display_values
                                ):

                                    display_values.append(
                                        supported_value
                                    )

                            if not display_values:

                                primary_value = str(
                                    response.get(
                                        "value",
                                        ""
                                    )
                                ).strip()

                                if primary_value:

                                    display_values.append(
                                        primary_value
                                    )

                            if len(display_values) <= 1:
                                show_response_value(
                                    response_type,
                                    (
                                        display_values[0]
                                        if display_values
                                        else None
                                    ),
                                )

                            else:

                                st.caption(
                                    "Multiple evidence-supported values/ranges "
                                    "were observed:"
                                )

                                for display_value in display_values:
                                    presentation = describe_response_value(
                                        response_type,
                                        display_value,
                                    )
                                    st.markdown(
                                        f"- **{presentation['display_value']}**"
                                    )
                                    if presentation["plain_language"]:
                                        st.caption(
                                            presentation["plain_language"]
                                        )
                                    if (
                                        presentation["readable_raw_value"]
                                        != presentation["display_value"]
                                    ):
                                        st.caption(
                                            "Evidence value: "
                                            f"{presentation['readable_raw_value']}"
                                        )

                            if (
                                response.get(
                                    "evidence_score"
                                )
                                is not None
                            ):

                                st.caption(
                                    "Evidence score: "
                                    f"{response['evidence_score']}"
                                )

                    else:

                        no_prediction_reason = getattr(
                            simulation_result,
                            "no_prediction_reason",
                            None
                        )

                        if (
                            no_prediction_reason
                            ==
                            "no_matching_evidence"
                        ):

                            st.warning(
                                "No predicted responses matched the "
                                "supplied water conditions."
                            )

                        elif (
                            no_prediction_reason
                            ==
                            "below_evidence_threshold"
                        ):

                            st.warning(
                                "Matching response evidence was found, "
                                "but no predicted responses met the "
                                "selected evidence threshold."
                            )

                        else:

                            # Compatibility fallback for older result
                            # objects that do not expose a reason.
                            st.warning(
                                "No predicted responses matched the "
                                "supplied water conditions at the "
                                "selected evidence threshold."
                            )

            # =================================================
            # BIOLOGICAL CONDITION + VISUAL STATE
            # =================================================

            st.divider()

            show_simulation_biological_visual_state(
                simulation_result
            )

            st.divider()

            # =================================================
            # V2 ILLUSTRATIVE VISUAL SIMULATION
            # =================================================

            show_v2_visual_simulation(
                simulation_result,
                minimum_evidence_score
            )

            st.divider()

            # =================================================
            # TRANSITION SUMMARY
            # =================================================

            st.subheader(
                "Transition summary"
            )

            number_of_predictions = len(
                next_state.organism_state
            )

            col1, col2, col3 = st.columns(
                3
            )

            with col1:

                st.metric(
                    "Water factors",
                    len(
                        current_state.water_state
                    )
                )

            with col2:

                st.metric(
                    "Predicted responses",
                    number_of_predictions
                )

            with col3:

                st.metric(
                    "Evidence threshold",
                    minimum_evidence_score
                )

            # =================================================
            # SUPPORTING EVIDENCE
            # =================================================

            raw_predictions = (
                simulation_result
                .raw_result
                .get(
                    "predictions",
                    []
                )
            )

            # Only show predictions that survived the
            # simulator's evidence threshold.
            retained_predictions = [
                prediction
                for prediction in raw_predictions
                if (
                    prediction.get(
                        "evidence_score",
                        0.0
                    )
                    >=
                    minimum_evidence_score
                )
            ]

            if retained_predictions:

                st.subheader(
                    "Evidence behind the transition"
                )

                st.caption(
                    "These records explain why each response "
                    "was added to the predicted next state."
                )

                for (
                    prediction_index,
                    prediction
                ) in enumerate(
                    retained_predictions,
                    start=1
                ):

                    with st.expander(
                        (
                            f"{prediction_index}. "
                            f"{humanize(prediction['response_type'])}"
                            f" — "
                            f"{humanize_value(prediction['response_value'])}"
                        )
                    ):

                        st.write(
                            "**Evidence score:** "
                            f"{prediction['evidence_score']}"
                        )

                        st.write(
                            "**Independent observations:** "
                            f"{prediction['independent_observations']}"
                        )

                        supporting_evidence = (
                            prediction.get(
                                "supporting_evidence",
                                []
                            )
                        )

                        for (
                            evidence_index,
                            evidence
                        ) in enumerate(
                            supporting_evidence,
                            start=1
                        ):

                            st.markdown(
                                f"### Evidence "
                                f"{evidence_index}"
                            )

                            st.write(
                                "**Observation:** "
                                f"`{evidence['observation_id']}`"
                            )

                            if evidence.get(
                                "case_id"
                            ):

                                st.write(
                                    "**Case:** "
                                    f"`{evidence['case_id']}`"
                                )

                            if evidence.get(
                                "case_concentration"
                            ):

                                st.write(
                                    "**Case concentration:** "
                                    f"{evidence['case_concentration']}"
                                )

                            st.write(
                                "**Relationship type:** "
                                f"{humanize(evidence['relationship_type'])}"
                            )

                            st.write(
                                "**Evidence contribution:** "
                                f"{round(evidence['evidence_strength'], 3)}"
                            )

                            concentration_specificity = (
                                evidence.get(
                                    "concentration_specificity"
                                )
                            )

                            if (
                                concentration_specificity
                                and
                                concentration_specificity
                                !=
                                "not_requested"
                            ):

                                st.write(
                                    "**Concentration matching:** "
                                    f"{humanize(concentration_specificity)}"
                                )

                            if evidence.get(
                                "source_url"
                            ):

                                st.link_button(
                                    "Open scientific source",
                                    evidence[
                                        "source_url"
                                    ],
                                    key=(
                                        f"simulation_source_"
                                        f"{prediction_index}_"
                                        f"{evidence_index}_"
                                        f"{evidence['observation_id']}"
                                    )
                                )

                            # =================================
                            # MICROSCOPY MEDIA
                            # =================================

                            st.markdown(
                                "**Associated microscopy media:**"
                            )

                            show_observation_media(
                                evidence[
                                    "observation_id"
                                ],
                                context_key=(
                                    f"simulation_"
                                    f"{prediction_index}_"
                                    f"{evidence_index}"
                                )
                            )

                            if (
                                evidence_index
                                <
                                len(
                                    supporting_evidence
                                )
                            ):

                                st.divider()


else:
    render_model_information()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
    Microorganism World Model V2 — evidence-based prototype.

    Results are limited by the observations currently stored in the database.
    Associations should not automatically be interpreted as direct causal or
    diagnostic relationships.

    General reference videos are provided for visual context only and are not
    treated as direct evidence for individual world-model predictions.

    V1 visual-state images are evidence-backed microscopy analogues selected
    from the project dataset. V2 can additionally create a conservative
    illustrative transformation when a predicted response has an explicit
    supported static-image rule. These transformed images are not scientifically
    validated exact predictions of future morphology.
    """
)
