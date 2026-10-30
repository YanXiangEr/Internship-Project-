"""
build_v2_visual_conditioning_dataset.py

Build the compact V2 visual-conditioning dataset.

Goal
----
Prepare evidence-backed records for the later visual simulator:

    microorganism
    + water condition
    + biological state
    + observed morphology / responses
             ↓
       visual simulation

Inputs
------
data/world_model_v1_dataset.csv

Output
------
data/v2_visual_conditioning_dataset.csv

This script:
- reuses the images already prepared
- does not download images
- does not train a model
- does not generate images
- does not invent missing morphology
"""

import csv
from collections import Counter
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

INPUT_DATASET = (
    DATA_DIR
    / "world_model_v1_dataset.csv"
)

OUTPUT_DATASET = (
    DATA_DIR
    / "v2_visual_conditioning_dataset.csv"
)


# ============================================================
# HELPERS
# ============================================================

def clean(value):

    if value is None:
        return ""

    return str(value).strip()


def as_bool(value):

    return clean(value).lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def print_section(title):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print()


def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        return list(
            csv.DictReader(file)
        )


def save_csv(path, rows):

    if not rows:
        return

    fields = []

    for row in rows:

        for field in row.keys():

            if field not in fields:
                fields.append(field)

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# TEXT BUILDERS
# ============================================================

def build_water_condition(row):

    parts = []

    qualitative = clean(
        row.get(
            "qualitative_condition"
        )
    )

    primary_factor = clean(
        row.get(
            "primary_water_factor"
        )
    )

    primary_value = clean(
        row.get(
            "primary_concentration_value"
        )
    )

    primary_unit = clean(
        row.get(
            "primary_concentration_unit"
        )
    )

    secondary_factor = clean(
        row.get(
            "secondary_water_factor"
        )
    )

    secondary_value = clean(
        row.get(
            "secondary_concentration_value"
        )
    )

    secondary_unit = clean(
        row.get(
            "secondary_concentration_unit"
        )
    )

    exposure_time = clean(
        row.get(
            "exposure_time"
        )
    )

    exposure_unit = clean(
        row.get(
            "exposure_unit"
        )
    )

    if qualitative:
        parts.append(
            qualitative
        )

    if primary_factor:

        factor_text = (
            primary_factor
        )

        if primary_value:

            factor_text += (
                f" {primary_value}"
            )

            if primary_unit:

                factor_text += (
                    f" {primary_unit}"
                )

        parts.append(
            factor_text
        )

    if secondary_factor:

        secondary_text = (
            secondary_factor
        )

        if secondary_value:

            secondary_text += (
                f" {secondary_value}"
            )

            if secondary_unit:

                secondary_text += (
                    f" {secondary_unit}"
                )

        parts.append(
            secondary_text
        )

    if exposure_time:

        exposure_text = (
            f"exposure {exposure_time}"
        )

        if exposure_unit:

            exposure_text += (
                f" {exposure_unit}"
            )

        parts.append(
            exposure_text
        )

    # Remove duplicates while preserving order.

    unique_parts = []

    for part in parts:

        if (
            part
            and
            part not in unique_parts
        ):

            unique_parts.append(
                part
            )

    return " | ".join(
        unique_parts
    )


def build_response_description(row):

    parts = []

    for field in [
        "response_labels",
        "visual_phenotype",
        "visual_feature_1",
        "visual_feature_2",
    ]:

        value = clean(
            row.get(
                field
            )
        )

        if (
            value
            and
            value not in parts
        ):

            parts.append(
                value
            )

    return " | ".join(
        parts
    )


def build_visual_prompt(row):

    """
    Create a future visual-conditioning prompt using only
    source-supported metadata.

    This is not sent to an image model yet.
    """

    species = clean(
        row.get(
            "species"
        )
    )

    condition = clean(
        row.get(
            "condition_label"
        )
    )

    water_condition = (
        build_water_condition(
            row
        )
    )

    morphology = (
        build_response_description(
            row
        )
    )

    parts = [
        f"microscopy image of {species}"
    ]

    if condition:

        parts.append(
            f"biological state: {condition}"
        )

    if water_condition:

        parts.append(
            f"environment: {water_condition}"
        )

    if morphology:

        parts.append(
            f"observed morphology: {morphology}"
        )

    return ". ".join(
        parts
    )


# ============================================================
# INCLUDE POLICY
# ============================================================

def include_record(row):

    # V2 requires a real available image.

    if not as_bool(
        row.get(
            "image_available"
        )
    ):

        return False

    local_path = clean(
        row.get(
            "local_path"
        )
    )

    if not local_path:

        return False

    path = Path(
        local_path
    )

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            / path
        )

    if not path.exists():

        return False

    # Need at least a known biological state.

    condition = clean(
        row.get(
            "condition_label"
        )
    )

    if condition not in {
        "healthy",
        "stressed",
    }:

        return False

    return True


# ============================================================
# NORMALISE
# ============================================================

def normalise_record(row):

    local_path = Path(
        clean(
            row.get(
                "local_path"
            )
        )
    )

    if not local_path.is_absolute():

        local_path = (
            PROJECT_ROOT
            / local_path
        )

    return {

        "record_id":
            clean(
                row.get(
                    "record_id"
                )
            ),

        "species":
            clean(
                row.get(
                    "species"
                )
            ),

        "biological_state":
            clean(
                row.get(
                    "condition_label"
                )
            ),

        "water_condition":
            build_water_condition(
                row
            ),

        "primary_water_factor":
            clean(
                row.get(
                    "primary_water_factor"
                )
            ),

        "primary_concentration_value":
            clean(
                row.get(
                    "primary_concentration_value"
                )
            ),

        "primary_concentration_unit":
            clean(
                row.get(
                    "primary_concentration_unit"
                )
            ),

        "secondary_water_factor":
            clean(
                row.get(
                    "secondary_water_factor"
                )
            ),

        "secondary_concentration_value":
            clean(
                row.get(
                    "secondary_concentration_value"
                )
            ),

        "secondary_concentration_unit":
            clean(
                row.get(
                    "secondary_concentration_unit"
                )
            ),

        "exposure_time":
            clean(
                row.get(
                    "exposure_time"
                )
            ),

        "exposure_unit":
            clean(
                row.get(
                    "exposure_unit"
                )
            ),

        "response_types":
            clean(
                row.get(
                    "response_types"
                )
            ),

        "response_labels":
            clean(
                row.get(
                    "response_labels"
                )
            ),

        "visual_phenotype":
            clean(
                row.get(
                    "visual_phenotype"
                )
            ),

        "visual_feature_1":
            clean(
                row.get(
                    "visual_feature_1"
                )
            ),

        "visual_feature_2":
            clean(
                row.get(
                    "visual_feature_2"
                )
            ),

        "visual_description":
            build_response_description(
                row
            ),

        "visual_prompt":
            build_visual_prompt(
                row
            ),

        "image_path":
            str(
                local_path
            ),

        "sha256":
            clean(
                row.get(
                    "sha256"
                )
            ),

        "split_group":
            clean(
                row.get(
                    "split_group"
                )
            ),

        "evidence_type":
            clean(
                row.get(
                    "evidence_type"
                )
            ),

        "dataset_origin":
            clean(
                row.get(
                    "dataset_origin"
                )
            ),

        "license":
            clean(
                row.get(
                    "license"
                )
            ),

        "provenance_url":
            clean(
                row.get(
                    "provenance_url"
                )
            ),
    }


# ============================================================
# MAIN
# ============================================================

def run():

    print_section(
        "BUILDING V2 VISUAL-CONDITIONING DATASET"
    )

    print(
        "V2 target:"
    )

    print()

    print(
        "  species + water quality"
    )

    print(
        "          ↓"
    )

    print(
        "  biological state / responses"
    )

    print(
        "          ↓"
    )

    print(
        "  visual appearance"
    )

    print()

    print(
        "No model will be trained."
    )

    print(
        "No new images will be downloaded."
    )

    print(
        "No synthetic images will be generated."
    )

    # ========================================================
    # LOAD
    # ========================================================

    rows = load_csv(
        INPUT_DATASET
    )

    print()

    print(
        f"V1 records loaded: "
        f"{len(rows)}"
    )

    usable_rows = [
        row
        for row in rows
        if include_record(
            row
        )
    ]

    output_rows = [
        normalise_record(
            row
        )
        for row in usable_rows
    ]

    # ========================================================
    # VALIDATION
    # ========================================================

    print_section(
        "V2 DATASET VALIDATION"
    )

    ids = [
        row[
            "record_id"
        ]
        for row in output_rows
    ]

    if len(ids) != len(
        set(ids)
    ):

        print(
            "[FAIL] Duplicate record IDs detected."
        )

        return

    print(
        "[PASS] Record IDs are unique."
    )

    missing_images = [
        row
        for row in output_rows
        if not Path(
            row[
                "image_path"
            ]
        ).exists()
    ]

    if missing_images:

        print(
            "[FAIL] Missing image files detected."
        )

        return

    print(
        "[PASS] Every V2 record has a readable local image path."
    )

    print(
        "[PASS] Every V2 record has a healthy/stressed state."
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print_section(
        "V2 VISUAL DATASET SUMMARY"
    )

    print(
        f"Usable visual-conditioning records: "
        f"{len(output_rows)}"
    )

    species_counts = Counter(
        row[
            "species"
        ]
        for row in output_rows
    )

    print()

    print(
        "Species:"
    )

    for species, count in (
        species_counts.items()
    ):

        print(
            f"  {species:<25} "
            f"{count}"
        )

    state_counts = Counter(
        row[
            "biological_state"
        ]
        for row in output_rows
    )

    print()

    print(
        "Biological states:"
    )

    for state, count in (
        state_counts.items()
    ):

        print(
            f"  {state:<12} "
            f"{count}"
        )

    origin_counts = Counter(
        row[
            "dataset_origin"
        ]
        for row in output_rows
    )

    print()

    print(
        "Dataset origin:"
    )

    for origin, count in (
        origin_counts.items()
    ):

        print(
            f"  {origin:<15} "
            f"{count}"
        )

    # ========================================================
    # VISUAL PROMPT EXAMPLES
    # ========================================================

    print_section(
        "EXAMPLE VISUAL-CONDITIONING RECORDS"
    )

    for row in (
        output_rows[:5]
    ):

        print(
            row[
                "record_id"
            ]
        )

        print(
            "  "
            +
            row[
                "visual_prompt"
            ]
        )

        print()

    # ========================================================
    # SAVE
    # ========================================================

    save_csv(
        OUTPUT_DATASET,
        output_rows,
    )

    print_section(
        "SAFETY CHECK"
    )

    print(
        "[PASS] Existing images were reused."
    )

    print(
        "[PASS] Missing morphology was not invented."
    )

    print(
        "[PASS] Water conditions remain tied to their "
        "original evidence records."
    )

    print(
        "[PASS] No image generator was trained."
    )

    print(
        "[PASS] No image was modified."
    )

    print_section(
        "V2 VISUAL-CONDITIONING DATASET COMPLETE"
    )

    print(
        "Output:"
    )

    print(
        OUTPUT_DATASET
    )

    print()

    print(
        "RESULT:"
    )

    print(
        "The existing microscopy evidence is now organised "
        "for V2 condition-dependent visual simulation."
    )

    print()

    print(
        "Next step: build a lightweight visual simulator "
        "that transforms/selects a reference appearance using "
        "the predicted biological state and morphology."
    )


if __name__ == "__main__":
    run()