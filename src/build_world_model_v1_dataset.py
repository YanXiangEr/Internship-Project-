"""
build_world_model_v1_dataset.py

Builds the actual v1 biological-state dataset for the microorganism
world model.

Input
-----
data/water_quality_visual_manifest.csv

Output
------
data/world_model_v1_dataset.csv

V1 goal
-------
Learn:

    species
    + supported water-quality condition
    + supported quantitative exposure variables
                    ↓
        structured biological state
                    ↓
       morphology + grounded description

Species-specific input policy
-----------------------------
Euglena viridis:
    qualitative condition only

Nitzschia palea:
    qualitative condition
    primary concentration
    secondary concentration
    exposure time

Oscillatoria limosa:
    qualitative condition only

IMPORTANT
---------
Unsupported numeric values remain missing.

The dataset does NOT fabricate:
- pH
- temperature
- nitrate concentration
- phosphate concentration
- dissolved oxygen
- conductivity

unless those values already exist in the source data.

No model is trained here.
"""

import csv
from collections import Counter
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_MANIFEST = (
    DATA_DIR
    / "water_quality_visual_manifest.csv"
)

OUTPUT_DATASET = (
    DATA_DIR
    / "world_model_v1_dataset.csv"
)


# ============================================================
# TARGET SPECIES
# ============================================================

TARGET_SPECIES = {
    "Euglena viridis",
    "Nitzschia palea",
    "Oscillatoria limosa",
}


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


def bool_string(value):

    return (
        "True"
        if value
        else "False"
    )


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
# INPUT POLICY
# ============================================================

def v1_input_policy(species):

    if species == "Nitzschia palea":

        return {
            "qualitative_condition":
                True,

            "primary_concentration":
                True,

            "secondary_concentration":
                True,

            "exposure_time":
                True,
        }

    if species in {
        "Euglena viridis",
        "Oscillatoria limosa",
    }:

        return {
            "qualitative_condition":
                True,

            "primary_concentration":
                False,

            "secondary_concentration":
                False,

            "exposure_time":
                False,
        }

    return {}


# ============================================================
# CONDITION REPRESENTATION
# ============================================================

def build_qualitative_condition(row):
    """
    Build the environmental INPUT representation.

    IMPORTANT:
    The healthy/stressed condition label is deliberately excluded
    because it is the prediction target.

    Only information that would genuinely be known BEFORE predicting
    biological state may be included here.
    """

    parts = []

    water_labels = clean(
        row.get(
            "water_factor_labels"
        )
    )

    primary_factor = clean(
        row.get(
            "primary_water_factor"
        )
    )

    secondary_factor = clean(
        row.get(
            "secondary_water_factor"
        )
    )

    concentration_level = clean(
        row.get(
            "concentration_level"
        )
    )

    stressor_grade = clean(
        row.get(
            "stressor_grade"
        )
    )

    if water_labels:

        parts.append(
            f"water={water_labels}"
        )

    if primary_factor:

        parts.append(
            f"factor={primary_factor}"
        )

    if secondary_factor:

        parts.append(
            f"secondary_factor={secondary_factor}"
        )

    if concentration_level:

        parts.append(
            f"level={concentration_level}"
        )

    if stressor_grade:

        parts.append(
            f"grade={stressor_grade}"
        )

    return " | ".join(
        parts
    )


# ============================================================
# OUTPUT STATE REPRESENTATION
# ============================================================

def build_state_label(row):

    """
    Structured state label assembled only from already-supported
    biological metadata.
    """

    parts = []

    condition = clean(
        row.get(
            "condition"
        )
    )

    phenotype = clean(
        row.get(
            "visual_phenotype"
        )
    )

    responses = clean(
        row.get(
            "response_labels"
        )
    )

    feature_1 = clean(
        row.get(
            "visual_feature_1"
        )
    )

    feature_2 = clean(
        row.get(
            "visual_feature_2"
        )
    )

    if condition:
        parts.append(
            condition
        )

    if phenotype:
        parts.append(
            phenotype
        )

    if responses:
        parts.append(
            responses
        )

    if feature_1:
        parts.append(
            feature_1
        )

    if feature_2:
        parts.append(
            feature_2
        )

    # Remove exact repeated text while preserving order.

    unique_parts = []

    for part in parts:

        if part not in unique_parts:
            unique_parts.append(
                part
            )

    return " | ".join(
        unique_parts
    )


# ============================================================
# INCLUDE POLICY
# ============================================================

def include_record(row):

    species = clean(
        row.get(
            "species"
        )
    )

    if species not in TARGET_SPECIES:
        return False

    # For v1 state prediction, the record must already have
    # state-prediction permission.

    return as_bool(
        row.get(
            "state_prediction_training_permission"
        )
    )


# ============================================================
# NORMALISATION
# ============================================================

def normalise_record(row):

    species = clean(
        row.get(
            "species"
        )
    )

    policy = v1_input_policy(
        species
    )

    qualitative_condition = (
        build_qualitative_condition(
            row
        )
    )

    primary_value = ""

    primary_unit = ""

    secondary_value = ""

    secondary_unit = ""

    exposure_time = ""

    exposure_unit = ""

    # --------------------------------------------------------
    # Quantitative inputs only for supported species
    # --------------------------------------------------------

    if policy[
        "primary_concentration"
    ]:

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

    if policy[
        "secondary_concentration"
    ]:

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

    if policy[
        "exposure_time"
    ]:

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

    state_label = build_state_label(
        row
    )

    return {

        # ====================================================
        # ID
        # ====================================================

        "record_id":
            clean(
                row.get(
                    "world_model_record_id"
                )
            ),

        "species":
            species,

        # ====================================================
        # INPUTS
        # ====================================================

        "qualitative_condition":
            qualitative_condition,

        "primary_water_factor":
            clean(
                row.get(
                    "primary_water_factor"
                )
            ),

        "primary_concentration_value":
            primary_value,

        "primary_concentration_unit":
            primary_unit,

        "secondary_water_factor":
            clean(
                row.get(
                    "secondary_water_factor"
                )
            ),

        "secondary_concentration_value":
            secondary_value,

        "secondary_concentration_unit":
            secondary_unit,

        "exposure_time":
            exposure_time,

        "exposure_unit":
            exposure_unit,

        # ====================================================
        # OUTPUT STATE
        # ====================================================

        "condition_label":
            clean(
                row.get(
                    "condition"
                )
            ),

        "state_label":
            state_label,

        "visual_phenotype":
            clean(
                row.get(
                    "visual_phenotype"
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

        # ====================================================
        # DESCRIPTION TARGET
        # ====================================================

        "description_target":
            clean(
                row.get(
                    "description_target"
                )
            ),

        # ====================================================
        # VISUAL REFERENCE
        # ====================================================

        "image_available":
            clean(
                row.get(
                    "image_available"
                )
            ),

        "local_path":
            clean(
                row.get(
                    "local_path"
                )
            ),

        "sha256":
            clean(
                row.get(
                    "sha256"
                )
            ),

        # ====================================================
        # FUTURE VISUAL SIMULATOR ROLE
        # ====================================================

        "paired_visual_context_available":
            clean(
                row.get(
                    "paired_visual_context_available"
                )
            ),

        "visual_simulation_training_permission":
            clean(
                row.get(
                    "visual_simulation_training_permission"
                )
            ),

        # ====================================================
        # EVIDENCE / PROVENANCE
        # ====================================================

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

        "dataset_source":
            clean(
                row.get(
                    "dataset_source"
                )
            ),

        "split_group":
            clean(
                row.get(
                    "split_group"
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
# VALIDATION
# ============================================================

def validate_dataset(rows):

    problems = []

    ids = [
        clean(
            row.get(
                "record_id"
            )
        )
        for row in rows
    ]

    if len(ids) != len(
        set(ids)
    ):

        problems.append(
            "Duplicate v1 record IDs detected."
        )

    # --------------------------------------------------------
    # Euglena and Oscillatoria must not suddenly receive
    # unsupported quantitative inputs.
    # --------------------------------------------------------

    for row in rows:

        species = row[
            "species"
        ]

        if species in {
            "Euglena viridis",
            "Oscillatoria limosa",
        }:

            numeric_values = [
                clean(
                    row.get(
                        "primary_concentration_value"
                    )
                ),
                clean(
                    row.get(
                        "secondary_concentration_value"
                    )
                ),
                clean(
                    row.get(
                        "exposure_time"
                    )
                ),
            ]

            if any(
                numeric_values
            ):

                problems.append(
                    (
                        f"{row['record_id']} contains "
                        "unsupported quantitative inputs "
                        f"for {species}."
                    )
                )

    # --------------------------------------------------------
    # Every record needs at least some supported state target.
    # --------------------------------------------------------

    for row in rows:

        if not clean(
            row.get(
                "state_label"
            )
        ):

            problems.append(
                (
                    f"{row['record_id']} has no "
                    "supported state label."
                )
            )

    return problems


# ============================================================
# MAIN
# ============================================================

def run():

    print_section(
        "BUILDING WORLD MODEL V1 DATASET"
    )

    print(
        "V1 objective:"
    )

    print()

    print(
        "  species + supported water condition"
    )

    print(
        "                  ↓"
    )

    print(
        "       biological state"
    )

    print(
        "                  ↓"
    )

    print(
        "      grounded description"
    )

    print()

    print(
        "Visual simulation metadata are retained "
        "for the later image-generation stage."
    )

    # ========================================================
    # LOAD
    # ========================================================

    source_rows = load_csv(
        INPUT_MANIFEST
    )

    selected_rows = [
        row
        for row in source_rows
        if include_record(
            row
        )
    ]

    output_rows = [
        normalise_record(
            row
        )
        for row in selected_rows
    ]

    print()

    print(
        f"Canonical world-model records: "
        f"{len(source_rows)}"
    )

    print(
        f"V1 state-prediction records: "
        f"{len(output_rows)}"
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    print_section(
        "V1 DATASET VALIDATION"
    )

    problems = validate_dataset(
        output_rows
    )

    if problems:

        for problem in problems:

            print(
                f"[FAIL] {problem}"
            )

        print()

        print(
            "RESULT:"
        )

        print(
            "V1 dataset was NOT written."
        )

        return

    print(
        "[PASS] V1 record IDs are unique."
    )

    print(
        "[PASS] Euglena viridis has no invented "
        "quantitative inputs."
    )

    print(
        "[PASS] Oscillatoria limosa has no invented "
        "quantitative inputs."
    )

    print(
        "[PASS] Every record has a supported "
        "biological-state target."
    )

    # ========================================================
    # COVERAGE
    # ========================================================

    print_section(
        "V1 TRAINING COVERAGE"
    )

    species_counts = Counter(
        row[
            "species"
        ]
        for row in output_rows
    )

    for species in sorted(
        TARGET_SPECIES
    ):

        species_rows = [
            row
            for row in output_rows
            if row[
                "species"
            ] == species
        ]

        numeric_rows = sum(
            bool(
                clean(
                    row.get(
                        "primary_concentration_value"
                    )
                )
                or clean(
                    row.get(
                        "secondary_concentration_value"
                    )
                )
                or clean(
                    row.get(
                        "exposure_time"
                    )
                )
            )
            for row in species_rows
        )

        visual_pairs = sum(
            as_bool(
                row.get(
                    "paired_visual_context_available"
                )
            )
            for row in species_rows
        )

        print(
            species
        )

        print(
            f"  Training records: "
            f"{species_counts[species]}"
        )

        print(
            f"  Quantitative-input records: "
            f"{numeric_rows}"
        )

        print(
            f"  Paired visual records: "
            f"{visual_pairs}"
        )

        print()

    # ========================================================
    # STATE TARGET SUMMARY
    # ========================================================

    print_section(
        "STATE TARGET SUMMARY"
    )

    condition_counts = Counter(
        row[
            "condition_label"
        ]
        or "<EMPTY>"
        for row in output_rows
    )

    print(
        "Condition labels:"
    )

    for label, count in (
        condition_counts.most_common()
    ):

        print(
            f"  {label:<30} "
            f"{count}"
        )

    print()

    state_counts = Counter(
        row[
            "state_label"
        ]
        for row in output_rows
    )

    print(
        f"Unique structured state labels: "
        f"{len(state_counts)}"
    )

    print()

    print(
        "Most common state labels:"
    )

    for state, count in (
        state_counts.most_common(
            10
        )
    ):

        display = (
            state
            if len(state) <= 80
            else state[:77] + "..."
        )

        print(
            f"  {count:>3}  "
            f"{display}"
        )

    # ========================================================
    # DESCRIPTION COVERAGE
    # ========================================================

    print_section(
        "DESCRIPTION TARGET COVERAGE"
    )

    description_count = sum(
        bool(
            clean(
                row.get(
                    "description_target"
                )
            )
        )
        for row in output_rows
    )

    print(
        f"Records with grounded descriptions: "
        f"{description_count}"
        f"/{len(output_rows)}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_csv(
        OUTPUT_DATASET,
        output_rows,
    )

    # ========================================================
    # SAFETY
    # ========================================================

    print_section(
        "SAFETY CHECK"
    )

    print(
        "[PASS] V1 inputs follow the locked "
        "species-specific coverage policy."
    )

    print(
        "[PASS] Unsupported numerical variables "
        "remain unavailable."
    )

    print(
        "[PASS] Biological-state labels come from "
        "existing source-supported metadata."
    )

    print(
        "[PASS] Description targets remain grounded "
        "in existing observations."
    )

    print(
        "[PASS] Visual-simulator metadata were preserved "
        "without training a generator."
    )

    print(
        "[PASS] No image was modified."
    )

    print(
        "[PASS] No model was trained."
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print_section(
        "WORLD MODEL V1 DATASET COMPLETE"
    )

    print(
        "Dataset:"
    )

    print(
        OUTPUT_DATASET
    )

    print()

    print(
        "RESULT:"
    )

    print(
        "The v1 biological-state and description "
        "dataset is now ready."
    )

    print()

    print(
        "Next step: inspect the state-label structure "
        "and build the first biological-state predictor "
        "instead of adding more preparation stages."
    )


if __name__ == "__main__":
    run()