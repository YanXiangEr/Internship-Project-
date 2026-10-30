"""
train_world_model_v1_state.py

Trains the first biological-state component of the microorganism
world model.

Goal
----
Predict:

    healthy
    stressed

from:

    species
    + source-supported qualitative water/environmental condition

Dataset
-------
data/world_model_v1_dataset.csv

Why a simple model?
-------------------
There are currently only 45 supervised biological-state records.
A compact logistic-regression baseline is more appropriate than a
large neural network.

The richer morphology and visual-response fields are preserved for
the later description / visual-simulation stages rather than being
treated as 28 sparse classes.

Important
---------
- split_group is used to prevent group leakage during evaluation
- class weighting handles healthy/stressed imbalance
- quantitative EXT014 panels are NOT relabelled as "stressed"
- no unsupported biological labels are invented
- no images are modified
"""

import csv
import json
import pickle

from collections import Counter
from pathlib import Path

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INPUT_DATASET = (
    DATA_DIR
    / "world_model_v1_dataset.csv"
)

MODEL_PATH = (
    MODEL_DIR
    / "world_model_v1_state_classifier.pkl"
)

METRICS_PATH = (
    DATA_DIR
    / "world_model_v1_state_metrics.json"
)

PREDICTIONS_PATH = (
    DATA_DIR
    / "world_model_v1_state_cv_predictions.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 1234

TARGET_LABELS = [
    "healthy",
    "stressed",
]

MAX_CV_SPLITS = 5


# ============================================================
# HELPERS
# ============================================================

def clean(value):

    if value is None:
        return ""

    return str(value).strip()


def print_section(title):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print()


# ============================================================
# CSV
# ============================================================

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
# DATA PREPARATION
# ============================================================

def build_input_text(row):
    """
    Combine only INPUT-side environmental information.

    Do NOT include:
    - response labels
    - morphology labels
    - description targets
    - state labels

    because those are outputs and would cause target leakage.
    """

    components = []

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

    secondary_factor = clean(
        row.get(
            "secondary_water_factor"
        )
    )

    if qualitative:

        components.append(
            qualitative
        )

    if primary_factor:

        components.append(
            f"primary_factor={primary_factor}"
        )

    if secondary_factor:

        components.append(
            f"secondary_factor={secondary_factor}"
        )

    return " | ".join(
        components
    )


def prepare_records(rows):

    prepared = []

    excluded = []

    for row in rows:

        condition = clean(
            row.get(
                "condition_label"
            )
        ).lower()

        if condition not in TARGET_LABELS:

            excluded.append(
                row
            )

            continue

        species = clean(
            row.get(
                "species"
            )
        )

        input_text = build_input_text(
            row
        )

        input_lower = input_text.lower()

        for forbidden_text in [
            "condition=healthy",
            "condition=stressed",
            "healthy_reference",
            "stressed_reference",
        ]:

            if forbidden_text in input_lower:

                raise RuntimeError(
                    f"Target leakage detected in "
                    f"{row.get('record_id')}: "
                    f"{forbidden_text}"
                )

        split_group = clean(
            row.get(
                "split_group"
            )
        )

        if not split_group:

            # Conservative one-record group fallback.
            split_group = (
                "record_"
                + clean(
                    row.get(
                        "record_id"
                    )
                )
            )

        prepared.append({

            "record_id":
                clean(
                    row.get(
                        "record_id"
                    )
                ),

            "species":
                species,

            "input_text":
                input_text,

            "target":
                condition,

            "split_group":
                split_group,

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
        })

    return (
        prepared,
        excluded,
    )


# ============================================================
# PIPELINE
# ============================================================

def build_model():

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "species",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                ["species"],
            ),
            (
                "condition_text",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
                "input_text",
            ),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_SEED,
    )

    return Pipeline(
        steps=[
            (
                "preprocess",
                preprocessing,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


# ============================================================
# PANDAS-LIKE INPUT
# ============================================================

def records_to_columns(records):

    import pandas as pd

    return pd.DataFrame(
        [
            {
                "species":
                    row["species"],

                "input_text":
                    row["input_text"],
            }
            for row in records
        ]
    )


# ============================================================
# GROUP-AWARE CV
# ============================================================

def choose_cv_splits(records):

    groups = [
        row[
            "split_group"
        ]
        for row in records
    ]

    unique_groups = len(
        set(groups)
    )

    labels = [
        row[
            "target"
        ]
        for row in records
    ]

    label_counts = Counter(
        labels
    )

    smallest_class = min(
        label_counts.values()
    )

    n_splits = min(
        MAX_CV_SPLITS,
        unique_groups,
        smallest_class,
    )

    if n_splits < 2:

        raise RuntimeError(
            "Not enough independent groups/classes "
            "for grouped cross-validation."
        )

    return n_splits


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    true_labels,
    predicted_labels,
):

    return {

        "accuracy":
            float(
                accuracy_score(
                    true_labels,
                    predicted_labels,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    true_labels,
                    predicted_labels,
                )
            ),

        "macro_precision":
            float(
                precision_score(
                    true_labels,
                    predicted_labels,
                    labels=TARGET_LABELS,
                    average="macro",
                    zero_division=0,
                )
            ),

        "macro_recall":
            float(
                recall_score(
                    true_labels,
                    predicted_labels,
                    labels=TARGET_LABELS,
                    average="macro",
                    zero_division=0,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    true_labels,
                    predicted_labels,
                    labels=TARGET_LABELS,
                    average="macro",
                    zero_division=0,
                )
            ),
    }


# ============================================================
# MAIN
# ============================================================

def run():

    print_section(
        "TRAINING WORLD MODEL V1 BIOLOGICAL-STATE PREDICTOR"
    )

    print(
        "Prediction target:"
    )

    print(
        "  healthy vs stressed"
    )

    print()

    print(
        "Inputs:"
    )

    print(
        "  species"
    )

    print(
        "  qualitative water/environmental context"
    )

    print()

    print(
        "Evaluation:"
    )

    print(
        "  stratified group-aware cross-validation"
    )

    print()

    print(
        "No quantitative EXT014 exposure is "
        "automatically converted into a stress label."
    )

    # ========================================================
    # LOAD
    # ========================================================

    raw_rows = load_csv(
        INPUT_DATASET
    )

    (
        records,
        excluded,
    ) = prepare_records(
        raw_rows
    )

    print_section(
        "DATASET"
    )

    print(
        f"V1 dataset rows: "
        f"{len(raw_rows)}"
    )

    print(
        f"Rows used for state classifier: "
        f"{len(records)}"
    )

    print(
        f"Rows excluded due to unsupported "
        f"condition target: "
        f"{len(excluded)}"
    )

    target_counts = Counter(
        row[
            "target"
        ]
        for row in records
    )

    print()

    print(
        "Target counts:"
    )

    for label in TARGET_LABELS:

        print(
            f"  {label:<12} "
            f"{target_counts[label]}"
        )

    species_counts = Counter(
        row[
            "species"
        ]
        for row in records
    )

    print()

    print(
        "Species counts:"
    )

    for species, count in (
        species_counts.items()
    ):

        print(
            f"  {species:<24} "
            f"{count}"
        )

    unique_groups = {
        row[
            "split_group"
        ]
        for row in records
    }

    print()

    print(
        f"Independent split groups: "
        f"{len(unique_groups)}"
    )

    # ========================================================
    # CV
    # ========================================================

    n_splits = choose_cv_splits(
        records
    )

    print_section(
        "GROUP-AWARE CROSS-VALIDATION"
    )

    print(
        f"Folds: {n_splits}"
    )

    X = records_to_columns(
        records
    )

    y = np.array(
        [
            row[
                "target"
            ]
            for row in records
        ]
    )

    groups = np.array(
        [
            row[
                "split_group"
            ]
            for row in records
        ]
    )

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    all_true = []
    all_predicted = []

    prediction_rows = []

    fold_metrics = []

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        cv.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):

        train_records = [
            records[index]
            for index in train_indices
        ]

        validation_records = [
            records[index]
            for index in validation_indices
        ]

        train_groups = {
            row[
                "split_group"
            ]
            for row in train_records
        }

        validation_groups = {
            row[
                "split_group"
            ]
            for row in validation_records
        }

        overlap = (
            train_groups
            & validation_groups
        )

        if overlap:

            raise RuntimeError(
                "Group leakage detected in CV fold."
            )

        model = build_model()

        model.fit(
            X.iloc[
                train_indices
            ],
            y[
                train_indices
            ],
        )

        predictions = model.predict(
            X.iloc[
                validation_indices
            ]
        )

        probabilities = (
            model.predict_proba(
                X.iloc[
                    validation_indices
                ]
            )
        )

        fold_true = (
            y[
                validation_indices
            ]
        )

        metrics = calculate_metrics(
            fold_true,
            predictions,
        )

        metrics[
            "fold"
        ] = fold_number

        metrics[
            "train_records"
        ] = len(
            train_indices
        )

        metrics[
            "validation_records"
        ] = len(
            validation_indices
        )

        metrics[
            "train_groups"
        ] = len(
            train_groups
        )

        metrics[
            "validation_groups"
        ] = len(
            validation_groups
        )

        fold_metrics.append(
            metrics
        )

        print(
            f"Fold {fold_number}"
        )

        print(
            f"  Train records: "
            f"{len(train_indices)}"
        )

        print(
            f"  Validation records: "
            f"{len(validation_indices)}"
        )

        print(
            f"  Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"  Balanced accuracy: "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"  Macro F1: "
            f"{metrics['macro_f1']:.4f}"
        )

        print()

        class_order = list(
            model.named_steps[
                "classifier"
            ].classes_
        )

        for local_index, record_index in enumerate(
            validation_indices
        ):

            record = records[
                record_index
            ]

            probability_map = {
                class_order[class_index]:
                    float(
                        probabilities[
                            local_index,
                            class_index,
                        ]
                    )
                for class_index in range(
                    len(class_order)
                )
            }

            true_label = (
                y[
                    record_index
                ]
            )

            predicted_label = (
                predictions[
                    local_index
                ]
            )

            all_true.append(
                true_label
            )

            all_predicted.append(
                predicted_label
            )

            prediction_rows.append({

                "record_id":
                    record[
                        "record_id"
                    ],

                "species":
                    record[
                        "species"
                    ],

                "fold":
                    fold_number,

                "split_group":
                    record[
                        "split_group"
                    ],

                "input_text":
                    record[
                        "input_text"
                    ],

                "true_state":
                    true_label,

                "predicted_state":
                    predicted_label,

                "correct":
                    str(
                        true_label
                        == predicted_label
                    ),

                "prob_healthy":
                    round(
                        probability_map.get(
                            "healthy",
                            0.0,
                        ),
                        6,
                    ),

                "prob_stressed":
                    round(
                        probability_map.get(
                            "stressed",
                            0.0,
                        ),
                        6,
                    ),
            })

    # ========================================================
    # OVERALL CV METRICS
    # ========================================================

    overall_metrics = calculate_metrics(
        all_true,
        all_predicted,
    )

    matrix = confusion_matrix(
        all_true,
        all_predicted,
        labels=TARGET_LABELS,
    )

    print_section(
        "CROSS-VALIDATED PERFORMANCE"
    )

    print(
        f"Accuracy: "
        f"{overall_metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{overall_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro precision: "
        f"{overall_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro recall: "
        f"{overall_metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{overall_metrics['macro_f1']:.4f}"
    )

    print()

    print(
        "Confusion matrix:"
    )

    print(
        "                 Pred healthy   Pred stressed"
    )

    print(
        f"True healthy     "
        f"{matrix[0, 0]:>12} "
        f"{matrix[0, 1]:>15}"
    )

    print(
        f"True stressed    "
        f"{matrix[1, 0]:>12} "
        f"{matrix[1, 1]:>15}"
    )

    print()

    report = classification_report(
        all_true,
        all_predicted,
        labels=TARGET_LABELS,
        output_dict=True,
        zero_division=0,
    )

    print(
        "Per-state recall:"
    )

    print(
        f"  healthy:  "
        f"{report['healthy']['recall']:.4f}"
    )

    print(
        f"  stressed: "
        f"{report['stressed']['recall']:.4f}"
    )

    # ========================================================
    # TRAIN FINAL MODEL ON ALL SUPERVISED DATA
    # ========================================================

    print_section(
        "TRAINING FINAL V1 STATE MODEL"
    )

    final_model = build_model()

    final_model.fit(
        X,
        y,
    )

    with open(
        MODEL_PATH,
        "wb",
    ) as file:

        pickle.dump(
            final_model,
            file,
        )

    print(
        "[PASS] Final state classifier trained "
        "on all supervised v1 records."
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    save_csv(
        PREDICTIONS_PATH,
        prediction_rows,
    )

    metrics_output = {

        "model":
            "species_plus_condition_logistic_regression",

        "records":
            len(records),

        "independent_groups":
            len(unique_groups),

        "cv_folds":
            n_splits,

        "target_counts":
            dict(
                target_counts
            ),

        "species_counts":
            dict(
                species_counts
            ),

        "overall_cv_metrics":
            overall_metrics,

        "confusion_matrix": {
            "labels":
                TARGET_LABELS,

            "matrix":
                matrix.tolist(),
        },

        "per_class": {
            "healthy":
                report[
                    "healthy"
                ],

            "stressed":
                report[
                    "stressed"
                ],
        },

        "fold_metrics":
            fold_metrics,

        "important_limitation":
            (
                "Model currently learns from qualitative "
                "condition evidence only. Quantitative EXT014 "
                "exposures remain reserved for the visual-"
                "conditioning stage unless explicit state "
                "labels are available."
            ),
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics_output,
            file,
            indent=2,
        )

    # ========================================================
    # SAFETY
    # ========================================================

    print_section(
        "SAFETY CHECK"
    )

    print(
        "[PASS] Target/output fields were excluded "
        "from model inputs."
    )

    print(
        "[PASS] Group-aware evaluation prevented "
        "related observations from crossing folds."
    )

    print(
        "[PASS] Healthy/stressed imbalance was "
        "handled using class weighting."
    )

    print(
        "[PASS] EXT014 exposure was not automatically "
        "converted into a stressed label."
    )

    print(
        "[PASS] No unsupported numeric inputs were invented."
    )

    print(
        "[PASS] No image was modified."
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print_section(
        "WORLD MODEL V1 STATE TRAINING COMPLETE"
    )

    print(
        "Model:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        "Cross-validation predictions:"
    )

    print(
        PREDICTIONS_PATH
    )

    print()

    print(
        "Metrics:"
    )

    print(
        METRICS_PATH
    )

    print()

    print(
        "RESULT:"
    )

    print(
        "The first biological-state prediction "
        "component of the microorganism world "
        "model has been trained."
    )

    print()

    print(
        "Next step: connect this predicted state "
        "to evidence-backed morphology and generate "
        "the grounded description returned to the user."
    )


if __name__ == "__main__":
    run()