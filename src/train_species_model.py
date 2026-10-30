"""
train_species_model.py

Trains the first microorganism species-recognition baseline.

Model:
    Pretrained ResNet18
    Frozen backbone
    3-class classifier head

Dataset:
    data/species_training_manifest.csv

Classes:
    0 = Euglena viridis
    1 = Nitzschia palea
    2 = Oscillatoria limosa

Training features:
- class-weighted cross-entropy
- validation monitoring
- accuracy
- macro precision
- macro recall
- macro F1
- per-class recall
- best-model checkpointing
- deterministic random seeds
- final test evaluation

This is the first actual training stage.
"""

import csv
import json
import random
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

from species_dataset import (
    SPECIES_CLASSES,
    SPECIES_TO_INDEX,
    build_dataloaders,
)

from species_model import (
    build_species_model,
    get_device,
)


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

BEST_MODEL_PATH = (
    MODEL_DIR
    / "species_resnet18_unique_best.pt"
)

HISTORY_PATH = (
    DATA_DIR
    / "species_training_history_unique.csv"
)

METRICS_PATH = (
    DATA_DIR
    / "species_test_metrics_unique.json"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

SEED = 1234

BATCH_SIZE = 8

NUM_WORKERS = 0

EPOCHS = 20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

FREEZE_BACKBONE = True

EARLY_STOPPING_PATIENCE = 6


# ============================================================
# RANDOM SEEDS
# ============================================================

def set_seed(seed):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# DISPLAY
# ============================================================

def print_section(title):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print()


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    train_loader,
    device,
):

    counts = torch.zeros(
        len(SPECIES_CLASSES),
        dtype=torch.float32,
    )

    for row in (
        train_loader.dataset.rows
    ):

        species = row[
            "species"
        ].strip()

        class_index = (
            SPECIES_TO_INDEX[
                species
            ]
        )

        counts[
            class_index
        ] += 1

    total = counts.sum()

    num_classes = len(
        SPECIES_CLASSES
    )

    weights = (
        total
        / (
            num_classes
            * counts
        )
    )

    return (
        counts,
        weights.to(device),
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def build_confusion_matrix(
    true_labels,
    predicted_labels,
    num_classes,
):

    matrix = np.zeros(
        (
            num_classes,
            num_classes,
        ),
        dtype=np.int64,
    )

    for true_label, predicted_label in zip(
        true_labels,
        predicted_labels,
    ):

        matrix[
            true_label,
            predicted_label,
        ] += 1

    return matrix


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    true_labels,
    predicted_labels,
):

    num_classes = len(
        SPECIES_CLASSES
    )

    confusion = (
        build_confusion_matrix(
            true_labels,
            predicted_labels,
            num_classes,
        )
    )

    total = confusion.sum()

    correct = np.trace(
        confusion
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    precisions = []
    recalls = []
    f1_scores = []

    per_class = {}

    for class_index, species in enumerate(
        SPECIES_CLASSES
    ):

        tp = confusion[
            class_index,
            class_index,
        ]

        fp = (
            confusion[
                :,
                class_index,
            ].sum()
            - tp
        )

        fn = (
            confusion[
                class_index,
                :,
            ].sum()
            - tp
        )

        precision = (
            tp
            / (
                tp + fp
            )
            if (
                tp + fp
            ) > 0
            else 0.0
        )

        recall = (
            tp
            / (
                tp + fn
            )
            if (
                tp + fn
            ) > 0
            else 0.0
        )

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            ) > 0
            else 0.0
        )

        precisions.append(
            precision
        )

        recalls.append(
            recall
        )

        f1_scores.append(
            f1
        )

        per_class[
            species
        ] = {
            "precision":
                float(
                    precision
                ),

            "recall":
                float(
                    recall
                ),

            "f1":
                float(
                    f1
                ),

            "support":
                int(
                    confusion[
                        class_index,
                        :,
                    ].sum()
                ),
        }

    return {
        "accuracy":
            float(
                accuracy
            ),

        "macro_precision":
            float(
                np.mean(
                    precisions
                )
            ),

        "macro_recall":
            float(
                np.mean(
                    recalls
                )
            ),

        "macro_f1":
            float(
                np.mean(
                    f1_scores
                )
            ),

        "confusion_matrix":
            confusion,

        "per_class":
            per_class,
    }


# ============================================================
# ONE TRAINING EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    running_loss = 0.0

    true_labels = []
    predicted_labels = []

    for batch in loader:

        images = (
            batch[
                "image"
            ]
            .to(
                device
            )
        )

        labels = (
            batch[
                "label"
            ]
            .to(
                device
            )
        )

        optimizer.zero_grad()

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        true_labels.extend(
            labels
            .detach()
            .cpu()
            .tolist()
        )

        predicted_labels.extend(
            predictions
            .detach()
            .cpu()
            .tolist()
        )

    average_loss = (
        running_loss
        / len(
            loader.dataset
        )
    )

    metrics = calculate_metrics(
        true_labels,
        predicted_labels,
    )

    metrics[
        "loss"
    ] = average_loss

    return metrics


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    true_labels = []
    predicted_labels = []

    with torch.no_grad():

        for batch in loader:

            images = (
                batch[
                    "image"
                ]
                .to(
                    device
                )
            )

            labels = (
                batch[
                    "label"
                ]
                .to(
                    device
                )
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels,
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            true_labels.extend(
                labels
                .detach()
                .cpu()
                .tolist()
            )

            predicted_labels.extend(
                predictions
                .detach()
                .cpu()
                .tolist()
            )

    average_loss = (
        running_loss
        / len(
            loader.dataset
        )
    )

    metrics = calculate_metrics(
        true_labels,
        predicted_labels,
    )

    metrics[
        "loss"
    ] = average_loss

    return metrics


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    epoch,
    validation_metrics,
    class_weights,
):

    checkpoint = {

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "species_classes":
            SPECIES_CLASSES,

        "species_to_index":
            SPECIES_TO_INDEX,

        "validation_accuracy":
            validation_metrics[
                "accuracy"
            ],

        "validation_macro_f1":
            validation_metrics[
                "macro_f1"
            ],

        "validation_loss":
            validation_metrics[
                "loss"
            ],

        "class_weights":
            class_weights
            .detach()
            .cpu()
            .tolist(),

        "freeze_backbone":
            FREEZE_BACKBONE,

        "image_size":
            224,
    }

    torch.save(
        checkpoint,
        BEST_MODEL_PATH,
    )


# ============================================================
# TRAINING HISTORY
# ============================================================

def save_history(rows):

    if not rows:
        return

    fields = list(
        rows[0].keys()
    )

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# CONFUSION MATRIX DISPLAY
# ============================================================

def print_confusion_matrix(
    matrix,
):

    print(
        f"{'True / Pred':<25}",
        end="",
    )

    for species in SPECIES_CLASSES:

        print(
            f"{species[:12]:>14}",
            end="",
        )

    print()

    for row_index, species in enumerate(
        SPECIES_CLASSES
    ):

        print(
            f"{species:<25}",
            end="",
        )

        for column_index in range(
            len(
                SPECIES_CLASSES
            )
        ):

            print(
                f"{matrix[row_index, column_index]:>14}",
                end="",
            )

        print()


# ============================================================
# MAIN
# ============================================================

def run():

    set_seed(
        SEED
    )

    print_section(
        "SPECIES-RECOGNITION BASELINE TRAINING"
    )

    device = get_device()

    print(
        f"Device: {device}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Weight decay: {WEIGHT_DECAY}"
    )

    print(
        f"Freeze backbone: {FREEZE_BACKBONE}"
    )

    print(
        f"Early-stopping patience: "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print_section(
        "DATA LOADERS"
    )

    loaders = build_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    train_loader = (
        loaders[
            "train"
        ]
    )

    validation_loader = (
        loaders[
            "validation"
        ]
    )

    test_loader = (
        loaders[
            "test"
        ]
    )

    print(
        f"Train:      "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation: "
        f"{len(validation_loader.dataset)}"
    )

    print(
        f"Test:       "
        f"{len(test_loader.dataset)}"
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    print_section(
        "CLASS-IMBALANCE HANDLING"
    )

    (
        class_counts,
        class_weights,
    ) = calculate_class_weights(
        train_loader,
        device,
    )

    for class_index, species in enumerate(
        SPECIES_CLASSES
    ):

        print(
            f"{species:<24}"
            f" count={int(class_counts[class_index].item()):<3}"
            f" weight={class_weights[class_index].item():.4f}"
        )

    # ========================================================
    # MODEL
    # ========================================================

    print_section(
        "MODEL"
    )

    model = build_species_model(
        freeze_backbone=FREEZE_BACKBONE,
    )

    model = model.to(
        device
    )

    trainable_parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    print(
        "Architecture: pretrained ResNet18"
    )

    print(
        f"Trainable tensors: "
        f"{len(trainable_parameters)}"
    )

    # ========================================================
    # LOSS / OPTIMIZER
    # ========================================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print_section(
        "TRAINING"
    )

    history = []

    best_macro_f1 = -1.0

    best_epoch = 0

    epochs_without_improvement = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        validation_metrics = (
            evaluate(
                model,
                validation_loader,
                criterion,
                device,
            )
        )

        history.append({

            "epoch":
                epoch,

            "train_loss":
                train_metrics[
                    "loss"
                ],

            "train_accuracy":
                train_metrics[
                    "accuracy"
                ],

            "train_macro_f1":
                train_metrics[
                    "macro_f1"
                ],

            "validation_loss":
                validation_metrics[
                    "loss"
                ],

            "validation_accuracy":
                validation_metrics[
                    "accuracy"
                ],

            "validation_macro_f1":
                validation_metrics[
                    "macro_f1"
                ],
        })

        print(
            f"Epoch "
            f"{epoch:02d}/{EPOCHS}"
        )

        print(
            f"  Train      "
            f"loss={train_metrics['loss']:.4f} "
            f"accuracy={train_metrics['accuracy']:.4f} "
            f"macro_F1={train_metrics['macro_f1']:.4f}"
        )

        print(
            f"  Validation "
            f"loss={validation_metrics['loss']:.4f} "
            f"accuracy={validation_metrics['accuracy']:.4f} "
            f"macro_F1={validation_metrics['macro_f1']:.4f}"
        )

        # ----------------------------------------------------
        # Best model by validation macro-F1
        # ----------------------------------------------------

        if (
            validation_metrics[
                "macro_f1"
            ]
            > best_macro_f1
        ):

            best_macro_f1 = (
                validation_metrics[
                    "macro_f1"
                ]
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            save_checkpoint(
                model,
                epoch,
                validation_metrics,
                class_weights,
            )

            print(
                "  [SAVE] New best model."
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()

            print(
                f"[EARLY STOP] Validation macro-F1 "
                f"did not improve for "
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )

            break

    save_history(
        history
    )

    # ========================================================
    # BEST CHECKPOINT
    # ========================================================

    print_section(
        "BEST MODEL"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation macro-F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Checkpoint: "
        f"{BEST_MODEL_PATH}"
    )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print_section(
        "FINAL TEST EVALUATION"
    )

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print(
        f"Test loss: "
        f"{test_metrics['loss']:.4f}"
    )

    print(
        f"Test accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test macro precision: "
        f"{test_metrics['macro_precision']:.4f}"
    )

    print(
        f"Test macro recall: "
        f"{test_metrics['macro_recall']:.4f}"
    )

    print(
        f"Test macro F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    # ========================================================
    # PER-CLASS TEST RESULTS
    # ========================================================

    print_section(
        "TEST PER-CLASS METRICS"
    )

    for species in SPECIES_CLASSES:

        metrics = (
            test_metrics[
                "per_class"
            ][
                species
            ]
        )

        print(
            species
        )

        print(
            f"  Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"  Recall:    "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"  F1:        "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"  Support:   "
            f"{metrics['support']}"
        )

        print()

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print_section(
        "TEST CONFUSION MATRIX"
    )

    print_confusion_matrix(
        test_metrics[
            "confusion_matrix"
        ]
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_for_json = {

        "best_epoch":
            best_epoch,

        "best_validation_macro_f1":
            best_macro_f1,

        "test_loss":
            test_metrics[
                "loss"
            ],

        "test_accuracy":
            test_metrics[
                "accuracy"
            ],

        "test_macro_precision":
            test_metrics[
                "macro_precision"
            ],

        "test_macro_recall":
            test_metrics[
                "macro_recall"
            ],

        "test_macro_f1":
            test_metrics[
                "macro_f1"
            ],

        "per_class":
            test_metrics[
                "per_class"
            ],

        "confusion_matrix":
            test_metrics[
                "confusion_matrix"
            ].tolist(),

        "species_classes":
            SPECIES_CLASSES,

        "class_weights":
            class_weights
            .detach()
            .cpu()
            .tolist(),
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics_for_json,
            file,
            indent=2,
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print_section(
        "TRAINING COMPLETE"
    )

    print(
        "Best model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print()

    print(
        "Training history:"
    )

    print(
        HISTORY_PATH
    )

    print()

    print(
        "Test metrics:"
    )

    print(
        METRICS_PATH
    )

    print()

    print(
        "RESULT:"
    )

    print(
        "The first leakage-safe, class-weighted "
        "species-recognition baseline has been trained "
        "and evaluated on the held-out test split."
    )

    print()

    print(
        "Next step: interpret the test confusion "
        "matrix and determine whether the frozen "
        "ResNet18 backbone should be fine-tuned."
    )


if __name__ == "__main__":
    run()