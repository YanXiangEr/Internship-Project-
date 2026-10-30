"""
species_model.py

Defines and validates the first transfer-learning baseline model
for microorganism species recognition.

Model:
    ResNet18 pretrained on ImageNet

Output classes:
    0 = Euglena viridis
    1 = Nitzschia palea
    2 = Oscillatoria limosa

This stage:
- builds the model
- replaces the final classification layer
- optionally freezes the convolutional backbone
- verifies trainable parameter counts
- loads one batch from species_dataset.py
- performs one forward pass
- verifies output dimensions

This stage DOES NOT train the model.
"""

import torch
import torch.nn as nn

from torchvision import models

from species_dataset import (
    SPECIES_CLASSES,
    build_dataloaders,
)


# ============================================================
# SETTINGS
# ============================================================

NUM_CLASSES = len(
    SPECIES_CLASSES
)

FREEZE_BACKBONE = True


# ============================================================
# HELPERS
# ============================================================

def print_section(title):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print()


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


# ============================================================
# MODEL
# ============================================================

def build_species_model(
    freeze_backbone=True,
):

    """
    Build pretrained ResNet18 for 3-class microorganism
    species recognition.
    """

    # --------------------------------------------------------
    # Load pretrained ImageNet weights
    # --------------------------------------------------------

    weights = (
        models.ResNet18_Weights.DEFAULT
    )

    model = models.resnet18(
        weights=weights
    )

    # --------------------------------------------------------
    # Freeze pretrained feature extractor
    # --------------------------------------------------------

    if freeze_backbone:

        for parameter in (
            model.parameters()
        ):

            parameter.requires_grad = False

    # --------------------------------------------------------
    # Replace final classifier
    # --------------------------------------------------------

    input_features = (
        model.fc.in_features
    )

    model.fc = nn.Linear(
        input_features,
        NUM_CLASSES,
    )

    # The new classifier remains trainable even if the
    # backbone is frozen.

    return model


# ============================================================
# PARAMETER COUNTS
# ============================================================

def parameter_counts(model):

    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    frozen = (
        total
        - trainable
    )

    return {
        "total":
            total,

        "trainable":
            trainable,

        "frozen":
            frozen,
    }


# ============================================================
# FORWARD VALIDATION
# ============================================================

def validate_forward_pass(
    model,
    device,
):

    loaders = build_dataloaders(
        batch_size=8,
        num_workers=0,
    )

    train_loader = (
        loaders[
            "train"
        ]
    )

    batch = next(
        iter(
            train_loader
        )
    )

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

    model.eval()

    with torch.no_grad():

        logits = model(
            images
        )

    return {
        "images":
            images,

        "labels":
            labels,

        "logits":
            logits,
    }


# ============================================================
# MAIN VALIDATION
# ============================================================

def run_validation():

    print_section(
        "SPECIES MODEL VALIDATION"
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = get_device()

    print(
        f"Device: {device}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    print_section(
        "BUILDING RESNET18 BASELINE"
    )

    print(
        "Architecture: ResNet18"
    )

    print(
        "Pretraining: ImageNet"
    )

    print(
        f"Output classes: "
        f"{NUM_CLASSES}"
    )

    print(
        f"Freeze backbone: "
        f"{FREEZE_BACKBONE}"
    )

    model = build_species_model(
        freeze_backbone=FREEZE_BACKBONE,
    )

    model = model.to(
        device
    )

    # ========================================================
    # CLASSIFIER
    # ========================================================

    print_section(
        "CLASSIFIER HEAD"
    )

    print(
        model.fc
    )

    print()

    print(
        "Species mapping:"
    )

    for index, species in enumerate(
        SPECIES_CLASSES
    ):

        print(
            f"  {index}: "
            f"{species}"
        )

    # ========================================================
    # PARAMETERS
    # ========================================================

    counts = parameter_counts(
        model
    )

    print_section(
        "MODEL PARAMETERS"
    )

    print(
        f"Total parameters:     "
        f"{counts['total']:,}"
    )

    print(
        f"Trainable parameters: "
        f"{counts['trainable']:,}"
    )

    print(
        f"Frozen parameters:    "
        f"{counts['frozen']:,}"
    )

    if FREEZE_BACKBONE:

        expected_trainable = (
            model.fc.in_features
            * NUM_CLASSES
            + NUM_CLASSES
        )

        if (
            counts[
                "trainable"
            ]
            == expected_trainable
        ):

            print()

            print(
                "[PASS] Only the new classifier "
                "head is trainable."
            )

        else:

            print()

            print(
                "[WARNING] Trainable parameter "
                "count differs from expected "
                "classifier-only count."
            )

    # ========================================================
    # FORWARD PASS
    # ========================================================

    print_section(
        "FORWARD PASS TEST"
    )

    result = validate_forward_pass(
        model,
        device,
    )

    images = result[
        "images"
    ]

    labels = result[
        "labels"
    ]

    logits = result[
        "logits"
    ]

    print(
        f"Input batch shape: "
        f"{tuple(images.shape)}"
    )

    print(
        f"Label batch shape: "
        f"{tuple(labels.shape)}"
    )

    print(
        f"Output logits shape: "
        f"{tuple(logits.shape)}"
    )

    # ========================================================
    # OUTPUT SHAPE CHECK
    # ========================================================

    expected_shape = (
        images.shape[0],
        NUM_CLASSES,
    )

    if (
        tuple(
            logits.shape
        )
        != expected_shape
    ):

        raise RuntimeError(
            "Unexpected model output shape. "
            f"Expected {expected_shape}, "
            f"got {tuple(logits.shape)}."
        )

    print()

    print(
        "[PASS] Model output shape is correct."
    )

    # ========================================================
    # PROBABILITIES
    # ========================================================

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    predicted_classes = torch.argmax(
        probabilities,
        dim=1,
    )

    print_section(
        "UNTRAINED MODEL SAMPLE OUTPUT"
    )

    print(
        "IMPORTANT:"
    )

    print(
        "These predictions are from the newly "
        "attached classifier BEFORE training."
    )

    print(
        "They have no biological meaning yet."
    )

    print()

    for index in range(
        min(
            5,
            images.shape[0],
        )
    ):

        true_index = int(
            labels[index].item()
        )

        predicted_index = int(
            predicted_classes[
                index
            ].item()
        )

        probs = (
            probabilities[
                index
            ]
            .detach()
            .cpu()
            .tolist()
        )

        print(
            f"Sample {index + 1}"
        )

        print(
            f"  True class: "
            f"{SPECIES_CLASSES[true_index]}"
        )

        print(
            f"  Predicted class: "
            f"{SPECIES_CLASSES[predicted_index]}"
        )

        print(
            "  Raw probabilities:"
        )

        for class_index, probability in enumerate(
            probs
        ):

            print(
                f"    "
                f"{SPECIES_CLASSES[class_index]:<24} "
                f"{probability:.4f}"
            )

        print()

    # ========================================================
    # LOSS FUNCTION CHECK
    # ========================================================

    print_section(
        "LOSS FUNCTION TEST"
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    loss = criterion(
        logits,
        labels,
    )

    print(
        f"Cross-entropy loss: "
        f"{loss.item():.6f}"
    )

    if torch.isfinite(
        loss
    ):

        print(
            "[PASS] Loss is finite."
        )

    else:

        raise RuntimeError(
            "Loss is not finite."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print_section(
        "MODEL VALIDATION SUMMARY"
    )

    print(
        "[PASS] Pretrained ResNet18 loaded."
    )

    print(
        "[PASS] Final layer replaced with "
        "3-class classifier."
    )

    if FREEZE_BACKBONE:

        print(
            "[PASS] Pretrained backbone frozen."
        )

    print(
        "[PASS] Training batch passed "
        "through the network."
    )

    print(
        "[PASS] Output dimensions are "
        "(batch_size, 3)."
    )

    print(
        "[PASS] Cross-entropy loss "
        "can be calculated."
    )

    print()

    print(
        "RESULT:"
    )

    print(
        "Species-recognition baseline model "
        "is structurally ready for training."
    )

    print()

    print(
        "Next step: implement the training loop "
        "with class-imbalance handling, validation "
        "monitoring and best-model checkpointing."
    )


if __name__ == "__main__":
    run_validation()