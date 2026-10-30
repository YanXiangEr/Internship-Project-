"""
world_model_v1_inference.py

First end-to-end inference component of the microorganism world model.

Inputs
------
- microorganism species
- qualitative water-quality/environmental context

Pipeline
--------
species + water context
        ↓
biological-state classifier
        ↓
predicted healthy / stressed state
        ↓
retrieve same-species, same-state evidence
        ↓
grounded biological / morphological description

This stage uses:
- the trained state classifier
- spreadsheet-derived internal observations
- Word-document-derived visual observations represented in the
  multimodal dataset
- approved external scientific evidence

IMPORTANT
---------
The description is retrieval-grounded.

The program does NOT invent:
- morphology
- pollutant concentrations
- pH
- temperature
- nutrient values
- exposure duration

unless they are present in retrieved evidence.

Visual simulation will be added after this inference layer is stable.
"""

import csv
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

STATE_MODEL_PATH = (
    MODEL_DIR
    / "world_model_v1_state_classifier.pkl"
)

WORLD_MODEL_DATASET = (
    DATA_DIR
    / "world_model_v1_dataset.csv"
)


# ============================================================
# SPECIES
# ============================================================

SUPPORTED_SPECIES = [
    "Euglena viridis",
    "Nitzschia palea",
    "Oscillatoria limosa",
]


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_TOP_K = 3


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
# LOAD DATA
# ============================================================

def load_dataset():

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
            csv.DictReader(file)
        )

    return rows


def load_state_model():

    if not STATE_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"State classifier not found:\n"
            f"{STATE_MODEL_PATH}"
        )

    with open(
        STATE_MODEL_PATH,
        "rb",
    ) as file:

        return pickle.load(file)


# ============================================================
# MODEL INPUT
# ============================================================

def build_model_input(
    species,
    water_context,
):

    """
    Build exactly the same input structure used by the state model.
    """

    return pd.DataFrame(
        [
            {
                "species":
                    species,

                "input_text":
                    water_context,
            }
        ]
    )


# ============================================================
# STATE PREDICTION
# ============================================================

def predict_state(
    model,
    species,
    water_context,
):

    X = build_model_input(
        species,
        water_context,
    )

    predicted_state = model.predict(
        X
    )[0]

    probabilities = model.predict_proba(
        X
    )[0]

    classes = list(
        model.named_steps[
            "classifier"
        ].classes_
    )

    probability_map = {
        label:
            float(
                probabilities[index]
            )
        for index, label
        in enumerate(classes)
    }

    return {

        "predicted_state":
            predicted_state,

        "healthy_probability":
            probability_map.get(
                "healthy",
                0.0,
            ),

        "stressed_probability":
            probability_map.get(
                "stressed",
                0.0,
            ),

        "confidence":
            max(
                probability_map.values()
            ),
    }


# ============================================================
# EVIDENCE TEXT
# ============================================================

def build_evidence_search_text(row):

    fields = [
        "qualitative_condition",
        "primary_water_factor",
        "secondary_water_factor",
    ]

    parts = []

    for field in fields:

        value = clean(
            row.get(field)
        )

        if value:

            parts.append(
                value
            )

    return " | ".join(
        parts
    )


# ============================================================
# EVIDENCE RETRIEVAL
# ============================================================

def retrieve_evidence(
    rows,
    species,
    predicted_state,
    water_context,
    top_k=DEFAULT_TOP_K,
):

    """
    Retrieve evidence from the same species and predicted state.

    Similarity is based ONLY on input-side environmental context.

    Output-side morphology is not used for similarity matching.
    """

    candidates = [
        row
        for row in rows
        if (
            clean(
                row.get(
                    "species"
                )
            )
            == species
            and
            clean(
                row.get(
                    "condition_label"
                )
            ).lower()
            == predicted_state.lower()
        )
    ]

    if not candidates:

        return []

    candidate_texts = [
        build_evidence_search_text(
            row
        )
        for row in candidates
    ]

    # Avoid an empty vocabulary if some rows contain no
    # environmental text.

    candidate_texts = [
        text
        if text
        else "unspecified_environment"
        for text in candidate_texts
    ]

    query_text = (
        water_context
        if clean(
            water_context
        )
        else "unspecified_environment"
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
    )

    matrix = vectorizer.fit_transform(
        candidate_texts
        + [
            query_text
        ]
    )

    candidate_vectors = matrix[
        :-1
    ]

    query_vector = matrix[
        -1
    ]

    similarities = cosine_similarity(
        candidate_vectors,
        query_vector,
    ).flatten()

    ranked_indices = np.argsort(
        similarities
    )[::-1]

    results = []

    seen_hashes = set()

    for index in ranked_indices:

        row = candidates[
            int(index)
        ]

        digest = clean(
            row.get(
                "sha256"
            )
        )

        # Avoid repeating exactly the same image evidence.

        if (
            digest
            and digest in seen_hashes
        ):

            continue

        if digest:

            seen_hashes.add(
                digest
            )

        result = dict(
            row
        )

        result[
            "retrieval_similarity"
        ] = float(
            similarities[
                index
            ]
        )

        results.append(
            result
        )

        if len(results) >= top_k:

            break

    return results


# ============================================================
# COLLECT SUPPORTED MORPHOLOGY
# ============================================================

def collect_supported_features(
    evidence_rows,
):

    features = []

    fields = [
        "visual_phenotype",
        "response_labels",
        "visual_feature_1",
        "visual_feature_2",
    ]

    for row in evidence_rows:

        for field in fields:

            value = clean(
                row.get(field)
            )

            if (
                value
                and value not in features
            ):

                features.append(
                    value
                )

    return features


# ============================================================
# DESCRIPTION
# ============================================================

def build_grounded_description(
    species,
    water_context,
    state_result,
    evidence_rows,
):

    predicted_state = (
        state_result[
            "predicted_state"
        ]
    )

    confidence = (
        state_result[
            "confidence"
        ]
    )

    description_parts = []

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    description_parts.append(
        (
            f"For {species}, the v1 biological-state model "
            f"predicts a {predicted_state} state "
            f"with model probability {confidence:.1%}."
        )
    )

    # --------------------------------------------------------
    # Input context
    # --------------------------------------------------------

    if clean(
        water_context
    ):

        description_parts.append(
            (
                "The supplied water-quality context is: "
                f"{water_context}."
            )
        )

    # --------------------------------------------------------
    # Evidence-backed morphology
    # --------------------------------------------------------

    supported_features = (
        collect_supported_features(
            evidence_rows
        )
    )

    if supported_features:

        displayed_features = (
            supported_features[:5]
        )

        feature_text = "; ".join(
            displayed_features
        )

        description_parts.append(
            (
                "Similar evidence-backed observations for "
                f"{species} in the predicted state report: "
                f"{feature_text}."
            )
        )

    else:

        description_parts.append(
            (
                "The current dataset does not contain a "
                "specific morphology annotation sufficiently "
                "matched to this predicted state, so no "
                "additional visual feature is inferred."
            )
        )

    # --------------------------------------------------------
    # Evidence qualification
    # --------------------------------------------------------

    if evidence_rows:

        evidence_types = []

        for row in evidence_rows:

            evidence_type = clean(
                row.get(
                    "evidence_type"
                )
            )

            if (
                evidence_type
                and
                evidence_type
                not in evidence_types
            ):

                evidence_types.append(
                    evidence_type
                )

        if evidence_types:

            description_parts.append(
                (
                    "The supporting records include: "
                    + ", ".join(
                        evidence_types
                    )
                    + "."
                )
            )

    # --------------------------------------------------------
    # Quantitative limitation
    # --------------------------------------------------------

    if species in {
        "Euglena viridis",
        "Oscillatoria limosa",
    }:

        description_parts.append(
            (
                "For this species, the current v1 state model "
                "is qualitative; it does not yet support "
                "numerical prediction from pH, temperature or "
                "pollutant concentration."
            )
        )

    elif species == "Nitzschia palea":

        description_parts.append(
            (
                "Quantitative Nitzschia exposure evidence is "
                "available separately for the upcoming visual-"
                "conditioning component, but the current "
                "healthy/stressed state classifier remains "
                "based on explicitly labelled qualitative "
                "state observations."
            )
        )

    return " ".join(
        description_parts
    )


# ============================================================
# EVIDENCE DISPLAY
# ============================================================

def print_evidence(
    evidence_rows,
):

    print_section(
        "RETRIEVED SUPPORTING EVIDENCE"
    )

    if not evidence_rows:

        print(
            "No matching evidence records were found."
        )

        return

    for index, row in enumerate(
        evidence_rows,
        start=1,
    ):

        print(
            f"EVIDENCE {index}"
        )

        print(
            f"  Record: "
            f"{clean(row.get('record_id'))}"
        )

        print(
            f"  State: "
            f"{clean(row.get('condition_label'))}"
        )

        print(
            f"  Similarity: "
            f"{float(row['retrieval_similarity']):.4f}"
        )

        print(
            f"  Evidence type: "
            f"{clean(row.get('evidence_type'))}"
        )

        print(
            f"  Source origin: "
            f"{clean(row.get('dataset_origin'))}"
        )

        morphology = clean(
            row.get(
                "state_label"
            )
        )

        if morphology:

            print(
                f"  Biological state: "
                f"{morphology}"
            )

        provenance = clean(
            row.get(
                "provenance_url"
            )
        )

        if provenance:

            print(
                f"  Provenance: "
                f"{provenance}"
            )

        print()


# ============================================================
# INFERENCE API
# ============================================================

def run_world_model(
    species,
    water_context,
    top_k=DEFAULT_TOP_K,
):

    if species not in SUPPORTED_SPECIES:

        raise ValueError(
            (
                f"Unsupported species: {species}\n"
                "Supported species are: "
                + ", ".join(
                    SUPPORTED_SPECIES
                )
            )
        )

    model = load_state_model()

    rows = load_dataset()

    state_result = predict_state(
        model,
        species,
        water_context,
    )

    evidence = retrieve_evidence(
        rows,
        species,
        state_result[
            "predicted_state"
        ],
        water_context,
        top_k=top_k,
    )

    description = (
        build_grounded_description(
            species,
            water_context,
            state_result,
            evidence,
        )
    )

    return {

        "species":
            species,

        "water_context":
            water_context,

        "predicted_state":
            state_result[
                "predicted_state"
            ],

        "healthy_probability":
            state_result[
                "healthy_probability"
            ],

        "stressed_probability":
            state_result[
                "stressed_probability"
            ],

        "confidence":
            state_result[
                "confidence"
            ],

        "description":
            description,

        "evidence":
            evidence,
    }


# ============================================================
# INTERACTIVE CLI
# ============================================================

def run_interactive():

    print_section(
        "MICROORGANISM WORLD MODEL V1"
    )

    print(
        "Supported species:"
    )

    for index, species in enumerate(
        SUPPORTED_SPECIES,
        start=1,
    ):

        print(
            f"  {index}. {species}"
        )

    print()

    species_input = input(
        "Enter species number or exact species name: "
    ).strip()

    if species_input.isdigit():

        species_index = (
            int(species_input)
            - 1
        )

        if (
            species_index < 0
            or species_index
            >= len(
                SUPPORTED_SPECIES
            )
        ):

            raise ValueError(
                "Invalid species number."
            )

        species = (
            SUPPORTED_SPECIES[
                species_index
            ]
        )

    else:

        species = species_input

    print()

    print(
        "Enter the qualitative water-quality context."
    )

    print(
        "Examples:"
    )

    print(
        "  sewage organic pollution"
    )

    print(
        "  eutrophication phosphate"
    )

    print(
        "  industrial effluent"
    )

    print(
        "  nitrate nitrite phosphate"
    )

    print()

    water_context = input(
        "Water context: "
    ).strip()

    # ========================================================
    # RUN
    # ========================================================

    result = run_world_model(
        species,
        water_context,
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print_section(
        "PREDICTED BIOLOGICAL STATE"
    )

    print(
        f"Species: "
        f"{result['species']}"
    )

    print(
        f"Predicted state: "
        f"{result['predicted_state']}"
    )

    print()

    print(
        f"Healthy probability: "
        f"{result['healthy_probability']:.4f}"
    )

    print(
        f"Stressed probability: "
        f"{result['stressed_probability']:.4f}"
    )

    print(
        f"Model confidence: "
        f"{result['confidence']:.4f}"
    )

    print_section(
        "GROUNDED DESCRIPTION"
    )

    print(
        result[
            "description"
        ]
    )

    print_evidence(
        result[
            "evidence"
        ]
    )

    # ========================================================
    # SAFETY / LIMITATIONS
    # ========================================================

    print_section(
        "MODEL SCOPE"
    )

    print(
        "This is a data-driven v1 prediction."
    )

    print()

    print(
        "Morphological statements above are retrieved "
        "from existing project/scientific evidence rather "
        "than invented by the model."
    )

    print()

    print(
        "Visual appearance simulation is not yet enabled."
    )

    print()

    print(
        "The next world-model component will use the "
        "paired microscopy evidence to construct the "
        "visual-conditioning representation."
    )


if __name__ == "__main__":
    run_interactive()