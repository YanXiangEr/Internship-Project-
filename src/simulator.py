"""
simulator.py

Simple simulation engine for the microorganism world model.

The simulator takes a WorldState describing the current
environment and uses the evidence-based world model to generate
a predicted next state.

The simulator now supports two related simulation modes:

1. Ordinary simulation

    WorldState(t)
          |
          v
    predict_response()
          |
          v
    evidence database
          |
          v
    WorldState(t+1)

2. Alternative-state / branching simulation

    WorldState(t)
          |
          v
    alternative-state evidence
          |
          +----> Possible State 1
          |
          +----> Possible State 2

The branching system is useful when the database explicitly
contains alternative organism states, such as Euglena viridis
Morphotype I and Morphotype II.

Important:
This is an evidence-based state transition system. It is not yet
a continuous-time biological simulation.

Alternative states are evidence-supported possibilities.
They are not probabilities.
"""


from copy import deepcopy
import pickle
import re
from pathlib import Path

import pandas as pd

from world_state import WorldState
from world_model import predict_response
from visual_state import resolve_visual_state

from state_hypotheses import (
    StateHypothesis,
    get_state_hypotheses_for_water
)


# ============================================================
# SIMULATION RESULT
# ============================================================

class SimulationResult:
    """
    Stores the result of one ordinary simulation step.

    This keeps:

        - the original state
        - the predicted next state
        - the raw world-model result
        - the predicted overall biological condition
        - where that condition decision came from
        - an evidence-backed visual state

    Keeping the raw result is useful because it contains
    supporting evidence that may later be displayed in the
    Streamlit interface.

    The visual state is currently an evidence-backed visual analogue.
    It is not yet a newly generated microscopy image.
    """

    def __init__(
        self,
        current_state,
        next_state,
        raw_result,
        predicted_condition=None,
        condition_confidence=None,
        condition_source=None,
        visual_state=None,
        classifier_environment_terms=None,
        classifier_input_supported=None,
        no_prediction_reason=None,
    ):

        self.current_state = (
            current_state
        )

        self.next_state = (
            next_state
        )

        self.raw_result = (
            raw_result
        )

        self.predicted_condition = (
            predicted_condition
        )

        self.condition_confidence = (
            condition_confidence
        )

        self.condition_source = (
            condition_source
        )

        self.visual_state = (
            visual_state
        )

        self.classifier_environment_terms = (
            classifier_environment_terms
            or []
        )

        self.classifier_input_supported = (
            classifier_input_supported
        )

        # When no response reaches the next state, retain the reason so
        # every interface can distinguish a value/context mismatch from
        # evidence that matched but fell below the selected threshold.
        self.no_prediction_reason = (
            no_prediction_reason
        )

    def to_dict(self):
        """
        Convert the complete simulation result into a dictionary.
        """

        return {
            "current_state":
                self.current_state.to_dict(),

            "next_state":
                self.next_state.to_dict(),

            "raw_result":
                deepcopy(
                    self.raw_result
                ),

            "predicted_condition":
                self.predicted_condition,

            "condition_confidence":
                self.condition_confidence,

            "condition_source":
                self.condition_source,

            "classifier_environment_terms":
                list(
                    self.classifier_environment_terms
                ),

            "classifier_input_supported":
                self.classifier_input_supported,

            "visual_state": (
                self.visual_state.to_dict()
                if self.visual_state is not None
                else None
            )
        }


# ============================================================
# BRANCHING SIMULATION RESULT
# ============================================================

class BranchingSimulationResult:
    """
    Stores alternative possible next states.

    Unlike SimulationResult, which contains one next_state,
    this object can contain several mutually alternative
    organism-state hypotheses.

    These alternatives are evidence-supported possibilities.
    They are not probabilities.
    """

    def __init__(
        self,
        current_state,
        possible_states,
        deterministic_result=None
    ):

        self.current_state = (
            current_state
        )

        self.possible_states = (
            possible_states
        )

        self.deterministic_result = (
            deterministic_result
        )

    @property
    def has_alternatives(self):
        """
        True when two or more explicit alternative states exist.
        """

        return (
            len(self.possible_states) >= 2
        )

    @property
    def number_of_possible_states(self):
        """
        Number of explicit state hypotheses returned.
        """

        return len(
            self.possible_states
        )

    def to_dict(self):
        """
        Convert the branching result into a dictionary.
        """

        return {
            "current_state":
                self.current_state.to_dict(),

            "has_alternatives":
                self.has_alternatives,

            "number_of_possible_states":
                self.number_of_possible_states,

            "possible_states": [
                state.to_dict()
                for state
                in self.possible_states
            ],

            "deterministic_result": (
                self.deterministic_result.to_dict()
                if self.deterministic_result
                is not None
                else None
            )
        }


# ============================================================
# BIOLOGICAL-STATE MODEL
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STATE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "world_model_v1_state_classifier.pkl"
)


def load_biological_state_model():
    """
    Load the trained healthy/stressed v1 classifier.

    Returning None keeps the evidence-based simulator usable even
    if the optional classifier checkpoint is temporarily unavailable.
    """

    if not STATE_MODEL_PATH.exists():

        return None

    with open(
        STATE_MODEL_PATH,
        "rb"
    ) as file:

        return pickle.load(
            file
        )


def build_state_model_input(
    species,
    water_state
):
    """
    Build the qualitative environmental input expected by the
    healthy/stressed v1 classifier.

    Numeric concentrations remain available to the main evidence
    model, but are deliberately not passed to this classifier because
    its current training dataset supports qualitative context only.
    """

    parts = []

    def expanded_text(value):
        """Preserve a database token and add its readable token form."""

        raw = str(value).strip()

        readable = re.sub(
            r"[_-]+",
            " ",
            raw,
        )

        readable = re.sub(
            r"\s+",
            " ",
            readable,
        ).strip()

        if not readable or readable.lower() == raw.lower():
            return raw

        return f"{raw} {readable}"

    for factor, value in water_state.items():

        factor_text = str(
            factor
        ).strip()

        if not factor_text:
            continue

        if isinstance(
            value,
            dict
        ):

            # The state classifier currently uses the factor itself,
            # not its numerical concentration.
            parts.append(
                expanded_text(
                    factor_text
                )
            )

        else:

            value_text = str(
                value
            ).strip()

            if (
                value_text
                and value_text.lower()
                not in {
                    "present",
                    "true",
                    "yes",
                    "1",
                }
            ):

                raw_phrase = (
                    f"{factor_text} {value_text}"
                )

                readable_phrase = (
                    f"{expanded_text(factor_text)} "
                    f"{expanded_text(value_text)}"
                )

                parts.append(
                    f"{raw_phrase} {readable_phrase}"
                )

            else:

                parts.append(
                    expanded_text(
                        factor_text
                    )
                )

    input_text = " | ".join(
        parts
    )

    return pd.DataFrame(
        [
            {
                "species":
                    species,

                "input_text":
                    input_text,
            }
        ]
    )


def classifier_environment_coverage(
    model,
    water_state
):
    """Inspect whether the text classifier recognises a supplied factor.

    Returns ``(None, [])`` when the model structure cannot be inspected, so
    older compatible checkpoints retain their previous behaviour. For the
    project's sklearn pipeline, True/False is returned together with the
    recognised factor-derived vocabulary terms.
    """

    try:
        preprocess = model.named_steps["preprocess"]
        transformers = getattr(
            preprocess,
            "transformers_",
            preprocess.transformers,
        )

        vectorizer = None

        for _name, transformer, columns in transformers:

            if columns != "input_text":
                continue

            if hasattr(transformer, "named_steps"):
                for step in reversed(
                    list(transformer.named_steps.values())
                ):
                    if hasattr(step, "vocabulary_"):
                        vectorizer = step
                        break

            elif hasattr(transformer, "vocabulary_"):
                vectorizer = transformer

            if vectorizer is not None:
                break

        if vectorizer is None:
            return None, []

        vocabulary = vectorizer.vocabulary_
        analyzer = vectorizer.build_analyzer()
        recognised = set()

        for factor in (water_state or {}):

            raw_factor = str(factor).strip()

            readable_factor = re.sub(
                r"[_-]+",
                " ",
                raw_factor,
            )

            factor_text = f"{raw_factor} {readable_factor}"

            for term in analyzer(factor_text):
                if term in vocabulary:
                    recognised.add(term)

        recognised_terms = sorted(
            recognised,
            key=lambda term: (
                " " in term,
                len(term),
                term,
            ),
        )

        return bool(recognised_terms), recognised_terms

    except (
        AttributeError,
        KeyError,
        TypeError,
    ):
        return None, []

def predict_biological_condition(
    model,
    species,
    water_state
):
    """
    Use the qualitative healthy/stressed classifier as a fallback.

    Direct adverse response evidence, when available, is handled
    separately and takes priority over this classifier.
    """

    if model is None:

        return {
            "condition": None,
            "confidence": None,
            "healthy_probability": None,
            "stressed_probability": None,
            "environment_input_supported": None,
            "recognized_environment_terms": [],
        }

    X = build_state_model_input(
        species,
        water_state
    )

    (
        environment_input_supported,
        recognized_environment_terms,
    ) = classifier_environment_coverage(
        model=model,
        water_state=water_state,
    )

    if environment_input_supported is False:

        return {
            "condition": None,
            "confidence": None,
            "healthy_probability": None,
            "stressed_probability": None,
            "environment_input_supported": False,
            "recognized_environment_terms": [],
        }

    prediction = model.predict(
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
        "condition":
            prediction,

        "confidence":
            max(
                probability_map.values()
            ),

        "healthy_probability":
            probability_map.get(
                "healthy"
            ),

        "stressed_probability":
            probability_map.get(
                "stressed"
            ),

        "environment_input_supported":
            environment_input_supported,

        "recognized_environment_terms":
            recognized_environment_terms,
    }


# ============================================================
# BIOLOGICAL-CONDITION EVIDENCE POLICY
# ============================================================

# These response types are explicitly adverse biological outcomes
# already represented in the project's evidence database.
#
# We deliberately keep this list conservative. A response is not
# classified as stress simply because it changes.

ADVERSE_RESPONSE_TYPES = {
    "cell_death",
    "cell_mortality_fraction",
    "reduced_viability",
    "mortality",
    "growth_inhibition",
    "growth_reduction",
    "chloroplast_damage",
    "chloroplast_loss",
    "burst_chloroplast",
    "frustule_breakage",
    "frustule_dispersion",
    "membrane_damage",
    "deformation",
}


# Only these relationship types establish that the supplied water factor is
# directly linked to the adverse response. Co-observed, observation-level and
# uncertain relationships remain useful evidence, but they must not be
# promoted to direct adverse-response provenance.
DIRECT_RESPONSE_RELATIONSHIP_TYPES = {
    "case_specific",
    "experimentally_observed",
}


def direct_response_evidence_score(
    prediction
):
    """Return a score calculated from direct supporting records only.

    The forward model's prediction score can combine direct and contextual
    observations. Condition provenance must therefore recalculate support from
    the prediction's individual evidence records instead of assuming that an
    adverse response name makes the complete prediction direct.
    """

    observation_scores = {}

    for evidence in prediction.get(
        "supporting_evidence",
        []
    ):

        relationship_type = str(
            evidence.get(
                "relationship_type",
                ""
            )
        ).strip().lower()

        if (
            relationship_type
            not in
            DIRECT_RESPONSE_RELATIONSHIP_TYPES
        ):
            continue

        try:
            evidence_strength = float(
                evidence.get(
                    "evidence_strength",
                    0.0
                )
            )
        except (
            TypeError,
            ValueError
        ):
            continue

        observation_id = str(
            evidence.get(
                "observation_id",
                ""
            )
        ).strip()

        # A stable fallback keeps separately returned direct records distinct
        # even if a future importer omits an observation identifier.
        observation_key = (
            observation_id
            or
            str(
                evidence.get(
                    "rule_id",
                    id(evidence)
                )
            )
        )

        observation_scores[
            observation_key
        ] = max(
            observation_scores.get(
                observation_key,
                0.0
            ),
            evidence_strength
        )

    if not observation_scores:
        return None

    scores = list(
        observation_scores.values()
    )

    score = (
        max(scores)
        +
        0.1
        *
        max(
            0,
            len(scores) - 1
        )
    )

    return round(
        min(score, 1.0),
        3
    )


def response_value_indicates_adverse_state(
    response_type,
    response_value
):
    """
    Conservative interpretation of explicitly adverse responses.

    The response TYPE must first be known to represent an adverse
    biological outcome. We do not infer stress from arbitrary changes.
    """

    if response_type not in ADVERSE_RESPONSE_TYPES:
        return False

    if response_value is None:
        return False

    value_text = str(
        response_value
    ).strip().lower()

    # Explicit absence should not trigger stress.

    non_adverse_values = {
        "",
        "none",
        "absent",
        "not_present",
        "not observed",
        "none_observed",
        "0",
        "0.0",
    }

    if value_text in non_adverse_values:
        return False

    # Numeric values such as cell-death percentages.

    try:

        numeric_value = float(
            value_text
        )

        return numeric_value > 0

    except ValueError:

        pass

    # Many stored values contain a number plus a unit, for example
    # "20.8 percent". Extract the first numeric token conservatively.

    number_match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
        value_text
    )

    if number_match is not None:

        try:

            numeric_value = float(
                number_match.group(0)
            )

            return numeric_value > 0

        except ValueError:

            pass

    # For an explicitly adverse response type, a supported non-absence
    # categorical value such as "present" is treated as adverse.

    return True


def condition_from_response_evidence(
    predictions,
    minimum_evidence_score=0.0
):
    """
    Look for direct adverse biological evidence in predict_response().

    Returns:
        dict when direct evidence supports a stressed state
        None otherwise

    IMPORTANT:
    None does NOT mean healthy.
    It only means direct adverse evidence was not found and the
    classifier may therefore be used as a fallback.
    """

    supporting_responses = []

    strongest_score = 0.0

    for prediction in predictions:

        response_type = prediction.get(
            "response_type"
        )

        response_value = prediction.get(
            "response_value"
        )

        if not response_type:
            continue

        direct_evidence_score = (
            direct_response_evidence_score(
                prediction
            )
        )

        # An adverse response supported only by co-observed,
        # observation-level, uncertain or untyped evidence is not direct.
        if direct_evidence_score is None:
            continue

        if (
            direct_evidence_score
            <
            minimum_evidence_score
        ):
            continue

        if response_value_indicates_adverse_state(
            response_type=response_type,
            response_value=response_value
        ):

            supporting_responses.append(
                {
                    "response_type":
                        response_type,

                    "response_value":
                        response_value,

                    "evidence_score":
                        direct_evidence_score,
                }
            )

            strongest_score = max(
                strongest_score,
                float(
                    direct_evidence_score
                )
            )

    if not supporting_responses:
        return None

    return {
        "condition":
            "stressed",

        "confidence":
            strongest_score,

        "source":
            "direct_response_evidence",

        "supporting_responses":
            supporting_responses,
    }



def condition_from_observation_context(
    predictions,
    minimum_evidence_score=0.0
):
    """
    Use explicit observation-condition provenance as a secondary
    evidence source when no adverse response type is present.

    This is intentionally weaker than direct adverse-response evidence.

    Policy
    ------
    - Only retained predictions at/above the current threshold are considered.
    - Supporting evidence should carry the explicit ``observations.condition``
      value returned by the evidence model.
    - Legacy ``_stressed_`` / ``_healthy_`` observation-ID tokens remain a
      backward-compatible fallback for older database records.
    - If stressed and healthy provenance are both present, return None rather
      than forcing one condition.
    - If only one condition is represented, return that condition with the
      strongest supporting evidence contribution found.

    This does NOT claim that a morphology such as trichome width is itself
    adverse. It says only that the predicted response is directly supported by
    an observation recorded under that biological-condition context.
    """

    observed_conditions = {
        "healthy": [],
        "stressed": [],
    }

    for prediction in predictions:

        prediction_score = prediction.get(
            "evidence_score",
            0.0
        )

        if prediction_score is None:
            prediction_score = 0.0

        if prediction_score < minimum_evidence_score:
            continue

        for evidence in prediction.get(
            "supporting_evidence",
            []
        ):

            condition = str(
                evidence.get(
                    "observation_condition",
                    ""
                )
            ).strip().lower()

            if condition not in {
                "healthy",
                "stressed",
            }:

                observation_id = str(
                    evidence.get(
                        "observation_id",
                        ""
                    )
                ).strip().lower()

                if not observation_id:
                    continue

                if "_stressed_" in observation_id:
                    condition = "stressed"
                elif "_healthy_" in observation_id:
                    condition = "healthy"
                else:
                    continue

            evidence_strength = evidence.get(
                "evidence_strength",
                prediction_score
            )

            try:
                evidence_strength = float(
                    evidence_strength
                )
            except (
                TypeError,
                ValueError
            ):
                evidence_strength = float(
                    prediction_score
                )

            observed_conditions[
                condition
            ].append(
                evidence_strength
            )

    has_healthy = bool(
        observed_conditions[
            "healthy"
        ]
    )

    has_stressed = bool(
        observed_conditions[
            "stressed"
        ]
    )

    # Mixed provenance is intentionally left unresolved.
    if has_healthy and has_stressed:
        return None

    if has_stressed:
        return {
            "condition":
                "stressed",

            "confidence":
                max(
                    observed_conditions[
                        "stressed"
                    ]
                ),

            "source":
                "observation_condition_evidence",
        }

    if has_healthy:
        return {
            "condition":
                "healthy",

            "confidence":
                max(
                    observed_conditions[
                        "healthy"
                    ]
                ),

            "source":
                "observation_condition_evidence",
        }

    return None


def predictions_have_explicit_contextual_evidence(
    predictions,
    minimum_evidence_score=0.0
):
    """
    Return True when retained predictions are explicitly contextual only.

    A contextual observation supports a response association but does not
    justify a healthy/stressed label. In that situation the simulator should
    leave the overall condition unresolved instead of asking the qualitative
    classifier to manufacture a binary interpretation of new evidence it was
    not trained to understand.

    If no explicit observation-condition metadata is present, return False so
    legacy database records retain their existing classifier fallback.
    """

    explicit_conditions = set()

    for prediction in predictions:

        prediction_score = prediction.get(
            "evidence_score",
            0.0
        )

        if prediction_score is None:
            prediction_score = 0.0

        if prediction_score < minimum_evidence_score:
            continue

        for evidence in prediction.get(
            "supporting_evidence",
            []
        ):

            condition = str(
                evidence.get(
                    "observation_condition",
                    ""
                )
            ).strip().lower()

            if condition:
                explicit_conditions.add(
                    condition
                )

    return (
        bool(explicit_conditions)
        and
        explicit_conditions.issubset(
            {"contextual"}
        )
    )


# ============================================================
# SIMPLE WORLD SIMULATOR
# ============================================================

class SimpleWorldSimulator:
    """
    Evidence-based microorganism world simulator.

    The simulator uses the current water-quality state to
    predict organism responses and writes those responses into
    a new WorldState.

    It also supports explicit alternative-state hypotheses,
    such as different morphotypes observed under the same
    environmental conditions.
    """

    def __init__(
        self,
        top_k=10,
        minimum_evidence_score=0.0
    ):
        """
        Parameters
        ----------
        top_k:
            Maximum number of predicted responses returned by
            the underlying world model.

        minimum_evidence_score:
            Predictions below this evidence score will not be
            written into the next state or retained in explicit
            alternative-state hypotheses.
        """

        self.top_k = top_k

        self.minimum_evidence_score = (
            minimum_evidence_score
        )

        # Load the optional qualitative healthy/stressed classifier once.
        # Direct response evidence remains authoritative when present.

        self.biological_state_model = (
            load_biological_state_model()
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    def validate_state(
        self,
        state
    ):
        """
        Check that the supplied object can be simulated.
        """

        if not isinstance(
            state,
            WorldState
        ):

            raise TypeError(
                "Simulation requires a WorldState."
            )

        if not state.species:

            raise ValueError(
                "WorldState has no microorganism species."
            )

        if not state.water_state:

            raise ValueError(
                "WorldState has no water-quality factors. "
                "At least one environmental factor is required "
                "for forward simulation."
            )


    # ========================================================
    # ONE ORDINARY SIMULATION STEP
    # ========================================================

    def simulate_step(
        self,
        state
    ):
        """
        Generate one predicted state transition.

        The original state is not modified.

        Example
        -------

        Current state:

            species = Nitzschia palea
            petrol = 10 uL/L

        Predicted next state:

            species = Nitzschia palea
            petrol = 10 uL/L
            cell_death = 20.8 percent
            oil_production = increased
            pore_percolation = present
        """

        # ----------------------------------------------------
        # VALIDATE INPUT STATE
        # ----------------------------------------------------

        self.validate_state(
            state
        )

        # ----------------------------------------------------
        # RUN EXISTING FORWARD WORLD MODEL
        # ----------------------------------------------------

        result = predict_response(
            species=state.species,

            water_state=deepcopy(
                state.water_state
            ),

            top_k=self.top_k
        )

        # ----------------------------------------------------
        # CREATE NEW STATE
        # ----------------------------------------------------

        next_state = state.copy()

        predictions = result.get(
            "predictions",
            []
        )

        # Keep one threshold-filtered list for the next-state writer,
        # adverse-condition resolver and response-aware visual matcher.

        retained_predictions = [
            prediction
            for prediction in predictions
            if (
                prediction.get(
                    "evidence_score",
                    0.0
                )
                >=
                self.minimum_evidence_score
            )
        ]

        if retained_predictions:

            no_prediction_reason = None

        elif predictions:

            no_prediction_reason = (
                "below_evidence_threshold"
            )

        else:

            no_prediction_reason = (
                "no_matching_evidence"
            )

        classifier_environment_terms = []
        classifier_input_supported = None

        # ----------------------------------------------------
        # WRITE PREDICTIONS INTO NEXT STATE
        # ----------------------------------------------------

        for prediction in retained_predictions:

            evidence_score = prediction.get(
                "evidence_score",
                0.0
            )

            response_type = prediction.get(
                "response_type"
            )

            response_value = prediction.get(
                "response_value"
            )

            independent_observations = (
                prediction.get(
                    "independent_observations"
                )
            )

            if response_type is None:

                continue

            next_state.set_response(
                response_type=response_type,

                value=response_value,

                evidence_score=evidence_score,

                independent_observations=(
                    independent_observations
                )
            )

        # ----------------------------------------------------
        # RESOLVE OVERALL BIOLOGICAL CONDITION
        # ----------------------------------------------------

        # Direct adverse response evidence takes priority over the
        # broader qualitative classifier. This prevents an evidence-
        # supported adverse response such as cell death from being
        # presented as a healthy state.

        direct_condition = (
            condition_from_response_evidence(
                predictions=retained_predictions,
                minimum_evidence_score=(
                    self.minimum_evidence_score
                )
            )
        )

        if direct_condition is not None:

            predicted_condition = (
                direct_condition[
                    "condition"
                ]
            )

            condition_confidence = (
                direct_condition[
                    "confidence"
                ]
            )

            condition_source = (
                direct_condition[
                    "source"
                ]
            )

        else:

            # ------------------------------------------------
            # SECONDARY DIRECT CONTEXT:
            # OBSERVATION CONDITION PROVENANCE
            # ------------------------------------------------
            #
            # A response such as trichome width is not inherently
            # adverse, so it should not be placed in
            # ADVERSE_RESPONSE_TYPES. However, if the retained
            # response evidence comes only from observations that
            # are explicitly recorded in the stressed (or healthy)
            # dataset, that provenance is more specific to the
            # current prediction than the broad qualitative
            # classifier.
            #
            # Mixed healthy/stressed provenance remains unresolved
            # here and falls through to the classifier.

            observation_condition = (
                condition_from_observation_context(
                    predictions=retained_predictions,
                    minimum_evidence_score=(
                        self.minimum_evidence_score
                    )
                )
            )

            if observation_condition is not None:

                predicted_condition = (
                    observation_condition[
                        "condition"
                    ]
                )

                condition_confidence = (
                    observation_condition[
                        "confidence"
                    ]
                )

                condition_source = (
                    observation_condition[
                        "source"
                    ]
                )

            else:

                contextual_only = (
                    predictions_have_explicit_contextual_evidence(
                        predictions=retained_predictions,
                        minimum_evidence_score=(
                            self.minimum_evidence_score
                        )
                    )
                )

                if contextual_only:

                    predicted_condition = None
                    condition_confidence = None
                    condition_source = (
                        "contextual_evidence_no_condition"
                    )

                else:

                    classifier_condition = (
                        predict_biological_condition(
                            model=(
                                self.biological_state_model
                            ),
                            species=state.species,
                            water_state=state.water_state
                        )
                    )

                    classifier_environment_terms = list(
                        classifier_condition.get(
                            "recognized_environment_terms",
                            [],
                        )
                    )

                    classifier_input_supported = (
                        classifier_condition.get(
                            "environment_input_supported"
                        )
                    )

                    predicted_condition = (
                        classifier_condition[
                            "condition"
                        ]
                    )

                    condition_confidence = (
                        classifier_condition[
                            "confidence"
                        ]
                    )

                    condition_source = (
                        "qualitative_state_classifier"
                        if predicted_condition is not None
                        else (
                            "qualitative_state_classifier_no_input_coverage"
                            if classifier_input_supported is False
                            else None
                        )
                    )

        # ----------------------------------------------------
        # RESOLVE EVIDENCE-BACKED VISUAL STATE
        # ----------------------------------------------------

        visual_state = (
            resolve_visual_state(
                species=state.species,
                predicted_condition=(
                    predicted_condition
                ),
                water_state=(
                    state.water_state
                ),
                predicted_responses=(
                    retained_predictions
                )
            )
        )

        # ----------------------------------------------------
        # RETURN COMPLETE TRANSITION
        # ----------------------------------------------------

        return SimulationResult(
            current_state=state.copy(),

            next_state=next_state,

            raw_result=result,

            predicted_condition=(
                predicted_condition
            ),

            condition_confidence=(
                condition_confidence
            ),

            condition_source=(
                condition_source
            ),

            classifier_environment_terms=(
                classifier_environment_terms
            ),

            classifier_input_supported=(
                classifier_input_supported
            ),

            no_prediction_reason=(
                no_prediction_reason
            ),

            visual_state=(
                visual_state
            )
        )


    # ========================================================
    # ALTERNATIVE / BRANCHING SIMULATION
    # ========================================================

    def simulate_possible_next_states(
        self,
        state
    ):
        """
        Find explicit alternative biological states associated
        with the current water-quality state.

        This method does NOT replace simulate_step().

        simulate_step()
            Returns the ordinary evidence-based next WorldState.

        simulate_possible_next_states()
            Runs the ordinary simulation and additionally checks
            whether the database contains explicit alternative
            states such as morphotypes.

        Example
        -------

        Euglena viridis
        +
        eutrophication

        may support:

            Morphotype I

        OR

            Morphotype II

        Important
        ---------
        The returned states are evidence-supported alternatives.

        They are not probabilities.

        The method also does not claim that the supplied
        environmental factor causes the morphotype. The current
        evidence may represent co-observation rather than a
        causal experimental relationship.
        """

        # ----------------------------------------------------
        # VALIDATE INPUT STATE
        # ----------------------------------------------------

        self.validate_state(
            state
        )

        # ----------------------------------------------------
        # RUN NORMAL SIMULATION
        # ----------------------------------------------------

        deterministic_result = (
            self.simulate_step(
                state
            )
        )

        # ----------------------------------------------------
        # GET WATER-QUALITY FACTOR NAMES
        # ----------------------------------------------------

        water_factors = list(
            state.water_state.keys()
        )

        # ----------------------------------------------------
        # FIND EXPLICIT STATE HYPOTHESES
        # ----------------------------------------------------

        groups = (
            get_state_hypotheses_for_water(
                species=state.species,

                water_factors=water_factors
            )
        )

        # ----------------------------------------------------
        # COLLECT STATES
        # ----------------------------------------------------

        possible_states = []

        seen_case_ids = set()

        for group in groups:

            for hypothesis in group.states:

                # --------------------------------------------
                # APPLY EVIDENCE THRESHOLD
                # --------------------------------------------

                retained_responses = []

                for response in hypothesis.responses:

                    evidence_strength = (
                        response.evidence_strength
                    )

                    # If there is no relationship evidence
                    # strength, we currently do not use the
                    # response for branching simulation.

                    if evidence_strength is None:

                        continue

                    if (
                        evidence_strength
                        <
                        self.minimum_evidence_score
                    ):

                        continue

                    retained_responses.append(
                        response
                    )

                # --------------------------------------------
                # NO SUPPORTED RESPONSES SURVIVED
                # --------------------------------------------

                if not retained_responses:

                    continue

                # --------------------------------------------
                # AVOID DUPLICATE HYPOTHESES
                # --------------------------------------------

                if (
                    hypothesis.case_id
                    in
                    seen_case_ids
                ):

                    continue

                seen_case_ids.add(
                    hypothesis.case_id
                )

                # --------------------------------------------
                # COPY THE HYPOTHESIS
                #
                # We create a new object instead of modifying
                # the original database-derived hypothesis.
                # --------------------------------------------

                filtered_hypothesis = (
                    StateHypothesis(

                        species=(
                            hypothesis.species
                        ),

                        observation_id=(
                            hypothesis.observation_id
                        ),

                        case_id=(
                            hypothesis.case_id
                        ),

                        case_label=(
                            hypothesis.case_label
                        ),

                        case_type=(
                            hypothesis.case_type
                        ),

                        morphotype=(
                            hypothesis.morphotype
                        ),

                        confidence=(
                            hypothesis.confidence
                        ),

                        responses=(
                            retained_responses
                        )
                    )
                )

                possible_states.append(
                    filtered_hypothesis
                )

        # ----------------------------------------------------
        # RETURN BRANCHING RESULT
        # ----------------------------------------------------

        return BranchingSimulationResult(

            current_state=(
                state.copy()
            ),

            possible_states=(
                possible_states
            ),

            deterministic_result=(
                deterministic_result
            )
        )


# ============================================================
# DISPLAY ORDINARY TRANSITION
# ============================================================

def print_transition(
    simulation_result
):
    """
    Print one ordinary state transition in a readable form.

    The biological condition and visual-state sections are displayed
    between the current state and the predicted t+1 state so that the
    evidence-based transition remains easy to inspect.
    """

    print()

    print(
        "=" * 70
    )

    print(
        "WORLD MODEL STATE TRANSITION"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "STATE AT t"
    )

    print()

    print(
        simulation_result
        .current_state
        .summary()
    )

    # ========================================================
    # BIOLOGICAL CONDITION
    # ========================================================

    print()

    print(
        "-" * 70
    )

    print()

    print(
        "PREDICTED BIOLOGICAL CONDITION"
    )

    print()

    if (
        simulation_result
        .predicted_condition
        is not None
    ):

        print(
            "Condition:",
            simulation_result
            .predicted_condition
        )

        if (
            simulation_result
            .condition_confidence
            is not None
        ):

            print(
                "Confidence / evidence score:",
                round(
                    simulation_result
                    .condition_confidence,
                    4
                )
            )

        print(
            "Condition source:",
            simulation_result
            .condition_source
        )

    else:

        print(
            "No biological-condition prediction available."
        )

    # ========================================================
    # VISUAL STATE
    # ========================================================

    print()

    print(
        "SELECTED VISUAL REFERENCE"
    )

    print()

    visual_state = (
        simulation_result
        .visual_state
    )

    if visual_state is None:

        print(
            "No visual-state prediction available."
        )

    else:

        print(
            "Reference record:",
            visual_state.evidence_record_id
        )

        print(
            "Image:",
            visual_state.image_path
        )

        print(
            "Similarity:",
            visual_state.evidence_similarity
        )

        if visual_state.phenotype:

            print(
                "Phenotype documented in reference:",
                visual_state.phenotype
            )

        if visual_state.visual_features:

            print(
                "Features documented in selected reference:"
            )

            for feature in (
                visual_state
                .visual_features
            ):

                print(
                    f"  - {feature}"
                )

            print(
                "These reference annotations are not additional effects "
                "predicted for the current water condition."
            )

        if visual_state.description:

            print()

            print(
                visual_state.description
            )

    print()

    print(
        "-" * 70
    )

    print()

    print(
        "                     WORLD MODEL"
    )

    print(
        "                          |"
    )

    print(
        "                          v"
    )

    print()

    print(
        "-" * 70
    )

    print()

    print(
        "PREDICTED STATE AT t+1"
    )

    print()

    print(
        simulation_result
        .next_state
        .summary()
    )


# ============================================================
# DISPLAY ALTERNATIVE STATES
# ============================================================

def print_possible_next_states(
    branching_result
):
    """
    Display alternative next-state hypotheses.
    """

    print()

    print(
        "=" * 70
    )

    print(
        "BRANCHING WORLD MODEL SIMULATION"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "CURRENT STATE"
    )

    print()

    print(
        branching_result
        .current_state
        .summary()
    )

    print()

    print(
        "-" * 70
    )

    print()

    # --------------------------------------------------------
    # NO ALTERNATIVE STATES
    # --------------------------------------------------------

    if not branching_result.possible_states:

        print(
            "No explicit alternative-state hypotheses "
            "were found."
        )

        print()

        print(
            "The ordinary evidence-based simulation "
            "remains available as the predicted next state."
        )

        return

    # --------------------------------------------------------
    # ALTERNATIVE STATES FOUND
    # --------------------------------------------------------

    print(
        "POSSIBLE NEXT STATES:"
    )

    print(
        branching_result
        .number_of_possible_states
    )

    print()

    if branching_result.has_alternatives:

        print(
            "These states are alternatives. "
            "They should not be merged into one organism."
        )

    else:

        print(
            "One explicit state hypothesis was found."
        )

    # --------------------------------------------------------
    # DISPLAY EACH HYPOTHESIS
    # --------------------------------------------------------

    for index, hypothesis in enumerate(
        branching_result.possible_states,
        start=1
    ):

        print()

        print(
            "=" * 70
        )

        print(
            f"POSSIBLE STATE {index}"
        )

        print(
            "=" * 70
        )

        print()

        print(
            f"State:      "
            f"{hypothesis.case_label}"
        )

        print(
            f"Morphotype: "
            f"{hypothesis.morphotype}"
        )

        print(
            f"Case ID:    "
            f"{hypothesis.case_id}"
        )

        print(
            f"Observation:"
            f" {hypothesis.observation_id}"
        )

        print(
            f"Case confidence: "
            f"{hypothesis.confidence}"
        )

        print()

        print(
            "Predicted / associated biological features:"
        )

        for response in hypothesis.responses:

            print()

            print(
                f"  {response.response_type}: "
                f"{response.value}"
            )

            print(
                f"    Category: "
                f"{response.response_category}"
            )

            print(
                f"    Evidence strength: "
                f"{response.evidence_strength}"
            )

            if response.associated_factors:

                print(
                    "    Associated water factors: "
                    +
                    ", ".join(
                        response.associated_factors
                    )
                )

    print()

    print(
        "-" * 70
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "The model does not currently assign probabilities "
        "to these alternative states."
    )

    print(
        "These states represent evidence-supported possibilities "
        "associated with the supplied environmental conditions."
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # CREATE SIMULATOR
    # --------------------------------------------------------

    simulator = SimpleWorldSimulator(
        top_k=10,
        minimum_evidence_score=0.0
    )


    # ========================================================
    # TEST 1
    # NITZSCHIA PALEA + PETROL 10 uL/L
    # ========================================================

    state_1 = WorldState(

        species="Nitzschia palea",

        water_state={
            "petrol": {
                "value": 10,
                "unit": "uL/L"
            }
        }
    )

    result_1 = simulator.simulate_step(
        state_1
    )

    print_transition(
        result_1
    )


    # ========================================================
    # VERIFY ORIGINAL STATE
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "IMMUTABILITY CHECK"
    )

    print(
        "=" * 70
    )

    print(
        "Responses in original state:",
        len(
            state_1.organism_state
        )
    )

    print(
        "Responses in predicted state:",
        len(
            result_1
            .next_state
            .organism_state
        )
    )


    # ========================================================
    # TEST 2
    # NITZSCHIA PALEA + PETROL 100 uL/L
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "SECOND SIMULATION"
    )

    print(
        "=" * 70
    )

    state_2 = WorldState(

        species="Nitzschia palea",

        water_state={
            "petrol": {
                "value": 100,
                "unit": "uL/L"
            }
        }
    )

    result_2 = simulator.simulate_step(
        state_2
    )

    print_transition(
        result_2
    )


    # ========================================================
    # TEST 3
    # EVIDENCE THRESHOLD
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "EVIDENCE THRESHOLD TEST"
    )

    print(
        "=" * 70
    )

    strong_simulator = SimpleWorldSimulator(
        top_k=10,
        minimum_evidence_score=0.9
    )

    result_3 = (
        strong_simulator.simulate_step(
            state_1
        )
    )

    print()

    print(
        "Minimum evidence score: 0.9"
    )

    print()

    print(
        result_3
        .next_state
        .summary()
    )


    # ========================================================
    # TEST 4
    # ALTERNATIVE EUGLENA STATES
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "ALTERNATIVE STATE SIMULATION TEST"
    )

    print(
        "=" * 70
    )

    euglena_state = WorldState(

        species="Euglena viridis",

        water_state={
            "eutrophication": {
                "value": "present"
            }
        }
    )

    branching_result = (
        simulator.simulate_possible_next_states(
            euglena_state
        )
    )

    print_possible_next_states(
        branching_result
    )


    # ========================================================
    # TEST 5
    # BRANCHING EVIDENCE THRESHOLD
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "BRANCHING EVIDENCE THRESHOLD TEST"
    )

    print(
        "=" * 70
    )

    strong_branching_result = (
        strong_simulator
        .simulate_possible_next_states(
            euglena_state
        )
    )

    print()

    print(
        "Minimum evidence score: 0.9"
    )

    print()

    print(
        "Possible states retained:",
        strong_branching_result
        .number_of_possible_states
    )

    if (
        strong_branching_result
        .number_of_possible_states
        == 0
    ):

        print()

        print(
            "Expected result: the current morphotype "
            "relationships have evidence strength 0.6, "
            "so they are removed by a 0.9 threshold."
        )
