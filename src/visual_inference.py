"""
visual_inference.py

Combines visually observed microorganism features with the
existing inverse world model.

Pipeline:

    VisualObservation
          ↓
    individual visual features
          ↓
    infer_water_quality()
          ↓
    combine evidence across features
          ↓
    ranked water-quality hypotheses

This module does NOT perform image recognition yet.

At this stage, visual features are supplied through a
VisualObservation object. Later, a computer-vision model can
populate the same object automatically.

Important:
    Visual match, feature confidence, and knowledge-base
    evidence are kept as separate concepts.

    The ranking score is only an interpretable ordering
    heuristic. It is NOT a biological probability.
"""


from copy import deepcopy

from world_model import infer_water_quality

from visual_observation import (
    VisualObservation
)


# ============================================================
# VISUAL WATER-QUALITY HYPOTHESIS
# ============================================================

class VisualWaterHypothesis:
    """
    One possible water-quality condition supported by visual
    microorganism observations.
    """

    def __init__(
        self,
        factor_name,
        factor_value=None
    ):

        self.factor_name = (
            factor_name
        )

        self.factor_value = (
            factor_value
        )

        self.matched_features = []

        self.supporting_evidence = []

        self.independent_observations = set()

        self.evidence_scores = []

        self.feature_confidences = []

        # Number of visual features supplied to the complete
        # visual-inference query.
        #
        # This is assigned by
        # infer_water_from_visual_observation().
        self.total_input_features = 0


    # ========================================================
    # ADD SUPPORT
    # ========================================================

    def add_support(
        self,
        feature,
        condition
    ):
        """
        Add support from one observed visual feature.
        """

        self.matched_features.append(
            {
                "response_type":
                    feature.response_type,

                "response_value":
                    feature.value,

                "feature_confidence":
                    feature.confidence,

                "feature_source":
                    feature.source,
            }
        )

        condition_score = (
            condition.get(
                "evidence_score"
            )
        )

        if condition_score is not None:

            self.evidence_scores.append(
                float(
                    condition_score
                )
            )

        if feature.confidence is not None:

            self.feature_confidences.append(
                float(
                    feature.confidence
                )
            )

        independent_count = (
            condition.get(
                "independent_observations"
            )
        )

        # Keep the actual observation IDs where available.
        evidence_records = condition.get(
            "supporting_evidence",
            []
        )

        for evidence in evidence_records:

            observation_id = (
                evidence.get(
                    "observation_id"
                )
            )

            if observation_id:

                self.independent_observations.add(
                    observation_id
                )

            self.supporting_evidence.append(
                deepcopy(
                    evidence
                )
            )

        # Fallback for models that only report a count rather
        # than the actual supporting observation IDs.
        if (
            not evidence_records
            and
            independent_count
        ):

            for index in range(
                int(
                    independent_count
                )
            ):

                fallback_id = (
                    "unknown_observation_"
                    f"{self.factor_name}_"
                    f"{len(self.independent_observations)}_"
                    f"{index}"
                )

                self.independent_observations.add(
                    fallback_id
                )


    # ========================================================
    # DERIVED PROPERTIES
    # ========================================================

    @property
    def matched_feature_count(self):
        """
        Number of distinct visual features supporting this
        hypothesis.
        """

        return len(
            self.matched_features
        )


    @property
    def visual_match_fraction(self):
        """
        Fraction of supplied visual features that support
        this water-quality hypothesis.

        Example:

            3 matching features out of 3 supplied
                -> 1.0

            1 matching feature out of 3 supplied
                -> 0.333...

        This measures visual support.

        It is NOT a biological probability.
        """

        if self.total_input_features <= 0:

            return 0.0

        return (
            self.matched_feature_count
            /
            self.total_input_features
        )


    @property
    def independent_observation_count(self):
        """
        Number of independent knowledge-base observations
        supporting this hypothesis where observation IDs are
        available.
        """

        return len(
            self.independent_observations
        )


    @property
    def mean_evidence_score(self):
        """
        Mean evidence strength from the existing world model.
        """

        if not self.evidence_scores:

            return 0.0

        return (
            sum(
                self.evidence_scores
            )
            /
            len(
                self.evidence_scores
            )
        )


    @property
    def mean_feature_confidence(self):
        """
        Mean confidence of the visual features supporting this
        hypothesis.

        At present these confidences may be manually supplied.

        Later they may come from a computer-vision model.
        """

        if not self.feature_confidences:

            return None

        return (
            sum(
                self.feature_confidences
            )
            /
            len(
                self.feature_confidences
            )
        )


    @property
    def combined_support_score(self):
        """
        Ranking heuristic retained for backward compatibility.

        The score rewards:

            - more matched visual features
            - stronger world-model evidence
            - higher visual-feature confidence

        Formula:

            matched_feature_count
                × mean_evidence_score
                × mean_feature_confidence

        This is NOT a biological probability and should only
        be used for ranking hypotheses.

        In user-facing output it is called the "Ranking score".
        """

        feature_factor = (
            self.matched_feature_count
        )

        evidence_factor = (
            self.mean_evidence_score
        )

        confidence = (
            self.mean_feature_confidence
        )

        if confidence is None:

            confidence = 1.0

        return (
            feature_factor
            *
            evidence_factor
            *
            confidence
        )


    @property
    def ranking_score(self):
        """
        User-facing alias for combined_support_score.

        Keeping the original property allows existing code to
        continue working while making the meaning clearer.
        """

        return (
            self.combined_support_score
        )


    # ========================================================
    # SERIALIZATION
    # ========================================================

    def to_dict(self):

        return {
            "factor_name":
                self.factor_name,

            "factor_value":
                self.factor_value,

            "matched_feature_count":
                self.matched_feature_count,

            "total_input_features":
                self.total_input_features,

            "visual_match_fraction":
                round(
                    self.visual_match_fraction,
                    4
                ),

            "matched_features":
                deepcopy(
                    self.matched_features
                ),

            "independent_observations":
                self.independent_observation_count,

            "mean_evidence_score":
                round(
                    self.mean_evidence_score,
                    4
                ),

            "mean_feature_confidence": (
                round(
                    self.mean_feature_confidence,
                    4
                )
                if
                self.mean_feature_confidence
                is not None
                else None
            ),

            # Keep both names for compatibility.
            "combined_support_score":
                round(
                    self.combined_support_score,
                    4
                ),

            "ranking_score":
                round(
                    self.ranking_score,
                    4
                ),

            "supporting_evidence":
                deepcopy(
                    self.supporting_evidence
                ),
        }


# ============================================================
# VISUAL INFERENCE RESULT
# ============================================================

class VisualInferenceResult:
    """
    Result of visual-feature -> water-quality inference.
    """

    def __init__(
        self,
        visual_observation,
        hypotheses,
        feature_results
    ):

        self.visual_observation = (
            visual_observation
        )

        self.hypotheses = (
            hypotheses
        )

        self.feature_results = (
            feature_results
        )


    def to_dict(self):

        return {
            "visual_observation":
                self.visual_observation.to_dict(),

            "hypotheses": [
                hypothesis.to_dict()
                for hypothesis
                in self.hypotheses
            ],

            "feature_results":
                deepcopy(
                    self.feature_results
                ),
        }


# ============================================================
# INFERENCE
# ============================================================

def infer_water_from_visual_observation(
    visual_observation,
    top_k_per_feature=20,
    top_k_final=20,
    minimum_world_evidence=0.0
):
    """
    Infer possible water-quality conditions from a
    VisualObservation.

    Each visual feature is first passed independently through
    the existing inverse world model.

    Results referring to the same water factor are then
    combined.

    Parameters
    ----------
    visual_observation:
        VisualObservation object.

    top_k_per_feature:
        Maximum inverse results retrieved for each feature.

    top_k_final:
        Maximum combined water hypotheses returned.

    minimum_world_evidence:
        Ignore inverse relationships below this evidence score.

    Important
    ---------
    The output distinguishes:

        1. Visual feature match
        2. Visual-feature confidence
        3. Knowledge-base evidence strength
        4. Independent evidence observations

    The ranking score is only an ordering heuristic.

    It is NOT a probability that the water contains a
    particular condition.
    """

    if not isinstance(
        visual_observation,
        VisualObservation
    ):

        raise TypeError(
            "visual_observation must be a VisualObservation."
        )

    if not visual_observation.features:

        raise ValueError(
            "VisualObservation contains no visual features."
        )

    # ========================================================
    # RUN INVERSE MODEL FEATURE BY FEATURE
    # ========================================================

    hypothesis_lookup = {}

    feature_results = []

    for feature in visual_observation.features:

        observed_response = {
            feature.response_type:
                feature.value
        }

        result = infer_water_quality(
            species=(
                visual_observation.species
            ),

            observed_response=(
                observed_response
            ),

            top_k=(
                top_k_per_feature
            )
        )

        feature_results.append(
            {
                "feature":
                    feature.to_dict(),

                "inverse_result":
                    deepcopy(
                        result
                    )
            }
        )

        conditions = result.get(
            "possible_water_conditions",
            []
        )

        for condition in conditions:

            evidence_score = (
                condition.get(
                    "evidence_score",
                    0.0
                )
            )

            if (
                evidence_score
                <
                minimum_world_evidence
            ):

                continue

            factor_name = (
                condition.get(
                    "factor_name"
                )
            )

            if not factor_name:

                continue

            factor_value = (
                condition.get(
                    "factor_value"
                )
            )

            # ------------------------------------------------
            # Group primarily by factor name.
            #
            # At this stage, values such as "present" usually
            # describe the same environmental hypothesis.
            # ------------------------------------------------

            key = (
                factor_name
            )

            if key not in hypothesis_lookup:

                hypothesis_lookup[
                    key
                ] = VisualWaterHypothesis(
                    factor_name=(
                        factor_name
                    ),

                    factor_value=(
                        factor_value
                    )
                )

            hypothesis_lookup[
                key
            ].add_support(
                feature=feature,
                condition=condition
            )


    # ========================================================
    # REMOVE DUPLICATE FEATURE SUPPORT
    # ========================================================

    for hypothesis in (
        hypothesis_lookup.values()
    ):

        unique_features = {}

        for feature_record in (
            hypothesis.matched_features
        ):

            key = (
                feature_record[
                    "response_type"
                ],

                str(
                    feature_record[
                        "response_value"
                    ]
                )
            )

            unique_features[
                key
            ] = (
                feature_record
            )

        hypothesis.matched_features = list(
            unique_features.values()
        )


    # ========================================================
    # ASSIGN TOTAL VISUAL FEATURE COUNT
    # ========================================================

    total_input_features = len(
        visual_observation.features
    )

    for hypothesis in (
        hypothesis_lookup.values()
    ):

        hypothesis.total_input_features = (
            total_input_features
        )


    # ========================================================
    # SORT HYPOTHESES
    # ========================================================

    hypotheses = sorted(
        hypothesis_lookup.values(),

        key=lambda hypothesis: (
            hypothesis.matched_feature_count,
            hypothesis.ranking_score,
            hypothesis.mean_evidence_score,
            hypothesis.independent_observation_count
        ),

        reverse=True
    )

    hypotheses = hypotheses[
        :top_k_final
    ]


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return VisualInferenceResult(
        visual_observation=(
            visual_observation
        ),

        hypotheses=(
            hypotheses
        ),

        feature_results=(
            feature_results
        )
    )


# ============================================================
# DISPLAY
# ============================================================

def print_visual_inference(
    result
):
    """
    Print visual inverse-inference results in a way that keeps
    visual match and biological evidence clearly separated.
    """

    print()

    print(
        "=" * 70
    )

    print(
        "VISUAL INVERSE WORLD MODEL"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Species: "
        f"{result.visual_observation.species}"
    )

    print()

    print(
        "Observed visual features:"
    )

    for feature in (
        result
        .visual_observation
        .features
    ):

        print(
            f"  - {feature.response_type}: "
            f"{feature.value}"
        )

        if (
            feature.confidence
            is not None
        ):

            print(
                f"      Feature confidence: "
                f"{feature.confidence}"
            )

    print()

    print(
        "Possible water-quality hypotheses:"
    )

    if not result.hypotheses:

        print()

        print(
            "  No supported water-quality hypotheses found."
        )

        return

    for index, hypothesis in enumerate(
        result.hypotheses,
        start=1
    ):

        print()

        print(
            f"{index}. "
            f"{hypothesis.factor_name}"
        )

        if (
            hypothesis.factor_value
            is not None
        ):

            print(
                f"   Value: "
                f"{hypothesis.factor_value}"
            )

        # ----------------------------------------------------
        # VISUAL MATCH
        # ----------------------------------------------------

        print(
            f"   Visual feature match: "
            f"{hypothesis.matched_feature_count}"
            f" / "
            f"{hypothesis.total_input_features}"
        )

        print(
            f"   Visual match fraction: "
            f"{round(hypothesis.visual_match_fraction, 3)}"
        )

        # ----------------------------------------------------
        # KNOWLEDGE-BASE EVIDENCE
        # ----------------------------------------------------

        print(
            f"   Mean world-model evidence: "
            f"{round(hypothesis.mean_evidence_score, 3)}"
        )

        # ----------------------------------------------------
        # VISUAL FEATURE CONFIDENCE
        # ----------------------------------------------------

        if (
            hypothesis.mean_feature_confidence
            is not None
        ):

            print(
                f"   Mean feature confidence: "
                f"{round(hypothesis.mean_feature_confidence, 3)}"
            )

        # ----------------------------------------------------
        # RANKING
        # ----------------------------------------------------

        print(
            f"   Ranking score: "
            f"{round(hypothesis.ranking_score, 3)}"
        )

        # ----------------------------------------------------
        # INDEPENDENT EVIDENCE
        # ----------------------------------------------------

        print(
            f"   Independent evidence observations: "
            f"{hypothesis.independent_observation_count}"
        )

        # ----------------------------------------------------
        # SUPPORTING FEATURES
        # ----------------------------------------------------

        print(
            "   Supporting visual features:"
        )

        for feature in (
            hypothesis.matched_features
        ):

            print(
                f"      - "
                f"{feature['response_type']}"
                f" = "
                f"{feature['response_value']}"
            )

    print()

    print(
        "-" * 70
    )

    print()

    print(
        "INTERPRETATION:"
    )

    print(
        "Visual feature match describes how many supplied "
        "visual features support each hypothesis."
    )

    print(
        "World-model evidence describes the strength of the "
        "relationships stored in the current knowledge base."
    )

    print(
        "Independent evidence observations describe how many "
        "distinct observations support the hypothesis where "
        "observation identifiers are available."
    )

    print(
        "The ranking score is only used to order hypotheses."
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "These outputs represent water-quality conditions "
        "associated with the observed visual features in "
        "the current evidence database."
    )

    print(
        "They are evidence-supported hypotheses, not "
        "biological probabilities."
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # EUGLENA MORPHOTYPE I EXAMPLE
    # ========================================================

    observation = VisualObservation(
        species="Euglena viridis",
        media_type="image",
        source_reference=(
            "manual_Morphotype_I_test"
        )
    )

    observation.add_feature(
        response_type="cell_shape",
        value="spindle",
        confidence=0.92,
        source="manual_test"
    )

    observation.add_feature(
        response_type=(
            "posterior_morphology"
        ),
        value=(
            "sharply_or_bluntly_pointed"
        ),
        confidence=0.88,
        source="manual_test"
    )

    observation.add_feature(
        response_type="cell_length",
        value="55-68 um",
        confidence=0.80,
        source="manual_test"
    )

    result = (
        infer_water_from_visual_observation(
            visual_observation=(
                observation
            ),

            top_k_per_feature=20,

            top_k_final=20,

            minimum_world_evidence=0.0
        )
    )

    print_visual_inference(
        result
    )