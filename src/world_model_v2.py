"""
world_model_v2.py

Combines the existing world-model components into one V2 result.

V2 objective:
    microorganism + water quality
            ↓
    biological state / responses
            ↓
    evidence-backed reference
            ↓
    illustrative visual simulation
            ↓
    biological interpretation
"""

from simulator import SimpleWorldSimulator
from v2_visual_simulator import simulate_visual_appearance
from world_state import WorldState


def responses_indicate_organism_absence(responses):
    """Detect only the explicit whole-organism presence=absent result."""

    absence_values = {
        "absent",
        "not detected",
        "not found",
        "not observed",
        "undetected",
    }

    for response in responses or []:

        response_type = str(
            response.get("response_type", "")
        ).strip().lower().replace("_", " ")

        response_value = str(
            response.get("response_value", "")
        ).strip().lower().replace("_", " ")

        if response_type == "presence" and response_value in absence_values:
            return True

    return False


def build_interpretation(
    species,
    water_quality,
    condition,
    responses,
    condition_source=None,
    no_prediction_reason=None,
):
    """Create a concise biological interpretation."""

    # --------------------------------------------------------
    # Water-quality description
    # --------------------------------------------------------

    water_parts = []

    for factor, details in water_quality.items():

        if isinstance(details, dict):
            value = details.get("value")
            unit = details.get("unit")

            if value is not None:
                text = f"{factor} at {value}"

                if unit:
                    text += f" {unit}"

                water_parts.append(text)

            else:
                water_parts.append(factor)

        else:
            water_parts.append(f"{factor}: {details}")

    water_text = ", ".join(water_parts)

    # --------------------------------------------------------
    # Response description
    # --------------------------------------------------------

    response_parts = []

    for response in responses:

        response_type = response.get(
            "response_type",
            ""
        ).replace("_", " ")

        response_value = response.get(
            "response_value",
            ""
        )

        if response_type and response_value:
            response_parts.append(
                f"{response_type} ({response_value})"
            )

    # --------------------------------------------------------
    # Build interpretation
    # --------------------------------------------------------

    if responses_indicate_organism_absence(responses):

        condition_phrase = (
            f"is predicted to be {condition}"
            if condition
            else "does not have a resolved healthy/stressed state"
        )

        return (
            f"Under the supplied water condition ({water_text}), "
            f"{species} {condition_phrase}. The evidence-backed response "
            "records the organism as absent from the supporting treatment "
            "observation. A condition-specific organism appearance is "
            "therefore not applicable, and absence is not being "
            "reinterpreted as mortality."
        )

    if not condition:

        if (
            condition_source
            ==
            "qualitative_state_classifier_no_input_coverage"
        ):

            return (
                f"Under the supplied water condition ({water_text}), no "
                f"overall healthy/stressed state is assigned for {species} "
                "because the qualitative classifier does not recognise a "
                "supported environmental factor term in the supplied input. "
                "A species-only baseline prediction is not used."
            )

        if response_parts:

            response_text = ", ".join(response_parts)

            response_sentence = (
                "The evidence-backed biological responses include "
                f"{response_text}. "
            )

        else:

            if (
                no_prediction_reason
                ==
                "no_matching_evidence"
            ):

                response_sentence = (
                    "No specific biological responses matched the "
                    "supplied water conditions. "
                )

            elif (
                no_prediction_reason
                ==
                "below_evidence_threshold"
            ):

                response_sentence = (
                    "Matching response evidence was found, but no "
                    "specific biological responses met the current "
                    "evidence threshold. "
                )

            else:

                response_sentence = (
                    "No specific biological responses matched the "
                    "supplied water conditions at the current evidence "
                    "threshold. "
                )

        return (
            f"Under the supplied water condition ({water_text}), "
            f"{species} does not have enough condition-specific evidence "
            f"to be classified as healthy or stressed. "
            f"{response_sentence}"
            "A state-neutral exact-species reference may be displayed, but "
            "no condition-specific visual transformation is justified."
        )

    condition_text = (
        f"is predicted to be {condition}"
    )

    if response_parts:

        response_text = ", ".join(response_parts)

        return (
            f"Under the supplied water condition ({water_text}), "
            f"{species} {condition_text}. "
            f"The evidence-backed biological responses include "
            f"{response_text}. "
            f"The visual layer selects a same-species reference by explicit "
            f"response and environmental-context agreement. Any documented "
            f"features in that reference remain reference annotations, not "
            f"additional effects predicted for this water condition. A V2 "
            f"transformation is applied only when an explicit static-image "
            f"response rule is available."
        )

    if (
        no_prediction_reason
        ==
        "no_matching_evidence"
    ):

        no_response_sentence = (
            "No specific biological responses matched the supplied "
            "water conditions. "
        )

    elif (
        no_prediction_reason
        ==
        "below_evidence_threshold"
    ):

        no_response_sentence = (
            "Matching response evidence was found, but no specific "
            "biological responses met the current evidence threshold. "
        )

    else:

        no_response_sentence = (
            "No specific biological responses matched the supplied water "
            "conditions at the current evidence threshold. "
        )

    return (
        f"Under the supplied water condition ({water_text}), "
        f"{species} {condition_text}. "
        f"{no_response_sentence}The visual output should therefore "
        f"be interpreted as an evidence-backed visual analogue rather "
        f"than a condition-specific morphological prediction."
    )


def infer_condition_from_responses(responses):
    """
    Infer the overall biological condition from the responses returned
    by the evidence-based simulator.

    This is used because WorldState stores biological responses but does
    not itself store a separate predicted_condition attribute.
    """

    if not responses:
        return "healthy"

    # Responses that clearly indicate biological stress or damage.
    stress_response_types = {
        "cell_death",
        "cell_mortality_fraction",
        "reduced_viability",
        "frustule_breakage",
        "morphological_change",
        "morphological_effect",
        "sheath_loss",
        "growth_reduction",
        "growth_inhibition",
        "lipid_body_change",
        "pore_percolation",
        "chloroplast_damage",
        "cell_damage",
        "membrane_damage",
        "oxidative_stress",
    }

    for response in responses:

        response_type = str(
            response.get("response_type", "")
        ).strip().lower()

        if response_type in stress_response_types:
            return "stressed"

    # If the evidence model returned biological responses but none are
    # explicitly classified above, avoid claiming damage unsupported
    # by the current rule set.
    return "healthy"


def simulate_world_v2(
    species,
    water_state,
    minimum_evidence_score=0.0,
):
    """
    Run the complete V2 pipeline.

    Returns one dictionary suitable for Streamlit.
    """

    simulator = SimpleWorldSimulator(
        minimum_evidence_score=minimum_evidence_score
    )

    # --------------------------------------------------------
    # 1. Create current world state
    # --------------------------------------------------------

    state = WorldState(
        species=species,
        water_state=water_state
    )

    # --------------------------------------------------------
    # 2. Existing biological simulation
    # --------------------------------------------------------

    result = simulator.simulate_step(
        state
    )

    # --------------------------------------------------------
    # 3. Extract predicted microorganism responses
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Extract predicted biological condition
    # --------------------------------------------------------

    condition = getattr(
        result,
        "predicted_condition",
        None
    )

    condition_confidence = getattr(
        result,
        "condition_confidence",
        None
    )

    condition_source = getattr(
        result,
        "condition_source",
        None
    )

    # --------------------------------------------------------
    # Extract predicted microorganism responses
    # --------------------------------------------------------

    clean_responses = []

    next_state = getattr(
        result,
        "next_state",
        None
    )

    if next_state is not None:

        for response_type, response_data in (
            next_state.organism_state.items()
        ):

            clean_responses.append({
                "response_type": response_type,
                "response_value": response_data.get(
                    "value"
                ),
                "evidence_score": response_data.get(
                    "evidence_score"
                ),
                "independent_observations": response_data.get(
                    "independent_observations"
                ),
            })

    # --------------------------------------------------------
    # 5. V2 visual simulation
    # --------------------------------------------------------

    v1_visual_state = getattr(
        result,
        "visual_state",
        None,
    )

    preferred_reference_record_id = getattr(
        v1_visual_state,
        "evidence_record_id",
        None,
    )

    visual = simulate_visual_appearance(
        species=species,
        biological_state=condition,
        predicted_responses=clean_responses,
        preferred_reference_record_id=(
            preferred_reference_record_id
        ),
    )

    # --------------------------------------------------------
    # 6. Biological interpretation
    # --------------------------------------------------------

    interpretation = build_interpretation(
        species=species,
        water_quality=water_state,
        condition=condition,
        responses=clean_responses,
        condition_source=condition_source,
        no_prediction_reason=getattr(
            result,
            "no_prediction_reason",
            None,
        ),
    )

    # --------------------------------------------------------
    # 7. Unified V2 output
    # --------------------------------------------------------

    return {
        "species": species,

        "water_quality": water_state,

        "predicted_condition": condition,
        "condition_confidence": condition_confidence,
        "condition_source": condition_source,

        "classifier_environment_terms": list(
            getattr(
                result,
                "classifier_environment_terms",
                [],
            )
            or []
        ),

        "classifier_input_supported": getattr(
            result,
            "classifier_input_supported",
            None,
        ),

        "no_prediction_reason": getattr(
            result,
            "no_prediction_reason",
            None,
        ),

        "predicted_responses": clean_responses,

        "organism_absent": visual.get(
            "organism_absent",
            False,
        ),

        "reference_record_id": visual.get(
            "reference_record_id"
        ),

        "reference_image_path": visual.get(
            "reference_image_path"
        ),

        "reference_response_match": visual.get(
            "reference_response_match"
        ),

        "simulated_image_path": visual.get(
            "simulated_image_path"
        ),

        "transformation_intensity": visual.get(
            "transformation_intensity"
        ),

        "applied_response_rules": visual.get(
            "applied_response_rules",
            []
        ),

        "interpretation": interpretation,

        "visual_note": visual.get(
            "note"
        ),
    }


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 78)
    print("WORLD MODEL V2 INTEGRATION TEST")
    print("=" * 78)

    result = simulate_world_v2(
        species="Nitzschia palea",
        water_state={
            "petrol": {
                "value": 100,
                "unit": "uL/L",
            }
        },
    )

    print()

    for key, value in result.items():
        print(f"{key}: {value}")

    print()
    print("=" * 78)
    print("V2 INTEGRATION TEST COMPLETE")
    print("=" * 78)
