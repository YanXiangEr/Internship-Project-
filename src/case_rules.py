# ============================================================
# CASE / TREATMENT ASSIGNMENT
# ============================================================
#
# Some spreadsheet rows contain multiple:
#
#   - treatments
#   - concentrations
#   - morphotypes
#   - microscopy panels
#
# We must not treat all factors and responses in such a row as
# belonging to one identical experimental state.
#
# This file defines explicit case splits for known complex rows.
#
# ============================================================


def insert_case(
    connection,
    case_id,
    observation_id,
    case_label=None,
    case_type=None,
    treatment_name=None,
    concentration_min=None,
    concentration_max=None,
    concentration_unit=None,
    duration_numeric=None,
    duration_unit=None,
    panel_label=None,
    morphotype=None,
    notes=None,
    confidence=1.0
):
    """
    Insert one experimental / morphological case.
    """

    connection.execute(
        """
        INSERT OR REPLACE INTO cases (
            case_id,
            observation_id,
            case_label,
            case_type,
            treatment_name,
            concentration_min,
            concentration_max,
            concentration_unit,
            duration_numeric,
            duration_unit,
            panel_label,
            morphotype,
            notes,
            confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            case_id,
            observation_id,
            case_label,
            case_type,
            treatment_name,
            concentration_min,
            concentration_max,
            concentration_unit,
            duration_numeric,
            duration_unit,
            panel_label,
            morphotype,
            notes,
            confidence
        )
    )


# ============================================================
# EUGLENA VIRIDIS — STRESSED ROW 6
# ============================================================

def split_euglena_stressed_row6(connection):

    observation_id = (
        "Euglena_viridis_stressed_row6"
    )

    morphotype_1 = (
        observation_id + "_morphotype_I"
    )

    morphotype_2 = (
        observation_id + "_morphotype_II"
    )

    # --------------------------------------------------------
    # Create cases
    # --------------------------------------------------------

    insert_case(
        connection,
        case_id=morphotype_1,
        observation_id=observation_id,
        case_label="Morphotype I",
        case_type="morphotype",
        morphotype="I",
        notes=(
            "Spindle cells. 55-68 um long, "
            "11-19 um wide. Posterior sharply or "
            "bluntly pointed. Emergent flagellum "
            "similar to or slightly shorter than body."
        )
    )

    insert_case(
        connection,
        case_id=morphotype_2,
        observation_id=observation_id,
        case_label="Morphotype II",
        case_type="morphotype",
        morphotype="II",
        notes=(
            "Cells 38-49 um long and 9-19 um wide. "
            "Posterior elongated to round during movement. "
            "Flagellum approximately twice body length."
        )
    )

    # --------------------------------------------------------
    # Morphotype I measurements
    # --------------------------------------------------------

    connection.execute(
        """
        UPDATE organism_responses
        SET case_id = ?
        WHERE observation_id = ?
          AND response_type = 'cell_length'
          AND value_min = 55
          AND value_max = 68
        """,
        (
            morphotype_1,
            observation_id
        )
    )

    connection.execute(
        """
        UPDATE organism_responses
        SET case_id = ?
        WHERE observation_id = ?
          AND response_type = 'cell_width'
          AND value_min = 11
          AND value_max = 19
        """,
        (
            morphotype_1,
            observation_id
        )
    )

    connection.execute(
        """
        UPDATE organism_responses
        SET case_id = ?
        WHERE observation_id = ?
          AND response_type =
              'flagellum_length_relative_to_body'
          AND value_text =
              'similar_or_slightly_shorter'
        """,
        (
            morphotype_1,
            observation_id
        )
    )

    # The "spindle" description specifically belongs
    # to morphotype I.
    connection.execute(
        """
        UPDATE organism_responses
        SET case_id = ?
        WHERE observation_id = ?
          AND response_type = 'cell_shape'
          AND value_text = 'spindle'
        """,
        (
            morphotype_1,
            observation_id
        )
    )

    # --------------------------------------------------------
    # Morphotype II measurements
    # --------------------------------------------------------

    connection.execute(
        """
        UPDATE organism_responses
        SET case_id = ?
        WHERE observation_id = ?
          AND response_type = 'cell_length'
          AND value_min = 38
          AND value_max = 49
        """,
        (
            morphotype_2,
            observation_id
        )
    )

    connection.execute(
        """
        UPDATE organism_responses
        SET case_id = ?
        WHERE observation_id = ?
          AND response_type = 'cell_width'
          AND value_min = 9
          AND value_max = 19
        """,
        (
            morphotype_2,
            observation_id
        )
    )

    connection.execute(
        """
        UPDATE organism_responses
        SET case_id = ?
        WHERE observation_id = ?
          AND response_type =
              'flagellum_length_relative_to_body'
          AND value_numeric = 2
        """,
        (
            morphotype_2,
            observation_id
        )
    )

    # Posterior description currently contains features
    # from both morphotypes in one normalized record.
    #
    # We remove that combined record and replace it with
    # two case-specific records.

    connection.execute(
        """
        DELETE FROM organism_responses
        WHERE observation_id = ?
          AND response_type = 'posterior_morphology'
        """,
        (observation_id,)
    )

    evidence = (
        "Morphotype I posterior sharply or bluntly "
        "pointed on movement. Morphotype II posterior "
        "elongated to round on movement."
    )

    connection.execute(
        """
        INSERT INTO organism_responses (
            observation_id,
            case_id,
            response_type,
            response_category,
            value_text,
            evidence_text,
            extraction_method,
            confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            observation_id,
            morphotype_1,
            "posterior_morphology",
            "morphology",
            "sharply_or_bluntly_pointed",
            evidence,
            "manual_case_rule",
            1.0
        )
    )

    connection.execute(
        """
        INSERT INTO organism_responses (
            observation_id,
            case_id,
            response_type,
            response_category,
            value_text,
            evidence_text,
            extraction_method,
            confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            observation_id,
            morphotype_2,
            "posterior_morphology",
            "morphology",
            "elongated_to_round",
            evidence,
            "manual_case_rule",
            1.0
        )
    )

    # Environmental factors remain observation-level.
    #
    # Both morphotypes were observed in the same urban
    # drainage environment, so we do NOT assign sewage /
    # nutrients to only one morphotype.


# ============================================================
# NITZSCHIA PALEA — STRESSED ROW 10
# CADMIUM: 10 AND 100 ug/L
# ============================================================

def split_nitzschia_stressed_row10(connection):

    observation_id = (
        "Nitzschia_palea_stressed_row10"
    )

    case_10 = observation_id + "_Cd_10"
    case_100 = observation_id + "_Cd_100"

    insert_case(
        connection,
        case_id=case_10,
        observation_id=observation_id,
        case_label="Cadmium 10 ug/L",
        case_type="concentration",
        treatment_name="cadmium",
        concentration_min=10,
        concentration_max=10,
        concentration_unit="ug/L"
    )

    insert_case(
        connection,
        case_id=case_100,
        observation_id=observation_id,
        case_label="Cadmium 100 ug/L",
        case_type="concentration",
        treatment_name="cadmium",
        concentration_min=100,
        concentration_max=100,
        concentration_unit="ug/L"
    )

    # Assign each water factor to its concentration case.

    connection.execute(
        """
        UPDATE water_quality_factors
        SET case_id = ?
        WHERE observation_id = ?
          AND factor_name = 'cadmium'
          AND value_numeric = 10
        """,
        (
            case_10,
            observation_id
        )
    )

    connection.execute(
        """
        UPDATE water_quality_factors
        SET case_id = ?
        WHERE observation_id = ?
          AND factor_name = 'cadmium'
          AND value_numeric = 100
        """,
        (
            case_100,
            observation_id
        )
    )

    # IMPORTANT:
    #
    # The figure shows deformity TYPES across cadmium
    # exposure, but our current spreadsheet wording does
    # not assign each deformity to one exact concentration.
    #
    # Therefore response case_id remains NULL.
    #
    # We intentionally do NOT say:
    #
    #   10 ug/L -> raphe deviation
    #
    # unless the source explicitly supports it.


# ============================================================
# NITZSCHIA PALEA — STRESSED ROW 18
# PETROL CONCENTRATION RESPONSE
# ============================================================

def split_nitzschia_stressed_row18(connection):

    observation_id = (
        "Nitzschia_palea_stressed_row18"
    )

    low_case = (
        observation_id + "_petrol_5_15"
    )

    above_15_case = (
        observation_id + "_petrol_above_15"
    )

    high_case = (
        observation_id + "_petrol_50_150"
    )

    # --------------------------------------------------------
    # Create cases
    # --------------------------------------------------------

    insert_case(
        connection,
        case_id=low_case,
        observation_id=observation_id,
        case_label="Petrol 5-15 uL/L",
        case_type="concentration_range",
        treatment_name="petrol",
        concentration_min=5,
        concentration_max=15,
        concentration_unit="uL/L",
        panel_label="A7"
    )

    insert_case(
        connection,
        case_id=above_15_case,
        observation_id=observation_id,
        case_label="Petrol >15 uL/L",
        case_type="concentration_threshold",
        treatment_name="petrol",
        concentration_min=15,
        concentration_max=None,
        concentration_unit="uL/L",
        panel_label="A7",
        notes=(
            "Spreadsheet wording states loss in cell "
            "viability with decrease or no change in "
            "lipid bodies above 15 uL/L."
        )
    )

    insert_case(
        connection,
        case_id=high_case,
        observation_id=observation_id,
        case_label="Petrol 50-150 uL/L",
        case_type="concentration_range",
        treatment_name="petrol",
        concentration_min=50,
        concentration_max=150,
        concentration_unit="uL/L",
        panel_label="A7"
    )

    # --------------------------------------------------------
    # Numeric cell-death values
    # --------------------------------------------------------

    connection.execute(
        """
        UPDATE organism_responses
        SET case_id = ?
        WHERE observation_id = ?
          AND response_type = 'cell_death'
          AND value_numeric = 20.8
        """,
        (
            low_case,
            observation_id
        )
    )

    connection.execute(
        """
        UPDATE organism_responses
        SET case_id = ?
        WHERE observation_id = ?
          AND response_type = 'cell_death'
          AND value_numeric = 97.6
        """,
        (
            high_case,
            observation_id
        )
    )

    # Oil production was explicitly reported at 5-15 uL/L.

    connection.execute(
        """
        UPDATE organism_responses
        SET case_id = ?
        WHERE observation_id = ?
          AND response_type = 'oil_production'
        """,
        (
            low_case,
            observation_id
        )
    )

    # Viability / lipid-body response reported above 15 uL/L.

    connection.execute(
        """
        UPDATE organism_responses
        SET case_id = ?
        WHERE observation_id = ?
          AND response_type = 'reduced_viability'
        """,
        (
            above_15_case,
            observation_id
        )
    )

    connection.execute(
        """
        UPDATE organism_responses
        SET case_id = ?
        WHERE observation_id = ?
          AND response_type = 'lipid_body_change'
        """,
        (
            above_15_case,
            observation_id
        )
    )

    # Keep the general petrol/hypoxia/oil factors at
    # observation level because they describe the overall
    # experimental setup.


# ============================================================
# NITZSCHIA PALEA — STRESSED ROW 20
# MULTIPLE EFFLUENT PANELS
# ============================================================

def split_nitzschia_stressed_row20(connection):

    observation_id = (
        "Nitzschia_palea_stressed_row20"
    )

    case_a4 = observation_id + "_A4"
    case_a6 = observation_id + "_A6"
    case_a7 = observation_id + "_A7"
    case_a8 = observation_id + "_A8"

    # --------------------------------------------------------
    # Cases
    # --------------------------------------------------------

    insert_case(
        connection,
        case_id=case_a4,
        observation_id=observation_id,
        case_label="A4",
        case_type="treatment",
        treatment_name="sewage",
        panel_label="E/F",
        notes=(
            "The stressor column labels A4 as sewage, "
            "while the figure-description text says "
            "'detergents (A4)' when discussing morphology. "
            "This inconsistency is preserved rather than "
            "silently resolved."
        ),
        confidence=0.7
    )

    insert_case(
        connection,
        case_id=case_a6,
        observation_id=observation_id,
        case_label="A6",
        case_type="treatment",
        treatment_name="propranolol_hydrochloride",
        panel_label="C"
    )

    insert_case(
        connection,
        case_id=case_a7,
        observation_id=observation_id,
        case_label="A7",
        case_type="treatment",
        treatment_name="petrol",
        panel_label="D"
    )

    insert_case(
        connection,
        case_id=case_a8,
        observation_id=observation_id,
        case_label="A8",
        case_type="treatment",
        treatment_name="sodium_metasilicate",
        concentration_min=1.06e-4,
        concentration_max=1.06e-4,
        concentration_unit="M",
        panel_label="A/B"
    )

    # --------------------------------------------------------
    # Assign existing WATER factors
    # --------------------------------------------------------

    connection.execute(
        """
        UPDATE water_quality_factors
        SET case_id = ?
        WHERE observation_id = ?
          AND factor_name IN (
              'sewage',
              'organic_pollution'
          )
        """,
        (
            case_a4,
            observation_id
        )
    )

    connection.execute(
        """
        UPDATE water_quality_factors
        SET case_id = ?
        WHERE observation_id = ?
          AND factor_name = 'propranolol'
        """,
        (
            case_a6,
            observation_id
        )
    )

    connection.execute(
        """
        UPDATE water_quality_factors
        SET case_id = ?
        WHERE observation_id = ?
          AND factor_name = 'petrol'
        """,
        (
            case_a7,
            observation_id
        )
    )

    connection.execute(
        """
        UPDATE water_quality_factors
        SET case_id = ?
        WHERE observation_id = ?
          AND factor_name = 'sodium_metasilicate'
        """,
        (
            case_a8,
            observation_id
        )
    )

    # --------------------------------------------------------
    # Frustule breakage belongs to A6 and A8.
    #
    # Existing normalizer created one record, so:
    #
    #   1. assign it to A6
    #   2. duplicate it for A8
    # --------------------------------------------------------

    response = connection.execute(
        """
        SELECT
            response_category,
            value_text,
            value_numeric,
            value_min,
            value_max,
            unit,
            qualifier,
            evidence_text,
            confidence

        FROM organism_responses

        WHERE observation_id = ?
          AND response_type = 'frustule_breakage'

        LIMIT 1
        """,
        (observation_id,)
    ).fetchone()

    if response is not None:

        connection.execute(
            """
            UPDATE organism_responses
            SET case_id = ?
            WHERE observation_id = ?
              AND response_type = 'frustule_breakage'
            """,
            (
                case_a6,
                observation_id
            )
        )

        (
            category,
            value_text,
            value_numeric,
            value_min,
            value_max,
            unit,
            qualifier,
            evidence_text,
            confidence
        ) = response

        connection.execute(
            """
            INSERT INTO organism_responses (
                observation_id,
                case_id,
                response_type,
                response_category,
                value_text,
                value_numeric,
                value_min,
                value_max,
                unit,
                qualifier,
                evidence_text,
                extraction_method,
                confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation_id,
                case_a8,
                "frustule_breakage",
                category,
                value_text,
                value_numeric,
                value_min,
                value_max,
                unit,
                qualifier,
                evidence_text,
                "manual_case_rule",
                confidence
            )
        )

    # --------------------------------------------------------
    # A7: petrol percolation into pores
    # --------------------------------------------------------

    connection.execute(
        """
        INSERT INTO organism_responses (
            observation_id,
            case_id,
            response_type,
            response_category,
            value_text,
            evidence_text,
            extraction_method,
            confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            observation_id,
            case_a7,
            "pore_percolation",
            "surface_interaction",
            "present",
            (
                "TEM image description reports "
                "percolation of petrol (A7) in "
                "pores of diatoms."
            ),
            "manual_case_rule",
            1.0
        )
    )

    # --------------------------------------------------------
    # A4: no apparent morphological effect
    # --------------------------------------------------------

    connection.execute(
        """
        INSERT INTO organism_responses (
            observation_id,
            case_id,
            response_type,
            response_category,
            value_text,
            evidence_text,
            extraction_method,
            confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            observation_id,
            case_a4,
            "morphological_effect",
            "morphology",
            "none_observed",
            (
                "Figure description states that panel "
                "(e) shows no effect on cell morphology "
                "in the presence of detergents (A4). "
                "The stressor field labels A4 as sewage, "
                "so treatment identity is flagged as "
                "source-inconsistent."
            ),
            "manual_case_rule",
            0.7
        )
    )


# ============================================================
# RUN ALL CASE RULES
# ============================================================

def apply_case_rules(connection):
    """
    Apply all currently known complex observation splits.

    Add new functions here as more complex rows are identified.
    """

    split_euglena_stressed_row6(
        connection
    )

    split_nitzschia_stressed_row10(
        connection
    )

    split_nitzschia_stressed_row18(
        connection
    )

    split_nitzschia_stressed_row20(
        connection
    )

    connection.commit()