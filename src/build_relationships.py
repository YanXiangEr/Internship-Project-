# ============================================================
# RELATIONSHIP / EVIDENCE BUILDER
# ============================================================
#
# Creates defensible links between:
#
#     water_quality_factors
#             ↓
#     organism_responses
#
# It deliberately avoids blindly taking the Cartesian product
# of every factor and every response in a complex observation.
#
# ============================================================


# ============================================================
# DEFAULT RELATIONSHIP WEIGHTS
# ============================================================

RELATIONSHIP_WEIGHTS = {

    "case_specific": 1.00,

    "experimentally_observed": 0.90,

    "co_observed": 0.60,

    "observation_level": 0.45,

    "uncertain": 0.25,
}


# ============================================================
# KNOWN OBSERVATIONAL ENVIRONMENT RECORDS
# ============================================================
#
# These stressed records describe organisms observed/collected
# in environmental conditions rather than controlled treatment
# experiments.
#
# Their factors and biological characteristics are therefore
# co-observed associations, not experimental treatment effects.
# ============================================================

CO_OBSERVED_SIMPLE_OBSERVATIONS = {
    "Euglena_viridis_stressed_row7",

    "Oscillatoria_limosa_stressed_row6",
    "Oscillatoria_limosa_stressed_row7",
    "Oscillatoria_limosa_stressed_row8",
    "Oscillatoria_limosa_stressed_row9",
    "Oscillatoria_limosa_stressed_row10",
    "Oscillatoria_limosa_stressed_row11",

    # Environmental field observations rather than
    # controlled treatment experiments.
    "Nitzschia_palea_stressed_row21",
    "Nitzschia_palea_stressed_row22",
}


# ============================================================
# PRIMARY TREATMENTS IN MULTI-FACTOR EXPERIMENTS
# ============================================================
#
# Some controlled experiments contain one actual treatment plus
# contextual/derived environmental factors.
#
# Only the primary treatment receives
# "experimentally_observed" evidence.
#
# Other factors from the same observation remain useful, but are
# treated as weaker co-observed contextual associations.
# ============================================================

PRIMARY_TREATMENT_FACTORS = {

    "Nitzschia_palea_stressed_row14": {
        "paint"
    },

    "Nitzschia_palea_stressed_row15": {
        "sewage"
    },

    "Nitzschia_palea_stressed_row18": {
        "petrol"
    },

    "Nitzschia_palea_stressed_row19": {
        "sodium_metasilicate"
    },
}


# ============================================================
# INSERT ONE RULE
# ============================================================

def insert_rule(
    connection,
    organism_id,
    observation_id,
    factor_id,
    response_id,
    relationship_type,
    case_id=None,
    direction="water_to_response",
    evidence_strength=None,
    notes=None,
    source_method="automatic_relationship_rule"
):
    """
    Insert one evidence relationship.
    """

    if evidence_strength is None:

        evidence_strength = RELATIONSHIP_WEIGHTS.get(
            relationship_type,
            0.5
        )

    connection.execute(
        """
        INSERT INTO evidence_rules (
            organism_id,
            observation_id,
            case_id,
            factor_id,
            response_id,
            relationship_type,
            evidence_strength,
            direction,
            notes,
            source_method
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            organism_id,
            observation_id,
            case_id,
            factor_id,
            response_id,
            relationship_type,
            evidence_strength,
            direction,
            notes,
            source_method
        )
    )


# ============================================================
# 1. CASE-SPECIFIC RELATIONSHIPS
# ============================================================

def create_case_specific_relationships(
    connection
):
    """
    Strongest relationship type.

    Only link a factor and response when they explicitly share
    the same case_id.
    """

    rows = connection.execute(
        """
        SELECT
            o.organism_id,
            w.observation_id,
            w.case_id,
            w.factor_id,
            r.response_id

        FROM water_quality_factors w

        JOIN organism_responses r
            ON w.observation_id = r.observation_id
           AND w.case_id = r.case_id

        JOIN observations o
            ON w.observation_id = o.observation_id

        WHERE
            w.case_id IS NOT NULL
            AND
            r.case_id IS NOT NULL
        """
    ).fetchall()

    count = 0

    for (
        organism_id,
        observation_id,
        case_id,
        factor_id,
        response_id
    ) in rows:

        insert_rule(
            connection,
            organism_id=organism_id,
            observation_id=observation_id,
            case_id=case_id,
            factor_id=factor_id,
            response_id=response_id,
            relationship_type="case_specific",
            notes=(
                "Water factor and organism response "
                "belong to the same explicitly defined case."
            )
        )

        count += 1

    return count


# ============================================================
# 2. SIMPLE OBSERVATIONS
# ============================================================

def create_simple_observation_relationships(
    connection
):
    """
    For observations that have NO explicit cases:

        factor(s)
             ↓
        response(s)

    We permit links because no internal treatment split has
    been identified.

    Stressed observations are treated as experimentally
    observed associations rather than automatically causal.
    """

    rows = connection.execute(
        """
        SELECT
            o.organism_id,
            o.observation_id,
            o.condition,

            w.factor_id,
            w.factor_name,
            r.response_id

        FROM observations o

        JOIN water_quality_factors w
            ON o.observation_id = w.observation_id

        JOIN organism_responses r
            ON o.observation_id = r.observation_id

        WHERE
            NOT EXISTS (
                SELECT 1
                FROM cases c
                WHERE c.observation_id =
                      o.observation_id
            )
        """
    ).fetchall()

    count = 0

    for (
        organism_id,
        observation_id,
        condition,
        factor_id,
        factor_name,
        response_id
    ) in rows:

        # ------------------------------------------------------------
        # Environmental field observations
        # ------------------------------------------------------------

        if (
            observation_id
            in
            CO_OBSERVED_SIMPLE_OBSERVATIONS
        ):

            relationship_type = (
                "co_observed"
            )

            notes = (
                "Water/environment factor and organism "
                "characteristic were observed together in "
                "an environmental observation. The record "
                "does not establish a controlled treatment "
                "effect or direct causality."
            )

        # ------------------------------------------------------------
        # Controlled multi-factor experiment
        # ------------------------------------------------------------

        elif (
            observation_id
            in
            PRIMARY_TREATMENT_FACTORS
        ):

            primary_factors = (
                PRIMARY_TREATMENT_FACTORS[
                    observation_id
                ]
            )

            if factor_name in primary_factors:

                relationship_type = (
                    "experimentally_observed"
                )

                notes = (
                    "This factor is the primary experimental "
                    "treatment reported for the observation."
                )

            else:

                relationship_type = (
                    "co_observed"
                )

                notes = (
                    "This factor describes contextual or "
                    "secondary environmental information "
                    "associated with the primary experimental "
                    "treatment. It is retained as supporting "
                    "context rather than treated as the direct "
                    "experimental treatment."
                )

        # ------------------------------------------------------------
        # Other stressed experimental observations
        # ------------------------------------------------------------

        elif condition == "stressed":

            relationship_type = (
                "experimentally_observed"
            )

            notes = (
                "Water/environment factor and response "
                "were reported in the same stressed "
                "experimental observation. Association "
                "does not by itself establish direct "
                "causality."
            )

        else:

            relationship_type = (
                "co_observed"
            )

            notes = (
                "Factor and organism characteristic were "
                "observed together."
            )

        insert_rule(
            connection,
            organism_id=organism_id,
            observation_id=observation_id,
            factor_id=factor_id,
            response_id=response_id,
            relationship_type=relationship_type,
            notes=notes
        )

        count += 1

    return count


# ============================================================
# 3. COMPLEX OBSERVATIONS:
#    OBSERVATION-LEVEL FACTOR + CASE RESPONSE
# ============================================================

def create_shared_environment_relationships(
    connection
):
    """
    Some factors apply to the whole observation while
    individual biological features are divided into cases.

    Example:
        Euglena stressed row 6

        sewage/eutrophication = observation-level

        Morphotype I / II responses = case-level

    Here we create weaker 'co_observed' relationships.
    """

    rows = connection.execute(
        """
        SELECT
            o.organism_id,
            o.observation_id,
            r.case_id,
            w.factor_id,
            r.response_id

        FROM observations o

        JOIN water_quality_factors w
            ON o.observation_id =
               w.observation_id

        JOIN organism_responses r
            ON o.observation_id =
               r.observation_id

        WHERE
            w.case_id IS NULL
            AND
            r.case_id IS NOT NULL
            AND
            EXISTS (
                SELECT 1
                FROM cases c
                WHERE c.observation_id =
                      o.observation_id
            )
        """
    ).fetchall()

    count = 0

    for (
        organism_id,
        observation_id,
        case_id,
        factor_id,
        response_id
    ) in rows:

        insert_rule(
            connection,
            organism_id=organism_id,
            observation_id=observation_id,
            case_id=case_id,
            factor_id=factor_id,
            response_id=response_id,
            relationship_type="co_observed",
            notes=(
                "Environmental factor applies to the "
                "overall observation while the organism "
                "response belongs to a specific case. "
                "Relationship is therefore associative, "
                "not case-specific causal evidence."
            )
        )

        count += 1

    return count


# ============================================================
# 4. CASE FACTOR + OBSERVATION-LEVEL RESPONSE
# ============================================================

def create_case_factor_observation_response_links(
    connection
):
    """
    Example:
        Nitzschia cadmium row 10

        Cd 10 and 100 ug/L have explicit cases,
        but individual deformity types are not assigned
        to one exact concentration.

    We create weaker observation-level relationships rather
    than claiming that each deformity belongs specifically to
    each concentration.
    """

    rows = connection.execute(
        """
        SELECT
            o.organism_id,
            o.observation_id,
            w.case_id,
            w.factor_id,
            r.response_id

        FROM observations o

        JOIN water_quality_factors w
            ON o.observation_id =
               w.observation_id

        JOIN organism_responses r
            ON o.observation_id =
               r.observation_id

        WHERE
            w.case_id IS NOT NULL
            AND
            r.case_id IS NULL
        """
    ).fetchall()

    count = 0

    for (
        organism_id,
        observation_id,
        case_id,
        factor_id,
        response_id
    ) in rows:

        insert_rule(
            connection,
            organism_id=organism_id,
            observation_id=observation_id,
            case_id=case_id,
            factor_id=factor_id,
            response_id=response_id,
            relationship_type="observation_level",
            notes=(
                "Treatment/concentration is case-specific, "
                "but the response was only reported at the "
                "overall observation level. Exact "
                "case-response assignment is unknown."
            )
        )

        count += 1

    return count


# ============================================================
# 5. MANUAL OVERRIDE / SPECIAL CASES
# ============================================================

def adjust_known_uncertain_relationships(
    connection
):
    """
    Downgrade relationships where source information contains
    known ambiguity.
    """

    uncertain_note = (
        "Source contains conflicting treatment labels for A4; "
        "relationship retained with reduced confidence."
    )

    connection.execute(
        """
        UPDATE evidence_rules

        SET
            relationship_type = ?,
            evidence_strength = ?,
            notes = ?

        WHERE case_id = ?
        """,
        (
            "uncertain",
            RELATIONSHIP_WEIGHTS["uncertain"],
            uncertain_note,
            "Nitzschia_palea_stressed_row20_A4"
        )
    )


# ============================================================
# DUPLICATE CLEANUP
# ============================================================

def remove_duplicate_relationships(
    connection
):
    """
    For the same factor-response-case combination, retain the
    strongest available relationship.

    Priority:

        case_specific
        experimentally_observed
        co_observed
        observation_level
        uncertain

    Exception:
        uncertain relationships are deliberately preserved
        when manually assigned afterwards.
    """

    rows = connection.execute(
        """
        SELECT
            rule_id,
            organism_id,
            observation_id,
            case_id,
            factor_id,
            response_id,
            evidence_strength

        FROM evidence_rules

        ORDER BY
            evidence_strength DESC,
            rule_id ASC
        """
    ).fetchall()

    seen = set()
    delete_ids = []

    for (
        rule_id,
        organism_id,
        observation_id,
        case_id,
        factor_id,
        response_id,
        evidence_strength
    ) in rows:

        key = (
            organism_id,
            observation_id,
            case_id,
            factor_id,
            response_id
        )

        if key in seen:
            delete_ids.append(
                rule_id
            )

        else:
            seen.add(
                key
            )

    for rule_id in delete_ids:

        connection.execute(
            """
            DELETE FROM evidence_rules
            WHERE rule_id = ?
            """,
            (rule_id,)
        )

def create_case_treatment_relationships(
    connection
):
    """
    Link an observation-level water factor to a case-specific
    response when the case explicitly identifies that same
    treatment.

    Example:

        observation factor:
            petrol

        case:
            Petrol 5-15 uL/L

        response:
            cell_death = 20.8%

    becomes:

        petrol 5-15 uL/L
            -> cell_death 20.8%

        relationship_type = case_specific

    Concentration information remains stored in the cases
    table and can be retrieved through case_id.
    """

    rows = connection.execute(
        """
        SELECT
            o.organism_id,
            o.observation_id,

            c.case_id,
            c.treatment_name,

            w.factor_id,
            w.factor_name,

            r.response_id

        FROM cases c

        JOIN observations o
            ON c.observation_id =
               o.observation_id

        JOIN organism_responses r
            ON r.case_id =
               c.case_id

        JOIN water_quality_factors w
            ON w.observation_id =
               c.observation_id

        WHERE
            r.case_id IS NOT NULL
            AND
            w.case_id IS NULL
            AND
            c.treatment_name IS NOT NULL
        """
    ).fetchall()

    count = 0

    for (
        organism_id,
        observation_id,
        case_id,
        treatment_name,
        factor_id,
        factor_name,
        response_id
    ) in rows:

        treatment = (
            str(treatment_name)
            .lower()
            .strip()
        )

        factor = (
            str(factor_name)
            .lower()
            .strip()
        )

        # ----------------------------------------------------
        # Normalize a few known naming differences.
        # ----------------------------------------------------

        aliases = {
            "propranolol_hydrochloride":
                "propranolol",

            "sodium_metasilicate":
                "sodium_metasilicate",

            "petrol":
                "petrol",

            "cadmium":
                "cadmium",

            "sewage":
                "sewage"
        }

        treatment_normalized = aliases.get(
            treatment,
            treatment
        )

        if factor != treatment_normalized:
            continue

        insert_rule(
            connection,
            organism_id=organism_id,
            observation_id=observation_id,
            case_id=case_id,
            factor_id=factor_id,
            response_id=response_id,
            relationship_type="case_specific",
            notes=(
                "The response belongs to a specific "
                "treatment/concentration case and the "
                "water factor matches that case's "
                "explicit treatment. Concentration "
                "information is stored in the cases table."
            ),
            source_method=(
                "case_treatment_relationship_rule"
            )
        )

        count += 1

    return count


# ============================================================
# MAIN
# ============================================================

def build_relationships(
    connection
):
    """
    Build the full evidence relationship layer.
    """

    # Always rebuild relationships from normalized data.
    connection.execute(
        """
        DELETE FROM evidence_rules
        """
    )

    case_count = (
        create_case_specific_relationships(
            connection
        )
    )

    case_treatment_count = (
        create_case_treatment_relationships(
            connection
        )
    )

    simple_count = (
        create_simple_observation_relationships(
            connection
        )
    )

    shared_count = (
        create_shared_environment_relationships(
            connection
        )
    )

    observation_level_count = (
        create_case_factor_observation_response_links(
            connection
        )
    )

    remove_duplicate_relationships(
        connection
    )

    adjust_known_uncertain_relationships(
        connection
    )

    connection.commit()

    final_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM evidence_rules
        """
    ).fetchone()[0]

    return {
        "case_specific_generated": case_count,

        "case_treatment_generated":
            case_treatment_count,

        "simple_generated": simple_count,

        "shared_environment_generated": shared_count,

        "observation_level_generated":
            observation_level_count,

        "final_rule_count": final_count,
    }