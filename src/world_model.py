import sqlite3
from collections import defaultdict

from config import DATABASE_FILE


# ============================================================
# DATABASE HELPERS
# ============================================================

def get_connection():
    """
    Open the SQLite database.
    """

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_species_name(species):
    """
    Normalize species names for matching.
    """

    return str(
        species
    ).strip().lower()


def normalize_feature_name(name):
    """
    Normalize factor / response names.
    """

    return str(
        name
    ).strip().lower()


# ============================================================
# TREATMENT-CONTEXT HELPERS
# ============================================================

# Some experimental treatment names are more specific than the
# normalized water-factor names stored in the database.
#
# These aliases are used ONLY for matching a case treatment to
# its corresponding water-quality factor. They do not overwrite
# or rename the source treatment stored in the database.
TREATMENT_FACTOR_ALIASES = {
    "propranolol_hydrochloride": "propranolol"
}


def normalize_treatment_name(name):
    """
    Normalize a treatment/factor name for treatment matching.

    Examples
    --------
    "Sodium Metasilicate" -> "sodium_metasilicate"
    "propranolol-hydrochloride" ->
        "propranolol_hydrochloride"
    """

    if name is None:
        return None

    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def canonical_treatment_factor_name(treatment_name):
    """
    Return the water-factor name that owns a treatment context.

    Example
    -------
    propranolol_hydrochloride -> propranolol
    petrol -> petrol
    cadmium -> cadmium
    """

    normalized = normalize_treatment_name(
        treatment_name
    )

    if normalized is None:
        return None

    return TREATMENT_FACTOR_ALIASES.get(
        normalized,
        normalized
    )


def factor_is_case_treatment(
    factor_name,
    treatment_name
):
    """
    Return True only when a water factor is the treatment that
    owns the experimental case's concentration.

    This prevents case concentrations such as 50-150 uL/L
    petrol from being incorrectly attached to co-observed
    factors such as hypoxia or oil.
    """

    if treatment_name is None:
        return False

    normalized_factor = normalize_treatment_name(
        factor_name
    )

    canonical_treatment = canonical_treatment_factor_name(
        treatment_name
    )

    return (
        normalized_factor
        ==
        canonical_treatment
    )


def get_factor_case_concentration(
    factor_name,
    treatment_name,
    case_min,
    case_max,
    case_unit
):
    """
    Return case concentration fields only when the supplied
    water factor is the treatment that owns that concentration.

    For co-observed contextual factors, return None values.
    """

    if not factor_is_case_treatment(
        factor_name,
        treatment_name
    ):
        return None, None, None

    return (
        case_min,
        case_max,
        case_unit
    )


# ============================================================
# SPECIES LOOKUP
# ============================================================

def get_organism_id(
    connection,
    species
):
    """
    Find organism_id from a scientific name.
    """

    species_normalized = (
        normalize_species_name(
            species
        )
    )

    rows = connection.execute(
        """
        SELECT
            organism_id,
            scientific_name

        FROM organisms
        """
    ).fetchall()

    for row in rows:

        stored_name = (
            normalize_species_name(
                row["scientific_name"]
            )
        )

        if stored_name == species_normalized:

            return row["organism_id"]

    raise ValueError(
        f"Species not found in database: "
        f"{species}"
    )


# ============================================================
# CONCENTRATION MATCHING
# ============================================================

def concentration_matches(
    requested_value,
    requested_unit,
    case_min,
    case_max,
    case_unit
):
    """
    Check whether a requested concentration is compatible
    with a case concentration.

    Returns
    -------
    True
        The case has a known concentration and it matches.

    False
        The case has a known concentration but it does not match.

    None
        The database evidence does not specify a concentration,
        so it cannot be considered an exact concentration match.
    """

    # --------------------------------------------------------
    # User did not request a numeric concentration.
    # --------------------------------------------------------

    if requested_value is None:
        return True

    # --------------------------------------------------------
    # User supplied a numeric concentration, but this case
    # does not contain any concentration information.
    # --------------------------------------------------------

    if case_min is None:
        return None

    # --------------------------------------------------------
    # Check units when both are available.
    # --------------------------------------------------------

    if (
        requested_unit is not None
        and
        case_unit is not None
    ):

        requested_unit_normalized = (
            str(requested_unit)
            .strip()
            .lower()
        )

        case_unit_normalized = (
            str(case_unit)
            .strip()
            .lower()
        )

        if (
            requested_unit_normalized
            !=
            case_unit_normalized
        ):
            return False

    requested_value = float(
        requested_value
    )

    case_min = float(
        case_min
    )

    # --------------------------------------------------------
    # Threshold case
    #
    # Example:
    #     >15 uL/L
    # --------------------------------------------------------

    if case_max is None:

        return (
            requested_value
            >
            case_min
        )

    case_max = float(
        case_max
    )

    # --------------------------------------------------------
    # Range or fixed-value case
    # --------------------------------------------------------

    return (
        case_min
        <= requested_value
        <= case_max
    )


# ============================================================
# RESPONSE VALUE FORMATTING
# ============================================================

def format_response_value(row):
    """
    Convert a stored organism response into a readable string.
    """

    # --------------------------------------------------------
    # Numeric value
    # --------------------------------------------------------

    if (
        row["response_value_numeric"]
        is not None
    ):

        value = (
            row["response_value_numeric"]
        )

        if row["response_unit"]:

            return (
                f"{value} "
                f"{row['response_unit']}"
            )

        return str(
            value
        )

    # --------------------------------------------------------
    # Numeric range
    # --------------------------------------------------------

    if (
        row["response_value_min"]
        is not None
        and
        row["response_value_max"]
        is not None
    ):

        value = (
            f"{row['response_value_min']}"
            f"-"
            f"{row['response_value_max']}"
        )

        if row["response_unit"]:

            value += (
                f" {row['response_unit']}"
            )

        return value

    # --------------------------------------------------------
    # Text value
    # --------------------------------------------------------

    if (
        row["response_value_text"]
        is not None
    ):

        return row[
            "response_value_text"
        ]

    return "observed"


# ============================================================
# WATER VALUE FORMATTING
# ============================================================

def format_water_value(row):
    """
    Convert a stored water factor into a readable string.
    """

    if (
        row["factor_value_numeric"]
        is not None
    ):

        value = (
            row["factor_value_numeric"]
        )

        if row["factor_unit"]:

            return (
                f"{value} "
                f"{row['factor_unit']}"
            )

        return str(
            value
        )

    if (
        row["factor_value_text"]
        is not None
    ):

        return row[
            "factor_value_text"
        ]

    return "present"


# ============================================================
# CASE CONCENTRATION FORMATTING
# ============================================================

def format_case_concentration(
    case_min,
    case_max,
    case_unit
):
    """
    Convert concentration case information into a readable
    string.

    Examples:
        5-15 uL/L
        >15 uL/L
        100 ug/L
    """

    if case_min is None:
        return None

    # Threshold
    if case_max is None:

        return (
            f">{case_min} "
            f"{case_unit or ''}"
        ).strip()

    # Exact concentration
    if float(case_min) == float(case_max):

        return (
            f"{case_min} "
            f"{case_unit or ''}"
        ).strip()

    # Range
    return (
        f"{case_min}-"
        f"{case_max} "
        f"{case_unit or ''}"
    ).strip()


# ============================================================
# FORWARD WORLD MODEL
#
# WATER QUALITY -> MICROORGANISM RESPONSE
# ============================================================

def predict_response(
    species,
    water_state,
    top_k=10
):
    """
    Rank likely microorganism responses for a specified
    water-quality state.

    Parameters
    ----------
    species : str

        Example:
            "Nitzschia palea"

    water_state : dict

        Presence/text example:

            {
                "insecticide": "present"
            }

        Numeric example:

            {
                "petrol": {
                    "value": 10,
                    "unit": "uL/L"
                }
            }

        Multiple-factor example:

            {
                "sewage": "present",
                "ammonium": "high",
                "phosphate": "high"
            }

    top_k : int

        Maximum number of predictions returned.

    Important
    ---------
    evidence_score is NOT a biological probability.

    It is a ranking score based on the strength and number
    of independent supporting database observations.
    """

    connection = get_connection()

    try:

        organism_id = get_organism_id(
            connection,
            species
        )

        all_matches = []

        # ====================================================
        # Process each requested environmental factor
        # ====================================================

        for (
            raw_factor_name,
            requested
        ) in water_state.items():

            factor_name = (
                normalize_feature_name(
                    raw_factor_name
                )
            )

            requested_value = None
            requested_unit = None
            requested_text = None

            # ------------------------------------------------
            # Dictionary form
            #
            # {
            #     "value": 10,
            #     "unit": "uL/L"
            # }
            # ------------------------------------------------

            if isinstance(
                requested,
                dict
            ):

                requested_value = (
                    requested.get(
                        "value"
                    )
                )

                requested_unit = (
                    requested.get(
                        "unit"
                    )
                )

                requested_text = (
                    requested.get(
                        "text"
                    )
                )

                if (
                    requested_text
                    is not None
                ):

                    requested_text = (
                        str(
                            requested_text
                        )
                        .strip()
                        .lower()
                    )

            # ------------------------------------------------
            # Simple text form
            #
            # "present"
            # "high"
            # ------------------------------------------------

            else:

                requested_text = (
                    str(
                        requested
                    )
                    .strip()
                    .lower()
                )

            # =================================================
            # Retrieve all relationships for this factor
            # =================================================

            rows = connection.execute(
                """
                SELECT
                    er.rule_id,
                    er.relationship_type,
                    er.evidence_strength,
                    er.observation_id,
                    er.case_id,

                    w.factor_name,

                    w.value_text
                        AS factor_value_text,

                    w.value_numeric
                        AS factor_value_numeric,

                    w.unit
                        AS factor_unit,

                    r.response_id,
                    r.response_type,
                    r.response_category,

                    r.value_text
                        AS response_value_text,

                    r.value_numeric
                        AS response_value_numeric,

                    r.value_min
                        AS response_value_min,

                    r.value_max
                        AS response_value_max,

                    r.unit
                        AS response_unit,

                    c.treatment_name,
                    c.concentration_min,
                    c.concentration_max,
                    c.concentration_unit,

                    o.source_url,
                    o.condition
                        AS observation_condition

                FROM evidence_rules er

                JOIN water_quality_factors w
                    ON er.factor_id =
                       w.factor_id

                JOIN organism_responses r
                    ON er.response_id =
                       r.response_id

                JOIN observations o
                    ON er.observation_id =
                       o.observation_id

                LEFT JOIN cases c
                    ON er.case_id =
                       c.case_id

                WHERE
                    er.organism_id = ?
                    AND
                    LOWER(w.factor_name) = ?
                """,
                (
                    organism_id,
                    factor_name
                )
            ).fetchall()

            # =================================================
            # Filter matching relationships
            # =================================================

            for row in rows:

                # --------------------------------------------
                # Match text-valued environmental conditions
                # --------------------------------------------

                if (
                    requested_text
                    is not None
                ):

                    stored_text = (
                        row[
                            "factor_value_text"
                        ]
                    )

                    if (
                        stored_text
                        is not None
                    ):

                        stored_text = (
                            str(
                                stored_text
                            )
                            .strip()
                            .lower()
                        )

                        # Generic presence queries are
                        # intentionally permissive.
                        generic_presence_values = {
                            "present",
                            "true",
                            "yes"
                        }

                        if (
                            requested_text
                            not in
                            generic_presence_values
                            and
                            stored_text
                            !=
                            requested_text
                        ):

                            continue

                # --------------------------------------------
                # Match concentration
                # --------------------------------------------

                concentration_match = True

                concentration_specificity = (
                    "not_requested"
                )

                # --------------------------------------------
                # A case concentration belongs only to the
                # factor that matches case.treatment_name.
                #
                # Example:
                #
                #   petrol = 50-150 uL/L
                #
                # may coexist with hypoxia and oil in the same
                # observation, but those contextual factors do
                # not inherit petrol's numerical concentration.
                # --------------------------------------------

                (
                    factor_case_min,
                    factor_case_max,
                    factor_case_unit
                ) = get_factor_case_concentration(
                    factor_name=row[
                        "factor_name"
                    ],
                    treatment_name=row[
                        "treatment_name"
                    ],
                    case_min=row[
                        "concentration_min"
                    ],
                    case_max=row[
                        "concentration_max"
                    ],
                    case_unit=row[
                        "concentration_unit"
                    ]
                )

                factor_is_treatment = (
                    factor_is_case_treatment(
                        row[
                            "factor_name"
                        ],
                        row[
                            "treatment_name"
                        ]
                    )
                )

                if (
                    requested_value
                    is not None
                ):

                    concentration_match = (
                        concentration_matches(
                            requested_value,
                            requested_unit,
                            factor_case_min,
                            factor_case_max,
                            factor_case_unit
                        )
                    )

                    # Known concentration contradicts query.
                    if (
                        concentration_match
                        is False
                    ):

                        continue

                    # Exact range / threshold match.
                    if (
                        concentration_match
                        is True
                    ):

                        concentration_specificity = (
                            "matched"
                        )

                    # Pollutant matches, but source has
                    # unspecified concentration.
                    elif (
                        concentration_match
                        is None
                    ):

                        concentration_specificity = (
                            "unspecified"
                        )

                # --------------------------------------------
                # Start from relationship evidence strength.
                # --------------------------------------------

                adjusted_strength = float(
                    row[
                        "evidence_strength"
                    ]
                )

                # --------------------------------------------
                # Numeric concentration requested, but the
                # supporting evidence contains no concentration.
                #
                # Keep as generic evidence, but penalize it.
                # --------------------------------------------

                if (
                    requested_value
                    is not None
                    and
                    concentration_specificity
                    ==
                    "unspecified"
                ):

                    adjusted_strength *= 0.5

                all_matches.append(
                    {
                        "input_factor":
                            factor_name,

                        "requested_value":
                            requested_value,

                        "requested_unit":
                            requested_unit,

                        "observation_id":
                            row[
                                "observation_id"
                            ],

                        "case_id":
                            row[
                                "case_id"
                            ],

                        # Concentration attached to the
                        # queried factor itself. This is None for
                        # co-observed contextual factors.
                        "case_concentration":
                            format_case_concentration(
                                factor_case_min,
                                factor_case_max,
                                factor_case_unit
                            ),

                        # Preserve the original experimental
                        # treatment context separately so the UI
                        # can still explain the case correctly.
                        "case_treatment_name":
                            row[
                                "treatment_name"
                            ],

                        "treatment_concentration":
                            format_case_concentration(
                                row[
                                    "concentration_min"
                                ],
                                row[
                                    "concentration_max"
                                ],
                                row[
                                    "concentration_unit"
                                ]
                            ),

                        "factor_is_treatment":
                            factor_is_treatment,

                        "response_type":
                            row[
                                "response_type"
                            ],

                        "response_category":
                            row[
                                "response_category"
                            ],

                        "response_value":
                            format_response_value(
                                row
                            ),

                        "evidence_strength":
                            adjusted_strength,

                        "original_evidence_strength":
                            row[
                                "evidence_strength"
                            ],

                        "relationship_type":
                            row[
                                "relationship_type"
                            ],

                        "concentration_specificity":
                            concentration_specificity,

                        "source_url":
                            row[
                                "source_url"
                            ],

                        # Preserve the biological-condition label stored
                        # on the source observation. Imported research IDs
                        # do not necessarily contain the legacy
                        # ``_healthy_`` / ``_stressed_`` naming tokens, so
                        # the simulator must use this explicit field rather
                        # than attempting to infer condition from the ID.
                        "observation_condition":
                            row[
                                "observation_condition"
                            ]
                    }
                )

        # ====================================================
        # AGGREGATION
        #
        # Aggregate by microorganism response.
        #
        # IMPORTANT:
        #
        # Several water factors can come from one paper /
        # observation.
        #
        # Example:
        #
        # sewage
        # ammonium
        # phosphate
        # nitrite
        #
        # may all describe one environment.
        #
        # We therefore count each observation only once when
        # determining independent support.
        # ====================================================

        grouped = defaultdict(
            lambda: {
                "observation_scores": {},
                "evidence": []
            }
        )

        for match in all_matches:

            response_key = (
                match[
                    "response_type"
                ],
                match[
                    "response_value"
                ]
            )

            observation_id = (
                match[
                    "observation_id"
                ]
            )

            current_score = (
                grouped[
                    response_key
                ][
                    "observation_scores"
                ].get(
                    observation_id,
                    0
                )
            )

            grouped[
                response_key
            ][
                "observation_scores"
            ][
                observation_id
            ] = max(
                current_score,
                match[
                    "evidence_strength"
                ]
            )

            grouped[
                response_key
            ][
                "evidence"
            ].append(
                match
            )

        # ====================================================
        # Build ranked predictions
        # ====================================================

        predictions = []

        for (
            response_type,
            response_value
        ), group in grouped.items():

            observation_scores = list(
                group[
                    "observation_scores"
                ].values()
            )

            strongest = max(
                observation_scores
            )

            independent_support = len(
                observation_scores
            )

            # ------------------------------------------------
            # Simple ranking rule:
            #
            # strongest evidence
            #
            # +
            #
            # small bonus for independent observations
            # ------------------------------------------------

            score = (
                strongest
                +
                (
                    0.1
                    *
                    max(
                        0,
                        independent_support - 1
                    )
                )
            )

            score = min(
                score,
                1.0
            )

            predictions.append(
                {
                    "response_type":
                        response_type,

                    "response_value":
                        response_value,

                    "evidence_score":
                        round(
                            score,
                            3
                        ),

                    "independent_observations":
                        independent_support,

                    "supporting_evidence":
                        group[
                            "evidence"
                        ]
                }
            )

        predictions.sort(
            key=lambda x: (
                x[
                    "evidence_score"
                ],
                x[
                    "independent_observations"
                ]
            ),
            reverse=True
        )

        return {
            "species":
                species,

            "water_state":
                water_state,

            "predictions":
                predictions[
                    :top_k
                ]
        }

    finally:

        connection.close()


# ============================================================
# INVERSE WORLD MODEL
#
# MICROORGANISM RESPONSE -> WATER QUALITY
# ============================================================

def normalize_unit(unit):
    """
    Normalize units for comparison.
    """
    if unit is None:
        return None

    return (
        str(unit)
        .strip()
        .lower()
        .replace("µ", "u")
        .replace("μ", "u")
    )


def parse_response_text_value(text):
    """
    Try to interpret a text response as either:

        13.0-16.0 um
        13.0–16.0 um
        20.8 percent
        2.0 body_lengths

    Returns a dictionary describing the parsed value,
    or None when the value is genuinely categorical text.
    """
    import re

    if text is None:
        return None

    cleaned = (
        str(text)
        .strip()
        .lower()
        .replace("µ", "u")
        .replace("μ", "u")
        .replace("–", "-")
        .replace("—", "-")
    )

    # --------------------------------------------------------
    # Numeric range
    # Example: 13.0-16.0 um
    # --------------------------------------------------------

    range_match = re.fullmatch(
        r"\s*"
        r"(-?\d+(?:\.\d+)?)"
        r"\s*-\s*"
        r"(-?\d+(?:\.\d+)?)"
        r"(?:\s+(.+?))?"
        r"\s*",
        cleaned
    )

    if range_match:

        return {
            "kind": "range",
            "min": float(range_match.group(1)),
            "max": float(range_match.group(2)),
            "unit": normalize_unit(
                range_match.group(3)
            )
        }

    # --------------------------------------------------------
    # Single numeric value
    # Example: 20.8 percent
    # --------------------------------------------------------

    numeric_match = re.fullmatch(
        r"\s*"
        r"(-?\d+(?:\.\d+)?)"
        r"(?:\s+(.+?))?"
        r"\s*",
        cleaned
    )

    if numeric_match:

        return {
            "kind": "numeric",
            "value": float(numeric_match.group(1)),
            "unit": normalize_unit(
                numeric_match.group(2)
            )
        }

    return None


def response_value_matches(
    row,
    requested_text=None,
    requested_value=None,
    requested_unit=None
):
    """
    Return True only when the requested biological response value
    matches the value stored in this database response row.

    Handles:
        - categorical text
        - exact numeric values
        - numeric ranges
        - numeric input falling inside a stored range
    """

    stored_text = row[
        "response_value_text"
    ]

    stored_numeric = row[
        "response_value_numeric"
    ]

    stored_min = row[
        "response_value_min"
    ]

    stored_max = row[
        "response_value_max"
    ]

    stored_unit = normalize_unit(
        row["response_unit"]
    )

    # ========================================================
    # NUMERIC INPUT FROM APP
    # ========================================================

    if requested_value is not None:

        requested_value = float(
            requested_value
        )

        requested_unit_normalized = normalize_unit(
            requested_unit
        )

        if (
            requested_unit_normalized is not None
            and
            stored_unit is not None
            and
            requested_unit_normalized != stored_unit
        ):
            return False

        # Exact stored numeric value
        if stored_numeric is not None:

            return (
                abs(
                    float(stored_numeric)
                    -
                    requested_value
                )
                <=
                1e-9
            )

        # Stored numeric range
        if (
            stored_min is not None
            and
            stored_max is not None
        ):

            return (
                float(stored_min)
                <=
                requested_value
                <=
                float(stored_max)
            )

        return False

    # ========================================================
    # TEXT INPUT FROM APP
    # ========================================================

    if requested_text is not None:

        requested_text_clean = (
            str(requested_text)
            .strip()
            .lower()
        )

        parsed = parse_response_text_value(
            requested_text_clean
        )

        # ----------------------------------------------------
        # Text actually represents a numeric range/value
        # ----------------------------------------------------

        if parsed is not None:

            parsed_unit = parsed[
                "unit"
            ]

            if (
                parsed_unit is not None
                and
                stored_unit is not None
                and
                parsed_unit != stored_unit
            ):
                return False

            if (
                parsed["kind"]
                ==
                "range"
            ):

                # Exact stored range required.
                if (
                    stored_min is None
                    or
                    stored_max is None
                ):
                    return False

                return (
                    abs(
                        float(stored_min)
                        -
                        parsed["min"]
                    )
                    <=
                    1e-9
                    and
                    abs(
                        float(stored_max)
                        -
                        parsed["max"]
                    )
                    <=
                    1e-9
                )

            if (
                parsed["kind"]
                ==
                "numeric"
            ):

                value = parsed[
                    "value"
                ]

                if stored_numeric is not None:

                    return (
                        abs(
                            float(stored_numeric)
                            -
                            value
                        )
                        <=
                        1e-9
                    )

                if (
                    stored_min is not None
                    and
                    stored_max is not None
                ):

                    return (
                        float(stored_min)
                        <=
                        value
                        <=
                        float(stored_max)
                    )

                return False

        # ----------------------------------------------------
        # Genuine categorical text
        # ----------------------------------------------------

        generic_presence_values = {
            "present",
            "true",
            "yes"
        }

        if stored_text is None:
            return False

        stored_text_clean = (
            str(stored_text)
            .strip()
            .lower()
        )

        if (
            requested_text_clean
            in
            generic_presence_values
        ):
            return (
                stored_text_clean
                in
                generic_presence_values
            )

        return (
            stored_text_clean
            ==
            requested_text_clean
        )

    return True

def infer_water_quality(
    species,
    observed_response,
    top_k=10
):
    """
    Rank possible water-quality conditions from an observed
    microorganism response.

    Parameters
    ----------
    species : str

    observed_response : dict

        Text example:

            {
                "sheath_absent": "present"
            }

        Numeric example:

            {
                "cell_death": {
                    "value": 97.6,
                    "unit": "percent"
                }
            }

    Important
    ---------
    The returned evidence_score is a database evidence ranking,
    NOT a probability that the water actually has that
    condition.
    """

    connection = get_connection()

    try:

        organism_id = get_organism_id(
            connection,
            species
        )

        all_matches = []

        # ====================================================
        # Process every observed biological response
        # ====================================================

        for (
            raw_response_type,
            requested
        ) in observed_response.items():

            response_type = (
                normalize_feature_name(
                    raw_response_type
                )
            )

            requested_value = None
            requested_unit = None
            requested_text = None

            if isinstance(
                requested,
                dict
            ):

                requested_value = (
                    requested.get(
                        "value"
                    )
                )

                requested_unit = (
                    requested.get(
                        "unit"
                    )
                )

                requested_text = (
                    requested.get(
                        "text"
                    )
                )

                if (
                    requested_text
                    is not None
                ):

                    requested_text = (
                        str(
                            requested_text
                        )
                        .strip()
                        .lower()
                    )

            else:

                requested_text = (
                    str(
                        requested
                    )
                    .strip()
                    .lower()
                )

            # =================================================
            # Retrieve all relevant evidence
            # =================================================

            rows = connection.execute(
                """
                SELECT
                    er.relationship_type,
                    er.evidence_strength,
                    er.observation_id,
                    er.case_id,

                    w.factor_name,

                    w.value_text
                        AS factor_value_text,

                    w.value_numeric
                        AS factor_value_numeric,

                    w.unit
                        AS factor_unit,

                    r.response_type,

                    r.value_text
                        AS response_value_text,

                    r.value_numeric
                        AS response_value_numeric,

                    r.value_min
                        AS response_value_min,

                    r.value_max
                        AS response_value_max,

                    r.unit
                        AS response_unit,

                    c.treatment_name,
                    c.concentration_min,
                    c.concentration_max,
                    c.concentration_unit,

                    o.source_url

                FROM evidence_rules er

                JOIN water_quality_factors w
                    ON er.factor_id =
                       w.factor_id

                JOIN organism_responses r
                    ON er.response_id =
                       r.response_id

                JOIN observations o
                    ON er.observation_id =
                       o.observation_id

                LEFT JOIN cases c
                    ON er.case_id =
                       c.case_id

                WHERE
                    er.organism_id = ?
                    AND
                    LOWER(r.response_type) = ?
                """,
                (
                    organism_id,
                    response_type
                )
            ).fetchall()

            # =================================================
            # Filter response value
            # =================================================

            for row in rows:

                if not response_value_matches(
                    row=row,
                    requested_text=requested_text,
                    requested_value=requested_value,
                    requested_unit=requested_unit
                ):
                    continue

                # --------------------------------------------
                # Only expose case concentration as a
                # concentration of this inferred factor when
                # the factor is the actual experimental
                # treatment.
                # --------------------------------------------

                (
                    factor_case_min,
                    factor_case_max,
                    factor_case_unit
                ) = get_factor_case_concentration(
                    factor_name=row[
                        "factor_name"
                    ],
                    treatment_name=row[
                        "treatment_name"
                    ],
                    case_min=row[
                        "concentration_min"
                    ],
                    case_max=row[
                        "concentration_max"
                    ],
                    case_unit=row[
                        "concentration_unit"
                    ]
                )

                concentration = (
                    format_case_concentration(
                        factor_case_min,
                        factor_case_max,
                        factor_case_unit
                    )
                )

                factor_is_treatment = (
                    factor_is_case_treatment(
                        row[
                            "factor_name"
                        ],
                        row[
                            "treatment_name"
                        ]
                    )
                )

                all_matches.append(
                    {
                        "factor_name":
                            row[
                                "factor_name"
                            ],

                        "factor_value":
                            format_water_value(
                                row
                            ),

                        "case_concentration":
                            concentration,

                        "case_treatment_name":
                            row[
                                "treatment_name"
                            ],

                        "treatment_concentration":
                            format_case_concentration(
                                row[
                                    "concentration_min"
                                ],
                                row[
                                    "concentration_max"
                                ],
                                row[
                                    "concentration_unit"
                                ]
                            ),

                        "factor_is_treatment":
                            factor_is_treatment,

                        "observation_id":
                            row[
                                "observation_id"
                            ],

                        "case_id":
                            row[
                                "case_id"
                            ],

                        "relationship_type":
                            row[
                                "relationship_type"
                            ],

                        "evidence_strength":
                            float(
                                row[
                                    "evidence_strength"
                                ]
                            ),

                        "source_url":
                            row[
                                "source_url"
                            ]
                    }
                )

        # ====================================================
        # Aggregate inferred water conditions
        # ====================================================

        grouped = defaultdict(
            lambda: {
                "observation_scores": {},
                "evidence": []
            }
        )

        for match in all_matches:

            key = (
                match[
                    "factor_name"
                ],

                match[
                    "factor_value"
                ],

                match[
                    "case_concentration"
                ]
            )

            observation_id = (
                match[
                    "observation_id"
                ]
            )

            current_score = (
                grouped[
                    key
                ][
                    "observation_scores"
                ].get(
                    observation_id,
                    0
                )
            )

            grouped[
                key
            ][
                "observation_scores"
            ][
                observation_id
            ] = max(
                current_score,
                match[
                    "evidence_strength"
                ]
            )

            grouped[
                key
            ][
                "evidence"
            ].append(
                match
            )

        # ====================================================
        # Build ranked water-condition predictions
        # ====================================================

        predictions = []

        for (
            factor_name,
            factor_value,
            case_concentration
        ), group in grouped.items():

            observation_scores = list(
                group[
                    "observation_scores"
                ].values()
            )

            strongest = max(
                observation_scores
            )

            independent_support = len(
                observation_scores
            )

            score = (
                strongest
                +
                (
                    0.1
                    *
                    max(
                        0,
                        independent_support - 1
                    )
                )
            )

            score = min(
                score,
                1.0
            )

            predictions.append(
                {
                    "factor_name":
                        factor_name,

                    "factor_value":
                        factor_value,

                    "concentration":
                        case_concentration,

                    "evidence_score":
                        round(
                            score,
                            3
                        ),

                    "independent_observations":
                        independent_support,

                    "supporting_evidence":
                        group[
                            "evidence"
                        ]
                }
            )

        predictions.sort(
            key=lambda x: (
                x[
                    "evidence_score"
                ],
                x[
                    "independent_observations"
                ]
            ),
            reverse=True
        )

        return {
            "species":
                species,

            "observed_response":
                observed_response,

            "possible_water_conditions":
                predictions[
                    :top_k
                ]
        }

    finally:

        connection.close()


# ============================================================
# PRINT FORWARD PREDICTIONS
# ============================================================

def print_forward_prediction(
    result
):
    """
    Pretty-print forward world-model output.
    """

    print()

    print(
        "=" * 70
    )

    print(
        "FORWARD WORLD MODEL"
    )

    print(
        "=" * 70
    )

    print(
        f"Species: "
        f"{result['species']}"
    )

    print(
        f"Water state: "
        f"{result['water_state']}"
    )

    print()

    predictions = (
        result[
            "predictions"
        ]
    )

    if not predictions:

        print(
            "No matching evidence was found "
            "in the database."
        )

        return

    for (
        index,
        prediction
    ) in enumerate(
        predictions,
        start=1
    ):

        print(
            f"{index}. "
            f"{prediction['response_type']}"
        )

        print(
            f"   Value: "
            f"{prediction['response_value']}"
        )

        print(
            f"   Evidence score: "
            f"{prediction['evidence_score']}"
        )

        print(
            f"   Independent observations: "
            f"{prediction['independent_observations']}"
        )

        relationship_types = sorted(
            {
                evidence[
                    "relationship_type"
                ]

                for evidence
                in prediction[
                    "supporting_evidence"
                ]
            }
        )

        print(
            f"   Evidence types: "
            f"{', '.join(relationship_types)}"
        )

        concentration_specificities = sorted(
            {
                evidence[
                    "concentration_specificity"
                ]

                for evidence
                in prediction[
                    "supporting_evidence"
                ]
            }
        )

        if (
            concentration_specificities
            !=
            ["not_requested"]
        ):

            print(
                f"   Concentration matching: "
                f"{', '.join(concentration_specificities)}"
            )

        # Show experimental treatment context separately from
        # the concentration of the queried factor.
        treatment_contexts = sorted(
            {
                (
                    evidence.get(
                        "case_treatment_name"
                    ),
                    evidence.get(
                        "treatment_concentration"
                    )
                )
                for evidence in prediction[
                    "supporting_evidence"
                ]
                if evidence.get(
                    "case_treatment_name"
                )
            },
            key=lambda item: (
                str(item[0]),
                str(item[1])
            )
        )

        for (
            treatment_name,
            treatment_concentration
        ) in treatment_contexts:

            if treatment_concentration:

                print(
                    f"   Treatment context: "
                    f"{treatment_name} = "
                    f"{treatment_concentration}"
                )

            else:

                print(
                    f"   Treatment context: "
                    f"{treatment_name}"
                )

        print()


# ============================================================
# PRINT INVERSE PREDICTIONS
# ============================================================

def print_inverse_prediction(
    result
):
    """
    Pretty-print inverse world-model output.
    """

    print()

    print(
        "=" * 70
    )

    print(
        "INVERSE WORLD MODEL"
    )

    print(
        "=" * 70
    )

    print(
        f"Species: "
        f"{result['species']}"
    )

    print(
        f"Observed response: "
        f"{result['observed_response']}"
    )

    print()

    conditions = (
        result[
            "possible_water_conditions"
        ]
    )

    if not conditions:

        print(
            "No matching evidence was found "
            "in the database."
        )

        return

    for (
        index,
        condition
    ) in enumerate(
        conditions,
        start=1
    ):

        print(
            f"{index}. "
            f"{condition['factor_name']}"
        )

        print(
            f"   Value: "
            f"{condition['factor_value']}"
        )

        if (
            condition[
                "concentration"
            ]
        ):

            print(
                f"   Concentration: "
                f"{condition['concentration']}"
            )

        print(
            f"   Evidence score: "
            f"{condition['evidence_score']}"
        )

        print(
            f"   Independent observations: "
            f"{condition['independent_observations']}"
        )

        relationship_types = sorted(
            {
                evidence[
                    "relationship_type"
                ]

                for evidence
                in condition[
                    "supporting_evidence"
                ]
            }
        )

        print(
            f"   Evidence types: "
            f"{', '.join(relationship_types)}"
        )

        treatment_contexts = sorted(
            {
                (
                    evidence.get(
                        "case_treatment_name"
                    ),
                    evidence.get(
                        "treatment_concentration"
                    )
                )
                for evidence in condition[
                    "supporting_evidence"
                ]
                if evidence.get(
                    "case_treatment_name"
                )
            },
            key=lambda item: (
                str(item[0]),
                str(item[1])
            )
        )

        for (
            treatment_name,
            treatment_concentration
        ) in treatment_contexts:

            if treatment_concentration:

                print(
                    f"   Experimental treatment context: "
                    f"{treatment_name} = "
                    f"{treatment_concentration}"
                )

            else:

                print(
                    f"   Experimental treatment context: "
                    f"{treatment_name}"
                )

        print()


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # TEST 1
    #
    # Nitzschia palea
    #
    # Petrol = 10 uL/L
    #
    # Expected strong evidence:
    #
    #   cell death = 20.8%
    #   oil production increased
    #
    # Generic petrol evidence such as pore percolation can
    # still appear, but should receive a lower score because
    # that source did not specify the concentration.
    # ========================================================

    result = predict_response(
        species="Nitzschia palea",

        water_state={
            "petrol": {
                "value": 10,
                "unit": "uL/L"
            }
        }
    )

    print_forward_prediction(
        result
    )


    # ========================================================
    # TEST 2
    #
    # Nitzschia palea
    #
    # Petrol = 100 uL/L
    #
    # Expected:
    #
    #   97.6% cell death
    #   reduced viability
    #   lipid-body change
    #
    # because 100 uL/L matches:
    #
    #   50-150 uL/L
    #
    # and:
    #
    #   >15 uL/L
    # ========================================================

    result = predict_response(
        species="Nitzschia palea",

        water_state={
            "petrol": {
                "value": 100,
                "unit": "uL/L"
            }
        }
    )

    print_forward_prediction(
        result
    )


    # ========================================================
    # TEST 3
    #
    # Oscillatoria limosa
    #
    # Observe absence of sheath and infer possible
    # environmental conditions.
    # ========================================================

    result = infer_water_quality(
        species="Oscillatoria limosa",

        observed_response={
            "sheath_absent":
                "present"
        }
    )

    print_inverse_prediction(
        result
    )
