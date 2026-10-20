"""
state_hypotheses.py

Alternative-state representation for the microorganism world model.

The purpose of this module is to represent cases where the same
environmental observation supports more than one possible organism
state, such as Euglena viridis Morphotype I and Morphotype II.

This module is read-only with respect to the database.
"""

import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from config import DATABASE_FILE


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class StateResponse:
    """
    One biological response belonging to a possible organism state.
    """

    response_type: str
    response_category: Optional[str]
    value: str
    confidence: Optional[float]

    associated_factors: list[str] = field(
        default_factory=list
    )

    evidence_strength: Optional[float] = None


@dataclass
class StateHypothesis:
    """
    One possible biological state of a microorganism.

    Example:

        Euglena viridis
        Morphotype I
            cell_length = 55-68 um
            cell_width = 11-19 um
            cell_shape = spindle
            ...
    """

    species: str
    observation_id: str

    case_id: str
    case_label: Optional[str]
    case_type: Optional[str]

    morphotype: Optional[str]

    confidence: Optional[float]

    responses: list[StateResponse] = field(
        default_factory=list
    )

    def to_dict(self):

        return {
            "species": self.species,
            "observation_id": self.observation_id,
            "case_id": self.case_id,
            "case_label": self.case_label,
            "case_type": self.case_type,
            "morphotype": self.morphotype,
            "confidence": self.confidence,

            "responses": [
                {
                    "response_type":
                        response.response_type,

                    "response_category":
                        response.response_category,

                    "value":
                        response.value,

                    "confidence":
                        response.confidence,

                    "associated_factors":
                        response.associated_factors,

                    "evidence_strength":
                        response.evidence_strength,
                }

                for response in self.responses
            ]
        }


@dataclass
class AlternativeStateGroup:
    """
    A collection of mutually alternative states belonging to the
    same microorganism observation.

    Example:

        Euglena_viridis_stressed_row6

            Possible state 1:
                Morphotype I

            Possible state 2:
                Morphotype II
    """

    species: str
    observation_id: str

    states: list[StateHypothesis] = field(
        default_factory=list
    )

    def to_dict(self):

        return {
            "species": self.species,
            "observation_id": self.observation_id,

            "states": [
                state.to_dict()
                for state in self.states
            ]
        }


# ============================================================
# DATABASE
# ============================================================

def get_connection():

    if not DATABASE_FILE.exists():

        raise FileNotFoundError(
            f"Database could not be found:\n"
            f"{DATABASE_FILE}"
        )

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# VALUE FORMATTING
# ============================================================

def format_number(value):

    if value is None:
        return None

    value = float(value)

    if value.is_integer():

        return str(
            int(value)
        )

    return str(value)


def format_response_value(row):
    """
    Convert a normalized organism response into a readable value.
    """

    if row["value_text"] is not None:

        value = str(
            row["value_text"]
        )

    elif row["value_numeric"] is not None:

        value = format_number(
            row["value_numeric"]
        )

        if row["unit"]:

            value += (
                f" {row['unit']}"
            )

    elif (
        row["value_min"] is not None
        and
        row["value_max"] is not None
    ):

        value = (
            f"{format_number(row['value_min'])}"
            f"-"
            f"{format_number(row['value_max'])}"
        )

        if row["unit"]:

            value += (
                f" {row['unit']}"
            )

    elif row["value_min"] is not None:

        value = (
            f"> {format_number(row['value_min'])}"
        )

        if row["unit"]:

            value += (
                f" {row['unit']}"
            )

    elif row["value_max"] is not None:

        value = (
            f"< {format_number(row['value_max'])}"
        )

        if row["unit"]:

            value += (
                f" {row['unit']}"
            )

    else:

        value = "unspecified"

    if row["qualifier"]:

        value += (
            f" [{row['qualifier']}]"
        )

    return value


# ============================================================
# LOAD RESPONSES FOR ONE STATE
# ============================================================

def load_state_responses(
    connection,
    case_id
):
    """
    Load biological responses belonging specifically to one case.

    A response can be associated with several water-quality factors,
    so the SQL may return several rows for the same response_id.
    These are combined into one StateResponse.
    """

    rows = connection.execute(
        """
        SELECT

            r.response_id,
            r.response_type,
            r.response_category,

            r.value_text,
            r.value_numeric,
            r.value_min,
            r.value_max,
            r.unit,
            r.qualifier,

            r.confidence,

            w.factor_name,

            er.evidence_strength

        FROM organism_responses r

        LEFT JOIN evidence_rules er
            ON er.response_id = r.response_id

        LEFT JOIN water_quality_factors w
            ON er.factor_id = w.factor_id

        WHERE r.case_id = ?

        ORDER BY
            r.response_id,
            w.factor_name
        """,
        (
            case_id,
        )
    ).fetchall()

    grouped = {}

    for row in rows:

        response_id = row[
            "response_id"
        ]

        if response_id not in grouped:

            grouped[
                response_id
            ] = {
                "row": row,
                "factors": set(),
                "evidence": []
            }

        if row["factor_name"]:

            grouped[
                response_id
            ][
                "factors"
            ].add(
                row["factor_name"]
            )

        if (
            row["evidence_strength"]
            is not None
        ):

            grouped[
                response_id
            ][
                "evidence"
            ].append(
                float(
                    row[
                        "evidence_strength"
                    ]
                )
            )

    responses = []

    for response_id in sorted(
        grouped
    ):

        item = grouped[
            response_id
        ]

        row = item[
            "row"
        ]

        evidence_values = item[
            "evidence"
        ]

        if evidence_values:

            evidence_strength = max(
                evidence_values
            )

        else:

            evidence_strength = None

        response = StateResponse(
            response_type=row[
                "response_type"
            ],

            response_category=row[
                "response_category"
            ],

            value=format_response_value(
                row
            ),

            confidence=row[
                "confidence"
            ],

            associated_factors=sorted(
                item[
                    "factors"
                ]
            ),

            evidence_strength=(
                evidence_strength
            )
        )

        responses.append(
            response
        )

    return responses


# ============================================================
# LOAD ONE STATE
# ============================================================

def load_state_hypothesis(
    connection,
    case_id
):
    """
    Load one case as a StateHypothesis.
    """

    case = connection.execute(
        """
        SELECT

            c.case_id,
            c.observation_id,
            c.case_label,
            c.case_type,
            c.morphotype,
            c.confidence,

            org.scientific_name
                AS species

        FROM cases c

        JOIN observations obs
            ON c.observation_id =
               obs.observation_id

        JOIN organisms org
            ON obs.organism_id =
               org.organism_id

        WHERE c.case_id = ?
        """,
        (
            case_id,
        )
    ).fetchone()

    if case is None:

        return None

    responses = load_state_responses(
        connection,
        case_id
    )

    return StateHypothesis(

        species=case[
            "species"
        ],

        observation_id=case[
            "observation_id"
        ],

        case_id=case[
            "case_id"
        ],

        case_label=case[
            "case_label"
        ],

        case_type=case[
            "case_type"
        ],

        morphotype=case[
            "morphotype"
        ],

        confidence=case[
            "confidence"
        ],

        responses=responses
    )


# ============================================================
# FIND ALTERNATIVE STATE GROUPS
# ============================================================

def get_alternative_state_groups(
    species=None
):
    """
    Return observations that contain two or more explicit
    morphotype cases.

    These are interpreted as alternative organism states.

    Parameters
    ----------
    species:
        Optional scientific name. If supplied, only alternative
        states for that microorganism are returned.
    """

    connection = get_connection()

    try:

        parameters = []

        species_filter = ""

        if species is not None:

            species_filter = """
                AND LOWER(org.scientific_name)
                    = LOWER(?)
            """

            parameters.append(
                species
            )

        rows = connection.execute(
            f"""
            SELECT

                c.case_id,
                c.observation_id,

                org.scientific_name
                    AS species

            FROM cases c

            JOIN observations obs
                ON c.observation_id =
                   obs.observation_id

            JOIN organisms org
                ON obs.organism_id =
                   org.organism_id

            WHERE
                (
                    c.case_type = 'morphotype'
                    OR
                    c.morphotype IS NOT NULL
                )

                {species_filter}

            ORDER BY
                org.scientific_name,
                c.observation_id,
                c.case_id
            """,
            parameters
        ).fetchall()

        # ----------------------------------------------------
        # Group cases by observation
        # ----------------------------------------------------

        grouped = {}

        for row in rows:

            key = (
                row["species"],
                row["observation_id"]
            )

            if key not in grouped:

                grouped[
                    key
                ] = []

            grouped[
                key
            ].append(
                row["case_id"]
            )

        alternative_groups = []

        # ----------------------------------------------------
        # Only observations with >= 2 morphotype cases are
        # genuinely alternative-state groups.
        # ----------------------------------------------------

        for (
            species_name,
            observation_id
        ), case_ids in grouped.items():

            if len(case_ids) < 2:

                continue

            group = AlternativeStateGroup(
                species=species_name,
                observation_id=observation_id
            )

            for case_id in case_ids:

                state = load_state_hypothesis(
                    connection,
                    case_id
                )

                if state is not None:

                    group.states.append(
                        state
                    )

            alternative_groups.append(
                group
            )

        return alternative_groups

    finally:

        connection.close()


# ============================================================
# FIND STATES ASSOCIATED WITH WATER QUALITY
# ============================================================

def get_state_hypotheses_for_water(
    species,
    water_factors
):
    """
    Find explicit morphotype states associated with supplied
    water-quality factors.

    Example:

        get_state_hypotheses_for_water(
            "Euglena viridis",
            ["eutrophication"]
        )

    This does NOT claim that the environmental factor causes the
    morphotype. It only returns explicit states whose evidence is
    associated with that factor.
    """

    if isinstance(
        water_factors,
        str
    ):

        water_factors = [
            water_factors
        ]

    requested_factors = {
        str(factor)
        .strip()
        .lower()

        for factor in water_factors
    }

    groups = (
        get_alternative_state_groups(
            species=species
        )
    )

    matching_groups = []

    for group in groups:

        matching_states = []

        for state in group.states:

            state_factors = set()

            for response in state.responses:

                state_factors.update(
                    factor.lower()
                    for factor
                    in response.associated_factors
                )

            if (
                requested_factors
                &
                state_factors
            ):

                matching_states.append(
                    state
                )

        if matching_states:

            matching_group = (
                AlternativeStateGroup(
                    species=group.species,
                    observation_id=(
                        group.observation_id
                    ),
                    states=matching_states
                )
            )

            matching_groups.append(
                matching_group
            )

    return matching_groups


# ============================================================
# DISPLAY
# ============================================================

def print_state_hypothesis(
    state,
    number=None
):

    if number is not None:

        print()
        print(
            f"POSSIBLE STATE {number}"
        )

    else:

        print()
        print(
            "POSSIBLE STATE"
        )

    print("-" * 70)

    print(
        f"Species:       "
        f"{state.species}"
    )

    print(
        f"State:         "
        f"{state.case_label}"
    )

    print(
        f"Morphotype:    "
        f"{state.morphotype}"
    )

    print(
        f"Observation:   "
        f"{state.observation_id}"
    )

    print(
        f"Confidence:    "
        f"{state.confidence}"
    )

    print()

    print(
        "Biological features:"
    )

    for response in state.responses:

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
            f"    Evidence: "
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


def print_alternative_group(
    group
):

    print()
    print("=" * 70)
    print("ALTERNATIVE ORGANISM STATES")
    print("=" * 70)

    print()

    print(
        f"Species:     "
        f"{group.species}"
    )

    print(
        f"Observation: "
        f"{group.observation_id}"
    )

    print()

    print(
        f"Possible states: "
        f"{len(group.states)}"
    )

    for index, state in enumerate(
        group.states,
        start=1
    ):

        print_state_hypothesis(
            state,
            number=index
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("STATE HYPOTHESIS TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # TEST 1
    # Load every explicit alternative-state group.
    # --------------------------------------------------------

    groups = (
        get_alternative_state_groups()
    )

    print()
    print(
        f"Alternative-state groups found: "
        f"{len(groups)}"
    )

    for group in groups:

        print_alternative_group(
            group
        )

    # --------------------------------------------------------
    # TEST 2
    # Ask for states associated with eutrophication.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "WATER-ASSOCIATED STATE TEST"
    )
    print("=" * 70)

    eutrophic_groups = (
        get_state_hypotheses_for_water(
            species="Euglena viridis",
            water_factors=[
                "eutrophication"
            ]
        )
    )

    print()
    print(
        "Query:"
    )

    print(
        "Species = Euglena viridis"
    )

    print(
        "Water factor = eutrophication"
    )

    print()

    print(
        f"Matching alternative-state groups: "
        f"{len(eutrophic_groups)}"
    )

    for group in eutrophic_groups:

        print_alternative_group(
            group
        )