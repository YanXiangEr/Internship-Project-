import sqlite3

from config import DATABASE_FILE


# ============================================================
# CONNECTION
# ============================================================

def get_connection():
    """
    Open SQLite database connection.
    """

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# ORGANISMS
# ============================================================

def get_species():
    """
    Return all microorganism species stored in the database.
    """

    connection = get_connection()

    try:

        rows = connection.execute(
            """
            SELECT scientific_name
            FROM organisms
            ORDER BY scientific_name;
            """
        ).fetchall()

        return [
            row["scientific_name"]
            for row in rows
        ]

    finally:

        connection.close()


# ============================================================
# WATER FACTORS
# ============================================================

def get_water_factors(
    species
):
    """
    Return water-quality factors that have evidence
    relationships for a particular microorganism.
    """

    connection = get_connection()

    try:

        rows = connection.execute(
            """
            SELECT DISTINCT
                w.factor_name

            FROM evidence_rules er

            JOIN organisms org
                ON er.organism_id =
                   org.organism_id

            JOIN water_quality_factors w
                ON er.factor_id =
                   w.factor_id

            WHERE
                LOWER(org.scientific_name)
                =
                LOWER(?)

            ORDER BY
                w.factor_name;
            """,
            (
                species,
            )
        ).fetchall()

        return [
            row["factor_name"]
            for row in rows
        ]

    finally:

        connection.close()


# ============================================================
# RESPONSE TYPES
# ============================================================

def get_response_types(
    species
):
    """
    Return microorganism responses that have evidence
    relationships for a species.
    """

    connection = get_connection()

    try:

        rows = connection.execute(
            """
            SELECT DISTINCT
                r.response_type

            FROM evidence_rules er

            JOIN organisms org
                ON er.organism_id =
                   org.organism_id

            JOIN organism_responses r
                ON er.response_id =
                   r.response_id

            WHERE
                LOWER(org.scientific_name)
                =
                LOWER(?)

            ORDER BY
                r.response_type;
            """,
            (
                species,
            )
        ).fetchall()

        return [
            row["response_type"]
            for row in rows
        ]

    finally:

        connection.close()


# ============================================================
# WATER FACTOR DETAILS
# ============================================================

def get_water_factor_details(
    species,
    factor_name
):
    """
    Return known values and concentration information for
    one water-quality factor.
    """

    connection = get_connection()

    try:

        rows = connection.execute(
            """
            SELECT DISTINCT

                w.value_text,

                w.value_numeric,

                w.unit AS factor_unit,

                c.concentration_min,

                c.concentration_max,

                c.concentration_unit

            FROM evidence_rules er

            JOIN organisms org
                ON er.organism_id =
                   org.organism_id

            JOIN water_quality_factors w
                ON er.factor_id =
                   w.factor_id

            LEFT JOIN cases c
                ON er.case_id =
                   c.case_id

            WHERE
                LOWER(org.scientific_name)
                =
                LOWER(?)

                AND

                LOWER(w.factor_name)
                =
                LOWER(?)

            ORDER BY
                c.concentration_min,
                w.value_numeric,
                w.value_text;
            """,
            (
                species,
                factor_name
            )
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


# ============================================================
# RESPONSE DETAILS
# ============================================================

def get_response_details(
    species,
    response_type
):
    """
    Return known values for a microorganism response.
    """

    connection = get_connection()

    try:

        rows = connection.execute(
            """
            SELECT DISTINCT

                r.value_text,

                r.value_numeric,

                r.value_min,

                r.value_max,

                r.unit

            FROM evidence_rules er

            JOIN organisms org
                ON er.organism_id =
                   org.organism_id

            JOIN organism_responses r
                ON er.response_id =
                   r.response_id

            WHERE
                LOWER(org.scientific_name)
                =
                LOWER(?)

                AND

                LOWER(r.response_type)
                =
                LOWER(?)

            ORDER BY
                r.value_numeric,
                r.value_min,
                r.value_text;
            """,
            (
                species,
                response_type
            )
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


# ============================================================
# MEDIA FOR AN OBSERVATION
# ============================================================

def get_media_for_observation(
    observation_id
):
    """
    Return images/videos linked to one observation.
    """

    connection = get_connection()

    try:

        rows = connection.execute(
            """
            SELECT
                media_id,
                media_type,
                url,
                local_path,
                notes,
                source_type

            FROM media

            WHERE
                observation_id = ?

            ORDER BY
                media_type,
                media_id;
            """,
            (
                observation_id,
            )
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()

# ============================================================
# GENERAL / ORGANISM-LEVEL MEDIA
# ============================================================

def get_general_media(
    species,
    condition=None
):
    """
    Return media that belong to an organism/condition but are
    not tied to a specific observation.

    This currently covers the general reference videos stored
    in the healthy Nitzschia and Euglena worksheets.
    """

    connection = get_connection()

    try:

        if condition is None:

            rows = connection.execute(
                """
                SELECT
                    m.media_id,
                    m.media_type,
                    m.url,
                    m.local_path,
                    m.notes,
                    m.source_type,
                    m.condition,
                    m.sheet_name

                FROM media m

                JOIN organisms org
                    ON m.organism_id =
                       org.organism_id

                WHERE
                    LOWER(org.scientific_name)
                    =
                    LOWER(?)

                    AND

                    m.observation_id IS NULL

                ORDER BY
                    m.condition,
                    m.media_type,
                    m.media_id;
                """,
                (
                    species,
                )
            ).fetchall()

        else:

            rows = connection.execute(
                """
                SELECT
                    m.media_id,
                    m.media_type,
                    m.url,
                    m.local_path,
                    m.notes,
                    m.source_type,
                    m.condition,
                    m.sheet_name

                FROM media m

                JOIN organisms org
                    ON m.organism_id =
                       org.organism_id

                WHERE
                    LOWER(org.scientific_name)
                    =
                    LOWER(?)

                    AND

                    LOWER(m.condition)
                    =
                    LOWER(?)

                    AND

                    m.observation_id IS NULL

                ORDER BY
                    m.media_type,
                    m.media_id;
                """,
                (
                    species,
                    condition
                )
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()