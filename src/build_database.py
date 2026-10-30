import json
import sqlite3
from pathlib import Path
from normalize_responses import (
    normalize_organism_response
)
import pandas as pd
from config import (
    EXCEL_FILE,
    WORD_FILE,
    DATABASE_FILE,
    DB_DIR,
    IMAGE_DIR,
)
from ingest_excel import read_all_sheets
from ingest_word import read_word_images
from normalize_water import normalize_water_quality
from case_rules import apply_case_rules
from build_relationships import (
    build_relationships
)

# ============================================================
# HELPERS
# ============================================================

def clean_value(value):
    """
    Convert pandas NaN values to None so SQLite can store them.
    """

    if pd.isna(value):
        return None

    return value


def row_to_json(row):
    """
    Preserve the complete original spreadsheet row as JSON.

    This means that if you add new spreadsheet columns later,
    the information is still retained even if there is not yet
    a dedicated SQLite column for it.
    """

    data = {}

    for column, value in row.items():

        value = clean_value(value)

        # Convert values that JSON may not understand directly.
        if isinstance(value, pd.Timestamp):
            value = value.isoformat()

        data[str(column)] = value

    return json.dumps(
        data,
        ensure_ascii=False,
        default=str
    )


def find_column(row, possible_names):
    """
    Find the first matching column from a collection of possible
    spreadsheet column names.

    Matching is case-insensitive and ignores surrounding spaces.
    """

    normalized_columns = {
        str(column).strip().lower(): column
        for column in row.index
    }

    for possible_name in possible_names:

        key = possible_name.strip().lower()

        if key in normalized_columns:

            actual_column = normalized_columns[key]

            value = clean_value(
                row.get(actual_column)
            )

            return value

    return None


# ============================================================
# DATABASE CREATION
# ============================================================

def create_database(connection):
    """
    Execute db/schema.sql to create all database tables.
    """

    schema_file = DB_DIR / "schema.sql"

    if not schema_file.exists():
        raise FileNotFoundError(
            f"Could not find schema file:\n{schema_file}"
        )

    with open(
        schema_file,
        "r",
        encoding="utf-8"
    ) as file:

        schema_sql = file.read()

    connection.executescript(schema_sql)


# ============================================================
# ORGANISMS
# ============================================================

def insert_organisms(
    connection,
    observations_df
):
    """
    Insert each unique microorganism into the organisms table.
    """

    organisms = (
        observations_df["species"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    for scientific_name in organisms:

        connection.execute(
            """
            INSERT OR IGNORE INTO organisms (
                scientific_name
            )
            VALUES (?)
            """,
            (scientific_name,)
        )

    connection.commit()


def get_organism_lookup(connection):
    """
    Return:
        {
            "Nitzschia palea": 1,
            ...
        }
    """

    rows = connection.execute(
        """
        SELECT
            organism_id,
            scientific_name
        FROM organisms
        """
    ).fetchall()

    return {
        scientific_name: organism_id
        for organism_id, scientific_name in rows
    }


# ============================================================
# OBSERVATIONS
# ============================================================

def insert_observations(
    connection,
    observations_df,
    organism_lookup
):
    """
    Insert every spreadsheet observation into SQLite.
    """

    inserted = 0

    for _, row in observations_df.iterrows():

        species = clean_value(
            row.get("species")
        )

        if species is None:
            continue

        organism_id = organism_lookup[
            str(species).strip()
        ]

        observation_id = clean_value(
            row.get("observation_id")
        )

        condition = clean_value(
            row.get("condition")
        )

        sheet_name = clean_value(
            row.get("sheet_name")
        )

        excel_row = clean_value(
            row.get("excel_row")
        )

        # ----------------------------------------------------
        # Try several plausible column names so minor changes
        # in the spreadsheet do not immediately break import.
        # ----------------------------------------------------

        source_url = find_column(
            row,
            [
                "Image URL",
                "Source URL",
                "URL",
                "Article URL",
                "Reference URL"
            ]
        )

        figure_no = find_column(
            row,
            [
                "Figure no.",
                "Figure No.",
                "Figure",
                "Figure number",
                "Figure Number"
            ]
        )

        description = find_column(
            row,
            [
                "Description",
                "Morphological Description",
                "Morphology",
                "Notes"
            ]
        )

        image_type = find_column(
            row,
            [
                "Image type",
                "Image Type",
                "Microscopy type",
                "Microscopy Type"
            ]
        )

        magnification = find_column(
            row,
            [
                "Magnification",
                "Scale",
                "Scale bar",
                "Scale Bar"
            ]
        )

        raw_json = row_to_json(row)

        connection.execute(
            """
            INSERT OR REPLACE INTO observations (
                observation_id,
                organism_id,
                condition,
                sheet_name,
                excel_row,
                source_url,
                figure_no,
                description,
                image_type,
                magnification,
                raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation_id,
                organism_id,
                condition,
                sheet_name,
                excel_row,
                source_url,
                figure_no,
                description,
                image_type,
                magnification,
                raw_json
            )
        )

        inserted += 1

    connection.commit()

    return inserted


# ============================================================
# MEDIA
# ============================================================

def insert_media(
    connection,
    media_df,
    organism_lookup
):
    """
    Insert images and videos into the generic media table.
    """

    if media_df is None or media_df.empty:
        return 0

    inserted = 0

    for _, row in media_df.iterrows():

        species = clean_value(
            row.get("species")
        )

        if species is None:
            continue

        species = str(species).strip()

        if species not in organism_lookup:
            continue

        organism_id = organism_lookup[
            species
        ]

        connection.execute(
            """
            INSERT OR REPLACE INTO media (
                media_id,
                observation_id,
                organism_id,
                condition,
                media_type,
                url,
                local_path,
                notes,
                source_type,
                sheet_name,
                excel_row
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean_value(row.get("media_id")),
                clean_value(row.get("observation_id")),
                organism_id,
                clean_value(row.get("condition")),
                clean_value(row.get("media_type")),
                clean_value(row.get("url")),
                clean_value(row.get("local_path")),
                clean_value(row.get("notes")),
                clean_value(row.get("source_type")),
                clean_value(row.get("sheet_name")),
                clean_value(row.get("excel_row"))
            )
        )

        inserted += 1

    connection.commit()

    return inserted


# ============================================================
# WATER QUALITY NORMALIZATION
# ============================================================

def insert_water_quality_factors(
    connection,
    observations_df
):
    """
    Normalize the Stressors/environment text and insert the
    resulting structured water-quality factors.

    One observation may produce many factors.
    """

    inserted = 0

    for _, row in observations_df.iterrows():

        observation_id = clean_value(
            row.get("observation_id")
        )

        if observation_id is None:
            continue

        stressor_text = find_column(
            row,
            [
                "Stressors",
                "Stressor",
                "Water Quality",
                "Water quality",
                "Environmental condition",
                "Environmental Condition",
                "Environmental conditions",
                "Environmental Conditions"
            ]
        )

        if stressor_text is None:
            continue

        factors = normalize_water_quality(
            observation_id,
            stressor_text
        )

        for factor in factors:

            connection.execute(
                """
                INSERT INTO water_quality_factors (
                    observation_id,
                    case_id,
                    factor_name,
                    factor_category,
                    value_text,
                    value_numeric,
                    unit,
                    qualifier,
                    evidence_text,
                    extraction_method,
                    confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    factor["observation_id"],
                    None,
                    factor["factor_name"],
                    factor["factor_category"],
                    factor["value_text"],
                    factor["value_numeric"],
                    factor["unit"],
                    factor["qualifier"],
                    factor["evidence_text"],
                    factor["extraction_method"],
                    factor["confidence"]
                )
            )

            inserted += 1

    connection.commit()

    return inserted

# ============================================================
# ORGANISM RESPONSE NORMALIZATION
# ============================================================

def insert_organism_responses(
    connection,
    observations_df
):
    """
    Normalize morphology / behaviour / biological-response
    descriptions from every observation.

    Healthy and stressed observations are both processed.
    """

    inserted = 0

    for _, row in observations_df.iterrows():

        observation_id = clean_value(
            row.get("observation_id")
        )

        species = clean_value(
            row.get("species")
        )

        if (
            observation_id is None
            or species is None
        ):
            continue

        description = find_column(
            row,
            [
                (
                    "Figure legend/Image description "
                    "(including morphological features "
                    "and other notes/observations)"
                ),
                "Figure legend/Image description",
                "Morphological Description",
                "Morphology",
                "Description"
            ]
        )

        if description is None:
            continue

        responses = normalize_organism_response(
            observation_id=observation_id,
            species=species,
            description=description
        )

        for response in responses:

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
                    response["observation_id"],
                    None,
                    response["response_type"],
                    response["response_category"],
                    response["value_text"],
                    response["value_numeric"],
                    response["value_min"],
                    response["value_max"],
                    response["unit"],
                    response["qualifier"],
                    response["evidence_text"],
                    response["extraction_method"],
                    response["confidence"]
                )
            )

            inserted += 1

    connection.commit()

    return inserted


# ============================================================
# DATABASE RESET
# ============================================================

def reset_database_file():
    """
    Rebuild from scratch each time.

    For this stage of the project, Excel/Word remain the editable
    source of truth and SQLite is generated from them.
    """

    DATABASE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if DATABASE_FILE.exists():
        DATABASE_FILE.unlink()


# ============================================================
# MAIN BUILD PIPELINE
# ============================================================

def build_database():

    print("=" * 65)
    print("MICROORGANISM WORLD MODEL DATABASE BUILDER")
    print("=" * 65)

    # --------------------------------------------------------
    # Check source files
    # --------------------------------------------------------

    if not EXCEL_FILE.exists():

        raise FileNotFoundError(
            f"\nExcel file not found:\n{EXCEL_FILE}"
        )

    if not WORD_FILE.exists():

        raise FileNotFoundError(
            f"\nWord file not found:\n{WORD_FILE}"
        )

    # --------------------------------------------------------
    # Step 1: Read Excel
    # --------------------------------------------------------

    print("\n[1/9] Reading Excel workbook...")

    observations_df, excel_media_df = (
        read_all_sheets(EXCEL_FILE)
    )

    print(
        f"      Observations found: "
        f"{len(observations_df)}"
    )

    if excel_media_df is not None:
        print(
            f"      Excel media found: "
            f"{len(excel_media_df)}"
        )

    # --------------------------------------------------------
    # Step 2: Read Word images
    # --------------------------------------------------------

    print("\n[2/9] Reading Word microscopy images...")

    word_media_df = read_word_images(
        WORD_FILE,
        output_directory=IMAGE_DIR
    )

    print(
        f"      Word images extracted: "
        f"{len(word_media_df)}"
    )

    # --------------------------------------------------------
    # Step 3: Combine media
    # --------------------------------------------------------

    print("\n[3/9] Combining media records...")

    media_frames = []

    if (
        excel_media_df is not None
        and not excel_media_df.empty
    ):
        media_frames.append(
            excel_media_df
        )

    if (
        word_media_df is not None
        and not word_media_df.empty
    ):
        media_frames.append(
            word_media_df
        )

    if media_frames:

        media_df = pd.concat(
            media_frames,
            ignore_index=True
        )

    else:

        media_df = pd.DataFrame()

    print(
        f"      Total media records: "
        f"{len(media_df)}"
    )

    # --------------------------------------------------------
    # Step 4: Create SQLite database
    # --------------------------------------------------------

    print("\n[4/9] Creating SQLite database...")

    reset_database_file()

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    # Enforce foreign keys.
    connection.execute(
        "PRAGMA foreign_keys = ON;"
    )

    try:

        create_database(
            connection
        )

        # ----------------------------------------------------
        # Step 5: Insert core data
        # ----------------------------------------------------

        print("\n[5/9] Inserting database records...")

        insert_organisms(
            connection,
            observations_df
        )

        organism_lookup = (
            get_organism_lookup(
                connection
            )
        )

        observation_count = (
            insert_observations(
                connection,
                observations_df,
                organism_lookup
            )
        )

        media_count = insert_media(
            connection,
            media_df,
            organism_lookup
        )

        # ----------------------------------------------------
        # Step 6: Normalize water quality
        # ----------------------------------------------------

        print(
            "\n[6/9] Normalizing "
            "water-quality information..."
        )

        water_factor_count = (
            insert_water_quality_factors(
                connection,
                observations_df
            )
        )

        # --------------------------------------------------------
        # Step 7: Normalize organism responses
        # --------------------------------------------------------

        print(
            "\n[7/9] Normalizing "
            "microorganism responses..."
        )

        response_count = (
            insert_organism_responses(
                connection,
                observations_df
            )
        )

        # --------------------------------------------------------
        # Step 8: Split complex observations into cases
        # --------------------------------------------------------

        print(
            "\n[8/9] Applying experimental "
            "case / treatment rules..."
        )

        apply_case_rules(
            connection
        )

        # --------------------------------------------------------
        # Step 9: Build evidence relationships
        # --------------------------------------------------------

        print(
            "\n[9/9] Building water-quality "
            "to organism-response relationships..."
        )

        relationship_stats = build_relationships(
            connection
        )

        # --------------------------------------------------------
        # Final counts AFTER case rules
        # --------------------------------------------------------

        organism_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM organisms
            """
        ).fetchone()[0]

        observation_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM observations
            """
        ).fetchone()[0]

        media_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM media
            """
        ).fetchone()[0]

        image_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM media
            WHERE media_type = 'image'
            """
        ).fetchone()[0]

        video_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM media
            WHERE media_type = 'video'
            """
        ).fetchone()[0]

        water_factor_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM water_quality_factors
            """
        ).fetchone()[0]

        response_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM organism_responses
            """
        ).fetchone()[0]

        case_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM cases
            """
        ).fetchone()[0]

        relationship_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM evidence_rules
            """
        ).fetchone()[0]

        print("\n" + "=" * 65)
        print("DATABASE BUILD COMPLETE")
        print("=" * 65)

        print(
            f"\nDatabase:\n{DATABASE_FILE}"
        )

        print(
            f"\nOrganisms:              "
            f"{organism_count}"
        )

        print(
            f"Observations:           "
            f"{observation_count}"
        )

        print(
            f"Media records:          "
            f"{media_count}"
        )

        print(
            f"  Images:               "
            f"{image_count}"
        )

        print(
            f"  Videos:               "
            f"{video_count}"
        )

        print(
            f"Water-quality factors:  "
            f"{water_factor_count}"
        )

        print(
            f"Organism responses:      "
            f"{response_count}"
        )

        print(
            f"Experimental cases:      "
            f"{case_count}"
        )

        print(
            f"Evidence relationships:   "
            f"{relationship_count}"
        )

    finally:

        connection.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    build_database()