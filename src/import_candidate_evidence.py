"""Safely import curated literature candidates into the world-model SQLite DB.

The command is dry-run by default. Pass --apply to write. Rows carrying
source warnings are excluded unless --include-warnings is also supplied.
Manual-review rows are never imported by this version.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "candidate_id", "species", "proposed_state", "evidence_type",
    "relationship_type", "default_rank_score", "primary_factor",
    "factor_category", "factor_value_min", "factor_value_max",
    "factor_unit", "factor_qualifier", "exposure_duration_value",
    "exposure_duration_unit", "response_type", "response_direction",
    "response_value_min", "response_value_max", "response_unit",
    "response_text", "strain", "source_id", "source_locator",
    "duplicate_status", "admission_status", "caution",
}

REQUIRED_TABLES = {
    "organisms", "observations", "water_quality_factors",
    "organism_responses", "cases", "evidence_rules",
}

DEFAULT_ADMISSION_STATUSES = {"admit", "admit_as_co_observed"}
WARNING_ADMISSION_STATUS = "admit_with_source_warning"
IMPORT_METHOD = "candidate_workbook_import_v1"
DEFAULT_PREFIX = "research_20260913"


def database_relationship_type(value: str | None) -> str:
    """Map research-layer wording onto the project's existing vocabulary."""
    normalized = (value or "").strip().lower()
    if normalized == "field_co_observed":
        return "co_observed"
    if normalized in {"experimentally_observed", "co_observed"}:
        return normalized
    raise ValueError(f"Unsupported relationship_type: {value!r}")


@dataclass
class ImportSummary:
    workbook_rows: int = 0
    eligible_rows: int = 0
    excluded_manual_review: int = 0
    excluded_warning_rows: int = 0
    observations: int = 0
    factors: int = 0
    responses: int = 0
    cases: int = 0
    evidence_rules: int = 0
    applied: bool = False
    backup_path: str | None = None


def clean(value: Any) -> Any:
    """Convert spreadsheet blanks/NaN values into Python None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if hasattr(value, "item"):
        return value.item()
    return value


def number(value: Any) -> float | None:
    value = clean(value)
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value: {value!r}")
    return result


def text_value(minimum: float | None, maximum: float | None, unit: str | None) -> str | None:
    if minimum is None and maximum is None:
        return None
    suffix = f" {unit}" if unit else ""
    if minimum is not None and maximum is not None and minimum != maximum:
        return f"{minimum:g}-{maximum:g}{suffix}"
    value = minimum if minimum is not None else maximum
    return f"{value:g}{suffix}"


def condition_for(proposed_state: str | None) -> str:
    """Use healthy/stressed only when explicitly supported; otherwise contextual."""
    state = (proposed_state or "").strip().lower()
    if state == "healthy_candidate":
        return "healthy"
    if state == "stressed":
        return "stressed"
    return "contextual"


def response_category(response_type: str) -> str:
    if response_type in {"cell_density", "cell_count", "growth", "growth_rate"}:
        return "growth"
    if response_type in {
        "photosynthesis", "photosynthetic_activity",
        "photosynthetic_electron_transport_rate", "chlorophyll_a_content",
    }:
        return "photosynthesis"
    if response_type in {"lipid_content", "oil_content", "carbon_allocation_to_lipids"}:
        return "metabolism"
    if response_type in {
        "MDA_content", "hydrogen_peroxide_content", "superoxide_anion_generation_rate",
        "SOD_activity", "POD_activity", "CAT_activity", "APX_activity",
        "glutathione_content",
        "ascorbic_acid_content", "proline_content", "soluble_protein", "soluble_sugar",
    }:
        return "stress_physiology"
    if response_type in {"geosmin_production", "extracellular_geosmin_release", "geosmin_concentration"}:
        return "secondary_metabolite"
    if response_type == "metal_biosorption":
        return "bioremediation"
    if response_type in {
        "BOD_removal_efficiency", "chloride_removal", "nitrate_removal",
    }:
        return "bioremediation"
    if response_type in {"presence", "relative_abundance"}:
        return "occurrence"
    if response_type in {"cell_mortality_fraction", "cell_death", "reduced_viability"}:
        return "viability"
    if response_type in {"cell_length", "cell_width", "cell_size_distribution"}:
        return "morphology"
    return "other"


def load_candidates(workbook_path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    candidates = pd.read_excel(workbook_path, sheet_name="Candidate observations")
    missing = REQUIRED_COLUMNS.difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidate workbook is missing columns: {sorted(missing)}")

    sources_df = pd.read_excel(workbook_path, sheet_name="Sources")
    if "source_id" not in sources_df or "url" not in sources_df:
        raise ValueError("Sources sheet must contain source_id and url columns")
    sources = {
        str(row["source_id"]).strip(): {str(k): clean(v) for k, v in row.items()}
        for _, row in sources_df.iterrows()
    }
    return candidates, sources


def select_candidates(candidates: pd.DataFrame, include_warnings: bool) -> tuple[pd.DataFrame, ImportSummary]:
    statuses = set(DEFAULT_ADMISSION_STATUSES)
    if include_warnings:
        statuses.add(WARNING_ADMISSION_STATUS)

    normalized = candidates["admission_status"].fillna("").astype(str).str.strip()
    selected = candidates[normalized.isin(statuses)].copy()
    summary = ImportSummary(
        workbook_rows=len(candidates),
        eligible_rows=len(selected),
        excluded_manual_review=int(normalized.str.contains("manual_review", regex=False).sum()),
        excluded_warning_rows=int((normalized == WARNING_ADMISSION_STATUS).sum()) if not include_warnings else 0,
    )
    return selected, summary


def validate_database(connection: sqlite3.Connection) -> dict[str, int]:
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = REQUIRED_TABLES.difference(tables)
    if missing:
        raise ValueError(f"Database is missing required tables: {sorted(missing)}")

    organism_lookup = {
        str(name).strip().lower(): int(organism_id)
        for organism_id, name in connection.execute(
            "SELECT organism_id, scientific_name FROM organisms"
        )
    }
    return organism_lookup


def make_backup(connection: sqlite3.Connection, database_path: Path) -> Path:
    """Create a transaction-consistent SQLite backup."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = database_path.with_name(f"{database_path.name}.backup-{stamp}")
    backup_connection = sqlite3.connect(backup_path)
    try:
        connection.backup(backup_connection)
    finally:
        backup_connection.close()
    return backup_path


def import_candidates(
    database_path: Path,
    workbook_path: Path,
    *,
    apply: bool = False,
    include_warnings: bool = False,
    backup: bool = True,
    prefix: str = DEFAULT_PREFIX,
) -> ImportSummary:
    candidates, sources = load_candidates(workbook_path)
    selected, summary = select_candidates(candidates, include_warnings)

    if not database_path.exists():
        raise FileNotFoundError(f"Database not found: {database_path}")

    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        organisms = validate_database(connection)
        missing_species = sorted({
            str(value).strip() for value in selected["species"]
            if str(value).strip().lower() not in organisms
        })
        if missing_species:
            raise ValueError(f"Species not present in database: {missing_species}")

        if not apply:
            summary.observations = summary.eligible_rows
            summary.factors = summary.eligible_rows
            summary.responses = summary.eligible_rows
            summary.cases = sum(
                1 for _, row in selected.iterrows()
                if number(row["factor_value_min"]) is not None or number(row["factor_value_max"]) is not None
            )
            summary.evidence_rules = summary.eligible_rows
            return summary

        if backup:
            summary.backup_path = str(make_backup(connection, database_path))

        connection.execute("BEGIN IMMEDIATE")
        for row_index, row in selected.iterrows():
            record = {str(k): clean(v) for k, v in row.items()}
            candidate_id = str(record["candidate_id"])
            species = str(record["species"])
            observation_id = f"{prefix}_{candidate_id}"
            case_id = f"{observation_id}_case"
            organism_id = organisms[species.lower()]
            source = sources.get(str(record["source_id"]), {})

            factor_min = number(record["factor_value_min"])
            factor_max = number(record["factor_value_max"])
            response_min = number(record["response_value_min"])
            response_max = number(record["response_value_max"])
            duration = number(record["exposure_duration_value"])
            rank_score = number(record["default_rank_score"])
            if rank_score is None:
                raise ValueError(f"{candidate_id} has no default_rank_score")
            if not 0 <= rank_score <= 1:
                raise ValueError(f"{candidate_id} rank score is outside 0..1")

            # Idempotent replacement scoped only to this importer's stable IDs.
            connection.execute("DELETE FROM observations WHERE observation_id = ?", (observation_id,))
            raw_json = json.dumps({"candidate": record, "source": source}, ensure_ascii=False, default=str)
            connection.execute(
                """INSERT INTO observations (
                    observation_id, organism_id, condition, sheet_name, excel_row,
                    source_url, figure_no, description, image_type, magnification, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    observation_id, organism_id, condition_for(record["proposed_state"]),
                    f"candidate_evidence_{prefix}", int(row_index) + 2,
                    source.get("url"), record["source_locator"], record["response_text"],
                    None, None, raw_json,
                ),
            )
            summary.observations += 1

            has_numeric_context = factor_min is not None or factor_max is not None
            active_case_id = case_id if has_numeric_context else None
            if has_numeric_context:
                connection.execute(
                    """INSERT INTO cases (
                        case_id, observation_id, case_label, case_type, treatment_name,
                        concentration_min, concentration_max, concentration_unit,
                        duration_numeric, duration_unit, panel_label, morphotype, notes, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        case_id, observation_id, candidate_id, "curated_literature_condition",
                        record["primary_factor"], factor_min, factor_max, record["factor_unit"],
                        duration, record["exposure_duration_unit"], None, record["strain"],
                        record["caution"], None,
                    ),
                )
                summary.cases += 1

            factor_text = text_value(factor_min, factor_max, record["factor_unit"])
            factor_numeric = factor_min if factor_min is not None and factor_min == factor_max else None
            factor_cursor = connection.execute(
                """INSERT INTO water_quality_factors (
                    observation_id, case_id, factor_name, factor_category, value_text,
                    value_numeric, unit, qualifier, evidence_text, extraction_method, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    observation_id, active_case_id, record["primary_factor"], record["factor_category"],
                    factor_text, factor_numeric, record["factor_unit"], record["factor_qualifier"],
                    record["response_text"], "curated_primary_literature", None,
                ),
            )
            factor_id = factor_cursor.lastrowid
            summary.factors += 1

            response_numeric = response_min if response_min is not None and response_min == response_max else None
            response_text_value = text_value(response_min, response_max, record["response_unit"])
            if response_text_value is None:
                response_text_value = record["response_direction"]
            qualifier = "; ".join(filter(None, [record["response_direction"], record["strain"]])) or None
            response_cursor = connection.execute(
                """INSERT INTO organism_responses (
                    observation_id, case_id, response_type, response_category, value_text,
                    value_numeric, value_min, value_max, unit, qualifier, evidence_text,
                    extraction_method, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    observation_id, active_case_id, record["response_type"],
                    response_category(str(record["response_type"])), response_text_value,
                    response_numeric, response_min, response_max, record["response_unit"],
                    qualifier, record["response_text"], "curated_primary_literature", None,
                ),
            )
            response_id = response_cursor.lastrowid
            summary.responses += 1

            note_parts = [
                "Imported from curator-reviewed candidate workbook.",
                "Evidence strength is a ranking value, not a probability.",
                record["caution"],
            ]
            connection.execute(
                """INSERT INTO evidence_rules (
                    organism_id, observation_id, case_id, factor_id, response_id,
                    relationship_type, evidence_strength, direction, notes, source_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    organism_id, observation_id, active_case_id, factor_id, response_id,
                    database_relationship_type(record["relationship_type"]), rank_score, "water_to_response",
                    " ".join(part for part in note_parts if part), IMPORT_METHOD,
                ),
            )
            summary.evidence_rules += 1

        connection.commit()
        summary.applied = True
        return summary
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=project_root / "data" / "curated" / "microorganism_model_candidate_evidence_2026-09-13.xlsx")
    parser.add_argument("--database", type=Path, default=project_root / "db" / "microorganism_world_model.db")
    parser.add_argument("--apply", action="store_true", help="Write eligible rows. Without this flag the command is read-only.")
    parser.add_argument("--include-warnings", action="store_true", help="Also import rows whose status is admit_with_source_warning.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create a timestamped database backup before applying.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Stable namespace for imported observation IDs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = import_candidates(
        args.database, args.workbook, apply=args.apply,
        include_warnings=args.include_warnings, backup=not args.no_backup,
        prefix=args.prefix,
    )
    print(json.dumps(asdict(summary), indent=2))
    if not args.apply:
        print("Dry run only. Re-run with --apply after reviewing this summary.")


if __name__ == "__main__":
    main()
