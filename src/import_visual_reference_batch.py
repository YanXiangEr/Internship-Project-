"""Import reviewed state-neutral microscopy references into V1, V2, and SQLite.

The importer is dry-run by default, idempotent by record/media ID, and never
creates biological observations, water factors, responses, cases, or evidence
rules.  It only expands the visual reference layer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "curated" / "visual_reference_candidates_2026-09-15.csv"
V1_PATH = PROJECT_ROOT / "data" / "world_model_v1_dataset.csv"
V2_PATH = PROJECT_ROOT / "data" / "v2_visual_conditioning_dataset.csv"
DB_PATH = PROJECT_ROOT / "db" / "microorganism_world_model.db"
ELIGIBLE_STATUS = "admit_state_neutral"


def clean(value):
    return "" if value is None else str(value).strip()


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_image_path(row):
    path = Path(clean(row.get("local_path")))
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_candidate(row):
    errors = []
    record_id = clean(row.get("record_id"))
    if not record_id:
        errors.append("missing record_id")
    if clean(row.get("condition_label")) != "state_neutral":
        errors.append("condition_label must be state_neutral")
    if clean(row.get("condition_label_supported")).lower() not in {"false", "0", "no"}:
        errors.append("state-neutral references must not claim a condition label")
    if not clean(row.get("license")):
        errors.append("missing license")
    path = candidate_image_path(row)
    if not path.exists():
        errors.append(f"missing image: {path}")
    elif clean(row.get("sha256")) and sha256(path) != clean(row.get("sha256")).lower():
        errors.append("SHA-256 mismatch")
    return errors


def v1_row(row):
    return {
        "record_id": clean(row["record_id"]),
        "species": clean(row["species"]),
        "qualitative_condition": "",
        "primary_water_factor": "",
        "primary_concentration_value": "",
        "primary_concentration_unit": "",
        "secondary_water_factor": "",
        "secondary_concentration_value": "",
        "secondary_concentration_unit": "",
        "exposure_time": "",
        "exposure_unit": "",
        "condition_label": "state_neutral",
        "state_label": "state_neutral",
        "visual_phenotype": "",
        "response_types": "",
        "response_labels": "",
        "visual_feature_1": "",
        "visual_feature_2": "",
        "description_target": "Exact-species state-neutral microscopy reference; no healthy/stressed or water-condition label is asserted.",
        "image_available": "True",
        "local_path": clean(row["local_path"]).replace("\\", "/"),
        "sha256": clean(row["sha256"]),
        "paired_visual_context_available": "False",
        "visual_simulation_training_permission": "False",
        "evidence_type": "external_state_neutral_reference",
        "dataset_origin": "external",
        "dataset_source": clean(row["source_id"]),
        "split_group": clean(row["source_id"]),
        "license": clean(row["license"]),
        "provenance_url": clean(row["source_url"]),
    }


def v2_row(row):
    return {
        "record_id": clean(row["record_id"]),
        "species": clean(row["species"]),
        "biological_state": "state_neutral",
        "water_condition": "",
        "primary_water_factor": "",
        "primary_concentration_value": "",
        "primary_concentration_unit": "",
        "secondary_water_factor": "",
        "secondary_concentration_value": "",
        "secondary_concentration_unit": "",
        "exposure_time": "",
        "exposure_unit": "",
        "response_types": "",
        "response_labels": "",
        "visual_phenotype": "",
        "visual_feature_1": "",
        "visual_feature_2": "",
        "visual_description": "Exact-species state-neutral microscopy reference; no healthy/stressed or water-condition label is asserted.",
        "visual_prompt": "",
        "image_path": clean(row["local_path"]).replace("\\", "/"),
        "sha256": clean(row["sha256"]),
        "split_group": clean(row["source_id"]),
        "evidence_type": "external_state_neutral_reference",
        "dataset_origin": "external",
        "license": clean(row["license"]),
        "provenance_url": clean(row["source_url"]),
    }


def append_missing(path, incoming):
    existing, fields = read_csv(path)
    existing_ids = {clean(row.get("record_id")) for row in existing}
    additions = [row for row in incoming if clean(row.get("record_id")) not in existing_ids]
    if additions:
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writerows(additions)
    return len(additions)


def backup(path, stamp):
    target = path.with_name(f"{path.name}.backup-{stamp}")
    shutil.copy2(path, target)
    return target


def upsert_media(connection, rows):
    organism_ids = dict(connection.execute("SELECT scientific_name, organism_id FROM organisms"))
    inserted = 0
    skipped = 0
    for row in rows:
        organism_id = organism_ids.get(clean(row["species"]))
        if organism_id is None:
            raise ValueError(f"Species is absent from organisms table: {row['species']}")
        media_id = clean(row["record_id"])
        exists = connection.execute("SELECT 1 FROM media WHERE media_id = ?", (media_id,)).fetchone()
        if exists:
            skipped += 1
            continue
        connection.execute(
            """INSERT INTO media
               (media_id, observation_id, organism_id, condition, media_type,
                url, local_path, notes, source_type, sheet_name, excel_row)
               VALUES (?, NULL, ?, 'state_neutral', 'image', ?, ?, ?,
                       'external_state_neutral_reference', NULL, NULL)""",
            (
                media_id,
                organism_id,
                clean(row["source_url"]),
                clean(row["local_path"]).replace("\\", "/"),
                f"{clean(row['source_title'])}; {clean(row['license'])}; {clean(row['attribution'])}",
            ),
        )
        inserted += 1
    return inserted, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    candidates_path = args.candidates if args.candidates.is_absolute() else PROJECT_ROOT / args.candidates
    rows, _ = read_csv(candidates_path)
    eligible = [row for row in rows if clean(row.get("admission_status")) == ELIGIBLE_STATUS]
    validation = {clean(row.get("record_id")): validate_candidate(row) for row in eligible}
    failures = {key: value for key, value in validation.items() if value}

    v1_existing, _ = read_csv(V1_PATH)
    v2_existing, _ = read_csv(V2_PATH)
    v1_ids = {clean(row.get("record_id")) for row in v1_existing}
    v2_ids = {clean(row.get("record_id")) for row in v2_existing}
    with sqlite3.connect(DB_PATH) as connection:
        db_ids = {row[0] for row in connection.execute("SELECT media_id FROM media")}

    summary = {
        "candidate_rows": len(rows),
        "eligible_state_neutral_rows": len(eligible),
        "held_rows": len(rows) - len(eligible),
        "validation_failures": failures,
        "v1_new_rows": sum(clean(row["record_id"]) not in v1_ids for row in eligible),
        "v2_new_rows": sum(clean(row["record_id"]) not in v2_ids for row in eligible),
        "database_new_media": sum(clean(row["record_id"]) not in db_ids for row in eligible),
        "biological_evidence_rules_added": 0,
        "applied": False,
        "backup_paths": [],
    }

    if failures:
        print(json.dumps(summary, indent=2))
        raise SystemExit("Validation failed; nothing was changed.")

    if args.apply:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backups = [backup(path, stamp) for path in (V1_PATH, V2_PATH, DB_PATH)]
        append_missing(V1_PATH, [v1_row(row) for row in eligible])
        append_missing(V2_PATH, [v2_row(row) for row in eligible])
        with sqlite3.connect(DB_PATH) as connection:
            upsert_media(connection, eligible)
            connection.commit()
        summary["applied"] = True
        summary["backup_paths"] = [str(path) for path in backups]

    print(json.dumps(summary, indent=2))
    if not args.apply:
        print("Dry run only. Re-run with --apply after reviewing this summary.")


if __name__ == "__main__":
    main()
