"""Create/review the transform-safety manifest and materialise approved crops.

Examples
--------
Preview the audit:
    python src/audit_visual_references.py

Write/update the manifest without approving any image:
    python src/audit_visual_references.py --write

After a human sets review_status=verified_crop, transform_eligible=true and a
crop_box in the manifest, create the non-destructive crop:
    python src/audit_visual_references.py --write --apply-approved-crops
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from visual_reference_safety import (
    DATA_DIR,
    POLICY_FIELDS,
    POLICY_FILE,
    VERIFIED_CROPS,
    analyse_layout_signals,
    clean,
    load_policy,
    parse_bool,
    parse_crop_box,
    resolve_project_image_path,
)


DATASET = DATA_DIR / "v2_visual_conditioning_dataset.csv"


def portable_path(path):
    try:
        return str(Path(path).resolve().relative_to(DATA_DIR.parent.resolve()))
    except (OSError, ValueError):
        return str(path)


def build_rows():
    existing = load_policy()
    with DATASET.open("r", encoding="utf-8-sig", newline="") as handle:
        dataset_rows = list(csv.DictReader(handle))

    rows = []
    seen = set()
    for dataset_row in dataset_rows:
        record_id = clean(dataset_row.get("record_id"))
        if not record_id or record_id in seen:
            continue
        seen.add(record_id)
        image_path = resolve_project_image_path(dataset_row.get("image_path"))
        if image_path is None or not image_path.is_file():
            profile = {
                "composite_layout_risk": True,
                "automated_risk_reasons": ["image_missing"],
            }
        else:
            with Image.open(image_path) as image:
                profile = analyse_layout_signals(image)

        old = existing.get(record_id, {})
        risks = " | ".join(profile.get("automated_risk_reasons", []))
        old_status = clean(old.get("review_status"))
        review_status = old_status or (
            "context_only"
            if profile.get("composite_layout_risk")
            else "review_required"
        )
        eligible = (
            parse_bool(old.get("transform_eligible"))
            and review_status in {"verified_whole", "verified_crop"}
        )
        rows.append({
            "record_id": record_id,
            "image_path": portable_path(image_path) if image_path else "",
            "review_status": review_status,
            "transform_eligible": "true" if eligible else "false",
            "crop_path": clean(old.get("crop_path")),
            "crop_box": clean(old.get("crop_box")),
            "automated_risk_reasons": risks,
            "review_notes": clean(old.get("review_notes")) or (
                "Automatically routed to context-only review."
                if risks
                else "Human approval is required before transformation."
            ),
            "display_status": clean(old.get("display_status")) or "source_figure",
            "display_crop_path": clean(old.get("display_crop_path")),
            "display_crop_box": clean(old.get("display_crop_box")),
            "display_panel_label": clean(old.get("display_panel_label")),
            "display_notes": clean(old.get("display_notes")),
        })
    return rows


def apply_approved_crops(rows):
    VERIFIED_CROPS.mkdir(parents=True, exist_ok=True)
    created = []
    errors = []
    for row in rows:
        if not (
            row["review_status"] == "verified_crop"
            and parse_bool(row["transform_eligible"])
        ):
            continue
        crop_box = parse_crop_box(row.get("crop_box"))
        source = resolve_project_image_path(row.get("image_path"))
        if crop_box is None or source is None or not source.is_file():
            errors.append(row["record_id"])
            row["transform_eligible"] = "false"
            continue
        with Image.open(source) as image:
            width, height = image.size
            left, top, right, bottom = crop_box
            if right > width or bottom > height:
                errors.append(row["record_id"])
                row["transform_eligible"] = "false"
                continue
            destination = VERIFIED_CROPS / f"{row['record_id']}_verified_crop.png"
            image.crop(crop_box).convert("RGB").save(destination, "PNG")
            row["crop_path"] = portable_path(destination)
            created.append(str(destination))
    return created, errors


def write_manifest(rows):
    POLICY_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = POLICY_FILE.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POLICY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(POLICY_FILE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--apply-approved-crops", action="store_true")
    args = parser.parse_args()

    rows = build_rows()
    created = []
    crop_errors = []
    if args.apply_approved_crops:
        created, crop_errors = apply_approved_crops(rows)
    if args.write:
        write_manifest(rows)

    summary = {
        "references_audited": len(rows),
        "context_only": sum(r["review_status"] == "context_only" for r in rows),
        "review_required": sum(r["review_status"] == "review_required" for r in rows),
        "verified_whole": sum(r["review_status"] == "verified_whole" for r in rows),
        "verified_crop": sum(r["review_status"] == "verified_crop" for r in rows),
        "transform_eligible": sum(parse_bool(r["transform_eligible"]) for r in rows),
        "crops_created": len(created),
        "crop_errors": crop_errors,
        "manifest_written": bool(args.write),
        "manifest_path": str(POLICY_FILE),
    }
    print(json.dumps(summary, indent=2))
    if crop_errors:
        raise SystemExit(1)
    if not args.write:
        print("Preview only. Re-run with --write after reviewing the summary.")


if __name__ == "__main__":
    main()
