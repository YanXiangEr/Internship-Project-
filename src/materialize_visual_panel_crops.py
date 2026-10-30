"""Materialise explicitly reviewed Word-figure panel crops.

The source evidence images are never overwritten. Crop coordinates live in a
reviewable CSV plan, and this command validates every box before writing a
separate display or transform crop.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

from docx_image_alignment import assess_transform_crop_quality

from visual_reference_safety import (
    POLICY_FIELDS,
    POLICY_FILE,
    load_policy,
    parse_crop_box,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAN_FILE = PROJECT_ROOT / "data" / "visual_reference_panel_crop_plan.csv"


def clean(value):
    return "" if value is None else str(value).strip()


def resolve_path(value):
    path = Path(clean(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plan(path=PLAN_FILE):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_box(box, image_size):
    if box is None:
        return False
    left, top, right, bottom = box
    width, height = image_size
    return 0 <= left < right <= width and 0 <= top < bottom <= height


def materialize(rows, apply=False):
    outputs = []
    errors = []
    for row in rows:
        record_id = clean(row.get("record_id"))
        source = resolve_path(row.get("source_image_path"))
        if not source.is_file():
            errors.append(f"{record_id}: source image is missing: {source}")
            continue
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            for crop_kind in ("display", "transform"):
                path_value = clean(row.get(f"{crop_kind}_crop_path"))
                box_value = clean(row.get(f"{crop_kind}_crop_box"))
                if not path_value and not box_value:
                    continue
                destination = resolve_path(path_value)
                box = parse_crop_box(box_value)
                if not path_value or not validate_box(box, image.size):
                    errors.append(
                        f"{record_id}: invalid {crop_kind} crop definition "
                        f"for source size {image.size}: {box_value!r}"
                    )
                    continue
                preview = image.crop(box)
                quality = (
                    assess_transform_crop_quality(preview)
                    if crop_kind == "transform"
                    else {"passed": True, "reasons": []}
                )
                if crop_kind == "transform" and not quality["passed"]:
                    errors.append(
                        f"{record_id}: transform crop failed quality gates: "
                        + ", ".join(quality["reasons"])
                    )
                    continue
                if apply:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(destination.name + ".tmp")
                    image.crop(box).save(temporary, format="PNG")
                    temporary.replace(destination)
                outputs.append(
                    {
                        "record_id": record_id,
                        "kind": crop_kind,
                        "source": str(source),
                        "destination": str(destination),
                        "crop_box": list(box),
                        "created": bool(apply),
                        "sha256": sha256(destination) if apply else None,
                        "quality": quality,
                    }
                )
    return outputs, errors


def update_policy(rows, apply=False):
    current = load_policy(POLICY_FILE)
    for plan in rows:
        record_id = clean(plan.get("record_id"))
        row = dict(current.get(record_id, {}))
        transform_status = clean(plan.get("transform_status")) or "context_only"
        row.update(
            {
                "record_id": record_id,
                "image_path": clean(plan.get("source_image_path")),
                "review_status": transform_status,
                "transform_eligible": (
                    "true" if transform_status == "verified_crop" else "false"
                ),
                "crop_path": clean(plan.get("transform_crop_path")),
                "crop_box": clean(plan.get("transform_crop_box")),
                "automated_risk_reasons": clean(
                    row.get("automated_risk_reasons")
                ) or "source_document_multi_panel",
                "review_notes": clean(row.get("review_notes")) or clean(
                    plan.get("rationale")
                ),
                "display_status": clean(plan.get("display_status")),
                "display_crop_path": clean(plan.get("display_crop_path")),
                "display_crop_box": clean(plan.get("display_crop_box")),
                "display_panel_label": clean(plan.get("panel_label")),
                "display_notes": clean(plan.get("rationale")),
            }
        )
        current[record_id] = row

    policy_rows = []
    for record_id in sorted(current):
        source = current[record_id]
        policy_rows.append({field: clean(source.get(field)) for field in POLICY_FIELDS})

    if apply:
        temporary = POLICY_FILE.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=POLICY_FIELDS)
            writer.writeheader()
            writer.writerows(policy_rows)
        temporary.replace(POLICY_FILE)
    return policy_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--update-policy", action="store_true")
    parser.add_argument("--plan", type=Path, default=PLAN_FILE)
    args = parser.parse_args()
    rows = load_plan(args.plan)
    outputs, errors = materialize(rows, apply=args.apply)
    policy_rows = update_policy(
        rows,
        apply=args.apply and args.update_policy,
    )
    print(
        json.dumps(
            {
                "records_in_plan": len(rows),
                "crops_validated": len(outputs),
                "crops_created": sum(item["created"] for item in outputs),
                "errors": errors,
                "policy_rows": len(policy_rows),
                "policy_updated": bool(args.apply and args.update_policy),
                "outputs": outputs,
            },
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)
    if not args.apply:
        print("Dry run only. Re-run with --apply to materialise reviewed crops.")


if __name__ == "__main__":
    main()
