"""Conservative transform eligibility for microscopy references.

Full evidence figures remain valid provenance, but whole-image visual
transformations are only allowed for references that have been explicitly
reviewed as clean single-field images or for separately saved reviewed crops.
Automated layout signals support review; they never approve a transformation.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from PIL import Image, ImageStat

from docx_image_alignment import assess_transform_crop_quality


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
POLICY_FILE = DATA_DIR / "visual_reference_transform_policy.csv"
PROCESSED_IMAGES = DATA_DIR / "processed" / "images"
VERIFIED_CROPS = DATA_DIR / "processed" / "visual_transform_crops"

POLICY_FIELDS = [
    "record_id",
    "image_path",
    "review_status",
    "transform_eligible",
    "crop_path",
    "crop_box",
    "automated_risk_reasons",
    "review_notes",
    "display_status",
    "display_crop_path",
    "display_crop_box",
    "display_panel_label",
    "display_notes",
]


def clean(value):
    return "" if value is None else str(value).strip()


def parse_bool(value):
    return clean(value).lower() in {"1", "true", "yes", "y"}


def resolve_project_image_path(value):
    """Resolve portable, relative, or old absolute project image paths."""

    raw = clean(value)
    if not raw:
        return None

    candidate = Path(raw)
    if candidate.is_file():
        return candidate

    if not candidate.is_absolute():
        project_candidate = PROJECT_ROOT / candidate
        if project_candidate.is_file():
            return project_candidate

    portable_name = raw.replace("\\", "/").rsplit("/", 1)[-1]
    fallback = PROCESSED_IMAGES / portable_name
    return fallback if fallback.is_file() else candidate


def analyse_layout_signals(image):
    """Return explainable conservative composite/annotation signals.

    The detector intentionally favours recall. Its output can block or route a
    reference for review, but it can never mark a reference transform-safe.
    """

    rgb = image.convert("RGB")
    width, height = rgb.size
    aspect_ratio = max(width, height) / max(1, min(width, height))

    sample = rgb.copy()
    sample.thumbnail((512, 512))
    sample_width, sample_height = sample.size
    pixels = sample.load()

    reasons = []
    if aspect_ratio >= 2.2:
        reasons.append("extreme_aspect_ratio")

    # Long saturated red rows/columns commonly indicate callout boxes or
    # panel borders. Biological red fluorescence is usually spatially local,
    # rather than a straight line spanning a large fraction of an axis.
    red_by_column = [0] * sample_width
    red_by_row = [0] * sample_height
    for y in range(sample_height):
        for x in range(sample_width):
            red, green, blue = pixels[x, y]
            is_red_markup = (
                red > 130
                and red - green > 35
                and red - blue > 35
                and red > green * 1.35
                and red > blue * 1.35
            )
            if is_red_markup:
                red_by_column[x] += 1
                red_by_row[y] += 1

    long_red_columns = sum(
        count / max(1, sample_height) >= 0.25
        for count in red_by_column
    )
    long_red_rows = sum(
        count / max(1, sample_width) >= 0.25
        for count in red_by_row
    )
    if long_red_columns >= 1 and long_red_rows >= 1:
        reasons.append("axis_aligned_annotation_or_callout")

    # Multiple strong internal seams in both directions are characteristic of
    # tiled figures. This is a review signal, not a scientific classifier.
    gray = sample.convert("L")
    gray_pixels = gray.load()
    vertical_scores = []
    for x in range(1, sample_width):
        vertical_scores.append(
            sum(
                abs(gray_pixels[x, y] - gray_pixels[x - 1, y])
                for y in range(sample_height)
            ) / max(1, sample_height)
        )
    horizontal_scores = []
    for y in range(1, sample_height):
        horizontal_scores.append(
            sum(
                abs(gray_pixels[x, y] - gray_pixels[x, y - 1])
                for x in range(sample_width)
            ) / max(1, sample_width)
        )

    def separated_internal_peaks(scores, axis_length):
        candidates = []
        margin = max(2, int(axis_length * 0.08))
        for index, score in enumerate(scores, start=1):
            if margin <= index <= axis_length - margin and score >= 24.0:
                if not candidates or index - candidates[-1] >= max(4, axis_length // 20):
                    candidates.append(index)
                elif score > scores[candidates[-1] - 1]:
                    candidates[-1] = index
        return candidates

    vertical_seams = separated_internal_peaks(vertical_scores, sample_width)
    horizontal_seams = separated_internal_peaks(horizontal_scores, sample_height)
    if len(vertical_seams) >= 2 and len(horizontal_seams) >= 1:
        reasons.append("multi_panel_grid_signal")

    hsv = sample.convert("HSV")
    saturation = hsv.getchannel("S")
    mean_saturation = ImageStat.Stat(saturation).mean[0] / 255.0

    # Fluorescence references in the reviewed project data have a predominantly
    # dark field with a small, saturated coloured signal. This remains a
    # conservative modality gate rather than an automatic image classifier:
    # explicit human approval and response-value agreement are still required.
    hsv_pixels = list(hsv.getdata())
    pixel_count = max(1, len(hsv_pixels))
    dark_background_fraction = sum(
        value < 40
        for _hue, _saturation, value in hsv_pixels
    ) / pixel_count
    saturated_signal_fraction = sum(
        saturation_value >= 80 and value >= 45
        for _hue, saturation_value, value in hsv_pixels
    ) / pixel_count
    mean_luminance = ImageStat.Stat(sample.convert("L")).mean[0] / 255.0
    fluorescence_like_supported = bool(
        mean_luminance <= 0.20
        and dark_background_fraction >= 0.55
        and saturated_signal_fraction >= 0.005
    )

    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(aspect_ratio, 4),
        "mean_saturation": round(mean_saturation, 4),
        "mean_luminance": round(mean_luminance, 4),
        "dark_background_fraction": round(dark_background_fraction, 4),
        "saturated_signal_fraction": round(saturated_signal_fraction, 4),
        "color_signal_supported": mean_saturation >= 0.05,
        "fluorescence_like_supported": fluorescence_like_supported,
        "composite_layout_risk": bool(reasons),
        "automated_risk_reasons": reasons,
        "long_red_columns": long_red_columns,
        "long_red_rows": long_red_rows,
        "vertical_seam_count": len(vertical_seams),
        "horizontal_seam_count": len(horizontal_seams),
        "organism_mask_available": False,
    }


def load_policy(path=POLICY_FILE):
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            clean(row.get("record_id")): row
            for row in csv.DictReader(handle)
            if clean(row.get("record_id"))
        }


def reference_policy(record_id, path=POLICY_FILE):
    row = load_policy(path).get(clean(record_id))
    if row is None:
        return {
            "record_id": clean(record_id),
            "review_status": "unreviewed",
            "transform_eligible": False,
            "crop_path": "",
            "crop_box": "",
            "automated_risk_reasons": "",
            "review_notes": (
                "No transform-safety review is recorded for this reference."
            ),
            "display_status": "source_figure",
            "display_crop_path": "",
            "display_crop_box": "",
            "display_panel_label": "",
            "display_notes": "No reviewed record-specific display crop is recorded.",
        }
    result = dict(row)
    result["transform_eligible"] = parse_bool(row.get("transform_eligible"))
    return result


def resolve_display_source(full_reference_path, policy):
    """Return a reviewed record-specific panel crop or the full evidence figure.

    A display crop is independent of transformation approval. It may retain
    labels or scale bars that are scientifically useful for display while still
    being barred from image manipulation.
    """

    status = clean((policy or {}).get("display_status")).lower()
    if status == "verified_panel":
        crop_path = resolve_project_image_path(
            (policy or {}).get("display_crop_path")
        )
        if crop_path is not None and crop_path.is_file():
            return crop_path
    return Path(full_reference_path)


def policy_allows_transform(policy):
    return bool(
        policy
        and policy.get("transform_eligible")
        and clean(policy.get("review_status")).lower()
        in {"verified_whole", "verified_crop"}
    )


def policy_reason(policy):
    status = clean((policy or {}).get("review_status")).lower() or "unreviewed"
    risks = clean((policy or {}).get("automated_risk_reasons"))
    notes = clean((policy or {}).get("review_notes"))
    if status == "context_only":
        base = (
            "The full reference is an evidence figure or annotated/composite "
            "image and is retained for context only."
        )
    elif status == "review_required":
        base = (
            "The reference requires human review before it can be used for a "
            "whole-image transformation."
        )
    elif status == "unreviewed":
        base = (
            "No explicit transform-safety review is recorded for this "
            "reference. Unreviewed images are context-only by default."
        )
    else:
        base = "The reference is not approved for visual transformation."
    details = []
    if risks:
        details.append("Automated risk signal(s): " + risks.replace("|", ", "))
    if notes:
        details.append(notes)
    return " ".join([base] + details)


def resolve_transform_source(full_reference_path, policy):
    """Return the reviewed image/crop path or None when transformation is barred."""

    if not policy_allows_transform(policy):
        return None
    status = clean(policy.get("review_status")).lower()
    if status == "verified_whole":
        return Path(full_reference_path)
    crop_path = resolve_project_image_path(policy.get("crop_path"))
    if crop_path is None or not crop_path.is_file():
        return None

    quality = assess_transform_crop_quality(crop_path)
    policy["runtime_crop_quality"] = quality
    if not quality["passed"]:
        policy["transform_eligible"] = False
        readable = ", ".join(
            reason.replace("_", " ") for reason in quality["reasons"]
        )
        policy["review_notes"] = (
            "The reviewed crop failed the current crop-quality gates "
            f"({readable}). Transformation was withheld pending review."
        )
        return None
    return crop_path


def parse_crop_box(value):
    numbers = [int(item) for item in re.findall(r"-?\d+", clean(value))]
    if len(numbers) != 4:
        return None
    left, top, right, bottom = numbers
    if left < 0 or top < 0 or right <= left or bottom <= top:
        return None
    return left, top, right, bottom
