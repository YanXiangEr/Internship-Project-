"""Dataset-wide Word-image alignment and crop quality checks.

The Word document remains the provenance source. The alignment manifest stores
the embedded-source checksum and the exact crop rectangle represented by
Word's ``srcRect`` values for every visual record. Automated checks may reject
a transform crop, but never promote an unreviewed image to transform-safe.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import deque
from pathlib import Path
from statistics import median

from PIL import Image, ImageChops


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALIGNMENT_MANIFEST = (
    PROJECT_ROOT / "data" / "visual_reference_docx_alignment_manifest.csv"
)


def clean(value):
    return "" if value is None else str(value).strip()


def parse_box(value):
    try:
        values = tuple(int(part.strip()) for part in clean(value).split(","))
    except ValueError:
        return None
    if len(values) != 4:
        return None
    left, top, right, bottom = values
    if left < 0 or top < 0 or right <= left or bottom <= top:
        return None
    return values


def resolve_path(value, project_root=PROJECT_ROOT):
    candidate = Path(clean(value))
    return candidate if candidate.is_absolute() else Path(project_root) / candidate


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_alignment_manifest(path=ALIGNMENT_MANIFEST):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def crop_matches_source(source_path, crop_path, box):
    box = parse_box(box) if isinstance(box, str) else box
    if box is None:
        return False
    with Image.open(source_path) as source, Image.open(crop_path) as crop:
        expected = source.convert("RGB").crop(box)
        actual = crop.convert("RGB")
        return expected.size == actual.size and ImageChops.difference(
            expected, actual
        ).getbbox() is None


def _near_white(pixel):
    red, green, blue = pixel
    return min(pixel) >= 238 and max(pixel) - min(pixel) <= 14


def _edge_separator_reasons(image):
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    reasons = []

    def column_fraction(x):
        return sum(_near_white(pixels[x, y]) for y in range(height)) / max(1, height)

    def row_fraction(y):
        return sum(_near_white(pixels[x, y]) for x in range(width)) / max(1, width)

    minimum_column_run = max(2, round(width * 0.012))
    minimum_row_run = max(2, round(height * 0.012))

    for side, indexes in (
        ("left", range(width)),
        ("right", range(width - 1, -1, -1)),
    ):
        run = 0
        for x in indexes:
            if column_fraction(x) >= 0.82:
                run += 1
            else:
                break
        if run >= minimum_column_run and run < width * 0.45:
            reasons.append(f"white_separator_at_{side}_edge")

    for side, indexes in (
        ("top", range(height)),
        ("bottom", range(height - 1, -1, -1)),
    ):
        run = 0
        for y in indexes:
            if row_fraction(y) >= 0.82:
                run += 1
            else:
                break
        if run >= minimum_row_run and run < height * 0.45:
            reasons.append(f"white_separator_at_{side}_edge")
    return reasons


def _largest_foreground_component(image):
    sample = image.convert("RGB")
    sample.thumbnail((320, 320))
    width, height = sample.size
    pixels = sample.load()

    border = []
    for x in range(width):
        border.extend((pixels[x, 0], pixels[x, height - 1]))
    for y in range(1, height - 1):
        border.extend((pixels[0, y], pixels[width - 1, y]))
    background = tuple(int(median(channel)) for channel in zip(*border))

    distances = []
    for y in range(height):
        for x in range(width):
            pixel = pixels[x, y]
            distances.append(math.sqrt(sum((pixel[i] - background[i]) ** 2 for i in range(3))))
    sorted_distances = sorted(distances)
    p85 = sorted_distances[min(len(sorted_distances) - 1, int(len(sorted_distances) * 0.85))]
    threshold = max(38.0, min(80.0, p85 * 0.65))
    mask = [distance >= threshold for distance in distances]

    visited = bytearray(width * height)
    largest = None
    for start, active in enumerate(mask):
        if not active or visited[start]:
            continue
        queue = deque([start])
        visited[start] = 1
        count = 0
        min_x = width
        min_y = height
        max_x = -1
        max_y = -1
        while queue:
            position = queue.popleft()
            y, x = divmod(position, width)
            count += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not dx and not dy:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbour = ny * width + nx
                        if mask[neighbour] and not visited[neighbour]:
                            visited[neighbour] = 1
                            queue.append(neighbour)
        candidate = (count, (min_x, min_y, max_x + 1, max_y + 1), (width, height))
        if largest is None or candidate[0] > largest[0]:
            largest = candidate
    return largest


def assess_transform_crop_quality(image_or_path):
    """Reject obvious clipping/separators without approving new crops."""

    if isinstance(image_or_path, (str, Path)):
        with Image.open(image_or_path) as opened:
            image = opened.convert("RGB")
    else:
        image = image_or_path.convert("RGB")

    width, height = image.size
    reasons = []
    if width < 48 or height < 48:
        reasons.append("crop_too_small")
    reasons.extend(_edge_separator_reasons(image))

    largest = _largest_foreground_component(image)
    component_fraction = 0.0
    padding = None
    if largest:
        count, (left, top, right, bottom), (sample_width, sample_height) = largest
        component_fraction = count / max(1, sample_width * sample_height)
        padding = {
            "left": left,
            "top": top,
            "right": sample_width - right,
            "bottom": sample_height - bottom,
        }
        minimum_padding = max(2, round(min(sample_width, sample_height) * 0.012))
        if component_fraction >= 0.004 and min(padding.values()) < minimum_padding:
            reasons.append("dominant_foreground_touches_crop_edge")
    else:
        reasons.append("no_foreground_component_detected")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "width": width,
        "height": height,
        "largest_component_fraction": round(component_fraction, 6),
        "foreground_padding_sample_pixels": padding,
    }


def audit_alignment(project_root=PROJECT_ROOT, manifest_path=ALIGNMENT_MANIFEST):
    rows = load_alignment_manifest(manifest_path)
    issues = []
    source_matches = 0
    display_matches = 0
    transform_passes = 0
    for row in rows:
        record_id = clean(row.get("record_id"))
        source = resolve_path(row.get("source_image_path"), project_root)
        if not source.is_file():
            issues.append(f"{record_id}: source image is missing")
            continue
        if sha256(source) != clean(row.get("source_sha256")):
            issues.append(f"{record_id}: processed source differs from the Word embedded source")
        else:
            source_matches += 1

        display_value = clean(row.get("display_crop_path"))
        if display_value:
            display = resolve_path(display_value, project_root)
            if not display.is_file():
                issues.append(f"{record_id}: Word-aligned display crop is missing")
            elif not crop_matches_source(source, display, row.get("word_crop_box")):
                issues.append(f"{record_id}: display crop pixels do not match the Word crop")
            elif sha256(display) != clean(row.get("display_crop_sha256")):
                issues.append(f"{record_id}: display crop checksum is unexpected")
            else:
                display_matches += 1

        transform_value = clean(row.get("transform_crop_path"))
        if transform_value:
            transform = resolve_path(transform_value, project_root)
            if not transform.is_file():
                issues.append(f"{record_id}: reviewed transform crop is missing")
            elif not crop_matches_source(source, transform, row.get("transform_crop_box")):
                issues.append(f"{record_id}: transform crop pixels do not match its source box")
            else:
                quality = assess_transform_crop_quality(transform)
                if quality["passed"]:
                    transform_passes += 1
                else:
                    issues.append(
                        f"{record_id}: transform crop failed quality gates: "
                        + ", ".join(quality["reasons"])
                    )
    return {
        "records": len(rows),
        "source_matches": source_matches,
        "display_crops_expected": sum(bool(clean(row.get("display_crop_path"))) for row in rows),
        "display_crops_matching": display_matches,
        "transform_crops_expected": sum(bool(clean(row.get("transform_crop_path"))) for row in rows),
        "transform_crops_passing": transform_passes,
        "issues": issues,
    }
