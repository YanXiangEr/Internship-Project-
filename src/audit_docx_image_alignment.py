"""Audit processed visual references against the Word-alignment manifest."""

from __future__ import annotations

import json

from docx_image_alignment import audit_alignment


def main():
    result = audit_alignment()
    print(json.dumps(result, indent=2))
    if result["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
