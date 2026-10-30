import re
from pathlib import Path
import pandas as pd
from docx import Document
from docx.oxml.ns import qn

from ingest_excel import slug, parse_sheet_name


def parse_excel_mapping(text):
    """Parse mappings such as 'Nitzschia palea, healthy; 6'."""
    text = str(text).strip()
    text = re.sub(r"^[A-Za-z0-9\-]+\s*:\s*", "", text)
    text = re.sub(r"\bNitschia\b", "Nitzschia", text, flags=re.IGNORECASE)

    match = re.search(
        r"(.+?,\s*(?:healthy|stressed))\s*;\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip(), int(match.group(2))


def read_word_images(word_file, output_directory):
    """Extract embedded images and map them to spreadsheet observations."""
    document = Document(word_file)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    records = []

    for table_number, table in enumerate(document.tables, start=1):
        if not table.rows:
            continue

        for word_row_number, row in enumerate(table.rows[1:], start=2):
            if len(row.cells) < 3:
                continue

            image_url = row.cells[0].text.strip() or None
            mapping_text = row.cells[2].text.strip()
            if not mapping_text:
                continue

            parsed = parse_excel_mapping(mapping_text)
            if parsed is None:
                print(f"Warning: could not parse Word mapping: {mapping_text!r}")
                continue

            sheet_name, excel_row = parsed
            species, condition = parse_sheet_name(sheet_name)
            observation_id = f"{slug(species)}_{condition}_row{excel_row}"

            blips = row.cells[1]._tc.xpath(".//a:blip")
            for image_number, blip in enumerate(blips, start=1):
                relationship_id = blip.get(qn("r:embed"))
                if not relationship_id:
                    continue

                relationship = document.part.rels[relationship_id]
                image_part = relationship.target_part
                extension = Path(str(image_part.partname)).suffix or ".bin"
                filename = f"{observation_id}_image{image_number}{extension}"
                filepath = output_directory / filename
                filepath.write_bytes(image_part.blob)

                records.append({
                    "media_id": f"{observation_id}_image{image_number}",
                    "observation_id": observation_id,
                    "species": species,
                    "condition": condition,
                    "media_type": "image",
                    "url": image_url,
                    "local_path": str(filepath),
                    "notes": None,
                    "source_type": "word",
                    "sheet_name": sheet_name,
                    "excel_row": excel_row,
                    "word_table": table_number,
                    "word_table_row": word_row_number,
                })

    return pd.DataFrame(records)
