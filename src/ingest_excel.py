import json
import re
from pathlib import Path
import pandas as pd


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip()).strip("_")


def parse_sheet_name(sheet_name: str):
    species, condition = sheet_name.rsplit(",", 1)
    return species.strip(), condition.strip().lower()


def _clean_column_name(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def read_sheet(excel_file, sheet_name):
    """Read one worksheet while preserving its original Excel row numbers.

    Returns:
        observations_df: one row per main-table record
        media_df: optional video records from a Videos section
    """
    raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
    species, condition = parse_sheet_name(sheet_name)

    header_row = None
    videos_row = None

    for i, row in raw.iterrows():
        values = [str(cell).strip() for cell in row if pd.notna(cell)]
        if header_row is None and "Image URL" in values:
            header_row = i
        if "Videos" in values:
            videos_row = i

    if header_row is None:
        raise ValueError(f"Could not find 'Image URL' header in {sheet_name!r}")

    table_end = videos_row if videos_row is not None else len(raw)
    headers = [_clean_column_name(v) for v in raw.iloc[header_row].tolist()]

    observation_records = []
    for pandas_index in range(header_row + 1, table_end):
        row = raw.iloc[pandas_index]
        if row.isna().all():
            continue

        record = {}
        for col_index, value in enumerate(row):
            if col_index >= len(headers):
                continue
            name = headers[col_index]
            if name is None:
                continue
            record[name] = None if pd.isna(value) else value

        # Skip separator/blank-looking rows that survived because of formatting.
        if not any(v is not None and str(v).strip() != "" for v in record.values()):
            continue

        excel_row = pandas_index + 1
        observation_id = f"{slug(species)}_{condition}_row{excel_row}"

        record.update({
            "observation_id": observation_id,
            "species": species,
            "condition": condition,
            "sheet_name": sheet_name,
            "excel_row": excel_row,
        })
        observation_records.append(record)

    media_records = []
    if videos_row is not None:
        video_counter = 0
        for pandas_index in range(videos_row + 1, len(raw)):
            for cell in raw.iloc[pandas_index]:
                if pd.isna(cell):
                    continue
                text = str(cell).strip()
                if not text.startswith("http"):
                    continue

                parts = text.split("\n", 1)
                url = parts[0].strip()
                notes = parts[1].strip() if len(parts) > 1 else None
                video_counter += 1

                media_records.append({
                    "media_id": f"{slug(species)}_{condition}_video{video_counter}",
                    "observation_id": None,
                    "species": species,
                    "condition": condition,
                    "media_type": "video",
                    "url": url,
                    "local_path": None,
                    "notes": notes,
                    "source_type": "excel",
                    "sheet_name": sheet_name,
                    "excel_row": pandas_index + 1,
                })

    return pd.DataFrame(observation_records), pd.DataFrame(media_records)


def read_all_sheets(excel_file):
    workbook = pd.ExcelFile(excel_file)
    observations = []
    media = []

    for sheet_name in workbook.sheet_names:
        obs_df, media_df = read_sheet(excel_file, sheet_name)
        observations.append(obs_df)
        if not media_df.empty:
            media.append(media_df)

    observations_df = pd.concat(observations, ignore_index=True) if observations else pd.DataFrame()
    media_df = pd.concat(media, ignore_index=True) if media else pd.DataFrame()
    return observations_df, media_df


def row_as_json(row, excluded_columns):
    payload = {}
    for key, value in row.items():
        if key in excluded_columns:
            continue
        if pd.isna(value):
            payload[key] = None
        elif hasattr(value, "item"):
            payload[key] = value.item()
        else:
            payload[key] = value
    return json.dumps(payload, ensure_ascii=False, default=str)
