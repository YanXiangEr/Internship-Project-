from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGE_DIR = PROCESSED_DIR / "images"
DB_DIR = PROJECT_ROOT / "db"

EXCEL_FILE = RAW_DIR / "phytoplankton_dataset.xlsx"
WORD_FILE = RAW_DIR / "phytoplankton_image_dataset.docx"
DATABASE_FILE = DB_DIR / "microorganism_world_model.db"
