PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS organisms (
    organism_id INTEGER PRIMARY KEY AUTOINCREMENT,
    scientific_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    organism_id INTEGER NOT NULL,
    condition TEXT NOT NULL,
    sheet_name TEXT NOT NULL,
    excel_row INTEGER NOT NULL,
    source_url TEXT,
    figure_no TEXT,
    description TEXT,
    image_type TEXT,
    magnification TEXT,
    raw_json TEXT,
    FOREIGN KEY (organism_id) REFERENCES organisms(organism_id),
    UNIQUE(sheet_name, excel_row)
);

CREATE TABLE IF NOT EXISTS media (
    media_id TEXT PRIMARY KEY,
    observation_id TEXT,
    organism_id INTEGER NOT NULL,
    condition TEXT NOT NULL,
    media_type TEXT NOT NULL,
    url TEXT,
    local_path TEXT,
    notes TEXT,
    source_type TEXT NOT NULL,
    sheet_name TEXT,
    excel_row INTEGER,
    FOREIGN KEY (observation_id) REFERENCES observations(observation_id),
    FOREIGN KEY (organism_id) REFERENCES organisms(organism_id)
);

CREATE TABLE IF NOT EXISTS water_quality_factors (
    factor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id TEXT NOT NULL,
    case_id TEXT,
    factor_name TEXT NOT NULL,
    factor_category TEXT,
    value_text TEXT,
    value_numeric REAL,
    unit TEXT,
    qualifier TEXT,
    evidence_text TEXT,
    extraction_method TEXT NOT NULL,
    confidence REAL,
    FOREIGN KEY (observation_id)
        REFERENCES observations(observation_id)
        ON DELETE CASCADE,
    FOREIGN KEY (case_id)
        REFERENCES cases(case_id)
        ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    observation_id TEXT NOT NULL,
    case_label TEXT,
    case_type TEXT,
    treatment_name TEXT,
    concentration_min REAL,
    concentration_max REAL,
    concentration_unit TEXT,
    duration_numeric REAL,
    duration_unit TEXT,
    panel_label TEXT,
    morphotype TEXT,
    notes TEXT,
    confidence REAL DEFAULT 1.0,
    FOREIGN KEY (observation_id)
        REFERENCES observations(observation_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS organism_responses (
    response_id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id TEXT NOT NULL,
    case_id TEXT,
    response_type TEXT NOT NULL,
    response_category TEXT,
    value_text TEXT,
    value_numeric REAL,
    value_min REAL,
    value_max REAL,
    unit TEXT,
    qualifier TEXT,
    evidence_text TEXT,
    extraction_method TEXT NOT NULL,
    confidence REAL,
    FOREIGN KEY (observation_id)
        REFERENCES observations(observation_id)
        ON DELETE CASCADE,
    FOREIGN KEY (case_id)
        REFERENCES cases(case_id)
        ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS evidence_rules (
    rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
    organism_id INTEGER NOT NULL,
    observation_id TEXT NOT NULL,
    case_id TEXT,
    factor_id INTEGER,
    response_id INTEGER,
    relationship_type TEXT NOT NULL,
    evidence_strength REAL,
    direction TEXT DEFAULT 'water_to_response',
    notes TEXT,
    source_method TEXT NOT NULL,
    FOREIGN KEY (organism_id)
        REFERENCES organisms(organism_id),
    FOREIGN KEY (observation_id)
        REFERENCES observations(observation_id)
        ON DELETE CASCADE,
    FOREIGN KEY (case_id)
        REFERENCES cases(case_id)
        ON DELETE SET NULL,
    FOREIGN KEY (factor_id)
        REFERENCES water_quality_factors(factor_id)
        ON DELETE CASCADE,
    FOREIGN KEY (response_id)
        REFERENCES organism_responses(response_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_observations_organism ON observations(organism_id);
CREATE INDEX IF NOT EXISTS idx_media_observation ON media(observation_id);
CREATE INDEX IF NOT EXISTS idx_water_observation ON water_quality_factors(observation_id);
CREATE INDEX IF NOT EXISTS idx_response_observation ON organism_responses(observation_id);
