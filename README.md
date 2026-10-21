# Microorganism World Model

An evidence-based prototype that connects water-quality conditions with
observed microorganism responses for *Nitzschia palea*, *Euglena viridis* and
*Oscillatoria limosa*.

The project combines a structured SQLite evidence database, forward and inverse
queries, a one-step world-state simulator, biological-condition integration,
evidence-backed visual-reference selection and conservative V2 illustrative
transformations.

## What the application can do

- **Forward model:** rank microorganism responses supported by supplied water
  conditions.
- **Inverse model:** rank water conditions associated with supplied biological
  responses.
- **World-state simulator:** apply the current evidence model once (`t → t+1`)
  and present the possible next responses, overall condition, evidence
  provenance and visual interpretation.
- **V1 visual reference:** select the closest evidence-backed microscopy
  analogue using species, condition, response agreement and environmental
  similarity.
- **V2 visual illustration:** apply a conservative visual rule only when the
  predicted response has an explicit static-image rule and the chosen image or
  crop passes the transformation safeguards.

## Scientific interpretation

This is a research prototype, not a diagnostic system.

- Evidence scores rank support in the current database. They are not
  probabilities or percentages.
- Associations should not automatically be interpreted as causal effects.
- Healthy/stressed states depend on the provenance and strength of the
  supporting evidence.
- A missing prediction can mean that no value/context matched or that matching
  evidence did not pass the selected threshold.
- V1 images are evidence-backed analogues, not exact future appearances.
- V2 images are conservative illustrations, not scientifically validated exact
  morphology predictions.
- The dataset and species coverage remain limited, so reported evaluation
  results should be interpreted cautiously.

## Installation on Windows

Python 3.11 is recommended for reproducing the current environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the complete regression suite:

```powershell
python validate_complete_world_model.py
```

Launch the application from the project root:

```powershell
streamlit run src/app.py
```

Streamlit normally opens `http://localhost:8501` in a browser.

## Guide to use the app

### What the model is doing

The application does not learn new biological facts by itself. It searches the
evidence records already stored in the project and reports relationships that
match the selected species, water condition, value or concentration, and
evidence threshold.

For example, if the user selects *Nitzschia palea* and an insecticide condition,
the forward model looks for observations in which that species and condition
were recorded together. It returns the supported responses, their original
values, the evidence score, and the observation that supplied each result. The
inverse model starts with a response and searches in the opposite direction.

The world-state simulator runs the same evidence logic once and presents the
result as a possible transition from `t` to `t+1`. This notation means “the next
model step”; it is not automatically one hour, one day or one generation.

The healthy/stressed label is a summary supported by the evidence provenance.
It should not be treated as a diagnosis. When the evidence is only contextual,
the application deliberately leaves the overall biological condition
unassigned.

### How to use the app

1. Choose the forward model, inverse model or world-state simulator.
2. Select one of the supported species.
3. Enter the water factor or biological response exactly as represented in the
   interface. Add a numerical value and unit when the evidence is
   concentration-specific.
4. Choose the evidence threshold. A higher threshold hides weaker contextual
   relationships; it does not make the remaining scores into probabilities.
5. Read the result together with its supporting observation, relationship type
   and evidence contribution.
6. For a simulation, compare the V1 reference, the V2 transformation reference
   and the safety decisions. They can be different records for a valid reason.

### How the visual outputs work

- **V1** selects a microscopy analogue using species, biological state,
  response-value agreement and environmental similarity. It is context, not a
  generated future image.
- **V2** can make a small illustrative change only when a predicted response
  has a rule, the actual transformation reference agrees with that response
  value, and the image passes every applicable safety gate.
- V2 may use a separate reviewed organism crop when the strongest V1 analogue
  is a composite or annotated figure. The app reports both record IDs.
- “No transformed image generated” is often the correct result. It prevents a
  visual claim when the evidence or image is unsuitable.

The six current V2 rule categories are:

| Rule | Stored response trigger | What is changed | Extra requirement |
| --- | --- | --- | --- |
| Structural | frustule breakage/dispersion, valve deformation, deformation | Very small shear and blur | Reviewed safe image or crop |
| Pigmentation | pigmentation change | Mild saturation reduction | Measurable colour signal |
| Chloroplast | chloroplast damage | Conservative saturation reduction | Measurable colour signal |
| Fluorescence attenuation | reduced chlorophyll fluorescence | Small brightness and saturation reduction | Reviewed fluorescence-like crop |
| Chlorophyll signal attenuation | chlorophyll decrease | Small brightness and saturation reduction | Reviewed fluorescence-like crop with value-level agreement |
| Chloroplast signal loss | chloroplast absent | Restrained coloured-signal reduction | Reviewed fluorescence-like crop |

The three fluorescence-dependent rules use a dark-field image heuristic only as an
additional safety gate. It is not a biological fluorescence measurement. The
chloroplast-signal-loss rule does not erase or redraw a chloroplast, and the
attenuation rules do not predict a quantitative intensity or chlorophyll
concentration.

### Reviewing or adding biological evidence

Before adding an observation, check the original paper or dataset and record:

- the exact species and accepted spelling;
- the water factor, treatment value, concentration and unit;
- exposure duration when available;
- the response type and the original response value;
- whether the relationship was experimentally observed, co-observed or only
  contextual;
- the source URL/DOI, figure or table location and a short provenance note;
- whether the observation supports a healthy/stressed label, or neither.

Do not convert “not observed” into mortality, combine different concentrations
into one exact rule, or infer causation from co-observation. Keep alternative
outcomes and ranges rather than choosing one convenient value.

Use the project’s current import script in dry-run mode first. Review excluded,
warning and manual-review rows before applying an import. After an applied
database change, keep the generated backup and run the complete validation
suite.

### Reviewing or adding images

Keep the complete original figure for provenance. If the Word document or
paper uses only one panel, create a separate display crop that matches that
panel. A display crop is not automatically safe for transformation.

A transformation crop requires explicit human review. It should contain one
relevant organism region with padding, exclude panel separators, labels and
scale bars, and remain linked to the complete source figure. Composite figures,
unreviewed images and state-neutral references should remain context-only.

Do not add a new V2 rule merely because an effect sounds visual. A rule should
be considered only when the response is visible in one still image, its
direction/value is represented in the transformation reference, and a small
deterministic operation can illustrate it without inventing structures.

### Checks to run after any biological or visual update

```powershell
python validate_complete_world_model.py
streamlit run src/app.py
```

Then manually test at least one expected match, one value mismatch, one
contextual-evidence case, one organism-absence case, one safe V2 transformation
and one transformation withheld by a visual-safety rule.

## Project structure

```text
src/                         Application and model logic
data/                        Evidence datasets and SQLite database
data/processed/images/       Complete evidence images
data/processed/visual_display_crops/
                             Word-aligned panels used for display
data/processed/visual_transform_crops/
                             Separately reviewed transformation crops
docs/                        Handover and demonstration guidance
validate_*.py                Focused regression suites
validate_complete_world_model.py
                             Consolidated regression runner
```

Display crops and transformation crops have different purposes. A panel may be
suitable for display while still being unsafe to transform. Complete source
figures are retained for provenance.

## Recommended demonstration

Use the cases in `docs/DEMONSTRATION_TESTS.md` or open **About this model** in
the application. Together, the cases demonstrate direct evidence, contextual
evidence, absence, threshold behaviour, reference matching and visual-safety
gates.

## Handover notes

- Do not run historical `apply_*.py` scripts on a current completed copy.
- Do not share virtual environments, caches, temporary outputs or backup
  directories.
- Keep the directory structure unchanged because datasets, images and policies
  are resolved relative to the project root.
- Local SQLite updates are not guaranteed to persist on Streamlit Community
  Cloud. Treat the local project and version-controlled repository as the
  authoritative source.

## Future work

Useful extensions include additional independently reviewed observations,
expert validation of evidence rules, temporal data for growth and behaviour,
organism segmentation masks, calibrated uncertainty and a persistent hosted
database for collaborative editing.

