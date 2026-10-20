# Demonstration Tests

Run these through the Streamlit interface after the automated validation suite
passes. Exact evidence wording may change if the database is expanded, but the
interpretation category should remain consistent.

## 1. Direct experimental evidence

- Mode: **Simulate world state**
- Species: *Nitzschia palea*
- Water factor: Salinity
- Value: 17 ‰
- Evidence threshold: 0.9
- Check: growth-rate and cell-mortality evidence are retained, a stressed state
  is supported, and population/time-dependent responses are not converted into
  single-cell morphology.

## 2. Contextual evidence without a health label

- Mode: **Simulate world state**
- Species: *Euglena viridis*
- Water factor: Heavy metal pollution
- Value: present
- Evidence threshold: 0.0
- Check: presence is explained as occurrence evidence, no unsupported
  healthy/stressed state is assigned, and a state-neutral reference is not
  transformed.

## 3. Whole-organism absence

- Mode: **Simulate world state**
- Species: *Oscillatoria limosa*
- Water factor: Arsenic treatment
- Value: present
- Check: `Presence = absent` is shown as **Not observed**, is not reinterpreted
  as mortality, and no condition-specific organism image is generated.

## 4. Response-matched reference with a withheld rule

- Mode: **Simulate world state**
- Species: *Nitzschia palea*
- Water factor: DDT
- Value: present
- Check: the selected analogue shares the predicted responses, but a visual
  rule is withheld if the reference has unresolved composite or annotation
  risk.

## 5. Composite-figure protection

- Mode: **Simulate world state**
- Species: *Nitzschia palea*
- Water factor: Fullerene C60
- Value: present
- Evidence threshold: 0.9
- Check: the six-panel evidence figure remains available for context, applied
  transformation intensity is zero, and no difference map is produced.

## 6. Exact value mismatch

- Mode: **Simulate world state**
- Species: *Nitzschia palea*
- Water factor: Salinity
- Value: 16.9 ‰
- Evidence threshold: 0.9
- Check: the app distinguishes the lack of matching condition-specific evidence
  from evidence that merely falls below the threshold.

## 7. Threshold exclusion

- Choose a case with matching evidence and raise the evidence threshold above
  its score.
- Check: the app states that matching evidence exists but did not meet the
  selected threshold.

## 8. Inverse boundary behaviour

- Mode: **Microorganism response → Water quality**
- Species: *Nitzschia palea*
- Response: Cell mortality fraction
- Value: 1.0 fraction, followed by 1.01 fraction
- Check: 1.0 returns supported salinity conditions while 1.01 does not produce
  a false match.

## Recording a demonstration

For each case, record the predicted responses, condition source, evidence
score, selected reference type, response/environment match scores, visual-rule
decision and whether a transformation was applied. Do not describe an evidence
score or visual-match score as a probability.

