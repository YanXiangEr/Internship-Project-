# GitHub and Streamlit Community Cloud deployment

This folder is the deployment copy of the Microorganism World Model. It omits
virtual environments, backups, patch installers, payload folders, cached files,
training-only ResNet checkpoints, raw source documents and previously generated
V2 output images.

## Upload to GitHub

Upload everything in this folder, keeping the directory structure unchanged.
The repository root must contain `requirements.txt`, `README.md`, `src`, `data`,
`db` and `models`.

## Deploy with Streamlit Community Cloud

1. Connect Streamlit Community Cloud to the GitHub account that owns the repository.
2. Create a new app and select the repository and branch.
3. Set the main file path to `src/app.py`.
4. Select Python 3.12 to match the environment used to validate this project.
5. Deploy the app.

The application should be run from the repository root. For local testing:

```powershell
python -m pip install -r requirements.txt
streamlit run src/app.py
```

## Files intentionally retained

- `db/microorganism_world_model.db`: live evidence database.
- `models/world_model_v1_state_classifier.pkl`: healthy/stressed classifier.
- `data/world_model_v1_dataset.csv`: V1 visual-reference dataset.
- `data/v2_visual_conditioning_dataset.csv`: V2 conditioning dataset.
- `data/visual_reference_transform_policy.csv`: transformation approvals and blocks.
- `data/visual_transform_candidate_metadata.csv`: reviewed V2 fallback metadata.
- `data/processed`: original evidence images and reviewed display/transform crops.
- `data/external_images`: licensed external species references.
- `data/curated` and source catalogues: evidence and visual provenance.

## Files intentionally omitted

- `.venv`, backups, caches and generated V2 output images.
- Old `apply_*.py` installers and their `*_payload` folders.
- Database and CSV backup copies.
- The three training-only ResNet species checkpoints.
- Raw Excel and Word source documents.
- Training, rebuilding and historical audit scripts that are not imported by the app.

Keep the full local internship project as the archival and development copy. This
repository is the smaller public deployment copy.
