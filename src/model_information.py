"""Human-readable project, interpretation and demonstration guidance."""

import streamlit as st


DEMONSTRATION_CASES = [
    {
        "Purpose": "Strong direct evidence",
        "Species": "Nitzschia palea",
        "Input": "Salinity = 17 ‰",
        "Expected result": "Growth-rate and mortality evidence; stressed condition",
    },
    {
        "Purpose": "Contextual evidence",
        "Species": "Euglena viridis",
        "Input": "Heavy metal pollution = present",
        "Expected result": "Presence evidence without an unsupported health label",
    },
    {
        "Purpose": "Whole-organism absence",
        "Species": "Oscillatoria limosa",
        "Input": "Arsenic treatment = present",
        "Expected result": "Not observed; no condition-specific organism image",
    },
    {
        "Purpose": "Successful safety-gated V2 transformation",
        "Species": "Nitzschia palea",
        "Input": "Insecticide = present",
        "Expected result": (
            "V1 contextual row-8 analogue; reviewed row-6 crop used by V2 "
            "for chloroplast-damage, frustule-dispersion and reduced-"
            "fluorescence cues"
        ),
    },
    {
        "Purpose": "Composite-reference protection",
        "Species": "Nitzschia palea",
        "Input": "Fullerene C60 = present",
        "Expected result": "Six-panel figure retained without transformation",
    },
    {
        "Purpose": "Exact value mismatch",
        "Species": "Nitzschia palea",
        "Input": "Salinity = 16.9 ‰",
        "Expected result": "No exact condition-specific prediction at a strict threshold",
    },
]


def render_model_information():
    """Render the in-app model card and handover-oriented test guide."""

    st.header("About this model")
    st.write(
        "This application is an evidence-based microorganism world-model "
        "prototype for Nitzschia palea, Euglena viridis and Oscillatoria "
        "limosa. It connects water conditions, observed responses, evidence "
        "provenance and microscopy references."
    )

    overview, interpretation, visual, rules, safety, tests = st.tabs(
        [
            "How it works",
            "How to interpret results",
            "V1 and V2 visuals",
            "V2 visual rules",
            "Visual safety checks",
            "Demonstration cases",
        ]
    )

    with overview:
        st.markdown(
            """
            1. The **forward model** ranks responses associated with supplied
               water conditions.
            2. The **inverse model** ranks water conditions associated with
               supplied organism responses.
            3. The **world-state simulator** applies the evidence model once to
               produce a possible next state (`t → t+1`). This is not a fixed
               amount of biological time.
            4. The condition layer distinguishes direct response evidence,
               observation-condition provenance, contextual support and the
               broader qualitative classifier.
            5. The visual layer selects an evidence-backed reference and only
               applies a V2 rule when both the response and image pass the
               current safeguards.
            """
        )

    with interpretation:
        st.warning(
            "Evidence scores rank support in the current database. They are "
            "not probabilities, diagnostic confidence values or biological "
            "effect sizes."
        )
        st.markdown(
            """
            - Predictions are limited to observations currently stored in the
              database.
            - An association is not automatically a direct causal relationship.
            - A missing prediction means either no matching evidence was found
              or matching evidence did not meet the selected threshold.
            - Healthy/stressed labels depend on the provenance and strength of
              the supporting evidence.
            - The current scope contains three species and a relatively small
              dataset, so results should be interpreted as prototype outputs.
            """
        )

    with visual:
        st.markdown(
            """
            **V1** selects the closest available microscopy analogue using
            species, state, response agreement and environmental similarity.

            **V2** first tries to use that V1 reference. If the V1 image is not
            approved for transformation, V2 may select a separate reviewed
            crop from the same species and biological state. This fallback is
            allowed only when the crop has value-level agreement with at least
            one predicted response that has a static-image rule. The original
            V1 analogue is still reported so the contextual evidence and the
            transformation source remain distinguishable.

            Three reviewed Figure 2 panel records are currently supplied to
            V2 through an approved metadata overlay. The overlay does not
            replace the main conditioning dataset: existing V2 rows remain
            authoritative, and an added row is accepted only when the visual
            safety policy explicitly approves its record.

            State-neutral references, organism-absence results, composite
            figures, unsafe crops and unsupported response types are preserved
            without transformation. A V2 image is an illustration, not a
            scientifically validated prediction of exact future morphology.
            """
        )

        st.info(
            "A stressed label alone never changes an image. V2 requires a "
            "matching response-specific rule and a reference that passes the "
            "safety checks."
        )

    with rules:
        st.write(
            "V2 currently has six deliberately narrow transformation-rule "
            "categories. These are image-processing cues rather than learned "
            "biological image-generation rules."
        )
        st.dataframe(
            [
                {
                    "Rule": "Structural",
                    "Triggering response types": (
                        "Frustule breakage; frustule dispersion; valve "
                        "deformation; deformation"
                    ),
                    "Illustrative operation": (
                        "Very small deterministic affine shear followed by "
                        "a slight blur; it does not draw cracks or fragments"
                    ),
                    "Extra modality requirement": "None",
                },
                {
                    "Rule": "Pigmentation",
                    "Triggering response types": "Pigmentation change",
                    "Illustrative operation": (
                        "Mild colour-saturation reduction; it does not invent "
                        "new pigments or structures"
                    ),
                    "Extra modality requirement": "Measurable colour signal",
                },
                {
                    "Rule": "Chloroplast",
                    "Triggering response types": "Chloroplast damage",
                    "Illustrative operation": (
                        "Conservative colour-saturation reduction; it does not "
                        "create, remove or redraw chloroplast geometry"
                    ),
                    "Extra modality requirement": "Measurable colour signal",
                },
                {
                    "Rule": "Fluorescence attenuation",
                    "Triggering response types": (
                        "Reduced chlorophyll fluorescence"
                    ),
                    "Illustrative operation": (
                        "Small brightness and saturation reduction; it is not "
                        "a quantitative fluorescence prediction"
                    ),
                    "Extra modality requirement": (
                        "Reviewed fluorescence-like crop"
                    ),
                },
                {
                    "Rule": "Chlorophyll signal attenuation",
                    "Triggering response types": "Chlorophyll decrease",
                    "Illustrative operation": (
                        "Small brightness and saturation reduction; it does "
                        "not convert chlorophyll concentration into pixels"
                    ),
                    "Extra modality requirement": (
                        "Reviewed fluorescence-like crop"
                    ),
                },
                {
                    "Rule": "Chloroplast signal loss",
                    "Triggering response types": "Chloroplast absent",
                    "Illustrative operation": (
                        "Restrained reduction of visible coloured signal; it "
                        "does not erase or redraw chloroplast geometry"
                    ),
                    "Extra modality requirement": (
                        "Reviewed fluorescence-like crop"
                    ),
                },
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            """
            **How transformation intensity is determined**

            - Only predicted responses covered by the rules above contribute.
            - The intensity uses their evidence scores and the number of
              transformable responses.
            - It is deliberately capped at **0.62**.
            - If no supported static response remains, the applied intensity is
              **0.0** and the reference pixels are preserved.

            **Responses deliberately not converted into a static image**

            Occurrence, abundance, cell density, mortality, viability, growth,
            motility, avoidance, photosynthesis inhibition, lipid-body change
            and other population-level, temporal, behavioural or non-static
            physiological responses may still influence the biological result
            or reference matching. They do not directly alter one still image.

            A chlorophyll decrease is rendered only when the actual V2
            transformation reference is a reviewed fluorescence-like crop and
            has value-level agreement for that response. Bright-field images,
            state-neutral images and unreviewed figures remain unchanged.
            """
        )

    with safety:
        st.write(
            "A visual rule is applied only after all relevant checks below "
            "pass. Automated signals can block or route an image for review; "
            "they never approve an image by themselves."
        )
        st.markdown(
            """
            1. **Appearance applicability:** if the evidence says the organism
               was absent, no condition-specific organism image is generated.
            2. **State support:** state-neutral references cannot justify a
               healthy- or stressed-specific transformation.
            3. **Explicit human-reviewed policy:** the record must be marked
               `verified_whole` or `verified_crop` and explicitly transform
               eligible. Missing, unreviewed, review-required and context-only
               records are denied by default.
            4. **Source availability:** the source image must exist. A
               `verified_crop` must also resolve to its separately stored,
               reviewed crop file.
            5. **Crop-quality gates:** a reviewed crop must be at least 48 × 48
               pixels, contain a detectable foreground component, retain
               padding around the dominant foreground, and avoid a white panel
               separator along its edges.
            6. **Composite and annotation protection:** whole images are
               withheld when the detector identifies an extreme aspect ratio
               (at least 2.2), axis-aligned red annotation/callout lines, or a
               multi-panel grid pattern. This protects labels, scale bars and
               unrelated panels.
            7. **Colour-modality check:** pigmentation and chloroplast-damage rules
               require a measurable colour signal (mean saturation of at least
               0.05). The fluorescence-attenuation and chloroplast-signal-loss
               rules additionally require a dark-background fluorescence-like
               profile: mean luminance no greater than 0.20, at least 55% dark
               pixels, and at least 0.5% saturated signal pixels. These are
               image-safety heuristics, not biological measurements.
               Structural rules do not require colour.
            8. **Response-value agreement:** a safe fallback reference must
               agree with the value of at least one predicted static response.
               Merely having the same species or stressed label is insufficient.
            9. **Rule scoping:** when V2 uses a different safe reference from
               V1, only the value-matched static responses from that fallback
               may drive the transformation. Other predicted responses remain
               reported but are not rendered through that crop.
            """
        )

        st.warning(
            "The rendering difference map and its changed-pixel measurements "
            "describe image-processing differences only. They are not "
            "biological effect sizes, probabilities, uncertainty maps or "
            "model-confidence scores."
        )

    with tests:
        st.write(
            "These cases exercise different behaviours of the application. "
            "Exact displayed evidence can change if the database is updated."
        )
        st.dataframe(DEMONSTRATION_CASES, use_container_width=True, hide_index=True)
