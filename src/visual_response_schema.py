"""
visual_response_schema.py

Defines how microorganism response types can be used by future
computer-vision, video-analysis, and visual-simulation components.

IMPORTANT:

These classifications describe modelling roles.

They do NOT claim that every response associated with an image is
directly visible in that image.

The schema is intentionally kept outside the SQLite database for now
so that classifications can be reviewed and changed safely.
"""


# ============================================================
# VISUAL ROLE CONSTANTS
# ============================================================

STATIC_VISUAL = "static_visual"

TEMPORAL_VISUAL = "temporal_visual"

INDIRECT_VISUAL = "indirect_visual"

NON_VISUAL = "non_visual"

UNKNOWN = "unknown"


# ============================================================
# RESPONSE SCHEMA
# ============================================================

VISUAL_RESPONSE_SCHEMA = {

    # --------------------------------------------------------
    # MORPHOMETRY
    # --------------------------------------------------------

    "cell_length": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Cell dimensions may be estimated from calibrated "
            "microscopy imagery when scale information is available."
        ),
    },

    "cell_width": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Cell dimensions may be estimated from calibrated "
            "microscopy imagery when scale information is available."
        ),
    },

    "trichome_width": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Trichome width is a morphological measurement, but "
            "quantitative prediction requires image calibration."
        ),
    },


    # --------------------------------------------------------
    # GENERAL MORPHOLOGY
    # --------------------------------------------------------

    "cell_shape": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Cell shape is directly relevant to visual morphology."
        ),
    },

    "posterior_morphology": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Posterior morphology describes visible organism shape."
        ),
    },

    "shape_variability": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "May be inferred from multiple organisms or alternative "
            "morphotypes rather than necessarily from one isolated cell."
        ),
    },

    "trichome_shape": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Trichome shape is suitable for morphological modelling."
        ),
    },

    "sheath_present": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Sheath presence is a structural morphological feature."
        ),
    },

    "sheath_absent": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Sheath absence may be used as a morphological feature "
            "when imaging conditions permit reliable identification."
        ),
    },

    "calyptra_absent": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Structural morphology candidate."
        ),
    },

    "crosswall_constriction": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Structural morphology candidate."
        ),
    },

    "frustule_intact": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Frustule structural condition may be represented visually."
        ),
    },

    "frustule_breakage": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Frustule breakage is a structural visual response."
        ),
    },

    "frustule_dispersion": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Spatial/structural state that may be represented visually."
        ),
    },

    "valve_deformation": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Valve deformation is a morphological abnormality."
        ),
    },

    "raphe_deviation": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Fine-scale morphological feature; suitability depends "
            "on image resolution and microscopy method."
        ),
    },

    "striae_abnormality": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Fine-scale structural feature requiring sufficient "
            "microscopy resolution."
        ),
    },

    "fibulae_abnormality": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Fine-scale structural feature requiring suitable imaging."
        ),
    },

    "fibulae_interruption": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Fine-scale structural feature requiring suitable imaging."
        ),
    },

    "morphological_effect": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "General morphology label. Interpretation should retain "
            "the original evidence context."
        ),
    },


    # --------------------------------------------------------
    # MOTILITY STRUCTURE
    # --------------------------------------------------------

    "flagellum_length_relative_to_body": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Potentially visible structural feature when the flagellum "
            "is resolved by the microscopy method."
        ),
    },


    # --------------------------------------------------------
    # TEMPORAL / BEHAVIOURAL RESPONSES
    # --------------------------------------------------------

    "gliding_motility": {
        "visual_role": TEMPORAL_VISUAL,
        "image_input": False,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Movement requires temporal information and should not "
            "normally be inferred from one static image."
        ),
    },

    "oscillatory_motility": {
        "visual_role": TEMPORAL_VISUAL,
        "image_input": False,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Oscillatory movement requires temporal observations."
        ),
    },

    "avoidance_behavior": {
        "visual_role": TEMPORAL_VISUAL,
        "image_input": False,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Behaviour requires observation across time."
        ),
    },

    "metaboly": {
        "visual_role": TEMPORAL_VISUAL,
        "image_input": False,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Dynamic deformation is best represented using temporal "
            "image sequences or video."
        ),
    },


    # --------------------------------------------------------
    # PIGMENT / OPTICAL FEATURES
    # --------------------------------------------------------

    "pigmentation_change": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Potential visual colour/intensity feature, but predictions "
            "must account for microscopy and illumination conditions."
        ),
    },

    "chloroplast_present": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Potentially observable structural feature depending "
            "on microscopy method."
        ),
    },

    "chloroplast_absent": {
        "visual_role": STATIC_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Potentially observable depending on microscopy method."
        ),
    },

    "chloroplast_damage": {
        "visual_role": INDIRECT_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "May have visual manifestations, but direct identification "
            "depends strongly on imaging modality and evidence."
        ),
    },

    "reduced_chlorophyll_fluorescence": {
        "visual_role": INDIRECT_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Potentially observable using appropriate fluorescence "
            "imaging rather than arbitrary microscopy images."
        ),
    },


    # --------------------------------------------------------
    # SURFACE INTERACTIONS
    # --------------------------------------------------------

    "surface_adsorption": {
        "visual_role": INDIRECT_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "May be visually represented when the microscopy method "
            "resolves the surface interaction."
        ),
    },

    "pore_percolation": {
        "visual_role": INDIRECT_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Fine-scale surface interaction; image interpretation "
            "depends on microscopy modality and resolution."
        ),
    },

    "eps_present": {
        "visual_role": INDIRECT_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Potentially visible depending on staining and microscopy."
        ),
    },

    "eps_interaction": {
        "visual_role": INDIRECT_VISUAL,
        "image_input": True,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Interaction may have visual evidence but should not "
            "automatically be treated as directly observable."
        ),
    },


    # --------------------------------------------------------
    # PHYSIOLOGY / METABOLISM / POPULATION RESPONSES
    # --------------------------------------------------------

    "cell_death": {
        "visual_role": INDIRECT_VISUAL,
        "image_input": False,
        "video_input": False,
        "animation_output": True,
        "notes": (
            "Cell death can be represented in simulation, but should "
            "not be inferred from morphology alone without validated "
            "visual indicators or an appropriate assay."
        ),
    },

    "reduced_viability": {
        "visual_role": NON_VISUAL,
        "image_input": False,
        "video_input": False,
        "animation_output": True,
        "notes": (
            "Viability is a biological state and is not assumed to be "
            "directly identifiable from an ordinary microscopy image."
        ),
    },

    "oil_production": {
        "visual_role": INDIRECT_VISUAL,
        "image_input": False,
        "video_input": False,
        "animation_output": True,
        "notes": (
            "Metabolic response. Visual inference requires evidence "
            "for a validated observable marker."
        ),
    },

    "lipid_body_change": {
        "visual_role": INDIRECT_VISUAL,
        "image_input": False,
        "video_input": False,
        "animation_output": True,
        "notes": (
            "May have visual manifestations under appropriate imaging "
            "but is not assumed detectable from arbitrary images."
        ),
    },

    "photosynthesis_inhibition": {
        "visual_role": NON_VISUAL,
        "image_input": False,
        "video_input": False,
        "animation_output": True,
        "notes": (
            "Physiological response rather than a direct morphological "
            "feature."
        ),
    },

    "chlorophyll_decrease": {
        "visual_role": INDIRECT_VISUAL,
        "image_input": False,
        "video_input": False,
        "animation_output": True,
        "notes": (
            "May affect optical properties, but quantitative inference "
            "requires appropriate imaging or measurement."
        ),
    },

    "chlorophyll_increase": {
        "visual_role": INDIRECT_VISUAL,
        "image_input": False,
        "video_input": False,
        "animation_output": True,
        "notes": (
            "May affect optical properties, but quantitative inference "
            "requires appropriate imaging or measurement."
        ),
    },

    "growth_increase": {
        "visual_role": INDIRECT_VISUAL,
        "image_input": False,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Growth generally requires comparison over time or "
            "population-level measurements."
        ),
    },

    "growth_reduction": {
        "visual_role": INDIRECT_VISUAL,
        "image_input": False,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Growth generally requires temporal or population-level "
            "evidence."
        ),
    },

    "population_reduction": {
        "visual_role": INDIRECT_VISUAL,
        "image_input": False,
        "video_input": True,
        "animation_output": True,
        "notes": (
            "Population change requires sampling/counting context and "
            "should not be inferred from an isolated organism image."
        ),
    },
}


# ============================================================
# LOOKUP FUNCTIONS
# ============================================================

def get_visual_schema(response_type):

    return VISUAL_RESPONSE_SCHEMA.get(
        response_type,
        {
            "visual_role": UNKNOWN,
            "image_input": False,
            "video_input": False,
            "animation_output": False,
            "notes": (
                "Response type has not yet been manually "
                "classified."
            ),
        }
    )


def get_visual_role(response_type):

    return get_visual_schema(
        response_type
    )["visual_role"]


def supports_image_input(response_type):

    return get_visual_schema(
        response_type
    )["image_input"]


def supports_video_input(response_type):

    return get_visual_schema(
        response_type
    )["video_input"]


def supports_animation_output(response_type):

    return get_visual_schema(
        response_type
    )["animation_output"]


# ============================================================
# SUMMARY
# ============================================================

def print_visual_schema_summary():

    role_counts = {}

    for response_type, config in (
        VISUAL_RESPONSE_SCHEMA.items()
    ):

        role = config[
            "visual_role"
        ]

        role_counts[
            role
        ] = (
            role_counts.get(
                role,
                0
            )
            + 1
        )

    print(
        "=" * 70
    )

    print(
        "VISUAL RESPONSE SCHEMA"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Classified response types: "
        f"{len(VISUAL_RESPONSE_SCHEMA)}"
    )

    print()

    print(
        "Visual roles:"
    )

    for role, count in sorted(
        role_counts.items()
    ):

        print(
            f"  {role:20} {count}"
        )

    print()

    image_count = sum(
        1
        for config
        in VISUAL_RESPONSE_SCHEMA.values()
        if config[
            "image_input"
        ]
    )

    video_count = sum(
        1
        for config
        in VISUAL_RESPONSE_SCHEMA.values()
        if config[
            "video_input"
        ]
    )

    animation_count = sum(
        1
        for config
        in VISUAL_RESPONSE_SCHEMA.values()
        if config[
            "animation_output"
        ]
    )

    print(
        f"Image-input candidates:     "
        f"{image_count}"
    )

    print(
        f"Video-input candidates:     "
        f"{video_count}"
    )

    print(
        f"Animation-output candidates:"
        f" {animation_count}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print_visual_schema_summary()