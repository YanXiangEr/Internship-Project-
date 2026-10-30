import re


# ============================================================
# RESPONSE ONTOLOGY
# ============================================================

RESPONSE_CATEGORIES = {

    # Overall state
    "cell_death": "viability",
    "reduced_viability": "viability",
    "growth_reduction": "growth",
    "growth_increase": "growth",
    "population_reduction": "growth",

    # Photosynthesis / pigments
    "chloroplast_damage": "photosynthesis",
    "chloroplast_absent": "photosynthesis",
    "chloroplast_present": "photosynthesis",
    "reduced_chlorophyll_fluorescence": "photosynthesis",
    "chlorophyll_increase": "photosynthesis",
    "chlorophyll_decrease": "photosynthesis",
    "photosynthesis_inhibition": "photosynthesis",

    # Diatom morphology
    "valve_deformation": "morphology",
    "striae_abnormality": "morphology",
    "fibulae_abnormality": "morphology",
    "fibulae_interruption": "morphology",
    "raphe_deviation": "morphology",
    "frustule_breakage": "morphology",
    "frustule_intact": "morphology",
    "frustule_dispersion": "morphology",

    # EPS / surfaces
    "eps_present": "extracellular_matrix",
    "eps_interaction": "extracellular_matrix",
    "surface_adsorption": "surface_interaction",

    # Euglena morphology
    "cell_shape": "morphology",
    "shape_variability": "morphology",
    "posterior_morphology": "morphology",
    "flagellum_length_relative_to_body": "motility_structure",
    "metaboly": "motility_behavior",
    "avoidance_behavior": "motility_behavior",

    # Oscillatoria morphology
    "sheath_present": "morphology",
    "sheath_absent": "morphology",
    "crosswall_constriction": "morphology",
    "calyptra_present": "morphology",
    "calyptra_absent": "morphology",
    "trichome_shape": "morphology",
    "gliding_motility": "motility_behavior",
    "oscillatory_motility": "motility_behavior",

    # Measurements
    "cell_length": "morphometry",
    "cell_width": "morphometry",
    "trichome_width": "morphometry",
    "trichome_length": "morphometry",
    "striae_density": "morphometry",
    "fibulae_density": "morphometry",

    # Other
    "lipid_body_change": "metabolism",
    "oil_production": "metabolism",
    "pigmentation_change": "pigmentation",
    "pore_percolation": "surface_interaction",
    "morphological_effect": "morphology",
}


# ============================================================
# HELPER
# ============================================================

def make_response(
    observation_id,
    response_type,
    value_text=None,
    value_numeric=None,
    value_min=None,
    value_max=None,
    unit=None,
    qualifier=None,
    evidence_text=None,
    confidence=1.0
):
    """
    Create one normalized microorganism-response record.
    """

    return {
        "observation_id": observation_id,

        "response_type": response_type,

        "response_category": RESPONSE_CATEGORIES.get(
            response_type,
            "other"
        ),

        "value_text": value_text,
        "value_numeric": value_numeric,

        "value_min": value_min,
        "value_max": value_max,

        "unit": unit,

        "qualifier": qualifier,

        "evidence_text": evidence_text,

        "extraction_method": "rule_based",

        "confidence": confidence
    }


# ============================================================
# TEXT CLEANING
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text)

    # Normalize common micro-symbol variants
    text = text.replace("µ", "u")
    text = text.replace("μ", "u")

    # Normalize different dash characters
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    return text.strip()


# ============================================================
# GENERIC MORPHOMETRIC EXTRACTION
# ============================================================

def extract_cell_dimensions(
    observation_id,
    text,
    responses
):
    """
    Extract explicit cell length/width statements.

    Examples:
        55-68 um in length
        11-19 um in width
        41.6 +/- 4.2 um cell length
    """

    lower = text.lower()

    # --------------------------------------------------------
    # Ranges:
    # "55-68 um in length"
    # "42-48 um long"
    # --------------------------------------------------------

    length_patterns = [

        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*u?m\s+"
        r"(?:in\s+)?(?:cell\s+)?(?:length|long)",

        r"(?:cell[s]?\s+)?"
        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*u?m\s+long"
    ]

    for pattern in length_patterns:

        for match in re.finditer(
            pattern,
            lower
        ):

            responses.append(
                make_response(
                    observation_id,
                    "cell_length",
                    value_min=float(match.group(1)),
                    value_max=float(match.group(2)),
                    unit="um",
                    evidence_text=text
                )
            )

    width_patterns = [

        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*u?m\s+"
        r"(?:in\s+)?(?:cell\s+)?(?:width|wide)",

        r"(?:cell[s]?\s+)?"
        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*u?m\s+wide"
    ]

    for pattern in width_patterns:

        for match in re.finditer(
            pattern,
            lower
        ):

            responses.append(
                make_response(
                    observation_id,
                    "cell_width",
                    value_min=float(match.group(1)),
                    value_max=float(match.group(2)),
                    unit="um",
                    evidence_text=text
                )
            )

    # --------------------------------------------------------
    # Mean +/- SD
    #
    # "41.6 +/- 4.2 um cell length"
    # --------------------------------------------------------

    mean_pattern = (
        r"(\d+(?:\.\d+)?)\s*"
        r"\+/-\s*"
        r"(\d+(?:\.\d+)?)\s*u?m\s*"
        r"(cell\s+)?(length|width)"
    )

    for match in re.finditer(
        mean_pattern,
        lower
    ):

        mean_value = float(
            match.group(1)
        )

        metric = match.group(4)

        response_type = (
            "cell_length"
            if metric == "length"
            else "cell_width"
        )

        responses.append(
            make_response(
                observation_id,
                response_type,
                value_numeric=mean_value,
                unit="um",
                qualifier=(
                    f"plus_minus_{match.group(2)}"
                ),
                evidence_text=text
            )
        )


# ============================================================
# NITZSCHIA PALEA
# ============================================================

def normalize_nitzschia(
    observation_id,
    text,
    responses
):

    lower = text.lower()

    # --------------------------------------------------------
    # Chloroplast
    # --------------------------------------------------------

    if (
        "burst chloroplast" in lower
        or "degraded the chloroplast" in lower
        or "chloroplast degradation" in lower
        or "damage of the chloroplast" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "chloroplast_damage",
                value_text="present",
                evidence_text=text
            )
        )

    if (
        "no chloroplast present" in lower
        or "no chloroplast" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "chloroplast_absent",
                value_text="present",
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Fluorescence / chlorophyll
    # --------------------------------------------------------

    if (
        "lower fluorescent cell intensity" in lower
        or "lower fluorescence" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "reduced_chlorophyll_fluorescence",
                value_text="present",
                evidence_text=text
            )
        )

    if (
        "lowering of chlorophyll content" in lower
        or "decrease in chlorophyll" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "chlorophyll_decrease",
                value_text="present",
                evidence_text=text
            )
        )

    if (
        "chlorophyll a content was significantly higher"
        in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "chlorophyll_increase",
                value_text="present",
                qualifier="significant",
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Photosynthesis
    # --------------------------------------------------------

    if (
        "inhibited photosynthesis" in lower
        or "inhibitory effect on photosynthesis" in lower
        or "inhibitory effects on photosynthesis" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "photosynthesis_inhibition",
                value_text="present",
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Cell death / viability
    # --------------------------------------------------------

    if (
        "dead nitzschia" in lower
        or "dead cell" in lower
        or "cell death" in lower
        or "cells started dying" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "cell_death",
                value_text="present",
                evidence_text=text
            )
        )

    if "97.6% cell death" in lower:

        responses.append(
            make_response(
                observation_id,
                "cell_death",
                value_numeric=97.6,
                unit="percent",
                qualifier="reported_at_high_exposure",
                evidence_text=text
            )
        )

    if "20.8% in amounts" in lower:

        responses.append(
            make_response(
                observation_id,
                "cell_death",
                value_numeric=20.8,
                unit="percent",
                qualifier="reported_at_lower_exposure",
                evidence_text=text
            )
        )

    if "loss in cell viability" in lower:

        responses.append(
            make_response(
                observation_id,
                "reduced_viability",
                value_text="present",
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Growth
    # --------------------------------------------------------

    if (
        "diminished n. palea growth" in lower
        or "diminishing growth" in lower
        or "inhibitory effect on" in lower
        and "growth" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "growth_reduction",
                value_text="present",
                evidence_text=text
            )
        )

    if (
        "population dropped substantially" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "population_reduction",
                value_text="present",
                evidence_text=text
            )
        )

    if (
        "cell density increased nearly 10 times"
        in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "growth_increase",
                value_numeric=10.0,
                unit="fold",
                qualifier="approximately",
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Valve deformation
    # --------------------------------------------------------

    if (
        "abnormal valve shape" in lower
        or "valve deform" in lower
        or "percentage deformities" in lower
        or "deformities were recorded" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "valve_deformation",
                value_text="present",
                evidence_text=text
            )
        )

    # Percentage deformity ranges
    deformity_ranges = re.findall(
        r"(\d+(?:\.\d+)?)\s*-\s*"
        r"(\d+(?:\.\d+)?)\s*%\s*"
        r"deformit",
        lower
    )

    for low_value, high_value in deformity_ranges:

        responses.append(
            make_response(
                observation_id,
                "valve_deformation",
                value_min=float(low_value),
                value_max=float(high_value),
                unit="percent",
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Striae / fibulae / raphe
    # --------------------------------------------------------

    if "irregular striae" in lower:

        responses.append(
            make_response(
                observation_id,
                "striae_abnormality",
                value_text="irregular",
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Striae density
    # --------------------------------------------------------
    # Examples supported:
    #   "35-36 striae per 10 um"
    #   "35 - 36 striae in 10 um"
    #   "35-36 striae per 10 um length of frustules"
    striae_density_patterns = [
        (
            r"(\d+(?:\.\d+)?)\s*-\s*"
            r"(\d+(?:\.\d+)?)\s*"
            r"striae\s+(?:per|in)\s+10\s*u?m"
        )
    ]

    for pattern in striae_density_patterns:

        for match in re.finditer(
            pattern,
            lower
        ):

            responses.append(
                make_response(
                    observation_id,
                    "striae_density",
                    value_min=float(match.group(1)),
                    value_max=float(match.group(2)),
                    unit="per_10_um",
                    evidence_text=text
                )
            )

    if "irregular striae and fibulae" in lower:

        responses.append(
            make_response(
                observation_id,
                "fibulae_abnormality",
                value_text="irregular",
                evidence_text=text
            )
        )

    if "fibulae interruption" in lower:

        responses.append(
            make_response(
                observation_id,
                "fibulae_interruption",
                value_text="present",
                evidence_text=text
            )
        )

    if "deviation of the raphe" in lower:

        qualifier = None

        if (
            "slight and severe deviation"
            in lower
        ):
            qualifier = "slight_to_severe"

        responses.append(
            make_response(
                observation_id,
                "raphe_deviation",
                value_text="present",
                qualifier=qualifier,
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Frustule
    # --------------------------------------------------------

    if (
        "breakage of cell walls" in lower
        or "frustule breakage" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "frustule_breakage",
                value_text="present",
                evidence_text=text
            )
        )

    if (
        "frustule intact" in lower
        or "intact frustule" in lower
        or "only left the cell frustule intact" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "frustule_intact",
                value_text="present",
                evidence_text=text
            )
        )

    if "frustule dispersion" in lower:

        responses.append(
            make_response(
                observation_id,
                "frustule_dispersion",
                value_text="present",
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # EPS / particle interaction
    # --------------------------------------------------------

    if (
        "eps network" in lower
        or "extracellular polymeric substances" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "eps_present",
                value_text="present",
                evidence_text=text
            )
        )

    if (
        "trapped into the eps" in lower
        or "captured in the eps" in lower
        or "accumulated in the eps" in lower
        or "affinity for the eps" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "eps_interaction",
                value_text="present",
                evidence_text=text
            )
        )

    if (
        "absorbed on the cell surface" in lower
        or "adsorbed on the cell surface" in lower
        or "on the cell walls" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "surface_adsorption",
                value_text="present",
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Pigmentation
    # --------------------------------------------------------

    if (
        "increasingly dark pigmentation"
        in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "pigmentation_change",
                value_text="darker",
                qualifier="increased_with_exposure",
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Oil / lipid production
    # --------------------------------------------------------

    if "increase in oil production" in lower:

        responses.append(
            make_response(
                observation_id,
                "oil_production",
                value_text="increased",
                evidence_text=text
            )
        )

    if (
        "decrease or no change in the number of lipid bodies"
        in lower
        or "decrease or no change in the number of lipid"
        in lower
        or "drecrease or no change in the number of lipid bodies"
        in lower
        or "drecrease or no change in the number of lipid"
        in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "lipid_body_change",
                value_text="decrease_or_no_change",
                evidence_text=text
            )
        )


# ============================================================
# EUGLENA VIRIDIS
# ============================================================

def normalize_euglena(
    observation_id,
    text,
    responses
):

    lower = text.lower()

    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    shape_terms = []

    possible_shapes = [
        "spindle",
        "fusiform",
        "broadly fusiform",
        "cylindrically fusiform",
        "cylindrical"
    ]

    for shape in possible_shapes:

        if shape in lower:
            shape_terms.append(shape)

    if shape_terms:

        responses.append(
            make_response(
                observation_id,
                "cell_shape",
                value_text="; ".join(
                    sorted(set(shape_terms))
                ),
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Morphotypes / variability
    # --------------------------------------------------------

    if "two morphotypes" in lower:

        responses.append(
            make_response(
                observation_id,
                "shape_variability",
                value_text="multiple_morphotypes",
                value_numeric=2,
                unit="morphotypes",
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Posterior morphology
    # --------------------------------------------------------

    posterior_descriptions = []

    if "sharply or bluntly pointed" in lower:
        posterior_descriptions.append(
            "sharply_or_bluntly_pointed"
        )

    if "posterior of cells were elongated to round" in lower:
        posterior_descriptions.append(
            "elongated_to_round"
        )

    if "tapering tail-piece" in lower:
        posterior_descriptions.append(
            "tapering_tail_piece"
        )

    if "pointed tail" in lower:
        posterior_descriptions.append(
            "pointed_tail"
        )

    if posterior_descriptions:

        responses.append(
            make_response(
                observation_id,
                "posterior_morphology",
                value_text="; ".join(
                    posterior_descriptions
                ),
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Flagellum
    # --------------------------------------------------------

    if (
        "flagellar length was about twice"
        in lower
        or "flagellum was about twice" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "flagellum_length_relative_to_body",
                value_numeric=2.0,
                unit="body_lengths",
                qualifier="approximately",
                evidence_text=text
            )
        )

    if (
        "flagellum was similar to the body"
        in lower
        or "flagellum was similar to body"
        in lower
        or "emergent flagellum was similar to the body"
        in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "flagellum_length_relative_to_body",
                value_text="similar_or_slightly_shorter",
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Metaboly
    # --------------------------------------------------------

    if "metaboly" in lower:

        responses.append(
            make_response(
                observation_id,
                "metaboly",
                value_text="observed",
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Avoidance / movement response
    # --------------------------------------------------------

    if (
        "moved away from the repelling agent"
        in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "avoidance_behavior",
                value_text="present",
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Chloroplast
    # --------------------------------------------------------

    if (
        "single, star-shaped chloroplast"
        in lower
        or "one stellate chloroplast"
        in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "chloroplast_present",
                value_text="single_stellate",
                evidence_text=text
            )
        )


# ============================================================
# OSCILLATORIA LIMOSA
# ============================================================

def normalize_oscillatoria(
    observation_id,
    text,
    responses
):

    lower = text.lower()

    # --------------------------------------------------------
    # Sheath
    # --------------------------------------------------------

    if (
        "sheath was absent" in lower
        or "without mucous sheath" in lower
        or "without mucilaginous sheath" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "sheath_absent",
                value_text="present",
                evidence_text=text
            )
        )

    elif (
        "covered by colourless sheath" in lower
        or "covered by colorless sheath" in lower
        or "colourless sheaths" in lower
        or "colorless sheaths" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "sheath_present",
                value_text="present",
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Cross-wall constriction
    # --------------------------------------------------------

    if (
        "not constricted at crosswalls" in lower
        or "without constriction at cell-cell junction" in lower
        or "cross walls non-constricted" in lower
        or "not constricted at the" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "crosswall_constriction",
                value_text="absent",
                evidence_text=text
            )
        )

    elif (
        "slightly constricted" in lower
        or "slight constrictions" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "crosswall_constriction",
                value_text="slight",
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Calyptra
    # --------------------------------------------------------

    if (
        "without calyptra" in lower
        or "without calyptras" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "calyptra_absent",
                value_text="present",
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Trichome shape
    # --------------------------------------------------------

    trichome_shapes = []

    if "trichomes" in lower or "trichome" in lower:

        if "straight" in lower:
            trichome_shapes.append(
                "straight"
            )

        if (
            "slightly curved" in lower
            or "slightly bent" in lower
            or "weakly curved" in lower
        ):
            trichome_shapes.append(
                "slightly_curved"
            )

    if trichome_shapes:

        responses.append(
            make_response(
                observation_id,
                "trichome_shape",
                value_text="; ".join(
                    sorted(set(trichome_shapes))
                ),
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Motility
    # --------------------------------------------------------

    if (
        "slowly gliding" in lower
        or "filaments moving" in lower
    ):

        responses.append(
            make_response(
                observation_id,
                "gliding_motility",
                value_text="observed",
                evidence_text=text
            )
        )

    if "oscillating" in lower:

        responses.append(
            make_response(
                observation_id,
                "oscillatory_motility",
                value_text="observed",
                evidence_text=text
            )
        )

    # --------------------------------------------------------
    # Trichome width
    # --------------------------------------------------------

    width_patterns = [

        # Keep the match within the same clause/sentence as "trichome".
        # This prevents a later cell-width phrase such as
        # "Cells ... 13.1-15.2 um long X 2.3-3.8 um wide"
        # from being mislabelled as trichome_width.
        r"(?:trichomes?|trichome)\s+[^.;]*?"
        r"(\d+(?:\.\d+)?)\s*-\s*"
        r"(\d+(?:\.\d+)?)\s*u?m\s+wide",

        r"width(?:\s+of\s+(?:the\s+)?trichome)?"
        r"\s*(?:was|is|of|ranging from)?\s*"
        r"(\d+(?:\.\d+)?)\s*-\s*"
        r"(\d+(?:\.\d+)?)\s*u?m"
    ]

    for pattern in width_patterns:

        for match in re.finditer(
            pattern,
            lower
        ):

            responses.append(
                make_response(
                    observation_id,
                    "trichome_width",
                    value_min=float(match.group(1)),
                    value_max=float(match.group(2)),
                    unit="um",
                    evidence_text=text
                )
            )


# ============================================================
# MAIN FUNCTION
# ============================================================

def normalize_organism_response(
    observation_id,
    species,
    description
):
    """
    Normalize one spreadsheet morphology/response description.

    Returns
    -------
    list[dict]

    Each dictionary becomes one organism_responses row.
    """

    if description is None:
        return []

    text = normalize_text(
        description
    )

    if not text:
        return []

    responses = []

    species_lower = str(
        species
    ).lower().strip()

    # Run the generic dimension extractor for all species so explicit
    # cell-length/cell-width measurements are retained.
    extract_cell_dimensions(
        observation_id,
        text,
        responses
    )

    if "nitzschia palea" in species_lower:

        normalize_nitzschia(
            observation_id,
            text,
            responses
        )

    elif "euglena viridis" in species_lower:

        normalize_euglena(
            observation_id,
            text,
            responses
        )

    elif "oscillatoria limosa" in species_lower:

        normalize_oscillatoria(
            observation_id,
            text,
            responses
        )

    # Oscillatoria descriptions can state trichome width as a plain
    # "... 13-16 um wide" phrase. The generic cell-width parser can then
    # create an identical cell_width record even though the source is
    # explicitly describing the trichome. Remove only those exact
    # cell_width duplicates that are matched by a trichome_width record.
    if "oscillatoria limosa" in species_lower:
        trichome_width_keys = {
            (
                response["value_numeric"],
                response["value_min"],
                response["value_max"],
                response["unit"],
            )
            for response in responses
            if response["response_type"] == "trichome_width"
        }

        responses = [
            response
            for response in responses
            if not (
                response["response_type"] == "cell_width"
                and (
                    response["value_numeric"],
                    response["value_min"],
                    response["value_max"],
                    response["unit"],
                ) in trichome_width_keys
            )
        ]

    return remove_duplicates(
        responses
    )


# ============================================================
# DUPLICATE HANDLING
# ============================================================

def remove_duplicates(
    responses
):

    seen = set()
    output = []

    for response in responses:

        key = (
            response["response_type"],
            response["value_text"],
            response["value_numeric"],
            response["value_min"],
            response["value_max"],
            response["unit"],
            response["qualifier"]
        )

        if key not in seen:

            seen.add(key)
            output.append(
                response
            )

    return output


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    example = """
    Types of deformities observed in N. palea cells exposed
    to cadmium using light microscopy, including irregular
    striae and fibulae, fibulae interruption, slight and
    severe deviation of the raphe, abnormal valve shape.
    """

    results = normalize_organism_response(
        observation_id=(
            "Nitzschia_palea_stressed_row10"
        ),
        species="Nitzschia palea",
        description=example
    )

    for result in results:
        print(result)