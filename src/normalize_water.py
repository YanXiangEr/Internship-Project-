import re


# ============================================================
# WATER QUALITY ONTOLOGY
# ============================================================
#
# This is our small controlled vocabulary.
#
# Raw spreadsheet terms may vary:
#
#   "oil contaminated"
#   "petrol"
#   "oil spillage"
#
# But the model can still understand that they belong to:
#
#   hydrocarbon_pollution
#
# ============================================================

FACTOR_CATEGORIES = {

    # Nutrients / eutrophication
    "eutrophication": "nutrient_pollution",
    "nitrate": "nutrient",
    "nitrite": "nutrient",
    "ammonium": "nutrient",
    "phosphate": "nutrient",
    "nitrogen": "nutrient",
    "phosphorus": "nutrient",

    # Organic / sewage pollution
    "sewage": "organic_pollution",
    "organic_pollution": "organic_pollution",
    "industrial_effluent": "pollution_source",
    "agricultural_runoff": "pollution_source",
    "municipal_waste": "pollution_source",
    "coliforms": "microbial_pollution",
    "biochemical_oxygen_demand": "oxygen_demand",

    # Metals
    "cadmium": "heavy_metal",
    "copper": "heavy_metal",
    "zinc": "heavy_metal",
    "silver": "heavy_metal",

    # Nanomaterials
    "silver_nanoparticles": "nanomaterial",
    "graphene_oxide": "nanomaterial",
    "fullerene_c60": "nanomaterial",
    "carbon_nanotubes": "nanomaterial",

    # Pesticides
    "insecticide": "pesticide",
    "ddt": "pesticide",

    # Hydrocarbons
    "petrol": "hydrocarbon_pollution",
    "oil": "hydrocarbon_pollution",

    # Other chemical pollution
    "detergent": "chemical_pollution",
    "propranolol": "pharmaceutical_pollution",
    "paint": "chemical_pollution",

    # Physical / general water conditions
    "salinity": "physical_chemical",
    "sodium_metasilicate": "chemical_pollution",
    "turbidity": "physical",
    "hypoxia": "oxygen_condition",

    # Ecological indicator
    "algal_bloom": "ecological_condition"
}


# ============================================================
# HELPER
# ============================================================

def make_factor(
    observation_id,
    factor_name,
    value_text=None,
    value_numeric=None,
    unit=None,
    qualifier=None,
    evidence_text=None,
    confidence=1.0
):
    """
    Create one normalized water-quality factor.
    """

    return {
        "observation_id": observation_id,
        "factor_name": factor_name,
        "factor_category": FACTOR_CATEGORIES.get(
            factor_name,
            "other"
        ),
        "value_text": value_text,
        "value_numeric": value_numeric,
        "unit": unit,
        "qualifier": qualifier,
        "evidence_text": evidence_text,
        "extraction_method": "rule_based",
        "confidence": confidence
    }


# ============================================================
# MAIN NORMALIZER
# ============================================================

def normalize_water_quality(
    observation_id,
    stressor_text
):
    """
    Convert raw stressor/environment text into standardized
    water-quality factors.

    One observation may produce many factors.
    """

    if stressor_text is None:
        return []

    text = str(stressor_text).strip()

    if not text:
        return []

    lower = text.lower()

    factors = []

    # ========================================================
    # EUTROPHICATION
    # ========================================================

    if "eutroph" in lower:

        confidence = 1.0

        if "possible eutroph" in lower:
            confidence = 0.6

        factors.append(
            make_factor(
                observation_id,
                "eutrophication",
                value_text="present",
                evidence_text=text,
                confidence=confidence
            )
        )

    # ========================================================
    # SEWAGE
    # ========================================================

    if "sewage" in lower:

        factors.append(
            make_factor(
                observation_id,
                "sewage",
                value_text="present",
                evidence_text=text
            )
        )

        factors.append(
            make_factor(
                observation_id,
                "organic_pollution",
                value_text="present",
                evidence_text=text,
                confidence=0.9
            )
        )

    # ========================================================
    # INDUSTRIAL EFFLUENT
    # ========================================================

    if (
        "industrial effluent" in lower
        or "industrial sewage" in lower
        or "industrial or domestic sewage" in lower
    ):

        factors.append(
            make_factor(
                observation_id,
                "industrial_effluent",
                value_text="present",
                evidence_text=text
            )
        )

    # ========================================================
    # AGRICULTURAL RUNOFF
    # ========================================================

    if "agricultural runoff" in lower:

        factors.append(
            make_factor(
                observation_id,
                "agricultural_runoff",
                value_text="present",
                evidence_text=text
            )
        )

    # ========================================================
    # MUNICIPAL WASTE
    # ========================================================

    if "municipal" in lower and "waste" in lower:

        factors.append(
            make_factor(
                observation_id,
                "municipal_waste",
                value_text="present",
                evidence_text=text
            )
        )

    # ========================================================
    # NUTRIENTS
    # ========================================================

    nutrient_patterns = {

        "ammonium": [
            "ammonium"
        ],

        "nitrite": [
            "nitrite",
            "no2-",
            "no2"
        ],

        "nitrate": [
            "nitrate",
            "no3-"
        ],

        "phosphate": [
            "phosphate",
            "po43-",
            "po4"
        ],

        "nitrogen": [
            "nitrogen"
        ],

        "phosphorus": [
            "phosphorus"
        ]
    }

    for nutrient, patterns in nutrient_patterns.items():

        matched = any(
            pattern in lower
            for pattern in patterns
        )

        # Match standalone NO3, but not chemical names such as AgNO3.
        if (
            nutrient == "nitrate"
            and
            re.search(
                r"(?<![a-z0-9])no3(?![a-z0-9])",
                lower
            )
        ):
            matched = True

        if not matched:
            continue

        qualifier = None

        # --------------------------------------------------------
        # NUTRIENT-SPECIFIC QUALIFIER
        # --------------------------------------------------------
        #
        # Do not apply a qualifier found anywhere in the complete
        # stressor text to every nutrient in that observation.
        #
        # Example:
        #   "slightly elevated nitrate and nitrite ... elevated phosphate"
        #
        # The previous implementation saw "slightly elevated" anywhere
        # in the sentence and therefore incorrectly labelled phosphate
        # as slightly_elevated too.
        #
        # Instead, first inspect a short local window around the nutrient
        # mention. This preserves qualifiers attached to that nutrient
        # while avoiding contamination from a different nutrient phrase.

        nutrient_mentions = []

        for pattern in patterns:

            for match in re.finditer(
                re.escape(pattern),
                lower
            ):
                nutrient_mentions.append(
                    (
                        match.start(),
                        match.end()
                    )
                )

        # Standalone NO3 is also a valid nitrate mention.
        if nutrient == "nitrate":

            for match in re.finditer(
                r"(?<![a-z0-9])no3(?![a-z0-9])",
                lower
            ):
                nutrient_mentions.append(
                    (
                        match.start(),
                        match.end()
                    )
                )

        local_windows = []

        for start, end in nutrient_mentions:

            window_start = max(
                0,
                start - 35
            )

            window_end = min(
                len(lower),
                end + 35
            )

            local_windows.append(
                lower[
                    window_start:
                    window_end
                ]
            )

        # Prefer the most specific phrase first.
        if any(
            "slightly elevated" in window
            for window in local_windows
        ):
            qualifier = "slightly_elevated"

        elif any(
            re.search(
                r"\bhigh\b",
                window
            )
            for window in local_windows
        ):
            qualifier = "high"

        elif any(
            re.search(
                r"\belevated\b",
                window
            )
            for window in local_windows
        ):
            qualifier = "elevated"

        factors.append(
            make_factor(
                observation_id,
                nutrient,
                value_text=qualifier or "present",
                qualifier=qualifier,
                evidence_text=text
            )
        )

    # ========================================================
    # CADMIUM
    # ========================================================

    if "cadmium" in lower or re.search(r"\bcd\b", lower):

        concentrations = re.findall(
            r"(\d+(?:\.\d+)?)\s*(?:ug|µg)\s*(?:cd)?\s*/\s*l",
            lower
        )

        if concentrations:

            for concentration in concentrations:

                factors.append(
                    make_factor(
                        observation_id,
                        "cadmium",
                        value_numeric=float(concentration),
                        unit="ug/L",
                        evidence_text=text
                    )
                )

        else:

            factors.append(
                make_factor(
                    observation_id,
                    "cadmium",
                    value_text="present",
                    evidence_text=text
                )
            )

    # ========================================================
    # COPPER
    # ========================================================

    if "copper" in lower:

        qualifier = (
            "elevated"
            if "elevated copper" in lower
            else None
        )

        factors.append(
            make_factor(
                observation_id,
                "copper",
                value_text=qualifier or "present",
                qualifier=qualifier,
                evidence_text=text
            )
        )

    # ========================================================
    # INSECTICIDE / DDT
    # ========================================================

    if "insecticide" in lower:

        factors.append(
            make_factor(
                observation_id,
                "insecticide",
                value_text="present",
                evidence_text=text
            )
        )

    if "ddt" in lower:

        factors.append(
            make_factor(
                observation_id,
                "ddt",
                value_text="present",
                evidence_text=text
            )
        )

    # ========================================================
    # GRAPHENE OXIDE
    # ========================================================

    if "graphene oxide" in lower:

        factors.append(
            make_factor(
                observation_id,
                "graphene_oxide",
                value_text="present",
                evidence_text=text
            )
        )

    # ========================================================
    # SILVER NANOPARTICLES
    # ========================================================

    if (
        "silver nanoparticle" in lower
        or "agnp" in lower
    ):

        factors.append(
            make_factor(
                observation_id,
                "silver_nanoparticles",
                value_text="present",
                evidence_text=text
            )
        )

    # Ionic silver is related but should be stored separately.
    if "agno3" in lower or "ag+" in lower:

        factors.append(
            make_factor(
                observation_id,
                "silver",
                value_text="ionic_silver_present",
                evidence_text=text
            )
        )

    # ========================================================
    # FULLERENE
    # ========================================================

    if (
        "c60" in lower
        or "fullerene" in lower
    ):

        factors.append(
            make_factor(
                observation_id,
                "fullerene_c60",
                value_text="present",
                evidence_text=text
            )
        )

    # ========================================================
    # CARBON NANOTUBES
    # ========================================================

    if (
        "carbon nanotube" in lower
        or "mwcnt" in lower
    ):

        factors.append(
            make_factor(
                observation_id,
                "carbon_nanotubes",
                value_text="present",
                evidence_text=text
            )
        )

    # ========================================================
    # PAINT
    # ========================================================

    if "paint" in lower:

        factors.append(
            make_factor(
                observation_id,
                "paint",
                value_text="present",
                evidence_text=text
            )
        )

    # ========================================================
    # DETERGENT
    # ========================================================

    if "detergent" in lower:

        factors.append(
            make_factor(
                observation_id,
                "detergent",
                value_text="present",
                evidence_text=text
            )
        )

    # ========================================================
    # PROPRANOLOL
    # ========================================================

    if "propranolol" in lower:

        factors.append(
            make_factor(
                observation_id,
                "propranolol",
                value_text="present",
                evidence_text=text
            )
        )

    # ========================================================
    # PETROL
    # ========================================================

    if "petrol" in lower:

        factors.append(
            make_factor(
                observation_id,
                "petrol",
                value_text="present",
                evidence_text=text
            )
        )

    # ========================================================
    # OIL
    # ========================================================

    if (
        "oil contaminated" in lower
        or "oil spillage" in lower
        or "oil:" in lower
    ):

        factors.append(
            make_factor(
                observation_id,
                "oil",
                value_text="present",
                evidence_text=text
            )
        )

    # ========================================================
    # HYPOXIA
    # ========================================================

    if "hypoxic" in lower or "hypoxia" in lower:

        factors.append(
            make_factor(
                observation_id,
                "hypoxia",
                value_text="present",
                evidence_text=text
            )
        )

    # ========================================================
    # SODIUM METASILICATE / SALINITY
    # ========================================================

    if (
        "sodium meta silicate" in lower
        or "sodium metal silicate" in lower
        or "sodium metasilicate" in lower
    ):

        factors.append(
            make_factor(
                observation_id,
                "sodium_metasilicate",
                value_text="present",
                evidence_text=text
            )
        )

    if "saline" in lower or "salinity" in lower:

        factors.append(
            make_factor(
                observation_id,
                "salinity",
                value_text="elevated_or_simulated",
                evidence_text=text,
                confidence=0.8
            )
        )

    # ========================================================
    # TURBIDITY
    # ========================================================

    if "turbidity" in lower:

        qualifier = (
            "high"
            if "high turbidity" in lower
            else "present"
        )

        factors.append(
            make_factor(
                observation_id,
                "turbidity",
                value_text=qualifier,
                qualifier=(
                    qualifier
                    if qualifier != "present"
                    else None
                ),
                evidence_text=text
            )
        )

    # ========================================================
    # BIOCHEMICAL OXYGEN DEMAND
    # ========================================================

    if (
        "biochemical oxygen demand" in lower
        or
        re.search(
            r"\bbod\b",
            lower
        )
    ):

        qualifier = (
            "increased"
            if "increased biochemical oxygen demand" in lower
            else "present"
        )

        factors.append(
            make_factor(
                observation_id,
                "biochemical_oxygen_demand",
                value_text=qualifier,
                qualifier=(
                    qualifier
                    if qualifier != "present"
                    else None
                ),
                evidence_text=text
            )
        )

    # ========================================================
    # COLIFORMS
    # ========================================================

    if "coliform" in lower:

        qualifier = (
            "high"
            if "higher concentration" in lower
            else "present"
        )

        factors.append(
            make_factor(
                observation_id,
                "coliforms",
                value_text=qualifier,
                qualifier=(
                    qualifier
                    if qualifier != "present"
                    else None
                ),
                evidence_text=text
            )
        )

    # ========================================================
    # ALGAL BLOOMS
    # ========================================================

    if "algal bloom" in lower:

        factors.append(
            make_factor(
                observation_id,
                "algal_bloom",
                value_text="present",
                evidence_text=text
            )
        )

    return remove_duplicates(factors)


# ============================================================
# DUPLICATE HANDLING
# ============================================================

def remove_duplicates(factors):
    """
    Prevent identical normalized facts from being inserted twice.
    """

    seen = set()
    output = []

    for factor in factors:

        key = (
            factor["factor_name"],
            factor["value_text"],
            factor["value_numeric"],
            factor["unit"]
        )

        if key not in seen:
            seen.add(key)
            output.append(factor)

    return output

if __name__ == "__main__":

    example = """
    Sewage and eutrophication: samples were collected
    from urban waterways receiving heavy inputs of
    industrial or domestic sewage. High ammonium,
    nitrite and phosphate concentrations were detected.
    """

    results = normalize_water_quality(
        "Euglena_viridis_stressed_row6",
        example
    )

    for result in results:
        print(result)