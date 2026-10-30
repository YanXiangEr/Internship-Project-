from world_model import (
    predict_response,
    infer_water_quality,
    print_forward_prediction,
    print_inverse_prediction
)


SPECIES_OPTIONS = {
    "1": "Nitzschia palea",
    "2": "Euglena viridis",
    "3": "Oscillatoria limosa"
}


def choose_species():
    """
    Ask the user to choose one of the three microorganisms.
    """

    print()
    print("Choose a microorganism:")
    print("1. Nitzschia palea")
    print("2. Euglena viridis")
    print("3. Oscillatoria limosa")

    while True:

        choice = input(
            "\nEnter 1, 2, or 3: "
        ).strip()

        if choice in SPECIES_OPTIONS:
            return SPECIES_OPTIONS[choice]

        print(
            "Invalid choice. Please enter 1, 2, or 3."
        )


def parse_numeric(value):
    """
    Try to convert text into a number.

    Returns None if conversion fails.
    """

    try:
        return float(value)

    except ValueError:
        return None


# ============================================================
# FORWARD MODE
# ============================================================

def run_forward_mode():

    print()
    print("=" * 70)
    print("FORWARD MODE")
    print("Water quality -> microorganism response")
    print("=" * 70)

    species = choose_species()

    water_state = {}

    print()
    print(
        "Enter one or more water-quality factors."
    )

    print(
        "Examples:"
    )

    print(
        "  petrol"
    )

    print(
        "  cadmium"
    )

    print(
        "  sewage"
    )

    print(
        "  eutrophication"
    )

    print(
        "  ammonium"
    )

    print(
        "  phosphate"
    )

    print()
    print(
        "Press Enter without typing a factor "
        "when you are finished."
    )

    while True:

        factor_name = input(
            "\nWater factor: "
        ).strip()

        if not factor_name:
            break

        print()
        print(
            "Does this factor have a numeric "
            "concentration/value?"
        )

        has_numeric = input(
            "Enter y or n: "
        ).strip().lower()

        if has_numeric == "y":

            while True:

                raw_value = input(
                    "Numeric value: "
                ).strip()

                numeric_value = (
                    parse_numeric(
                        raw_value
                    )
                )

                if numeric_value is not None:
                    break

                print(
                    "Please enter a valid number."
                )

            unit = input(
                "Unit (for example uL/L, ug/L): "
            ).strip()

            water_state[
                factor_name
            ] = {
                "value": numeric_value,
                "unit": unit or None
            }

        else:

            text_value = input(
                "Value "
                "(press Enter for 'present'): "
            ).strip()

            if not text_value:
                text_value = "present"

            water_state[
                factor_name
            ] = text_value

    if not water_state:

        print()
        print(
            "No water-quality factors were entered."
        )

        return

    print()
    print(
        "Running forward world model..."
    )

    result = predict_response(
        species=species,
        water_state=water_state,
        top_k=10
    )

    print_forward_prediction(
        result
    )


# ============================================================
# INVERSE MODE
# ============================================================

def run_inverse_mode():

    print()
    print("=" * 70)
    print("INVERSE MODE")
    print(
        "Microorganism response -> possible water quality"
    )
    print("=" * 70)

    species = choose_species()

    observed_response = {}

    print()
    print(
        "Enter one or more observed microorganism responses."
    )

    print()
    print(
        "Examples:"
    )

    print(
        "  sheath_absent"
    )

    print(
        "  cell_death"
    )

    print(
        "  valve_deformation"
    )

    print(
        "  chloroplast_damage"
    )

    print(
        "  reduced_viability"
    )

    print(
        "  growth_reduction"
    )

    print()
    print(
        "Press Enter without typing a response "
        "when you are finished."
    )

    while True:

        response_type = input(
            "\nObserved response: "
        ).strip()

        if not response_type:
            break

        print()
        print(
            "Does this response have a numeric value?"
        )

        has_numeric = input(
            "Enter y or n: "
        ).strip().lower()

        if has_numeric == "y":

            while True:

                raw_value = input(
                    "Numeric value: "
                ).strip()

                numeric_value = (
                    parse_numeric(
                        raw_value
                    )
                )

                if numeric_value is not None:
                    break

                print(
                    "Please enter a valid number."
                )

            unit = input(
                "Unit "
                "(for example percent, um): "
            ).strip()

            observed_response[
                response_type
            ] = {
                "value": numeric_value,
                "unit": unit or None
            }

        else:

            text_value = input(
                "Value "
                "(press Enter for 'present'): "
            ).strip()

            if not text_value:
                text_value = "present"

            observed_response[
                response_type
            ] = text_value

    if not observed_response:

        print()
        print(
            "No microorganism responses were entered."
        )

        return

    print()
    print(
        "Running inverse world model..."
    )

    result = infer_water_quality(
        species=species,
        observed_response=observed_response,
        top_k=10
    )

    print_inverse_prediction(
        result
    )


# ============================================================
# MAIN MENU
# ============================================================

def main():

    while True:

        print()
        print("=" * 70)
        print("MICROORGANISM WORLD MODEL v1")
        print("=" * 70)

        print()
        print("1. Forward model")
        print(
            "   Water quality -> microorganism response"
        )

        print()
        print("2. Inverse model")
        print(
            "   Microorganism response -> water quality"
        )

        print()
        print("3. Exit")

        choice = input(
            "\nChoose 1, 2, or 3: "
        ).strip()

        if choice == "1":

            run_forward_mode()

        elif choice == "2":

            run_inverse_mode()

        elif choice == "3":

            print()
            print(
                "Exiting world model."
            )

            break

        else:

            print()
            print(
                "Invalid choice. Please enter 1, 2, or 3."
            )


if __name__ == "__main__":
    main()