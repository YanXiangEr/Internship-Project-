"""
world_state.py

Defines the state representation used by the microorganism
world-model simulator.

A WorldState describes:

    1. Which microorganism is being simulated
    2. The current water-quality/environmental conditions
    3. The currently known microorganism responses

This module does NOT perform prediction itself.
Prediction/state transitions will be handled by simulator.py.
"""


from copy import deepcopy


# ============================================================
# WORLD STATE
# ============================================================

class WorldState:
    """
    Representation of one state of the microorganism world.

    Example
    -------

    state = WorldState(
        species="Nitzschia palea",
        water_state={
            "petrol": {
                "value": 10,
                "unit": "uL/L"
            }
        }
    )
    """

    def __init__(
        self,
        species,
        water_state=None,
        organism_state=None
    ):

        if not species:
            raise ValueError(
                "WorldState requires a microorganism species."
            )

        self.species = species

        self.water_state = (
            deepcopy(water_state)
            if water_state is not None
            else {}
        )

        self.organism_state = (
            deepcopy(organism_state)
            if organism_state is not None
            else {}
        )


    # ========================================================
    # WATER-QUALITY STATE
    # ========================================================

    def set_water_factor(
        self,
        factor,
        value,
        unit=None
    ):
        """
        Add or update a water-quality factor.

        Examples
        --------

        state.set_water_factor(
            "petrol",
            10,
            "uL/L"
        )

        state.set_water_factor(
            "eutrophication",
            "present"
        )
        """

        if not factor:
            raise ValueError(
                "Water factor cannot be empty."
            )

        if unit is not None:

            self.water_state[
                factor
            ] = {
                "value": value,
                "unit": unit
            }

        else:

            self.water_state[
                factor
            ] = value


    def remove_water_factor(
        self,
        factor
    ):
        """
        Remove a water-quality factor if it exists.
        """

        self.water_state.pop(
            factor,
            None
        )


    def get_water_factor(
        self,
        factor
    ):
        """
        Return a water-quality factor value.
        """

        return self.water_state.get(
            factor
        )


    # ========================================================
    # ORGANISM STATE
    # ========================================================

    def set_response(
        self,
        response_type,
        value,
        evidence_score=None,
        independent_observations=None
    ):
        """
        Add or update a microorganism response.

        Backward-compatible multi-value behaviour
        -----------------------------------------
        ``organism_state`` remains keyed by response type so existing
        simulator/UI code can continue using::

            organism_state["cell_length"]["value"]

        When the same response type is supported with a *different*
        value, the alternatives are additionally preserved under
        ``supported_values`` instead of silently overwriting one
        observation with another.

        The top-level ``value`` remains the primary value:
        - a higher evidence score replaces a weaker primary value;
        - equal evidence leaves the first encountered value primary;
        - all distinct supported values are retained.

        This is appropriate for evidence such as two independent
        observed cell-length ranges. The model does not average those
        ranges or imply that one is the uniquely predicted value.
        """

        if not response_type:
            raise ValueError(
                "Response type cannot be empty."
            )

        candidate = {
            "value": value
        }

        if evidence_score is not None:
            candidate[
                "evidence_score"
            ] = evidence_score

        if independent_observations is not None:
            candidate[
                "independent_observations"
            ] = independent_observations

        existing = self.organism_state.get(
            response_type
        )

        # First value for this response type.
        if existing is None:

            response_record = deepcopy(
                candidate
            )

            response_record[
                "supported_values"
            ] = [
                deepcopy(
                    candidate
                )
            ]

            self.organism_state[
                response_type
            ] = response_record

            return

        # Preserve any values already accumulated by a newer state.
        supported_values = deepcopy(
            existing.get(
                "supported_values",
                [
                    {
                        key: deepcopy(value_)
                        for key, value_
                        in existing.items()
                        if key != "supported_values"
                    }
                ]
            )
        )

        # Avoid exact duplicate value records. If the same value is seen
        # again with stronger evidence, retain the stronger metadata.
        candidate_value = str(
            candidate.get(
                "value"
            )
        )

        matching_index = None

        for index, stored in enumerate(
            supported_values
        ):

            if str(
                stored.get(
                    "value"
                )
            ) == candidate_value:

                matching_index = index
                break

        if matching_index is None:

            supported_values.append(
                deepcopy(
                    candidate
                )
            )

        else:

            stored = supported_values[
                matching_index
            ]

            stored_score = stored.get(
                "evidence_score"
            )

            candidate_score = candidate.get(
                "evidence_score"
            )

            if (
                candidate_score is not None
                and (
                    stored_score is None
                    or candidate_score > stored_score
                )
            ):

                supported_values[
                    matching_index
                ] = deepcopy(
                    candidate
                )

        # Keep a single primary value for backward compatibility.
        existing_score = existing.get(
            "evidence_score"
        )

        candidate_score = candidate.get(
            "evidence_score"
        )

        should_replace_primary = (
            candidate_score is not None
            and (
                existing_score is None
                or candidate_score > existing_score
            )
        )

        if should_replace_primary:

            primary = deepcopy(
                candidate
            )

        else:

            primary = {
                key: deepcopy(value_)
                for key, value_
                in existing.items()
                if key != "supported_values"
            }

        primary[
            "supported_values"
        ] = supported_values

        self.organism_state[
            response_type
        ] = primary


    def remove_response(
        self,
        response_type
    ):
        """
        Remove an organism response if it exists.
        """

        self.organism_state.pop(
            response_type,
            None
        )


    def get_response(
        self,
        response_type
    ):
        """
        Return a microorganism response.
        """

        return self.organism_state.get(
            response_type
        )


    # ========================================================
    # COPYING
    # ========================================================

    def copy(self):
        """
        Return an independent copy of the world state.

        This is important for simulation because the simulator
        should create a new state rather than modifying the
        original state in place.
        """

        return WorldState(
            species=self.species,
            water_state=deepcopy(
                self.water_state
            ),
            organism_state=deepcopy(
                self.organism_state
            )
        )


    # ========================================================
    # SERIALIZATION
    # ========================================================

    def to_dict(self):
        """
        Convert the WorldState into a normal Python dictionary.

        This will later make the state easy to:

            - display in Streamlit
            - save as JSON
            - pass into another model
            - use as ML training data
        """

        return {
            "species":
                self.species,

            "water_state":
                deepcopy(
                    self.water_state
                ),

            "organism_state":
                deepcopy(
                    self.organism_state
                )
        }


    @classmethod
    def from_dict(
        cls,
        data
    ):
        """
        Create a WorldState from a dictionary.
        """

        if not isinstance(
            data,
            dict
        ):
            raise TypeError(
                "WorldState.from_dict() requires a dictionary."
            )

        return cls(
            species=data[
                "species"
            ],

            water_state=data.get(
                "water_state",
                {}
            ),

            organism_state=data.get(
                "organism_state",
                {}
            )
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    def summary(self):
        """
        Return a readable multi-line representation of the
        current state.
        """

        lines = []

        lines.append(
            "=" * 70
        )

        lines.append(
            "WORLD STATE"
        )

        lines.append(
            "=" * 70
        )

        lines.append(
            f"Species: {self.species}"
        )

        lines.append("")

        # ----------------------------------------------------
        # WATER STATE
        # ----------------------------------------------------

        lines.append(
            "Water quality:"
        )

        if not self.water_state:

            lines.append(
                "  No water-quality factors specified."
            )

        else:

            for factor, value in (
                self.water_state.items()
            ):

                if isinstance(
                    value,
                    dict
                ):

                    numeric_value = value.get(
                        "value"
                    )

                    unit = value.get(
                        "unit"
                    )

                    if unit:

                        lines.append(
                            f"  {factor}: "
                            f"{numeric_value} "
                            f"{unit}"
                        )

                    else:

                        lines.append(
                            f"  {factor}: "
                            f"{numeric_value}"
                        )

                else:

                    lines.append(
                        f"  {factor}: {value}"
                    )

        lines.append("")

        # ----------------------------------------------------
        # ORGANISM STATE
        # ----------------------------------------------------

        lines.append(
            "Microorganism responses:"
        )

        if not self.organism_state:

            lines.append(
                "  No responses currently known."
            )

        else:

            for (
                response_type,
                response
            ) in self.organism_state.items():

                supported_values = response.get(
                    "supported_values",
                    []
                )

                distinct_values = []

                for supported in supported_values:

                    supported_value = supported.get(
                        "value"
                    )

                    if (
                        supported_value
                        not in distinct_values
                    ):

                        distinct_values.append(
                            supported_value
                        )

                if len(distinct_values) > 1:

                    lines.append(
                        f"  {response_type}: "
                        f"{' | '.join(str(v) for v in distinct_values)}"
                    )

                    lines.append(
                        "    Multiple evidence-supported values/ranges "
                        "were observed; they are preserved separately."
                    )

                else:

                    lines.append(
                        f"  {response_type}: "
                        f"{response.get('value')}"
                    )

                if (
                    response.get(
                        "evidence_score"
                    )
                    is not None
                ):

                    lines.append(
                        "    Evidence score: "
                        f"{response['evidence_score']}"
                    )

                if (
                    response.get(
                        "independent_observations"
                    )
                    is not None
                ):

                    lines.append(
                        "    Independent observations: "
                        f"{response['independent_observations']}"
                    )

        return "\n".join(
            lines
        )


    def __repr__(self):
        """
        Short representation useful while debugging.
        """

        return (
            "WorldState("
            f"species={self.species!r}, "
            f"water_factors={len(self.water_state)}, "
            f"responses={len(self.organism_state)}"
            ")"
        )


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # CREATE INITIAL STATE
    # --------------------------------------------------------

    state = WorldState(
        species="Nitzschia palea"
    )

    # --------------------------------------------------------
    # ADD WATER CONDITION
    # --------------------------------------------------------

    state.set_water_factor(
        factor="petrol",
        value=10,
        unit="uL/L"
    )

    print()
    print(
        "INITIAL STATE"
    )

    print()

    print(
        state.summary()
    )

    # --------------------------------------------------------
    # COPY STATE
    # --------------------------------------------------------

    next_state = state.copy()

    # --------------------------------------------------------
    # ADD EXAMPLE RESPONSES
    # --------------------------------------------------------

    next_state.set_response(
        response_type="cell_death",
        value="20.8 percent",
        evidence_score=1.0,
        independent_observations=1
    )

    next_state.set_response(
        response_type="oil_production",
        value="increased",
        evidence_score=1.0,
        independent_observations=1
    )

    next_state.set_response(
        response_type="pore_percolation",
        value="present",
        evidence_score=0.5,
        independent_observations=1
    )

    print()
    print(
        "PREDICTED NEXT STATE"
    )

    print()

    print(
        next_state.summary()
    )

    # --------------------------------------------------------
    # VERIFY ORIGINAL STATE WAS NOT MODIFIED
    # --------------------------------------------------------

    print()
    print(
        "ORIGINAL STATE STILL HAS:"
    )

    print(
        f"{len(state.organism_state)} "
        "organism responses."
    )

    print()
    print(
        "NEXT STATE HAS:"
    )

    print(
        f"{len(next_state.organism_state)} "
        "organism responses."
    )

    # --------------------------------------------------------
    # DICTIONARY REPRESENTATION
    # --------------------------------------------------------

    print()
    print(
        "DICTIONARY REPRESENTATION"
    )

    print()

    print(
        next_state.to_dict()
    )