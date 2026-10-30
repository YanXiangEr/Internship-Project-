"""
visual_observation.py

Structured representation of visual features observed from a
microorganism image or video.

This module does not perform computer vision yet.

It defines the contract between:

    image/video perception
            ↓
    visual biological features
            ↓
    inverse microorganism world model

Only response types permitted by visual_response_schema.py can
be inserted as image or video observations.
"""


from copy import deepcopy

from visual_response_schema import (
    get_visual_schema,
    supports_image_input,
    supports_video_input,
)


# ============================================================
# VISUAL FEATURE
# ============================================================

class VisualFeature:
    """
    One observed biological feature.

    Example:

        cell_shape = spindle
    """

    def __init__(
        self,
        response_type,
        value,
        confidence=None,
        source="manual",
        notes=None
    ):

        self.response_type = (
            response_type
        )

        self.value = (
            value
        )

        self.confidence = (
            confidence
        )

        self.source = (
            source
        )

        self.notes = (
            notes
        )

    def to_dict(self):

        return {
            "response_type":
                self.response_type,

            "value":
                self.value,

            "confidence":
                self.confidence,

            "source":
                self.source,

            "notes":
                self.notes,
        }

    def __repr__(self):

        return (
            "VisualFeature("
            f"response_type={self.response_type!r}, "
            f"value={self.value!r}, "
            f"confidence={self.confidence!r}"
            ")"
        )


# ============================================================
# VISUAL OBSERVATION
# ============================================================

class VisualObservation:
    """
    Represents visual information extracted from one image
    or video.

    At this stage, features can be entered manually.

    Later, a computer-vision model will populate the same
    structure automatically.
    """

    def __init__(
        self,
        species,
        media_type="image",
        features=None,
        source_reference=None
    ):

        if not species:

            raise ValueError(
                "VisualObservation requires a species."
            )

        media_type = (
            str(media_type)
            .strip()
            .lower()
        )

        if media_type not in {
            "image",
            "video"
        }:

            raise ValueError(
                "media_type must be 'image' or 'video'."
            )

        self.species = species

        self.media_type = (
            media_type
        )

        self.source_reference = (
            source_reference
        )

        self.features = []

        if features:

            for feature in features:

                if isinstance(
                    feature,
                    VisualFeature
                ):

                    self.add_feature_object(
                        feature
                    )

                elif isinstance(
                    feature,
                    dict
                ):

                    self.add_feature(
                        response_type=(
                            feature[
                                "response_type"
                            ]
                        ),

                        value=(
                            feature[
                                "value"
                            ]
                        ),

                        confidence=(
                            feature.get(
                                "confidence"
                            )
                        ),

                        source=(
                            feature.get(
                                "source",
                                "manual"
                            )
                        ),

                        notes=(
                            feature.get(
                                "notes"
                            )
                        )
                    )

                else:

                    raise TypeError(
                        "features must contain "
                        "VisualFeature objects or dictionaries."
                    )


    # ========================================================
    # VALIDATION
    # ========================================================

    def validate_response_type(
        self,
        response_type
    ):
        """
        Check whether this feature is permitted for the
        current media type.
        """

        schema = get_visual_schema(
            response_type
        )

        if self.media_type == "image":

            if not supports_image_input(
                response_type
            ):

                raise ValueError(
                    f"{response_type!r} is not currently "
                    "approved as an image-input feature. "
                    f"Visual role: "
                    f"{schema['visual_role']}."
                )

        elif self.media_type == "video":

            if not supports_video_input(
                response_type
            ):

                raise ValueError(
                    f"{response_type!r} is not currently "
                    "approved as a video-input feature. "
                    f"Visual role: "
                    f"{schema['visual_role']}."
                )


    # ========================================================
    # ADD FEATURE
    # ========================================================

    def add_feature(
        self,
        response_type,
        value,
        confidence=None,
        source="manual",
        notes=None
    ):
        """
        Add one visually observed biological feature.
        """

        if not response_type:

            raise ValueError(
                "response_type cannot be empty."
            )

        self.validate_response_type(
            response_type
        )

        if confidence is not None:

            confidence = float(
                confidence
            )

            if not (
                0.0
                <=
                confidence
                <=
                1.0
            ):

                raise ValueError(
                    "Feature confidence must be "
                    "between 0 and 1."
                )

        feature = VisualFeature(
            response_type=(
                response_type
            ),

            value=value,

            confidence=(
                confidence
            ),

            source=source,

            notes=notes
        )

        # Replace an existing feature of the same type rather
        # than silently storing contradictory duplicate values.

        for index, existing in enumerate(
            self.features
        ):

            if (
                existing.response_type
                ==
                response_type
            ):

                self.features[
                    index
                ] = feature

                return feature

        self.features.append(
            feature
        )

        return feature


    def add_feature_object(
        self,
        feature
    ):

        if not isinstance(
            feature,
            VisualFeature
        ):

            raise TypeError(
                "feature must be a VisualFeature."
            )

        return self.add_feature(
            response_type=(
                feature.response_type
            ),

            value=(
                feature.value
            ),

            confidence=(
                feature.confidence
            ),

            source=(
                feature.source
            ),

            notes=(
                feature.notes
            )
        )


    # ========================================================
    # GET FEATURE
    # ========================================================

    def get_feature(
        self,
        response_type
    ):

        for feature in self.features:

            if (
                feature.response_type
                ==
                response_type
            ):

                return feature

        return None


    # ========================================================
    # REMOVE FEATURE
    # ========================================================

    def remove_feature(
        self,
        response_type
    ):

        self.features = [
            feature
            for feature
            in self.features
            if (
                feature.response_type
                !=
                response_type
            )
        ]


    # ========================================================
    # WORLD-MODEL RESPONSE FORMAT
    # ========================================================

    def to_observed_response_dict(
        self
    ):
        """
        Convert visual features into the dictionary format
        currently expected by infer_water_quality().

        Example:

            {
                "cell_shape": "spindle",
                "posterior_morphology":
                    "sharply_or_bluntly_pointed"
            }
        """

        return {
            feature.response_type:
                feature.value

            for feature in self.features
        }


    # ========================================================
    # SERIALIZATION
    # ========================================================

    def to_dict(self):

        return {
            "species":
                self.species,

            "media_type":
                self.media_type,

            "source_reference":
                self.source_reference,

            "features": [
                feature.to_dict()
                for feature
                in self.features
            ]
        }


    @classmethod
    def from_dict(
        cls,
        data
    ):

        return cls(
            species=data[
                "species"
            ],

            media_type=data.get(
                "media_type",
                "image"
            ),

            source_reference=data.get(
                "source_reference"
            ),

            features=deepcopy(
                data.get(
                    "features",
                    []
                )
            )
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    def summary(self):

        lines = []

        lines.append(
            "=" * 70
        )

        lines.append(
            "VISUAL OBSERVATION"
        )

        lines.append(
            "=" * 70
        )

        lines.append(
            f"Species: {self.species}"
        )

        lines.append(
            f"Media type: {self.media_type}"
        )

        if self.source_reference:

            lines.append(
                f"Source: "
                f"{self.source_reference}"
            )

        lines.append("")

        lines.append(
            "Observed visual features:"
        )

        if not self.features:

            lines.append(
                "  No visual features recorded."
            )

        for feature in self.features:

            lines.append(
                f"  {feature.response_type}: "
                f"{feature.value}"
            )

            if (
                feature.confidence
                is not None
            ):

                lines.append(
                    f"    Confidence: "
                    f"{feature.confidence}"
                )

            lines.append(
                f"    Source: "
                f"{feature.source}"
            )

            schema = get_visual_schema(
                feature.response_type
            )

            lines.append(
                f"    Visual role: "
                f"{schema['visual_role']}"
            )

        return "\n".join(
            lines
        )


    def __repr__(self):

        return (
            "VisualObservation("
            f"species={self.species!r}, "
            f"media_type={self.media_type!r}, "
            f"features={len(self.features)}"
            ")"
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # TEST 1
    # VALID IMAGE FEATURES
    # --------------------------------------------------------

    observation = VisualObservation(
        species="Euglena viridis",
        media_type="image",
        source_reference=(
            "example_microscopy_image"
        )
    )

    observation.add_feature(
        response_type="cell_shape",
        value="spindle",
        confidence=0.92,
        source="manual_test"
    )

    observation.add_feature(
        response_type=(
            "posterior_morphology"
        ),
        value=(
            "sharply_or_bluntly_pointed"
        ),
        confidence=0.88,
        source="manual_test"
    )

    observation.add_feature(
        response_type=(
            "cell_length"
        ),
        value="55-68 um",
        confidence=0.80,
        source="manual_test"
    )

    print()

    print(
        observation.summary()
    )

    print()

    print(
        "WORLD-MODEL RESPONSE DICTIONARY"
    )

    print()

    print(
        observation
        .to_observed_response_dict()
    )

    # --------------------------------------------------------
    # TEST 2
    # SERIALIZATION
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "SERIALIZATION TEST"
    )

    print(
        "=" * 70
    )

    serialized = (
        observation.to_dict()
    )

    reconstructed = (
        VisualObservation.from_dict(
            serialized
        )
    )

    print()

    print(
        reconstructed.summary()
    )

    # --------------------------------------------------------
    # TEST 3
    # INVALID STATIC-IMAGE FEATURE
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "IMAGE SAFETY TEST"
    )

    print(
        "=" * 70
    )

    try:

        observation.add_feature(
            response_type=(
                "oscillatory_motility"
            ),
            value="observed",
            confidence=0.9
        )

        print()

        print(
            "[FAIL] A temporal motility feature "
            "was accepted from a static image."
        )

    except ValueError as error:

        print()

        print(
            "[PASS] Static image rejected "
            "temporal motility feature."
        )

        print()

        print(
            error
        )