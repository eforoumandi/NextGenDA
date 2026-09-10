from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class ObservationTargetError(
    ValueError
):
    pass


@dataclass(
    frozen=True,
    slots=True,
)
class ObservationTarget:
    gage_id: str

    flowpath_id: str
    flowpath_attribute_link: str

    observation_nexus_id: str
    divide_id: str

    poi_id: str
    vpuid: str


def _required_string(
    mapping: Mapping[str, Any],
    key: str,
) -> str:
    if key not in mapping:
        raise ObservationTargetError(
            f"Required target field {key!r} is absent."
        )

    value = str(
        mapping[
            key
        ]
    ).strip()

    if not value:
        raise ObservationTargetError(
            f"Required target field {key!r} is empty."
        )

    return value


def resolve_observation_target(
    manifest: Mapping[str, Any],
) -> ObservationTarget:

    request = manifest.get(
        "request"
    )

    crosswalk = manifest.get(
        "target_crosswalk"
    )

    if not isinstance(
        request,
        Mapping,
    ):
        raise ObservationTargetError(
            "Prepared manifest request section is absent."
        )

    if not isinstance(
        crosswalk,
        Mapping,
    ):
        raise ObservationTargetError(
            "Prepared manifest target_crosswalk section is absent."
        )


    selector_type = str(
        request.get(
            "selector_type",
            "",
        )
    ).strip().lower()

    selector_value = str(
        request.get(
            "selector_value",
            "",
        )
    ).strip()

    crosswalk_gage = _required_string(
        crosswalk,
        "gage_id",
    )


    #
    # When preparation was explicitly requested by gauge,
    # both independently stored values must agree.
    #
    if selector_type == "gage":

        if not selector_value:
            raise ObservationTargetError(
                "Gauge selector has no selector_value."
            )

        if selector_value != crosswalk_gage:
            raise ObservationTargetError(
                "Prepared request gauge and hydrofabric "
                "crosswalk gauge disagree: "
                f"{selector_value!r} != {crosswalk_gage!r}."
            )


    problems = crosswalk.get(
        "problems",
        [],
    )

    if problems:
        raise ObservationTargetError(
            "Target hydrofabric crosswalk contains problems: "
            + repr(
                problems
            )
        )


    return ObservationTarget(
        gage_id=crosswalk_gage,

        flowpath_id=_required_string(
            crosswalk,
            "flowpath_id",
        ),

        flowpath_attribute_link=_required_string(
            crosswalk,
            "flowpath_attribute_link",
        ),

        observation_nexus_id=_required_string(
            crosswalk,
            "observation_nexus_id",
        ),

        divide_id=_required_string(
            crosswalk,
            "divide_id",
        ),

        poi_id=_required_string(
            crosswalk,
            "poi_id",
        ),

        vpuid=_required_string(
            crosswalk,
            "vpuid",
        ),
    )
