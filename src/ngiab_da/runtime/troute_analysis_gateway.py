"""Real t-route forecast extraction and transactional analysis-state updates."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .troute_ensemble import (
    BaselineTRouteEnsembleRuntime,
    TRouteEnsembleStepResult,
)


class TRouteAnalysisGatewayError(RuntimeError):
    """Raised when routing forecast or analysis state is invalid."""


def _readonly_array(values: Any, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TRouteForecastAnalysisState:
    """Immutable routing forecast presented to deterministic DA."""

    member_ids: tuple[str, ...]
    target_time: float
    segment_ids: np.ndarray
    discharge: np.ndarray
    q0: np.ndarray
    gage_to_segment: Mapping[str, int]

    def __post_init__(self) -> None:
        member_ids = tuple(str(value) for value in self.member_ids)
        segment_ids = _readonly_array(
            self.segment_ids,
            dtype=np.int64,
        )
        discharge = _readonly_array(
            self.discharge,
            dtype=np.float64,
        )
        q0 = _readonly_array(
            self.q0,
            dtype=np.float64,
        )

        member_count = len(member_ids)
        segment_count = int(segment_ids.size)

        if member_count == 0:
            raise TRouteAnalysisGatewayError(
                "At least one routing member is required."
            )
        if len(set(member_ids)) != member_count:
            raise TRouteAnalysisGatewayError(
                "Routing member IDs must be unique."
            )
        if segment_ids.ndim != 1 or segment_count == 0:
            raise TRouteAnalysisGatewayError(
                "segment_ids must be a non-empty vector."
            )
        if discharge.shape != (member_count, segment_count):
            raise TRouteAnalysisGatewayError(
                "discharge must have shape (member, segment)."
            )
        if q0.shape != (member_count, segment_count, 3):
            raise TRouteAnalysisGatewayError(
                "q0 must have shape (member, segment, 3)."
            )
        if not np.isfinite(discharge).all():
            raise TRouteAnalysisGatewayError(
                "Forecast discharge contains non-finite values."
            )
        if not np.isfinite(q0).all():
            raise TRouteAnalysisGatewayError(
                "Forecast q0 contains non-finite values."
            )

        gage_map = MappingProxyType(
            {
                str(gage): int(segment)
                for gage, segment in self.gage_to_segment.items()
            }
        )
        segment_set = set(int(value) for value in segment_ids)
        if any(
            segment not in segment_set
            for segment in gage_map.values()
        ):
            raise TRouteAnalysisGatewayError(
                "A gage maps outside the routing domain."
            )

        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(self, "target_time", float(self.target_time))
        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "discharge", discharge)
        object.__setattr__(self, "q0", q0)
        object.__setattr__(self, "gage_to_segment", gage_map)

    @property
    def member_count(self) -> int:
        return len(self.member_ids)

    @property
    def segment_count(self) -> int:
        return int(self.segment_ids.size)

    def gage_prediction(self, gage_id: str) -> np.ndarray:
        """Return the modeled discharge ensemble at one gage."""

        key = str(gage_id)
        try:
            segment_id = self.gage_to_segment[key]
        except KeyError as exc:
            raise TRouteAnalysisGatewayError(
                f"Unknown routing gage: {key}"
            ) from exc

        positions = np.flatnonzero(
            self.segment_ids == segment_id
        )
        if positions.size != 1:
            raise TRouteAnalysisGatewayError(
                f"Gage segment {segment_id} is not unique."
            )

        return _readonly_array(
            self.discharge[:, int(positions[0])],
            dtype=np.float64,
        )

    def flattened_q0(self) -> np.ndarray:
        """Return q0 as ``(member, segment*3)`` for DA algorithms."""

        return _readonly_array(
            self.q0.reshape(self.member_count, -1),
            dtype=np.float64,
        )


class BaselineTRouteForecastAnalysisGateway:
    """Connect a real routing ensemble to forecast and analysis operations."""

    def __init__(
        self,
        ensemble: BaselineTRouteEnsembleRuntime,
    ) -> None:
        self._ensemble = ensemble
        self._last_step: TRouteEnsembleStepResult | None = None

    @property
    def ensemble(self) -> BaselineTRouteEnsembleRuntime:
        return self._ensemble

    @property
    def last_step(self) -> TRouteEnsembleStepResult:
        if self._last_step is None:
            raise TRouteAnalysisGatewayError(
                "No routing forecast has been completed."
            )
        return self._last_step

    def advance_forecast(
        self,
        lateral_inflow_by_member: Mapping[str, Any],
        until: float,
        *,
        segment_ids: Any | None = None,
    ) -> TRouteForecastAnalysisState:
        """Advance real members and return an immutable DA forecast."""

        step = self._ensemble.advance(
            lateral_inflow_by_member,
            until,
            segment_ids=segment_ids,
        )
        self._last_step = step

        return TRouteForecastAnalysisState(
            member_ids=step.member_ids,
            target_time=step.target_time,
            segment_ids=step.segment_ids,
            discharge=step.discharge_matrix(),
            q0=step.warm_state_tensor(),
            gage_to_segment=step.gage_to_segment,
        )

    def current_state(self) -> TRouteForecastAnalysisState:
        """Read the current model state using the last forecast discharge."""

        step = self.last_step
        q0 = np.stack(
            [
                np.asarray(
                    member.model.get_value("q0"),
                    dtype=np.float64,
                ).reshape(member.domain.size, 3)
                for member in self._ensemble.members
            ],
            axis=0,
        )

        return TRouteForecastAnalysisState(
            member_ids=step.member_ids,
            target_time=self._ensemble.current_time,
            segment_ids=step.segment_ids,
            discharge=step.discharge_matrix(),
            q0=q0,
            gage_to_segment=step.gage_to_segment,
        )


    def apply_analyzed_q0(
            self,
            analyzed_q0: Any,
        ) -> TRouteForecastAnalysisState:
            """Transactionally replace routing warm state through BMI only.

            The derived t-route BMI boundary owns synchronization between the
            public BMI ``q0`` representation and the live routing kernel state.
            """

            forecast = self.current_state()

            proposed = np.asarray(
                analyzed_q0,
                dtype=np.float64,
            )


            if proposed.shape != forecast.q0.shape:

                raise TRouteAnalysisGatewayError(
                    "Analyzed q0 must have shape "
                    f"{forecast.q0.shape}; received {proposed.shape}."
                )


            if not np.isfinite(
                proposed
            ).all():

                raise TRouteAnalysisGatewayError(
                    "Analyzed q0 contains non-finite values."
                )


            routing_analysis_canonical_quantum = (
                1.0e-5
            )


            proposed = (
                np.rint(
                    proposed
                    / routing_analysis_canonical_quantum
                )
                * routing_analysis_canonical_quantum
            )


            segment_ids = np.asarray(
                forecast.segment_ids,
                dtype=np.int64,
            )


            snapshots: list[
                tuple[
                    Any,
                    np.ndarray,
                    np.ndarray,
                ]
            ] = []


            for member in self._ensemble.members:

                model = member.model


                old_q0 = np.asarray(
                    model.get_value(
                        "q0"
                    ),
                    dtype=np.float64,
                ).copy()


                old_index = np.asarray(
                    model.get_value(
                        "q0_index"
                    ),
                    dtype=np.int64,
                ).copy()


                snapshots.append(
                    (
                        model,
                        old_q0,
                        old_index,
                    )
                )


            try:

                for position, member in enumerate(
                    self._ensemble.members
                ):

                    model = snapshots[
                        position
                    ][0]


                    member_analysis = np.array(
                        proposed[
                            position
                        ],
                        dtype=np.float64,
                        copy=True,
                    )


                    #
                    # Establish canonical ordering first.  The derived BMI
                    # boundary performs live-state synchronization when q0
                    # itself is subsequently written.
                    #
                    model.set_value(
                        "q0_index",
                        np.array(
                            segment_ids,
                            copy=True,
                        ),
                    )


                    model.set_value(
                        "q0",
                        member_analysis.reshape(
                            -1
                        ),
                    )


                    actual_index = np.asarray(
                        model.get_value(
                            "q0_index"
                        ),
                        dtype=np.int64,
                    )


                    actual_q0 = np.asarray(
                        model.get_value(
                            "q0"
                        ),
                        dtype=np.float64,
                    ).reshape(
                        forecast.segment_count,
                        3,
                    )


                    if not np.array_equal(
                        actual_index,
                        segment_ids,
                    ):

                        raise TRouteAnalysisGatewayError(
                            "A routing member changed segment order "
                            "during BMI analysis writeback: "
                            f"{member.member_id}."
                        )


                    if not np.array_equal(
                        actual_q0,
                        member_analysis,
                    ):

                        raise TRouteAnalysisGatewayError(
                            "A routing member did not accept analyzed "
                            "BMI q0 exactly: "
                            f"{member.member_id}."
                        )


            except Exception as exc:

                rollback_errors: list[
                    str
                ] = []


                for (
                    model,
                    old_q0,
                    old_index,
                ) in snapshots:

                    try:

                        model.set_value(
                            "q0_index",
                            old_index,
                        )

                        model.set_value(
                            "q0",
                            old_q0,
                        )

                    except Exception as rollback_exc:

                        rollback_errors.append(
                            type(
                                rollback_exc
                            ).__name__
                        )


                if rollback_errors:

                    raise TRouteAnalysisGatewayError(
                        "Analyzed q0 application failed and BMI rollback "
                        f"also failed: {rollback_errors}."
                    ) from exc


                raise TRouteAnalysisGatewayError(
                    "Analyzed q0 application failed; BMI state was "
                    "rolled back for every routing member."
                ) from exc


            return self.current_state()
