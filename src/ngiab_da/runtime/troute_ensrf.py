"""Localized deterministic EnSRF analysis for the real t-route ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ngiab_da.filters import EnSRFResult, SerialEnSRF
from ngiab_da.localization import gaspari_cohn

from .troute_analysis_gateway import (
    BaselineTRouteForecastAnalysisGateway,
    TRouteForecastAnalysisState,
)
from ngiab_da.localization.along_stream import along_stream_distance_matrix


class TRouteEnSRFAnalysisError(RuntimeError):
    """Raised when a real routing EnSRF analysis cannot be completed."""


def _readonly_array(values: Any, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TRouteLocalizedEnSRFOutcome:
    """Forecast, localization, filter diagnostics, and applied analysis."""

    gage_ids: tuple[str, ...]
    observations: np.ndarray
    error_std: np.ndarray
    segment_distances: np.ndarray
    state_localization: np.ndarray
    forecast: TRouteForecastAnalysisState
    filter_result: EnSRFResult
    applied: TRouteForecastAnalysisState

    def __post_init__(self) -> None:
        gage_ids = tuple(str(value) for value in self.gage_ids)
        observations = _readonly_array(
            self.observations,
            dtype=np.float64,
        )
        error_std = _readonly_array(
            self.error_std,
            dtype=np.float64,
        )
        distances = _readonly_array(
            self.segment_distances,
            dtype=np.float64,
        )
        localization = _readonly_array(
            self.state_localization,
            dtype=np.float64,
        )

        observation_count = len(gage_ids)
        state_count = self.forecast.segment_count * 3

        if observation_count == 0:
            raise TRouteEnSRFAnalysisError(
                "At least one routing observation is required."
            )
        if len(set(gage_ids)) != observation_count:
            raise TRouteEnSRFAnalysisError(
                "Routing gage IDs must be unique."
            )
        if observations.shape != (observation_count,):
            raise TRouteEnSRFAnalysisError(
                "Observation values do not align with gage IDs."
            )
        if error_std.shape != (observation_count,):
            raise TRouteEnSRFAnalysisError(
                "Observation errors do not align with gage IDs."
            )
        if distances.shape != (
            observation_count,
            self.forecast.segment_count,
        ):
            raise TRouteEnSRFAnalysisError(
                "Segment distances have an invalid shape."
            )
        if localization.shape != (
            observation_count,
            state_count,
        ):
            raise TRouteEnSRFAnalysisError(
                "State localization has an invalid shape."
            )
        if np.any(error_std <= 0.0):
            raise TRouteEnSRFAnalysisError(
                "Observation-error standard deviations must be positive."
            )
        if np.any(localization < 0.0) or np.any(localization > 1.0):
            raise TRouteEnSRFAnalysisError(
                "Localization weights must lie in [0, 1]."
            )
        if self.filter_result.analysis_values.shape != (
            self.forecast.member_count,
            state_count,
        ):
            raise TRouteEnSRFAnalysisError(
                "EnSRF analysis state has an invalid shape."
            )
        if not np.array_equal(
            self.applied.member_ids,
            self.forecast.member_ids,
        ):
            raise TRouteEnSRFAnalysisError(
                "Applied analysis changed routing member order."
            )

        object.__setattr__(self, "gage_ids", gage_ids)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "error_std", error_std)
        object.__setattr__(self, "segment_distances", distances)
        object.__setattr__(self, "state_localization", localization)

    def forecast_predictions(self) -> np.ndarray:
        """Return modeled discharge with shape ``(member, observation)``."""

        return _readonly_array(
            np.column_stack(
                [
                    self.forecast.gage_prediction(gage_id)
                    for gage_id in self.gage_ids
                ]
            ),
            dtype=np.float64,
        )

    def analysis_predictions(self) -> np.ndarray:
        """Return EnSRF-updated observation equivalents."""

        return self.filter_result.analysis_predicted_observations


class BaselineTRouteLocalizedEnSRF:
    """Apply network-localized deterministic EnSRF to real t-route q0."""

    def __init__(
        self,
        gateway: BaselineTRouteForecastAnalysisGateway,
        *,
        cutoff_distance_m: float,
        filter_: SerialEnSRF | None = None,
        nonnegative_projection: bool = True,
    ) -> None:
        cutoff = float(cutoff_distance_m)
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError(
                "cutoff_distance_m must be finite and positive."
            )

        self._gateway = gateway
        self._cutoff_distance_m = cutoff
        self._filter = SerialEnSRF() if filter_ is None else filter_
        self._nonnegative_projection = bool(nonnegative_projection)

    @property
    def cutoff_distance_m(self) -> float:
        return self._cutoff_distance_m

    def analyze(
        self,
        forecast: TRouteForecastAnalysisState,
        *,
        observations: Mapping[str, float],
        error_std: Mapping[str, float],
    ) -> TRouteLocalizedEnSRFOutcome:
        """Assimilate routing discharge and transactionally apply analyzed q0."""

        gage_ids = tuple(str(key) for key in observations)
        if not gage_ids:
            raise TRouteEnSRFAnalysisError(
                "At least one routing observation is required."
            )
        if set(gage_ids) != set(str(key) for key in error_std):
            raise TRouteEnSRFAnalysisError(
                "Observation and error mappings must use identical gage IDs."
            )

        observation_values = np.asarray(
            [float(observations[gage_id]) for gage_id in gage_ids],
            dtype=np.float64,
        )
        observation_errors = np.asarray(
            [float(error_std[gage_id]) for gage_id in gage_ids],
            dtype=np.float64,
        )

        if not np.isfinite(observation_values).all():
            raise TRouteEnSRFAnalysisError(
                "Routing observations must be finite."
            )
        if (
            not np.isfinite(observation_errors).all()
            or np.any(observation_errors <= 0.0)
        ):
            raise TRouteEnSRFAnalysisError(
                "Routing observation errors must be finite and positive."
            )

        predicted = np.column_stack(
            [
                forecast.gage_prediction(gage_id)
                for gage_id in gage_ids
            ]
        )
        distances = self.segment_distances(
            forecast,
            gage_ids,
        )
        segment_weights = gaspari_cohn(
            distances,
            self._cutoff_distance_m,
        )
        state_localization = np.repeat(
            segment_weights,
            repeats=3,
            axis=1,
        )
        observation_localization = self._observation_localization(
            forecast,
            gage_ids,
        )

        result = self._filter.update(
            state_values=forecast.flattened_q0(),
            predicted_observations=predicted,
            observations=observation_values,
            error_std=observation_errors,
            localization_weights=state_localization,
            observation_localization_weights=(
                observation_localization
            ),
        )

        analyzed_q0 = result.analysis_values.reshape(
            forecast.member_count,
            forecast.segment_count,
            3,
        )
        if self._nonnegative_projection:
            analyzed_q0 = np.maximum(analyzed_q0, 0.0)

        applied = self._gateway.apply_analyzed_q0(
            analyzed_q0
        )

        return TRouteLocalizedEnSRFOutcome(
            gage_ids=gage_ids,
            observations=observation_values,
            error_std=observation_errors,
            segment_distances=distances,
            state_localization=state_localization,
            forecast=forecast,
            filter_result=result,
            applied=applied,
        )

    def segment_distances(
        self,
        forecast: TRouteForecastAnalysisState,
        gage_ids: Sequence[str],
    ) -> np.ndarray:
        """Return strict Along-The-Stream center-to-center distances.

        A gauge may influence its complete contributing upstream tree and
        its single downstream routing path.  Localization is not permitted
        to travel downstream through a confluence and then reverse upstream
        into another tributary.
        """

        members = self._gateway.ensemble.members

        if not members:
            raise TRouteEnSRFAnalysisError(
                "The routing ensemble has no members."
            )

        domain = members[0].domain

        if not np.array_equal(
            domain.segment_ids,
            forecast.segment_ids,
        ):
            raise TRouteEnSRFAnalysisError(
                "Forecast segment order differs from the live domain."
            )

        source_segments: list[int] = []

        for gage_id in gage_ids:
            try:
                source_segments.append(
                    int(
                        forecast.gage_to_segment[
                            str(gage_id)
                        ]
                    )
                )
            except KeyError as exc:
                raise TRouteEnSRFAnalysisError(
                    f"Unknown routing gage: {gage_id}"
                ) from exc

        try:
            return along_stream_distance_matrix(
                segment_ids=domain.segment_ids,
                segment_toids=domain.segment_toids,
                segment_lengths=(
                    domain.hydraulic_arrays["dx"]
                ),
                source_segment_ids=tuple(
                    source_segments
                ),
            )
        except Exception as exc:
            raise TRouteEnSRFAnalysisError(
                "Strict along-stream localization "
                "could not be constructed."
            ) from exc

    def _observation_localization(
        self,
        forecast: TRouteForecastAnalysisState,
        gage_ids: Sequence[str],
    ) -> np.ndarray:
        distances = self.segment_distances(
            forecast,
            gage_ids,
        )
        gage_positions = []
        for gage_id in gage_ids:
            segment = forecast.gage_to_segment[str(gage_id)]
            positions = np.flatnonzero(
                forecast.segment_ids == segment
            )
            if positions.size != 1:
                raise TRouteEnSRFAnalysisError(
                    f"Gage segment {segment} is not unique."
                )
            gage_positions.append(int(positions[0]))

        pairwise = np.empty(
            (len(gage_ids), len(gage_ids)),
            dtype=np.float64,
        )
        for row in range(len(gage_ids)):
            pairwise[row, :] = distances[
                row,
                gage_positions,
            ]

        return gaspari_cohn(
            pairwise,
            self._cutoff_distance_m,
        )

    @staticmethod
    def _dijkstra(
        graph: Mapping[int, Sequence[tuple[int, float]]],
        source: int,
    ) -> Mapping[int, float]:
        if source not in graph:
            raise TRouteEnSRFAnalysisError(
                f"Localization source segment is outside the domain: {source}."
            )

        distances: dict[int, float] = {source: 0.0}
        queue: list[tuple[float, int]] = [(0.0, source)]

        while queue:
            distance, segment = heappop(queue)
            if distance != distances[segment]:
                continue

            for neighbor, edge_distance in graph[segment]:
                proposed = distance + edge_distance
                if proposed < distances.get(neighbor, np.inf):
                    distances[neighbor] = proposed
                    heappush(queue, (proposed, neighbor))

        return MappingProxyType(distances)
