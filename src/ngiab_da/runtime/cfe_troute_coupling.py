"""Shared-member coupling from real CFE runoff to real t-route qlat."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .cfe_analysis_gateway import (
    BaselineCFEForecastAnalysisGateway,
    CFEForecastAnalysisState,
)
from .cfe_qlat import (
    BaselineCFEQlatOperator,
    CFEQlatPrediction,
)
from .troute_analysis_gateway import (
    BaselineTRouteForecastAnalysisGateway,
    TRouteForecastAnalysisState,
)


class CFEToTRouteCouplingError(RuntimeError):
    """Raised when CFE and t-route cannot advance as one member set."""


def _readonly_array(values: Any, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CFEToTRouteCycleResult:
    """One synchronized hydrologic-routing forecast cycle."""

    member_ids: tuple[str, ...]
    target_time: float
    cfe_forecast: CFEForecastAnalysisState
    cfe_qlat: CFEQlatPrediction
    routing_qlat: np.ndarray
    routing_forecast: TRouteForecastAnalysisState

    def __post_init__(self) -> None:
        member_ids = tuple(str(value) for value in self.member_ids)
        target_time = float(self.target_time)
        routing_qlat = _readonly_array(
            self.routing_qlat,
            dtype=np.float64,
        )

        if not member_ids:
            raise CFEToTRouteCouplingError(
                "At least one shared member is required."
            )
        if len(set(member_ids)) != len(member_ids):
            raise CFEToTRouteCouplingError(
                "Shared member IDs must be unique."
            )
        if self.cfe_forecast.member_ids != member_ids:
            raise CFEToTRouteCouplingError(
                "CFE forecast member order differs from the shared order."
            )
        if self.cfe_qlat.member_ids != member_ids:
            raise CFEToTRouteCouplingError(
                "CFE qlat member order differs from the shared order."
            )
        if self.routing_forecast.member_ids != member_ids:
            raise CFEToTRouteCouplingError(
                "Routing forecast member order differs from the shared order."
            )
        if routing_qlat.shape != (
            len(member_ids),
            self.routing_forecast.segment_count,
        ):
            raise CFEToTRouteCouplingError(
                "Routing qlat has an invalid shape."
            )
        if not np.isfinite(routing_qlat).all():
            raise CFEToTRouteCouplingError(
                "Routing qlat contains non-finite values."
            )
        if np.any(routing_qlat < 0.0):
            raise CFEToTRouteCouplingError(
                "Routing qlat contains negative values."
            )
        if not np.isclose(
            self.cfe_forecast.target_time,
            target_time,
        ):
            raise CFEToTRouteCouplingError(
                "CFE forecast did not reach the coupled target time."
            )
        if not np.isclose(
            self.routing_forecast.target_time,
            target_time,
        ):
            raise CFEToTRouteCouplingError(
                "Routing forecast did not reach the coupled target time."
            )

        segment_positions = np.flatnonzero(
            self.routing_forecast.segment_ids
            == self.cfe_qlat.segment_id
        )
        if segment_positions.size != 1:
            raise CFEToTRouteCouplingError(
                "The CFE catchment segment is not unique in t-route."
            )

        position = int(segment_positions[0])
        np.testing.assert_allclose(
            routing_qlat[:, position],
            self.cfe_qlat.qlat_m3s,
            rtol=0.0,
            atol=0.0,
        )

        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(self, "target_time", target_time)
        object.__setattr__(self, "routing_qlat", routing_qlat)

    @property
    def coupled_segment_position(self) -> int:
        positions = np.flatnonzero(
            self.routing_forecast.segment_ids
            == self.cfe_qlat.segment_id
        )
        return int(positions[0])

    def routing_qlat_for(self, member_id: str) -> np.ndarray:
        key = str(member_id)
        try:
            position = self.member_ids.index(key)
        except ValueError as exc:
            raise CFEToTRouteCouplingError(
                f"Unknown shared member: {key}"
            ) from exc

        return _readonly_array(
            self.routing_qlat[position],
            dtype=np.float64,
        )


class BaselineCFEToTRouteCoupler:
    """Advance one real CFE catchment into one real routing domain."""

    def __init__(
        self,
        cfe_gateway: BaselineCFEForecastAnalysisGateway,
        qlat_operator: BaselineCFEQlatOperator,
        routing_gateway: BaselineTRouteForecastAnalysisGateway,
    ) -> None:
        self._cfe_gateway = cfe_gateway
        self._qlat_operator = qlat_operator
        self._routing_gateway = routing_gateway
        self._validate_static_contract()

    @property
    def member_ids(self) -> tuple[str, ...]:
        return self._cfe_gateway.ensemble.member_ids

    def advance(
        self,
        forcing_by_member: Mapping[str, Mapping[str, float]],
        *,
        until: float,
    ) -> CFEToTRouteCycleResult:
        """Run CFE, convert Q_OUT to qlat, and route it."""

        target = float(until)
        if not np.isfinite(target):
            raise CFEToTRouteCouplingError(
                "Coupled target time must be finite."
            )

        cfe_forecast = self._cfe_gateway.advance_forecast(
            forcing_by_member,
            until=target,
        )
        cfe_qlat = self._qlat_operator.predict(cfe_forecast)

        routing_members = self._routing_gateway.ensemble.member_ids
        if cfe_qlat.member_ids != routing_members:
            raise CFEToTRouteCouplingError(
                "CFE and t-route member orders differ."
            )

        routing_segments = (
            self._routing_gateway.ensemble.members[0].domain.segment_ids
        )
        segment_positions = np.flatnonzero(
            routing_segments == cfe_qlat.segment_id
        )
        if segment_positions.size != 1:
            raise CFEToTRouteCouplingError(
                "The CFE catchment segment is not unique in t-route."
            )

        routing_qlat = np.zeros(
            (len(routing_members), routing_segments.size),
            dtype=np.float64,
        )
        routing_qlat[:, int(segment_positions[0])] = (
            cfe_qlat.qlat_m3s
        )

        qlat_by_member = {
            member_id: routing_qlat[position].copy()
            for position, member_id in enumerate(routing_members)
        }
        routing_forecast = self._routing_gateway.advance_forecast(
            qlat_by_member,
            target,
        )

        return CFEToTRouteCycleResult(
            member_ids=routing_members,
            target_time=target,
            cfe_forecast=cfe_forecast,
            cfe_qlat=cfe_qlat,
            routing_qlat=routing_qlat,
            routing_forecast=routing_forecast,
        )

    def _validate_static_contract(self) -> None:
        cfe_members = self._cfe_gateway.ensemble.member_ids
        routing_members = self._routing_gateway.ensemble.member_ids

        if cfe_members != routing_members:
            raise CFEToTRouteCouplingError(
                "CFE and t-route must use the same ordered member IDs."
            )
        if (
            self._cfe_gateway.ensemble.catchment_id
            != self._qlat_operator.catchment_id
        ):
            raise CFEToTRouteCouplingError(
                "The qlat operator does not match the CFE catchment."
            )

        routing_domain = (
            self._routing_gateway.ensemble.members[0].domain
        )
        positions = np.flatnonzero(
            routing_domain.segment_ids
            == self._qlat_operator.segment_id
        )
        if positions.size != 1:
            raise CFEToTRouteCouplingError(
                "The qlat operator segment is absent or duplicated "
                "in the routing domain."
            )
