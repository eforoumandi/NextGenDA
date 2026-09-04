"""Real CFE forecast extraction and transactional assimilation-state updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .cfe_ensemble import (
    BaselineCFEEnsembleRuntime,
    CFEEnsembleStepResult,
)


class CFEAnalysisGatewayError(RuntimeError):
    """Raised when a CFE forecast or analysis state is invalid."""


def _readonly_array(values: Any, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CFEForecastAnalysisState:
    """Immutable CFE forecast exposed to the particle-filter layer."""

    member_ids: tuple[str, ...]
    catchment_id: str
    target_time: float
    storage_state: np.ndarray
    q_out: np.ndarray
    q_out_units: str

    def __post_init__(self) -> None:
        member_ids = tuple(str(value) for value in self.member_ids)
        catchment_id = str(self.catchment_id)
        target_time = float(self.target_time)
        storage = _readonly_array(
            self.storage_state,
            dtype=np.float64,
        )
        q_out = _readonly_array(
            self.q_out,
            dtype=np.float64,
        )

        member_count = len(member_ids)
        if member_count == 0:
            raise CFEAnalysisGatewayError(
                "At least one CFE member is required."
            )
        if len(set(member_ids)) != member_count:
            raise CFEAnalysisGatewayError(
                "CFE member IDs must be unique."
            )
        if not np.isfinite(target_time):
            raise CFEAnalysisGatewayError(
                "target_time must be finite."
            )
        if storage.shape != (member_count, 2):
            raise CFEAnalysisGatewayError(
                "storage_state must have shape (member, 2)."
            )
        if q_out.shape != (member_count,):
            raise CFEAnalysisGatewayError(
                "q_out must have shape (member,)."
            )
        if not np.isfinite(storage).all():
            raise CFEAnalysisGatewayError(
                "CFE storage state contains non-finite values."
            )
        if np.any(storage < 0.0):
            raise CFEAnalysisGatewayError(
                "CFE storage state contains negative values."
            )
        if not np.isfinite(q_out).all():
            raise CFEAnalysisGatewayError(
                "CFE Q_OUT contains non-finite values."
            )

        object.__setattr__(self, "member_ids", member_ids)
        object.__setattr__(self, "catchment_id", catchment_id)
        object.__setattr__(self, "target_time", target_time)
        object.__setattr__(self, "storage_state", storage)
        object.__setattr__(self, "q_out", q_out)
        object.__setattr__(self, "q_out_units", str(self.q_out_units))

    @property
    def member_count(self) -> int:
        return len(self.member_ids)

    def state_for(self, member_id: str) -> np.ndarray:
        """Return ``[soil, groundwater]`` for one member."""

        key = str(member_id)
        try:
            position = self.member_ids.index(key)
        except ValueError as exc:
            raise CFEAnalysisGatewayError(
                f"Unknown CFE member: {key}"
            ) from exc

        return _readonly_array(
            self.storage_state[position],
            dtype=np.float64,
        )


class BaselineCFEForecastAnalysisGateway:
    """Connect a real CFE ensemble to forecast and PF state operations."""

    def __init__(
        self,
        ensemble: BaselineCFEEnsembleRuntime,
    ) -> None:
        self._ensemble = ensemble
        self._last_step: CFEEnsembleStepResult | None = None

    @property
    def ensemble(self) -> BaselineCFEEnsembleRuntime:
        return self._ensemble

    @property
    def last_step(self) -> CFEEnsembleStepResult:
        if self._last_step is None:
            raise CFEAnalysisGatewayError(
                "No CFE forecast has been completed."
            )
        return self._last_step

    def advance_forecast(
        self,
        forcing_by_member: Mapping[str, Mapping[str, float]],
        *,
        until: float | None = None,
    ) -> CFEForecastAnalysisState:
        """Advance the real ensemble and expose its PF forecast."""

        step = self._ensemble.advance(
            forcing_by_member,
            until=until,
        )
        self._last_step = step
        return self._state_from_step(step)

    def current_state(self) -> CFEForecastAnalysisState:
        """Read live assimilation storage after the latest forecast."""

        step = self.last_step
        storage = np.stack(
            [
                member.state_adapter.capture().vector
                for member in self._ensemble.members
            ],
            axis=0,
        )

        return CFEForecastAnalysisState(
            member_ids=step.member_ids,
            catchment_id=step.catchment_id,
            target_time=self._ensemble.current_time,
            storage_state=storage,
            q_out=step.output_vector("Q_OUT"),
            q_out_units=self._q_out_units(),
        )

    def apply_analyzed_state(
        self,
        analyzed_storage_state: Any,
    ) -> CFEForecastAnalysisState:
        """Transactionally replace soil/GW storage for every member.

        This updates only the two CFE storage variables exposed by BMI.
        Queue and Nash-cascade process memory remain outside this adapter.
        """

        forecast = self.current_state()
        proposed = np.asarray(
            analyzed_storage_state,
            dtype=np.float64,
        )

        if proposed.shape != forecast.storage_state.shape:
            raise CFEAnalysisGatewayError(
                "Analyzed CFE storage must have shape "
                f"{forecast.storage_state.shape}; "
                f"received {proposed.shape}."
            )
        if not np.isfinite(proposed).all():
            raise CFEAnalysisGatewayError(
                "Analyzed CFE storage contains non-finite values."
            )
        if np.any(proposed < 0.0):
            raise CFEAnalysisGatewayError(
                "Analyzed CFE storage contains negative values."
            )

        snapshots = [
            member.state_adapter.capture()
            for member in self._ensemble.members
        ]

        try:
            for position, member in enumerate(
                self._ensemble.members
            ):
                member.state_adapter.restore_vector(
                    proposed[position]
                )
                actual = member.state_adapter.capture().vector
                np.testing.assert_allclose(
                    actual,
                    proposed[position],
                    rtol=0.0,
                    atol=1.0e-12,
                )
        except Exception as exc:
            rollback_errors: list[str] = []

            for member, snapshot in zip(
                self._ensemble.members,
                snapshots,
            ):
                try:
                    member.state_adapter.restore(snapshot)
                except Exception as rollback_exc:
                    rollback_errors.append(
                        f"{member.member_id}:"
                        f"{type(rollback_exc).__name__}"
                    )

            if rollback_errors:
                raise CFEAnalysisGatewayError(
                    "CFE analysis application failed and rollback "
                    f"also failed: {rollback_errors}."
                ) from exc

            raise CFEAnalysisGatewayError(
                "CFE analysis application failed; all exposed "
                "storage states were rolled back."
            ) from exc

        return self.current_state()

    def _state_from_step(
        self,
        step: CFEEnsembleStepResult,
    ) -> CFEForecastAnalysisState:
        return CFEForecastAnalysisState(
            member_ids=step.member_ids,
            catchment_id=step.catchment_id,
            target_time=step.target_time,
            storage_state=step.state_matrix(),
            q_out=step.output_vector("Q_OUT"),
            q_out_units=self._q_out_units(),
        )

    def _q_out_units(self) -> str:
        members = self._ensemble.members
        if not members:
            raise CFEAnalysisGatewayError(
                "The CFE ensemble contains no members."
            )

        units = tuple(
            member.model.get_var_units("Q_OUT")
            for member in members
        )
        if len(set(units)) != 1:
            raise CFEAnalysisGatewayError(
                f"CFE members disagree on Q_OUT units: {units}."
            )
        return units[0]
