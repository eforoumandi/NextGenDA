"""Generic runoff-generation particle-filter binding utilities.

Contains PF bookkeeping, deterministic seeding, contiguous cycle
registration, qlat extraction, and routing-posterior-to-qlat
regression. It contains no process rerun or generation handshake.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from ngiab_da.coupling.dual_filter import RoutingPosteriorQlat
from ngiab_da.engine.cycle import CycleWindow


class RunoffPFBindingError(RuntimeError):
    """Raised when the runoff-PF binding contract is violated."""



_RAW_OBSERVATION_KEY_TOKENS = (
    "discharge",
    "streamflow",
    "usgs",
    "value_cms",
    "error_stddev_cms",
)

def _readonly(values: Any, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array

def _assert_no_raw_observation_fields(value: Any) -> None:
    """Reject raw-discharge/USGS field names from PF durable payloads."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            token = str(key).strip().lower()
            if any(
                marker in token
                for marker in _RAW_OBSERVATION_KEY_TOKENS
            ):
                raise RunoffPFBindingError(
                    "Raw discharge/USGS fields are prohibited from "
                    f"CFE PF durable payloads: {key!r}."
                )
            _assert_no_raw_observation_fields(item)
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_raw_observation_fields(item)


class RunoffPFBinding:
    """Model-independent runoff-generation PF state and utilities."""

    def __init__(
        self,
        member_ids: Sequence[str],
        output_root: str | Path,
        *,
        prior_weights: Any | None = None,
        resampling_threshold_fraction: float = 0.5,
        force_resampling: bool = False,
        pf_observation_relative_error: float = 0.10,
        pf_prediction_relative_error: float = 0.10,
        pf_minimum_error_std_m3s: float = 1.0,
        pf_random_seed: int | None = None,
        minimum_error_std_m3s: float = 1.0e-8,
        relative_error_floor: float = 0.02,
        covariance_regularization_fraction: float = 1.0e-12,
        enabled: bool = True,
    ) -> None:
        ids = tuple(str(value).strip() for value in member_ids)
        if not ids or any(not value for value in ids):
            raise ValueError(
                "member_ids must contain non-empty strings."
            )
        if len(set(ids)) != len(ids):
            raise ValueError("member_ids must be unique.")

        threshold = float(resampling_threshold_fraction)
        if (
            not math.isfinite(threshold)
            or threshold <= 0.0
            or threshold > 1.0
        ):
            raise ValueError(
                "resampling_threshold_fraction must lie in (0, 1]."
            )

        pf_observation_error = float(
            pf_observation_relative_error
        )
        pf_prediction_error = float(
            pf_prediction_relative_error
        )
        pf_minimum_error = float(
            pf_minimum_error_std_m3s
        )

        if pf_random_seed is None:
            resolved_pf_random_seed = None
        else:
            if isinstance(pf_random_seed, bool):
                raise TypeError(
                    "pf_random_seed must be an integer or None."
                )
            resolved_pf_random_seed = int(pf_random_seed)
            if resolved_pf_random_seed < 0:
                raise ValueError(
                    "pf_random_seed must be nonnegative."
                )

        minimum_error = float(minimum_error_std_m3s)
        relative_error = float(relative_error_floor)
        regularization = float(
            covariance_regularization_fraction
        )

        if (
            not math.isfinite(pf_observation_error)
            or pf_observation_error < 0.0
        ):
            raise ValueError(
                "pf_observation_relative_error must be "
                "finite and nonnegative."
            )

        if (
            not math.isfinite(pf_prediction_error)
            or pf_prediction_error < 0.0
        ):
            raise ValueError(
                "pf_prediction_relative_error must be "
                "finite and nonnegative."
            )

        if (
            not math.isfinite(pf_minimum_error)
            or pf_minimum_error <= 0.0
        ):
            raise ValueError(
                "pf_minimum_error_std_m3s must be "
                "finite and positive."
            )
        if not math.isfinite(minimum_error) or minimum_error <= 0.0:
            raise ValueError(
                "minimum_error_std_m3s must be finite and positive."
            )
        if not math.isfinite(relative_error) or relative_error < 0.0:
            raise ValueError(
                "relative_error_floor must be finite and nonnegative."
            )
        if not math.isfinite(regularization) or regularization < 0.0:
            raise ValueError(
                "covariance_regularization_fraction must be finite "
                "and nonnegative."
            )

        if prior_weights is None:
            weights = np.full(
                len(ids),
                1.0 / len(ids),
                dtype=np.float64,
            )
        else:
            weights = np.asarray(
                prior_weights,
                dtype=np.float64,
            )
            if weights.shape != (len(ids),):
                raise ValueError(
                    "prior_weights must contain one value per member."
                )
            if not np.isfinite(weights).all() or np.any(weights < 0.0):
                raise ValueError(
                    "prior_weights must be finite and nonnegative."
                )
            total = float(np.sum(weights))
            if total <= 0.0:
                raise ValueError(
                    "prior_weights must have positive total mass."
                )
            weights = weights / total

        root = Path(output_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

        self._member_ids = ids
        self._root = root
        self._posterior_weights = _readonly(
            weights,
            dtype=np.float64,
        )
        self._pf_observation_relative_error = (
            pf_observation_error
        )
        self._pf_prediction_relative_error = (
            pf_prediction_error
        )
        self._pf_minimum_error_std_m3s = (
            pf_minimum_error
        )
        self._pf_random_seed = resolved_pf_random_seed

        self._resampling_threshold = threshold
        self._force_resampling = bool(force_resampling)
        self._minimum_error = minimum_error
        self._relative_error = relative_error
        self._regularization = regularization
        self._enabled = bool(enabled) and len(ids) >= 2
        self._source_cycles: list[CycleWindow] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def member_ids(self) -> tuple[str, ...]:
        return self._member_ids


    @property
    def pf_observation_relative_error(self) -> float:
        return float(self._pf_observation_relative_error)


    @property
    def pf_prediction_relative_error(self) -> float:
        return float(self._pf_prediction_relative_error)


    @property
    def pf_minimum_error_std_m3s(self) -> float:
        return float(self._pf_minimum_error_std_m3s)

    @property
    def source_cycles(self) -> tuple[CycleWindow, ...]:
        return tuple(self._source_cycles)

    def register_cycle(self, cycle: CycleWindow) -> None:
        """Record every model interval for deterministic lineage lineage."""

        if not isinstance(cycle, CycleWindow):
            raise TypeError("cycle must be a CycleWindow.")
        if not self._source_cycles:
            self._source_cycles.append(cycle)
            return

        previous = self._source_cycles[-1]
        if cycle == previous:
            return
        if (
            cycle.cycle_index != previous.cycle_index + 1
            or cycle.start_time != previous.end_time
        ):
            raise RunoffPFBindingError(
                "runoff PF lineage cycles must be contiguous."
            )
        self._source_cycles.append(cycle)

    @staticmethod
    def _seed(
        *,
        run_id: str,
        cycle: CycleWindow,
        purpose: str,
        pf_random_seed: int | None = None,
    ) -> int:
        if pf_random_seed is None:
            namespace = run_id
        else:
            namespace = f"fixed-seed:{pf_random_seed}"

        material = (
            f"{namespace}|{cycle.cycle_id}|{purpose}"
        ).encode("utf-8")
        digest = hashlib.sha256(material).digest()
        return int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        )

    def _qlat_matrix(
        self,
        lateral_by_member: Mapping[str, Any],
        *,
        segment_ids: Any,
        location_segment_ids: Sequence[int],
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        segments = np.asarray(segment_ids, dtype=np.int64)
        if segments.ndim != 1 or segments.size == 0:
            raise RunoffPFBindingError(
                "Routing segment IDs must be a non-empty vector."
            )
        if np.unique(segments).size != segments.size:
            raise RunoffPFBindingError(
                "Routing segment IDs must be unique."
            )
        index = {
            int(segment): position
            for position, segment in enumerate(segments)
        }

        requested: list[int] = []
        seen: set[int] = set()
        for raw in location_segment_ids:
            segment = int(raw)
            if segment in seen:
                continue
            if segment not in index:
                raise RunoffPFBindingError(
                    "PF qlat location is outside the routing domain: "
                    f"{segment}."
                )
            requested.append(segment)
            seen.add(segment)
        if not requested:
            raise RunoffPFBindingError(
                "At least one PF qlat location is required."
            )

        rows = []
        positions = [index[segment] for segment in requested]
        for member_id in self._member_ids:
            try:
                lateral = np.asarray(
                    lateral_by_member[member_id],
                    dtype=np.float64,
                )
            except KeyError as exc:
                raise RunoffPFBindingError(
                    f"Missing qlat member: {member_id}."
                ) from exc
            if lateral.shape != segments.shape:
                raise RunoffPFBindingError(
                    "Member qlat shape differs from routing segments."
                )
            if (
                not np.isfinite(lateral).all()
                or np.any(lateral < 0.0)
            ):
                raise RunoffPFBindingError(
                    "Member qlat must be finite and nonnegative."
                )
            rows.append(lateral[positions])

        return (
            np.asarray(rows, dtype=np.float64),
            tuple(str(value) for value in requested),
        )

    def _routing_posterior_analysis(
        self,
        forecast_qlat: np.ndarray,
        location_ids: tuple[str, ...],
        routing_outcome: Any,
        *,
        location_localization_weights: Any | None = None,
    ) -> tuple[RoutingPosteriorQlat, np.ndarray]:
        """Derive routing-posterior qlat summary and full ensemble.

        The routing EnSRF updates discharge.  A regression through the
        forecast cross-covariance maps that discharge increment back into
        qlat space.  The resulting qlat ensemble is retained for the CFE
        particle likelihood so its spatial covariance, rather than an
        independent-location approximation, controls the update.
        """

        try:
            forecast_discharge = np.asarray(
                routing_outcome.forecast_predictions(),
                dtype=np.float64,
            )
            posterior_discharge = np.asarray(
                routing_outcome.analysis_predictions(),
                dtype=np.float64,
            )
        except AttributeError as exc:
            raise RunoffPFBindingError(
                "Routing outcome lacks forecast/posterior predictions."
            ) from exc

        member_count = len(self._member_ids)

        if (
            forecast_discharge.ndim != 2
            or forecast_discharge.shape[0] != member_count
            or forecast_discharge.shape[1] < 1
        ):
            raise RunoffPFBindingError(
                "Routing forecast predictions have an invalid shape."
            )

        if posterior_discharge.shape != forecast_discharge.shape:
            raise RunoffPFBindingError(
                "Routing posterior prediction shape differs."
            )

        if (
            forecast_qlat.ndim != 2
            or forecast_qlat.shape[0] != member_count
            or forecast_qlat.shape[1] != len(location_ids)
        ):
            raise RunoffPFBindingError(
                "Forecast qlat has an invalid shape."
            )

        if not (
            np.isfinite(forecast_discharge).all()
            and np.isfinite(posterior_discharge).all()
            and np.isfinite(forecast_qlat).all()
        ):
            raise RunoffPFBindingError(
                "Routing-posterior regression inputs must be finite."
            )

        qlat_anomalies = (
            forecast_qlat
            - np.mean(
                forecast_qlat,
                axis=0,
            )
        )

        discharge_anomalies = (
            forecast_discharge
            - np.mean(
                forecast_discharge,
                axis=0,
            )
        )

        denominator = member_count - 1

        cross_covariance = (
            qlat_anomalies.T
            @ discharge_anomalies
            / denominator
        )

        discharge_covariance = (
            discharge_anomalies.T
            @ discharge_anomalies
            / denominator
        )

        covariance_scale = max(
            float(
                np.trace(
                    discharge_covariance
                )
            ),
            float(
                np.max(
                    np.abs(
                        discharge_covariance
                    )
                )
            ),
            np.finfo(np.float64).eps,
        )

        stabilized = (
            discharge_covariance
            + np.eye(
                discharge_covariance.shape[0]
            )
            * covariance_scale
            * self._regularization
        )

        gain = (
            cross_covariance
            @ np.linalg.pinv(
                stabilized,
                hermitian=True,
            )
        )

        # Optional hydrologically causal localization in qlat space.
        #
        # The legacy/default path is exactly unchanged when no weights
        # are supplied.  When supplied, each qlat regression row is
        # tapered before both the posterior increment and projected
        # uncertainty are computed.
        if location_localization_weights is not None:

            localization = np.asarray(
                location_localization_weights,
                dtype=np.float64,
            ).reshape(-1)

            if localization.shape != (
                len(location_ids),
            ):
                raise RunoffPFBindingError(
                    "qlat localization weights do not "
                    "align with location_ids."
                )

            if (
                not np.isfinite(
                    localization
                ).all()
                or np.any(
                    localization < 0.0
                )
                or np.any(
                    localization > 1.0
                )
            ):
                raise RunoffPFBindingError(
                    "qlat localization weights must "
                    "be finite and lie in [0, 1]."
                )

            if not np.any(
                localization > 0.0
            ):
                raise RunoffPFBindingError(
                    "At least one qlat localization "
                    "weight must be positive."
                )

            # Preserve exact legacy arithmetic for an all-one taper.
            if not np.all(
                localization == 1.0
            ):
                gain = (
                    gain
                    * localization[
                        :,
                        np.newaxis,
                    ]
                )

        posterior_qlat = (
            forecast_qlat
            + (
                posterior_discharge
                - forecast_discharge
            )
            @ gain.T
        )

        posterior_qlat = np.maximum(
            posterior_qlat,
            0.0,
        )

        if not np.isfinite(
            posterior_qlat
        ).all():
            raise RunoffPFBindingError(
                "Routing-posterior qlat ensemble contains "
                "non-finite values."
            )

        values = np.mean(
            posterior_qlat,
            axis=0,
        )

        spread = np.std(
            posterior_qlat,
            axis=0,
            ddof=1,
        )

        routing_error_std = np.maximum(
            spread,
            np.maximum(
                np.abs(values)
                * self._relative_error,
                self._minimum_error,
            ),
        )

        # The public PF uncertainty controls are specified in routed
        # discharge units. Propagate their independent variances through
        # the same linear regression gain that maps routed-discharge
        # increments into qlat space:
        #
        #   Var(e_qlat,j)
        #       = sum_k gain[j, k]^2 Var(e_Q,k)
        #
        # Observation and prediction errors are treated as independent.
        # The configured minimum is therefore applied in discharge space,
        # never directly as an absolute qlat likelihood floor.
        forecast_discharge_mean = np.mean(
            forecast_discharge,
            axis=0,
        )

        posterior_discharge_mean = np.mean(
            posterior_discharge,
            axis=0,
        )

        observation_error_std_m3s = np.maximum(
            np.abs(
                posterior_discharge_mean
            )
            * self._pf_observation_relative_error,
            self._pf_minimum_error_std_m3s,
        )

        prediction_error_std_m3s = (
            np.abs(
                forecast_discharge_mean
            )
            * self._pf_prediction_relative_error
        )

        discharge_error_variance = (
            np.square(
                observation_error_std_m3s
            )
            + np.square(
                prediction_error_std_m3s
            )
        )

        projected_pf_error_variance = np.maximum(
            np.square(gain)
            @ discharge_error_variance,
            0.0,
        )

        projected_pf_error_std = np.sqrt(
            projected_pf_error_variance
        )

        errors = np.sqrt(
            np.square(
                routing_error_std
            )
            + np.square(
                projected_pf_error_std
            )
        )

        errors = np.maximum(
            errors,
            self._minimum_error,
        )

        if not np.isfinite(errors).all():
            raise RunoffPFBindingError(
                "Projected runoff PF likelihood errors contain "
                "non-finite values."
            )

        feedback = RoutingPosteriorQlat(
            location_ids=location_ids,
            values=np.asarray(
                values,
                dtype=np.float64,
            ),
            error_std=np.asarray(
                errors,
                dtype=np.float64,
            ),
        )

        return (
            feedback,
            np.asarray(
                posterior_qlat,
                dtype=np.float64,
            ),
        )


__all__ = (
    "RunoffPFBinding",
    "RunoffPFBindingError",
    "_assert_no_raw_observation_fields",
    "_readonly",
)
