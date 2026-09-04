"""Validated user configuration for production ensemble perturbations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any


class PerturbationConfigurationError(
    ValueError
):
    """Invalid production ensemble/perturbation configuration."""


def _finite_float(
    value: object,
    *,
    name: str,
) -> float:

    if isinstance(
        value,
        bool,
    ):

        raise PerturbationConfigurationError(
            f"{name} must be numeric, not boolean."
        )


    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise PerturbationConfigurationError(
            f"{name} must be numeric."
        ) from exc


    if not math.isfinite(
        result
    ):

        raise PerturbationConfigurationError(
            f"{name} must be finite."
        )


    return result


@dataclass(
    frozen=True,
    slots=True,
)
class AssimilationPerturbationConfig:
    """
    User-selectable production perturbation configuration.

    Defaults reproduce the currently validated generalized
    ensemble/forcing/state-perturbation settings.
    """

    ensemble_size: int = 50

    forcing_phi: float = 0.73

    precipitation_cv: float = 0.45

    temperature_sigma_k: float = 1.0

    forcing_spatial_correlation: float = 0.27

    precip_temperature_correlation: float = -0.10

    sacsma_state_std_fraction: float = (
        0.0017712117239475898
    )

    sacsma_state_correlation_seconds: float = (
        10800.0
    )

    sacsma_state_truncation_sigma: float = 2.5


    def __post_init__(
        self,
    ) -> None:

        if (
            isinstance(
                self.ensemble_size,
                bool,
            )
            or
            not isinstance(
                self.ensemble_size,
                int,
            )
        ):

            raise PerturbationConfigurationError(
                "ensemble_size must be an integer."
            )


        if self.ensemble_size < 2:

            raise PerturbationConfigurationError(
                "ensemble_size must be at least 2."
            )


        phi = _finite_float(
            self.forcing_phi,
            name="forcing_phi",
        )


        if not -1.0 < phi < 1.0:

            raise PerturbationConfigurationError(
                "forcing_phi must lie strictly within (-1, 1)."
            )


        precip_cv = _finite_float(
            self.precipitation_cv,
            name="precipitation_cv",
        )


        if precip_cv < 0.0:

            raise PerturbationConfigurationError(
                "precipitation_cv must be nonnegative."
            )


        temperature_sigma = _finite_float(
            self.temperature_sigma_k,
            name="temperature_sigma_k",
        )


        if temperature_sigma < 0.0:

            raise PerturbationConfigurationError(
                "temperature_sigma_k must be nonnegative."
            )


        spatial = _finite_float(
            self.forcing_spatial_correlation,
            name="forcing_spatial_correlation",
        )


        if not 0.0 < spatial < 1.0:

            raise PerturbationConfigurationError(
                "forcing_spatial_correlation must lie in (0, 1)."
            )


        cross = _finite_float(
            self.precip_temperature_correlation,
            name="precip_temperature_correlation",
        )


        if not -1.0 < cross < 1.0:

            raise PerturbationConfigurationError(
                "precip_temperature_correlation must lie "
                "strictly within (-1, 1)."
            )


        state_std = _finite_float(
            self.sacsma_state_std_fraction,
            name="sacsma_state_std_fraction",
        )


        if state_std < 0.0:

            raise PerturbationConfigurationError(
                "sacsma_state_std_fraction must be nonnegative."
            )


        state_tau = _finite_float(
            self.sacsma_state_correlation_seconds,
            name="sacsma_state_correlation_seconds",
        )


        if state_tau <= 0.0:

            raise PerturbationConfigurationError(
                "sacsma_state_correlation_seconds must be positive."
            )


        truncation = _finite_float(
            self.sacsma_state_truncation_sigma,
            name="sacsma_state_truncation_sigma",
        )


        if truncation <= 0.0:

            raise PerturbationConfigurationError(
                "sacsma_state_truncation_sigma must be positive."
            )


        object.__setattr__(
            self,
            "forcing_phi",
            phi,
        )

        object.__setattr__(
            self,
            "precipitation_cv",
            precip_cv,
        )

        object.__setattr__(
            self,
            "temperature_sigma_k",
            temperature_sigma,
        )

        object.__setattr__(
            self,
            "forcing_spatial_correlation",
            spatial,
        )

        object.__setattr__(
            self,
            "precip_temperature_correlation",
            cross,
        )

        object.__setattr__(
            self,
            "sacsma_state_std_fraction",
            state_std,
        )

        object.__setattr__(
            self,
            "sacsma_state_correlation_seconds",
            state_tau,
        )

        object.__setattr__(
            self,
            "sacsma_state_truncation_sigma",
            truncation,
        )


    def to_contract_payload(
        self,
    ) -> dict[
        str,
        int | float,
    ]:

        return {
            "ensemble_size":
                self.ensemble_size,

            "forcing_phi":
                self.forcing_phi,

            "precipitation_cv":
                self.precipitation_cv,

            "temperature_sigma_k":
                self.temperature_sigma_k,

            "forcing_spatial_correlation":
                self.forcing_spatial_correlation,

            "precip_temperature_correlation":
                self.precip_temperature_correlation,

            "sacsma_state_std_fraction":
                self.sacsma_state_std_fraction,

            "sacsma_state_correlation_seconds":
                self.sacsma_state_correlation_seconds,

            "sacsma_state_truncation_sigma":
                self.sacsma_state_truncation_sigma,
        }


    @classmethod
    def from_contract_payload(
        cls,
        payload: Mapping[
            str,
            Any,
        ] | None,
    ) -> "AssimilationPerturbationConfig":

        if payload is None:

            return cls()


        if not isinstance(
            payload,
            Mapping,
        ):

            raise PerturbationConfigurationError(
                "perturbation_configuration must be an object."
            )


        values = (
            cls()
            .to_contract_payload()
        )


        unknown = sorted(
            set(
                payload
            )
            -
            set(
                values
            )
        )


        if unknown:

            raise PerturbationConfigurationError(
                "Unknown perturbation_configuration fields: "
                +
                ", ".join(
                    unknown
                )
            )


        values.update(
            dict(
                payload
            )
        )


        return cls(
            **values
        )


    @classmethod
    def add_cli_arguments(
        cls,
        parser: Any,
    ) -> None:

        default = cls()


        parser.add_argument(
            "--ensemble-size",
            type=int,
            default=default.ensemble_size,
        )


        parser.add_argument(
            "--forcing-phi",
            type=float,
            default=default.forcing_phi,
        )


        parser.add_argument(
            "--precipitation-cv",
            type=float,
            default=default.precipitation_cv,
        )


        parser.add_argument(
            "--temperature-sigma-k",
            type=float,
            default=default.temperature_sigma_k,
        )


        parser.add_argument(
            "--forcing-spatial-correlation",
            type=float,
            default=default.forcing_spatial_correlation,
        )


        parser.add_argument(
            "--precip-temperature-correlation",
            type=float,
            default=default.precip_temperature_correlation,
        )


        parser.add_argument(
            "--sacsma-state-std-fraction",
            type=float,
            default=default.sacsma_state_std_fraction,
        )


        parser.add_argument(
            "--sacsma-state-correlation-seconds",
            type=float,
            default=default.sacsma_state_correlation_seconds,
        )


        parser.add_argument(
            "--sacsma-state-truncation-sigma",
            type=float,
            default=default.sacsma_state_truncation_sigma,
        )


    @classmethod
    def from_namespace(
        cls,
        namespace: object,
    ) -> "AssimilationPerturbationConfig":

        return cls(
            ensemble_size=namespace.ensemble_size,

            forcing_phi=namespace.forcing_phi,

            precipitation_cv=namespace.precipitation_cv,

            temperature_sigma_k=namespace.temperature_sigma_k,

            forcing_spatial_correlation=(
                namespace.forcing_spatial_correlation
            ),

            precip_temperature_correlation=(
                namespace.precip_temperature_correlation
            ),

            sacsma_state_std_fraction=(
                namespace.sacsma_state_std_fraction
            ),

            sacsma_state_correlation_seconds=(
                namespace.sacsma_state_correlation_seconds
            ),

            sacsma_state_truncation_sigma=(
                namespace.sacsma_state_truncation_sigma
            ),
        )


DEFAULT_PERTURBATION_CONFIG = (
    AssimilationPerturbationConfig()
)
