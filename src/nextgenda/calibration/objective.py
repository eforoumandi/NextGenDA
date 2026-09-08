from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from nextgenda.evaluation.metrics import (
    correlation,
    kge,
    nse,
    pbias,
    rmse,
)


class CalibrationObjectiveError(
    ValueError
):
    """Invalid calibration objective input."""


@dataclass(
    frozen=True,
    slots=True,
)
class CalibrationObjectiveResult:

    paired_count: int

    calibration_start_utc: str
    calibration_end_exclusive_utc: str

    correlation: float
    nse: float

    kge_2009: float
    kge_alpha: float
    kge_beta: float

    rmse_cms: float
    pbias_percent: float

    volume_ratio: float
    peak_ratio: float

    objective_1_minus_kge: float


def _utc_timestamp(
    value: Any,
    *,
    name: str,
) -> pd.Timestamp:

    result = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(
        result
    ):

        raise CalibrationObjectiveError(
            f"{name} is not a valid datetime: {value!r}"
        )

    return pd.Timestamp(
        result
    )


def load_observations(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load a calibration observation series.

    Supported schemas
    -----------------
    Current NextGenDA observation cache:
        observed_at
        value_cms
        optional is_usable
        optional quality_weight

    Historical joint-DDS observation file:
        value_date
        obs_flow

    Returned schema:
        time
        obs_flow_cms

    All timestamps are converted to UTC.
    """

    target = (
        Path(
            path
        )
        .expanduser()
        .resolve()
    )

    if not target.is_file():

        raise CalibrationObjectiveError(
            f"Observation file does not exist: {target}"
        )

    frame = pd.read_csv(
        target
    )

    columns = {
        str(
            column
        ).strip().lower():
            column
        for column in frame.columns
    }


    if (
        "observed_at" in columns
        and
        "value_cms" in columns
    ):

        time_column = columns[
            "observed_at"
        ]

        value_column = columns[
            "value_cms"
        ]

        usable = pd.Series(
            True,
            index=frame.index,
            dtype=bool,
        )

        if "is_usable" in columns:

            usable &= (
                pd.to_numeric(
                    frame[
                        columns[
                            "is_usable"
                        ]
                    ],
                    errors="coerce",
                )
                >
                0
            )

        if "quality_weight" in columns:

            usable &= (
                pd.to_numeric(
                    frame[
                        columns[
                            "quality_weight"
                        ]
                    ],
                    errors="coerce",
                )
                >
                0.0
            )

        frame = frame.loc[
            usable
        ].copy()

    elif (
        "value_date" in columns
        and
        "obs_flow" in columns
    ):

        time_column = columns[
            "value_date"
        ]

        value_column = columns[
            "obs_flow"
        ]

    else:

        raise CalibrationObjectiveError(
            "Unsupported observation schema. "
            "Expected either "
            "{observed_at, value_cms} or "
            "{value_date, obs_flow}. "
            f"columns={list(frame.columns)!r}"
        )


    time = pd.to_datetime(
        frame[
            time_column
        ],
        errors="coerce",
        utc=True,
    )

    flow = pd.to_numeric(
        frame[
            value_column
        ],
        errors="coerce",
    )


    result = pd.DataFrame(
        {
            "time":
                time,

            "obs_flow_cms":
                flow,
        }
    )


    result = result.loc[
        result[
            "time"
        ].notna()
        &
        np.isfinite(
            result[
                "obs_flow_cms"
            ]
        )
    ].copy()


    if result[
        "time"
    ].duplicated().any():

        duplicate = (
            result.loc[
                result[
                    "time"
                ].duplicated(
                    keep=False
                ),
                "time",
            ]
            .astype(
                str
            )
            .tolist()
        )

        raise CalibrationObjectiveError(
            "Observation timestamps are duplicated: "
            f"{duplicate[:10]}"
        )


    result = (
        result
        .sort_values(
            "time"
        )
        .reset_index(
            drop=True
        )
    )


    if len(
        result
    ) < 2:

        raise CalibrationObjectiveError(
            "At least two usable observations are required."
        )


    return result


def extract_routed_flow(
    routing_path: str | Path,
    *,
    feature_id: int,
) -> pd.DataFrame:
    """
    Extract one routed discharge series from a t-route NetCDF.

    The historical calibration contract requires exactly one occurrence
    of the requested routing feature.
    """

    target = (
        Path(
            routing_path
        )
        .expanduser()
        .resolve()
    )

    if not target.is_file():

        raise CalibrationObjectiveError(
            f"Routing NetCDF does not exist: {target}"
        )


    with xr.open_dataset(
        target,
        decode_times=False,
    ) as dataset:

        for variable in (
            "feature_id",
            "time",
            "flow",
        ):

            if variable not in dataset.variables:

                raise CalibrationObjectiveError(
                    f"Routing output lacks variable {variable!r}."
                )


        feature_ids = np.asarray(
            dataset[
                "feature_id"
            ].values,
            dtype=np.int64,
        ).reshape(
            -1
        )

        indices = np.flatnonzero(
            feature_ids
            ==
            int(
                feature_id
            )
        )


        if len(
            indices
        ) != 1:

            raise CalibrationObjectiveError(
                f"Expected routing feature {feature_id} exactly once; "
                f"found {len(indices)}."
            )


        time_variable = dataset[
            "time"
        ]

        raw_time = np.asarray(
            time_variable.values,
            dtype=float,
        ).reshape(
            -1
        )

        units = str(
            time_variable.attrs.get(
                "units",
                "",
            )
        )


        match = re.fullmatch(
            r"\s*seconds\s+since\s+(.+?)\s*",
            units,
            flags=re.IGNORECASE,
        )


        if match is None:

            raise CalibrationObjectiveError(
                f"Unexpected routing time units: {units!r}"
            )


        origin = _utc_timestamp(
            match.group(
                1
            ),
            name="routing time origin",
        )


        timestamps = (
            origin
            +
            pd.to_timedelta(
                raw_time,
                unit="s",
            )
        )


        routed_flow = np.asarray(
            dataset[
                "flow"
            ].isel(
                feature_id=int(
                    indices[
                        0
                    ]
                )
            ).values,
            dtype=float,
        ).reshape(
            -1
        )


    if (
        len(
            timestamps
        )
        !=
        len(
            routed_flow
        )
    ):

        raise CalibrationObjectiveError(
            "Routing time and flow lengths differ."
        )


    result = pd.DataFrame(
        {
            "time":
                pd.DatetimeIndex(
                    timestamps
                ),

            "sim_flow_cms":
                routed_flow,
        }
    )


    if result[
        "time"
    ].duplicated().any():

        raise CalibrationObjectiveError(
            "Routing timestamps are duplicated."
        )


    return (
        result
        .sort_values(
            "time"
        )
        .reset_index(
            drop=True
        )
    )


def align_exact_timestamp(
    simulation: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    calibration_start: Any,
    calibration_end_exclusive: Any,
) -> pd.DataFrame:
    """
    Historical joint-DDS alignment contract:

      1. exact timestamp inner merge
      2. one-to-one validation
      3. start inclusive
      4. end exclusive
      5. finite pairs only

    There is deliberately no nearest-neighbor matching,
    temporal averaging, or interpolation.
    """

    required_simulation = {
        "time",
        "sim_flow_cms",
    }

    required_observations = {
        "time",
        "obs_flow_cms",
    }


    if not required_simulation.issubset(
        simulation.columns
    ):

        raise CalibrationObjectiveError(
            "Simulation frame lacks required columns."
        )


    if not required_observations.issubset(
        observations.columns
    ):

        raise CalibrationObjectiveError(
            "Observation frame lacks required columns."
        )


    simulation = simulation.copy()
    observations = observations.copy()


    simulation[
        "time"
    ] = pd.to_datetime(
        simulation[
            "time"
        ],
        errors="coerce",
        utc=True,
    )

    observations[
        "time"
    ] = pd.to_datetime(
        observations[
            "time"
        ],
        errors="coerce",
        utc=True,
    )


    if simulation[
        "time"
    ].duplicated().any():

        raise CalibrationObjectiveError(
            "Simulation timestamps are duplicated."
        )


    if observations[
        "time"
    ].duplicated().any():

        raise CalibrationObjectiveError(
            "Observation timestamps are duplicated."
        )


    aligned = simulation.merge(
        observations,
        on="time",
        how="inner",
        validate="one_to_one",
    )


    start = _utc_timestamp(
        calibration_start,
        name="calibration_start",
    )

    end = _utc_timestamp(
        calibration_end_exclusive,
        name="calibration_end_exclusive",
    )


    if end <= start:

        raise CalibrationObjectiveError(
            "Calibration end must be after calibration start."
        )


    aligned = aligned.loc[
        (
            aligned[
                "time"
            ]
            >= start
        )
        &
        (
            aligned[
                "time"
            ]
            < end
        )
    ].copy()


    aligned = aligned.loc[
        np.isfinite(
            aligned[
                "obs_flow_cms"
            ]
        )
        &
        np.isfinite(
            aligned[
                "sim_flow_cms"
            ]
        )
    ].copy()


    return (
        aligned
        .sort_values(
            "time"
        )
        .reset_index(
            drop=True
        )
    )


def evaluate_aligned(
    aligned: pd.DataFrame,
    *,
    calibration_start: Any,
    calibration_end_exclusive: Any,
    expected_pairs: int | None = None,
) -> CalibrationObjectiveResult:

    if len(
        aligned
    ) < 2:

        raise CalibrationObjectiveError(
            "At least two exact timestamp pairs are required."
        )


    if (
        expected_pairs is not None
        and
        len(
            aligned
        )
        != int(
            expected_pairs
        )
    ):

        raise CalibrationObjectiveError(
            f"Expected {expected_pairs} exact calibration pairs; "
            f"found {len(aligned)}."
        )


    obs = aligned[
        "obs_flow_cms"
    ].to_numpy(
        dtype=float
    )

    sim = aligned[
        "sim_flow_cms"
    ].to_numpy(
        dtype=float
    )


    obs_mean = float(
        np.mean(
            obs
        )
    )

    sim_mean = float(
        np.mean(
            sim
        )
    )

    obs_std = float(
        np.std(
            obs,
            ddof=0,
        )
    )

    sim_std = float(
        np.std(
            sim,
            ddof=0,
        )
    )


    if obs_mean == 0.0:

        raise CalibrationObjectiveError(
            "Observed-flow mean is zero."
        )


    if obs_std <= 0.0:

        raise CalibrationObjectiveError(
            "Observed-flow standard deviation is not positive."
        )


    if sim_std <= 0.0:

        raise CalibrationObjectiveError(
            "Simulated-flow standard deviation is not positive."
        )


    corr = float(
        correlation(
            obs,
            sim,
        )
    )

    alpha = float(
        sim_std
        /
        obs_std
    )

    beta = float(
        sim_mean
        /
        obs_mean
    )

    kge_value = float(
        kge(
            obs,
            sim,
        )
    )

    nse_value = float(
        nse(
            obs,
            sim,
        )
    )

    rmse_value = float(
        rmse(
            obs,
            sim,
        )
    )

    pbias_value = float(
        pbias(
            obs,
            sim,
        )
    )

    volume_ratio = float(
        np.sum(
            sim
        )
        /
        np.sum(
            obs
        )
    )

    peak_ratio = float(
        np.max(
            sim
        )
        /
        np.max(
            obs
        )
    )

    objective = float(
        1.0
        -
        kge_value
    )


    values = np.asarray(
        [
            corr,
            alpha,
            beta,
            kge_value,
            nse_value,
            rmse_value,
            pbias_value,
            volume_ratio,
            peak_ratio,
            objective,
        ],
        dtype=float,
    )


    if not np.isfinite(
        values
    ).all():

        raise CalibrationObjectiveError(
            "Calibration metrics contain nonfinite values."
        )


    start = _utc_timestamp(
        calibration_start,
        name="calibration_start",
    )

    end = _utc_timestamp(
        calibration_end_exclusive,
        name="calibration_end_exclusive",
    )


    return CalibrationObjectiveResult(
        paired_count=int(
            len(
                aligned
            )
        ),

        calibration_start_utc=(
            start.isoformat()
        ),

        calibration_end_exclusive_utc=(
            end.isoformat()
        ),

        correlation=corr,

        nse=nse_value,

        kge_2009=kge_value,

        kge_alpha=alpha,

        kge_beta=beta,

        rmse_cms=rmse_value,

        pbias_percent=pbias_value,

        volume_ratio=volume_ratio,

        peak_ratio=peak_ratio,

        objective_1_minus_kge=(
            objective
        ),
    )


def evaluate_routing_objective(
    *,
    routing_path: str | Path,
    feature_id: int,
    observation_path: str | Path,
    calibration_start: Any,
    calibration_end_exclusive: Any,
    expected_pairs: int | None = None,
) -> CalibrationObjectiveResult:

    simulation = extract_routed_flow(
        routing_path,
        feature_id=feature_id,
    )

    observations = load_observations(
        observation_path
    )

    aligned = align_exact_timestamp(
        simulation,
        observations,
        calibration_start=(
            calibration_start
        ),
        calibration_end_exclusive=(
            calibration_end_exclusive
        ),
    )

    return evaluate_aligned(
        aligned,
        calibration_start=(
            calibration_start
        ),
        calibration_end_exclusive=(
            calibration_end_exclusive
        ),
        expected_pairs=(
            expected_pairs
        ),
    )


def objective_result_to_dict(
    result: CalibrationObjectiveResult,
) -> dict[str, Any]:

    return asdict(
        result
    )
