"""Versioned Unix-domain socket protocol for sequential NGen ensembles."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import argparse
import copy
import json
import math
from pathlib import Path
import socket
import struct
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from ngiab_da.integration.ancestry_payload import (
    ANCESTRY_PAYLOAD_KEY,
    AncestryPayloadError,
    normalize_ancestry_payload,
)


PROTOCOL_VERSION = 1
_FRAME_HEADER = struct.Struct("!Q")
MAX_FRAME_BYTES = 64 * 1024 * 1024


class SequentialEnsembleProtocolError(RuntimeError):
    """Raised when the member/sidecar protocol contract is violated."""


class SequentialEnsembleDuplicateError(
    SequentialEnsembleProtocolError
):
    """Raised for a conflicting duplicate member submission."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def encode_frame(payload: Mapping[str, Any]) -> bytes:
    """Encode one deterministic length-prefixed UTF-8 JSON frame."""

    encoded = _canonical_json(payload).encode("utf-8")
    if not encoded or len(encoded) > MAX_FRAME_BYTES:
        raise SequentialEnsembleProtocolError(
            f"Invalid frame size: {len(encoded)}."
        )
    return _FRAME_HEADER.pack(len(encoded)) + encoded


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise SequentialEnsembleProtocolError(
                "Socket closed before the complete frame arrived."
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_frame(connection: socket.socket) -> dict[str, Any]:
    """Receive and decode one framed request or response."""

    header = _recv_exact(connection, _FRAME_HEADER.size)
    (size,) = _FRAME_HEADER.unpack(header)
    if size <= 0 or size > MAX_FRAME_BYTES:
        raise SequentialEnsembleProtocolError(
            f"Invalid incoming frame size: {size}."
        )
    encoded = _recv_exact(connection, size)
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SequentialEnsembleProtocolError(
            "Incoming frame is not valid UTF-8 JSON."
        ) from error
    if not isinstance(value, dict):
        raise SequentialEnsembleProtocolError(
            "Protocol payload must be a JSON object."
        )
    return value


def send_frame(
    connection: socket.socket,
    payload: Mapping[str, Any],
) -> None:
    """Send one complete framed request or response."""

    connection.sendall(encode_frame(payload))


def _string(
    payload: Mapping[str, Any],
    field_name: str,
) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise SequentialEnsembleProtocolError(
            f"{field_name} must be a non-empty string."
        )
    return value


def _integer(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    minimum: int | None = None,
) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool):
        raise SequentialEnsembleProtocolError(
            f"{field_name} must be an integer."
        )
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise SequentialEnsembleProtocolError(
            f"{field_name} must be an integer."
        ) from error
    if str(result) != str(value) and not isinstance(value, int):
        try:
            if float(value) != result:
                raise ValueError
        except (TypeError, ValueError):
            raise SequentialEnsembleProtocolError(
                f"{field_name} must be an integer."
            )
    if minimum is not None and result < minimum:
        raise SequentialEnsembleProtocolError(
            f"{field_name} must be at least {minimum}."
        )
    return result


def _number(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    minimum: float | None = None,
) -> float:
    value = payload.get(field_name)
    if isinstance(value, bool):
        raise SequentialEnsembleProtocolError(
            f"{field_name} must be numeric."
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SequentialEnsembleProtocolError(
            f"{field_name} must be numeric."
        ) from error
    if not math.isfinite(result):
        raise SequentialEnsembleProtocolError(
            f"{field_name} must be finite."
        )
    if minimum is not None and result < minimum:
        raise SequentialEnsembleProtocolError(
            f"{field_name} must be at least {minimum}."
        )
    return result



ROUTING_QLAT_ONLY_REQUEST_KIND = "routing_qlat_only"
SACSMA_STATE_AND_QLAT_REQUEST_KIND = "sacsma_state_and_qlat"


def _validate_sacsma_member_request(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one live SAC-SMA state + qlat request."""

    if _integer(
        payload,
        "protocol_version",
    ) != PROTOCOL_VERSION:

        raise SequentialEnsembleProtocolError(
            "Unsupported member protocol version."
        )

    normalized: dict[str, Any] = {
        "protocol_version":
            PROTOCOL_VERSION,

        "request_kind":
            SACSMA_STATE_AND_QLAT_REQUEST_KIND,

        "run_id":
            _string(
                payload,
                "run_id",
            ),

        "member_id":
            _string(
                payload,
                "member_id",
            ),

        "generation":
            _integer(
                payload,
                "generation",
                minimum=0,
            ),

        "cycle_index":
            _integer(
                payload,
                "cycle_index",
                minimum=0,
            ),

        "analysis_epoch_seconds":
            _integer(
                payload,
                "analysis_epoch_seconds",
            ),
    }

    states = payload.get(
        "catchment_states"
    )

    if (
        not isinstance(
            states,
            list,
        )
        or not states
    ):

        raise SequentialEnsembleProtocolError(
            "catchment_states must be a non-empty array."
        )

    state_names = (
        "uztwc",
        "uzfwc",
        "lztwc",
        "lzfsc",
        "lzfpc",
        "adimc",
    )

    normalized_states: list[
        dict[str, Any]
    ] = []

    state_keys: set[
        tuple[str, int]
    ] = set()

    for raw_state in states:

        if not isinstance(
            raw_state,
            Mapping,
        ):

            raise SequentialEnsembleProtocolError(
                "Each SAC-SMA catchment state must be an object."
            )

        state: dict[str, Any] = {
            "catchment_id":
                _string(
                    raw_state,
                    "catchment_id",
                ),

            "module_index":
                _integer(
                    raw_state,
                    "module_index",
                    minimum=0,
                ),
        }

        for state_name in state_names:

            state[
                state_name
            ] = _number(
                raw_state,
                state_name,
                minimum=0.0,
            )

        if ANCESTRY_PAYLOAD_KEY in raw_state:

            try:

                state[
                    ANCESTRY_PAYLOAD_KEY
                ] = normalize_ancestry_payload(
                    raw_state[
                        ANCESTRY_PAYLOAD_KEY
                    ]
                )

            except AncestryPayloadError as error:

                raise SequentialEnsembleProtocolError(
                    "Invalid optional complete-ancestry payload "
                    f"for catchment {state['catchment_id']!r}: "
                    f"{error}"
                ) from error


        key = (
            state[
                "catchment_id"
            ],
            state[
                "module_index"
            ],
        )

        if key in state_keys:

            raise SequentialEnsembleProtocolError(
                f"Duplicate SAC-SMA catchment state: {key!r}."
            )

        state_keys.add(
            key
        )

        normalized_states.append(
            state
        )

    qlat = payload.get(
        "catchment_qlat"
    )

    if (
        not isinstance(
            qlat,
            list,
        )
        or not qlat
    ):

        raise SequentialEnsembleProtocolError(
            "catchment_qlat must be a non-empty array."
        )

    normalized_qlat: list[
        dict[str, Any]
    ] = []

    qlat_ids: set[str] = set()

    for raw_qlat in qlat:

        if not isinstance(
            raw_qlat,
            Mapping,
        ):

            raise SequentialEnsembleProtocolError(
                "Each catchment qlat entry must be an object."
            )

        catchment_id = _string(
            raw_qlat,
            "catchment_id",
        )

        if catchment_id in qlat_ids:

            raise SequentialEnsembleProtocolError(
                f"Duplicate catchment qlat: {catchment_id!r}."
            )

        qlat_ids.add(
            catchment_id
        )

        available_value = raw_qlat.get(
            "available"
        )

        if isinstance(
            available_value,
            str,
        ):

            if (
                available_value.lower()
                not in {
                    "true",
                    "false",
                }
            ):

                raise SequentialEnsembleProtocolError(
                    "catchment_qlat.available must be boolean."
                )

            available = (
                available_value.lower()
                == "true"
            )

        elif isinstance(
            available_value,
            bool,
        ):

            available = available_value

        else:

            raise SequentialEnsembleProtocolError(
                "catchment_qlat.available must be boolean."
            )

        units = _string(
            raw_qlat,
            "units",
        )

        if units != "m":

            raise SequentialEnsembleProtocolError(
                "SAC-SMA qlat values must use units='m'."
            )

        source_variable = _string(
            raw_qlat,
            "source_variable",
        )

        if source_variable != "tci":

            raise SequentialEnsembleProtocolError(
                "SAC-SMA qlat source_variable must be 'tci'."
            )

        normalized_qlat.append(
            {
                "catchment_id":
                    catchment_id,

                "value":
                    _number(
                        raw_qlat,
                        "value",
                        minimum=0.0,
                    ),

                "units":
                    units,

                "source_variable":
                    source_variable,

                "available":
                    available,
            }
        )

    if (
        qlat_ids
        != {
            state[
                "catchment_id"
            ]
            for state in (
                normalized_states
            )
        }
    ):

        raise SequentialEnsembleProtocolError(
            "SAC-SMA catchment_qlat IDs must equal "
            "catchment-state IDs."
        )

    normalized[
        "catchment_states"
    ] = sorted(
        normalized_states,
        key=lambda item: (
            item[
                "catchment_id"
            ],
            item[
                "module_index"
            ],
        ),
    )

    normalized[
        "catchment_qlat"
    ] = sorted(
        normalized_qlat,
        key=lambda item:
            item[
                "catchment_id"
            ],
    )

    return normalized


def validate_member_request(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one explicit SAC-SMA-state or routing-only payload."""

    request_kind = _string(
        payload,
        "request_kind",
    )

    if request_kind == SACSMA_STATE_AND_QLAT_REQUEST_KIND:
        return _validate_sacsma_member_request(payload)

    if request_kind != ROUTING_QLAT_ONLY_REQUEST_KIND:
        raise SequentialEnsembleProtocolError(
            f"Unsupported request_kind: {request_kind!r}."
        )

    normalized: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "request_kind": ROUTING_QLAT_ONLY_REQUEST_KIND,
        "run_id": _string(payload, "run_id"),
        "member_id": _string(payload, "member_id"),
        "generation": _integer(
            payload,
            "generation",
            minimum=0,
        ),
        "cycle_index": _integer(
            payload,
            "cycle_index",
            minimum=0,
        ),
        "analysis_epoch_seconds": _integer(
            payload,
            "analysis_epoch_seconds",
        ),
    }

    states = payload.get("catchment_states", [])
    if not isinstance(states, list):
        raise SequentialEnsembleProtocolError(
            "catchment_states must be an array when supplied."
        )
    if states:
        raise SequentialEnsembleProtocolError(
            "routing_qlat_only requests must not contain "
            "hydrologic model states."
        )

    qlat = payload.get("catchment_qlat")
    if not isinstance(qlat, list) or not qlat:
        raise SequentialEnsembleProtocolError(
            "catchment_qlat must be a non-empty array."
        )

    normalized_qlat: list[dict[str, Any]] = []
    qlat_ids: set[str] = set()

    for raw_qlat in qlat:
        if not isinstance(raw_qlat, Mapping):
            raise SequentialEnsembleProtocolError(
                "Each catchment qlat entry must be an object."
            )

        catchment_id = _string(
            raw_qlat,
            "catchment_id",
        )
        if catchment_id in qlat_ids:
            raise SequentialEnsembleProtocolError(
                f"Duplicate catchment qlat: {catchment_id!r}."
            )
        qlat_ids.add(catchment_id)

        available_value = raw_qlat.get("available")
        if isinstance(available_value, str):
            if available_value.lower() not in {
                "true",
                "false",
            }:
                raise SequentialEnsembleProtocolError(
                    "catchment_qlat.available must be boolean."
                )
            available = available_value.lower() == "true"
        elif isinstance(available_value, bool):
            available = available_value
        else:
            raise SequentialEnsembleProtocolError(
                "catchment_qlat.available must be boolean."
            )

        units = _string(raw_qlat, "units")
        if units != "m":
            raise SequentialEnsembleProtocolError(
                "routing_qlat_only values must use units='m'."
            )

        source_variable = _string(
            raw_qlat,
            "source_variable",
        )
        if source_variable != "tci":
            raise SequentialEnsembleProtocolError(
                "routing_qlat_only source_variable must be 'tci'."
            )

        normalized_qlat.append(
            {
                "catchment_id": catchment_id,
                "value": _number(
                    raw_qlat,
                    "value",
                    minimum=0.0,
                ),
                "units": units,
                "source_variable": source_variable,
                "available": available,
            }
        )

    normalized["catchment_states"] = []
    normalized["catchment_qlat"] = sorted(
        normalized_qlat,
        key=lambda item: item["catchment_id"],
    )
    return normalized




def _forecast_only_response(
    request: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": request["run_id"],
        "member_id": request["member_id"],
        "generation": request["generation"],
        "cycle_index": request["cycle_index"],
        "analysis_epoch_seconds": request[
            "analysis_epoch_seconds"
        ],
        "status": "forecast_only",
        "reason": reason,
    }


AnalysisCallback = Callable[
    [tuple[dict[str, Any], ...], tuple[str, ...]],
    Mapping[str, Sequence[Mapping[str, Any]]],
]


def identity_analysis(
    requests: tuple[dict[str, Any], ...],
    member_ids: tuple[str, ...],
) -> Mapping[str, Sequence[Mapping[str, Any]]]:
    """Return member states unchanged in explicit member order."""

    return {
        member_id: copy.deepcopy(request["catchment_states"])
        for member_id, request in zip(member_ids, requests)
    }


@dataclass
class _CycleState:
    requests: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    request_json: dict[str, str] = field(default_factory=dict)
    responses: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    analyzing: bool = False
    timed_out: bool = False
    delivered_members: set[str] = field(
        default_factory=set
    )
    completed: bool = False


class SequentialEnsembleBarrier:
    """Thread-safe deterministic barrier for one sequential ensemble."""

    # Keep a small retry/idempotency horizon without retaining the
    # complete request/response history for an entire long simulation.
    #
    # Each SAC-SMA cycle contains 20 member request trees carrying
    # catchment states and qlat, canonical JSON copies of those
    # requests, and response state trees.  Retaining every completed
    # cycle causes process memory to grow with simulation length.
    _COMPLETED_CYCLE_RETENTION = 8

    def __init__(
        self,
        member_ids: Sequence[str],
        *,
        timeout_seconds: float,
        analyzer: AnalysisCallback = identity_analysis,
        event_callback: Callable[
            [Mapping[str, Any]],
            None,
        ]
        | None = None,
    ) -> None:
        resolved = tuple(str(value) for value in member_ids)
        if not resolved or any(not value for value in resolved):
            raise ValueError(
                "member_ids must contain non-empty strings."
            )
        if len(set(resolved)) != len(resolved):
            raise ValueError("member_ids must be unique.")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be finite and positive."
            )
        if not callable(analyzer):
            raise TypeError("analyzer must be callable.")

        self._member_ids = resolved
        self._member_set = frozenset(resolved)
        self._timeout_seconds = float(timeout_seconds)
        self._analyzer = analyzer
        self._event_callback = event_callback
        self._condition = threading.Condition()
        self._cycles: dict[
            tuple[str, int, int, int],
            _CycleState,
        ] = {}

    @property
    def member_ids(self) -> tuple[str, ...]:
        return self._member_ids

    def _emit(self, event: Mapping[str, Any]) -> None:
        if self._event_callback is not None:
            self._event_callback(dict(event))

    def _analyze(
        self,
        key: tuple[str, int, int, int],
        cycle: _CycleState,
    ) -> None:
        """Run one analyzer without monopolizing the barrier condition."""

        ordered_requests = tuple(
            cycle.requests[member_id]
            for member_id in self._member_ids
        )

        request_kinds = {
            str(
                request[
                    "request_kind"
                ]
            )
            for request in ordered_requests
        }

        if len(request_kinds) != 1:
            raise SequentialEnsembleProtocolError(
                "All synchronized members must use one request_kind."
            )

        request_kind = next(iter(request_kinds))

        response_state_key = (
            "sacsma_analysis_states"
            if request_kind
            == SACSMA_STATE_AND_QLAT_REQUEST_KIND
            else None
        )

        analyzed = self._analyzer(
            ordered_requests,
            self._member_ids,
        )
        if set(analyzed) != self._member_set:
            raise SequentialEnsembleProtocolError(
                "Analyzer result must cover every member exactly once."
            )

        prepared: dict[str, dict[str, Any]] = {}
        for member_id, request in zip(
            self._member_ids,
            ordered_requests,
        ):
            states = list(analyzed[member_id])

            response: dict[str, Any] = {
                "protocol_version": PROTOCOL_VERSION,
                "run_id": request["run_id"],
                "member_id": member_id,
                "generation": request["generation"],
                "cycle_index": request["cycle_index"],
                "analysis_epoch_seconds": request[
                    "analysis_epoch_seconds"
                ],
                "status": "analysis",
            }

            if response_state_key is not None:
                response[
                    response_state_key
                ] = states

            elif states:
                raise SequentialEnsembleProtocolError(
                    "routing_qlat_only analyzer result must not "
                    "contain hydrologic model states."
                )

            prepared[
                member_id
            ] = response

        # The expensive/blocking analyzer ran outside this condition.
        # Commit responses atomically only after it returns.
        with self._condition:
            if cycle.timed_out:
                return
            cycle.responses.update(prepared)
            self._emit(
                {
                    "event": "barrier_release",
                    "run_id": key[0],
                    "generation": key[1],
                    "cycle_index": key[2],
                    "analysis_epoch_seconds": key[3],
                    "member_ids": list(self._member_ids),
                    "status": "analysis",
                }
            )

    def _prune_completed_cycles_locked(self) -> None:
        """Bound retained completed cycles while holding the condition."""

        completed_keys = [
            key
            for key, value in self._cycles.items()
            if value.completed
        ]

        excess = (
            len(completed_keys)
            - self._COMPLETED_CYCLE_RETENTION
        )

        if excess <= 0:
            return

        for old_key in completed_keys[:excess]:
            self._cycles.pop(
                old_key,
                None,
            )

    def submit(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Submit one member and block until analysis or timeout."""

        request = validate_member_request(payload)
        member_id = request["member_id"]
        if member_id not in self._member_set:
            raise SequentialEnsembleProtocolError(
                f"Unexpected member_id: {member_id!r}."
            )

        key = (
            request["run_id"],
            request["generation"],
            request["cycle_index"],
            request["analysis_epoch_seconds"],
        )
        canonical = _canonical_json(request)
        deadline = time.monotonic() + self._timeout_seconds
        should_analyze = False

        # Only barrier bookkeeping is protected by the global condition.
        # The analyzer itself runs after this block so another generation
        # can enter while the current generation is intentionally held.
        with self._condition:
            cycle = self._cycles.setdefault(key, _CycleState())
            existing = cycle.request_json.get(member_id)
            if existing is not None and existing != canonical:
                raise SequentialEnsembleDuplicateError(
                    "Conflicting duplicate request for "
                    f"{(key, member_id)!r}."
                )

            if existing is None:
                cycle.requests[member_id] = request
                cycle.request_json[member_id] = canonical
                self._emit(
                    {
                        "event": "member_arrival",
                        "run_id": key[0],
                        "generation": key[1],
                        "cycle_index": key[2],
                        "analysis_epoch_seconds": key[3],
                        "member_id": member_id,
                        "arrival_count": len(cycle.requests),
                    }
                )

            if cycle.timed_out and member_id not in cycle.responses:
                cycle.responses[member_id] = _forecast_only_response(
                    request,
                    reason="ensemble_barrier_timeout",
                )

            if (
                not cycle.timed_out
                and not cycle.responses
                and not cycle.analyzing
                and set(cycle.requests) == self._member_set
            ):
                cycle.analyzing = True
                should_analyze = True

        if should_analyze:
            analysis_error: Exception | None = None
            try:
                self._analyze(key, cycle)
            except Exception as error:
                analysis_error = error

            with self._condition:
                if analysis_error is not None:
                    cycle.timed_out = True
                    cycle.responses.clear()
                    reason = (
                        "ensemble_analysis_error:"
                        f"{type(analysis_error).__name__}"
                    )
                    for arrived_member, arrived_request in (
                        cycle.requests.items()
                    ):
                        cycle.responses[arrived_member] = (
                            _forecast_only_response(
                                arrived_request,
                                reason=reason,
                            )
                        )
                    self._emit(
                        {
                            "event": "barrier_release",
                            "run_id": key[0],
                            "generation": key[1],
                            "cycle_index": key[2],
                            "analysis_epoch_seconds": key[3],
                            "member_ids": list(self._member_ids),
                            "status": "forecast_only",
                            "reason": reason,
                            "error_type": (
                                type(analysis_error).__name__
                            ),
                            "error_message": str(analysis_error),
                        }
                    )

                cycle.analyzing = False
                self._condition.notify_all()

        with self._condition:
            while member_id not in cycle.responses:
                # Preserve the old semantic that analyzer execution itself
                # owns the barrier.  A deliberate PF hold has its own
                # bounded timeout and must not degrade into the ordinary
                # member-arrival timeout while analysis is in progress.
                if cycle.analyzing:
                    self._condition.wait()
                    continue

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if not cycle.responses:
                        cycle.timed_out = True
                        for arrived_member, arrived_request in (
                            cycle.requests.items()
                        ):
                            cycle.responses[arrived_member] = (
                                _forecast_only_response(
                                    arrived_request,
                                    reason=(
                                        "ensemble_barrier_timeout"
                                    ),
                                )
                            )
                        self._emit(
                            {
                                "event": "barrier_release",
                                "run_id": key[0],
                                "generation": key[1],
                                "cycle_index": key[2],
                                "analysis_epoch_seconds": key[3],
                                "member_ids": [
                                    member
                                    for member in self._member_ids
                                    if member in cycle.requests
                                ],
                                "status": "forecast_only",
                                "reason": (
                                    "ensemble_barrier_timeout"
                                ),
                            }
                        )
                        self._condition.notify_all()
                    elif member_id not in cycle.responses:
                        cycle.responses[member_id] = (
                            _forecast_only_response(
                                request,
                                reason=(
                                    "ensemble_barrier_timeout"
                                ),
                            )
                        )
                    break

                self._condition.wait(remaining)

            response = cycle.responses.get(member_id)
            if response is None:
                raise SequentialEnsembleProtocolError(
                    "Barrier completed without a member response."
                )

            result = copy.deepcopy(
                response
            )

            cycle.delivered_members.add(
                member_id
            )

            if (
                cycle.delivered_members
                == self._member_set
            ):
                cycle.completed = True
                self._prune_completed_cycles_locked()

            return result


class SequentialEnsembleSidecarServer:
    """Concurrent Unix-domain socket server for ensemble members."""

    def __init__(
        self,
        socket_path: str | Path,
        barrier: SequentialEnsembleBarrier,
        *,
        connection_timeout_seconds: float,
        max_requests: int | None = None,
        stop_path: str | Path | None = None,
        event_log: str | Path | None = None,
        event_callback: Callable[
            [Mapping[str, Any]],
            None,
        ]
        | None = None,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.barrier = barrier
        if (
            not math.isfinite(connection_timeout_seconds)
            or connection_timeout_seconds <= 0
        ):
            raise ValueError(
                "connection_timeout_seconds must be positive."
            )
        self.connection_timeout_seconds = float(
            connection_timeout_seconds
        )
        if max_requests is not None and max_requests <= 0:
            raise ValueError("max_requests must be positive.")
        self.max_requests = max_requests
        self.stop_path = (
            None
            if stop_path is None
            else Path(stop_path).expanduser().resolve()
        )
        self.event_log = (
            None if event_log is None else Path(event_log)
        )
        self.event_callback = event_callback
        self._event_lock = threading.Lock()
        self.ready_event = threading.Event()

    def append_event(self, event: Mapping[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(dict(event))
        if self.event_log is None:
            return
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        line = _canonical_json(event) + "\n"
        with self._event_lock:
            with self.event_log.open(
                "a",
                encoding="utf-8",
            ) as stream:
                stream.write(line)

    def _handle(self, connection: socket.socket) -> None:
        with connection:
            connection.settimeout(
                self.connection_timeout_seconds
            )
            try:
                request = receive_frame(connection)
                response = self.barrier.submit(request)
            except Exception as error:
                response = {
                    "protocol_version": PROTOCOL_VERSION,
                    "status": "error",
                    "error": (
                        f"{type(error).__name__}: {error}"
                    ),
                }
            send_frame(connection, response)
            self.append_event(
                {
                    "event": "response_sent",
                    "member_id": response.get("member_id"),
                    "cycle_index": response.get("cycle_index"),
                    "status": response.get("status"),
                }
            )

    def serve(self) -> None:
        self.ready_event.clear()
        self.socket_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        if self.socket_path.exists():
            self.socket_path.unlink()

        accepted = 0
        worker_count = max(4, len(self.barrier.member_ids) * 2)
        try:
            with socket.socket(
                socket.AF_UNIX,
                socket.SOCK_STREAM,
            ) as listener:
                listener.bind(str(self.socket_path))
                listener.listen(worker_count)
                if self.stop_path is not None:
                    listener.settimeout(0.2)
                self.ready_event.set()
                with ThreadPoolExecutor(
                    max_workers=worker_count
                ) as executor:
                    while (
                        (
                            self.max_requests is None
                            or accepted < self.max_requests
                        )
                        and (
                            self.stop_path is None
                            or not self.stop_path.exists()
                        )
                    ):
                        try:
                            connection, _ = listener.accept()
                        except socket.timeout:
                            continue
                        accepted += 1
                        executor.submit(self._handle, connection)
        finally:
            if self.socket_path.exists():
                self.socket_path.unlink()


def _parse_members(value: str) -> tuple[str, ...]:
    members = tuple(
        item.strip()
        for item in value.split(",")
        if item.strip()
    )
    if not members:
        raise argparse.ArgumentTypeError(
            "At least one member ID is required."
        )
    if len(set(members)) != len(members):
        raise argparse.ArgumentTypeError(
            "Member IDs must be unique."
        )
    return members


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the sequential NGen ensemble Unix-socket sidecar."
        )
    )
    parser.add_argument("--socket", required=True)
    parser.add_argument(
        "--members",
        required=True,
        type=_parse_members,
    )
    parser.add_argument(
        "--barrier-timeout",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--connection-timeout",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--max-requests",
        type=int,
    )
    parser.add_argument("--event-log")
    args = parser.parse_args(argv)

    event_log = (
        None if args.event_log is None else Path(args.event_log)
    )
    event_lock = threading.Lock()

    def emit(event: Mapping[str, Any]) -> None:
        if event_log is None:
            return
        event_log.parent.mkdir(parents=True, exist_ok=True)
        with event_lock:
            with event_log.open("a", encoding="utf-8") as stream:
                stream.write(_canonical_json(event) + "\n")

    barrier = SequentialEnsembleBarrier(
        args.members,
        timeout_seconds=args.barrier_timeout,
        event_callback=emit,
    )
    server = SequentialEnsembleSidecarServer(
        args.socket,
        barrier,
        connection_timeout_seconds=args.connection_timeout,
        max_requests=args.max_requests,
        event_callback=emit,
    )
    server.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

