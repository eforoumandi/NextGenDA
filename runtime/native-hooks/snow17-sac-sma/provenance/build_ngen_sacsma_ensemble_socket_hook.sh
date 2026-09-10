#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
)"

BUILDER_IMAGE="${NGIAB_DA_BUILDER_IMAGE:-ngiab-da-builder:validated}"

if [ "$#" -ne 1 ]; then

    echo "Usage: $0 OUTPUT_LIBRARY" >&2
    exit 2

fi

OUTPUT_LIBRARY="$(
    realpath -m "$1"
)"

OUTPUT_PARENT="$(
    dirname "$OUTPUT_LIBRARY"
)"

OUTPUT_NAME="$(
    basename "$OUTPUT_LIBRARY"
)"

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"


for required in \
    "$REPOSITORY/native/ngiab_da_step_hook_api_v2.h" \
    "$REPOSITORY/native/ngiab_da_sacsma_ensemble_socket_hook.cpp"
do

    if [ ! -f "$required" ]; then

        echo "ERROR: Required source is missing: $required" >&2
        exit 1

    fi

done


mkdir -p "$OUTPUT_PARENT"


docker run --rm \
    --mount \
    "type=bind,source=$REPOSITORY,target=/workspace/repository,readonly" \
    --mount \
    "type=bind,source=$OUTPUT_PARENT,target=/workspace/output" \
    --env "OUTPUT_NAME=$OUTPUT_NAME" \
    --env "HOST_UID=$HOST_UID" \
    --env "HOST_GID=$HOST_GID" \
    --entrypoint bash \
    "$BUILDER_IMAGE" \
    -lc '
        set -euo pipefail

        test -f \
            /boost_1_86_0/boost/property_tree/json_parser.hpp

        g++ \
            -std=c++17 \
            -O2 \
            -fPIC \
            -shared \
            -I /workspace/repository/native \
            -I /boost_1_86_0 \
            /workspace/repository/native/ngiab_da_sacsma_ensemble_socket_hook.cpp \
            -o "/workspace/output/$OUTPUT_NAME"

        chown \
            "$HOST_UID:$HOST_GID" \
            "/workspace/output/$OUTPUT_NAME"
    '


echo "sacsma_ensemble_socket_hook_library=$OUTPUT_LIBRARY"

echo "NGEN_SACSMA_ENSEMBLE_SOCKET_HOOK_BUILD=PASS"
