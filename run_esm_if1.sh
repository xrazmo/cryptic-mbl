#!/bin/bash
# Wrapper for ESM-IF1's frozen structure encoder using conda run.
# Same pattern as run_metal3d.sh -- the esm-if1 conda environment exists
# because the main cryptic-mbl environment's torch (2.13.0+cu132) has no
# prebuilt torch_scatter wheel and building it from source fails (system
# CUDA toolkit 12.9 vs torch's CUDA 13.2). esm-if1 is pinned to
# torch 2.9.1+cu128 with matching prebuilt torch_scatter/torch_cluster/
# torch_geometric wheels, and biotite<1.0 (esm.inverse_folding.util
# imports the now-renamed filter_backbone).

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
WORKER="$SCRIPT_DIR/scripts/esm_if1_worker.py"
ESM_IF1_ENV="esm-if1"

if [ ! -f "$WORKER" ]; then
    echo "Error: worker script not found at $WORKER"
    exit 1
fi

to_absolute() {
    if [[ "$1" == /* ]]; then
        echo "$1"
    else
        echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    fi
}

declare -a args
skip_next=false
for i in "$@"; do
    if [ "$skip_next" = true ]; then
        args+=("$(to_absolute "$i")")
        skip_next=false
    elif [[ "$i" == "--pdb" ]] || [[ "$i" == "--out" ]]; then
        args+=("$i")
        skip_next=true
    else
        args+=("$i")
    fi
done

cmd="cd '$SCRIPT_DIR' && python '$WORKER'"
for arg in "${args[@]}"; do
    arg_escaped="${arg//\'/\'\"\'\"\'}"
    cmd="$cmd '$arg_escaped'"
done

conda run -n "$ESM_IF1_ENV" bash -c "$cmd"
