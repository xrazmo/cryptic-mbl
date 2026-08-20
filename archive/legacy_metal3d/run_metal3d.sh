#!/bin/bash
# Wrapper for Metal3D using conda run (with proper argument quoting)

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
METAL3D_DIR="$SCRIPT_DIR/assets/metal-site-prediction/Metal3D"
METAL3D_ENV="metal3d"

# Verify Metal3D directory exists
if [ ! -d "$METAL3D_DIR" ]; then
    echo "Error: Metal3D directory not found at $METAL3D_DIR"
    exit 1
fi

# Function to convert relative path to absolute
to_absolute() {
    if [[ "$1" == /* ]]; then
        echo "$1"
    else
        echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    fi
}

# Parse arguments and convert paths to absolute
declare -a args
skip_next=false

for i in "$@"; do
    if [ "$skip_next" = true ]; then
        args+=("$(to_absolute "$i")")
        skip_next=false
    elif [[ "$i" == "--pdb" ]] || [[ "$i" == "--probefile" ]] || [[ "$i" == "--probefile_out" ]]; then
        args+=("$i")
        skip_next=true
    else
        args+=("$i")
    fi
done

# Build properly quoted command line
# This preserves argument boundaries when passed through bash -c
cmd="cd '$METAL3D_DIR' && python metal3d.py"
for arg in "${args[@]}"; do
    # Escape single quotes
    arg_escaped="${arg//\'/\'\"\'\"\'}"
    cmd="$cmd '$arg_escaped'"
done

# Add --softexit flag to skip interactive prompt
cmd="$cmd --softexit"

# Run metal3d.py in the metal3d environment via conda run
conda run -n "$METAL3D_ENV" bash -c "$cmd"
