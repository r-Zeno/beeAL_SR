#!/bin/bash

INPUT_DIR="~/beeAL_sim/input"
OUTPUT_DIR="~/beeAL_sim/simulations_output"
LATEST_DIR="~/beeAL_sim/curr_results"

mkdir -p "$(eval echo "$INPUT_DIR")"
mkdir -p "$(eval echo "$OUTPUT_DIR")"
mkdir -p "$(eval echo "$LATEST_DIR")"

EVAL_INPUT_DIR=$(eval echo "$INPUT_DIR")
EVAL_OUTPUT_DIR=$(eval echo "$OUTPUT_DIR")
EVAL_LATEST_DIR=$(eval echo "$LATEST_DIR")

rm -f "${EVAL_LATEST_DIR}"/*
echo "Previous results cleared"

echo "Starting Simulations..."
docker run \
  --gpus all \
  --rm \
  -v "${EVAL_INPUT_DIR}/parameters.json:/app/parameters.json" \
  -v "${EVAL_OUTPUT_DIR}:/simulations" \
  beeal-simulator:latest
echo "Simulations Finished"

echo "Looking for results..."
LATEST_SIM_DIR=$(find "${EVAL_OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

if [ -d "$LATEST_SIM_DIR" ]; then
  echo "latest simulation directory: $LATEST_SIM_DIR"

  if ! command -v jq &> /dev/null
  then
    echo "ERROR: jq is not installed"
    exit 1
  fi

  cp "$LATEST_SIM_DIR"/sim_settings.json "${EVAL_LATEST_DIR}/"
  cp "$LATEST_SIM_DIR"/plot_names.json "${EVAL_LATEST_DIR}/"
  cp "$LATEST_SIM_DIR"/mean_vp_dist_x_noiselvls_*.npy "${EVAL_LATEST_DIR}/"
  cp "$LATEST_SIM_DIR"/single_vp_dist_values_*.npy "${EVAL_LATEST_DIR}/"

    PLOT_NAMES_JSON="${LATEST_SIM_DIR}/plot_names.json"

    if [ -f "$PLOT_NAMES_JSON" ]; then
      echo "Copying plots listed in plot_names.json..."

      jq -r '.[] | .[]' "$PLOT_NAMES_JSON" | while read -r plot_filename; do

        if [ -f "${LATEST_SIM_DIR}/${plot_filename}" ]; then
          cp "${LATEST_SIM_DIR}/${plot_filename}" "${EVAL_LATEST_DIR}/"
          echo "Copied ${plot_filename}"
        else
          echo "WARNING: Could not find plot file ${plot_filename}"
        fi

      done

    else
      echo "WARNING: plot_names.json not found. Cannot copy plot files."
    fi

    echo "Latest files are in ${EVAL_LATEST_DIR}"
  else
    echo "WARNING: Could not find any new simulation output directories."
  fi
  