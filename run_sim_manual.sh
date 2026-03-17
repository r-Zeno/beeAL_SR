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

echo "You are in manual mode, results will be saved in "$LATEST_DIR" and will have to be transferred individually!"
rm -rf "${EVAL_LATEST_DIR:?}"/*
echo "Previous results cleared from ${EVAL_LATEST_DIR}"

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
  echo "Latest simulation directory found: $LATEST_SIM_DIR"
  echo "Copying all results to staging directory..."

  cp -r "${LATEST_SIM_DIR}"/* "${EVAL_LATEST_DIR}/"
  
  echo "All current simulation files are now in ${EVAL_LATEST_DIR}."
else
  echo "WARNING: Could not find any new simulation output directories."
fi
