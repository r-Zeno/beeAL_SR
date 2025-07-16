#!/bin/bash

REMOTE_USER="zeno"
REMOTE_HOST="100.71.200.87"
REMOTE_PROJECT_PATH="beeAL_SR"
LOCAL_PARAMS_FILE="parameters.json"
DATE_TAG=$(date +"sim_%Y%m%d_%H%M%S")
LOCAL_OUTPUT_DIR="$DATE_TAG"

if [ ! -f "$LOCAL_PARAMS_FILE" ]; then
    echo "Error: Parameters file not found at '$LOCAL_PARAMS_FILE'"
    exit 1
fi

mkdir -p "$LOCAL_OUTPUT_DIR"
echo "Results will be saved in './${LOCAL_OUTPUT_DIR}/'"

echo "Uploading '$LOCAL_PARAMS_FILE' to remote worker..."
scp "$LOCAL_PARAMS_FILE" ${REMOTE_USER}@${REMOTE_HOST}:beeAL_sim/input/parameters.json
echo "Upload complete"

echo "Worker starting remote simulations..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "bash ~/${REMOTE_PROJECT_PATH}/run_sim.sh"
echo "Simulations finished"

echo "Downloading results from worker..."
scp ${REMOTE_USER}@${REMOTE_HOST}:beeAL_sim/curr_results/* "$LOCAL_OUTPUT_DIR/"
echo "Download complete"

echo "All done. Your results are in the './${LOCAL_OUTPUT_DIR}/' directory."
