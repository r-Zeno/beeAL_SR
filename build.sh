#!/bin/bash
set -e

PYTHON_INCLUDE=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
NUMPY_INCLUDE=$(python3 -c "import numpy; print(numpy.get_include())")

PYBIND11_INCLUDE="pybind11/include"

MODULE_NAME="pdv_cuda"

EXTENSION_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUTPUT_FILE="${MODULE_NAME}${EXTENSION_SUFFIX}"

nvcc -O3 -shared -std=c++17 -Xcompiler -fPIC \
    -I${PYTHON_INCLUDE} \
    -I${NUMPY_INCLUDE} \
    -I${PYBIND11_INCLUDE} \
    Components/PDV.cu \
    -o ${OUTPUT_FILE}

echo "Build complete: ${OUTPUT_FILE}"
