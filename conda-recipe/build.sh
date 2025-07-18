#!/bin/bash

# Build script for Unix-like systems (Linux, macOS)
set -euo pipefail

# Install the package
$PYTHON -m pip install . -vv --no-deps --no-build-isolation

# Compile numba functions ahead of time if needed
$PYTHON -c "import pandas_plus.groupby.numba; print('Numba functions loaded successfully')"