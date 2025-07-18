@echo off
REM Build script for Windows

REM Install the package
%PYTHON% -m pip install . -vv --no-deps --no-build-isolation
if errorlevel 1 exit 1

REM Compile numba functions ahead of time if needed
%PYTHON% -c "import pandas_plus.groupby.numba; print('Numba functions loaded successfully')"
if errorlevel 1 exit 1