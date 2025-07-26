# GroupBy Package Development Guidelines

## Build/Test/Lint Commands
- Run all tests: `conda activate pandas-plus-dev && cd /path/to/groupby && python -m pytest tests/`
- Run single test: `conda activate pandas-plus-dev && cd /path/to/groupby && python -m pytest tests/test_file.py::TestClass::test_method -v`
- Run with parallelism: `conda activate pd22 pandas-plus-dev cd /path/to/groupby && python -m pytest tests/ -xvs`

## Code Style Guidelines
- **Imports**: Order by standard library, third-party, local modules. Group imports within these categories.
- **Typing**: Use Python type hints (e.g., `ArrayType1D`, Callable, List). Use Union with pipe operator (`|`).
- **Error Handling**: Use appropriate exceptions with descriptive messages. Check input dimensions and types.
- **Naming Conventions**: Use snake_case for functions/variables, PascalCase for classes. Prefix private functions with underscore.
- **Documentation**: Add docstrings for public functions and classes (see core.py for example style).
- **Testing**: Write unit tests for each function. Use parametrized tests where applicable. Use the pytest framework when writing tests
- **NumPy/Pandas**: Follow NumPy conventions for array handling. Handle NaN values consistently.

## Dependencies
Python 3.6+, NumPy, pandas, numba, pytest
