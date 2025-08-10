# LSSVM - Least Squares Support Vector Machine

[![Code Quality](https://github.com/RomuloDrumond/LSSVM/actions/workflows/code_quality.yml/badge.svg)](https://github.com/RomuloDrumond/LSSVM/actions/workflows/code_quality.yml)

This repository contains a Python implementation of the **Least Squares Support Vector Machine (LSSVM)** model for both **CPU** and **GPU**. You can find theory and usage examples in the `LSSVC.ipynb` Jupyter notebook.

For a better viewing experience of the notebook: https://nbviewer.jupyter.org/github/RomuloDrumond/LSSVM/blob/master/LSSVC.ipynb

## Installation

To use LSSVM in your project, install it from PyPI (when available):

```bash
# Install from PyPI (when available)
pip install lssvm
```

Alternatively, you can install it directly from the source:

```bash
git clone https://github.com/RomuloDrumond/LSSVM.git
cd LSSVM
pip install .
```

## Contributing and Development

Contributions are welcome! We use modern Python development practices with a focus on code quality and comprehensive testing.

### Quick Start for Contributors

1.  **Fork and clone** the repository.
2.  **Set up the development environment**. This project uses [uv](https://docs.astral.sh/uv/) for dependency management.
    ```bash
    # If you don't have uv, install it first
    make install-uv

    # Install dependencies
    make install-dev
    ```
3.  **Create a new branch** for your feature or bug fix.
4.  **Implement your changes**. Make sure to add tests for any new functionality.
5.  **Run checks** before committing. This ensures your changes meet our quality standards.
    ```bash
    make pre-commit
    ```
6.  **Submit a pull request**.

### Development Commands

The `Makefile` contains several commands to streamline development. Run `make help` to see all options.

-   `make check`: Run linting and type checking.
-   `make test`: Run all tests.
-   `make test-fast`: Run fast unit and integration tests.
-   `make test-slow`: Run slow benchmark and complex dataset tests.
