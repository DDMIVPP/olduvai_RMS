#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_CMD=""

if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_CMD="python3.12"
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import struct,sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and struct.calcsize("P") * 8 == 64 else 1)' >/dev/null 2>&1; then
    PYTHON_CMD="python3"
fi

if [[ -z "$PYTHON_CMD" ]]; then
    echo "A 64-bit Python 3.12 interpreter was not found."
    echo "Install Python 3.12, then run this script again."
    exit 1
fi

echo "Using Python:"
"$PYTHON_CMD" --version

if [[ ! -x ".venv/bin/python" ]]; then
    echo
    echo "Creating the project virtual environment .venv ..."
    "$PYTHON_CMD" -m venv .venv
fi

if ! .venv/bin/python -c 'import struct,sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and struct.calcsize("P") * 8 == 64 else 1)'; then
    echo
    echo "The existing .venv does not use 64-bit Python 3.12."
    echo "Rename or remove that .venv, then run this installer again."
    exit 1
fi

echo
echo "Upgrading pip in .venv ..."
.venv/bin/python -m pip install --upgrade pip

echo
echo "Installing the pinned packages from requirements.txt ..."
if ! .venv/bin/python -m pip install --only-binary=:all: -r requirements.txt; then
    echo
    echo "Binary-only installation failed. Retrying with the standard installer ..."
    .venv/bin/python -m pip install -r requirements.txt
fi

echo
echo "Verifying required imports ..."
.venv/bin/python -c "import matplotlib,numpy,openpyxl,pandas,scipy,sklearn,statsmodels; print('Dependency import check passed.')"

echo
echo "Installation completed successfully."
echo "Run the analysis with:"
echo "  ./.venv/bin/python olduvai_RMS.py"
