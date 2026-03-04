#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Checking Python 3..."
if ! command -v python3 >/dev/null 2>&1; then
    echo
    echo "python3 was not found. Please install Python 3 first."
    exit 1
fi

python3 --version
echo

echo "Checking pip..."
if ! python3 -m pip --version >/dev/null 2>&1; then
    echo "pip is not available. Trying to enable it with ensurepip..."
    python3 -m ensurepip --upgrade || true
fi

if ! python3 -m pip --version >/dev/null 2>&1; then
    echo
    echo "pip is still unavailable. Please install pip for your Python 3 environment."
    exit 1
fi

echo "Upgrading pip..."
python3 -m pip install --upgrade pip

echo
read -p "Create and use a virtual environment (.venv)? [Y/n] " USE_VENV
if [[ -z "$USE_VENV" || "$USE_VENV" =~ ^[Yy]$ ]]; then
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Virtual environment activated: $SCRIPT_DIR/.venv"
fi

echo
echo "Installing required Python modules from requirements.txt ..."
if python3 -m pip install --only-binary=:all: -r requirements.txt; then
    echo
    echo "Installation completed successfully."
else
    echo
    echo "Binary-only installation failed. Retrying with the standard installer..."
    python3 -m pip install -r requirements.txt
    echo
    echo "Installation completed successfully."
fi
