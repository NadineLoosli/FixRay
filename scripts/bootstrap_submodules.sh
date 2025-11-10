#!/usr/bin/env bash
set -euo pipefail
# Initialize and update submodules, then install editable if setup.py/pyproject exists
git submodule init
git submodule update --remote --recursive
# Install fracture-segmentation in editable mode if it contains a setup.py or pyproject.toml
if [ -d "libs/fracture-segmentation" ]; then
  if [ -f "libs/fracture-segmentation/setup.py" ] || [ -f "libs/fracture-segmentation/pyproject.toml" ]; then
    python -m pip install -e libs/fracture-segmentation
  else
    echo "Submodule cloned, but no setup.py/pyproject.toml found. Add the submodule path to PYTHONPATH or install manually."
  fi
fi

echo "Submodules initialized/updated. If fracture-segmentation provides a packaging entrypoint, it was installed editable."
