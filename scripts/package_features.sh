#!/usr/bin/env bash
# Packages the cached mel-spectrograms so they can be uploaded to Google Drive
# and used by the Colab notebook, skipping the 5.6 GB dataset download there.
#
# Produces processed_features.tar.gz (~370 MB) in the project root.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -f data/processed/us8k_X.npy ]; then
    echo "No feature cache found. Run this first:"
    echo "  .venv/bin/python run_experiments.py --stage prep"
    exit 1
fi

OUT="processed_features.tar.gz"
echo "packaging data/processed/ ..."
tar -czf "$OUT" data/processed/

echo
echo "created: $OUT ($(du -h "$OUT" | cut -f1))"
echo
echo "Next: upload it to the root of your Google Drive, then run cell 5B"
echo "in notebooks/colab_run_experiments.ipynb instead of cell 5A."
