#!/usr/bin/env bash
# Waits for the dataset download to finish, verifies the extraction, then
# caches features and runs a fast end-to-end check.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "[wait] waiting for download_data.sh to finish..."
while pgrep -f "download_data.sh" >/dev/null; do sleep 30; done
echo "[wait] download script exited at $(date)"

if [ ! -f data/raw/UrbanSound8K/metadata/UrbanSound8K.csv ]; then
    echo "[FAIL] UrbanSound8K metadata missing — extraction did not complete."
    ls -la data/raw/ || true
    exit 1
fi

N_WAV=$(find data/raw/UrbanSound8K/audio -name '*.wav' | wc -l | tr -d ' ')
echo "[ok] extracted: $N_WAV wav files (expected 8732)"
if [ "$N_WAV" -lt 8000 ]; then
    echo "[FAIL] too few wav files — archive likely truncated."
    exit 1
fi

echo "[prep] caching mel-spectrograms..."
.venv/bin/python run_experiments.py --stage prep 2>&1 | grep -vE '^\s*extracting' | tail -25 || exit 1

echo "[check] fast end-to-end run (1 fold, 3 epochs)..."
.venv/bin/python run_experiments.py --stage compare --quick 2>&1 | tail -30

echo "[done] ready for the full run at $(date)"
