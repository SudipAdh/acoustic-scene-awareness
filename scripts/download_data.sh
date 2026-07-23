#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Downloads the two datasets used in this project.
#
#   ESC-50        ~600 MB   2,000 clips / 50 classes   (CC BY-NC)
#   UrbanSound8K  ~6 GB     8,732 clips / 10 classes   (CC BY-NC 3.0)
#
# Both are fetched from their official public mirrors and need no login.
# Re-running is safe: existing downloads are resumed, existing extracts skipped.
# ---------------------------------------------------------------------------
set -euo pipefail

RAW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"
mkdir -p "$RAW_DIR"

ESC50_URL="https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
US8K_URL="https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1"

# ---------- ESC-50 ----------
if [ -d "$RAW_DIR/ESC-50-master" ]; then
    echo "[skip] ESC-50 already extracted"
else
    echo "[1/2] Downloading ESC-50 (~600 MB)..."
    curl -L -C - -o "$RAW_DIR/ESC-50.zip" "$ESC50_URL"
    echo "[1/2] Extracting ESC-50..."
    unzip -q -o "$RAW_DIR/ESC-50.zip" -d "$RAW_DIR"
    rm -f "$RAW_DIR/ESC-50.zip"
fi

# ---------- UrbanSound8K ----------
if [ -d "$RAW_DIR/UrbanSound8K" ]; then
    echo "[skip] UrbanSound8K already extracted"
else
    echo "[2/2] Downloading UrbanSound8K (~6 GB, this takes a while)..."
    curl -L -C - -o "$RAW_DIR/UrbanSound8K.tar.gz" "$US8K_URL"
    echo "[2/2] Extracting UrbanSound8K..."
    tar -xzf "$RAW_DIR/UrbanSound8K.tar.gz" -C "$RAW_DIR"
    rm -f "$RAW_DIR/UrbanSound8K.tar.gz"
fi

echo
echo "Done. Datasets are in: $RAW_DIR"
du -sh "$RAW_DIR"/* 2>/dev/null || true
