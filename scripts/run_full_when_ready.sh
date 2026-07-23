#!/usr/bin/env bash
# Waits for the download + prep chain to finish, verifies the cached features,
# then runs the full 10-fold experiment suite.
#
# Safe to run unattended: every stage is checkpointed to results/logs/ and
# results/figures/ as it completes, so a failure late in the run does not
# discard earlier results.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "waiting for download + prep chain to finish..."
while pgrep -f "download_data.sh|wait_then_prep.sh" >/dev/null; do sleep 60; done
log "chain finished"

# --- verify the extraction actually produced a usable dataset -------------
if [ ! -f data/raw/UrbanSound8K/metadata/UrbanSound8K.csv ]; then
    log "FAIL: UrbanSound8K metadata missing — extraction did not complete."
    exit 1
fi
N_WAV=$(find data/raw/UrbanSound8K/audio -name '*.wav' | wc -l | tr -d ' ')
log "extracted $N_WAV wav files (expected 8732)"
if [ "$N_WAV" -lt 8000 ]; then
    log "FAIL: too few wav files — archive likely truncated."
    exit 1
fi

# --- ensure features are cached (the prep chain normally does this) -------
if [ ! -f data/processed/us8k_X.npy ]; then
    log "feature cache missing — running prep"
    .venv/bin/python run_experiments.py --stage prep 2>&1 \
        | grep -vE '^\s*extracting' | tail -20 || exit 1
fi
log "feature cache ready: $(du -sh data/processed/us8k_X.npy | cut -f1)"

# --- full suite, stage by stage so partial results survive a failure ------
for stage in explore compare final cluster anomaly; do
    log "=== STAGE: $stage ==="
    if .venv/bin/python run_experiments.py --stage "$stage" 2>&1 \
         | grep -vE '^\s*extracting|^\[cache\]' ; then
        log "stage $stage OK"
    else
        log "stage $stage FAILED (continuing to next stage)"
    fi
done

log "ALL STAGES COMPLETE"
log "figures: $(ls results/figures/*.png 2>/dev/null | wc -l | tr -d ' ') png"
log "logs:    $(ls results/logs/*.json 2>/dev/null | wc -l | tr -d ' ') json"
