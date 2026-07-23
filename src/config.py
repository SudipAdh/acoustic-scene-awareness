"""Central configuration for the acoustic scene awareness project.

Every tunable lives here so experiments stay reproducible and the report can
quote exact settings.
"""
from pathlib import Path

import torch

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"

US8K_DIR = RAW_DIR / "UrbanSound8K"
US8K_AUDIO = US8K_DIR / "audio"
US8K_META = US8K_DIR / "metadata" / "UrbanSound8K.csv"

ESC50_DIR = RAW_DIR / "ESC-50-master"
ESC50_AUDIO = ESC50_DIR / "audio"
ESC50_META = ESC50_DIR / "meta" / "esc50.csv"

for _d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Audio / mel-spectrogram front-end
# --------------------------------------------------------------------------
# UrbanSound8K clips are up to 4 s. Fixing every clip to 4 s at 22.05 kHz keeps
# the spectrogram a constant shape, which the CNN requires.
SAMPLE_RATE = 22050
CLIP_SECONDS = 4.0
N_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64
F_MIN = 20
F_MAX = SAMPLE_RATE // 2

# Resulting spectrogram shape -> (N_MELS, N_FRAMES)
N_FRAMES = int(N_SAMPLES / HOP_LENGTH) + 1

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 8
SEED = 42

# UrbanSound8K ships 10 predefined folds. The official protocol is 10-fold
# cross-validation using these folds — never a random split, because clips cut
# from the same source recording live in the same fold. Random splitting leaks
# them across train/test and inflates accuracy.
N_FOLDS = 10

# For fast iteration we can evaluate on a subset of folds; the full run uses all.
QUICK_FOLDS = [1]

# --------------------------------------------------------------------------
# Device — Apple Silicon GPU via Metal (MPS), else CPU
# --------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()

# --------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------
US8K_CLASSES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]

# Human-facing labels for the live demo, plus which sounds are treated as
# safety-critical alerts for the assistive use case.
DEMO_LABELS = {
    "air_conditioner": "Air conditioner",
    "car_horn": "Car horn",
    "children_playing": "Children playing",
    "dog_bark": "Dog barking",
    "drilling": "Drilling",
    "engine_idling": "Engine idling",
    "gun_shot": "Gunshot",
    "jackhammer": "Jackhammer",
    "siren": "Siren",
    "street_music": "Street music",
}

ALERT_CLASSES = {"car_horn", "gun_shot", "siren", "dog_bark"}

# ESC-50 classes used for the novel-sound (anomaly) experiment. These are
# household/safety sounds the UrbanSound8K model has never been trained on, so
# a good novelty detector should flag them as unknown.
ESC50_NOVEL_CLASSES = [
    "door_wood_knock",
    "glass_breaking",
    "crying_baby",
    "clock_alarm",
    "door_wood_creaks",
    "water_drops",
    "footsteps",
    "can_opening",
]
