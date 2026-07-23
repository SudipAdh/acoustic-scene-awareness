# Acoustic Scene Awareness

**Urban and Household Sound Recognition using Convolutional Neural Networks with Autoencoder-Based Novel-Sound Detection**

Coursework for **STW7088CEM — Artificial Neural Networks**, MSc Data Science and Computational Intelligence, Softwarica College of IT & E-Commerce (in collaboration with Coventry University).

---

## What this project does

A sound-awareness system that recognises everyday urban and household sounds directly from audio and raises a **visual alert** when something important happens — a siren, car horn, dog bark, or gunshot. The motivating use case is assistive technology for Deaf and hard-of-hearing users, who cannot rely on an audible cue; the same model also fits smart-home safety monitoring.

Raw audio is converted into **log-mel spectrograms** — a 2-D picture of frequency against time — so a convolutional network can learn each sound's signature without any hand-engineered audio features. This is the core reason the project is a neural-network problem rather than a classical machine-learning one: no tabular feature set is ever constructed.

A closed-set classifier has a dangerous failure mode for this use case: shown a sound outside its ten classes it must still answer with one of them, often confidently. A **convolutional autoencoder** trained only on the known classes addresses this — unfamiliar sounds reconstruct poorly, so reconstruction error becomes an "I have not heard this before" signal.

## Tasks performed

The brief requires more than one task. This project performs four:

| # | Task | Method |
|---|---|---|
| 1 | **Classification** | `AudioCNN` — 4-block CNN over log-mel spectrograms |
| 2 | **Anomaly / novelty detection** | `ConvAutoencoder` — reconstruction error as a novelty score |
| 3 | **Clustering** | k-means + t-SNE over the CNN's learned 256-d embeddings |
| 4 | **Architecture comparison** | MLP baseline vs CNN, plus an augmentation ablation |

## Datasets

Both are open, free, and require no account. `scripts/download_data.sh` fetches them.

| Dataset | Contents | Role | License |
|---|---|---|---|
| [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) | 8,732 clips, 10 classes, 10 official folds | Main classification dataset | CC BY-NC 3.0 |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | 2,000 clips, 50 classes | Unseen sounds for novelty detection | CC BY-NC |

Mirror used for UrbanSound8K: [Zenodo record 1203745](https://zenodo.org/records/1203745) (also on [Kaggle](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)).

**Citation requirement.** UrbanSound8K: Salamon, Jacoby & Bello (2014), *A Dataset and Taxonomy for Urban Sound Research*, ACM-MM. ESC-50: Piczak (2015), *ESC: Dataset for Environmental Sound Classification*, ACM-MM.

## Evaluation protocol

UrbanSound8K ships **10 predefined folds**, and this project uses them. That matters: many clips are cut from the same source recording, and a random train/test split scatters those slices across both sides, leaking information and inflating accuracy. All reported numbers use the official 10-fold cross-validation, with normalisation statistics computed on the training split only.

## Setup

Requires Python 3.11. On macOS, install [`uv`](https://docs.astral.sh/uv/) (`brew install uv`) — it provides a self-contained Python and avoids the broken `pyexpat` in Homebrew's Python builds.

```bash
git clone <repo-url>
cd acoustic-scene-awareness

uv venv --python 3.11 --python-preference only-managed
uv pip install -r requirements.txt

bash scripts/download_data.sh        # ~6.5 GB, one time
```

## Running

```bash
# fast sanity check (1 fold, 3 epochs)
.venv/bin/python run_experiments.py --quick

# individual stages
.venv/bin/python run_experiments.py --stage prep       # cache spectrograms
.venv/bin/python run_experiments.py --stage explore    # dataset figures
.venv/bin/python run_experiments.py --stage compare    # MLP vs CNN
.venv/bin/python run_experiments.py --stage final      # full 10-fold run
.venv/bin/python run_experiments.py --stage cluster    # k-means + t-SNE
.venv/bin/python run_experiments.py --stage anomaly    # novelty detection

# everything, full protocol
.venv/bin/python run_experiments.py
```

### Live demo

```bash
.venv/bin/python demo/live_demo.py                 # microphone
.venv/bin/python demo/live_demo.py --file clip.wav # replay a file
.venv/bin/python demo/live_demo.py --list-devices
```

A window opens showing a live 4-second spectrogram, per-class confidence bars, and a status banner that turns red for safety-critical sounds and amber for sounds the autoencoder flags as unfamiliar. Requires a model — run `--stage final` first.

## Repository layout

```
src/config.py       all hyperparameters and paths
src/data.py         audio loading, mel-spectrograms, augmentation, datasets
src/models.py       MLPBaseline, AudioCNN, ConvAutoencoder
src/train.py        training loop and 10-fold cross-validation
src/evaluate.py     pooled metrics, confusion matrix, embedding extraction
src/anomaly.py      autoencoder novelty detection + softmax baseline
src/cluster.py      k-means, t-SNE, cluster quality metrics
src/viz.py          all report figures (shared visual system)
run_experiments.py  staged experiment runner
demo/live_demo.py   real-time microphone dashboard
scripts/            dataset download, pipeline smoke test
results/figures/    generated figures (PNG + PDF)
results/logs/       metrics as JSON
```

## Hardware

Developed and trained on an **Apple MacBook (M1, 16 GB)** using PyTorch's **MPS** backend for GPU acceleration. No paid compute was used. Feature extraction takes well under a minute for the full dataset; the cached spectrograms make repeat runs fast.

## Reproducibility

- Every random seed is fixed in `src/config.py` (`SEED = 42`), and each fold seeds off it deterministically.
- Feature extraction settings (sample rate, FFT size, hop, mel bands) are single-sourced in `config.py`.
- All metrics are written to `results/logs/*.json`; all figures to `results/figures/`.
- `scripts/smoke_test.py` exercises every stage without needing UrbanSound8K.

## License

Code released under the MIT License. The datasets remain under their own CC BY-NC terms and are not redistributed here.
