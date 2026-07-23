#!/usr/bin/env python
"""Real-time acoustic scene awareness from the laptop microphone.

Holds a rolling 4-second window of audio, converts it to a log-mel spectrogram
on the same front-end used in training, and runs both networks each refresh:

  * the CNN classifies the sound and drives the confidence bars
  * the autoencoder scores how familiar it is, so an unrecognised sound is
    reported as UNKNOWN rather than forced into one of the ten labels

Run:
    python demo/live_demo.py                 # live microphone
    python demo/live_demo.py --list-devices
    python demo/live_demo.py --file clip.wav # replay a file instead
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C  # noqa: E402
from src import viz  # noqa: E402
from src.data import audio_to_melspec  # noqa: E402
from src.models import AudioCNN, ConvAutoencoder  # noqa: E402

REFRESH_SEC = 0.5          # how often inference runs
SMOOTHING = 0.6            # EMA over probabilities, damps single-frame flicker


# ---------------------------------------------------------------------------
class Engine:
    """Loads the trained models and turns a waveform into a prediction."""

    def __init__(self, cnn_path, ae_path=None, device=C.DEVICE):
        self.device = device

        ck = torch.load(cnn_path, map_location=device, weights_only=False)
        self.mean, self.std = ck["mean"], ck["std"]
        self.cnn = AudioCNN(n_classes=ck.get("n_classes", 10)).to(device)
        self.cnn.load_state_dict(ck["state_dict"])
        self.cnn.eval()

        self.ae = None
        self.threshold = None
        if ae_path and Path(ae_path).exists():
            ack = torch.load(ae_path, map_location=device, weights_only=False)
            self.ae = ConvAutoencoder().to(device)
            self.ae.load_state_dict(ack["state_dict"])
            self.ae.eval()
            self.threshold = ack.get("threshold")

        self.probs = np.zeros(len(C.US8K_CLASSES))

    @torch.no_grad()
    def infer(self, wave: np.ndarray):
        spec = audio_to_melspec(wave)
        x = torch.from_numpy((spec - self.mean) / self.std).float()[None, None].to(self.device)

        p = torch.softmax(self.cnn(x), dim=1)[0].cpu().numpy()
        self.probs = SMOOTHING * self.probs + (1 - SMOOTHING) * p  # EMA

        novelty, is_novel = None, False
        if self.ae is not None:
            novelty = float(self.ae.reconstruction_error(x).item())
            if self.threshold is not None:
                is_novel = novelty > self.threshold

        return spec, self.probs, novelty, is_novel


# ---------------------------------------------------------------------------
class AudioStream:
    """Rolling microphone buffer of the last CLIP_SECONDS seconds."""

    def __init__(self, device=None, samplerate=C.SAMPLE_RATE):
        self.samplerate = samplerate
        self.buffer = np.zeros(C.N_SAMPLES, dtype=np.float32)
        self.lock = threading.Lock()
        self.q: queue.Queue = queue.Queue()
        self.device = device
        self.stream = None
        self.level = 0.0

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        mono = indata[:, 0].copy()
        with self.lock:
            n = len(mono)
            self.buffer = np.roll(self.buffer, -n)
            self.buffer[-n:] = mono
            self.level = float(np.abs(mono).max())

    def start(self):
        import sounddevice as sd

        self.stream = sd.InputStream(
            channels=1, samplerate=self.samplerate, device=self.device,
            blocksize=int(self.samplerate * 0.1), callback=self._callback,
        )
        self.stream.start()
        print(f"[audio] listening at {self.samplerate} Hz")

    def read(self) -> np.ndarray:
        with self.lock:
            return self.buffer.copy()

    def stop(self):
        if self.stream:
            self.stream.stop(); self.stream.close()


# ---------------------------------------------------------------------------
class Dashboard:
    """Live figure: spectrogram, confidence bars, and a status banner."""

    def __init__(self, engine: Engine, source, is_file=False):
        self.engine = engine
        self.source = source
        self.is_file = is_file
        viz.use_report_style()

        self.fig = plt.figure(figsize=(13, 6.6))
        self.fig.canvas.manager.set_window_title("Acoustic Scene Awareness — live")
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 5.2],
                                   width_ratios=[1.35, 1], hspace=0.28, wspace=0.22)

        # Status banner
        self.ax_banner = self.fig.add_subplot(gs[0, :])
        self.ax_banner.axis("off")
        self.banner = self.ax_banner.text(
            0.5, 0.5, "listening…", ha="center", va="center",
            fontsize=25, fontweight="bold", color=viz.INK,
            transform=self.ax_banner.transAxes,
            bbox=dict(boxstyle="round,pad=0.55", fc="#eef3fa", ec=viz.SERIES[0], lw=2.5))

        # Spectrogram
        self.ax_spec = self.fig.add_subplot(gs[1, 0])
        self.im = self.ax_spec.imshow(
            np.full((C.N_MELS, C.N_FRAMES), -80.0), origin="lower", aspect="auto",
            cmap="magma", vmin=-80, vmax=0, extent=[0, C.CLIP_SECONDS, 0, C.N_MELS])
        self.ax_spec.set_title("Live log-mel spectrogram (last 4 s)", loc="left", pad=10)
        self.ax_spec.set_xlabel("Time (s)"); self.ax_spec.set_ylabel("Mel band")
        self.ax_spec.grid(False)

        # Confidence bars
        self.ax_bars = self.fig.add_subplot(gs[1, 1])
        labels = [C.DEMO_LABELS[c] for c in C.US8K_CLASSES]
        self.bars = self.ax_bars.barh(labels, np.zeros(len(labels)),
                                      color=viz.SERIES[0], height=0.62)
        self.val_txt = [
            self.ax_bars.text(0.6, b.get_y() + b.get_height() / 2, "", va="center",
                              fontsize=9, color=viz.INK_2)
            for b in self.bars
        ]
        self.ax_bars.set_xlim(0, 1)
        self.ax_bars.set_xlabel("Confidence")
        self.ax_bars.set_title("CNN class confidence", loc="left", pad=10)
        self.ax_bars.grid(axis="y", visible=False)
        for s in ("top", "right"):
            self.ax_bars.spines[s].set_visible(False)

        self.file_pos = 0

    def _next_wave(self) -> np.ndarray:
        if not self.is_file:
            return self.source.read()
        # File mode: sweep a window across the clip so the demo animates.
        wave = self.source
        step = int(C.SAMPLE_RATE * REFRESH_SEC)
        start = self.file_pos % max(len(wave), 1)
        self.file_pos += step
        w = wave[start:start + C.N_SAMPLES]
        if len(w) < C.N_SAMPLES:
            w = np.pad(w, (0, C.N_SAMPLES - len(w)))
        return w

    def update(self, _frame):
        wave = self._next_wave()

        if np.abs(wave).max() < 1e-4:                     # effectively silence
            self.banner.set_text("listening…")
            self.banner.set_color(viz.INK)
            self.banner.get_bbox_patch().set(fc="#f2f2ef", ec=viz.AXIS)
            for b, t in zip(self.bars, self.val_txt):
                b.set_width(0); t.set_text("")
            return

        spec, probs, novelty, is_novel = self.engine.infer(wave)
        self.im.set_data(spec)

        top = int(np.argmax(probs))
        conf = float(probs[top])
        cls = C.US8K_CLASSES[top]

        for b, t, p in zip(self.bars, self.val_txt, probs):
            b.set_width(p)
            b.set_color(viz.SERIES[0] if p < 0.5 else viz.SERIES[1])
            t.set_x(min(p + 0.02, 0.88))
            t.set_text(f"{p*100:.0f}%" if p > 0.02 else "")

        if is_novel:
            txt, fc, ec, col = ("❓ UNKNOWN SOUND", "#fdf3e3",
                                viz.STATUS["warning"], "#7a5200")
        elif cls in C.ALERT_CLASSES and conf > 0.45:
            txt = f"⚠️  {C.DEMO_LABELS[cls].upper()}"
            fc, ec, col = "#fdeaea", viz.STATUS["critical"], "#8c1d1d"
        elif conf > 0.35:
            txt, fc, ec, col = (C.DEMO_LABELS[cls], "#eef3fa", viz.SERIES[0], viz.INK)
        else:
            txt, fc, ec, col = ("listening…", "#f2f2ef", viz.AXIS, viz.INK)

        if conf > 0.35 and not is_novel:
            txt += f"   {conf*100:.0f}%"
        if novelty is not None:
            txt += f"      novelty {novelty:.3f}"

        self.banner.set_text(txt)
        self.banner.set_color(col)
        self.banner.get_bbox_patch().set(fc=fc, ec=ec)

    def run(self):
        from matplotlib.animation import FuncAnimation

        self.anim = FuncAnimation(self.fig, self.update,
                                  interval=int(REFRESH_SEC * 1000),
                                  cache_frame_data=False)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(C.MODELS_DIR / "cnn_demo.pt"))
    ap.add_argument("--autoencoder", default=str(C.MODELS_DIR / "autoencoder.pt"))
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--file", default=None, help="replay a wav file instead of the mic")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    if not Path(args.model).exists():
        sys.exit(f"Model not found: {args.model}\n"
                 f"Train one first:  python run_experiments.py --stage final")

    # pyplot is already imported (via src.viz), so the backend has to be
    # switched rather than set — matplotlib.use() would be a no-op here.
    for backend in (("MacOSX", "TkAgg") if sys.platform == "darwin" else ("TkAgg",)):
        try:
            plt.switch_backend(backend)
            break
        except Exception:
            continue
    if matplotlib.get_backend().lower() in ("agg", "template"):
        sys.exit("No interactive matplotlib backend available — the live demo "
                 "needs a display. Try: uv pip install pyqt6")
    print(f"[ui] backend: {matplotlib.get_backend()}")

    engine = Engine(args.model, args.autoencoder)
    print(f"[model] loaded on {engine.device}"
          f"{' (+ autoencoder)' if engine.ae is not None else ''}")

    if args.file:
        import librosa
        wave, _ = librosa.load(args.file, sr=C.SAMPLE_RATE, mono=True)
        print(f"[file] {args.file} — {len(wave)/C.SAMPLE_RATE:.1f}s")
        Dashboard(engine, wave, is_file=True).run()
        return

    stream = AudioStream(device=args.device)
    stream.start()
    try:
        Dashboard(engine, stream).run()
    finally:
        stream.stop()
        print("[audio] stopped")


if __name__ == "__main__":
    main()
