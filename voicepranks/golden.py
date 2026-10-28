"""Write golden reference renders for ports of the effect to other languages.

Usage:
    python -m voicepranks.golden OUTDIR [--sample-rate 48000] [--seconds 1.0]

Writes into OUTDIR:
  - `input__<signal>.wav`: the test signals (sine, saw, chirp), 16-bit.
  - `<case>__<signal>.wav`: each signal rendered through each case, via
    `voicepranks.api.render` (same length as the input).
  - `manifest.json`: for every render, the case's params, the filter's
    `latency_samples`, the dominant frequency of the middle half, and a
    20 ms RMS envelope -- the features a port compares where it can't be
    sample-exact (the gibberish engines' RNG).

Cases isolate each stage of the plain path (pitch, EQ, growl, distortion,
nasal bands, reverb) on the neutral baseline, then cover every preset. Each
case's params in the manifest are the filter's effective values (baseline
filled in, laugh off), exactly what was rendered. Inputs are quantized to
16-bit *before* rendering, so a port that reads `input__*.wav` back sees
exactly the samples the reference processed.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from . import __version__
from .api import VoiceFilter, render
from .audio.wavio import read_wav, write_wav
from .presets import PRESETS

_ENV_MS = 20.0

# Each stage in isolation, layered on the neutral baseline so nothing else
# colours it.
STAGE_CASES: Dict[str, Dict[str, Any]] = {
    "pitch_up": {"effect.max_semitones": 7.0},
    "pitch_down": {"effect.max_semitones": -7.0},
    "eq": {"effect.eq_enabled": True, "effect.eq_gain_db": 6.0, "effect.eq_center_hz": 2500.0, "effect.eq_q": 1.1},
    "growl": {"growl.enabled": True, "growl.rate_hz": 70.0, "growl.depth": 0.55},
    "distortion": {"distortion.enabled": True, "distortion.drive": 11.0},
    "nasal": {"effect.nasality": 0.85},
    "reverb": {"reverb.enabled": True, "reverb.mix": 0.22, "reverb.room_size": 0.6, "reverb.damp": 0.6},
}


def _signals(sample_rate: int, seconds: float) -> Dict[str, np.ndarray]:
    n = int(sample_rate * seconds)
    t = np.arange(n, dtype=np.float64) / sample_rate
    saw_phase = (150.0 * t) % 1.0
    # Linear chirp 100 Hz -> 1000 Hz over the clip.
    f0, f1 = 100.0, 1000.0
    chirp_phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / seconds * t * t)
    raw = {
        "sine": 0.5 * np.sin(2.0 * np.pi * 220.0 * t),
        "saw": 0.5 * (2.0 * saw_phase - 1.0),
        "chirp": 0.5 * np.sin(chirp_phase),
    }
    return {name: sig.astype(np.float32) for name, sig in raw.items()}


def _dominant_hz(signal: np.ndarray, sample_rate: int) -> float:
    mid = signal[signal.shape[0] // 4: 3 * signal.shape[0] // 4].astype(np.float64)
    if mid.shape[0] < 2:
        return 0.0
    mag = np.abs(np.fft.rfft(mid * np.hanning(mid.shape[0])))
    return float(np.fft.rfftfreq(mid.shape[0], d=1.0 / sample_rate)[int(np.argmax(mag))])


def _rms_envelope(signal: np.ndarray, sample_rate: int) -> List[float]:
    win = max(int(sample_rate * _ENV_MS / 1000.0), 1)
    usable = signal.shape[0] - signal.shape[0] % win
    frames = signal[:usable].astype(np.float64).reshape(-1, win)
    return [round(float(v), 6) for v in np.sqrt(np.mean(frames * frames, axis=1))]


def write_golden(outdir: str, sample_rate: int = 48000, seconds: float = 1.0) -> Dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)
    requested: Dict[str, Dict[str, Any]] = dict(STAGE_CASES)
    requested.update({f"preset_{name}": dict(values) for name, values in PRESETS.items()})
    # Record each case as the filter actually runs it (neutral baseline filled
    # in, laugh forced off), so the manifest describes exactly what rendered.
    filters = {case: VoiceFilter(sample_rate, params=p) for case, p in requested.items()}
    cases = {case: dict(vf.values) for case, vf in filters.items()}

    renders = []
    for sig_name, sig in _signals(sample_rate, seconds).items():
        in_path = os.path.join(outdir, f"input__{sig_name}.wav")
        write_wav(in_path, sample_rate, sig)
        _, quantized = read_wav(in_path)
        for case, params in cases.items():
            out = render(quantized, sample_rate, params=params)
            file = f"{case}__{sig_name}.wav"
            write_wav(os.path.join(outdir, file), sample_rate, out)
            renders.append({
                "case": case,
                "signal": sig_name,
                "input": os.path.basename(in_path),
                "file": file,
                "latency_samples": filters[case].latency_samples,
                "dominant_hz": round(_dominant_hz(out, sample_rate), 3),
                "rms_envelope": _rms_envelope(out, sample_rate),
            })

    manifest = {
        "version": __version__,
        "sample_rate": sample_rate,
        "seconds": seconds,
        "envelope_ms": _ENV_MS,
        "cases": cases,
        "renders": renders,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Write golden reference renders for effect ports.")
    parser.add_argument("outdir")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    manifest = write_golden(args.outdir, args.sample_rate, args.seconds)
    print(f"wrote {len(manifest['renders'])} renders to {args.outdir}")


if __name__ == "__main__":
    main()
