"""Tests for the public embedding API (`voicepranks.api`) and its CLI."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from voicepranks.api import VoiceFilter, render
from voicepranks.audio.wavio import read_wav, write_wav
from voicepranks.presets import preset_names

SR = 48000


def _tone(seconds: float = 1.0, freq: float = 200.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _dominant(y: np.ndarray) -> float:
    mid = y[len(y) // 4: 3 * len(y) // 4].astype(np.float64)
    mag = np.abs(np.fft.rfft(mid * np.hanning(len(mid))))
    return float(np.fft.rfftfreq(len(mid), 1.0 / SR)[np.argmax(mag)])


def test_importing_api_does_not_pull_in_sounddevice():
    code = "import sys, voicepranks.api; assert 'sounddevice' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("preset", preset_names())
def test_render_keeps_length_and_tail(preset):
    x = _tone()
    y = render(x, SR, preset=preset)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y)) and float(np.max(np.abs(y))) <= 1.0
    # The buffered tail is flushed, not dropped: the last 100 ms still sounds.
    assert float(np.sqrt(np.mean(y[-SR // 10:] ** 2))) > 0.01


def test_render_tail_s_extends_output():
    x = _tone(0.5)
    y = render(x, SR, preset="scary", tail_s=0.25)
    assert y.shape[0] == x.shape[0] + SR // 4


def test_render_is_deterministic():
    x = _tone()
    assert np.array_equal(render(x, SR, preset="animalese"), render(x, SR, preset="animalese"))


def test_preset_applies_full_intensity():
    # Plain pitch-up at full intensity: 7 semitones moves 200 Hz to ~300 Hz.
    y = render(_tone(), SR, params={"effect.max_semitones": 7.0})
    assert _dominant(y) == pytest.approx(200 * 2 ** (7 / 12), rel=0.05)


def test_gibberish_preset_routes_through_shuffle():
    vf = VoiceFilter(SR, preset="minion")
    assert vf._effect.gibberish and vf._effect.use_shuffle
    assert vf.latency_samples > SR // 10  # shuffle buffers whole chunk windows


def test_params_override_preset():
    vf = VoiceFilter(SR, preset="scary", params={"effect.max_semitones": -3.0})
    assert vf._effect.max_semitones == -3.0
    assert vf.values["distortion.enabled"] is True


def test_laugh_off_unless_requested():
    goofy = VoiceFilter(SR, preset="goofy")
    assert goofy._effect.laugh_enabled is False
    assert goofy.values["laugh.enabled"] is False  # values report what's in effect
    # laugh=True turns it on, whatever the preset says.
    assert VoiceFilter(SR, preset="scary", laugh=True)._effect.laugh_enabled is True
    assert VoiceFilter(SR, laugh=True)._effect.laugh_enabled is True


def test_no_preset_is_the_unchanged_voice():
    # Neutral baseline, not MinionEffect's +8.5-semitone construction default.
    assert _dominant(render(_tone(), SR)) == pytest.approx(200, rel=0.03)
    growl_only = VoiceFilter(SR, params={"growl.enabled": True, "growl.depth": 0.3})
    assert growl_only._effect.max_semitones == 0.0


@pytest.mark.parametrize("text, want", [("False", False), ("false", False), ("0", False), ("true", True), (1, True)])
def test_bool_params_parse_strictly(text, want):
    assert VoiceFilter(SR, params={"distortion.enabled": text})._effect.distortion_enabled is want


@pytest.mark.parametrize("params", [{"distortion.enabled": "maybe"}, {"shuffle.shuffle_k": 2.9}, {"reverb.mix": "loud"}])
def test_malformed_values_rejected(params):
    with pytest.raises(ValueError):
        VoiceFilter(SR, params=params)


@pytest.mark.parametrize(
    "params, err",
    [
        ({"nope.knob": 1.0}, KeyError),
        ({"effect.max_semitones": 99.0}, ValueError),
        ({"io.blocksize": 256}, KeyError),
    ],
)
def test_bad_params_rejected(params, err):
    with pytest.raises(err):
        VoiceFilter(SR, params=params)


def test_streaming_matches_render():
    x = _tone()
    vf = VoiceFilter(SR, preset="goofy")
    streamed = np.concatenate([vf.process(x[i:i + 300]) for i in range(0, len(x), 300)] + [vf.flush()])
    assert np.allclose(streamed[: len(x)], render(x, SR, preset="goofy"), atol=1e-5)


def test_reset_restores_deterministic_output():
    x = _tone(0.5)
    vf = VoiceFilter(SR, preset="animalese")
    a = np.concatenate([vf.process(x), vf.flush()])
    vf.reset()
    b = np.concatenate([vf.process(x), vf.flush()])
    assert np.array_equal(a, b)


def test_render_cli(tmp_path):
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    write_wav(str(src), SR, _tone(0.5))
    subprocess.run(
        [sys.executable, "-m", "voicepranks.render", str(src), str(dst),
         "--preset", "scary", "--param", "effect.max_semitones=-5", "--tail-s", "0.1"],
        check=True, capture_output=True,
    )
    sr, y = read_wav(str(dst))
    assert sr == SR and y.shape[0] == SR // 2 + SR // 10


def test_render_cli_rejects_unknown_param(tmp_path):
    src = tmp_path / "in.wav"
    write_wav(str(src), SR, _tone(0.1))
    proc = subprocess.run(
        [sys.executable, "-m", "voicepranks.render", str(src), str(tmp_path / "o.wav"), "--param", "bogus=1"],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0 and "error: unknown param: 'bogus'" in proc.stderr


def test_render_cli_bool_param_false_turns_stage_off(tmp_path):
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    write_wav(str(src), SR, _tone(0.3))
    subprocess.run(
        [sys.executable, "-m", "voicepranks.render", str(src), str(dst), "--preset", "scary",
         "--param", "distortion.enabled=False", "--param", "growl.enabled=False",
         "--param", "reverb.enabled=False", "--param", "effect.eq_enabled=False"],
        check=True, capture_output=True,
    )
    _, y = read_wav(str(dst))
    assert np.allclose(y, render(read_wav(str(src))[1], SR, params={"effect.max_semitones": -7.0}), atol=1e-4)


def test_render_cli_unreadable_wav_is_a_clean_error(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav")
    proc = subprocess.run(
        [sys.executable, "-m", "voicepranks.render", str(bad), str(tmp_path / "o.wav")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2 and "can't read" in proc.stderr and "Traceback" not in proc.stderr
