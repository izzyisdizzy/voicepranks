"""Public, embeddable API for the voice effect -- the supported surface for
using VoicePranks from another program (offline sound-effect rendering, a
game's voice chat, a batch tool).

Everything here is pure numpy: importing this module never imports
`sounddevice` (that lives only in `audio/engine.py`), opens no device and
starts no thread.

    from voicepranks.api import VoiceFilter, render

    # Offline: one call, output is the same length as the input.
    out = render(mono, 48000, preset="scary")

    # Streaming: feed blocks, collect whatever comes out.
    vf = VoiceFilter(48000, preset="minion")
    for block in blocks:
        sink.write(vf.process(block))   # may be shorter/longer/empty
    sink.write(vf.flush())              # drain the buffered tail

Params are the same dotted names as `params.PARAM_SPECS`; presets are the
names in `presets.PRESETS`. Explicit `params` override the preset's values.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from .dsp.effect import MinionEffect
from .params import PARAM_SPECS_BY_NAME, build_effect_registry
from .presets import get_preset, preset_names

__all__ = ["VoiceFilter", "render", "preset_names"]

# Block size `render` feeds the effect with -- matches the selftest CLI.
_RENDER_BLOCK = 1024
# Upper bound on how long `flush` will feed silence looking for the tail. The
# slowest engine (shuffle at k=6 x 80 ms + fade) buffers ~0.55 s, so 5 s is
# generous headroom while still guaranteeing termination.
_MAX_FLUSH_S = 5.0


def _coerce(name: str, value: Any) -> Any:
    """Validate one param against its spec and coerce it to the spec's kind."""
    spec = PARAM_SPECS_BY_NAME.get(name)
    if spec is None:
        raise KeyError(f"unknown param: {name!r}")
    if spec.kind == "bool":
        return bool(value)
    v = int(value) if spec.kind == "int" else float(value)
    if not (spec.min <= v <= spec.max):
        raise ValueError(f"param {name!r}={v} outside [{spec.min}, {spec.max}]")
    return v


class VoiceFilter:
    """A configured, streaming instance of the voice effect.

    `process()` takes mono float32 samples in [-1, 1] at `sample_rate` and
    returns processed samples. The effect buffers internally (WSOLA, the
    gibberish engines), so a call can return fewer, more, or zero samples;
    over a whole stream the output is time-aligned with the input -- it
    starts at input sample 0 and just trails it by up to `latency_samples`.

    Intensity is fixed at 1.0 (the full preset character). The random laugh
    overlay is off unless `laugh=True`: it fires on a time-spent-talking timer and
    can read a user-recorded clip from `~/.voicepranks`, neither of which
    belongs in a reproducible render.
    """

    def __init__(
        self,
        sample_rate: int,
        preset: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        laugh: bool = False,
    ) -> None:
        self.sample_rate = int(sample_rate)
        values: Dict[str, Any] = dict(get_preset(preset)) if preset else {}
        for name, value in (params or {}).items():
            values[name] = value
        self.values = {name: _coerce(name, v) for name, v in values.items()}

        self._effect = MinionEffect(self.sample_rate)
        registry = build_effect_registry(self._effect)
        for name, value in self.values.items():
            if name == "gibberish":
                self._effect.set_gibberish(value)
            elif name in registry:
                registry[name].set(value)
            else:
                raise KeyError(f"param {name!r} is engine/session state, not part of the effect")
        if not laugh:
            self._effect.set_laugh_enabled(False)
        self._effect.set_intensity(1.0)

    @property
    def latency_samples(self) -> int:
        """Worst-case samples the output trails the input by: the active
        gibberish engine's buffering, or one WSOLA frame plus its search
        window on the plain path."""
        pitch = self._effect.pitch
        wsola = pitch.L + 2 * pitch.tol
        engine = int(round(self._effect.latency_ms() * self.sample_rate / 1000.0))
        return max(wsola, engine)

    def process(self, block: np.ndarray) -> np.ndarray:
        """Process one block of mono samples; see the class docstring."""
        mono = np.asarray(block, dtype=np.float32).reshape(-1)
        return self._effect.process(mono)

    def flush(self) -> np.ndarray:
        """Return the samples still buffered inside the effect by feeding it
        silence for `latency_samples` (plus one block of slack). Call at the
        end of a stream; the filter can be reused afterwards via `reset()`."""
        need = self.latency_samples + _RENDER_BLOCK
        silence = np.zeros(_RENDER_BLOCK, dtype=np.float32)
        out = [self._effect.process(silence) for _ in range(-(-need // _RENDER_BLOCK))]
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        """Clear all streaming state, keeping the configuration. Output after
        a reset is deterministic (the effect's RNGs are re-seeded)."""
        self._effect.reset()


def render(
    mono: np.ndarray,
    sample_rate: int,
    preset: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
    tail_s: float = 0.0,
) -> np.ndarray:
    """Render a whole clip through a fresh `VoiceFilter`.

    Returns `len(mono) + tail_s * sample_rate` samples, time-aligned with the
    input. Use `tail_s` to keep a reverb tail ringing past the dry clip's end.
    Deterministic: the same input and settings always give the same output.
    """
    vf = VoiceFilter(sample_rate, preset=preset, params=params)
    signal = np.asarray(mono, dtype=np.float32).reshape(-1)
    want = signal.shape[0] + int(round(max(tail_s, 0.0) * sample_rate))

    chunks = [vf.process(signal[i:i + _RENDER_BLOCK]) for i in range(0, signal.shape[0], _RENDER_BLOCK)]
    have = sum(c.shape[0] for c in chunks)
    silence = np.zeros(_RENDER_BLOCK, dtype=np.float32)
    max_blocks = int(_MAX_FLUSH_S * sample_rate) // _RENDER_BLOCK + want // _RENDER_BLOCK
    for _ in range(max_blocks):
        if have >= want:
            break
        c = vf.process(silence)
        chunks.append(c)
        have += c.shape[0]

    out = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    if out.shape[0] < want:
        out = np.concatenate([out, np.zeros(want - out.shape[0], dtype=np.float32)])
    return out[:want]
