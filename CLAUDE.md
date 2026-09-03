# CLAUDE.md

## Project Status

VoicePranks is a real-time "Minion voice" microphone filter for macOS and
Windows. It captures the physical mic, transforms it with hand-rolled DSP, and
writes to a virtual audio device so Discord / Zoom / OBS see it as a microphone
input. Tech decisions, and why:

- **Pure Python, numpy + sounddevice only.** The DSP is hand-rolled — **no scipy,
  no librosa**. Keeps the PyInstaller bundles small and the install story for
  non-developers to "unzip and double-click".
- **The virtual cable is a system driver, never bundled.** BlackHole on macOS,
  VB-CABLE on Windows. Users install it themselves; the app only selects it.
- **One parameter registry drives everything.** See the invariant below.

### Layout

- `voicepranks/params.py` — `PARAM_SPECS`, the **single source of truth** for
  every tunable knob (name, group, label, kind, range, default, whether applying
  rebuilds a DSP stage). Params use a dotted `group.field` convention
  (`minionese.chunk_ms`); the `global` group has no prefix.
- `voicepranks/dsp/` — the DSP stages. `effect.py` is the chain; `pitch.py` is
  the WSOLA pitch shifter; `biquad.py` (`PeakingEQ`), `ramp.py`, `distortion.py`,
  `reverb.py`, `growl.py`, `laugh.py`, and the two gibberish engines
  `minionese.py` (STFT formant-replacement) and `minionese_shuffle.py`
  (time-domain WSOLA + syllable shuffle).
- `voicepranks/audio/` — `engine.py` (the `VoiceEngine`, driven via
  `set_param`/`snapshot`/`apply_preset`), `devices.py`, `wavio.py`.
- `voicepranks/ui/app.py` — Tkinter GUI. `control_server.py` + `webui/` — the
  HTTP control surface. Both drive the engine *only* through `PARAM_SPECS`.
- `voicepranks/presets.py` — curated `{param: value}` bundles of sound character.
- `tests/` — pytest, one module per concern.
- `build-macos.sh`, `build-windows.ps1`, `voicepranks.spec`, `DISTRIBUTING.md` —
  the PyInstaller bundle pipeline.

### Build / run / test

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m voicepranks              # run the GUI
pytest                             # the test suite (testpaths=tests)
pytest tests/test_pitch.py -k wsola # one area
./build-macos.sh                   # build the distributable .app
```

### Invariants to preserve

- **`PARAM_SPECS` is the only knob list.** Add a parameter there and it appears
  in the Tkinter UI and the HTTP control server automatically. Never hardcode a
  param in `ui/app.py` or `control_server.py`, and never reintroduce a module
  constant the UI can't see.
- **No scipy / librosa.** DSP stays hand-rolled on numpy. A new dependency here
  is an architectural change, not a convenience.
- **The pitch shift is WSOLA, flat — no EQ.** This was settled by ear during live
  tuning: WSOLA alone reads as correct, and the presence-EQ boost slightly hurt.
  The `PeakingEQ` stage is left wired at 0 dB so it can return without touching
  `engine.py`/`ui/app.py`. **Do not re-litigate this** — granular and
  phase-vocoder pitch engines and the EQ boost were all evaluated and rejected.
- **Presets set sound character only.** They never touch `io.*`, `enabled`,
  `monitor`, or `intensity` — applying a preset must not disturb device
  selection, on/off state, or the live master `intensity` control.
- **Bundled assets resolve through `resources.py`.** Frozen builds read from
  `_MEIPASS`, not the source tree; a bare relative path works in dev and breaks
  in the shipped app. Any new asset goes through the resolver *and* the spec.
- **Defaults keep the plain voice unchanged.** `Distortion` and `Reverb` are off
  by default; the "scary" preset is what turns them on alongside a negative
  `max_semitones`.
