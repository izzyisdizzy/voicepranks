"""Tests for the cross-language contract: `export_spec` and `golden`."""
from __future__ import annotations

import json
import os

from voicepranks.audio.wavio import read_wav
from voicepranks.export_spec import build_spec
from voicepranks.golden import write_golden
from voicepranks.params import PARAM_SPECS
from voicepranks.presets import PRESETS


def test_spec_round_trips_every_param_and_preset():
    spec = json.loads(json.dumps(build_spec()))
    assert [p["name"] for p in spec["params"]] == [s.name for s in PARAM_SPECS]
    first = spec["params"][0]
    assert set(first) == {"name", "group", "label", "kind", "min", "max", "step", "default", "needs_reset"}
    assert spec["presets"] == {k: dict(v) for k, v in PRESETS.items()}


def test_golden_writes_manifest_and_wavs(tmp_path):
    manifest = write_golden(str(tmp_path), sample_rate=16000, seconds=0.3)
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["sample_rate"] == 16000
    assert {r["case"] for r in manifest["renders"]} >= {"pitch_up", "reverb", "preset_minion"}
    # Cases record what actually rendered: the laugh is off even for goofy.
    assert manifest["cases"]["preset_goofy"]["laugh.enabled"] is False
    assert manifest["cases"]["pitch_up"]["distortion.enabled"] is False
    for r in manifest["renders"]:
        sr, y = read_wav(os.path.join(tmp_path, r["file"]))
        _, x = read_wav(os.path.join(tmp_path, r["input"]))
        assert sr == 16000 and y.shape == x.shape
        assert len(r["rms_envelope"]) == y.shape[0] // 320
