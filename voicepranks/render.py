"""Render a WAV file through the voice effect (no audio hardware needed).

Usage:
    python -m voicepranks.render IN.wav OUT.wav --preset scary
    python -m voicepranks.render IN.wav OUT.wav --preset minion --param shuffle.semitones=7
    python -m voicepranks.render IN.wav OUT.wav --params voice.json --tail-s 1.5

Input is 16-bit PCM WAV (first channel if multichannel); output is mono
16-bit at the input's sample rate, the same length as the input plus
`--tail-s`. Convert other formats first, e.g.
`ffmpeg -i in.mp3 -ac 1 -ar 48000 -c:a pcm_s16le in.wav`.

`--params` is a JSON object of `{param: value}` using `PARAM_SPECS` names;
`--param NAME=VALUE` entries are applied on top of it, and both override the
preset. This is a thin CLI over `voicepranks.api.render`.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

from .api import preset_names, render
from .audio.wavio import read_wav, write_wav


def _parse_value(text: str) -> Any:
    """Parse a `--param` value as JSON (numbers, true/false), else keep the string."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Render a WAV file through the VoicePranks voice effect.")
    parser.add_argument("input", help="input WAV (16-bit PCM)")
    parser.add_argument("output", help="output WAV (mono 16-bit)")
    parser.add_argument("--preset", choices=preset_names(), help="named preset to start from")
    parser.add_argument("--params", help="JSON file of {param: value} overrides")
    parser.add_argument(
        "--param", action="append", default=[], metavar="NAME=VALUE", help="one param override (repeatable)"
    )
    parser.add_argument("--tail-s", type=float, default=0.0, help="seconds of effect tail to keep past the input's end")
    args = parser.parse_args(argv)

    params: Dict[str, Any] = {}
    if args.params:
        with open(args.params, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            parser.error("--params must be a JSON object")
        params.update(loaded)
    for item in args.param:
        name, sep, value = item.partition("=")
        if not sep:
            parser.error(f"--param expects NAME=VALUE, got {item!r}")
        params[name.strip()] = _parse_value(value.strip())

    sample_rate, mono = read_wav(args.input)
    try:
        out = render(mono, sample_rate, preset=args.preset, params=params, tail_s=args.tail_s)
    except (KeyError, ValueError) as e:
        parser.error(str(e))
    write_wav(args.output, sample_rate, out)
    print(f"{args.input} -> {args.output}: {out.shape[0] / sample_rate:.2f} s @ {sample_rate} Hz")


if __name__ == "__main__":
    main()
