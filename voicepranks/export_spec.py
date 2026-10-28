"""Export the parameter registry and presets as JSON.

Usage:
    python -m voicepranks.export_spec OUT.json

This is the contract for ports of the effect to other languages (e.g. a C#
voice-chat filter): `params` mirrors `PARAM_SPECS` and `presets` mirrors
`PRESETS`, so a port can load the same presets instead of hand-copying them
and drifting.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Any, Dict, Optional

from . import __version__
from .params import PARAM_SPECS
from .presets import PRESETS


def build_spec() -> Dict[str, Any]:
    return {
        "version": __version__,
        "params": [dataclasses.asdict(spec) for spec in PARAM_SPECS],
        "presets": {name: dict(values) for name, values in PRESETS.items()},
    }


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Export PARAM_SPECS + PRESETS as JSON.")
    parser.add_argument("output", help="JSON file to write ('-' for stdout)")
    args = parser.parse_args(argv)
    text = json.dumps(build_spec(), indent=2) + "\n"
    if args.output == "-":
        print(text, end="")
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
