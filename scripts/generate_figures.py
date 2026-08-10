#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dufppi.figures import generate_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    generate_all(args.release_root)


if __name__ == "__main__":
    main()
