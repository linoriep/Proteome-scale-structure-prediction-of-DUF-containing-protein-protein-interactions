#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXCLUDED_PARTS = {".git", ".venv", "__pycache__", "results"}
EXCLUDED_SUFFIXES = {".pyc"}
OUTPUTS = {Path("metadata/checksums.sha256"), Path("metadata/file_manifest.tsv")}


def included_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file() or relative in OUTPUTS:
            continue
        if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        files.append(relative)
    return sorted(files, key=lambda path: path.as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.release_root.resolve()
    rows = []
    for relative in included_files(root):
        content = (root / relative).read_bytes()
        rows.append((relative.as_posix(), len(content), hashlib.sha256(content).hexdigest()))

    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    checksums = "".join(f"{digest}  {path}\n" for path, _, digest in rows)
    manifest = "path\tsize_bytes\tsha256\n" + "".join(
        f"{path}\t{size}\t{digest}\n" for path, size, digest in rows
    )
    (metadata / "checksums.sha256").write_text(checksums)
    (metadata / "file_manifest.tsv").write_text(manifest)
    print(f"Recorded {len(rows)} files")


if __name__ == "__main__":
    main()
