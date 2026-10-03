#!/usr/bin/env bash
set -euo pipefail

# An incremental render must not clean other chapters' staging directories.
# Limit cleanup to this render's outputs, and verify published asset copies.
python3 - <<'PY'
import filecmp
import os
from pathlib import Path
import shutil

output = Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", "docs"))
rendered = os.environ.get("QUARTO_PROJECT_OUTPUT_FILES", "").splitlines()
for rendered_file in rendered:
    try:
        relative = Path(rendered_file).resolve().relative_to(output.resolve())
    except ValueError:
        continue
    if relative.parent != Path(".") or relative.suffix != ".html":
        continue
    qmd = relative.with_suffix(".qmd")
    if not qmd.is_file():
        continue
    source = qmd.with_name(qmd.stem + "_files")
    if not source.is_dir():
        continue
    destination = output / source.name
    files = [
        path for path in source.rglob("*") if path.is_file()
        and "execute-results" not in path.relative_to(source).parts
    ]
    if not files:
        continue
    if all(
        (destination / path.relative_to(source)).is_file()
        and filecmp.cmp(path, destination / path.relative_to(source), shallow=False)
        for path in files
    ):
        shutil.rmtree(source)
PY
