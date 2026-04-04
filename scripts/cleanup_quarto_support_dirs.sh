#!/usr/bin/env bash
set -euo pipefail

shopt -s nullglob

# Quarto's current execution outputs live under _freeze/.
# Remove stale top-level *_files folders so ad hoc renders do not clutter the repo root.
for qmd in *.qmd; do
  support_dir="${qmd%.qmd}_files"
  if [[ -d "${support_dir}" ]]; then
    rm -rf -- "${support_dir}"
  fi
done
