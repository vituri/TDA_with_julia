# Reproducing this book

The examples were checked with Julia 1.12.5 and Quarto 1.9.35. Install Julia,
Quarto, and a Python environment with `jupyter` and `jupyter-cache`, then run
from the book directory:

```sh
julia scripts/setup.jl
quarto render --cache-refresh
```

The setup script verifies and extracts the six JuliaTDA source snapshots into
the ignored `.book-packages/` directory, activates the book project, installs
its dependencies, and registers the `tda-book` Jupyter kernel used by Quarto.
`Project.toml` records direct dependencies; `Manifest.toml` pins the remaining
Julia dependencies. The first setup includes compilation and downloads, and
the digit chapter also downloads MNIST on its first run. Quarto uses the Jupyter
engine explicitly; executable chapter frontmatter also names the kernel so a
standalone chapter render cannot select an unrelated Julia environment. See
[Quarto’s Julia setup instructions](https://quarto.org/docs/computations/julia.html#using-the-jupyter-engine)
if Jupyter is not yet installed.

`package-sources.tar` contains runtime source files and original licenses, not
compiled code. `source-lock.toml` records package identities, repository
revisions, source hashes, and the archive hash. These snapshots include local
changes, including the TDARipserer and TDAPersistenceDiagrams module renames.
The recorded Git revisions alone cannot recreate those changes; the archive
is the reproducible source for this edition. Setup checks an existing extracted
snapshot and refuses to overwrite edits.

## Working on the packages

Authors with all six package repositories beside the book can instead run:

```sh
julia scripts/setup.jl --local
```

Set `TDA_PACKAGE_ROOT` to use a different parent directory. This intentionally
uses the current development sources and rewrites the manifest's local paths.
Before publishing a new edition, freeze the intended package sources and
restore the portable manifest:

```sh
python3 scripts/freeze_sources.py
# Move aside an older .book-packages/ directory if the snapshot has changed.
julia scripts/setup.jl
quarto render --cache-refresh
```

Keep the archive, lock file, Project, and Manifest together. A reader copying
the book does not need the author's sibling repositories. `--no-kernel` skips
kernel registration when only Julia scripts will be run.
