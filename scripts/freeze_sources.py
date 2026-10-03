#!/usr/bin/env python3
"""Freeze the runtime sources used by this book, including local changes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tomllib

BOOK = Path(__file__).resolve().parents[1]
NAMES = ("MetricSpaces", "TDAPersistenceDiagrams", "TDARipserer",
         "TDAmapper", "TDAplots", "ToMATo")


def git(path, *args):
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True).strip()


def runtime_files(root):
    paths = [root / "Project.toml"]
    paths.extend(root.glob("LICENSE*"))
    if (root / "Artifacts.toml").exists():
        paths.append(root / "Artifacts.toml")
    for directory in ("src", "ext", "deps"):
        if (root / directory).exists():
            paths.extend(p for p in (root / directory).rglob("*") if p.is_file())
    return sorted(set(paths), key=lambda p: p.relative_to(root).as_posix())


def digest(root, paths):
    result = hashlib.sha256()
    for path in paths:
        result.update(path.relative_to(root).as_posix().encode())
        result.update(b"\0")
        result.update(hashlib.sha256(path.read_bytes()).digest())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=BOOK.parent)
    args = parser.parse_args()
    output = BOOK / "environment"
    output.mkdir(exist_ok=True)
    archive = output / "package-sources.tar"
    records = []
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as tar:
        for name in NAMES:
            root = args.package_root / (name + ".jl")
            paths = runtime_files(root)
            project = tomllib.loads((root / "Project.toml").read_text())
            url = git(root, "remote", "get-url", "origin")
            if url.startswith("git@github.com:"):
                url = "https://github.com/" + url.split(":", 1)[1]
            records.append(dict(
                name=name, uuid=project["uuid"], version=project["version"],
                url=url, revision=git(root, "rev-parse", "HEAD"),
                includes_local_changes=bool(git(root, "status", "--porcelain")),
                source_sha256=digest(root, paths)))
            for path in paths:
                data = path.read_bytes()
                member = tarfile.TarInfo(
                    f"{name}.jl/{path.relative_to(root).as_posix()}")
                member.size = len(data)
                member.mode = path.stat().st_mode & 0o777
                member.mtime = member.uid = member.gid = 0
                tar.addfile(member, io.BytesIO(data))
    lines = [
        "# Runtime source snapshots; the hashes include the listed local changes.",
        "format_version = 1",
        'archive = "package-sources.tar"',
        f"archive_sha256 = {json.dumps(hashlib.sha256(archive.read_bytes()).hexdigest())}",
    ]
    for record in records:
        lines.append(f"\n[packages.{record.pop('name')}]")
        for key, value in record.items():
            lines.append(f"{key} = {str(value).lower() if isinstance(value, bool) else json.dumps(value)}")
    (output / "source-lock.toml").write_text("\n".join(lines) + "\n")
    print(f"Frozen {len(records)} packages in {archive} ({archive.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
