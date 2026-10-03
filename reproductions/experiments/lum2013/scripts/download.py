#!/usr/bin/env python3
"""Acquire original GEO inputs and the series-level overall-relapse table.

Run from any directory. Python stdlib only; existing source bytes are verified
against sources.toml rather than silently replaced. --refresh is explicit.
"""
from pathlib import Path
import gzip, hashlib, sys, tomllib, urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCES = {
    "GSE2034_series_matrix.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE2nnn/GSE2034/matrix/GSE2034_series_matrix.txt.gz",
    "GSE2034_family.soft.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE2nnn/GSE2034/soft/GSE2034_family.soft.gz",
    "GPL96.annot.gz": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz",
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

DATA.mkdir(parents=True, exist_ok=True)
record_path = DATA / "sources.toml"
record = tomllib.loads(record_path.read_text()) if record_path.exists() else {}
for filename, url in SOURCES.items():
    target = DATA / filename
    if not target.exists() or "--refresh" in sys.argv:
        urllib.request.urlretrieve(url, target)
    actual = digest(target)
    expected = record.get(filename, {}).get("sha256")
    if expected is not None and expected != actual and "--refresh" not in sys.argv:
        raise RuntimeError(f"Changed source bytes: {filename}. Investigate before --refresh.")
    print(filename, target.stat().st_size, actual)

table = []
inside = False
with gzip.open(DATA / "GSE2034_family.soft.gz", "rt") as stream:
    for line in stream:
        if line.startswith("!series_table_begin"):
            inside = "Patient clinical parameters" in line
        elif line.startswith("!series_table_end"):
            inside = False
        elif inside:
            table.append(line)
assert len(table) == 287, "Expected a header and 286 clinical records"
clinical = DATA / "clinical.tsv"
clinical.write_text("".join(table))

with record_path.open("w") as stream:
    stream.write('accession = "GSE2034"\nplatform = "GPL96"\nretrieval_date = "2026-10-03"\n')
    for filename, url in SOURCES.items():
        stream.write(f'\n["{filename}"]\nurl = "{url}"\nsha256 = "{digest(DATA / filename)}"\nbytes = {(DATA / filename).stat().st_size}\n')
    stream.write(f'\n["clinical.tsv"]\nsource = "GSE2034_family.soft.gz named series table"\nsha256 = "{digest(clinical)}"\nrows = 286\n')
print("clinical.tsv: 286 records; labels are overall relapse, not bone relapse")
