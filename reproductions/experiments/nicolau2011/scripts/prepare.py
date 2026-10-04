#!/usr/bin/env python3
"""Recover original NKI295/BCN13 data; construct auditable Julia inputs.

Python standard library only, plus R/readxl for the archived XLS clinical table.
The checked-in derived inputs make R unnecessary for subsequent Julia runs.
Large archives stay in data/raw (ignored). No downloaded code is executed.
"""
import csv
import gzip
import hashlib
import io
import json
import pathlib
import platform
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
COMMIT = "2914487d9138dce2175f3a1e7918277187a11b87"
URLS = {
    "seventyGeneData_1.40.0.tar.gz": "https://bioconductor.posit.co/packages/3.19/data/experiment/src/contrib/seventyGeneData_1.40.0.tar.gz",
    "NormalBreastData.zip": f"https://raw.githubusercontent.com/monkgroupie/publication_code/{COMMIT}/NormalBreastData.zip",
    "GPL2567_family.soft.gz": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL2nnn/GPL2567/soft/GPL2567_family.soft.gz",
    "author_dsga.R": f"https://raw.githubusercontent.com/monkgroupie/publication_code/{COMMIT}/reuten.DSGA_decomposition.nicolau.R",
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def gzwrite(path, text):
    # Set gzip mtime/name explicitly to preserve byte-for-byte deterministic inputs.
    with open(path, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as z:
        z.write(text.encode())


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    previous = json.loads((DATA / "sources.json").read_text()) if (DATA / "sources.json").exists() else {}
    sources = {}
    for name, url in URLS.items():
        path = RAW / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        sources[name] = dict(url=url, sha256=sha(path), bytes=path.stat().st_size)
        if name in previous.get("sources", {}):
            assert sources[name]["sha256"] == previous["sources"][name]["sha256"], name
    # The independent original platform records exactly the 24,479 assay probes.
    # Ignore sample sections in the large SOFT family archive.
    platform_rows = []
    in_table = False
    with gzip.open(RAW / "GPL2567_family.soft.gz", "rt") as f:
        for line in f:
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if in_table:
                platform_rows.append(line.rstrip("\n"))
    table = list(csv.DictReader(platform_rows, delimiter="\t"))
    probes = {r["ACCESSION"] for r in table}
    assert len(table) == len(probes) == 24479
    (DATA / "platform_probes.tsv").write_text("ACCESSION\tGENE_SYMBOL\n" + "".join(f'{r["ACCESSION"]}\t{r["GENE_SYMBOL"]}\n' for r in table))
    # Extract only the original tumor zip, clinical XLS and annotation ExpressionSet.
    with tempfile.TemporaryDirectory(prefix="nicolau2011-", dir="/mnt/Dados/tmp") as tmp:
        temp = pathlib.Path(tmp)
        with tarfile.open(RAW / "seventyGeneData_1.40.0.tar.gz") as tar:
            def extract(name):
                b = tar.extractfile("seventyGeneData/" + name).read()
                out = temp / pathlib.Path(name).name
                out.write_bytes(b)
                sources[pathlib.Path(name).name] = dict(archive="seventyGeneData_1.40.0.tar.gz", member="seventyGeneData/" + name, sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))
                return out
            expression_zip = extract("inst/extdata/vanDeVijver/ZipFiles295Samples.zip")
            clinical_zip = extract("inst/extdata/vanDeVijver/nejm_table1.zip")
            annotation_rda = extract("data/vanDeVijver.rda")
        with zipfile.ZipFile(clinical_zip) as z:
            clinical_xls = temp / "Table1_ClinicalData_Table.xls"
            b = z.read(clinical_xls.name)
            clinical_xls.write_bytes(b)
            sources[clinical_xls.name] = dict(archive="nejm_table1.zip", sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))
        r_info = subprocess.check_output(["Rscript", str(ROOT / "scripts/export_annotation.R"), str(annotation_rda), str(DATA / "nki_annotation.tsv"), str(clinical_xls), str(DATA / "clinical.tsv")], text=True).strip()
        ids, raw_probes, values, validity = [], [], [], []
        with zipfile.ZipFile(expression_zip) as z:
            for index in range(1, 7):
                name = f"Table_NKI_295_{index}.txt"
                b = z.read(name)
                sources[name] = dict(archive="ZipFiles295Samples.zip", sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))
                rows = csv.reader(io.StringIO(b.decode()), delimiter="\t")
                h = next(rows)
                ids.extend(h[2::5])
                next(rows)
                count = 0
                for row in rows:
                    if index == 1:
                        raw_probes.append(row[0]); values.append([]); validity.append([])
                    else:
                        assert raw_probes[count] == row[0]
                    for j in range(2, len(row) - 4, 5):
                        values[count].append(row[j].strip())
                        validity[count].append(row[j + 4].strip() == "1")
                    count += 1
                assert count == 24496
        assert len(ids) == len(set(ids)) == 295
        header = "probe\t" + "\t".join(ids) + "\n"
        kept, missing, excluded, low_quality = [], 0, [], []
        for probe, row, valid in zip(raw_probes, values, validity):
            if probe not in probes:
                excluded.append(probe)
                continue
            if sum(valid) / 295 < 0.70:
                low_quality.append(probe)
                continue
            kept.append(probe)
            missing += len(valid) - sum(valid)
            header += probe + "\t" + "\t".join(v if good else "NaN" for v, good in zip(row, valid)) + "\n"
        assert len(kept) == 24453 and len(low_quality) == 26 and len(excluded) == 17
        gzwrite(DATA / "nki_log10.tsv.gz", header)
        (DATA / "excluded_controls.txt").write_text("\n".join(excluded) + "\n")
        (DATA / "low_quality_probes.txt").write_text("\n".join(low_quality) + "\n")
    with zipfile.ZipFile(RAW / "NormalBreastData.zip") as z:
        for name in ("BCN.ugc219.pcl", "BCN.70fltr.report.html", "BCN.Samples.xlsx"):
            b = z.read("NormalBreastData/" + name)
            sources[name] = dict(archive="NormalBreastData.zip", sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))
            if name.endswith(".pcl"):
                gzwrite(DATA / (name + ".gz"), b.decode())
            elif name.endswith(".html"):
                (DATA / name).write_bytes(b)
    metadata = dict(python=platform.python_version(), r=r_info, normal_repository_commit=COMMIT,
                    raw_nki_rows=24496, assay_probes=len(probes), retained_probes=len(kept), excluded_controls=excluded, low_quality_probes=low_quality,
                    patients=len(ids), missing_valid_measurements=missing,
                    mapping="NKI HUGO.gene.symbol from seventyGeneData to unique exact BCN build219 gene symbols; NOT historical NKI UniGene219 mapping",
                    sources=sources)
    metadata["derived"] = {p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(DATA.iterdir()) if p.is_file() and p.name not in ("sources.json","checksums.toml")}
    (DATA / "checksums.toml").write_text("".join(f'"{n}" = "{v["sha256"]}"\n' for n,v in metadata["derived"].items()))
    (DATA / "sources.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k:v for k,v in metadata.items() if k not in ("sources","derived")},indent=2))


if __name__ == "__main__":
    main()
