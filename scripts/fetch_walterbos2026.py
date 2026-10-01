"""Fetch the Walterbos et al. (2026) HSE06 halide double perovskite table and write the columns we use.

Source: L. Walterbos, A. McEwan, R. Shinde, J. George, "Spin-Polarized Electronic Structure and Chemical
Bonding Data for 2,500+ Halide Double Perovskites", arXiv:2606.11928 (doi 10.48550/arxiv.2606.11928).
Archive: Zenodo record 20598121 (doi 10.5281/zenodo.20598121), "Luccerboi/HDP_WorkFLow_Analysis v0.1.0",
licence CC BY 4.0. File used: AnalysisResults/HDP_CombinedInfo_260510.csv.

Output (committed, CC BY 4.0 attribution in data/external/walterbos2026/README.md):
    data/external/walterbos2026/hdp_hse06_subset.csv
Columns: comp_name_full, element.B1, element.B2, element.X, bandgap (HSE06, spin-polarised, eV),
cond_type, spin_forbidden, geom_stable. Nothing is transformed; rows and values are copied as published.

Run:  python scripts/fetch_walterbos2026.py      (network; about 60 MB download)
The sha256 of the source CSV is checked so a changed upstream file fails loudly instead of silently.
"""
from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/external/walterbos2026"
URL = "https://zenodo.org/records/20598121/files/Luccerboi/HDP_WorkFLow_Analysis-v0.1.0.zip?download=1"
MEMBER = "Luccerboi-HDP_WorkFLow_Analysis-bb977ab/AnalysisResults/HDP_CombinedInfo_260510.csv"
SRC_SHA256 = None  # filled from the first verified download; see README.md
COLS = ["comp_name_full", "element.B1", "element.B2", "element.X", "bandgap", "cond_type", "spin_forbidden",
        "geom_stable"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("downloading", URL)
    req = urllib.request.Request(URL, headers={"User-Agent": "perovskite-screening-ml-tea/2 (+https://github.com/havidaqoma/perovskite-screening-ml-tea)"})
    blob = urllib.request.urlopen(req, timeout=600).read()
    raw = zipfile.ZipFile(io.BytesIO(blob)).read(MEMBER)
    digest = hashlib.sha256(raw).hexdigest()
    expected = (OUT / "SOURCE_SHA256").read_text().strip() if (OUT / "SOURCE_SHA256").exists() else SRC_SHA256
    if expected and digest != expected:
        print(f"FAIL: source CSV sha256 {digest} != recorded {expected}")
        return 1
    df = pd.read_csv(io.BytesIO(raw))
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        print("FAIL: upstream columns missing:", missing)
        return 1
    df[COLS].to_csv(OUT / "hdp_hse06_subset.csv", index=False, lineterminator="\n")
    (OUT / "SOURCE_SHA256").write_text(digest + "\n")
    print(f"wrote {len(df)} rows to {OUT / 'hdp_hse06_subset.csv'}; source sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
