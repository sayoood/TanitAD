#!/usr/bin/env python3
"""Re-root the navtest metric cache's metadata CSV after the navsim archive MOVED (P6 layout fix, 2026-11-06). Path roots only.

WHY. `navsim`'s MetricCacheLoader reads <cache>/metadata/*.csv and opens every `metric_cache.pkl` at the ABSOLUTE path the CSV row names
(written by the devkit as str(WindowsPath), e.g. `D:\\Archive\\devbox-C\\navsim\\exp\\w3_navtest_v1\\metric_cache_navtest\\<log>\\unknown\\<token>\\metric_cache.pkl`).
Move the archive and all 12,146 rows dangle: the harness then cannot score anything. This tool rewrites ONLY the root prefix of each row, as BYTES, so
the header, the CRLF line endings, the ROW ORDER and every byte after the prefix are unchanged (a regenerated CSV via write_cache_metadata.py would
sort the rows; preserving the order means no downstream step can depend on a changed one, so nothing is re-derived here).

    python eval/rewrite_cache_metadata_root.py [--to-root F:/Archive/devbox-C/navsim] [--dry-run]

The target root defaults to the same resolver the read path uses ($TANITAD_NAVSIM_ROOT, else the first existing of E:/ F:/ D:/ Archive/devbox-C/navsim).
The source root is read from the CSV itself (everything before `exp\\w3_navtest_v1`). The original is copied ONCE to
`<cache>/metadata_bak_dpaths_<stamp>/` (NOT into metadata/: the loader opens the first file whose name CONTAINS ".csv", so a `*.csv.bak` beside the CSV is a trap);
an existing backup is never overwritten. Verified on CONTENT, refuses (exit 1, original untouched) unless: row count unchanged, every new path exists on disk,
12,146 distinct tokens equal the old ones, and every row differs from its old row in the root prefix only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time


def navsim_root() -> str:
    r = os.environ.get("TANITAD_NAVSIM_ROOT")
    cands = [r.replace("\\", "/").rstrip("/")] if r else [f"{d}:/Archive/devbox-C/navsim" for d in "EFD"]
    for c in cands:
        if os.path.isdir(c):
            return c
    raise SystemExit("REFUSING: navsim archive root not found (tried: " + ", ".join(cands) + "); set TANITAD_NAVSIM_ROOT. No silent default.")


def reroot(data: bytes, to_root: str):
    """returns (new_bytes, from_root, n_rows). `to_root` is a forward-slash root; rows are rewritten with backslashes like the devkit's own."""
    lines = data.split(b"\r\n") if b"\r\n" in data else data.split(b"\n")
    sep = b"\r\n" if b"\r\n" in data else b"\n"
    marker = b"\\exp\\w3_navtest_v1\\"
    first = next(ln for ln in lines[1:] if ln.strip())
    k = first.find(marker)
    if k < 0:
        raise SystemExit("REFUSING: the first row does not contain \\exp\\w3_navtest_v1\\ : not a navtest metric-cache metadata CSV")
    from_root = first[:k]
    to = to_root.replace("/", "\\").encode("utf-8")
    out, n = [lines[0]], 0
    for ln in lines[1:]:
        if not ln.strip():
            out.append(ln)
            continue
        if not ln.startswith(from_root + marker):
            raise SystemExit(f"REFUSING: a row does not start with the source root {from_root!r}: {ln[:120]!r}")
        out.append(to + ln[len(from_root):])
        n += 1
    return sep.join(out), from_root.decode("utf-8"), n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--to-root", default=None, help="forward-slash navsim root; default = the read path's resolver")
    ap.add_argument("--cache-name", default="navtest")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    to_root = (a.to_root or navsim_root()).replace("\\", "/").rstrip("/")
    cache = f"{to_root}/exp/w3_navtest_v1/metric_cache_{a.cache_name}"
    mdir = f"{cache}/metadata"
    csvs = [f for f in sorted(os.listdir(mdir)) if f.endswith(".csv")]
    if len(csvs) != 1:
        raise SystemExit(f"REFUSING: expected exactly 1 *.csv in {mdir}, found {csvs}")
    csv_p = f"{mdir}/{csvs[0]}"
    old = open(csv_p, "rb").read()
    new, from_root, n = reroot(old, to_root)
    rows_old = [r for r in old.decode("utf-8").splitlines()[1:] if r.strip()]
    rows_new = [r for r in new.decode("utf-8").splitlines()[1:] if r.strip()]
    toks = lambda rows: [re.split(r"[\\/]", r.strip())[-2] for r in rows]          # noqa: E731
    rep = {"csv": csv_p, "from_root": from_root, "to_root": to_root.replace("/", "\\"), "n_rows_old": len(rows_old), "n_rows_new": len(rows_new),
           "n_rows_new_exist_on_disk": sum(1 for r in rows_new if os.path.exists(r.strip())),
           "n_distinct_tokens_new": len(set(toks(rows_new))), "tokens_equal_old_in_order": toks(rows_new) == toks(rows_old),
           "suffix_identical_all_rows": all(o[len(from_root):] == n_[len(to_root):] for o, n_ in zip(rows_old, rows_new)),
           "header_identical": old.split(b"\n")[0] == new.split(b"\n")[0], "crlf_preserved": (b"\r\n" in old) == (b"\r\n" in new),
           "old_sha256": hashlib.sha256(old).hexdigest(), "new_sha256": hashlib.sha256(new).hexdigest(), "dry_run": bool(a.dry_run)}
    rep["PASS"] = bool(rep["n_rows_old"] == rep["n_rows_new"] == rep["n_rows_new_exist_on_disk"] == rep["n_distinct_tokens_new"]
                       and rep["tokens_equal_old_in_order"] and rep["suffix_identical_all_rows"] and rep["header_identical"] and rep["crlf_preserved"])
    if from_root.replace("\\", "/") == to_root:
        rep["note"] = "already rooted at the target: nothing to do"
        print(json.dumps(rep, indent=1))
        return 0 if rep["PASS"] else 1
    if not rep["PASS"] or a.dry_run:
        print(json.dumps(rep, indent=1))
        return 0 if (rep["PASS"] and a.dry_run) else 1
    bak_dir = f"{cache}/metadata_bak_dpaths_{time.strftime('%Y%m%d')}"
    bak = f"{bak_dir}/{csvs[0]}.bak_dpaths_{time.strftime('%Y%m%d')}"
    os.makedirs(bak_dir, exist_ok=True)
    if not os.path.exists(bak):
        shutil.copy2(csv_p, bak)
    assert open(bak, "rb").read() == old, "the backup does not equal the original: refusing to overwrite"
    tmp = f"{mdir}/_rewrite.tmp"                                                      # no ".csv" in the name: the loader must never see it
    with open(tmp, "wb") as fh:
        fh.write(new)
    os.replace(tmp, csv_p)
    assert open(csv_p, "rb").read() == new
    rep["backup"] = bak
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
