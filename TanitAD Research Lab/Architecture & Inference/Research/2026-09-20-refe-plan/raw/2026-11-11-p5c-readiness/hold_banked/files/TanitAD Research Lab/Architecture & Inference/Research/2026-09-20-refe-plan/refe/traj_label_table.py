#!/usr/bin/env python3
"""A ROW -> TRAJECTORY LABEL TABLE for REFe's student trainer (P5-C, 2026-11-11; PLAN: scorer_gym_20261107/p5c/P5C_READINESS.md section 3).

WHY. P5-C trains the student on the P5-S tracker-aware labels (Tc20s, cubic) INSTEAD of each bank row's raw teacher `traj`.  The labeller (a separate tool) writes one [20, 3]
label per bank row; this module banks them as a gzipped JSON and is the ONE reader the trainer (`train.py --traj-label-table`) uses.  The trainer's loss is unchanged: only the
tensor that `TargetBank.__getitem__` returns in slot 3 differs.  Imports only the stdlib (nothing numeric), like ego_yaw_table.py.

THE KEY.  A bank row is identified by (log_name, token, rank) and PROVEN to be the row the label was computed from by the sha256 of the row's OWN `traj` bytes (each float as a
little-endian float64, row-major, `traj_sha256` below).  Rank-1 goal twins share the rank-0 token, so a token key would collide; the sha makes the label provably the one computed
from the row it replaces.  The table stores key = "log|token|rank|sha256" -> label.  Three outcomes at lookup (`Table.lookup`):
  * the full key is present                                  -> the label;
  * (log, token, rank) is present under ANOTHER traj sha256   -> REFUSE (the table was built from a different traj than this row's: silent substitution is the failure);
  * (log, token, rank) is absent                              -> None: the row KEEPS its raw `traj`, and the trainer reports the coverage.
A table may declare `require_coverage` (a fraction in meta, set at build time): the trainer REFUSES a bank the table covers less than that.  Because it lives in the table it is
bound by the table's sha256, so no second trainer flag exists.

THE TABLE'S sha256 is the sha256 of the FILE (deterministic: sorted keys, gzip mtime 0, no wall clock); the trainer prints it at startup and writes it into the checkpoint meta and run identity.

    python traj_label_table.py build    --bank-dir <dir> --labels <jsonl> --out T.json.gz [--require-coverage 1.0] [--label-name Tc20s] [--allow-partial]
        <jsonl>: one object per line {"log_name", "token", "rank", "label": [[x, y, yaw] x T], "traj_sha256" (optional: if present it must equal the bank row's)}
    python traj_label_table.py identity --bank-dir <dir> --out T.json.gz           # label := the row's own traj (the control table: must reproduce flag-off bit for bit)
    python traj_label_table.py verify   --table T.json.gz --bank-dir <dir>         # check-bank: coverage, sha mismatches, shape / finite / yaw-branch; rc 0 only if no refusal
    python traj_label_table.py info     --table T.json.gz
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SCHEMA = "traj-label-table-v1"
YAW_COL = 2                                     # a trajectory pose is (x, y, yaw)
MAX_DYAW = math.pi                              # |label yaw - raw yaw| must stay below this on every pose (a larger gap is a wrap-branch mismatch, ~2 pi)


class TableRefused(Exception):
    """a table that is malformed, built from different trajectories than the rows it meets, or that does not cover what it was asked to"""


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def traj_sha256(traj) -> str:
    """sha256 of a row's `traj` as little-endian float64, row-major.  Stdlib only; the same bytes for a JSON list-of-lists and for anything that iterates to the same numbers."""
    h = hashlib.sha256()
    for pose in traj:
        for v in pose:
            h.update(struct.pack("<d", float(v)))
    return h.hexdigest()


def row_triple(r: dict) -> str:
    return f"{r.get('log_name', '')}|{r.get('token', '')}|{int(r.get('rank', 0))}"


def row_key(r: dict) -> str:
    return f"{row_triple(r)}|{traj_sha256(r['traj'])}"


def _check_label(label, traj, where: str):
    """a label must be [T, 3] like the row's own traj, finite, and on the same yaw branch (|d yaw| < pi on every pose)."""
    if not isinstance(label, (list, tuple)) or len(label) != len(traj):
        raise TableRefused(f"{where}: label has {len(label) if isinstance(label, (list, tuple)) else '?'} poses, the row's traj has {len(traj)}")
    worst = 0.0
    for i, (lp, tp) in enumerate(zip(label, traj)):
        if len(lp) != 3 or len(tp) != 3:
            raise TableRefused(f"{where}: pose {i} is not (x, y, yaw): {len(lp)} vs {len(tp)} values")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in lp):
            raise TableRefused(f"{where}: pose {i} holds a non-finite or non-numeric value {lp!r}")
        worst = max(worst, abs(float(lp[YAW_COL]) - float(tp[YAW_COL])))
    if worst >= MAX_DYAW:
        raise TableRefused(f"{where}: |label yaw - raw yaw| reaches {worst:.4f} rad >= pi: a wrap-branch mismatch, not a repair")
    return worst


class Table:
    """The loaded table: `labels` {full key: label}, `by_triple` {(log|token|rank): {sha256, ...}}, `meta`, `sha256` (of the file)."""

    def __init__(self, labels: dict, meta: dict, sha: str):
        self.labels, self.meta, self.sha256 = labels, meta, sha
        self.by_triple: dict = {}
        for k in labels:
            triple, _, sha_ = k.rpartition("|")
            self.by_triple.setdefault(triple, set()).add(sha_)

    def __len__(self):
        return len(self.labels)

    def lookup(self, r: dict):
        """-> the label (list of [x, y, yaw]) for this bank row, or None (the row keeps its raw traj).  REFUSES a row whose (log, token, rank) is in the table under another traj sha256."""
        triple = row_triple(r)
        shas = self.by_triple.get(triple)
        if not shas:
            return None
        sha = traj_sha256(r["traj"])
        if sha not in shas:
            raise TableRefused(f"row {triple}: the table's label was computed from a traj with sha256 {sorted(shas)[0][:12]}..., this row's own traj is {sha[:12]}...: REFUSED, never substituted")
        return self.labels[f"{triple}|{sha}"]


# ------------------------------------------------------------------------------------------------ the file
def save(path, labels: dict, meta: dict) -> str:
    """Write the gzipped JSON deterministically and return the file's sha256."""
    for k, v in labels.items():
        if not isinstance(v, list) or not all(isinstance(p, list) and len(p) == 3 and all(isinstance(x, float) and math.isfinite(x) for x in p) for p in v):
            raise TableRefused(f"key {k}: the label is not a finite [T, 3] list of floats")
    meta = dict(meta, schema=SCHEMA, n_keys=len(labels))
    blob = json.dumps({"meta": meta, "table": labels}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            gz.write(blob)
    return sha256_file(path)


def load(path, expect_sha256: str | None = None) -> Table:
    sha = sha256_file(path)
    if expect_sha256 is not None and sha != expect_sha256:
        raise TableRefused(f"{path}: sha256 {sha[:12]} != the pinned {expect_sha256[:12]}")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    meta, labels = d.get("meta") or {}, d.get("table")
    if meta.get("schema") != SCHEMA or not isinstance(labels, dict):
        raise TableRefused(f"{path}: not a {SCHEMA} file (schema {meta.get('schema')!r})")
    if meta.get("n_keys") != len(labels):
        raise TableRefused(f"{path}: meta says {meta.get('n_keys')} keys, the table holds {len(labels)}")
    for k in labels:
        if k.count("|") < 3:
            raise TableRefused(f"{path}: malformed key {k!r} (want log|token|rank|sha256)")
    return Table(labels, meta, sha)


def apply_to_rows(table: Table, rows: list) -> dict:
    """The constructor's whole audit, over the bank's FINAL rows.  -> {"labels": {row index: label}, "coverage": {...}} .  Raises TableRefused on a traj-sha mismatch, on a bad label
    (shape / non-finite / yaw branch) and on a coverage below the table's own `require_coverage`.  Rows the table does not carry are simply absent from `labels` (they keep raw traj)."""
    out: dict = {}
    n_diff = 0
    for i, r in enumerate(rows):
        lab = table.lookup(r)
        if lab is None:
            continue
        _check_label(lab, r["traj"], f"row {row_triple(r)}")
        out[i] = lab
        if lab != [list(map(float, p)) for p in r["traj"]]:
            n_diff += 1
    n = len(rows)
    cov = {"rows": n, "covered": len(out), "kept_raw": n - len(out), "coverage": (len(out) / n) if n else 0.0, "differs_from_raw": n_diff,
           "table_keys": len(table), "table_sha256": table.sha256, "label": table.meta.get("label_name")}
    need = float(table.meta.get("require_coverage", 0.0) or 0.0)
    if cov["coverage"] < need:
        raise TableRefused(f"the table covers {len(out):,}/{n:,} rows ({100 * cov['coverage']:.3f} %), below the {100 * need:.3f} % it declares as required (meta.require_coverage); "
                           f"REFUSED rather than train a mixture the table did not promise")
    return {"labels": out, "coverage": cov}


# ------------------------------------------------------------------------------------------------ build / identity / verify
def read_bank_rows(bank_dir: str) -> list:
    rows = []
    for p in sorted(glob.glob(os.path.join(bank_dir, "targets_rank*.jsonl"))):       # the trainer's own glob (non-recursive)
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(ln) for ln in f if ln.strip()]
    if not rows:
        raise TableRefused(f"{bank_dir}: no targets_rank*.jsonl rows")
    return rows


def build_table(rows: list, labels_by_triple: dict, meta_extra: dict | None = None, require_coverage: float = 0.0, allow_partial: bool = False):
    """rows = bank rows; labels_by_triple = {(log|token|rank): (label, optional traj_sha256 or None)}.  -> (labels {key: label}, meta, report)."""
    out: dict = {}
    missing = []
    for r in rows:
        t = row_triple(r)
        got = labels_by_triple.get(t)
        if got is None:
            missing.append(t)
            continue
        lab, claimed = got
        sha = traj_sha256(r["traj"])
        if claimed is not None and claimed != sha:
            raise TableRefused(f"{t}: the labeller's traj_sha256 {claimed[:12]} is not this bank row's {sha[:12]}: the label was computed from another traj")
        lab = [[float(x) for x in p] for p in lab]
        _check_label(lab, r["traj"], f"row {t}")
        out[f"{t}|{sha}"] = lab
    if missing and not allow_partial:
        raise TableRefused(f"{len(missing):,} of {len(rows):,} bank rows have no label (e.g. {missing[:3]}); pass allow_partial / --allow-partial to build a partial table")
    meta = dict(meta_extra or {}, require_coverage=float(require_coverage), bank_rows=len(rows), covered=len(out), partial=bool(missing))
    return out, meta, {"rows": len(rows), "covered": len(out), "missing": len(missing), "missing_examples": missing[:5]}


def verify(table_path: str, bank_dir: str) -> dict:
    t = load(table_path)
    rows = read_bank_rows(bank_dir)
    rep = {"table": str(table_path), "table_sha256": t.sha256, "bank_dir": str(bank_dir), "refused": None}
    try:
        res = apply_to_rows(t, rows)
        rep.update(res["coverage"])
    except TableRefused as e:
        rep["refused"] = str(e)
    return rep


def _cmd_build(a):
    rows = read_bank_rows(a.bank_dir)
    labels: dict = {}
    with open(a.labels, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            d = json.loads(ln)
            labels[f"{d['log_name']}|{d['token']}|{int(d.get('rank', 0))}"] = (d["label"], d.get("traj_sha256"))
    tab, meta, rep = build_table(rows, labels, {"label_name": a.label_name, "labels_file": os.path.basename(a.labels)}, a.require_coverage, a.allow_partial)
    sha = save(a.out, tab, meta)
    print(json.dumps(dict(rep, out=str(a.out), table_sha256=sha), indent=1))
    return 0


def _cmd_identity(a):
    rows = read_bank_rows(a.bank_dir)
    labs = {row_triple(r): ([[float(x) for x in p] for p in r["traj"]], None) for r in rows}
    tab, meta, rep = build_table(rows, labs, {"label_name": "identity"}, 1.0)
    sha = save(a.out, tab, meta)
    print(json.dumps(dict(rep, out=str(a.out), table_sha256=sha), indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--bank-dir", required=True)
    b.add_argument("--labels", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--require-coverage", type=float, default=0.0)
    b.add_argument("--label-name", default="")
    b.add_argument("--allow-partial", action="store_true")
    i = sub.add_parser("identity")
    i.add_argument("--bank-dir", required=True)
    i.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--table", required=True)
    v.add_argument("--bank-dir", required=True)
    n = sub.add_parser("info")
    n.add_argument("--table", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "build":
            return _cmd_build(a)
        if a.cmd == "identity":
            return _cmd_identity(a)
        if a.cmd == "verify":
            rep = verify(a.table, a.bank_dir)
            print(json.dumps(rep, indent=1))
            return 1 if rep["refused"] else 0
        t = load(a.table)
        print(json.dumps(dict(t.meta, sha256=t.sha256, keys=len(t)), indent=1))
        return 0
    except TableRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
