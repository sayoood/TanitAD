#!/usr/bin/env python3
"""run_v1_fix1 -- three INSTRUMENT fixes for W3's navtest driver, without touching ``run_v1.py``'s bytes.

WHY A WRAPPER AND NOT AN EDIT. ``run_v1.py``'s sha256 (``125670597d99b243...``) is on record as the
validated harness in three places: the Thor admission record
``2026-10-04-navsim-thor-backend/raw/THOR_BACKEND_VALIDATED.json`` / ``..._VALIDATION.json``
(``byte_identical_code_on_thor``, ``ADMITTED_UNDER_POLICY``); the Colab code-bundle MANIFEST
``colab/raw/2026-09-27-navtest-p0/code_manifest.json`` (written by ``colab/navtest/pack_code.py``); and
``w3_run_v1_sha256`` in 24 banked ``*.thor.json`` score-provenance files. No launch-gate token binds it and
nothing compares it at run time, but an in-place edit would leave every one of those records naming bytes
that are no longer at that path and would desynchronise the Thor copy from the repo. So the original stays
byte-for-byte what was validated, ``load`` verifies that (sha256 pin), and the fixes are applied to the
IMPORTED module object. Everything that decides a score is still W3's, untouched.

THE THREE FIXES (all MEASURED failures of RL-1 V1 navtest, 2026-10-18; ``Project Steering/TRAPS.md``,
"A W3 NAVTEST RUN KILLED BY ITS RAM GUARD ..."):

  A  cross-drive ``os.path.relpath(dst, PKG)`` (run_v1.py:459) raised ``ValueError`` on Windows when the
     run's ``--out`` and the package sit on different drives -- AFTER the devkit finished, BEFORE the
     C2/C3 row-count checks ran, so the report was lost. Now: fall back to the absolute posix path and
     record ``csv_path_kind: "abs"`` (``"rel"`` otherwise).
  B  a reconstructed PARTIAL csv left in the devkit run dir was re-copied by the harness's glob fallback
     (``sorted(glob(run_out/*.csv))[-1]``) on later attempts. Now: at run start every ``*.csv`` already in
     ``<EXP>/runs/<label>/`` -- and a previous ``<RAW>/<label>/<label>.csv`` copy -- is MOVED into a
     timestamped ``quarantine_<stamp>/`` beside it, with sha256 / bytes / token-row count of each, and the
     list is written into ``<label>.counts.json`` (``stale_csv_quarantine``). Nothing is deleted.
  C  C2 already FAILs a csv whose token-row count differs from the expected count, but the report still
     carried the partial run's ``summary`` / ``summary_x100_4dp`` and PRINTED its average (two partial
     averages, 88.44 and 88.47, were read before the full read and had to be disclosed). Now: when the csv
     rows / valid rows / average rows / token set disagree with ``n_expected``, both summary blocks are
     removed from counts.json and from the printed line, ``summary_suppressed`` says why, status is FAIL.

USE (a drop-in for the three lines that load run_v1.py by path)::

    spec = importlib.util.spec_from_file_location("w3_fix1", os.path.join(W3_CODE, "run_v1_fix1.py"))
    fx = importlib.util.module_from_spec(spec); spec.loader.exec_module(fx)
    w3 = fx.load()            # the patched W3 module; set w3.RAW / w3.PKG / w3.EXP exactly as before
    w3.cmd_score(ns)

or on the command line, with run_v1.py's own sub-commands::   python run_v1_fix1.py score --label ...

A drive-repointed COPY of run_v1.py (e.g. score_e.py's ``w3e/code``, ``D:/`` -> ``E:/`` by sed) is a different byte
sequence: ``load`` refuses it unless ``allow_drift=True`` or ``RUN_V1_FIX1_ALLOW_DRIFT=1``; counts.json then
records ``run_v1_matches_pinned: false`` beside the sha256 that actually ran.

Python 3.9-compatible (the NAVSIM venv). Tests: ``tests/test_run_v1_fix1.py`` (a RED mutation arm per fix).
"""
from __future__ import annotations

import contextlib
import csv
import glob
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL = os.path.join(HERE, "run_v1.py")
ORIGINAL_SHA256 = "125670597d99b243eaf68bf77973753b314e41f3ed060d0073ad627da601914e"
FIX_VERSION = 1
FIXES = ("A-xdrive-relpath", "B-stale-csv-quarantine", "C-no-summary-on-count-mismatch")


class Fix1DriftError(RuntimeError):
    """run_v1.py is not the validated byte sequence these fixes were written against."""


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def verify_original(path: str = ORIGINAL, allow_drift: bool = False) -> dict:
    """sha256 of run_v1.py against the validated pin. A CRLF checkout (core.autocrlf) is accepted when its
    LF-normalised bytes match -- that is the SAME program; any other difference raises."""
    raw = open(path, "rb").read()
    sha, sha_lf = _sha256_bytes(raw), _sha256_bytes(raw.replace(b"\r\n", b"\n"))
    ok = ORIGINAL_SHA256 in (sha, sha_lf)
    if not ok and not (allow_drift or os.environ.get("RUN_V1_FIX1_ALLOW_DRIFT") == "1"):
        raise Fix1DriftError(f"{path}: sha256 {sha} != pinned {ORIGINAL_SHA256}; run_v1_fix1 was written "
                             f"against the validated bytes. Re-validate before changing the pin.")
    return {"path": path, "sha256": sha, "sha256_lf_normalised": sha_lf, "matches_pinned": ok}


# --------------------------------------------------------------------------- #
# FIX A -- cross-drive relpath. The ONLY os.path function run_v1 overrides is relpath; the shim stands in
# for the module's ``os`` name and forwards everything else to the live ``os``.
class _PathShim:
    def __getattr__(self, name):
        return getattr(os.path, name)

    def relpath(self, path, start=os.curdir):
        try:
            return os.path.relpath(path, start)
        except ValueError:                         # "path is on mount 'C:', start on mount 'E:'"
            return os.path.abspath(path).replace("\\", "/")


class _OsShim:
    path = _PathShim()

    def __getattr__(self, name):
        return getattr(os, name)


def _is_abs_posix(p: str) -> bool:
    return p.startswith("/") or re.match(r"^[A-Za-z]:/", p) is not None


# --------------------------------------------------------------------------- #
# FIX B -- stale-csv quarantine.
def _csv_facts(p: str) -> dict:
    facts = {"bytes": os.path.getsize(p), "sha256": _sha256_file(p)}
    try:
        with open(p, encoding="utf-8", newline="") as fh:
            facts["token_rows"] = sum(1 for r in csv.DictReader(fh) if r.get("token") != "average")
    except Exception as e:                                           # noqa: BLE001
        facts["token_rows"] = None
        facts["unreadable"] = repr(e)[:120]
    return facts


def quarantine_files(files, parent: str, stamp: str) -> list:
    """MOVE ``files`` into ``<parent>/quarantine_<stamp>[_n]/`` (created only if there is something to
    move); return one record per file. Nothing is ever deleted or overwritten."""
    files = sorted({f for f in files if os.path.isfile(f)})
    if not files:
        return []
    qdir, n = os.path.join(parent, "quarantine_" + stamp), 1
    while os.path.exists(qdir):
        n += 1
        qdir = os.path.join(parent, f"quarantine_{stamp}_{n}")
    os.makedirs(qdir)
    moved = []
    for f in files:
        facts = _csv_facts(f)
        dst = os.path.join(qdir, os.path.basename(f))
        shutil.move(f, dst)
        moved.append(dict(facts, **{"from": f.replace("\\", "/"), "to": dst.replace("\\", "/")}))
    return moved


def quarantine_stale_csvs(run_out: str, out_dir: str, label: str, stamp: str) -> list:
    stale_run = glob.glob(os.path.join(glob.escape(run_out), "*.csv"))
    moved = quarantine_files(stale_run, run_out, stamp)
    moved += quarantine_files([os.path.join(out_dir, f"{label}.csv")], out_dir, stamp)
    return moved


# --------------------------------------------------------------------------- #
# FIX C -- a count-mismatched csv emits no summary.
def count_mismatch_reason(rep: dict):
    """None when the csv the harness copied agrees with the expected token set; else why not."""
    if "csv" not in rep:
        return None                                     # no csv, so no summary was ever computed
    want = rep.get("n_expected")
    for key, exp in (("csv_token_rows", want), ("csv_valid_rows", want), ("csv_average_rows", 1)):
        if rep.get(key) != exp:
            return f"{key}={rep.get(key)} != {exp}"
    if rep.get("C3_missing") or rep.get("C3_unexpected"):
        return f"token set differs: missing {rep.get('C3_missing')}, unexpected {rep.get('C3_unexpected')}"
    return None


PRINT_KEYS = ("label", "status", "log_successful", "log_failed", "csv_valid_rows", "C1_max_abs_delta",
              "summary_x100_4dp", "wall_s", "failures")


def _wrap_cmd_score(m):
    orig_cmd_score = m.cmd_score

    def cmd_score(a) -> int:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        run_out = f"{m.EXP}/runs/{a.label}"
        out_dir = os.path.join(m.RAW, a.label)
        quarantined = quarantine_stale_csvs(run_out, out_dir, a.label, stamp)
        if quarantined:                      # also on stdout, so the driver's log has it even if the run dies
            print(json.dumps({"fix1_quarantine": len(quarantined), "moved": quarantined}), flush=True)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = orig_cmd_score(a)
        except BaseException:
            sys.stdout.write(buf.getvalue())
            raise
        counts = os.path.join(out_dir, f"{a.label}.counts.json")
        if not os.path.exists(counts):
            sys.stdout.write(buf.getvalue())
            return rc
        rep = json.load(open(counts, encoding="utf-8"))
        rep["harness_fix1"] = {"version": FIX_VERSION, "fixes": list(FIXES),
                               "run_v1_sha256": m.FIX1["original"]["sha256"],
                               "run_v1_matches_pinned": m.FIX1["original"]["matches_pinned"],
                               "run_v1_fix1_sha256": _sha256_file(__file__)}
        rep["stale_csv_quarantine"] = {"n_moved": len(quarantined), "moved": quarantined}
        if "csv" in rep:
            rep["csv_path_kind"] = "abs" if _is_abs_posix(str(rep["csv"])) else "rel"
        reason = count_mismatch_reason(rep)
        if reason:
            for k in ("summary", "summary_x100_4dp"):
                rep.pop(k, None)
            rep["summary_suppressed"] = {"reason": reason, "csv_token_rows": rep.get("csv_token_rows"),
                                         "n_expected": rep.get("n_expected")}
            if rep.get("status") != "FAIL":
                rep["status"] = "FAIL"
                rep.setdefault("failures", []).append("FIX1-C: csv disagrees with the expected token set: "
                                                      + reason)
            rc = rc or 1
        json.dump(rep, open(counts, "w", encoding="utf-8"), indent=1, default=str)
        if reason:
            line = {k: rep.get(k) for k in PRINT_KEYS if k != "summary_x100_4dp"}
            line["summary_suppressed"] = rep["summary_suppressed"]
            print(json.dumps(line, default=str))
        else:
            sys.stdout.write(buf.getvalue())
        return rc

    return cmd_score


def load(orig_path: str = ORIGINAL, name: str = "run_v1_fix1_w3", allow_drift: bool = False):
    """Import the byte-verified ``run_v1.py`` and apply fixes A, B and C to the module object."""
    info = verify_original(orig_path, allow_drift=allow_drift)
    spec = importlib.util.spec_from_file_location(name, orig_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    m.FIX1 = {"version": FIX_VERSION, "fixes": FIXES, "original": info}
    m.os = _OsShim()
    m.cmd_score = _wrap_cmd_score(m)
    return m


def main(argv=None) -> int:
    return load().main(argv)


if __name__ == "__main__":
    sys.exit(main())
