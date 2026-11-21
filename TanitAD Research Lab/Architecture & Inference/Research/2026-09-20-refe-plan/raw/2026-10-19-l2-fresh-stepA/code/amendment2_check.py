#!/usr/bin/env python3
"""Checks for `PREREG_L2_FRESH_AMENDMENT2.md` (the next-arm mapping of O3 only).  Reads NO label and NO score.

`check_amendment` REFUSES (AmendmentRefused, exit status 5) unless
  (1) the file's sha256 (CRLF folded to LF) equals the given sha,
  (2) it carries "Status: SEALED" and not "Status: DRAFT",
  (3) it names the PARENT's sha256 and that equals the parent file on disk (the parent is the file the read's seal guard checks; it must be unchanged),
  (4) it carries the MM ruling VERBATIM (the ruling file's text appears in it),
  (5) it names the design calculation's code sha256 and result sha256 and both equal the files on disk (the result json as written, or LF-folded as staged),
  (6) its "Selected branch:" line equals `branch.selected` of the calculation's JSON, AND that branch is what the ruling's rule gives from the JSON's projected half-widths (the larger of the two seeds vs 0.025),
  (7) it states the projected half-widths of both seeds exactly as the JSON has them (4 decimals).
`sealed_before_read` is the temporal half: no label / score artifact of the run dir may be older than the seal.
This module does NOT replace the read's own guard (`analyse_fresh.py --seal-check-only`), which covers the parent and the five code pins; Amendment 2 touches none of them.

    python amendment2_check.py --amendment PREREG_L2_FRESH_AMENDMENT2.md --sha256 <hex> [--run-dir <W> --sealed-at <iso>]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys

W = "C:/Users/Admin/par18/l2fresh"
THRESHOLD = 0.025
LABEL_SCORE_FILES = ("bundle/bundle.npz", "perts.npz", "labels.npz", "scores.npz", "fresh_ep_scores.json", "fresh_ep_scores.npz", "feats/feats.npz", "FRESH_RESULT.json")


class AmendmentRefused(SystemExit):
    def __init__(self, msg: str):
        super().__init__(5)
        self.msg = msg

    def __str__(self):
        return self.msg


def sha_lf(p):
    return hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def sha_raw(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha_crlf(p):
    """the file with every line ending expanded to CRLF (the form the dev box wrote the result json in, recovered from an LF-folded staged copy)"""
    lf = open(p, "rb").read().replace(bytes([13, 10]), bytes([10]))
    return hashlib.sha256(lf.replace(bytes([10]), bytes([13, 10]))).hexdigest()


def branch_from_json(calc: dict) -> str:
    """the ruling's rule: B-ALL if the projected half-width <= 0.025 (the larger of the two seeds' projections), else the unseen-stratum data job"""
    hw = [calc["arms"][s]["projection"]["proportional_form"]["HW"] for s in ("H_s0", "H_s1")]
    return "B-ALL" if max(hw) <= THRESHOLD else "UNSEEN-STRATUM DATA JOB"


def check_amendment(amendment: str, sha_given: str, parent: str, ruling: str, calc_json: str, calc_code: str) -> dict:
    if not os.path.exists(amendment):
        raise AmendmentRefused(f"AMENDMENT REFUSED (5): file not found {amendment!r}")
    got = sha_lf(amendment)
    if got != str(sha_given).lower():
        raise AmendmentRefused(f"AMENDMENT REFUSED (5): sha256 {got[:12]} != given {str(sha_given)[:12]}")
    txt = open(amendment, encoding="utf-8").read()
    if "Status: DRAFT" in txt:
        raise AmendmentRefused("AMENDMENT REFUSED (5): draft marker present")
    if "Status: SEALED" not in txt:
        raise AmendmentRefused("AMENDMENT REFUSED (5): no SEALED marker")
    psha = sha_lf(parent)
    if psha not in txt:
        raise AmendmentRefused("AMENDMENT REFUSED (5): the amendment does not name the parent's CURRENT sha256 (the parent changed, or it was never named)")
    rtxt = open(ruling, "rb").read().replace(b"\r\n", b"\n").decode("utf-8").rstrip("\n")
    if rtxt not in txt:
        raise AmendmentRefused("AMENDMENT REFUSED (5): the MM ruling is not carried verbatim")
    # the sealed text pins the result json's RAW bytes (the file as the dev box wrote it: CRLF); the staged repo copy is LF-folded, so either form of the on-disk file is accepted (a changed number changes both)
    for label, p, fns in (("calculation code", calc_code, (sha_lf,)), ("calculation result", calc_json, (sha_raw, sha_lf, sha_crlf))):
        if not any(fn(p) in txt for fn in fns):
            raise AmendmentRefused(f"AMENDMENT REFUSED (5): the amendment does not name the CURRENT sha256 of the {label} ({os.path.basename(p)})")
    calc = json.load(open(calc_json, encoding="utf-8"))
    m = re.search(r"\*\*Selected branch: (.+?)\.\*\*", txt)
    if not m:
        raise AmendmentRefused("AMENDMENT REFUSED (5): no 'Selected branch:' line")
    want = branch_from_json(calc)
    if m.group(1) != calc["branch"]["selected"] or m.group(1) != want:
        raise AmendmentRefused(f"AMENDMENT REFUSED (5): branch {m.group(1)!r} vs result json {calc['branch']['selected']!r} vs the ruling's rule {want!r}")
    for s in ("H_s0", "H_s1"):
        hw = f"{calc['arms'][s]['projection']['proportional_form']['HW']:.4f}"
        if hw not in txt:
            raise AmendmentRefused(f"AMENDMENT REFUSED (5): the projected half-width {hw} of {s} is not stated as the result json has it")
    if not calc.get("identity", {}).get("pass"):
        raise AmendmentRefused("AMENDMENT REFUSED (5): the calculation's identity check did not pass")
    return {"amendment_sha256": got, "parent_sha256": psha, "branch": m.group(1)}


def parse_sealed_at(txt: str) -> datetime.datetime:
    m = re.search(r"SEALED_AT (\S+)", txt)
    if not m:
        raise ValueError("no SEALED_AT line")
    return datetime.datetime.fromisoformat(m.group(1))


def sealed_before_read(run_dir: str, sealed_at: datetime.datetime) -> dict:
    """every label / score artifact that exists in the run dir must be NEWER than the seal.  Vacuously true while none exists.  Only mtimes are read."""
    bad, seen = [], []
    for rel in LABEL_SCORE_FILES:
        p = os.path.join(run_dir, rel)
        if os.path.exists(p):
            seen.append(rel)
            mt = datetime.datetime.fromtimestamp(os.path.getmtime(p)).astimezone()
            if mt <= sealed_at:
                bad.append({"file": rel, "mtime": mt.isoformat(timespec="seconds")})
    return {"ok": not bad, "artifacts_present": seen, "older_than_seal": bad}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--amendment", default=f"{W}/PREREG_L2_FRESH_AMENDMENT2.md")
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--parent", default=f"{W}/PREREG_L2_FRESH.md")
    ap.add_argument("--ruling", default=f"{W}/MM_RULING_L2_FRESH_D.md")
    ap.add_argument("--calc-json", default=f"{W}/raw/design_calc_ball.json")
    ap.add_argument("--calc-code", default=f"{W}/code/design_calc_ball.py")
    ap.add_argument("--run-dir")
    ap.add_argument("--sealed-at-file", default=f"{W}/SEALED_AT_AMENDMENT2.txt")
    a = ap.parse_args(argv)
    info = check_amendment(a.amendment, a.sha256, a.parent, a.ruling, a.calc_json, a.calc_code)
    if a.run_dir:
        info["sealed_before_read"] = sealed_before_read(a.run_dir, parse_sealed_at(open(a.sealed_at_file, encoding="utf-8").read()))
    print("ZZAMENDMENT2_ACCEPTED", json.dumps(info), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AmendmentRefused as e:
        print(e.msg, file=sys.stderr, flush=True)
        sys.exit(5)
