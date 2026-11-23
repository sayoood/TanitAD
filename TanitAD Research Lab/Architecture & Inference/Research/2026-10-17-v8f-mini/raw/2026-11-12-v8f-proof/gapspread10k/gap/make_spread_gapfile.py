"""make_spread_gapfile.py -- card PREREG_V8F_PROOF_ADD134 (GAPSPREAD10K), control p0: the gap file VGAPSP90.npz.

On every row with an output: the base gap of VGAPLE.npz times max(0.60, 1 - k s), s the relative spread of the three fit files of card ADD106,
k the one number for which the mean factor over all rows with an output is 0.90 (no label enters the file).
The tools of cards ADD132 and ADD133 are imported unchanged. The label rows are read for the control alone (the long and the short part of row 201).

  build --base --fit (x3) --rec106 --rb-args --out-dir   -> VGAPSP90.npz and GAPSPREAD10K_GAPFILE_RECORD.json; refuses when the out folder exists
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import io
import json
import os
import sys
import types

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD134"
GB_PREFIX = "25a6db5a30692653"
KEYS = ("clip_index", "frame", "gap", "has_out", "clips", "seed")
ROW_TOL = 1e-9
EXPECT = {"k": 0.6033, "long": 0.492, "short": 3.105, "rows": 3841}
K_TOL = 1e-4
PART_TOL = 1e-3
RANGE = (0.0, 60.0)


def _sha(p):
    import hashlib
    h = hashlib.sha256()
    with io.open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_gb():
    m = sys.modules.get("read_gapspreadb_t0")
    p = getattr(m, "__file__", None) if m is not None else None
    if not (p and os.path.isfile(p)):
        for p in (os.environ.get("GAPSPREADB_TOOL"), "/workspace/v8proof/gapspread/kit/read_gapspreadb_t0.py", "F:/TanitAD-scratch/v8proof/gapspread/kit/read_gapspreadb_t0.py"):
            if p and os.path.isfile(p):
                break
        else:
            raise SystemExit("REFUSED: read_gapspreadb_t0.py not found (set GAPSPREADB_TOOL)")
        d = os.path.dirname(os.path.abspath(p))
        if d not in sys.path:
            sys.path.insert(0, d)
        spec = importlib.util.spec_from_file_location("read_gapspreadb_t0", p)
        m = importlib.util.module_from_spec(spec)
        sys.modules["read_gapspreadb_t0"] = m
        spec.loader.exec_module(m)
    if not _sha(p).startswith(GB_PREFIX):
        raise SystemExit("REFUSED: %s is not the tool of card ADD133 (%s)" % (p, GB_PREFIX))
    return m


GB = load_gb()
GS, MJ, RB, BS = GB.GS, GB.MJ, GB.RB, GB.BS


def build(x, a, expect=None, fault=None):
    expect = expect or EXPECT
    if os.path.exists(x.out_dir):
        raise SystemExit("REFUSED: %s exists" % x.out_dir)
    W = GS.World(x, a)
    B = W.files[GS.BASE_NAME]
    bad = []
    miss = [k for k in KEYS if k not in B]
    if miss:
        raise SystemExit("REFUSED: the base file lacks %s" % miss)
    s0 = GS.ctl_s0(W, BS.try_json(x.rec106))
    if not s0["pass"]:
        bad.append("files: " + s0["failures"][0])
    k = GS.solve_k(W.s[W.has])
    if k is None:
        raise SystemExit("REFUSED: no k reaches the dose")
    g = np.where(W.has, GS.sp_gap(W.m, W.s, k, fault), np.nan)
    os.makedirs(x.out_dir)
    S = {key: np.array(v, copy=True) for key, v in B.items()}
    S["gap"] = g.astype(np.float64)
    path = os.path.join(x.out_dir, "VGAPSP90.npz")
    np.savez(path, **S)
    F = MJ.load(path)
    if sorted(F) != sorted(B):
        bad.append("the new file does not have the arrays of the base file")
    for key in B:
        if key != "gap" and (key not in F or not np.array_equal(np.asarray(F[key]), np.asarray(B[key]))):
            bad.append("the new file differs from the base file in %s" % key)
    fg = np.asarray(F["gap"], np.float64)
    if not np.array_equal(np.isfinite(fg), W.has):
        bad.append("the new gap is not a number on other rows than those with no output")
    second = GS.sp_gap_rows(W.m, W.s, k)
    dev = float(np.abs(fg[W.has] - second[W.has]).max())
    if not dev <= ROW_TOL:
        bad.append("the new gap differs from the second routine by %.3g m" % dev)
    dose = float(GS.factor_of(W.s[W.has], k).mean())
    if abs(dose - GS.DOSE) > GS.DOSE_TOL:
        bad.append("the mean factor is not the dose")
    if abs(k - expect["k"]) > K_TOL:
        bad.append("k is not the k of row 201")
    lp = float(GS.long_part(fg[W.rows], W.lab).mean()) if len(W.rows) else float("nan")
    sp = float(GS.short_part(fg[W.rows], W.lab).mean()) if len(W.rows) else float("nan")
    if len(W.rows) != expect["rows"] or not abs(lp - expect["long"]) <= PART_TOL or not abs(sp - expect["short"]) <= PART_TOL:
        bad.append("the long and the short part on the label rows do not reproduce row 201")
    inb = np.isfinite(W.m) & (W.m >= RANGE[0]) & (W.m <= RANGE[1])
    inn = np.isfinite(fg) & (fg >= RANGE[0]) & (fg <= RANGE[1])
    rec = {"card": CARD, "pass_p0": not bad, "failures": bad,
           "files": {"VGAPSP90": {"file": "VGAPSP90.npz", "sha256": _sha(path)}},
           "inputs": {"base": W.sha[GS.BASE_NAME], "fits": {n: W.sha[n] for n in GS.FIT_NAMES}, "rec106": _sha(x.rec106) if x.rec106 and os.path.isfile(x.rec106) else None,
                      "tool": _sha(os.path.abspath(__file__)), "read_gapspreadb_t0.py": _sha(GB.__file__), "read_gapspread_t0.py": _sha(GS.__file__)},
           "k": float(k), "dose": GS.DOSE, "floor": GS.FLOOR, "max_abs_dev_second_routine_m": dev,
           "counts": {"n_rows": int(len(W.m)), "n_rows_with_output": int(W.has.sum()), "n_label_rows": int(len(W.rows)),
                      "n_rows_in_the_rule_range_in_the_base_alone": int((W.has & inb & ~inn).sum()), "n_rows_in_the_rule_range_in_the_new_file_alone": int((W.has & ~inb & inn).sum())},
           "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    BS.wt(os.path.join(x.out_dir, "GAPSPREAD10K_GAPFILE_RECORD.json"), json.dumps(rec, indent=1, default=BS._jdefault))
    return rec


def print_record(rec):
    print("ctl p0 PASS" if rec["pass_p0"] else "ctl p0 FAIL %s" % rec["failures"][0][:150])
    print("counts %s" % json.dumps(rec["counts"], sort_keys=True))
    print("file VGAPSP90.npz sha256 begins %s" % rec["files"]["VGAPSP90"]["sha256"][:16])
    print("ZZ SPREAD-GAPFILE %s" % ("ALL-PASS" if rec["pass_p0"] else "FAILED"))
    return rec["pass_p0"]


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("mode", choices=("build",))
    ap.add_argument("--base", default="")
    ap.add_argument("--fit", action="append", default=[])
    ap.add_argument("--rec106", default="")
    ap.add_argument("--rb-args", default="")
    ap.add_argument("--out-dir", default="")
    v = ap.parse_args(argv)
    if not v.base or len(v.fit) != 3 or not v.rb_args or not v.out_dir:
        print("REFUSED: --base, three --fit, --rb-args and --out-dir are required")
        return 2
    x = types.SimpleNamespace(base=v.base, fit=v.fit, rec106=v.rec106, rb_args=v.rb_args, out_dir=v.out_dir, tests_out=[])
    return 0 if print_record(build(x, MJ.rb_args(v.rb_args, v.out_dir + "_scratch"))) else 1


if __name__ == "__main__":
    sys.exit(main())
