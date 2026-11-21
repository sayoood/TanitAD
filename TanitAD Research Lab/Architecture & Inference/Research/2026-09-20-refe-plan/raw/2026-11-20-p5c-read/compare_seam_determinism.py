"""P5-C read: is the REFe navtest seam deterministic at inference?  Attempt-1 C1 seam vs the re-run C1 seam.

PRE-REGISTERED 2026-11-19 ~19:05Z by the MM, BEFORE the re-run's seam npz existed (attempt 4 was at log 105/136).
Same checkpoint (arms/C1/model_final.pt), same tip export (751f3ae7), same machine, same venv; the only intended
difference is the run itself (and one token that attempt 1 lost to a low-commit image-load miss).

Comparison is over tokens present in BOTH seams whose input fingerprint matches (a fingerprint mismatch means the
inputs differ, so that token is no determinism test; the count is reported).  Classes, committed in advance:
  IDENTICAL  poses, pick, proposals and logits bit-identical on every compared token
  DRIFT      not identical, but max |d pose| <= 1e-3 m AND pick agreement >= 99.9 %
  DIFFERENT  anything else  ->  the P5-C read's C1-C0 comparison carries an INFERENCE variance the episode bootstrap
             cannot see (CLAUDE.md, third variance); an effect below the measured spread is not quotable as a lever
             effect without an inference replicate.
Controls that must read known values (the script refuses otherwise):
  self       attempt 1 vs itself                       -> IDENTICAL, max |d pose| exactly 0.0
  perturbed  attempt 1 vs itself with one pose + 0.01 m -> DIFFERENT, max |d pose| ~= 0.01

Commit gate inside the runner: refuses below --min-commit-gb (default 4.5) of free commit (dev box, 2026-11-19).
Usage: python compare_seam_determinism.py <old.npz> <new.npz> --out <json>  (the .proposals.npz siblings are read too)
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np

MISSED_IN_ATTEMPT1 = "d88f286835b95ff3"


def commit_free_gb() -> float:
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise OSError("GlobalMemoryStatusEx failed")
    return m.ullAvailPageFile / 2 ** 30


def load(seam: Path) -> dict:
    dump = seam.with_name(seam.name.replace(".npz", ".proposals.npz"))
    out = {}
    with np.load(seam) as z:
        out.update(token=z["token"].astype(str), fp=z["fingerprint"].astype(str), poses=z["poses"].copy())
    with np.load(dump) as d:
        if not (d["token"].astype(str) == out["token"]).all():
            raise RuntimeError(f"{dump}: token order differs from {seam}")
        out.update(pick=d["pick"].copy(), proposals=d["proposals"].copy(), logits=d["logits"].copy())
    if len(set(out["token"])) != len(out["token"]):
        raise RuntimeError(f"{seam}: duplicate tokens")
    return out


def compare(a: dict, b: dict) -> dict:
    ia = {t: i for i, t in enumerate(a["token"])}
    ib = {t: i for i, t in enumerate(b["token"])}
    common = sorted(set(ia) & set(ib))
    fp_ok = [t for t in common if a["fp"][ia[t]] == b["fp"][ib[t]]]
    xa = np.array([ia[t] for t in fp_ok], dtype=np.int64)
    xb = np.array([ib[t] for t in fp_ok], dtype=np.int64)
    res = {"n_old": len(ia), "n_new": len(ib), "n_common": len(common), "n_fingerprint_mismatch": len(common) - len(fp_ok),
           "n_compared": len(fp_ok), "only_old": sorted(set(ia) - set(ib))[:20], "only_new": sorted(set(ib) - set(ia))[:20]}
    if not len(fp_ok):
        res["class"] = "NOT-COMPUTABLE"
        return res
    dpose = np.abs(a["poses"][xa].astype(np.float64) - b["poses"][xb].astype(np.float64))
    per_tok = dpose.reshape(len(fp_ok), -1).max(axis=1)
    pick_eq = a["pick"][xa] == b["pick"][xb]
    ident = {k: bool(np.array_equal(a[k][xa], b[k][xb])) for k in ("poses", "pick", "proposals", "logits")}
    res.update(identical=ident,
               max_abs_dpose_m=float(dpose.max()), median_tok_max_dpose_m=float(np.median(per_tok)),
               p99_tok_max_dpose_m=float(np.quantile(per_tok, 0.99)), n_tok_dpose_gt_1mm=int((per_tok > 1e-3).sum()),
               n_tok_dpose_gt_10cm=int((per_tok > 0.1).sum()), pick_agreement=float(pick_eq.mean()),
               n_pick_differ=int((~pick_eq).sum()),
               max_abs_dlogit=float(np.abs(a["logits"][xa].astype(np.float64) - b["logits"][xb].astype(np.float64)).max()))
    if all(ident.values()):
        res["class"] = "IDENTICAL"
    elif res["max_abs_dpose_m"] <= 1e-3 and res["pick_agreement"] >= 0.999:
        res["class"] = "DRIFT"
    else:
        res["class"] = "DIFFERENT"
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-commit-gb", type=float, default=4.5)
    a = ap.parse_args(argv)
    cf = commit_free_gb()
    if cf < a.min_commit_gb:
        print(f"ZZREFUSED commit free {cf:.2f} GB < {a.min_commit_gb} GB (dev-box commit gate)")
        return 3
    old = load(Path(a.old))
    ctl_self = compare(old, old)
    if ctl_self["class"] != "IDENTICAL" or ctl_self["max_abs_dpose_m"] != 0.0:
        raise RuntimeError(f"control 'self' must read IDENTICAL / 0.0, got {ctl_self['class']} {ctl_self['max_abs_dpose_m']}")
    pert = dict(old, poses=old["poses"].copy())
    pert["poses"][0, 3, 0] += np.float32(0.01)
    ctl_pert = compare(old, pert)
    if ctl_pert["class"] != "DIFFERENT" or abs(ctl_pert["max_abs_dpose_m"] - 0.01) > 1e-4:
        raise RuntimeError(f"control 'perturbed' must read DIFFERENT / ~0.01, got {ctl_pert['class']} {ctl_pert['max_abs_dpose_m']}")
    del pert
    new = load(Path(a.new))
    res = compare(old, new)
    res.update(old=a.old, new=a.new, controls={"self": "IDENTICAL 0.0", "perturbed": f"DIFFERENT {ctl_pert['max_abs_dpose_m']:.6f}"},
               missed_token=MISSED_IN_ATTEMPT1, missed_token_in_new=bool(MISSED_IN_ATTEMPT1 in set(new["token"])),
               commit_free_gb_at_start=round(cf, 2), preregistered="see module docstring (written before the new seam existed)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"ZZDET {res['class']} compared {res['n_compared']} max|dpose| {res.get('max_abs_dpose_m')} "
          f"pick agree {res.get('pick_agreement')} missed token present {res['missed_token_in_new']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
