#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mapft_kplanid.py -- PREREG_MAPFT s2 controls K-PLAN-ID and K-RAST-MOVED (dev box, CPU), each with controls that must read a KNOWN value.

  K-PLAN-ID (sealed text): the MAPSEL roll of MAPFT on eval139 (os-only, per-frame feed, inference seed 0, the registered window list) emits,
  on EVERY window, a plan, fan and `sel` BIT-EQUAL to the base trace F:/refcv9_mapsel/trace_s0.npz (and its MAPPATH derivative).
  Failure = the planner reads the fine-tuned modules: STOP, report, no bar is read.
  K-RAST-MOVED: the MAPFT own rasters differ from the base rasters on some window (else the run trained nothing).

SUBCOMMANDS (all print a JSON record and an opaque ZZMAPFT-...ZZ marker; the verdict is the RECORD, not an exit code)
  compare  --base <trace_s0.npz> --new <trace.npz> --json <out>            K-PLAN-ID leg 1 (+ the in-run controls C1 self-compare, C2 one-bit mutation)
  mappath  --base-derivative <trace_s0_mappath.npz> --new <trace.npz> --work <dir> [--script mappath_transform.py] [--base <trace_s0.npz>]
                                                                         K-PLAN-ID leg 2: derive the new trace's MAPPATH twin with the REGISTERED script and compare it with the banked one;
                                                                         with --base also runs the CONTROL derive(base) == banked derivative
  rast     --base-dir <rast_s0> --new-dir <rast_x> --json <out>             K-RAST-MOVED, with the CONTROL that the GT plane is bit-equal on every window (alignment)
  self-test                                                               literal expectations on synthetic data + one mutant per rule

BIT-EQUAL means raw BYTES (dtype, shape and tobytes()): a -0.0 differs from +0.0 and a NaN equals the same NaN -- the sealed text says bit-equal.
PRIMARY arrays (the K-PLAN-ID verdict) are EXACTLY the sealed three: plan8, fan, sel -- no stricter, no looser.  Every OTHER numeric array (sel_base included)
is reported BESIDE (never silently dropped): a difference there does not change the sealed verdict, but it is printed and goes to the Master Mind.
Window alignment (e_i, t, ws and n_windows) is a PRECONDITION: a mismatch there is INVALID, not a K-PLAN-ID failure.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PRIMARY = ("plan8", "fan", "sel")            # EXACTLY the sealed text: "a plan, fan and `sel`"
ALIGN = ("e_i", "t", "ws")
SKIP = ("meta_json",)
OWN_PLANES = ("pdrv", "pedge", "psw", "code")
GT_PLANE = "gt"


def _bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def _rowdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """bool [n]: per-window (axis 0) byte inequality; assumes equal shape/dtype."""
    n = a.shape[0]
    av = np.ascontiguousarray(a).reshape(n, -1).view(np.uint8) if a.size else np.zeros((n, 0), np.uint8)
    bv = np.ascontiguousarray(b).reshape(n, -1).view(np.uint8) if b.size else np.zeros((n, 0), np.uint8)
    return (av != bv).any(axis=1)


def load(path) -> dict:
    z = np.load(str(path), allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["_meta"] = json.loads(str(d["meta_json"])) if "meta_json" in d else {}
    return d


def compare_arrays(base: dict, new: dict, primary=PRIMARY) -> dict:
    """The K-PLAN-ID comparison on two loaded traces -> record (no I/O)."""
    rec: dict = {"alignment": {}, "arrays": {}, "primary": list(primary)}
    keys_b = sorted(k for k in base if not k.startswith("_") and k not in SKIP)
    keys_n = sorted(k for k in new if not k.startswith("_") and k not in SKIP)
    rec["keys_equal"] = keys_b == keys_n
    if not rec["keys_equal"]:
        rec["keys_only_base"], rec["keys_only_new"] = sorted(set(keys_b) - set(keys_n)), sorted(set(keys_n) - set(keys_b))
    nb = int(base["_meta"].get("n_windows", -1)) if base["_meta"] else -1
    nn = int(new["_meta"].get("n_windows", -1)) if new["_meta"] else -1
    rec["n_windows"] = {"base": nb, "new": nn}
    rec["final"] = {"base": base["_meta"].get("final"), "new": new["_meta"].get("final")}
    aligned = rec["keys_equal"] and nb == nn and nb > 0 and base["_meta"].get("final") is True and new["_meta"].get("final") is True
    for k in ALIGN:
        ok = k in base and k in new and _bytes_equal(base[k], new[k])
        rec["alignment"][k] = ok
        aligned = aligned and ok
    rec["aligned"] = bool(aligned)
    if not aligned:
        rec["K_PLAN_ID"] = "INVALID"
        rec["why"] = "window alignment / key set / finality precondition failed: the two traces are not the same window list (this is NOT a planner-identity failure)"
        return rec
    for k in keys_b:
        a, b = base[k], new[k]
        row: dict = {"equal": _bytes_equal(a, b), "dtype": [str(a.dtype), str(b.dtype)], "shape": [list(a.shape), list(b.shape)]}
        if not row["equal"] and a.shape == b.shape and a.dtype == b.dtype and a.ndim >= 1:
            dw = _rowdiff(a, b)
            row["n_windows_differing"] = int(dw.sum())
            row["first_differing_windows"] = [int(i) for i in np.flatnonzero(dw)[:5]]
            if np.issubdtype(a.dtype, np.floating):
                d = np.abs(a.astype(np.float64) - b.astype(np.float64))
                row["max_abs_diff"] = float(np.nanmax(d)) if d.size else 0.0
        rec["arrays"][k] = row
    miss = [k for k in primary if k not in rec["arrays"]]
    rec["primary_missing"] = miss
    rec["primary_equal"] = {k: rec["arrays"][k]["equal"] for k in primary if k in rec["arrays"]}
    rec["beside_differing"] = sorted(k for k, r in rec["arrays"].items() if k not in primary and not r["equal"])
    rec["K_PLAN_ID"] = "PASS" if (not miss and all(rec["primary_equal"].values())) else "FAIL"
    return rec


def controls(base: dict) -> dict:
    """C1: the comparator reads base == base (all arrays equal).  C2: ONE flipped bit in plan8 of window 0 -> plan8 differs on exactly 1 window, K_PLAN_ID FAIL."""
    c1 = compare_arrays(base, base)
    c1_ok = c1["K_PLAN_ID"] == "PASS" and all(r["equal"] for r in c1["arrays"].values())
    mut = dict(base)
    p = np.array(base["plan8"], copy=True)
    flat = p.reshape(p.shape[0], -1).view(np.uint32) if p.dtype == np.float32 else None
    if flat is None:
        return {"C1_self_compare_all_equal": c1_ok, "C2_one_bit_mutation_detected": False, "error": "plan8 is not float32"}
    flat[0, 0] ^= np.uint32(1)                       # one mantissa bit of one value
    mut["plan8"] = p
    c2 = compare_arrays(base, mut)
    c2_ok = (c2["K_PLAN_ID"] == "FAIL" and c2["arrays"]["plan8"].get("n_windows_differing") == 1 and c2["arrays"]["plan8"]["first_differing_windows"] == [0]
             and c2["arrays"]["fan"]["equal"] and c2["arrays"]["sel"]["equal"])
    return {"C1_self_compare_all_equal": bool(c1_ok), "C2_one_bit_mutation_detected": bool(c2_ok)}


def cmd_compare(a) -> int:
    base, new = load(a.base), load(a.new)
    rec = compare_arrays(base, new)
    rec["controls"] = controls(base)
    ctl = all(rec["controls"].values())
    rec["controls_ok"] = bool(ctl)
    rec["base"], rec["new"] = str(a.base), str(a.new)
    if not ctl:
        rec["K_PLAN_ID_FINAL"] = "INVALID(controls)"
    else:
        rec["K_PLAN_ID_FINAL"] = rec["K_PLAN_ID"]
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("K_PLAN_ID_FINAL", "controls", "primary_equal", "beside_differing", "n_windows") if k in rec}, indent=1))
    print(f"ZZMAPFT-KPLANID-{rec['K_PLAN_ID_FINAL'].split('(')[0]}ZZ")
    return 0 if rec["K_PLAN_ID_FINAL"] == "PASS" else 1


def _derive(script: str, src: Path, dst: Path, rec: Path) -> dict:
    r = subprocess.run([sys.executable, script, str(src), str(dst), str(rec)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if not dst.is_file() or not rec.is_file():
        return {"error": f"derivation produced no output (rc={r.returncode}): {(r.stderr or r.stdout)[-300:]}"}
    return {"record": json.loads(rec.read_text(encoding="utf-8"))}


def cmd_mappath(a) -> int:
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    out: dict = {"script": a.script, "base_derivative": str(a.base_derivative), "new": str(a.new)}
    ok = True
    if a.base:       # CONTROL: the registered derivation applied to the BASE trace must reproduce the banked derivative bit for bit
        d0 = _derive(a.script, Path(a.base), work / "derived_from_base.npz", work / "derived_from_base.rec.json")
        if "error" in d0:
            out["control_derive_base"] = d0
            ok = False
        else:
            c = compare_arrays(load(a.base_derivative), load(work / "derived_from_base.npz"))
            out["control_derive_base"] = {"all_arrays_equal": all(r["equal"] for r in c.get("arrays", {}).values()) and bool(c.get("arrays")),
                                          "K_PLAN_ID": c["K_PLAN_ID"], "beside_differing": c.get("beside_differing")}
            ok = ok and out["control_derive_base"]["all_arrays_equal"]
    d1 = _derive(a.script, Path(a.new), work / "derived_from_new.npz", work / "derived_from_new.rec.json")
    if "error" in d1:
        out["new_derive"] = d1
        ok = False
    else:
        c = compare_arrays(load(a.base_derivative), load(work / "derived_from_new.npz"))
        out["new_vs_banked_derivative"] = {"K_PLAN_ID": c["K_PLAN_ID"], "primary_equal": c.get("primary_equal"), "beside_differing": c.get("beside_differing"),
                                           "K-ID_native_unchanged_bit_equal": d1["record"].get("K-ID_native_unchanged_bit_equal")}
        ok = ok and c["K_PLAN_ID"] == "PASS"
    out["MAPPATH_LEG"] = "PASS" if ok else "FAIL"
    (work / "mappath_leg.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print("ZZMAPFT-KPLANID-MAPPATH-PASSZZ" if ok else "ZZMAPFT-KPLANID-MAPPATH-FAILZZ")
    return 0 if ok else 1


def cmd_rast(a) -> int:
    bd, nd = Path(a.base_dir), Path(a.new_dir)
    fb, fn = sorted(p.name for p in bd.glob("e*_w*.npz")), sorted(p.name for p in nd.glob("e*_w*.npz"))
    rec: dict = {"n_base": len(fb), "n_new": len(fn), "same_file_set": fb == fn}
    if not fb or fb != fn:
        rec["K_RAST_MOVED"] = "INVALID"
        rec["why"] = "empty or different raster file sets: the two rolls did not cover the same windows"
        Path(a.json).write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(json.dumps(rec, indent=1))
        print("ZZMAPFT-KRASTMOVED-INVALIDZZ")
        return 1
    n_changed = {p: 0 for p in OWN_PLANES}
    n_windows_moved = 0
    gt_mismatch = 0
    gt_present = 0
    frac_sum = {p: 0.0 for p in OWN_PLANES}
    first = []
    for i, name in enumerate(fb):
        zb, zn = np.load(bd / name), np.load(nd / name)
        moved = False
        for p in OWN_PLANES:
            if p in zb.files and p in zn.files:
                ch = zb[p] != zn[p]
                if ch.any():
                    n_changed[p] += 1
                    frac_sum[p] += float(ch.mean())
                    moved = True
        if GT_PLANE in zb.files or GT_PLANE in zn.files:
            gt_present += 1
            if not (GT_PLANE in zb.files and GT_PLANE in zn.files and np.array_equal(zb[GT_PLANE], zn[GT_PLANE])):
                gt_mismatch += 1
        if moved:
            n_windows_moved += 1
            if len(first) < 5:
                first.append(name)
    rec.update(windows_with_any_own_plane_changed=n_windows_moved, per_plane_windows_changed=n_changed,
               mean_changed_pixel_share_over_changed_windows={p: (frac_sum[p] / n_changed[p] if n_changed[p] else 0.0) for p in OWN_PLANES},
               first_moved=first, gt_plane={"windows_with_gt": gt_present, "mismatching": gt_mismatch})
    control_ok = gt_mismatch == 0 and gt_present > 0                 # alignment control: the GT plane is the SAME physical plane in both rolls
    rec["control_gt_plane_bit_equal_on_every_window"] = bool(control_ok)
    rec["K_RAST_MOVED"] = "INVALID(control)" if not control_ok else ("PASS" if n_windows_moved >= 1 else "FAIL")
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    print(f"ZZMAPFT-KRASTMOVED-{rec['K_RAST_MOVED'].split('(')[0]}ZZ")
    return 0 if rec["K_RAST_MOVED"] == "PASS" else 1


# ------------------------------------------------------------------------------------------------------------------------------------
def _synth(n=6, seed=0):
    r = np.random.default_rng(seed)
    d = {"e_i": np.arange(n, dtype=np.int64) // 3, "t": np.arange(n, dtype=np.int64), "ws": np.arange(n, dtype=np.int64) * 5,
         "plan8": r.normal(size=(n, 8, 2)).astype(np.float32), "sel": r.integers(0, 149, n).astype(np.int64), "sel_base": r.integers(0, 149, n).astype(np.int64),
         "fan": r.normal(size=(n, 149, 8, 2)).astype(np.float32), "sel_score": r.normal(size=(n, 149)).astype(np.float32),
         "bank": r.normal(size=(n, 117, 8, 2)).astype(np.float16)}
    d["_meta"] = {"n_windows": n, "final": True}
    return d


def _copy(d):
    return {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else dict(v)) for k, v in d.items()}


def self_test(_a) -> int:
    bad = []

    def expect(name, got, want):
        ok = got == want
        print(f"{'OK ' if ok else 'BAD'} {name}: expected {want!r}, read {got!r}")
        if not ok:
            bad.append(name)

    b = _synth()
    expect("identical traces -> PASS", compare_arrays(b, _copy(b))["K_PLAN_ID"], "PASS")
    m = _copy(b); m["plan8"].reshape(6, -1).view(np.uint32)[2, 3] ^= np.uint32(1)
    r = compare_arrays(b, m)
    expect("MUT one bit of plan8 -> FAIL", r["K_PLAN_ID"], "FAIL")
    expect("   ... and it names window 2", r["arrays"]["plan8"]["first_differing_windows"], [2])
    m = _copy(b); m["sel"][4] = (m["sel"][4] + 1) % 149
    expect("MUT sel changed -> FAIL", compare_arrays(b, m)["K_PLAN_ID"], "FAIL")
    m = _copy(b); m["fan"][5, 100, 7, 1] += np.float32(1e-7)
    expect("MUT fan changed by 1e-7 -> FAIL", compare_arrays(b, m)["K_PLAN_ID"], "FAIL")
    m = _copy(b); m["sel_score"][0, 0] += np.float32(1.0)
    r = compare_arrays(b, m)
    expect("sel_score differs, primary equal -> PASS with it listed BESIDE", (r["K_PLAN_ID"], r["beside_differing"]), ("PASS", ["sel_score"]))
    m = _copy(b); m["sel_base"][1] = (m["sel_base"][1] + 1) % 149
    r = compare_arrays(b, m)
    expect("sel_base is NOT in the sealed three: differs -> PASS, listed BESIDE", (r["K_PLAN_ID"], r["beside_differing"]), ("PASS", ["sel_base"]))
    expect("the primary set is exactly the sealed three", PRIMARY, ("plan8", "fan", "sel"))
    m = _copy(b); m["plan8"][0, 0, 0] = np.float32(-0.0) if b["plan8"][0, 0, 0] == 0 else b["plan8"][0, 0, 0]
    z = _copy(b); z["plan8"][0, 0, 0] = np.float32(0.0); y = _copy(z); y["plan8"][0, 0, 0] = np.float32(-0.0)
    expect("+0.0 vs -0.0 are NOT bit-equal -> FAIL", compare_arrays(z, y)["K_PLAN_ID"], "FAIL")
    n1 = _copy(b); n1["plan8"][1, 1, 1] = np.float32("nan"); n2 = _copy(n1)
    expect("same NaN in both -> PASS (bytes, not ==)", compare_arrays(n1, n2)["K_PLAN_ID"], "PASS")
    m = _copy(b); m["ws"][3] += 1
    expect("MUT window list differs -> INVALID (not FAIL)", compare_arrays(b, m)["K_PLAN_ID"], "INVALID")
    m = _copy(b); m["_meta"]["final"] = False
    expect("MUT partial trace -> INVALID", compare_arrays(b, m)["K_PLAN_ID"], "INVALID")
    m = _copy(b); del m["bank"]
    expect("MUT missing array -> INVALID", compare_arrays(b, m)["K_PLAN_ID"], "INVALID")
    expect("controls on synthetic base", controls(b), {"C1_self_compare_all_equal": True, "C2_one_bit_mutation_detected": True})
    print("ZZMAPFT-KPLANID-SELFTEST-OKZZ" if not bad else f"ZZMAPFT-KPLANID-SELFTEST-FAILZZ {bad}")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare"); c.add_argument("--base", required=True); c.add_argument("--new", required=True); c.add_argument("--json", required=True)
    m = sub.add_parser("mappath"); m.add_argument("--base-derivative", required=True); m.add_argument("--new", required=True); m.add_argument("--work", required=True)
    m.add_argument("--script", default="C:/Users/Admin/refcv8_pf/refcv9_mapedge/mappath_transform.py"); m.add_argument("--base", default=None)
    r = sub.add_parser("rast"); r.add_argument("--base-dir", required=True); r.add_argument("--new-dir", required=True); r.add_argument("--json", required=True)
    sub.add_parser("self-test")
    a = ap.parse_args(argv)
    return {"compare": cmd_compare, "mappath": cmd_mappath, "rast": cmd_rast, "self-test": self_test}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
