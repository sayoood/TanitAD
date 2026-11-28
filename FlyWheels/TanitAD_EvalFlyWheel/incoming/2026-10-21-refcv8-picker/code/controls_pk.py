"""controls_pk.py -- PREREG R8-PICKER section 6 controls on a capture (K1, K3, K4, K5, K6a/b/c) -- shared by the CPU smoke and the analysis.

  K1   wrapped vs UNWRAPPED roll of the same device / seed / windows: every array of every npz the unwrapped roll wrote is bit-identical
       (equal_nan) in the wrapped roll's dump.
  K3   K3a: anchor_traj[sel_idx] == traj exactly; K3b: traj == the roll's own sidecar nav_true.traj exactly.
  K4   argmax(refc_v3.e9_rank(sel_score_v3, {reach_keep, ceil_keep, r8_emit_keep})) == sel_idx on every window (the model's own function, CPU);
       a disagreement where both indices carry the same rank value is a TIE (counted, not failed).
  K5   the CORE rank re-built in the forward's order (sel_score; reach_keep; ceil_keep; emit with the core's dead-row rule) -> argmax == sel_idx_base.
  K6a  v_max_valid == 0 windows: ceil_keep all True (Addendum 1: vacuous on eval139, n reported).
  K6b  (Addendum 1) windows where EVERY candidate's refcv6_selection.planned_max_speed (horizons 5..60 ticks, tick 0.1 s) <= the fed ceiling
       (clamp_posthoc_r8.fed_ceiling): ceil_keep all True.  Beside (informative, re-derives the producer's rule): agreement of ceil_keep with
       (pm <= ceiling) | empty-row.
  K6c  the SHIP pick lies in e9_keep on every window.

    python controls_pk.py smoke --root C:/Users/Admin/refcv8_pf/picker/smoke --out-json <K1_SMOKE.json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PKGW = "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code"
HZ8 = [5, 10, 15, 20, 30, 40, 50, 60]


def k1_compare(unwrapped_dump: str, wrapped_dump: str) -> dict:
    import build_picker_trees as BT
    c = BT.compare_trees(unwrapped_dump, wrapped_dump)
    return {"n_files": c["n_files"], "n_arrays": c["n_arrays"], "missing_in_wrapped": c["missing_in_b"], "differing": c["differing"],
            "PASS": bool(c["n_files"] > 0 and not c["missing_in_b"] and not c["differing"])}


def _float_env(da: str, db: str) -> tuple:
    """(max |delta| over every float array of every npz in da that also exists in db, number of integer / bool arrays that differ)."""
    mx, nint = 0.0, 0
    for f in sorted(glob.glob(os.path.join(da, "**", "*.npz"), recursive=True)):
        g = os.path.join(db, os.path.relpath(f, da))
        if not os.path.exists(g):
            continue
        A, B = np.load(f, allow_pickle=True), np.load(g, allow_pickle=True)
        for k in A.files:
            if k not in B.files:
                continue
            x, y = A[k], B[k]
            if x.dtype.kind == "f" and x.shape == y.shape:
                if not np.array_equal(x, y, equal_nan=True):
                    mx = max(mx, float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))))
            elif x.dtype.kind in "iub" and not np.array_equal(x, y):
                nint += 1
    return mx, nint


def _torch_e9(z):
    import torch
    from tanitad.refs import refc_v3 as RV3
    out = {"reach_keep": torch.from_numpy(np.asarray(z["reach_keep"], bool)),
           "ceil_keep": torch.from_numpy(np.asarray(z["ceil_keep"], bool)),
           "r8_emit_keep": torch.from_numpy(np.asarray(z["r8_emit_keep"], bool))}
    blended = torch.from_numpy(np.asarray(z["sel_score_v3"], np.float32))
    rank, _ = RV3.e9_rank(blended, out)
    rank0, _ = RV3.e9_rank(torch.zeros_like(blended), out)
    return rank.numpy(), np.isfinite(rank0.numpy()), RV3.__file__


def core_rank(z) -> np.ndarray:
    s = np.asarray(z["sel_score"], np.float32)
    ninf = np.float32(-np.inf)
    r = np.where(np.asarray(z["reach_keep"], bool), s, ninf)
    r = np.where(np.asarray(z["ceil_keep"], bool), r, ninf)
    ek = np.asarray(z["r8_emit_keep"], bool)
    r2 = np.where(ek, r, ninf)
    dead = ~np.isfinite(r2).any(1)
    return np.where(dead[:, None], np.where(ek, s, ninf), r2)


def _agree_with_ties(rank: np.ndarray, idx: np.ndarray) -> dict:
    am = rank.argmax(1)
    ar = np.arange(len(idx))
    dis = am != idx
    tie = dis & (rank[ar, am] == rank[ar, idx])
    return {"n": int(len(idx)), "n_disagree": int(dis.sum()), "n_ties": int(tie.sum()), "agree_frac": float((~dis).mean()),
            "PASS": bool((dis & ~tie).sum() == 0)}


def fed_ceiling_eval(vmax: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if PKGW not in sys.path:
        sys.path.insert(0, PKGW)
    import clamp_posthoc_r8 as CP
    rc = CP.run_cfg()
    lim, info = CP.fed_ceiling(np.asarray(vmax, np.float32), np.asarray(valid, np.float32), rc)
    return lim.numpy().astype(np.float64), {"run_cfg_horizons": rc["horizons"], **{k: info[k] for k in ("n_rows", "n_known", "n_unknown_inf", "limit_values_ms")}}


def capture_controls(z, side_traj: np.ndarray, vmax: np.ndarray, valid: np.ndarray) -> dict:
    """z = capture npz; side_traj = the roll's own sidecar nav_true.traj (same row order, asserted by the wrapper); vmax / valid = the windows' fed label."""
    import torch
    from tanitad.refs.refcv6_selection import planned_max_speed
    fan = np.asarray(z["anchor_traj"], np.float32)
    n = fan.shape[0]
    ar = np.arange(n)
    sel = np.asarray(z["sel_idx"]).astype(np.int64).reshape(-1)
    base = np.asarray(z["sel_idx_base"]).astype(np.int64).reshape(-1)
    traj = np.asarray(z["traj"], np.float32)
    out = {"n_windows": int(n)}
    out["K3"] = {"K3a_max_abs_m": float(np.abs(fan[ar, sel] - traj).max()), "K3b_traj_equals_sidecar": bool(np.array_equal(traj, np.asarray(side_traj, np.float32)))}
    out["K3"]["PASS"] = bool(out["K3"]["K3a_max_abs_m"] == 0.0 and out["K3"]["K3b_traj_equals_sidecar"])
    rank, e9_keep_re, rv3 = _torch_e9(z)
    out["K4"] = _agree_with_ties(rank, sel)
    out["K4"]["refc_v3"] = rv3
    out["K4"]["e9_keep_recomputed_equals_captured"] = bool(np.array_equal(e9_keep_re, np.asarray(z["e9_keep"], bool)))
    out["K4"]["rolltime_e9_argmax_agree"] = (float((np.asarray(z["e9_argmax"]).reshape(-1) == sel).mean()) if "e9_argmax" in z.files else None)
    out["K4"]["PASS"] = bool(out["K4"]["PASS"] and out["K4"]["e9_keep_recomputed_equals_captured"])
    out["K5"] = _agree_with_ties(core_rank(z), base)
    if "traj_base" in z.files:
        out["K5"]["fan_at_base_minus_traj_base_max_abs_m"] = float(np.abs(fan[ar, base] - np.asarray(z["traj_base"], np.float32)).max())
    ck = np.asarray(z["ceil_keep"], bool)
    v0m = np.asarray(valid, np.float64) < 0.5
    out["K6a"] = {"n_windows_v_max_valid_0": int(v0m.sum()), "all_true_there": bool(ck[v0m].all()) if v0m.any() else None,
                  "note": "Addendum 1: vacuous on eval139 when n = 0 (not counted as evidence)"}
    lim, linfo = fed_ceiling_eval(vmax, valid)
    pm = planned_max_speed(torch.from_numpy(fan.astype(np.float32)), horizons=HZ8, tick_s=0.1).numpy().astype(np.float64)
    nb = (pm <= lim[:, None]).all(1)
    rule = (pm <= lim[:, None]) | (~(pm <= lim[:, None]).any(1))[:, None]
    out["K6b"] = {"n_nonbinding_windows": int(nb.sum()), "ceil_keep_all_true_on_nonbinding_frac": (float(ck[nb].all(1).mean()) if nb.any() else None),
                  "PASS": (bool(ck[nb].all()) if nb.any() else None), "fed_ceiling": linfo,
                  "beside_ceil_keep_equals_rule_cell_frac": float((ck == rule).mean()),
                  "beside_ceil_keep_equals_rule_window_frac": float((ck == rule).all(1).mean()),
                  "beside_n_binding_windows": int((~nb).sum())}
    ek = np.asarray(z["e9_keep"], bool)
    out["K6c"] = {"ship_pick_in_e9_keep_frac": float(ek[ar, sel].mean()), "PASS": bool(ek[ar, sel].all())}
    out["e9_keep"] = {"mean_count": float(ek.sum(1).mean()), "min_count": int(ek.sum(1).min()), "n_windows_lt_32": int((ek.sum(1) < 32).sum())}
    out["PASS_blocking"] = bool(out["K3"]["PASS"] and out["K4"]["PASS"] and out["K5"]["PASS"] and out["K6c"]["PASS"] and out["K6b"]["PASS"] is not False)
    return out


def cmd_smoke(a) -> int:
    t0 = time.time()
    res = {"tool": "controls_pk.py smoke", "generated": time.strftime("%FT%T%z"), "root": a.root, "seeds": {}}
    ok = True
    for sd in (0, 1):
        w, u = os.path.join(a.root, "wrapped", f"s{sd}"), os.path.join(a.root, "unwrapped", f"s{sd}")
        r = {"form": "FULL ARM" if sd == 0 else "--os-only"}
        # ADDENDUM 2 (a4982749...): K1 PASSES iff the wrapped roll is bit-identical to AT LEAST ONE unwrapped roll of the same form, the integer
        # arrays agree across ALL of them, and no wrapped-vs-unwrapped float difference exceeds the unwrapped-vs-unwrapped envelope.
        unw = [d for d in sorted(glob.glob(os.path.join(a.root, "unwrapped*", f"s{sd}"))) if os.path.exists(os.path.join(d, "dump", "manifest.json"))]
        r["K1_vs"] = {}
        for d in unw:
            c = k1_compare(os.path.join(d, "dump"), os.path.join(w, "dump"))
            c["max_abs_float_diff"], c["n_int_arrays_differing"] = _float_env(os.path.join(d, "dump"), os.path.join(w, "dump"))
            r["K1_vs"][os.path.relpath(d, a.root)] = c
        env = {}
        for i in range(len(unw)):
            for j in range(i + 1, len(unw)):
                env[f"{os.path.relpath(unw[i], a.root)}~{os.path.relpath(unw[j], a.root)}"] = _float_env(os.path.join(unw[i], "dump"), os.path.join(unw[j], "dump"))
        env_max = max([v[0] for v in env.values()], default=0.0)
        any_bit = any(c["PASS"] for c in r["K1_vs"].values())
        ints_ok = all(c["n_int_arrays_differing"] == 0 for c in r["K1_vs"].values()) and all(v[1] == 0 for v in env.values())
        within = all(c["max_abs_float_diff"] <= env_max for c in r["K1_vs"].values())
        first = r["K1_vs"].get(os.path.join("unwrapped", f"s{sd}"), {})
        r["K1"] = {"as_first_coded_vs_unwrapped": {"PASS": first.get("PASS"), "n_differing": len(first.get("differing", [])),
                                                    "max_abs_float_diff": first.get("max_abs_float_diff")},
                   "unwrapped_vs_unwrapped_envelope": {"pairs": env, "max_abs": env_max},
                   "bit_identical_to_at_least_one_unwrapped": any_bit, "integer_arrays_identical_everywhere": ints_ok,
                   "within_envelope": within, "rule": "ADDENDUM 2 (a4982749...)", "PASS": bool(any_bit and ints_ok and within),
                   "n_files": first.get("n_files"), "n_arrays": first.get("n_arrays"), "differing": first.get("differing", [])}
        z = np.load(os.path.join(w, "dump", "fan_extras.npz"))
        side = np.load(os.path.join(w, "dump", "refcv6_extras.npz"))
        r["capture"] = capture_controls(z, side["nav_true.traj"], side["v_max_ms"], side["v_max_valid"])
        r["meta"] = json.load(open(os.path.join(w, f"fan_roll_s{sd}.json"), encoding="utf-8"))
        r["meta"] = {k: r["meta"].get(k) for k in ("n_windows", "conds_seen", "n_calls_without_nav_true", "keys_missing_counts", "K3a_max_abs_fan_at_sel_minus_traj_m",
                                                  "K3b_traj_equals_sidecar_nav_true", "K4_rolltime_e9_argmax_equals_sel_idx_frac", "wall_s", "this_script_sha256",
                                                  "refc_v3_sha256", "torch", "host", "child_argv")}
        r["picks_preview"] = {"sel_idx": z["sel_idx"].tolist(), "sel_idx_base": z["sel_idx_base"].tolist(), "e9_keep_count": z["e9_keep"].sum(1).tolist(),
                              "ceil_keep_count": z["ceil_keep"].sum(1).tolist(), "v_max_ms": side["v_max_ms"].round(3).tolist()}
        r["PASS"] = bool(r["K1"]["PASS"] and r["capture"]["PASS_blocking"])
        ok = ok and r["PASS"]
        res["seeds"][str(sd)] = r
    res["ALL_PASS"] = ok
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out_json, "w", encoding="utf-8"), indent=1, default=str)
    for sd, r in res["seeds"].items():
        c = r["capture"]
        print(f"s{sd} K1(Add.2) {r['K1']['PASS']} [first-coded {r['K1']['as_first_coded_vs_unwrapped']}; envelope {r['K1']['unwrapped_vs_unwrapped_envelope']['max_abs']}] K3 {c['K3']['PASS']} "
              f"K4 {c['K4']['PASS']} ({c['K4']['n_disagree']} dis) K5 {c['K5']['PASS']} ({c['K5']['n_disagree']} dis) K6a n={c['K6a']['n_windows_v_max_valid_0']} "
              f"K6b {c['K6b']['PASS']} (n_nonbinding {c['K6b']['n_nonbinding_windows']}) K6c {c['K6c']['PASS']}")
    print("ZZPKSMOKECTL_" + ("PASS" if ok else "FAIL") + "ZZ", flush=True)
    return 0 if ok else 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("smoke",))
    ap.add_argument("--root", default="C:/Users/Admin/refcv8_pf/picker/smoke")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    return cmd_smoke(a)


if __name__ == "__main__":
    sys.exit(main())
