#!/usr/bin/env python3
"""feas2_t1_read.py -- the read of H-FEAS2-T1-1 (PREREG_FROZEN.md): X = S + {kbox, abox, kabox}, Y = KAPPA-T1's fresh S runs (reused), DAG-OACC and SOUP, s0 / s1.

KAPPA-T1's reader (kappa_t1_read.py under READER_FIX_FLOOR.md) with the lever generalised to three modes and Y read from KAPPA-T1's out dir.  Per-window arrays: SMOOTH-T1's
extract_smooth.py (read-only; own work dir and logs; --floor-guard-m 1.0, the gross guard; the same-session check is KF against the fresh Y).  Rule and rows: SMOOTH-T1's pure
layer (read_a, read_xy, arm_outcome) on gapdecomp.decompose (paired episode-cluster bootstrap, B=2000, seed 0; answers another draw of EPISODES only).  Known answers KA1 (a fresh
rerun of ONE arm-seed equals the reused Y, bit for bit), KA2 (plan == mode(P + SG(plan_raw - P), tick v0)), KF, K3, KB1 and the Y pins (FROZEN_SHA.txt) are computed before any row.
Tier T1-kin; evidence class MEASURED (ours).
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
SMT = "E:/Projects/TanitAD-artifacts/dagland/smooth_t1"
KAP_PK = "E:/Projects/TanitAD-artifacts/dagland/kappa_t1"
OUT_Y = "F:/TanitAD-scratch/kappa_t1/out"                 # KAPPA-T1's fresh S runs (SREP_*), reused as Y
OUT_X = "F:/TanitAD-scratch/feas2_t1/out"
WORK = "F:/TanitAD-scratch/feas2_t1/work/rig"
TREE_STACK = "F:/TanitAD-scratch/feas2_t1/tree_dag/stack"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
MODES = ("kbox", "abox", "kabox")
PFX = {"kbox": "KBOX", "abox": "ABOX", "kabox": "KABOX"}
# key -> (tag suffix, arm, seed)
ARMSEEDS = {"CTRL_s0": ("DAGOACC_s0", "CTRL", 0), "CTRL_s1": ("DAGOACC_s1", "CTRL", 1), "SOUP_s0": ("SOUP_s0", "SOUP", 0), "SOUP_s1": ("SOUP_s1", "SOUP", 1)}
KA1_TAG, KA1_REF = "SREP2_DAGOACC_s0", "SREP_DAGOACC_s0"
TOL = 1e-4
FL_GROSS_M = 1.0   # READER_FIX_FLOOR.md section 3.1


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SR = _load("smooth_t1_read", f"{SMT}/code/smooth_t1_read.py")          # read-only import: pure layer, load_dump, cmp_arrays, sha256_file
GK = _load("gate_kappa", f"{HERE}/gate_kappa.py")


def extract(tag, arm, seed, out_dir, arr_dir):
    dst = f"{arr_dir}/{tag}.npz"
    if os.path.exists(dst):
        return dst
    os.makedirs(arr_dir, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    cmd = [PY, "-I", f"{SMT}/code/extract_smooth.py", "--arm", arm, "--seed", str(seed), "--out", dst, "--dump-dir", f"{out_dir}/{tag}/dump_{tag}",
           "--ticks", f"{out_dir}/{tag}/ticks/ticks_{tag}.npz", "--work-dir", WORK, "--floor-guard-m", str(FL_GROSS_M)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    open(f"{PK}/logs/extract_{tag}.log", "w", encoding="utf-8", newline="\n").write(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(f"extract {tag} failed rc={r.returncode}: {r.stderr[-400:]}")
    return dst


def lever(plan_s, v0, mode):
    """The card's operators, spelled here (feasible_decode from the arm tree), dt 0.2, mu=None; float32 cast as the tool returns."""
    import torch
    if TREE_STACK not in sys.path:
        sys.path.insert(0, TREE_STACK)
    from tanitad.refs import feasible_decode as FD
    inf = float("inf")
    p = np.asarray(plan_s, np.float64)
    full = torch.from_numpy(np.concatenate([np.zeros_like(p[:, :1]), p], 1))
    out = full
    if mode in ("kbox", "kabox"):
        out = FD.project_feasible(full, dt=0.2, a_max=inf, kappa_max=FD.KAPPA_MAX_1PM, mu=None, clamp_entry=False)
    if mode in ("abox", "kabox"):
        out = FD.project_feasible(out, v0=torch.from_numpy(np.asarray(v0, np.float64)), dt=0.2, a_max=FD.A_MAX_MPS2, kappa_max=inf, mu=None, clamp_entry=True)
    return out[:, 1:].numpy().astype(np.float32).astype(np.float64)


def y_pins():
    """The Y runs reused from KAPPA-T1 are the ones the card froze: their done_* files and extracted arrays hash to FROZEN_SHA.txt's pins."""
    pins = {}
    for ln in open(f"{PK}/FROZEN_SHA.txt", encoding="utf-8"):
        p = ln.split()
        if len(p) >= 5 and p[0] == "Y" and p[1] == "arrays":
            pins[f"arrays:{p[2]}"] = p[4]
        elif len(p) >= 4 and p[0] == "Y" and p[1].startswith("done_"):
            pins[f"done:{p[1]}"] = p[3]
    out, ok = {}, True
    for t in ("DAGOACC_s0", "DAGOACC_s1", "SOUP_s0", "SOUP_s1"):
        for key, path in ((f"done:done_SREP_{t}.txt", f"{OUT_Y}/done_SREP_{t}.txt"), (f"arrays:SREP_{t}.npz", f"{KAP_PK}/raw/arrays/SREP_{t}.npz")):
            got, want = SR.sha256_file(path), pins.get(key)
            out[key] = {"sha256": got, "pinned": want, "PASS": bool(want is not None and got == want)}
            ok &= out[key]["PASS"]
    return out, ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_X)
    ap.add_argument("--raw", default=f"{PK}/raw")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    want = [ln.split()[2] for ln in open(f"{PK}/FROZEN_SHA.txt", encoding="utf-8") if ln.startswith("PREREG_FROZEN.md sha256")][0]
    got = SR.sha256_file(f"{PK}/PREREG_FROZEN.md")
    if want != got:
        print(f"REFUSED: PREREG_FROZEN.md {got} != frozen {want}", file=sys.stderr)
        return 2
    from scipy.signal import savgol_filter
    arr_dir = f"{a.raw}/arrays"
    res = {"hypothesis": "H-FEAS2-T1-1", "levers": "kbox / abox / kabox after S", "tier": "T1-kin", "evidence_class": "MEASURED (ours; artifacts in MANIFEST.txt)", "prereg_sha256": got,
           "estimator": f"paired episode-cluster bootstrap, B={a.n_boot}, seed 0 (taniteval.ci draw stream via gapdecomp.decompose); answers another draw of EPISODES only",
           "run_clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "controls": {}, "arms": {}, "descriptive": {}}
    GD = SR.load_gd()
    MAN = json.load(open(SR.MANIFEST, encoding="utf-8"))
    pins, pins_ok = y_pins()
    res["controls"]["KY_reused_Y_equals_frozen_pins"] = {"files": pins, "PASS": bool(pins_ok)}
    # ---- KA1: ONE fresh rerun of an arm-seed equals the reused Y, bit for bit
    _, d1 = SR.load_dump(f"{OUT_X}/{KA1_TAG}/dump_{KA1_TAG}")
    _, d0 = SR.load_dump(f"{OUT_Y}/{KA1_REF}/dump_{KA1_REF}")
    same_d, dev_d, _, bad_d, only_d = SR.cmp_arrays(d1, d0)
    t1 = {q: v for q, v in np.load(f"{OUT_X}/{KA1_TAG}/ticks/ticks_{KA1_TAG}.npz", allow_pickle=True).items() if q not in ("tag", "ckpt")}
    t0 = {q: v for q, v in np.load(f"{OUT_Y}/{KA1_REF}/ticks/ticks_{KA1_REF}.npz", allow_pickle=True).items() if q not in ("tag", "ckpt")}
    same_t, dev_t, _, bad_t, only_t = SR.cmp_arrays(t1, t0)
    res["controls"]["KA1_fresh_rerun_equals_reused_Y"] = {"rerun": KA1_TAG, "reused": KA1_REF, "dump_bit_identical": same_d, "dump_max_abs_dev_m": dev_d, "dump_differing_keys": bad_d,
                                                          "ticks_bit_identical": same_t, "ticks_max_abs_dev": dev_t, "ticks_differing_keys": bad_t, "key_set_diff": only_d + only_t,
                                                          "PASS": bool((same_d and same_t) or (dev_d <= TOL and dev_t <= TOL and not only_d and not only_t))}
    ka2, kf = {}, {}
    for k, (suf, arm, seed) in ARMSEEDS.items():
        yt = f"SREP_{suf}"
        _, dy = SR.load_dump(f"{OUT_Y}/{yt}/dump_{yt}")
        ty = {q: v for q, v in np.load(f"{OUT_Y}/{yt}/ticks/ticks_{yt}.npz", allow_pickle=True).items() if q not in ("tag", "ckpt")}
        db_ref = SR.load_dump(MAN["arms"]["CTRL"][str(seed)]["t1"]["dump_dir"])[1]
        wk = ("clip_index", "ws", "eid", "g", "v0")
        for mode in MODES:
            xt = f"{PFX[mode]}_{suf}"
            tx = dict(np.load(f"{a.out_dir}/{xt}/ticks/ticks_{xt}.npz", allow_pickle=True))
            _, dx = SR.load_dump(f"{a.out_dir}/{xt}/dump_{xt}")
            s_rec = (tx["P"] + savgol_filter(tx["plan_raw"] - tx["P"], 7, 2, axis=1, mode="interp")).astype(np.float32).astype(np.float64)
            d_s = float(np.abs(s_rec - tx["plan_s"]).max())
            d_l = float(np.abs(lever(tx["plan_s"], tx["v0"], mode) - tx["plan"]).max())
            k0 = tx["tick"] == 0
            d_t0 = float(np.abs(tx["plan_raw"][k0] - ty["plan_raw"][ty["tick"] == 0]).max())
            kk = f"{k}:{mode}"
            ka2[kk] = {"plan_s_vs_P_plus_SG_max_abs_m": d_s, "plan_vs_lever_of_plan_s_max_abs_m": d_l, "lever_changed_plan_max_abs_m": float(np.abs(tx["plan"] - tx["plan_s"]).max()),
                       "tick0_plan_raw_vs_Y_max_abs_m": d_t0, "floor_equals_Y": bool(np.array_equal(dx["fl"], dy["fl"])), "floor_max_abs_dev_m": float(np.abs(dx["fl"] - dy["fl"]).max()),
                       "PASS": bool(d_s <= TOL and d_l <= TOL and d_t0 <= TOL and float(np.abs(dx["fl"] - dy["fl"]).max()) <= TOL)}
            kf[kk] = {"windows_X_equal_fresh_Y": bool(all(np.array_equal(dx[q], dy[q]) for q in wk)), "windows_fresh_Y_equal_banked_CTRL": bool(all(np.array_equal(dy[q], db_ref[q]) for q in wk)),
                      "floor_X_vs_fresh_Y_max_abs_m": float(np.abs(dx["fl"] - dy["fl"]).max()), "floor_fresh_Y_vs_banked_CTRL_max_abs_m": float(np.abs(dy["fl"] - db_ref["fl"]).max()),
                      "floor_X_vs_banked_CTRL_max_abs_m": float(np.abs(dx["fl"] - db_ref["fl"]).max()), "floor_fresh_Y_vs_banked_CTRL_mean_abs_m": float(np.abs(dy["fl"] - db_ref["fl"]).mean())}
            kf[kk]["PASS"] = bool(kf[kk]["windows_X_equal_fresh_Y"] and kf[kk]["windows_fresh_Y_equal_banked_CTRL"] and kf[kk]["floor_X_vs_fresh_Y_max_abs_m"] <= TOL
                                  and kf[kk]["floor_fresh_Y_vs_banked_CTRL_max_abs_m"] <= FL_GROSS_M and kf[kk]["floor_X_vs_banked_CTRL_max_abs_m"] <= FL_GROSS_M)
    res["controls"]["KA2_issued_plan_is_lever_of_S"] = {"runs": ka2, "PASS": all(v["PASS"] for v in ka2.values())}
    res["controls"]["KF_floor_window_check_vs_fresh_Y"] = {"runs": kf, "gross_guard_m": FL_GROSS_M, "PASS": all(v["PASS"] for v in kf.values())}
    if SR.sha256_file(SR.SOUP_READ) != SR.PIN_SOUP_READ:
        print("REFUSED: SOUP_READ.json is not the pinned banked read", file=sys.stderr)
        return 2
    bk_tbl = {"CTRL": json.load(open(SR.SOUP_READ, encoding="utf-8"))["tables"]["CTRL_P10"], "SOUP": json.load(open(SR.SOUP_READ, encoding="utf-8"))["tables"]["SOUP_P10"]}
    k3 = {}
    for arm_, s_ in (("CTRL", 0), ("CTRL", 1), ("SOUP", 0), ("SOUP", 1)):
        z = np.load(f"{SR.GD_DIR}/raw/arrays/{arm_}_s{s_}.npz", allow_pickle=True)
        cls_, _ = GD.classify(z["v0"], z["g"])
        for i, lab in enumerate(GD.LABELS):
            r = GD.decompose(z[f"a_{i}"], z[f"b_{i}"], z["eid"], cls_, GD.POOL[lab], a.n_boot)
            bk = bk_tbl[arm_][f"s{s_}"][lab]
            k3[f"{arm_}_s{s_}:{lab}"] = bool(abs(r["delta"] - bk["delta"]) <= SR.TOL_ROW and abs(r["lo"] - bk["lo"]) <= SR.TOL_ROW and abs(r["hi"] - bk["hi"]) <= SR.TOL_ROW and r["separated"] == bk["separated"])
    res["controls"]["K3_Y_banked_minus_P10_reproduces_SOUP_READ"] = {"n_rows": len(k3), "n_pass": sum(k3.values()), "failed": [q for q, v in k3.items() if not v], "PASS": all(k3.values())}

    def refuse(msg):
        json.dump(res, open(f"{a.raw}/FEAS2_T1.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
        print("REFUSED:", msg, {q: v["PASS"] for q, v in res["controls"].items()}, file=sys.stderr)
        return 3

    if not all(v["PASS"] for v in res["controls"].values()):
        return refuse("a known answer failed")
    kb1 = {}
    for k, (suf, arm, seed) in ARMSEEDS.items():
        zl = np.load(f"{SR.GD_DIR}/raw/arrays/{arm}_s{seed}.npz", allow_pickle=True)
        tgs = [("Y", f"SREP_{suf}", OUT_Y, f"{KAP_PK}/raw/arrays")] + [(f"X:{m}", f"{PFX[m]}_{suf}", a.out_dir, arr_dir) for m in MODES]
        for role, tg, od, ad in tgs:
            zz = np.load(extract(tg, arm, seed, od, ad), allow_pickle=True)
            lab_ok = tuple(str(q) for q in zz["labels"]) == tuple(str(q) for q in zl["labels"])
            eq = {str(l): bool(np.array_equal(zz[f"b_{i}"], zl[f"b_{i}"], equal_nan=True)) for i, l in enumerate(zl["labels"]) if not str(l).startswith("S-POS")}
            win = bool(all(np.array_equal(zz[q], zl[q]) for q in ("eid", "v0", "g", "ws", "clip_index")))
            kb1[f"{k}:{role}"] = {"labels_equal": lab_ok, "windows_equal_landed": win, "n_rows_b_equal": sum(eq.values()), "n_rows": len(eq), "failed": [q for q, v in eq.items() if not v], "PASS": bool(lab_ok and win and all(eq.values()))}
    res["controls"]["KB1_P10_side_equals_landed_banked_b"] = {"runs": kb1, "PASS": all(v["PASS"] for v in kb1.values())}
    if not all(v["PASS"] for v in res["controls"].values()):
        return refuse("KB1 failed")
    per_arm = {m: {"CTRL": {}, "SOUP": {}} for m in MODES}
    for k, (suf, arm, seed) in ARMSEEDS.items():
        zy = np.load(extract(f"SREP_{suf}", arm, seed, OUT_Y, f"{KAP_PK}/raw/arrays"), allow_pickle=True)
        _, dy = SR.load_dump(f"{OUT_Y}/SREP_{suf}/dump_SREP_{suf}")
        py = GK.partition(dy["cl"], dy["fl"], dy["g"])
        for mode in MODES:
            xt = f"{PFX[mode]}_{suf}"
            zx = np.load(extract(xt, arm, seed, a.out_dir, arr_dir), allow_pickle=True)
            labels = tuple(str(q) for q in zx["labels"])
            assert labels == GD.LABELS and np.array_equal(zx["eid"], zy["eid"])
            cls, _ = GD.classify(zx["v0"], zx["g"])
            rows = {"X_minus_P10": {}, "X_minus_Y": {}, "Y_minus_P10": {}}
            for i, lab in enumerate(labels):
                for nm, (u, v) in (("X_minus_P10", (zx[f"a_{i}"], zx[f"b_{i}"])), ("X_minus_Y", (zx[f"a_{i}"], zy[f"a_{i}"])), ("Y_minus_P10", (zy[f"a_{i}"], zy[f"b_{i}"]))):
                    r = GD.decompose(u, v, zx["eid"], cls, GD.POOL[lab], a.n_boot)
                    rows[nm][lab] = {q: r[q] for q in ("delta", "lo", "hi", "separated")}
            ra, rxy = SR.read_a(GD, rows["X_minus_P10"]), SR.read_xy(GD, rows["X_minus_Y"])
            per_arm[mode][arm][seed] = {"A": ra, "XY": rxy}
            _, dx = SR.load_dump(f"{a.out_dir}/{xt}/dump_{xt}")
            px = GK.partition(dx["cl"], dx["fl"], dx["g"])
            res["arms"][f"{k}:{mode}"] = {"mode": mode, "x_tag": xt, "y_tag": f"SREP_{suf}", "rows": rows, "read_a": ra, "read_xy": rxy}
            res["descriptive"][f"{k}:{mode}"] = {"curvature_partition_X": GK.public(px), "curvature_partition_Y": GK.public(py)}
            print(k, mode, "ADE6 X-P10 %+.4f X-Y %+.4f | curv X-P10 %+.4f X-Y %+.4f | M1 X-Y %+.4f | sep-worse X-P10 %s" % (rows["X_minus_P10"]["ADE6"]["delta"], rows["X_minus_Y"]["ADE6"]["delta"],
                  rows["X_minus_P10"]["curvature MAE"]["delta"], rows["X_minus_Y"]["curvature MAE"]["delta"], rows["X_minus_Y"]["M1 crawl share"]["delta"], ra["separated_worse_rows"]), flush=True)
    res["outcome"] = {m: {SR.ARM_NAME[arm]: SR.arm_outcome(per_arm[m][arm]) for arm in ("CTRL", "SOUP")} for m in MODES}
    json.dump(res, open(f"{a.raw}/FEAS2_T1.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    print("OUTCOME", {m: {q: v["outcome"] for q, v in o.items()} for m, o in res["outcome"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
