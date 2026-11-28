#!/usr/bin/env python3
"""histswap_lf_read.py -- the read of H-HISTSWAP-LF-1 (PREREG_HISTSWAP_LF_FROZEN.md): two leak-free history arms on S, HCV (``--histswap cv_exec``, deployable) and HLVL (``--histswap cv_logged``, ORACLE level only),
Y = KAPPA-T1's fresh S runs (reused), DAG-OACC and SOUP, s0 / s1.

HISTSWAP's reader (histswap_t1_read.py) with the single lever replaced by two history forms (FEAS2's MODES loop).  Per-window arrays: SMOOTH-T1's extract_smooth.py (read-only; own work dir;
--floor-guard-m 1.0).  Rows and rule: SMOOTH-T1's pure layer (read_a, read_xy, arm_outcome: DOMINANT > HARMS > MOVED > NO EFFECT) on gapdecomp.decompose (paired episode-cluster bootstrap, B = 2000, seed 0;
answers another draw of EPISODES only), plus this file's pure ``decide_m1`` (HISTSWAP's M1 reading, per arm) and ``interpret`` (the card's committed table).
Known answers computed before any row: KY (Y pins), KA1 (a fresh ``--histswap off`` rerun with the LF tool equals the reused Y), KC1 (the flag reached the run and v0_in is what the card says), KC2 (the CV history
reached the network: the tick-0 plan differs from Y's), KC3 (the plan is committed to the executed state: tick 1's executed position = tick 0's pose composed with the tick-0 plan), KF (floor), K3, KB1.
Tier T1-kin; evidence class MEASURED (ours).  HLVL is an ORACLE (the logged speed is the human's): its absolute rows are not performance.  S-POS rows of either arm mix the CV pose the S head read with the executed
pose the extractor frames them in: UNVERIFIED, outside the reading.
"""
import argparse
import importlib.util
import json
import math
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
OUT_X = "F:/TanitAD-scratch/histswap_lf/out"
WORK = "F:/TanitAD-scratch/histswap_lf/work/rig"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
# key -> (tag suffix, arm, seed)
ARMSEEDS = {"CTRL_s0": ("DAGOACC_s0", "CTRL", 0), "CTRL_s1": ("DAGOACC_s1", "CTRL", 1), "SOUP_s0": ("SOUP_s0", "SOUP", 0), "SOUP_s1": ("SOUP_s1", "SOUP", 1)}
#: mode key -> (the --histswap value, the run-tag prefix)
MODES = {"HCV": ("cv_exec", "HCV"), "HLVL": ("cv_logged", "HLV")}
KA1_TAG, KA1_REF = "SREP5_DAGOACC_s0", "SREP_DAGOACC_s0"
TOL = 1e-4
FL_GROSS_M = 1.0   # READER_FIX_FLOOR.md section 3.1
M1 = "M1 crawl share"
SHARE_BAR = 0.5
REPLAN_M = 10
EXEC_TOL_M = 5e-3


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ pure layer (unit-checkable, no file access)
def m1_cell(xy, yp):
    """One arm-seed cell from the M1 rows of X - Y (``xy``) and Y - P10 (``yp``), each {delta, lo, hi, separated}.
    ``better``: X - Y separated on the better side (M1 is a share, lower is better: delta < 0).  ``share_closed`` = -(X - Y) / (Y - P10): the part of Y's excess over P10 that X removes;
    undefined (None) when Y - P10 is not > 0 (no excess to close), and an undefined cell counts as 0 in the mean (the conservative reading)."""
    better = bool((xy["lo"] > 0 or xy["hi"] < 0) and xy["delta"] < 0)
    ex = float(yp["delta"])
    share = (-float(xy["delta"]) / ex) if ex > 0 else None
    return {"better": better, "xy_delta": float(xy["delta"]), "yp_delta": ex, "yp_separated_worse": bool(yp["separated"] and ex > 0), "share_closed": share}


def decide_m1(cells):
    """HISTSWAP's M1 reading of ONE arm over the four cells (DAG-OACC s0 / s1, SOUP s0 / s1).
    CLOSES iff X - Y is separated-better on M1 at ALL four cells AND the mean share closed (undefined = 0) is >= 0.5 (HISTSWAP's SUPPORTED condition, verbatim);
    NOT CLOSES iff X - Y is separated-better at fewer than 3 of the 4 cells; anything else is MIXED."""
    if len(cells) != 4:
        raise ValueError("the reading is defined on exactly four cells")
    n_better = sum(c["better"] for c in cells.values())
    shares = [0.0 if c["share_closed"] is None else c["share_closed"] for c in cells.values()]
    mean_share = float(np.mean(shares))
    pooled_den = sum(c["yp_delta"] for c in cells.values())
    pooled = (-sum(c["xy_delta"] for c in cells.values()) / pooled_den) if pooled_den > 0 else None
    if n_better == 4 and mean_share >= SHARE_BAR:
        out = "CLOSES"
    elif n_better < 3:
        out = "NOT CLOSES"
    else:
        out = "MIXED"
    return {"reading": out, "closes": out == "CLOSES", "n_cells_separated_better": int(n_better), "mean_share_closed": mean_share, "pooled_share_closed": pooled, "share_bar": SHARE_BAR}


def interpret(hcv, hlvl):
    """The card's committed interpretation from the two arms' ``decide_m1`` blocks.
    HCV closes                      -> (A) INERTIA: O follows the TREND of its own executed history; HCV is a deployable inference lever (DOMINANT is reported separately by the outcome rule).
    HCV does not close, HLVL does   -> (B) LEVEL: O needs the human's speed LEVEL; it must come from perception or training, not from the history form.
    neither closes                  -> (C) INTERACTION: the trend + level interaction of HISTSWAP's full logged history carries it.
    ``hcv_mixed`` / ``hlvl_mixed`` flag an arm that did not CLOSE but did not fail outright either (the reading stands; the flag is stated beside it)."""
    if hcv["closes"]:
        r = "A_INERTIA"
    elif hlvl["closes"]:
        r = "B_LEVEL"
    else:
        r = "C_INTERACTION"
    return {"reading": r, "hcv_closes": bool(hcv["closes"]), "hlvl_closes": bool(hlvl["closes"]), "hcv_mixed": hcv["reading"] == "MIXED", "hlvl_mixed": hlvl["reading"] == "MIXED",
            "hlvl_discordant": bool(hcv["closes"] and not hlvl["closes"])}


def exec_chain_dev(t, m: int = REPLAN_M):
    """KC3: per window, the executed position at tick 1 (the ticks file's ``pose`` at tick 1) minus [the tick-0 executed pose composed with the tick-0 plan's knot ``m``] -- the plan is committed from the executed state.
    ``t`` = the ticks dict (``tick``, ``win``, ``pose`` [N, 4], ``plan`` [N, 30, 2]).  Returns the per-window deviation (m) array."""
    tick, win = np.asarray(t["tick"]), np.asarray(t["win"])
    i0 = {int(win[i]): int(i) for i in np.where(tick == 0)[0]}
    i1 = {int(win[i]): int(i) for i in np.where(tick == 1)[0]}
    dev = []
    for w, a in i0.items():
        if w not in i1:
            continue
        b = i1[w]
        p0 = np.asarray(t["pose"][a], np.float64)
        k = np.asarray(t["plan"][a][m - 1], np.float64)
        c, s = math.cos(p0[2]), math.sin(p0[2])
        want = p0[:2] + np.array([k[0] * c - k[1] * s, k[0] * s + k[1] * c])
        dev.append(float(np.hypot(*(np.asarray(t["pose"][b][:2], np.float64) - want))))
    return np.asarray(dev)


# ------------------------------------------------------------------------------------------------ heavy layer
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


def _dev(p, q):
    """max abs difference of two arrays; inf when their NaN masks differ, 0.0 when both are all-NaN in the same places."""
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    if p.shape != q.shape or not np.array_equal(np.isnan(p), np.isnan(q)):
        return float("inf")
    d = np.abs(p - q)
    return float(d[np.isfinite(d)].max()) if np.isfinite(d).any() else 0.0


def tickload(path):
    return {q: v for q, v in np.load(path, allow_pickle=True).items() if q not in ("tag", "ckpt")}


def swap_stats(out_dir, tag):
    h = json.load(open(f"{out_dir}/{tag}/t1v8f_{tag}.json", encoding="utf-8"))["t1v8f"]["histswap"]
    return h["bucket_stats_model_ticks"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_X)
    ap.add_argument("--raw", default=f"{PK}/raw")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    want = [ln.split()[2] for ln in open(f"{PK}/FROZEN_SHA.txt", encoding="utf-8") if ln.startswith("PREREG_HISTSWAP_LF_FROZEN.md sha256")][0]
    got = SR.sha256_file(f"{PK}/PREREG_HISTSWAP_LF_FROZEN.md")
    if want != got:
        print(f"REFUSED: PREREG_HISTSWAP_LF_FROZEN.md {got} != frozen {want}", file=sys.stderr)
        return 2
    arr_dir = f"{a.raw}/arrays"
    res = {"hypothesis": "H-HISTSWAP-LF-1", "lever": "--histswap cv_exec (HCV, deployable) and cv_logged (HLVL, ORACLE level only): a constant-twist ego history; the plan is committed to the executed state", "tier": "T1-kin",
           "evidence_class": "MEASURED (ours; artifacts in MANIFEST.txt)", "prereg_sha256": got,
           "estimator": f"paired episode-cluster bootstrap, B={a.n_boot}, seed 0 (taniteval.ci draw stream via gapdecomp.decompose); answers another draw of EPISODES only; X and Y share the checkpoint, so no training-seed variance is in X - Y",
           "run_clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "controls": {}, "arms": {}, "descriptive": {}}
    GD = SR.load_gd()
    MAN = json.load(open(SR.MANIFEST, encoding="utf-8"))
    pins, pins_ok = y_pins()
    res["controls"]["KY_reused_Y_equals_frozen_pins"] = {"files": pins, "PASS": bool(pins_ok)}
    # ---- KA1: ONE fresh --histswap off rerun (the LF tool) of an arm-seed equals the reused Y
    _, d1 = SR.load_dump(f"{OUT_X}/{KA1_TAG}/dump_{KA1_TAG}")
    _, d0 = SR.load_dump(f"{OUT_Y}/{KA1_REF}/dump_{KA1_REF}")
    same_d, dev_d, _, bad_d, only_d = SR.cmp_arrays(d1, d0)
    t1 = tickload(f"{OUT_X}/{KA1_TAG}/ticks/ticks_{KA1_TAG}.npz")
    t0 = tickload(f"{OUT_Y}/{KA1_REF}/ticks/ticks_{KA1_REF}.npz")
    same_t, dev_t, _, bad_t, only_t = SR.cmp_arrays(t1, t0)
    res["controls"]["KA1_fresh_off_rerun_equals_reused_Y"] = {"rerun": KA1_TAG, "reused": KA1_REF, "dump_bit_identical": same_d, "dump_max_abs_dev_m": dev_d, "dump_differing_keys": bad_d,
                                                              "ticks_bit_identical": same_t, "ticks_max_abs_dev": dev_t, "ticks_differing_keys": bad_t, "key_set_diff": only_d + only_t,
                                                              "PASS": bool((same_d and same_t) or (dev_d <= TOL and dev_t <= TOL and not only_d and not only_t))}
    kc1, kc2, kc3, kf = {}, {}, {}, {}
    for k, (suf, arm, seed) in ARMSEEDS.items():
        yt = f"SREP_{suf}"
        _, dy = SR.load_dump(f"{OUT_Y}/{yt}/dump_{yt}")
        ty = tickload(f"{OUT_Y}/{yt}/ticks/ticks_{yt}.npz")
        db_ref = SR.load_dump(MAN["arms"]["CTRL"][str(seed)]["t1"]["dump_dir"])[1]
        wk = ("clip_index", "ws", "eid", "g", "v0")
        y0 = ty["tick"] == 0
        for mode, (flag, pfx) in MODES.items():
            xt = f"{pfx}_{suf}"
            kk = f"{k}:{mode}"
            tx = tickload(f"{a.out_dir}/{xt}/ticks/ticks_{xt}.npz")
            _, dx = SR.load_dump(f"{a.out_dir}/{xt}/dump_{xt}")
            x0, later = tx["tick"] == 0, tx["tick"] >= 1
            d_in = np.abs(tx["v0_in"].astype(np.float64) - tx["v0"].astype(np.float64))
            if mode == "HCV":
                ok1 = bool(d_in.max() <= TOL)
            else:
                ok1 = bool(d_in[x0].max() <= TOL and (d_in[later] > 0.05).mean() > 0.05)
            kc1[kk] = {"str_histswap": str(tx["histswap"]), "v0_in_minus_v0_max_abs_all_ticks": float(d_in.max()), "v0_in_minus_v0_max_abs_tick0": float(d_in[x0].max()),
                       "frac_later_ticks_v0_in_ne_v0_gt_0p05": float((d_in[later] > 0.05).mean()), "mean_v0_exec_later_mps": float(tx["v0"][later].mean()), "mean_v0_in_later_mps": float(tx["v0_in"][later].mean()),
                       "PASS": bool(str(tx["histswap"]) == flag and ok1)}
            dev_plan0 = _dev(tx["plan_raw"][x0], ty["plan_raw"][y0]) if int(x0.sum()) == int(y0.sum()) else float("inf")
            kc2[kk] = {"tick0_plan_raw_max_abs_dev_vs_Y_m": dev_plan0, "n_tick0_rows": int(x0.sum()), "PASS": bool(int(x0.sum()) == int(y0.sum()) and dev_plan0 > 1e-3)}
            ec = exec_chain_dev(tx)
            kc3[kk] = {"n_windows": int(len(ec)), "max_dev_m": float(ec.max()) if len(ec) else None, "frac_within_tol": float((ec <= EXEC_TOL_M).mean()) if len(ec) else None, "tol_m": EXEC_TOL_M,
                       "PASS": bool(len(ec) > 0 and (ec <= EXEC_TOL_M).mean() >= 0.999)}
            kf[kk] = {"windows_X_equal_fresh_Y": bool(all(np.array_equal(dx[q], dy[q]) for q in wk)), "windows_fresh_Y_equal_banked_CTRL": bool(all(np.array_equal(dy[q], db_ref[q]) for q in wk)),
                      "floor_X_vs_fresh_Y_max_abs_m": float(np.abs(dx["fl"] - dy["fl"]).max()), "floor_fresh_Y_vs_banked_CTRL_max_abs_m": float(np.abs(dy["fl"] - db_ref["fl"]).max()),
                      "floor_X_vs_banked_CTRL_max_abs_m": float(np.abs(dx["fl"] - db_ref["fl"]).max())}
            kf[kk]["PASS"] = bool(kf[kk]["windows_X_equal_fresh_Y"] and kf[kk]["windows_fresh_Y_equal_banked_CTRL"] and kf[kk]["floor_X_vs_fresh_Y_max_abs_m"] <= TOL
                                  and kf[kk]["floor_fresh_Y_vs_banked_CTRL_max_abs_m"] <= FL_GROSS_M and kf[kk]["floor_X_vs_banked_CTRL_max_abs_m"] <= FL_GROSS_M)
    res["controls"]["KC1_flag_reached_the_run_and_v0_in_is_as_the_card_says"] = {"runs": kc1, "PASS": all(v["PASS"] for v in kc1.values())}
    res["controls"]["KC2_the_CV_history_reached_the_network"] = {"runs": kc2, "PASS": all(v["PASS"] for v in kc2.values())}
    res["controls"]["KC3_plan_committed_to_the_executed_state"] = {"runs": kc3, "PASS": all(v["PASS"] for v in kc3.values())}
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
        json.dump(res, open(f"{a.raw}/HISTSWAP_LF_T1.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
        print("REFUSED:", msg, {q: v["PASS"] for q, v in res["controls"].items()}, file=sys.stderr)
        return 3

    if not all(v["PASS"] for v in res["controls"].values()):
        return refuse("a known answer failed")
    kb1 = {}
    for k, (suf, arm, seed) in ARMSEEDS.items():
        zl = np.load(f"{SR.GD_DIR}/raw/arrays/{arm}_s{seed}.npz", allow_pickle=True)
        tgs = [("Y", f"SREP_{suf}", OUT_Y, f"{KAP_PK}/raw/arrays")] + [(f"X:{m}", f"{MODES[m][1]}_{suf}", a.out_dir, arr_dir) for m in MODES]
        for role, tg, od, ad in tgs:
            zz = np.load(extract(tg, arm, seed, od, ad), allow_pickle=True)
            lab_ok = tuple(str(q) for q in zz["labels"]) == tuple(str(q) for q in zl["labels"])
            eq = {str(l): bool(np.array_equal(zz[f"b_{i}"], zl[f"b_{i}"], equal_nan=True)) for i, l in enumerate(zl["labels"]) if not str(l).startswith("S-POS")}
            win = bool(all(np.array_equal(zz[q], zl[q]) for q in ("eid", "v0", "g", "ws", "clip_index")))
            kb1[f"{k}:{role}"] = {"labels_equal": lab_ok, "windows_equal_landed": win, "n_rows_b_equal": sum(eq.values()), "n_rows": len(eq), "failed": [q for q, v in eq.items() if not v], "PASS": bool(lab_ok and win and all(eq.values()))}
    res["controls"]["KB1_P10_side_equals_landed_banked_b"] = {"runs": kb1, "PASS": all(v["PASS"] for v in kb1.values())}
    if not all(v["PASS"] for v in res["controls"].values()):
        return refuse("KB1 failed")
    cells = {m: {} for m in MODES}
    per_arm = {m: {"CTRL": {}, "SOUP": {}} for m in MODES}
    for k, (suf, arm, seed) in ARMSEEDS.items():
        zy = np.load(extract(f"SREP_{suf}", arm, seed, OUT_Y, f"{KAP_PK}/raw/arrays"), allow_pickle=True)
        _, dy = SR.load_dump(f"{OUT_Y}/SREP_{suf}/dump_SREP_{suf}")
        ty = tickload(f"{OUT_Y}/SREP_{suf}/ticks/ticks_SREP_{suf}.npz")
        for mode, (flag, pfx) in MODES.items():
            xt = f"{pfx}_{suf}"
            zx = np.load(extract(xt, arm, seed, a.out_dir, arr_dir), allow_pickle=True)
            labels = tuple(str(q) for q in zx["labels"])
            assert labels == GD.LABELS and np.array_equal(zx["eid"], zy["eid"])
            cls, _ = GD.classify(zx["v0"], zx["g"])
            rows, full_m1 = {"X_minus_P10": {}, "X_minus_Y": {}, "Y_minus_P10": {}}, {}
            for i, lab in enumerate(labels):
                for nm, (u, v) in (("X_minus_P10", (zx[f"a_{i}"], zx[f"b_{i}"])), ("X_minus_Y", (zx[f"a_{i}"], zy[f"a_{i}"])), ("Y_minus_P10", (zy[f"a_{i}"], zy[f"b_{i}"]))):
                    r = GD.decompose(u, v, zx["eid"], cls, GD.POOL[lab], a.n_boot)
                    rows[nm][lab] = {q: r[q] for q in ("delta", "lo", "hi", "separated")}
                    if lab == M1:
                        full_m1[nm] = r                                         # the per-class split (L launch / S stay / O other-stopped / M moving) of the decision row, whole
            ra, rxy = SR.read_a(GD, rows["X_minus_P10"]), SR.read_xy(GD, rows["X_minus_Y"])
            cells[mode][k] = m1_cell(rows["X_minus_Y"][M1], rows["Y_minus_P10"][M1])
            per_arm[mode][arm][seed] = {"A": ra, "XY": rxy}
            _, dx = SR.load_dump(f"{a.out_dir}/{xt}/dump_{xt}")
            tx = tickload(f"{a.out_dir}/{xt}/ticks/ticks_{xt}.npz")
            lt_x, lt_y = tx["tick"] >= 1, ty["tick"] >= 1
            o_res = lambda t, m_: float((t["plan_raw"][m_][:, -1, 0] - t["P"][m_][:, -1, 0]).mean())  # noqa: E731  (O's own along-track contribution at 6 s, ticks >= 1; descriptive)
            res["arms"][f"{k}:{mode}"] = {"mode": mode, "flag": flag, "x_tag": xt, "y_tag": f"SREP_{suf}", "rows": rows, "read_a_X_vs_P10": ra, "read_xy_X_minus_Y": rxy, "m1_cell": cells[mode][k],
                                          "swap_bucket_differs_over_swap_ticks": swap_stats(a.out_dir, xt)}
            res["descriptive"][f"{k}:{mode}"] = {"M1_per_class_split": full_m1, "curvature_partition_X": GK.public(GK.partition(dx["cl"], dx["fl"], dx["g"])), "curvature_partition_Y": GK.public(GK.partition(dy["cl"], dy["fl"], dy["g"])),
                                                 "o_residual_x30_mean_m_ticks_ge1": {"X": o_res(tx, lt_x), "Y": o_res(ty, lt_y)},
                                                 "P_x30_mean_m_ticks_ge1": {"X": float(tx["P"][lt_x][:, -1, 0].mean()), "Y": float(ty["P"][lt_y][:, -1, 0].mean())},
                                                 "plan_x30_mean_m_ticks_ge1": {"X": float(tx["plan"][lt_x][:, -1, 0].mean()), "Y": float(ty["plan"][lt_y][:, -1, 0].mean())},
                                                 "note": "S-POS rows of X mix the CV pose the S head read with the executed pose the extractor frames them in: UNVERIFIED, not part of the reading"}
            print(k, mode, "M1 X-Y %+.4f [%+.4f, %+.4f] sep %s | Y-P10 %+.4f | share closed %s | ADE6 X-Y %+.4f | ADE6 X-P10 %+.4f" % (
                rows["X_minus_Y"][M1]["delta"], rows["X_minus_Y"][M1]["lo"], rows["X_minus_Y"][M1]["hi"], rows["X_minus_Y"][M1]["separated"], rows["Y_minus_P10"][M1]["delta"],
                cells[mode][k]["share_closed"], rows["X_minus_Y"]["ADE6"]["delta"], rows["X_minus_P10"]["ADE6"]["delta"]), flush=True)
    res["outcome"] = {m: {SR.ARM_NAME[arm]: SR.arm_outcome(per_arm[m][arm]) for arm in ("CTRL", "SOUP")} for m in MODES}
    res["m1_reading"] = {m: decide_m1(cells[m]) for m in MODES}
    res["interpretation"] = interpret(res["m1_reading"]["HCV"], res["m1_reading"]["HLVL"])
    json.dump(res, open(f"{a.raw}/HISTSWAP_LF_T1.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    print("OUTCOME", {m: {q: v["outcome"] for q, v in o.items()} for m, o in res["outcome"].items()})
    print("M1 READING", {m: (r["reading"], round(r["mean_share_closed"], 3), r["n_cells_separated_better"]) for m, r in res["m1_reading"].items()})
    print("INTERPRETATION", res["interpretation"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
