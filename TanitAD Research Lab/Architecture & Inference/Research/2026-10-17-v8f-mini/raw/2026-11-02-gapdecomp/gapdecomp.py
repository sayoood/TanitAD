#!/usr/bin/env python3
"""gapdecomp.py -- H-GAPDECOMP-1: Read A's gap against P10, split by window class (PREREG_FROZEN.md; zero training, CPU only).

Pure layer (numpy only, unit-tested by test_gapdecomp.py): classify, decompose, verdicts.  Heavy layer (main): reads the per-window arrays that extract_arrays.py wrote from the
instruments of the SOUP / LAUNCH-INTENT readers, runs the known-answer controls K1-K5, then the split.  Nothing here trains or rolls out a model.

    python gapdecomp.py --arrays F:/TanitAD-scratch/gapdecomp/arrays --out CLASS_SPLIT.json

Identity: a row is mean over its n valid windows of d = a - b, so  Delta = sum_c w_c Delta_c  with w_c = n_c / n.
Bootstrap: the instrument's own draw stream (taniteval.ci: default_rng(seed), choice(uniq, n_ep) over the row's own episode pool), class statistics recomputed on every draw.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ART = "E:/Projects/TanitAD-artifacts"
CI_PATH = f"{ART}/dagt1_tree_b43c7a9a/taniteval/taniteval/ci.py"
SOUP_READ = "F:/TanitAD-scratch/soup_t1_out/read/SOUP_READ.json"
LI_READ = "F:/TanitAD-scratch/launch_intent_t1_out/read_A1/LAUNCH_INTENT_READ.json"
CRAWL01 = "F:/TanitAD-scratch/gapdecomp/banked/crawl01_arrays.npz"
CRAWL01_SHA = "4ed58b65a83817b7516afe69ebf662b6e4876722d92e7e92ba8d90abdbdcc0ae"
PIN_SOUP_READ = "dca10c07f5fabda9ddc9ea2725f425f53f1462bec8488d58ebc92ecd8d33de6a"
PIN_LI_READ = "e451e1a6ba44e1d30e97561456476dc4223cd916bd3be96163cec4b02844bf23"

CLASSES = ("L", "S", "O", "M")
CLASS_NAME = {"L": "GT-launch", "S": "GT-stay", "O": "other-stopped", "M": "moving, no launch"}
TRIGGER = ("L", "S")
OTHER = ("O", "M")
LABELS = ("ADE6", "speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG", "heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE",
          "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)", "TAC-LAT", "TAC-LON", "S-POS@30", "S-POS@50", "S-POS@80", "M1 crawl share")
HIGHER_BETTER = ("TAC-LAT", "TAC-LON")
VERDICT_ROWS = ("ADE6", "yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share")
SEEDS = (0, 1)
ARMS = ("CTRL", "REP", "SOUP")
ARM_NAME = {"CTRL": "DAG-OACC", "REP": "REP", "SOUP": "SOUP"}
POOL = {l: ("all" if l.startswith("S-POS") else "masked") for l in LABELS}
N_BOOT, SEED = 2000, 0
TOL = 6e-5                      # the banked values are rounded to 4 dp
SHARE_HI, SHARE_LO, CI_MID = 0.7, 0.3, 0.5
STOP_V, LAUNCH_V0_MAX, DV, DT = 0.5, 6.0, 1.0, 0.2


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------ pure layer
def classify(v0, g):
    """-> (cls [N] int8 index into CLASSES, masks dict).  Definitions of H-CRAWL-01 (crawl_diag.py:144, mlaunch.py:84).  Raises on a non-finite v0 or a track of another shape."""
    v0 = np.asarray(v0, np.float64)
    g = np.asarray(g, np.float64)
    if v0.ndim != 1 or g.shape != (len(v0), 30, 2):
        raise ValueError(f"v0 {v0.shape} / g {g.shape}: expected [N] and [N, 30, 2]")
    if not np.isfinite(v0).all():
        raise ValueError("a window with a non-finite v0 has no class")
    last = np.linalg.norm(g[:, 29] - g[:, 28], axis=-1) / DT
    K = np.concatenate([np.zeros((len(g), 1, 2)), g], axis=1)
    top = (np.linalg.norm(np.diff(K, axis=1), axis=-1) / DT).max(axis=1)
    launch = (v0 < LAUNCH_V0_MAX) & (last >= v0 + DV)
    stay = (v0 < STOP_V) & (top < v0 + DV)
    other = (v0 < STOP_V) & ~launch & ~stay
    moving = (v0 >= STOP_V) & ~launch
    masks = {"L": launch, "S": stay, "O": other, "M": moving}
    total = launch.astype(int) + stay + other + moving
    if not (total == 1).all():
        raise ValueError(f"the four classes do not partition the windows: {int((total != 1).sum())} windows are in 0 or >= 2 classes")
    cls = np.zeros(len(v0), np.int8)
    for i, c in enumerate(CLASSES):
        cls[masks[c]] = i
    return cls, masks


def sep_worse(label, delta, lo, hi):
    """Separated on the worse side of the row's direction (TAC rows: higher is better; every other row: lower is better)."""
    sep = bool(lo > 0 or hi < 0)
    return bool(sep and ((delta < 0) if label in HIGHER_BETTER else (delta > 0)))


def draw_counts(uniq, n_boot=N_BOOT, seed=SEED):
    """[n_boot, n_ep] episode multiplicities of the instrument's own draw stream (taniteval.ci._draws): default_rng(seed), choice(uniq, n_ep, replace=True) per draw."""
    rng = np.random.default_rng(seed)
    n_ep = len(uniq)
    M = np.zeros((n_boot, n_ep))
    for b in range(n_boot):
        pick = rng.choice(uniq, size=n_ep, replace=True)
        M[b] = np.bincount(np.searchsorted(uniq, pick), minlength=n_ep)
    return M


_DRAWS = {}


def cached_draws(uniq, n_boot, seed):
    key = (n_boot, seed, len(uniq), hashlib.sha1("|".join(map(str, uniq)).encode()).hexdigest())
    if key not in _DRAWS:
        _DRAWS[key] = draw_counts(uniq, n_boot, seed)
    return _DRAWS[key]


def _pct(v):
    v = np.asarray(v, np.float64)
    f = v[np.isfinite(v)]
    if not len(f):
        return None, None, int(len(v))
    lo, hi = np.percentile(f, [2.5, 97.5])
    return float(lo), float(hi), int(len(v) - len(f))


def decompose(a, b, eid, cls, pool="masked", n_boot=N_BOOT, seed=SEED):
    """Class split of  mean(a - b)  over the pairs where both are finite.  pool 'masked': the draws resample the episodes that hold a valid pair (t1k_read.paired);
    pool 'all': all episodes of eid, ineligible windows staying in the pool (t1hd_read.srow).  -> dict (floats unrounded)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    cls = np.asarray(cls)
    if not (a.shape == b.shape == cls.shape == np.asarray(eid).shape):
        raise ValueError("a, b, cls, eid must be aligned")
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        raise ValueError("no valid pair")
    ep = np.asarray([str(x) for x in eid])
    uniq = np.unique(ep[ok] if pool == "masked" else ep)
    n_ep = len(uniq)
    ei = np.searchsorted(uniq, ep)
    C = len(CLASSES)
    S, N = np.zeros((n_ep, C)), np.zeros((n_ep, C))
    d = np.where(ok, a - b, 0.0)
    np.add.at(S, (ei[ok], cls[ok]), d[ok])
    np.add.at(N, (ei[ok], cls[ok]), 1.0)
    Sc, Nc = S.sum(0), N.sum(0)
    n = Nc.sum()
    delta = float(Sc.sum() / n)
    M = cached_draws(uniq, n_boot, seed)
    Sb, Nb = M @ S, M @ N
    Ntot, Stot = Nb.sum(1), Sb.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_b = Stot / Ntot
        dc_b = np.where(Nb > 0, Sb / np.where(Nb > 0, Nb, 1.0), np.nan)
        con_b = Sb / Ntot[:, None]
        sh_b = np.where(Stot[:, None] != 0, Sb / np.where(Stot != 0, Stot, 1.0)[:, None], np.nan)
        ti = [CLASSES.index(c) for c in TRIGGER]
        oi = [CLASSES.index(c) for c in OTHER]
        trig_b = np.where(Stot != 0, Sb[:, ti].sum(1) / np.where(Stot != 0, Stot, 1.0), np.nan)
        dprime_b = Sb[:, oi].sum(1) / Ntot
    lo, hi, _ = _pct(d_b)
    out = {"n_windows": int(n), "n_episodes_pool": int(n_ep), "pool": pool, "delta": delta, "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0),
           "classes": {}, "n_boot": int(n_boot), "seed": int(seed)}
    for i, c in enumerate(CLASSES):
        cl_lo, cl_hi, cl_nan = _pct(dc_b[:, i])
        co_lo, co_hi, _ = _pct(con_b[:, i])
        sh_lo, sh_hi, sh_nan = _pct(sh_b[:, i])
        out["classes"][c] = {"n": int(Nc[i]), "n_episodes": int((N[:, i] > 0).sum()), "w": float(Nc[i] / n),
                             "delta": (float(Sc[i] / Nc[i]) if Nc[i] > 0 else None), "delta_lo": cl_lo, "delta_hi": cl_hi, "delta_n_nan_draws": cl_nan,
                             "contribution": float(Sc[i] / n), "contribution_lo": co_lo, "contribution_hi": co_hi,
                             "share": (float(Sc[i] / Sc.sum()) if Sc.sum() != 0 else None), "share_lo": sh_lo, "share_hi": sh_hi, "share_n_nan_draws": sh_nan}
    t_lo, t_hi, t_nan = _pct(trig_b)
    out["trigger"] = {"share": (float(Sc[ti].sum() / Sc.sum()) if Sc.sum() != 0 else None), "lo": t_lo, "hi": t_hi, "n_nan_draws": t_nan, "contribution": float(Sc[ti].sum() / n),
                      "w": float(Nc[ti].sum() / n)}
    dp_lo, dp_hi, _ = _pct(dprime_b)
    dprime = float(Sc[oi].sum() / n)
    out["trigger_zeroed"] = {"delta": dprime, "lo": dp_lo, "hi": dp_hi, "separated": bool(dp_lo > 0 or dp_hi < 0)}
    out["_share_sum_max_abs_dev_over_draws"] = float(np.nanmax(np.abs(sh_b.sum(1) - 1.0))) if np.isfinite(sh_b).any() else None
    out["_class_sum_identity_abs_dev"] = float(abs(sum(out["classes"][c]["w"] * (out["classes"][c]["delta"] or 0.0) for c in CLASSES) - delta))
    return out


def trigger_verdict(row):
    """PREREG_FROZEN verdict of one row at one seed from decompose()'s dict."""
    if not row["separated"]:
        return "NOT DEFINED"
    t = row["trigger"]
    if t["share"] is None or t["lo"] is None:
        return "NOT DEFINED"
    if t["share"] >= SHARE_HI and t["lo"] > CI_MID:
        return "TRIGGER-BOUND"
    if t["share"] <= SHARE_LO and t["hi"] < CI_MID:
        return "NOT TRIGGER-BOUND"
    return "MIXED"


def arm_verdict(per_seed):
    """per_seed = {0: verdict, 1: verdict} -> the arm's verdict."""
    v = [per_seed[s] for s in SEEDS]
    if all(x == "TRIGGER-BOUND" for x in v):
        return "TRIGGER-BOUND"
    if all(x == "NOT TRIGGER-BOUND" for x in v):
        return "NOT TRIGGER-BOUND"
    nd = [f"s{s}" for s in SEEDS if per_seed[s] == "NOT DEFINED"]
    if nd:
        return "NOT DEFINED at " + ", ".join(nd) + (" (other seed: " + ", ".join(per_seed[s] for s in SEEDS if per_seed[s] != "NOT DEFINED") + ")" if len(nd) < 2 else "")
    return "MIXED"


# ------------------------------------------------------------------------------------------------------------------------------ heavy layer
def load_ci():
    spec = importlib.util.spec_from_file_location("tanitad_ci", CI_PATH)
    m = importlib.util.module_from_spec(spec)
    sys.modules["tanitad_ci"] = m
    spec.loader.exec_module(m)
    return m


def instrument_row(CI, a, b, eid, pool):
    """What the readers do with one row's arrays: t1k_read.paired (masked) or t1hd_read.srow (NaN-paired, all episodes) -> taniteval.ci.paired_episode_cluster_bootstrap."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if pool == "masked":
        return CI.paired_episode_cluster_bootstrap(a[m], b[m], np.asarray(eid)[m], n_boot=N_BOOT, seed=SEED)
    return CI.paired_episode_cluster_bootstrap(np.where(m, a, np.nan), np.where(m, b, np.nan), np.asarray(eid), n_boot=N_BOOT, seed=SEED)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frozen-dir", default=HERE)
    a_ = ap.parse_args(argv)
    t_start = time.time()
    fz = open(f"{a_.frozen_dir}/FROZEN_SHA.txt", encoding="utf-8").read()
    want = fz.split("sha256:")[1].split()[0]
    got = sha256_file(f"{a_.frozen_dir}/PREREG_FROZEN.md")
    if want != got:
        print(f"REFUSED: PREREG_FROZEN.md sha256 {got} != frozen {want}", file=sys.stderr)
        return 2
    frozen_clock = [ln for ln in fz.splitlines() if ln.startswith("wall_clock:")][0].split(":", 1)[1].strip()
    if sha256_file(CRAWL01) != CRAWL01_SHA or sha256_file(LI_READ) != PIN_LI_READ or sha256_file(SOUP_READ) != PIN_SOUP_READ:
        print("REFUSED: a pinned banked input has another sha256", file=sys.stderr)
        return 2
    CI = load_ci()
    sw = json.load(open(SOUP_READ, encoding="utf-8"))["tables"]
    li = json.load(open(LI_READ, encoding="utf-8"))["read_a"]
    banked = {"CTRL": sw["CTRL_P10"], "SOUP": sw["SOUP_P10"], "REP": li["REP_P10"]}
    res = {"hypothesis": "H-GAPDECOMP-1", "tier": "T1-kin", "evidence_class": "MEASURED (ours; arrays in F:/TanitAD-scratch/gapdecomp/arrays, shas in MANIFEST.txt)",
           "estimator": f"paired episode-cluster bootstrap, B={N_BOOT}, seed {SEED}, the instrument's own draw stream per row; answers another draw of episodes only",
           "prereg_sha256": got, "prereg_frozen_clock": frozen_clock, "run_clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "inputs": {"soup_read_sha256": PIN_SOUP_READ, "li_read_sha256": PIN_LI_READ, "crawl01_arrays_sha256": CRAWL01_SHA}, "controls": {}, "arms": {}, "verdicts": {}, "beside": {}}
    ctl = res["controls"]
    z0 = np.load(CRAWL01)
    for arm in ARMS:
        res["arms"][arm] = {}
        for s in SEEDS:
            p = f"{a_.arrays}/{arm}_s{s}.npz"
            z = np.load(p)
            labs = tuple(str(x) for x in z["labels"])
            if labs != LABELS:
                print(f"REFUSED: {p} rows {labs} != {LABELS}", file=sys.stderr)
                return 2
            eid, v0, g = z["eid"], z["v0"], z["g"]
            cls, masks = classify(v0, g)
            tag = f"{arm}_s{s}"
            counts = {c: int(masks[c].sum()) for c in CLASSES}
            ctl[f"K1:{tag}"] = {"PASS": bool(sum(counts.values()) == 2399 and np.array_equal(masks["L"], z0["launch"]) and np.array_equal(masks["S"], z0["stay"]) and np.array_equal(eid, z0["eid"])
                                             and int(masks["L"].sum()) == 327 and len(np.unique(eid[masks["L"]])) == 49), "counts": counts,
                                "episodes": {c: int(len(np.unique(eid[masks[c]]))) for c in CLASSES}}
            rows = {}
            for i, lab in enumerate(LABELS):
                a, b = z[f"a_{i}"], z[f"b_{i}"]
                pool = POOL[lab]
                r = decompose(a, b, eid, cls, pool)
                bk = banked[arm][f"s{s}"][lab]
                inst = instrument_row(CI, a, b, eid, pool)
                dp = inst.get("display_dp", 4)
                tol_i = 0.5 * 10 ** (-dp) + 1e-12
                k2 = {"delta": abs(r["delta"] - bk["delta"]) <= TOL, "lo": abs(r["lo"] - bk["lo"]) <= TOL, "hi": abs(r["hi"] - bk["hi"]) <= TOL, "separated": r["separated"] == bk["separated"],
                      "separated_worse": sep_worse(lab, r["delta"], r["lo"], r["hi"]) == bool(bk["separated_worse"])}
                k4 = {"delta": abs(r["delta"] - inst["delta"]) <= tol_i, "lo": abs(r["lo"] - inst["lo"]) <= tol_i, "hi": abs(r["hi"] - inst["hi"]) <= tol_i}
                bn = bk.get("n")
                k2["n"] = True if (not isinstance(bn, str) or "/" not in bn or bn.startswith("ticks")) else bn == f"{r['n_windows']}/{r['n_episodes_pool']}"
                # (X, X): the arm against itself reads exactly 0.0 in every class, interval [0, 0]
                zr = decompose(a, a, eid, cls, pool)
                k3 = bool(zr["delta"] == 0.0 and zr["lo"] == 0.0 and zr["hi"] == 0.0 and all((zr["classes"][c]["delta"] in (0.0, None)) for c in CLASSES))
                k5 = bool(r["_class_sum_identity_abs_dev"] <= 1e-12 and (r["_share_sum_max_abs_dev_over_draws"] is None or r["_share_sum_max_abs_dev_over_draws"] <= 1e-9)
                          and sum(r["classes"][c]["n"] for c in CLASSES) == r["n_windows"])
                r["_known_answer"] = {"K2_vs_banked": k2, "K3_xx_zero": k3, "K4_vs_instrument": k4, "K5_identity": k5, "banked": {k: bk[k] for k in ("delta", "lo", "hi", "separated", "separated_worse", "n")},
                                      "instrument": {k: inst[k] for k in ("delta", "lo", "hi")}}
                r["decomposable"] = bool(all(k2.values()) and k3 and all(k4.values()) and k5)
                if not r["decomposable"]:
                    r["not_decomposable_reason"] = {"K2": k2, "K3": k3, "K4": k4, "K5": k5}
                    for c in CLASSES:                                                  # no class number is quoted for a row that failed a known answer
                        r["classes"][c] = {"n": r["classes"][c]["n"], "REFUSED": True}
                    r["trigger"], r["trigger_zeroed"] = None, None
                rows[lab] = r
            res["arms"][arm][f"s{s}"] = rows
            ctl[f"K2K3K4K5:{tag}"] = {"PASS": all(rows[l]["decomposable"] for l in LABELS), "failed_rows": [l for l in LABELS if not rows[l]["decomposable"]]}
    all_pass = all(v["PASS"] for v in ctl.values())
    res["controls_PASS"] = bool(all_pass)
    # verdicts and the beside block
    for arm in ARMS:
        res["verdicts"][arm] = {}
        for lab in VERDICT_ROWS:
            ps = {}
            for s in SEEDS:
                row = res["arms"][arm][f"s{s}"][lab]
                ps[s] = trigger_verdict(row) if row["decomposable"] else "NOT DECOMPOSABLE"
            res["verdicts"][arm][lab] = {"s0": ps[0], "s1": ps[1], "arm": arm_verdict(ps) if all(v != "NOT DECOMPOSABLE" for v in ps.values()) else "NOT DECOMPOSABLE"}
        d2 = {}
        for s in SEEDS:
            rows = res["arms"][arm][f"s{s}"]
            worse = [l for l in LABELS if rows[l]["decomposable"] and sep_worse(l, rows[l]["delta"], rows[l]["lo"], rows[l]["hi"])]
            worse_z = [l for l in LABELS if rows[l]["decomposable"] and sep_worse(l, rows[l]["trigger_zeroed"]["delta"], rows[l]["trigger_zeroed"]["lo"], rows[l]["trigger_zeroed"]["hi"])]
            d2[f"s{s}"] = {"D2_worse_rows": worse, "D2": not worse, "D2_trigger_zeroed_worse_rows": worse_z, "D2_trigger_zeroed": not worse_z}
        d2["both_D2"] = bool(d2["s0"]["D2"] and d2["s1"]["D2"])
        d2["both_D2_trigger_zeroed"] = bool(d2["s0"]["D2_trigger_zeroed"] and d2["s1"]["D2_trigger_zeroed"])
        res["beside"][arm] = {"evidence_class": "ESTIMATED (decomposition arithmetic, not a retrained model)", **d2}
    res["wall_s"] = round(time.time() - t_start, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a_.out)), exist_ok=True)
    with open(a_.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("GAPDECOMP", "controls PASS" if all_pass else "CONTROLS FAILED", "->", a_.out)
    if not all_pass:
        for k, v in ctl.items():
            if not v["PASS"]:
                print("  FAILED", k, json.dumps(v)[:400])
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
