"""hierwhere.py -- PREREG_V8F_PROOF_ADD6 (HIERWHERE).  CPU, zero training, zero rollouts.  Tier T1-kin (logged frames): a diagnostic, never driving performance.

Usage:  python hierwhere.py run [--out-dir DIR] [--nboot 2000] [--smoke N]
Reads the banked DAG and FLAT-DAG dumps (read-only), recomputes the banked ADE6 difference (control h1) FIRST, then R3 (+h2), R1, R2, h3, h4, h5, the family rows.
Definitions and deviations: HIER_DEFS.md.  Exit 0 = all controls pass, 2 = a control failed (h1 failing stops the run before any read).
"""
import argparse
import glob
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
ART = "E:/Projects/TanitAD-artifacts"
PINNED_TREE = ART + "/dagt1_tree_b43c7a9a"            # the tool tree of the FLATDAG read (ci.py sha 38db571e..., four_families.py sha 96839329...)
sys.path.insert(0, PINNED_TREE + "/taniteval")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
from taniteval import ci as CI  # noqa: E402
from taniteval import four_families as FF  # noqa: E402

DIRS = {"DAG": ART + "/v8f-dagger-t1/dump_DAG_s{s}_m10_causal_heads", "FLAT": ART + "/v8f-flatdag-t1/dump_FLATDAG_s{s}_m10_causal_heads"}
STOPDEC = ART + "/dagland/motion_c/theory/stopbal/read/out/stopdec/stopdec_BAL-NULL_s{s}.npz"
SEEDS = (0, 1)
DT = 0.2
NB, BSEED = 2000, 0
F_SEED, F_TAC = 0.12075, 0.287
BANKED = {0: (-0.3451, -0.4421, -0.2506), 1: (-0.0544, -0.1605, 0.0547)}
THR = {"stop_mps": 0.5, "dv_mps": 2.0, "head_deg": 10.0}
LON = ("STOP", "DECEL", "ACCEL", "KEEP")
LAT = ("LEFT", "RIGHT", "STRAIGHT")
FAM_ROWS = (("LONGITUDINAL", "speed_mae_mps"), ("LONGITUDINAL", "along_mae_m"), ("LATERAL", "cross_mae_m"), ("LATERAL", "heading_mae_deg"),
            ("LATERAL", "curvature_mae_1pm"), ("LATERAL", "yaw_rate_mae_degps"))


# ------------------------------------------------------------------------------------------------ loading
def load_arm(d, smoke=0):
    fs = sorted(glob.glob(d + "/ep*.npz"))
    assert len(fs) == 141, (d, len(fs))
    if smoke:
        fs = fs[:smoke]
    out = {k: [] for k in ("g", "cl", "ws", "eid", "clip_index", "v0")}
    for f in fs:
        with np.load(f) as z:
            for k in out:
                out[k].append(np.asarray(z[k]))
    return {k: np.concatenate(v) for k, v in out.items()}


# ------------------------------------------------------------------------------------------------ geometry and modes (HIER_DEFS.md section 2)
def kinematics(traj):
    """traj [n,30,2] ego-frame metres, knot k at t=(k+1)*0.2 s.  -> speed [n,30] (step k covers (t_k-0.2, t_k], origin prepended, as four_families._seq_geometry),
    dheading_deg [n] (heading change from the first to the last VALID step, unwrapped over the valid steps; 0 when fewer than 2 valid steps)."""
    p = np.asarray(traj, np.float64)
    p = np.concatenate([np.zeros_like(p[:, :1]), p], axis=1)
    d = p[:, 1:] - p[:, :-1]
    ds = np.linalg.norm(d, axis=-1)
    speed = ds / DT
    valid = ds > FF.MIN_DS_MPS * DT
    head = np.arctan2(d[..., 1], d[..., 0])
    dh = np.zeros(len(p))
    for i in range(len(p)):
        v = np.flatnonzero(valid[i])
        if len(v) >= 2:
            u = np.unwrap(head[i, v])
            dh[i] = u[-1] - u[0]
    return speed, np.degrees(dh)


def classify(speed, dhead_deg, scale=1.0):
    """-> (lon [n] in 0..3 = STOP, DECEL, ACCEL, KEEP ; lat [n] in 0..2 = LEFT, RIGHT, STRAIGHT).  Thresholds x scale (h5)."""
    s_stop, s_dv, s_h = THR["stop_mps"] * scale, THR["dv_mps"] * scale, THR["head_deg"] * scale
    n = len(speed)
    # STOP: below s_stop at some time in [2 s, 6 s] and below to 6 s  <=>  exists j in 9..29 (t_j >= 2.0) with speed[j:] < s_stop for all later knots
    below = speed < s_stop
    suffix_all = np.logical_and.accumulate(below[:, ::-1], axis=1)[:, ::-1]          # suffix_all[:, j] = all(below[:, j:])
    stop = suffix_all[:, 9:].any(axis=1)
    dv = speed[:, 29] - speed[:, 0]
    lon = np.full(n, 3, np.int64)
    lon[dv >= s_dv] = 2
    lon[dv <= -s_dv] = 1
    lon[stop] = 0
    lat = np.full(n, 2, np.int64)
    lat[dhead_deg >= s_h] = 0
    lat[dhead_deg <= -s_h] = 1
    return lon, lat


def modes(traj, scale=1.0):
    sp, dh = kinematics(traj)
    return classify(sp, dh, scale)


def joint(lon, lat):
    return lon * 3 + lat


KEEP_STRAIGHT = 3 * 3 + 2


# ------------------------------------------------------------------------------------------------ errors
def comps(X, G):
    """per-window components of one arm against the logged future (copy of t1k_read.comps: four_families' definitions and masks; NaN where a mask is empty)."""
    Xt, Gt = torch.as_tensor(np.asarray(X, np.float64)), torch.as_tensor(np.asarray(G, np.float64))
    P, Q = FF._seq_geometry(Xt, DT), FF._seq_geometry(Gt, DT)
    e = torch.linalg.norm(Xt - Gt, dim=-1)
    both, bp = P["valid"] & Q["valid"], P["pair_valid"] & Q["pair_valid"]
    dh = P["heading"] - Q["heading"]
    dh = ((dh + math.pi) % (2 * math.pi) - math.pi).abs()

    def mm(x, m):
        s, c = (x * m).sum(1), m.sum(1)
        return torch.where(c > 0, s / c.clamp_min(1), torch.full_like(s, float("nan")))
    out = {"ade2_m": e[:, :10].mean(1), "ade6_m": e.mean(1), "ade26_m": e[:, 10:].mean(1),
           "speed_mae_mps": (P["speed"] - Q["speed"]).abs().mean(1), "along_mae_m": (P["along"] - Q["along"]).abs().mean(1),
           "cross_mae_m": (P["cross"] - Q["cross"]).abs().mean(1),
           "heading_mae_deg": torch.rad2deg(mm(dh, both)), "yaw_rate_mae_degps": torch.rad2deg(mm((P["yaw_rate"] - Q["yaw_rate"]).abs(), bp)),
           "curvature_mae_1pm": mm((P["curvature"] - Q["curvature"]).abs(), bp)}
    return {k: v.numpy().astype(np.float64) for k, v in out.items()}


# ------------------------------------------------------------------------------------------------ the estimator: paired episode-cluster bootstrap on taniteval.ci's own draws
def boot(eid, stats, n_boot=NB, seed=BSEED, alpha=0.05):
    """stats: {name: fn(sel)->float}.  The SAME resampled episodes for every statistic (taniteval.ci._draws: episodes with replacement, B n_boot, seed).
    -> {name: {point (full-sample), lo, hi, separated}} ; point = fn(all windows)."""
    uniq, idx = CI.episode_index(eid)
    allw = np.arange(len(eid))
    pts = {k: float(f(allw)) for k, f in stats.items()}
    acc = {k: np.empty(n_boot) for k in stats}
    for b, sel in enumerate(CI._draws(uniq, idx, n_boot, seed)):
        for k, f in stats.items():
            acc[k][b] = f(sel)
    out = {}
    for k in stats:
        a = acc[k]
        a = a[np.isfinite(a)]
        lo, hi = (float(x) for x in np.percentile(a, [100 * alpha / 2, 100 * (1 - alpha / 2)])) if len(a) else (float("nan"), float("nan"))
        out[k] = {"point": pts[k], "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0), "n_boot_finite": int(len(a))}
    return out


def mean_over(v, m=None):
    """fn(sel) -> mean of v over the finite entries of v (and mask m) among the selected windows."""
    v = np.asarray(v, np.float64)
    ok = np.isfinite(v) if m is None else (np.isfinite(v) & np.asarray(m, bool))
    vv = np.where(ok, v, 0.0)
    w = ok.astype(np.float64)

    def f(sel):
        c = w[sel].sum()
        return vv[sel].sum() / c if c > 0 else float("nan")
    return f


def total_over(v, m):
    """fn(sel) -> (1/N_sel) * sum over selected windows in mask m of v   (a term of the exact decomposition: divisor is ALL selected windows)."""
    vv = np.where(np.asarray(m, bool), np.asarray(v, np.float64), 0.0)

    def f(sel):
        return vv[sel].sum() / len(sel)
    return f


def nums(eid, m, v=None):
    """n windows (finite v, mask m) and n episodes."""
    ok = np.asarray(m, bool) if v is None else (np.asarray(m, bool) & np.isfinite(v))
    return int(ok.sum()), int(len(np.unique(np.asarray(eid)[ok])))


def row(r, n=None):
    d = {"point": round(r["point"], 6), "lo": round(r["lo"], 6), "hi": round(r["hi"], 6), "separated": r["separated"]}
    if n:
        d["n_windows"], d["n_episodes"] = n
    return d


# ------------------------------------------------------------------------------------------------ the decomposition (R3)
def decompose(eD, eF, cD, cF):
    """per-window terms; WITHIN + SELECTION + BOTH-MISS = eD - eF window by window (a partition of the windows by cD, cF)."""
    d = eD - eF
    w = cD & cF
    s = cD ^ cF
    bm = ~cD & ~cF
    return d, {"WITHIN": w, "SELECTION": s, "BOTH-MISS": bm}


def r3_rows(eid, eD, eF, cD, cF, n_boot):
    d, sets = decompose(eD, eF, cD, cF)
    stats = {"D": total_over(d, np.ones(len(d), bool))}
    for k, m in sets.items():
        stats[k] = total_over(d, m)
    r = boot(eid, stats, n_boot)
    out = {k: row(v) for k, v in r.items()}
    for k, m in sets.items():
        out[k]["n_windows"], out[k]["n_episodes"] = nums(eid, m)
    # control h2: the exact identity, on the point values (summed in float64 over the windows)
    ident = float(abs(sum(float(np.where(m, d, 0.0).sum()) for m in sets.values()) / len(d) - d.mean()))
    return out, ident


def outcome(r0, r1):
    """PREREG outcomes.  r0, r1: R3 ADE6 rows of seed 0 / seed 1 ({'D','WITHIN','SELECTION','BOTH-MISS'})."""
    s0, w0, s1, w1 = r0["SELECTION"], r0["WITHIN"], r1["SELECTION"], r1["WITHIN"]
    sc = {"s0_SELECTION_lt_0": s0["point"] < 0, "s0_SELECTION_interval_excludes_0": s0["hi"] < 0 or s0["lo"] > 0,
          "s0_abs_SELECTION_ge_abs_WITHIN": abs(s0["point"]) >= abs(w0["point"]), "s1_SELECTION_le_WITHIN": s1["point"] <= w1["point"]}
    wc = {"s0_WITHIN_lt_0": w0["point"] < 0, "s0_WITHIN_interval_excludes_0": w0["hi"] < 0 or w0["lo"] > 0, "s0_abs_WITHIN_gt_abs_SELECTION": abs(w0["point"]) > abs(s0["point"])}
    name = "SELECTION-CARRIES" if all(sc.values()) else ("WITHIN-CARRIES" if all(wc.values()) else "MIXED")
    return name, sc, wc


def r1_outcome(crit_minus_bulk_s0, diff_crit, diff_bulk):
    ok = bool(diff_crit < diff_bulk and crit_minus_bulk_s0["hi"] < 0)
    return "CRIT-CONCENTRATED" if ok else "CRIT-FLAT"


def swap_arms(eD, eF, cD, cF, seed):
    rng = np.random.default_rng(seed)
    sw = rng.random(len(eD)) < 0.5
    return np.where(sw, eF, eD), np.where(sw, eD, eF), np.where(sw, cF, cD), np.where(sw, cD, cF), int(sw.sum())


# ------------------------------------------------------------------------------------------------ the run
def jdump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def prefix_free_commit_gb():
    try:
        import subprocess
        r = subprocess.run(["powershell", "-NoProfile", "-Command", "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)"], capture_output=True, encoding="utf-8", timeout=60)
        return int(r.stdout.strip())
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--nboot", type=int, default=NB)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    nb = a.nboot
    res = {"card": "PREREG_V8F_PROOF_ADD6.md sha256 787f925f8853...", "tier": "T1-kin (logged frames; a diagnostic, never driving performance)",
           "estimator": f"paired episode-cluster bootstrap on taniteval.ci draws (pinned tree {PINNED_TREE}), B {nb}, seed {BSEED}; one training run per arm per seed: a separated interval is necessary, not sufficient",
           "floors": {"seed_floor_F_m": F_SEED, "tactical_replicate_floor_m": F_TAC, "banked_bar_2F_m": 2 * F_SEED}, "STRATEGIC": "PAUSED (PI item 25)", "thresholds": THR,
           "smoke": a.smoke, "controls": {}, "seeds": {}}
    outp = a.out_dir + "/RESULT_HIERWHERE.json"
    A = {}
    for s in SEEDS:
        dag, flat = load_arm(DIRS["DAG"].format(s=s), a.smoke), load_arm(DIRS["FLAT"].format(s=s), a.smoke)
        same = all(np.array_equal(dag[k], flat[k]) for k in ("g", "ws", "eid", "clip_index", "v0"))
        assert same, "the two arms' windows / logged future differ"
        A[s] = (dag, flat)
        res["seeds"][f"s{s}"] = {"n_windows": int(len(dag["eid"])), "n_episodes": int(len(np.unique(dag["eid"])))}
    # ---------------------------------------------------------------- control h1 (FIRST)
    pre = {}
    for s in SEEDS:
        dag, flat = A[s]
        cD, cF = comps(dag["cl"], dag["g"]), comps(flat["cl"], flat["g"])
        pre[s] = (cD, cF)
        eid = dag["eid"]
        pb = CI.paired_episode_cluster_bootstrap(cD["ade6_m"], cF["ade6_m"], eid, n_boot=nb, seed=BSEED)
        mine = boot(eid, {"d": mean_over(cD["ade6_m"] - cF["ade6_m"])}, nb)["d"]
        bk = BANKED[s]
        dev = max(abs(pb["delta"] - bk[0]), abs(pb["lo"] - bk[1]), abs(pb["hi"] - bk[2]))
        dev_mine = max(abs(mine["point"] - pb["delta"]), abs(mine["lo"] - pb["lo"]), abs(mine["hi"] - pb["hi"]))
        # the kinematics used for the modes against four_families._seq_geometry (speed)
        sp = kinematics(dag["g"])[0]
        spf = FF._seq_geometry(torch.as_tensor(dag["g"]), DT)["speed"].numpy()
        res["controls"][f"h1_s{s}"] = {"PASS": bool(dev <= 1e-4 + 1e-12), "reader": {"delta": pb["delta"], "lo": pb["lo"], "hi": pb["hi"]}, "banked": list(bk), "max_abs_dev_m": dev,
                                       "own_bootstrap_vs_taniteval_max_abs_dev_m": dev_mine, "kinematics_speed_vs_four_families_max_abs": float(np.abs(sp - spf).max())}
    res["controls"]["h1_PASS"] = bool(all(res["controls"][f"h1_s{s}"]["PASS"] for s in SEEDS) and not a.smoke)
    if a.smoke:
        res["controls"]["h1_PASS"] = None
    jdump(outp, res)
    if res["controls"]["h1_PASS"] is False:
        print("h1 FAILED; stopping before any read:", json.dumps({k: res["controls"][k] for k in res["controls"] if k.startswith("h1_s")}))
        return 2
    print("h1", json.dumps({s: res["controls"][f"h1_s{s}"]["reader"] for s in SEEDS}), f"{time.time() - t0:.0f}s", flush=True)
    # ---------------------------------------------------------------- modes
    M = {}
    for s in SEEDS:
        dag, flat = A[s]
        sp_g, dh_g = kinematics(dag["g"])
        sp_d, dh_d = kinematics(dag["cl"])
        sp_f, dh_f = kinematics(flat["cl"])
        M[s] = {"g": (sp_g, dh_g), "D": (sp_d, dh_d), "F": (sp_f, dh_f)}
    res["v0_vs_first_segment_speed_logged"] = {f"s{s}": float(np.abs(M[s]["g"][0][:, 0] - A[s][0]["v0"]).mean()) for s in SEEDS}
    res["modes"] = {}
    for s in SEEDS:
        lg, tg = classify(*M[s]["g"])
        res["modes"][f"s{s}"] = {"logged_joint_counts": {f"{LON[i]}/{LAT[j]}": int(((lg == i) & (tg == j)).sum()) for i in range(4) for j in range(3)},
                                 "logged_lon_counts": {LON[i]: int((lg == i).sum()) for i in range(4)}, "logged_lat_counts": {LAT[j]: int((tg == j).sum()) for j in range(3)},
                                 "n_SELECTION_CRITICAL": int((joint(lg, tg) != KEEP_STRAIGHT).sum()), "n_BULK": int((joint(lg, tg) == KEEP_STRAIGHT).sum())}
    # ---------------------------------------------------------------- per-seed reads
    for s in SEEDS:
        dag, flat = A[s]
        eid = dag["eid"]
        cD, cF = pre[s]
        rs = res["seeds"][f"s{s}"]

        def stage(scale, thr_tag):
            lg, tg = classify(*M[s]["g"], scale)
            ld, td = classify(*M[s]["D"], scale)
            lf, tf = classify(*M[s]["F"], scale)
            jg = joint(lg, tg)
            return {"lg": lg, "tg": tg, "ld": ld, "td": td, "lf": lf, "tf": tf, "jg": jg, "crit": jg != KEEP_STRAIGHT,
                    "cD": (joint(ld, td) == jg), "cF": (joint(lf, tf) == jg)}
        S = stage(1.0, "stated")
        # -------- R3 + h2
        r3 = {}
        h2 = {}
        for nm, key in (("ADE6", "ade6_m"), ("ADE_2_6s", "ade26_m")):
            rows_, ident = r3_rows(eid, cD[key], cF[key], S["cD"], S["cF"], nb)
            r3[nm] = rows_
            h2[nm] = ident
        rs["R3"] = r3
        res["controls"][f"h2_s{s}"] = {"PASS": bool(all(v <= 1e-9 for v in h2.values())), "abs_identity_residual_m": h2}
        print(f"s{s} R3", json.dumps({k: (v["point"], v["lo"], v["hi"]) for k, v in r3["ADE6"].items()}), f"{time.time() - t0:.0f}s", flush=True)
        jdump(outp, res)
        # -------- R1 (ADE6 diff on crit / bulk / per logged class) + family rows on the two groups
        d6 = cD["ade6_m"] - cF["ade6_m"]
        crit, bulk = S["crit"], ~S["crit"]
        stats = {"crit": mean_over(d6, crit), "bulk": mean_over(d6, bulk)}
        stats["crit_minus_bulk"] = (lambda fc, fb: (lambda sel: fc(sel) - fb(sel)))(stats["crit"], stats["bulk"])
        for i, n_ in enumerate(LON):
            stats["lon_" + n_] = mean_over(d6, S["lg"] == i)
        for j, n_ in enumerate(LAT):
            stats["lat_" + n_] = mean_over(d6, S["tg"] == j)
        groups = {"crit": crit, "bulk": bulk}
        for fam, key in FAM_ROWS:
            dd = cD[key] - cF[key]
            for g, m in groups.items():
                stats[f"fam_{key}_{g}"] = mean_over(dd, m)
        r = boot(eid, stats, nb)
        rr = {"SELECTION-CRITICAL": row(r["crit"], nums(eid, crit)), "BULK": row(r["bulk"], nums(eid, bulk)), "crit_minus_bulk": row(r["crit_minus_bulk"])}
        rr["per_logged_lon_class"] = {n_: row(r["lon_" + n_], nums(eid, S["lg"] == i)) for i, n_ in enumerate(LON)}
        rr["per_logged_lat_class"] = {n_: row(r["lat_" + n_], nums(eid, S["tg"] == j)) for j, n_ in enumerate(LAT)}
        rs["R1"] = rr
        rs["families"] = {"note": "DAG minus FLAT-DAG, lower is better, mean over windows of the group with a finite value; TACTICAL = R2; STRATEGIC PAUSED (PI item 25)",
                          "rows": {f"{fam}:{key}": {g: row(r[f"fam_{key}_{g}"], nums(eid, m, cD[key] - cF[key])) for g, m in groups.items()} for fam, key in FAM_ROWS}}
        print(f"s{s} R1", f"{time.time() - t0:.0f}s", flush=True)
        jdump(outp, res)
        # -------- R2 selection rates (1 - mean c_a) and paired difference
        errD_l, errF_l = (S["ld"] != S["lg"]).astype(float), (S["lf"] != S["lg"]).astype(float)
        errD_t, errF_t = (S["td"] != S["tg"]).astype(float), (S["tf"] != S["tg"]).astype(float)
        errD_j, errF_j = (~S["cD"]).astype(float), (~S["cF"]).astype(float)
        st = {}
        for nm, (xd, xf) in {"longitudinal": (errD_l, errF_l), "lateral": (errD_t, errF_t), "joint": (errD_j, errF_j)}.items():
            st[nm + "_DAG"], st[nm + "_FLAT"] = mean_over(xd), mean_over(xf)
            st[nm + "_diff"] = mean_over(xd - xf)
        r = boot(eid, st, nb)
        rs["R2"] = {nm: {"DAG": row(r[nm + "_DAG"]), "FLAT-DAG": row(r[nm + "_FLAT"]), "DAG_minus_FLAT": row(r[nm + "_diff"])} for nm in ("longitudinal", "lateral", "joint")}
        rs["R2"]["n_windows"], rs["R2"]["n_episodes"] = int(len(eid)), int(len(np.unique(eid)))
        # confusion of the logged joint mode against each arm (counts of window pairs), for the record
        rs["R2"]["confusion_lon"] = {"DAG": np.bincount(S["lg"] * 4 + S["ld"], minlength=16).reshape(4, 4).tolist(), "FLAT-DAG": np.bincount(S["lg"] * 4 + S["lf"], minlength=16).reshape(4, 4).tolist(), "rows": "logged " + str(LON), "cols": "arm " + str(LON)}
        jdump(outp, res)
        # -------- h3 stop label
        z = np.load(STOPDEC.format(s=s))
        assert np.array_equal(z["clip_index"], dag["clip_index"]) and np.array_equal(z["ws"], dag["ws"]) or a.smoke, "STOPDEC rows differ from the dump windows"
        if not a.smoke:
            lab = z["stop_hit"].astype(bool)
            mine_stop = S["lg"] == 0
            inter, uni = int((lab & mine_stop).sum()), int((lab | mine_stop).sum())
            h3 = {"n_label_stop": int(lab.sum()), "n_mode_stop": int(mine_stop.sum()), "intersection": inter, "union": uni, "jaccard": inter / uni, "recall_of_label": inter / max(int(lab.sum()), 1),
                  "precision_vs_label": inter / max(int(mine_stop.sum()), 1), "accuracy_all_windows": float((lab == mine_stop).mean())}
            h3["PASS"] = bool(h3["jaccard"] >= 0.90)
            h3["definition_gap_flag"] = not h3["PASS"]
            res["controls"][f"h3_s{s}"] = h3
            sub = {"label_STOP_windows": lab, "label_nonSTOP_windows": ~lab}
            stl = {k: mean_over(d6, m) for k, m in sub.items()}
            r = boot(eid, stl, nb)
            rs["R1_on_existing_stop_label"] = {k: row(r[k], nums(eid, sub[k])) for k in sub}
            rs["R1_on_existing_stop_label"]["note"] = "required when h3 jaccard < 0.90; printed always"
        # -------- h4 label swap
        h4 = {}
        for sw_seed in range(20):
            eDs, eFs, cDs, cFs, nsw = swap_arms(cD["ade6_m"], cF["ade6_m"], S["cD"], S["cF"], sw_seed)
            rows_, _ = r3_rows(eid, eDs, eFs, cDs, cFs, nb)
            h4[f"swap_seed_{sw_seed}"] = {"n_swapped": nsw, "SELECTION": row(rows_["SELECTION"]), "WITHIN": row(rows_["WITHIN"]),
                                          "both_include_0": bool(rows_["SELECTION"]["lo"] <= 0 <= rows_["SELECTION"]["hi"] and rows_["WITHIN"]["lo"] <= 0 <= rows_["WITHIN"]["hi"])}
            if sw_seed == 0:
                res["controls"][f"h4_s{s}"] = {"PASS": h4["swap_seed_0"]["both_include_0"], "swap_seed_0": h4["swap_seed_0"]}
        ex = {k: v for k, v in h4.items() if k != "swap_seed_0"}
        res["controls"][f"h4_s{s}"]["extra_seeds_informational"] = ex
        res["controls"][f"h4_s{s}"]["calibration_19_extra_swap_seeds"] = {"n_both_include_0": int(sum(v["both_include_0"] for v in ex.values())), "n": len(ex),
                                                                           "mean_point_SELECTION": float(np.mean([v["SELECTION"]["point"] for v in ex.values()])),
                                                                           "mean_point_WITHIN": float(np.mean([v["WITHIN"]["point"] for v in ex.values()]))}
        print(f"s{s} h3/h4", f"{time.time() - t0:.0f}s", flush=True)
        jdump(outp, res)
        # -------- h5 sensitivity
        h5 = {}
        for scale in (0.75, 1.25):
            S2 = stage(scale, "x")
            rows_, ident = r3_rows(eid, cD["ade6_m"], cF["ade6_m"], S2["cD"], S2["cF"], nb)
            crit2 = S2["crit"]
            r = boot(eid, {"crit": mean_over(d6, crit2), "bulk": mean_over(d6, ~crit2)}, nb)
            h5[f"x{scale}"] = {"R3_ADE6": rows_, "identity_residual": ident, "n_crit": int(crit2.sum()), "R1_crit": row(r["crit"]), "R1_bulk": row(r["bulk"]),
                               "R2_joint_error_rate": {"DAG": float(1 - S2["cD"].mean()), "FLAT-DAG": float(1 - S2["cF"].mean())}}
        sgn = lambda x: int(np.sign(x))  # noqa: E731
        base = r3["ADE6"]
        for k, v in h5.items():
            v["sign_change_vs_stated"] = {t: sgn(v["R3_ADE6"][t]["point"]) != sgn(base[t]["point"]) for t in ("SELECTION", "WITHIN")}
        rs["h5"] = h5
        jdump(outp, res)
    # ---------------------------------------------------------------- outcomes and controls
    r0, r1 = res["seeds"]["s0"]["R3"]["ADE6"], res["seeds"]["s1"]["R3"]["ADE6"]
    name, sc, wc = outcome(r0, r1)
    rr0 = res["seeds"]["s0"]["R1"]
    r1name = r1_outcome(rr0["crit_minus_bulk"], rr0["SELECTION-CRITICAL"]["point"], rr0["BULK"]["point"])
    res["OUTCOME_R3"] = {"name": name, "SELECTION-CARRIES_conditions": sc, "WITHIN-CARRIES_conditions": wc}
    res["OUTCOME_R1"] = r1name
    h5o = {}
    for k in ("x0.75", "x1.25"):
        n_, _, _ = outcome(res["seeds"]["s0"]["h5"][k]["R3_ADE6"], res["seeds"]["s1"]["h5"][k]["R3_ADE6"])
        h5o[k] = n_
    res["OUTCOME_R3_h5_sensitivity"] = h5o
    res["controls"]["h5_declared"] = True
    flags = [bool(v["PASS"]) for k, v in res["controls"].items() if isinstance(v, dict) and "PASS" in v and not k.startswith("h3")]
    flags.append(res["controls"]["h1_PASS"] is not False)
    res["controls"]["h3_agreement_flag_is_definition_gap_not_failure"] = True
    res["runtime_s"] = round(time.time() - t0, 1)
    res["controls"]["ALL_PASS"] = bool(all(flags))
    jdump(outp, res)
    write_verdict(res, a.out_dir + "/VERDICT_HIERWHERE.md")
    print(f"OUTCOME R3 {name} | R1 {r1name} | controls {'PASS' if res['controls']['ALL_PASS'] else 'FAIL'} | {res['runtime_s']}s")
    return 0 if res["controls"]["ALL_PASS"] else 2


def fmt(r, nd=4):
    return f"{r['point']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}]" + (" sep" if r["separated"] else "")


def write_verdict(res, path):
    L = []
    P = L.append
    S0, S1 = res["seeds"]["s0"], res["seeds"]["s1"]
    P("# VERDICT_HIERWHERE (PREREG_V8F_PROOF_ADD6; MEASURED, tier T1-kin on logged frames, a diagnostic, never driving performance)")
    P("")
    P(f"Estimator: {res['estimator']}. n = {S0['n_windows']} windows, {S0['n_episodes']} episodes per seed. Floors beside every term: seed floor F = {F_SEED} m (banked bar 2F = {2 * F_SEED:.4f} m), tactical replicate floor {F_TAC} m. STRATEGIC: PAUSED (PI item 25).")
    P("One training run per arm per seed, no replicate arm: a separated interval is necessary and not sufficient. Nothing here says the hierarchy beats flat. Definitions and deviations: HIER_DEFS.md.")
    P("")
    P(f"## Outcomes: R3 = **{res['OUTCOME_R3']['name']}**, R1 = **{res['OUTCOME_R1']}**; controls {'ALL PASS' if res['controls']['ALL_PASS'] else 'A CONTROL FAILED'}")
    failed = [k for k, v in res["controls"].items() if isinstance(v, dict) and v.get("PASS") is False and not k.startswith("h3")]
    P("Controls failed (the card says the read is then not valid until the control reads its known value): " + (", ".join(failed) if failed else "none") + ". h3 is a flagged definition gap, not a failure (card rule).")
    P(f"h5 sensitivity of the R3 name: x0.75 -> {res['OUTCOME_R3_h5_sensitivity']['x0.75']}, x1.25 -> {res['OUTCOME_R3_h5_sensitivity']['x1.25']}; the decided outcome is the stated-threshold one.")
    P("")
    P("## R3 exact decomposition of D = ADE6(DAG) - ADE6(FLAT-DAG), metres, point [95 % CI]")
    P("| term | n windows (ep) s0 | s0 | n windows (ep) s1 | s1 |")
    P("|---|---|---|---|---|")
    for t in ("D", "WITHIN", "SELECTION", "BOTH-MISS"):
        a, b = S0["R3"]["ADE6"][t], S1["R3"]["ADE6"][t]
        na = f"{a['n_windows']} ({a['n_episodes']})" if "n_windows" in a else f"{S0['n_windows']} ({S0['n_episodes']})"
        nbb = f"{b['n_windows']} ({b['n_episodes']})" if "n_windows" in b else f"{S1['n_windows']} ({S1['n_episodes']})"
        P(f"| {t} | {na} | {fmt(a)} | {nbb} | {fmt(b)} |")
    P("")
    P("Same three terms, 2-6 s ADE (knots t = 2.2 to 6.0 s):")
    P("| term | s0 | s1 |")
    P("|---|---|---|")
    for t in ("D", "WITHIN", "SELECTION", "BOTH-MISS"):
        P(f"| {t} | {fmt(S0['R3']['ADE_2_6s'][t])} | {fmt(S1['R3']['ADE_2_6s'][t])} |")
    P("")
    P("## R3 card bars next to measured values")
    P("| outcome | condition | measured | result |")
    P("|---|---|---|---|")
    r0, r1 = S0["R3"]["ADE6"], S1["R3"]["ADE6"]
    meas = {"s0_SELECTION_lt_0": f"SELECTION s0 {r0['SELECTION']['point']:+.4f}", "s0_SELECTION_interval_excludes_0": f"[{r0['SELECTION']['lo']:+.4f}, {r0['SELECTION']['hi']:+.4f}]",
            "s0_abs_SELECTION_ge_abs_WITHIN": f"|SEL| {abs(r0['SELECTION']['point']):.4f} vs |WITHIN| {abs(r0['WITHIN']['point']):.4f}", "s1_SELECTION_le_WITHIN": f"SEL {r1['SELECTION']['point']:+.4f} vs WITHIN {r1['WITHIN']['point']:+.4f} (s1)"}
    for k, v in res["OUTCOME_R3"]["SELECTION-CARRIES_conditions"].items():
        P(f"| SELECTION-CARRIES | {k} | {meas[k]} | {'PASS' if v else 'FAIL'} |")
    measw = {"s0_WITHIN_lt_0": f"WITHIN s0 {r0['WITHIN']['point']:+.4f}", "s0_WITHIN_interval_excludes_0": f"[{r0['WITHIN']['lo']:+.4f}, {r0['WITHIN']['hi']:+.4f}]",
             "s0_abs_WITHIN_gt_abs_SELECTION": f"|WITHIN| {abs(r0['WITHIN']['point']):.4f} vs |SEL| {abs(r0['SELECTION']['point']):.4f}"}
    for k, v in res["OUTCOME_R3"]["WITHIN-CARRIES_conditions"].items():
        P(f"| WITHIN-CARRIES | {k} | {measw[k]} | {'PASS' if v else 'FAIL'} |")
    rr = S0["R1"]
    cmb = rr["crit_minus_bulk"]
    P(f"| CRIT-CONCENTRATED | s0 crit diff more negative than bulk diff | crit {rr['SELECTION-CRITICAL']['point']:+.4f} vs bulk {rr['BULK']['point']:+.4f} | {'PASS' if rr['SELECTION-CRITICAL']['point'] < rr['BULK']['point'] else 'FAIL'} |")
    P(f"| CRIT-CONCENTRATED | paired (crit - bulk) interval excludes 0 on the negative side | {fmt(cmb)} | {'PASS' if cmb['hi'] < 0 else 'FAIL'} |")
    P("")
    P("## R1 WHERE: ADE6(DAG) - ADE6(FLAT-DAG), metres, by window class")
    for s in SEEDS:
        R = res["seeds"][f"s{s}"]["R1"]
        P(f"### seed {s}")
        P("| window class (logged) | n windows | n episodes | difference [95 % CI] |")
        P("|---|---|---|---|")
        for nm in ("SELECTION-CRITICAL", "BULK"):
            P(f"| {nm} | {R[nm]['n_windows']} | {R[nm]['n_episodes']} | {fmt(R[nm])} |")
        P(f"| crit - bulk | | | {fmt(R['crit_minus_bulk'])} |")
        for grp in ("per_logged_lon_class", "per_logged_lat_class"):
            for nm, v in R[grp].items():
                P(f"| {nm} | {v['n_windows']} | {v['n_episodes']} | {fmt(v)} |")
        P("")
    P("## R2 selection (mode) error rates, 1 - mean c_a (fraction of windows whose arm trajectory misses the logged mode)")
    P("| seed | family | DAG | FLAT-DAG | DAG - FLAT-DAG [95 % CI] |")
    P("|---|---|---|---|---|")
    for s in SEEDS:
        R = res["seeds"][f"s{s}"]["R2"]
        for nm in ("longitudinal", "lateral", "joint"):
            P(f"| s{s} | {nm} | {R[nm]['DAG']['point']:.4f} | {R[nm]['FLAT-DAG']['point']:.4f} | {fmt(R[nm]['DAG_minus_FLAT'])} |")
    P("")
    P("## Four families on the two window groups, DAG - FLAT-DAG, point [95 % CI] (LONGITUDINAL: speed, along-track; LATERAL: cross-track, heading, curvature, yaw rate; TACTICAL = R2; STRATEGIC PAUSED)")
    P("| seed | row | group | n windows (ep) | difference |")
    P("|---|---|---|---|---|")
    for s in SEEDS:
        for k, v in res["seeds"][f"s{s}"]["families"]["rows"].items():
            for g in ("crit", "bulk"):
                P(f"| s{s} | {k} | {g} | {v[g]['n_windows']} ({v[g]['n_episodes']}) | {fmt(v[g])} |")
    P("")
    P("## Controls")
    C = res["controls"]
    for s in SEEDS:
        c = C[f"h1_s{s}"]
        P(f"* h1 s{s}: {'PASS' if c['PASS'] else 'FAIL'}; reproduced {c['reader']['delta']:+.4f} [{c['reader']['lo']:+.4f}, {c['reader']['hi']:+.4f}] vs banked {c['banked'][0]:+.4f} [{c['banked'][1]:+.4f}, {c['banked'][2]:+.4f}], max |dev| {c['max_abs_dev_m']:.2e} m (bar 1e-4); own bootstrap vs taniteval.ci max |dev| {c['own_bootstrap_vs_taniteval_max_abs_dev_m']:.1e}.")
    for s in SEEDS:
        c = C[f"h2_s{s}"]
        P(f"* h2 s{s}: {'PASS' if c['PASS'] else 'FAIL'}; |WITHIN + SELECTION + BOTH-MISS - D| = {c['abs_identity_residual_m']['ADE6']:.1e} m (ADE6), {c['abs_identity_residual_m']['ADE_2_6s']:.1e} m (2-6 s); bar 1e-9.")
    for s in SEEDS:
        c = C[f"h3_s{s}"]
        P(f"* h3 s{s}: mode-function STOP vs existing stop-window label: Jaccard {c['jaccard']:.4f} (bar 0.90; {'PASS' if c['PASS'] else 'FLAGGED definition gap'}), label {c['n_label_stop']}, mode {c['n_mode_stop']}, both {c['intersection']}, recall {c['recall_of_label']:.4f}, precision {c['precision_vs_label']:.4f}, accuracy over all windows {c['accuracy_all_windows']:.4f}.")
        R = res["seeds"][f"s{s}"]["R1_on_existing_stop_label"]
        P(f"  R1 on the existing label (s{s}): STOP-label windows {fmt(R['label_STOP_windows'])} (n {R['label_STOP_windows']['n_windows']}); other windows {fmt(R['label_nonSTOP_windows'])} (n {R['label_nonSTOP_windows']['n_windows']}).")
    for s in SEEDS:
        c = C[f"h4_s{s}"]
        P(f"* h4 s{s} (swap seed 0): {'PASS' if c['PASS'] else 'FAIL'}; SELECTION {fmt(c['swap_seed_0']['SELECTION'])}, WITHIN {fmt(c['swap_seed_0']['WITHIN'])}; n swapped {c['swap_seed_0']['n_swapped']}. Informational (not part of the card control): over 19 further swap seeds both intervals include 0 in {c['calibration_19_extra_swap_seeds']['n_both_include_0']} of {c['calibration_19_extra_swap_seeds']['n']}; mean swap point SELECTION {c['calibration_19_extra_swap_seeds']['mean_point_SELECTION']:+.4f}, WITHIN {c['calibration_19_extra_swap_seeds']['mean_point_WITHIN']:+.4f}.")
    P("")
    P("## h5 sensitivity (all thresholds x0.75 and x1.25; decided at the stated thresholds)")
    P("| seed | scale | n SELECTION-CRITICAL | WITHIN | SELECTION | BOTH-MISS | crit diff | bulk diff | sign change WITHIN / SELECTION |")
    P("|---|---|---|---|---|---|---|---|---|")
    for s in SEEDS:
        base = res["seeds"][f"s{s}"]["R3"]["ADE6"]
        P(f"| s{s} | x1.0 (stated) | {res['modes'][f's{s}']['n_SELECTION_CRITICAL']} | {fmt(base['WITHIN'])} | {fmt(base['SELECTION'])} | {fmt(base['BOTH-MISS'])} | {fmt(res['seeds'][f's{s}']['R1']['SELECTION-CRITICAL'])} | {fmt(res['seeds'][f's{s}']['R1']['BULK'])} | |")
        for k, v in res["seeds"][f"s{s}"]["h5"].items():
            R3 = v["R3_ADE6"]
            sc_ = v["sign_change_vs_stated"]
            P(f"| s{s} | {k} | {v['n_crit']} | {fmt(R3['WITHIN'])} | {fmt(R3['SELECTION'])} | {fmt(R3['BOTH-MISS'])} | {fmt(v['R1_crit'])} | {fmt(v['R1_bulk'])} | {sc_['WITHIN']} / {sc_['SELECTION']} |")
    chg = []
    for s in SEEDS:
        for k, v in res["seeds"][f"s{s}"]["h5"].items():
            for t in ("WITHIN", "SELECTION", "BOTH-MISS"):
                if v["R3_ADE6"][t]["separated"] != res["seeds"][f"s{s}"]["R3"]["ADE6"][t]["separated"]:
                    chg.append(f"s{s} {k} {t} separated -> {v['R3_ADE6'][t]['separated']}")
            for t in ("WITHIN", "SELECTION"):
                if v["sign_change_vs_stated"][t]:
                    chg.append(f"s{s} {k} {t} SIGN CHANGE")
    P("")
    P("h5 named changes (sign changes of SELECTION or WITHIN, and changes of whether an interval excludes 0): " + ("; ".join(chg) if chg else "none"))
    P("")
    P("## Logged mode counts (stated thresholds)")
    for s in SEEDS:
        m = res["modes"][f"s{s}"]
        P(f"* s{s}: SELECTION-CRITICAL {m['n_SELECTION_CRITICAL']}, BULK {m['n_BULK']}; longitudinal {m['logged_lon_counts']}; lateral {m['logged_lat_counts']}.")
    P("")
    P("## Limits")
    P("A post-hoc partition of a banked comparison. Mode thresholds are a choice; h5 shows how much rides on it. The interval answers another draw of episodes, never another training run or another inference run. One training run per arm per seed. The tier is T1-kin on logged frames. Several classes hold few windows (see n); an interval from a handful of episodes is wide. Windows are partitioned post hoc by a function of the logged future and of each arm's own rollout; the partition is therefore not independent of the error it partitions (BOTH-MISS and SELECTION windows are selected on the arm outputs). Deviations R1 to R5 are in HIER_DEFS.md.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
