"""P1 return map, per SPEC_P1.md (sha256 in SPEC_P1.sha256). CPU only, no model. Reads the stage-2/3/4 T2 records.

Run:  CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe p1_return_map.py
Writes P1_RESULT.json, P1_RETURN_MAP.md next to itself. The schema is read as t2v8f_lead.rows_for_records / t2v8f_score read it
(rec scene / f0 / track [N,4] (x,y,yaw,v at 10 Hz) / plans [3,30,2] / extra[k].step / extra[k].v0_ms; scene_<idx>.npz gt_T, gt_ts_us);
ego_from_rig is COPIED from t2v8f_lead.py (backward-difference speed), not imported, so nothing under t2v8f/code is touched.
"""
import glob
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ART = "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f"
ROOTS = {2: f"{ART}/stage2/pull_box2_run2/pull2/rollouts", 3: "C:/Users/Admin/t2v8f_pulls/stage3/pull_box2/pull3/rollouts",
         4: "C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/rollouts"}
ARMS = {"P10": (2, 3), "C0_s0": (2, 3), "C0_s1": (2, 3), "C0C_s0": (4,), "C0C_s1": (4,)}
EXC = json.load(open(f"{ART}/stage4/EXCLUSIONS4.json"))["excluded"]
B_BOOT, SEED = 2000, 0
EDGES = np.arange(0.0, 12.5, 0.5)
NB = len(EDGES) - 1
BAND = (3.0, 8.0)
DT_TICK = 20          # rig ticks between planner ticks (2.0 s)


def ego_v(T, tus):
    """COPY of t2v8f_lead.ego_from_rig 'v': backward difference |p[i]-p[i-1]|/0.1, row 0 repeats row 1."""
    T = np.asarray(T, np.float64)
    x, y = T[:, 0, 3], T[:, 1, 3]
    d = np.hypot(np.diff(x), np.diff(y)) / 0.1
    return np.concatenate([[d[0]], d])


def cell_ok(d):
    if os.path.isfile(f"{d}/FAIL") or not os.path.isfile(f"{d}/DONE"):
        return False
    t = open(f"{d}/DONE", encoding="utf-8").read()
    a, b = t.split("records=")[1].split()[0].split("/")
    n = sum(1 for _ in open(f"{d}/records.jsonl", encoding="utf-8"))
    return int(a) == int(b) == n


def plan_speed_at_2s(plan):
    p = np.asarray(plan, np.float64)
    return float(np.hypot(*(p[11] - p[7]))) / 0.8


_HUMAN = {}


def human_v(cell_dir, idx):
    if idx not in _HUMAN:
        z = np.load(f"{cell_dir}/scene_{idx}.npz")
        _HUMAN[idx] = ego_v(z["gt_T"], z["gt_ts_us"])
    return _HUMAN[idx]


def load_arm(arm):
    rows, notes = [], {"cells": 0, "excluded_cells": 0, "not_done": [], "records": 0, "no_scene_npz": []}
    for st in ARMS[arm]:
        for d in sorted(glob.glob(f"{ROOTS[st]}/*/objects/T2/{arm}")):
            d = d.replace("\\", "/")
            s8 = d.split("/")[-4]
            if s8 in EXC:
                notes["excluded_cells"] += 1
                continue
            if not cell_ok(d):
                notes["not_done"].append(d)
                continue
            notes["cells"] += 1
            for line in open(f"{d}/records.jsonl", encoding="utf-8"):
                r = json.loads(line)
                tr = np.asarray(r["track"], np.float64)
                P = r["plans"]
                try:
                    hv = human_v(d, int(r["scene"]))
                except Exception:
                    notes["no_scene_npz"].append(d)
                    hv = None
                notes["records"] += 1
                for k, e in enumerate(r["extra"][:3]):
                    i = 2 * int(e["step"])
                    if i + DT_TICK >= len(tr):
                        continue
                    f = int(r["f0"]) + DT_TICK * k
                    hk = (hv[f], hv[f + DT_TICK]) if (hv is not None and f + DT_TICK < len(hv)) else (np.nan, np.nan)
                    rows.append((int(r["scene"]), int(r["f0"]), k, float(e["v0_ms"]), plan_speed_at_2s(P[k]), float(tr[i + DT_TICK][3]), hk[0], hk[1]))
    a = np.array(rows, np.float64) if rows else np.zeros((0, 8))
    return {"scene": a[:, 0].astype(int), "f0": a[:, 1].astype(int), "k": a[:, 2].astype(int), "v": a[:, 3], "gp": a[:, 4], "ge": a[:, 5], "hv": a[:, 6], "hn": a[:, 7]}, notes


# ----------------------------------------------------------------------------------------------- statistics
def bin_of(v):
    b = np.floor(v / 0.5).astype(int)
    ok = (v >= 0) & (v < 12.0)
    return np.where(ok, b, -1)


def scene_stats(D, mask, y, sc_index, n_sc):
    """per-scene sufficient statistics of d = y - v: bin sums/counts [n_sc, NB], band sums/counts [n_sc], linear-fit moments over [0,12) [n_sc, 5]."""
    v = D["v"][mask]
    yy = y[mask]
    si = sc_index[mask]
    b = bin_of(v)
    ok = b >= 0
    d = yy - v
    S = np.zeros((n_sc, NB)); N = np.zeros((n_sc, NB))
    np.add.at(S, (si[ok], b[ok]), d[ok]); np.add.at(N, (si[ok], b[ok]), 1.0)
    inb = (v >= BAND[0]) & (v < BAND[1])
    Sb = np.bincount(si[inb], weights=d[inb], minlength=n_sc); Nb = np.bincount(si[inb], minlength=n_sc).astype(float)
    M = np.zeros((n_sc, 5))
    for j, w in enumerate((np.ones_like(v), v, v * v, yy, v * yy)):
        M[:, j] = np.bincount(si[ok], weights=w[ok], minlength=n_sc)
    return {"S": S, "N": N, "Sb": Sb, "Nb": Nb, "M": M}


def local_moments(D, mask, y, sc_index, n_sc, lo, hi):
    v = D["v"][mask]; yy = y[mask]; si = sc_index[mask]
    ok = (v >= lo) & (v < hi)
    M = np.zeros((n_sc, 5))
    for j, w in enumerate((np.ones_like(v), v, v * v, yy, v * yy)):
        M[:, j] = np.bincount(si[ok], weights=w[ok], minlength=n_sc)
    return M


def fit_from_M(M):
    """M summed over scenes [..., 5] = n, sv, svv, sg, svg  -> a, b, fixed point v*, (nan when not defined)"""
    n, sv, svv, sg, svg = (M[..., j] for j in range(5))
    den = n * svv - sv * sv
    with np.errstate(all="ignore"):
        b = (n * svg - sv * sg) / den
        a = (sg - b * sv) / n
        vstar = np.where(b < 1.0, a / (1.0 - b), np.nan)
    return a, b, vstar


def pct(x, q=(2.5, 97.5)):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, q[0])), float(np.percentile(x, q[1]))] if len(x) else [float("nan")] * 2


def analyse(D, y_key, mask=None, label=""):
    """one arm, one map (y_key 'gp' or 'ge' or the human 'hn'): point + bootstrap table."""
    mask = np.ones(len(D["v"]), bool) if mask is None else mask
    y = D[y_key]
    scenes = np.unique(D["scene"][mask])
    n_sc = len(scenes)
    sc_index = np.searchsorted(scenes, D["scene"])
    st = scene_stats(D, mask, y, sc_index, n_sc)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_sc, (B_BOOT, n_sc))
    out = {"n_windows": int(mask.sum()), "n_scenes": int(n_sc), "n_ge12": int((D["v"][mask] >= 12.0).sum())}
    # per bin
    Sb, Nb = st["S"].sum(0), st["N"].sum(0)
    bS, bN = st["S"][idx].sum(1), st["N"][idx].sum(1)
    with np.errstate(all="ignore"):
        mean = Sb / Nb
        bm = bS / bN
    out["bins"] = [{"lo": float(EDGES[i]), "hi": float(EDGES[i + 1]), "n": int(Nb[i]), "mean_g_minus_v": None if Nb[i] == 0 else float(mean[i]),
                    "ci": None if Nb[i] < 5 else pct(bm[:, i])} for i in range(NB)]
    # band
    band = st["Sb"].sum() / max(st["Nb"].sum(), 1.0)
    bb = st["Sb"][idx].sum(1) / np.maximum(st["Nb"][idx].sum(1), 1.0)
    inb = [b for b in out["bins"] if b["lo"] >= BAND[0] and b["hi"] <= BAND[1] and b["n"] > 0]
    out["band"] = {"n_windows": int(st["Nb"].sum()), "mean_g_minus_v": float(band), "ci": pct(bb), "bins_occupied": len(inb),
                   "bins_negative": int(sum(1 for b in inb if b["mean_g_minus_v"] < 0)), "bins_ci_upper_lt0": int(sum(1 for b in inb if b["ci"] and b["ci"][1] < 0))}
    # fit
    a, b, vs = fit_from_M(st["M"].sum(0))
    ba, bb_, bvs = fit_from_M(st["M"][idx].sum(1))
    out["fit_global"] = {"a": float(a), "slope_b": float(b), "v_star": None if not np.isfinite(vs) else float(vs), "slope_ci": pct(bb_), "v_star_ci": pct(bvs),
                         "v_star_defined_frac": float(np.isfinite(bvs).mean())}
    # local slope near v*
    if np.isfinite(vs):
        lo, hi = vs - 1.5, vs + 1.5
        Ml = local_moments(D, mask, y, sc_index, n_sc, lo, hi)
        la, lb, lvs = fit_from_M(Ml.sum(0))
        _, lbb, lvss = fit_from_M(Ml[idx].sum(1))
        out["fit_local"] = {"window": [float(lo), float(hi)], "n": int(Ml[:, 0].sum()), "slope": float(lb), "slope_ci": pct(lbb), "v_star_local": None if not np.isfinite(lvs) else float(lvs)}
    else:
        out["fit_local"] = None
    # bin crossing
    cross = None
    prev = None
    for i in range(NB):
        if Nb[i] >= 10:
            if prev is not None and mean[prev] > 0 and mean[i] <= 0:
                x0, x1 = (EDGES[prev] + EDGES[prev + 1]) / 2, (EDGES[i] + EDGES[i + 1]) / 2
                cross = float(x0 + (x1 - x0) * mean[prev] / (mean[prev] - mean[i]))
                break
            prev = i
    out["bin_crossing_v_star"] = cross
    out["_boot_idx"] = (scenes, idx)
    out["_stats"] = st
    return out


def human_median(D, mask, scenes_idx=None):
    """matched-window human speed: median over windows (all ticks), and over windows whose logged speed is in the band."""
    m = mask & np.isfinite(D["hv"])
    res = {"n": int(m.sum()), "median": float(np.median(D["hv"][m])), "q25": float(np.percentile(D["hv"][m], 25)), "q75": float(np.percentile(D["hv"][m], 75)),
           "mean": float(D["hv"][m].mean())}
    mb = m & (D["hv"] >= BAND[0]) & (D["hv"] < BAND[1])
    res["band_median"] = float(np.median(D["hv"][mb])) if mb.any() else None
    res["n_in_band_by_human"] = int(mb.sum())
    res["frac_human_below_0p5"] = float((D["hv"][m] < 0.5).mean())
    return res


def vstar_minus_human_ci(D, a_res, y_key, mask):
    """bootstrap of (v* - human median) with the SAME scene resamples as analyse()."""
    scenes, idx = a_res["_boot_idx"]
    st = a_res["_stats"]
    _, _, bvs = fit_from_M(st["M"][idx].sum(1))
    sc_index = np.searchsorted(scenes, D["scene"])
    m = mask & np.isfinite(D["hv"])
    per = [D["hv"][m & (sc_index == i)] for i in range(len(scenes))]
    med = np.array([np.median(np.concatenate([per[j] for j in row])) if sum(len(per[j]) for j in row) else np.nan for row in idx])
    diff = bvs - med
    return {"median_h_ci": pct(med), "v_star_minus_human_median_ci": pct(diff), "frac_replicates_vstar_defined": float(np.isfinite(bvs).mean())}


def band_contrast(DA, DB, y_key, maskA=None, maskB=None):
    """paired (same keys) contrast B_A - B_B of the band mean of (g - v); scene-cluster bootstrap on the shared scenes."""
    ka = {(s, f, k): i for i, (s, f, k) in enumerate(zip(DA["scene"], DA["f0"], DA["k"]))}
    kb = {(s, f, k): i for i, (s, f, k) in enumerate(zip(DB["scene"], DB["f0"], DB["k"]))}
    keys = sorted(set(ka) & set(kb))
    ia = np.array([ka[x] for x in keys]); ib = np.array([kb[x] for x in keys])
    sc = np.array([x[0] for x in keys])
    scenes = np.unique(sc)
    si = np.searchsorted(scenes, sc)
    n_sc = len(scenes)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_sc, (B_BOOT, n_sc))
    out = {"n_shared_windows": len(keys), "n_shared_scenes": int(n_sc)}
    res = {}
    for nm, DD, ii in (("A", DA, ia), ("B", DB, ib)):
        v = DD["v"][ii]; d = DD[y_key][ii] - v
        inb = (v >= BAND[0]) & (v < BAND[1])
        res[nm] = (np.bincount(si[inb], weights=d[inb], minlength=n_sc), np.bincount(si[inb], minlength=n_sc).astype(float))
    pa = res["A"][0].sum() / max(res["A"][1].sum(), 1.0); pb = res["B"][0].sum() / max(res["B"][1].sum(), 1.0)
    ba = res["A"][0][idx].sum(1) / np.maximum(res["A"][1][idx].sum(1), 1.0)
    bb = res["B"][0][idx].sum(1) / np.maximum(res["B"][1][idx].sum(1), 1.0)
    out.update({"band_A": float(pa), "band_B": float(pb), "n_band_A": int(res["A"][1].sum()), "n_band_B": int(res["B"][1].sum()), "diff_A_minus_B": float(pa - pb), "diff_ci": pct(ba - bb)})
    return out


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def main():
    data, notes = {}, {}
    for arm in ARMS:
        data[arm], notes[arm] = load_arm(arm)
        print(arm, notes[arm]["cells"], "cells", notes[arm]["records"], "records", len(data[arm]["v"]), "tick-rows", "not_done", len(notes[arm]["not_done"]), flush=True)
    R = {"spec_sha256": open(f"{HERE}/SPEC_P1.sha256").read().split()[0], "bootstrap": {"B": B_BOOT, "seed": SEED, "unit": "scene"}, "arms": {}, "contrasts": {}, "notes": notes}
    # human map on the C0C windows (control) -- uses hv -> hn
    for arm, D in data.items():
        entry = {"n_tick_rows": int(len(D["v"]))}
        full = np.ones(len(D["v"]), bool)
        for ykey, nm in (("gp", "g_plan"), ("ge", "g_exec")):
            a = analyse(D, ykey, full)
            a.update(vstar_vs_human=vstar_minus_human_ci(D, a, ykey, full) if a["fit_global"]["v_star"] is not None else None)
            entry[nm] = a
            for sub, m in (("tick0", D["k"] == 0), ("ticks12", D["k"] >= 1)):
                if m.sum() > 20:
                    s = analyse(D, ykey, m)
                    entry[f"{nm}__{sub}"] = {"band": s["band"], "fit_global": s["fit_global"], "n_windows": s["n_windows"]}
        # human map on the same windows (control): y = logged speed at +2 s, v = logged speed now
        H = dict(D); H["v"] = D["hv"]; H["gp"] = D["hn"]
        mh = np.isfinite(H["v"]) & np.isfinite(H["gp"])
        entry["human_return_map"] = analyse(H, "gp", mh)
        entry["human_speed"] = human_median(D, full)
        R["arms"][arm] = entry
    # contrasts: C0C vs P10 (paired on shared (scene, f0, tick))
    for s in ("s0", "s1"):
        for ykey, nm in (("gp", "g_plan"), ("ge", "g_exec")):
            R["contrasts"][f"C0C_{s}_minus_P10__{nm}"] = band_contrast(data[f"C0C_{s}"], data["P10"], ykey)
            R["contrasts"][f"C0_{s}_minus_P10__{nm}"] = band_contrast(data[f"C0_{s}"], data["P10"], ykey)
    # verdicts
    V = {}
    for ykey, nm in (("gp", "g_plan"), ("ge", "g_exec")):
        legs = {}
        As = [R["arms"][f"C0C_{s}"][nm]["band"] for s in ("s0", "s1")]
        holds = all(b["ci"][1] < 0 for b in As)
        fails = all(b["mean_g_minus_v"] >= 0 for b in As)
        legs["A_contraction"] = "HOLDS" if holds else "FAILS" if fails else "INCONCLUSIVE"
        Bs = []
        for s in ("s0", "s1"):
            e = R["arms"][f"C0C_{s}"][nm]
            vs = e["fit_global"]["v_star"]
            vh = e.get("vstar_vs_human")
            Bs.append(vs is not None and BAND[0] <= vs < BAND[1] and vh is not None and vh["v_star_minus_human_median_ci"][1] < 0)
        legs["B_fixed_point_below_human"] = "HOLDS" if all(Bs) else "NOT_MET"
        Cs = [R["contrasts"][f"C0C_{s}_minus_P10__{nm}"]["diff_ci"][1] < 0 for s in ("s0", "s1")]
        legs["C_P10_nearer_identity"] = "HOLDS" if all(Cs) else "NOT_MET"
        V[nm] = {"legs": legs, "verdict": "HOLDS" if all(v == "HOLDS" for v in legs.values()) else "FAILS" if legs["A_contraction"] == "FAILS" else "PARTIAL"}
    R["verdict"] = V
    json.dump(strip(R), open(f"{HERE}/P1_RESULT.json", "w"), indent=1)
    print(json.dumps(V, indent=1))
    return R


if __name__ == "__main__":
    main()
