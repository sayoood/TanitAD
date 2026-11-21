"""diag_read.py -- DSCALE10K DATA-HURTS diagnostics (per-goal-class F1 by provenance; TAC-LAT / TAC-LON per country and platform).
Rules: DIAG_ADDENDUM.md (frozen before any number; sha256 in DIAG_ADDENDUM.md.sha256).  CPU, zero training.  MEASURED (ours; banked T1 tick dumps).
Estimator: paired episode-cluster bootstrap, B 2,000, seed 0, same resampler as taniteval.ci (ci._draws / ci.episode_index of the read tree).
Usage: python diag_read.py --part f1|strata|both [--n-boot 2000] [--out RESULT_DIAG.json]"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.dirname(HERE)
R2 = DS + "/read2"
for p in (R2, R2 + "/theory/obd/read", R2 + "/theory/otband/read"):
    if p not in sys.path:
        sys.path.insert(0, p)
import ds_lib as L  # noqa: E402

INP = "F:/TanitAD-scratch/strat1/read/inputs"
ST = {0: DS + "/read", 1: DS + "/read2"}               # seed -> folder holding t1/ (dumps, ticks)
ARMS = {0: ("B_s0", "U_s0", "B_rep"), 1: ("B_s1", "U_s1")}
GOAL22 = ("FOLLOW_LANE", "TURN_L", "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R", "YIELD", "STOP_POINT", "SPEED_BAND", "CORRIDOR_OFFSET", "EVADE_IN_CORRIDOR", "OVERTAKE_VEHICLE",
          "MERGE", "GAP_TARGET", "REACT_ON_ONCOMING", "TAKE_EXIT_L", "TAKE_EXIT_R", "TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED", "TRAFFIC_LIGHT_REACT_YELLOW",
          "TRAFFIC_LIGHT_REACT_GREEN", "LANE_CHANGE_L", "LANE_CHANGE_R")
GEO = ("FOLLOW_LANE", "TURN_L", "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R", "STOP_POINT", "SPEED_BAND")      # GOAL_PROV == geometry (build_v9_labels.py); every other class is vlm or sam3|vlm
THR = 0.5


def tick_path(seed, arm, st):
    return f"{ST[seed]}/t1/ticks/ticks_DS-{arm}-{st}_m10_causal_heads.npz"


def dump_dir(seed, arm, st):
    return f"{ST[seed]}/t1/dump_DS-{arm}-{st}_m10_causal_heads"


def sha(p):
    return L.sha256_file(p)


# ------------------------------------------------------------------------------------------------------------------------------ pure logic
def f1_from(tp, fp, fn):
    """vectorised F1: 2tp/(2tp+fp+fn); 0 when tp == 0 and fp+fn > 0; nan when the three are all 0"""
    den = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), np.nan)
    return f


def group_macro(f, idx):
    v = f[..., idx]
    return np.nanmean(v, axis=-1) if len(idx) else np.full(v.shape[:-1], np.nan)


def counts(prob, gy, gw):
    """prob [N,22]; gy, gw [N] uint bitmasks -> tp, fp, fn [N,22] (windows scored only where the label bit exists)"""
    bits = np.arange(22)
    y = ((gy[:, None].astype(np.int64) >> bits) & 1).astype(bool)
    w = ((gw[:, None].astype(np.int64) >> bits) & 1).astype(bool)
    p = prob > THR
    return (w & p & y).astype(np.float64), (w & p & ~y).astype(np.float64), (w & ~p & y).astype(np.float64)


def boot_f1(CI, arms, pairs, eid, groups, nb):
    """arms {name: (tp, fp, fn)}; pairs [(a, b)]; groups {name: class idx array}.  The SAME episode draws for every pair.  -> {pair: {key: row}}; key = class name or group name."""
    uniq, idx_by_ep = CI.episode_index(eid)
    keys = list(groups)
    D = {pr: {k: np.empty(nb) for k in keys} for pr in pairs}
    for i, sel in enumerate(CI._draws(uniq, idx_by_ep, nb, 0)):
        F = {}
        for a, (tp, fp, fn) in arms.items():
            F[a] = f1_from(tp[sel].sum(0), fp[sel].sum(0), fn[sel].sum(0))
        for pr in pairs:
            a, b = pr
            for k, idx in groups.items():
                D[pr][k][i] = group_macro(F[a], idx) - group_macro(F[b], idx)
    out = {}
    for pr in pairs:
        a, b = pr
        Fa, Fb = f1_from(*[x.sum(0) for x in arms[a]]), f1_from(*[x.sum(0) for x in arms[b]])
        out[pr] = {}
        for k, idx in groups.items():
            d = D[pr][k]
            fin = d[np.isfinite(d)]
            pt = float(group_macro(Fa, idx) - group_macro(Fb, idx))
            if fin.size == 0 or not np.isfinite(pt):
                out[pr][k] = {"delta": None, "undefined": True}
                continue
            lo, hi = (float(x) for x in np.percentile(fin, [2.5, 97.5]))
            out[pr][k] = {"a": float(group_macro(Fa, idx)), "b": float(group_macro(Fb, idx)), "delta": pt, "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0),
                          "negative_separated": bool(pt < 0 and hi < 0), "n_nonfinite_draws": int(d.size - fin.size)}
    return out


def label(d0, d1):
    """addendum item 6.  d0, d1: {'VLM': row, 'GEO': row} at seed 0 / seed 1 (stage B)"""
    loses = {g: bool(d0[g].get("negative_separated") and d1[g].get("negative_separated")) for g in ("VLM", "GEO")}
    geo_any_negsep = bool(d0["GEO"].get("negative_separated") or d1["GEO"].get("negative_separated"))
    if loses["VLM"] and not geo_any_negsep:
        return "DILUTION-CONSISTENT", loses
    if (loses["VLM"] and loses["GEO"]) or (not loses["VLM"] and not loses["GEO"]):
        return "NOT-DILUTION", loses
    return "MIXED", loses


# ------------------------------------------------------------------------------------------------------------------------------ loaders
def load_ticks0(seed, arm, st, ref=None):
    z = np.load(tick_path(seed, arm, st))
    m = z["tick"] == 0
    r = {k: z[k][m] for k in ("t_goal_prob", "win", "ci", "t0", "age_t")}
    assert (r["win"] == np.arange(len(r["win"]))).all() and (r["age_t"] == 0).all(), (seed, arm, st)
    r["clip_names"] = [str(x) for x in z["clip_names"]]
    if ref is not None:
        assert np.array_equal(r["ci"], ref["ci"]) and np.array_equal(r["t0"], ref["t0"]) and r["clip_names"] == ref["clip_names"], f"windows differ: {seed} {arm} {st}"
    return r


def load_eid(d):
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    assert len(fs) == L.EVAL_N_EPISODES
    zs = [np.load(f) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("eid", "clip_index", "ws")}


def part_f1(CI, nb, rep):
    J = np.load(INP + "/join_eval141.npz")
    rep["inputs_sha256"] = {"join_eval141.npz": sha(INP + "/join_eval141.npz")}
    ref = load_ticks0(0, "B_s0", "B")
    assert [str(x) for x in J["clip"]] == ref["clip_names"], "join clip order differs from the ticks clip_names"
    dd = load_eid(dump_dir(0, "B_s0", "B"))
    assert np.array_equal(dd["clip_index"], ref["ci"]) and np.array_equal(dd["ws"], ref["t0"])
    eid = dd["eid"]
    row = J["off"][ref["ci"]] + ref["t0"]
    gy, gw = J["goal_y"][row], J["goal_w"][row]
    bits = np.arange(22)
    wmat = ((gw[:, None].astype(np.int64) >> bits) & 1).astype(bool)
    ymat = ((gy[:, None].astype(np.int64) >> bits) & 1).astype(bool)
    n_scored, n_pos = wmat.sum(0), (wmat & ymat).sum(0)
    support = {GOAL22[c]: {"n_scored_windows": int(n_scored[c]), "n_pos_windows": int(n_pos[c]), "n_pos_episodes": int(len(np.unique(eid[wmat[:, c] & ymat[:, c]]))),
                           "source": "geometry" if GOAL22[c] in GEO else "vlm"} for c in range(22)}
    in_macro = [c for c in range(22) if n_pos[c] > 0]
    geo_idx = np.array([c for c in in_macro if GOAL22[c] in GEO], int)
    vlm_idx = np.array([c for c in in_macro if GOAL22[c] not in GEO], int)
    groups = {"VLM": vlm_idx, "GEO": geo_idx}
    groups.update({GOAL22[c]: np.array([c]) for c in in_macro})
    out = {"n_windows": int(len(eid)), "n_episodes": int(len(np.unique(eid))), "threshold": THR, "support": support,
           "excluded_n0": [{"class": GOAL22[c], "n_pos": 0, "n_scored_windows": int(n_scored[c])} for c in range(22) if n_pos[c] == 0],
           "macro_geometry": [GOAL22[c] for c in geo_idx], "macro_vlm": [GOAL22[c] for c in vlm_idx],
           "scoring": "window scored for class c where goal_w bit c is set at join row off[ci]+t0; pred = t_goal_prob at tick 0 > 0.5"}
    arms = {}
    for stg in ("B", "A"):
        for seed, names in ARMS.items():
            for a in names:
                if stg == "A" and a == "B_rep" and False:
                    continue
                t = load_ticks0(seed, a, stg, ref)
                arms[(a, stg)] = counts(t["t_goal_prob"], gy, gw)
                rep["inputs_sha256"][os.path.basename(tick_path(seed, a, stg))] = sha(tick_path(seed, a, stg))
    # controls
    ora = counts(np.where(ymat, 1.0, 0.0), gy, gw)
    fo = f1_from(*[x.sum(0) for x in ora])
    out["K_oracle"] = {"every_supported_class_f1_is_1": bool(all(fo[c] == 1.0 for c in in_macro)), "min_f1": float(min(fo[c] for c in in_macro))}
    tp, fp, fn = arms[("U_s0", "B")]
    out["rows"] = {}
    pairs = {}
    for stg in ("B", "A"):
        pr = [(f"U_s0.{stg}", f"B_s0.{stg}"), (f"U_s1.{stg}", f"B_s1.{stg}"), (f"B_rep.{stg}", f"B_s0.{stg}")]
        ar = {f"{a}.{s}": v for (a, s), v in arms.items() if s == stg}
        t0 = time.time()
        res = boot_f1(CI, ar, pr, eid, groups, nb)
        res[(f"U_s0.{stg}", f"U_s0.{stg}")] = None
        selfc = boot_f1(CI, {f"U_s0.{stg}": arms[("U_s0", stg)]}, [(f"U_s0.{stg}", f"U_s0.{stg}")], eid, groups, nb)
        pairs[stg] = {"U-B_s0": res[pr[0]], "U-B_s1": res[pr[1]], "floor_Brep-B_s0": res[pr[2]], "self_U_s0": selfc[(f"U_s0.{stg}", f"U_s0.{stg}")]}
        print(f"f1 stage {stg} done {time.time() - t0:.0f}s", flush=True)
    out["pairs"] = pairs
    out["K_self"] = {"stage_B_all_keys_delta_0_ci_0": bool(all(v.get("delta") == 0.0 and v.get("lo") == 0.0 and v.get("hi") == 0.0 for v in pairs["B"]["self_U_s0"].values()))}
    lab, loses = label({g: pairs["B"]["U-B_s0"][g] for g in ("VLM", "GEO")}, {g: pairs["B"]["U-B_s1"][g] for g in ("VLM", "GEO")})
    out["label"] = lab
    out["groups_lose_at_both_seeds"] = loses
    out["per_class_stage_B_f1"] = {GOAL22[c]: {f"{a}": float(f1_from(*[x.sum(0) for x in arms[(a, 'B')]])[c]) for a in ("B_s0", "U_s0", "B_rep", "B_s1", "U_s1")} for c in in_macro}
    rep["f1"] = out
    return eid, ref, row


# ------------------------------------------------------------------------------------------------------------------------------ strata
def part_strata(CI, nb, rep, eid_ref, ref):
    import ds_read as R
    import obd_geom as G
    plat = np.load(INP + "/strat_platform_by_clip.npz")
    lab = np.load(INP + "/strat_labels_table.npz")
    rep["inputs_sha256"].update({"strat_platform_by_clip.npz": sha(INP + "/strat_platform_by_clip.npz"), "strat_labels_table.npz": sha(INP + "/strat_labels_table.npz")})
    geom = G.build()
    dg = R.load_dump_n(dump_dir(0, "B_s0", "B"), L.EVAL_N_EPISODES)
    J = geom.join(dg)
    glat, glon = geom.labels(dg["g"])
    cells = {}
    for seed, names in ARMS.items():
        for a in names:
            if a == "B_rep":
                continue
            cells[a] = R.load_arm_cells(geom, dg, J, glat, glon, dump_dir(seed, a, "B"), L.EVAL_N_EPISODES)
            print("loaded", a, flush=True)
    eid = dg["eid"]
    assert np.array_equal(eid, eid_ref)
    banked = json.load(open(R2 + "/out/RESULT_DSCALE10K_B2.json", encoding="utf-8"))["pairs"]
    kre = {}
    for s in (0, 1):
        u, b = f"U_s{s}", f"B_s{s}"
        for c in ("TAC-LAT", "TAC-LON"):
            r = CI.paired_episode_cluster_bootstrap(cells[u][c], cells[b][c], eid, n_boot=nb, seed=0)
            bk = banked[f"U-B_s{s}.B"][c]
            kre[f"{u}-{b}.{c}"] = {"recomputed_delta": float(cells[u][c].mean() - cells[b][c].mean()), "banked_delta": bk["delta"], "recomputed_lo_hi": [r["lo"], r["hi"]], "banked_lo_hi": [bk["lo"], bk["hi"]],
                                   "equal": bool(abs(float(cells[u][c].mean() - cells[b][c].mean()) - bk["delta"]) < 1e-12 and abs(r["lo"] - bk["lo"]) < 1e-9 and abs(r["hi"] - bk["hi"]) < 1e-9)}
    # strata
    ci = ref["ci"]
    el_off, el_n = lab["eval_off"], lab["eval_n"]
    ok = lab["eval_country_valid"].astype(bool) & (lab["eval_flag"] == 0)
    vc = [str(x) for x in lab["vocab_country"]]
    assert [str(x) for x in lab["eval_clip"]] == ref["clip_names"] and [str(x) for x in plat["eval_clip"]] == ref["clip_names"]
    clip_country = np.full(141, -1)
    for c in range(141):
        m = ok[el_off[c]:el_off[c] + el_n[c]]
        ids = lab["eval_country_id"][el_off[c]:el_off[c] + el_n[c]][m].astype(int)
        if len(ids):
            clip_country[c] = np.bincount(ids, minlength=len(vc)).argmax()
    ctry = np.array([vc[clip_country[c]] if clip_country[c] >= 0 else "(unlabelled)" for c in range(141)])[ci]
    vp = [str(x) for x in plat["vocab_platform"]]
    pl = np.array([vp[int(i)] for i in plat["eval_platform_id"]])[ci]
    rep["strata_layer"] = ("country = clip-level modal of the STRAT labels table eval_country_id (valid, flag 0) [F:/TanitAD-scratch/strat1/read/inputs/strat_labels_table.npz, the STRAT1 read input; a per-frame label release layer]; "
                           "platform = strat_platform_by_clip.npz eval_platform_id [same folder]. Corpus clip metadata for the 10k set (country, platform_class) lives in datascale/ds_select.py order_all_eligible.parquet.")
    res = {"K_RE": kre, "strata": {}}
    for name, arr in (("country", ctry), ("platform", pl)):
        rows = []
        for v in sorted(set(arr)):
            m = arr == v
            row = {"stratum": v, "n_windows": int(m.sum()), "n_episodes": int(len(np.unique(eid[m])))}
            for c in ("TAC-LAT", "TAC-LON"):
                for s in (0, 1):
                    u, b = f"U_s{s}", f"B_s{s}"
                    if row["n_episodes"] >= 2:
                        r = CI.paired_episode_cluster_bootstrap(cells[u][c][m], cells[b][c][m], eid[m], n_boot=nb, seed=0)
                        pt = float(cells[u][c][m].mean() - cells[b][c][m].mean())
                        row[f"{c}_s{s}"] = {"delta": pt, "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"])}
                    else:
                        row[f"{c}_s{s}"] = None
            flag, same = [], []
            for c in ("TAC-LAT", "TAC-LON"):
                a0, a1 = row[f"{c}_s0"], row[f"{c}_s1"]
                if a0 and a1:
                    if a0["separated"] and a1["separated"] and a0["delta"] > 0 and a1["delta"] > 0 and row["n_episodes"] >= 5:
                        flag.append(c)
                    if a0["separated"] and a1["separated"] and a0["delta"] < 0 and a1["delta"] < 0:
                        same.append(c)
            row["FLAG_opposite_direction_both_seeds"] = flag
            row["same_direction_both_seeds_descriptive"] = same
            rows.append(row)
        res["strata"][name] = rows
    res["n_strata_read"] = {k: len(v) for k, v in res["strata"].items()}
    res["n_cells_read"] = sum(len(v) for v in res["strata"].values()) * 4
    rep["strata"] = res


def md(rep):
    o = ["# DSCALE10K DATA-HURTS diagnostics (RESULT_DIAG)", "",
         f"Evidence class {rep['evidence_class']}. Tier {rep['tier']}. Estimator {rep['estimator']}. Rules frozen in `DIAG_ADDENDUM.md` (sha256 `{rep['addendum_sha256'][:12]}`) before any number.", ""]
    f = rep.get("f1")
    if f:
        o += ["## 1. Per-goal-class F1 by source of supervision (stage B, deciding)", "",
              f"**Label: {f['label']}** (groups that lose at both seeds: {f['groups_lose_at_both_seeds']}). n = {f['n_windows']} windows, {f['n_episodes']} episodes; a class is scored only on windows where its label exists.", "",
              f"Controls: oracle reads F1 1.0 on every supported class = {f['K_oracle']['every_supported_class_f1_is_1']}; U vs itself reads delta 0 CI [0,0] = {f['K_self']['stage_B_all_keys_delta_0_ci_0']}.", "",
              "| group | seed 0 D (U-B) [95% CI] | seed 1 D (U-B) [95% CI] | floor |B_rep - B_s0| (beside) | classes |", "|---|---|---|---|---|"]
        pb = f["pairs"]["B"]

        def cell(r):
            return "n/a" if not r or r.get("delta") is None else f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]{' sep' if r['separated'] else ''}"
        for g, cl in (("VLM", f["macro_vlm"]), ("GEO", f["macro_geometry"])):
            fl = pb["floor_Brep-B_s0"][g]
            o.append(f"| {g} macro-F1 | {cell(pb['U-B_s0'][g])} | {cell(pb['U-B_s1'][g])} | {abs(fl['delta']):.4f} | {', '.join(cl)} |")
        o += ["", "Per class (stage B; F1 of each arm, D = U - B with CI; support = positive windows / scored windows / positive episodes):", "",
              "| class | source | pos / scored / ep | B_s0 | U_s0 | D s0 | B_s1 | U_s1 | D s1 |", "|---|---|---|---|---|---|---|---|---|"]
        for c, v in f["per_class_stage_B_f1"].items():
            sp = f["support"][c]
            o.append(f"| {c} | {sp['source']} | {sp['n_pos_windows']} / {sp['n_scored_windows']} / {sp['n_pos_episodes']} | {v['B_s0']:.3f} | {v['U_s0']:.3f} | {cell(pb['U-B_s0'][c])} | {v['B_s1']:.3f} | {v['U_s1']:.3f} | {cell(pb['U-B_s1'][c])} |")
        o += ["", "Excluded from the macro (n_pos = 0): " + ", ".join(f"{e['class']} (n_pos 0, scored {e['n_scored_windows']})" for e in f["excluded_n0"]), "",
              "Stage A, beside (T mechanism stage): VLM D s0 " + cell(f["pairs"]["A"]["U-B_s0"]["VLM"]) + "; s1 " + cell(f["pairs"]["A"]["U-B_s1"]["VLM"]) + "; GEO D s0 " + cell(f["pairs"]["A"]["U-B_s0"]["GEO"]) + "; s1 " + cell(f["pairs"]["A"]["U-B_s1"]["GEO"]) + ".", ""]
    s = rep.get("strata")
    if s:
        o += ["## 2. TAC-LAT / TAC-LON accuracy, U - B, per country and platform (stage B, descriptive only)", "", f"Layer: {rep['strata_layer']}", "",
              f"K-RE (recomputed against the banked read): all equal = {all(v['equal'] for v in s['K_RE'].values())}. Strata read: {s['n_strata_read']}, cells {s['n_cells_read']} (multiplicity; no stratum is promoted).", ""]
        for nm, rows in s["strata"].items():
            o += [f"### {nm}", "", "| stratum | n win | n ep | LAT s0 | LAT s1 | LON s0 | LON s1 | flag |", "|---|---|---|---|---|---|---|---|"]

            def c2(r):
                return "n/a" if not r else f"{r['delta']:+.3f}{'*' if r['separated'] else ''}"
            for r in rows:
                fl = ("FLAG " + ",".join(r["FLAG_opposite_direction_both_seeds"])) if r["FLAG_opposite_direction_both_seeds"] else ("same-dir " + ",".join(r["same_direction_both_seeds_descriptive"]) if r["same_direction_both_seeds_descriptive"] else "")
                o.append(f"| {r['stratum']} | {r['n_windows']} | {r['n_episodes']} | {c2(r['TAC-LAT_s0'])} | {c2(r['TAC-LAT_s1'])} | {c2(r['TAC-LON_s0'])} | {c2(r['TAC-LON_s1'])} | {fl} |")
            o.append("")
        o += ["`*` = the paired episode-cluster CI excludes 0 inside the stratum. Strata under 5 episodes cannot be flagged."]
    return "\n".join(o) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="both", choices=("f1", "strata", "both"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=HERE + "/RESULT_DIAG.json")
    a = ap.parse_args()
    CI = L.load_ci()
    rep = {"tool": "diag_read.py", "evidence_class": "MEASURED (ours; banked T1 tick dumps, zero training, CPU)",
           "tier": "T1-kin tick 0 (heads) / T1-kin trajectory cells (TAC-LAT, TAC-LON), 141 episodes / 2,399 windows",
           "estimator": f"paired episode-cluster bootstrap, B={a.n_boot}, seed 0, taniteval.ci resampler ({CI.SOURCE}); 'another draw of EPISODES' only",
           "addendum_sha256": sha(HERE + "/DIAG_ADDENDUM.md"), "inputs_sha256": {}}
    t0 = time.time()
    eid, ref, _ = part_f1(CI, a.n_boot, rep)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    if a.part in ("strata", "both"):
        part_strata(CI, a.n_boot, rep, eid, ref)
    rep["wall_s"] = time.time() - t0
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    open(a.out.replace(".json", ".md"), "w", encoding="utf-8").write(md(rep))
    print("WROTE", a.out, f"{rep['wall_s']:.0f}s")
    print("ZZ DIAG DONE ZZ")


if __name__ == "__main__":
    main()
