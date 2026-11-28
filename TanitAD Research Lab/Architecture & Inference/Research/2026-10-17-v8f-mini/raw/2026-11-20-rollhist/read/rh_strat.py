"""rh_strat.py -- STRATEGIC family of the ROLLHIST read (card section 3.2/3.3: ADD66 `pos` at 30/50/80 m at the third tick = ROLL2, the ROLL name), from the tick files of the 24 rollouts.
It IMPORTS the frozen reader of card ADD66 (code/integ10k/strat_roll_read.py, copied unedited, sha256 in MANIFEST_SHA256.txt) and re-uses its functions (label_join, compute_cells, roll_diff, boot_cols,
pair_read, run_s4, run_s8, s3_check ...); only the run names, the per-seed mode (Q5) and the pair list are new.  No rollout, CPU, numpy + torch(load of tracks).
Controls first: s4 (synthetic known values + planted fault), s8 (self-pair exactly 0, swap = minus), per file s1 s2 s6, s3 on the first RH file, s5 (T0 cells equal across forms of one checkpoint: exactly 0.0),
s7 (reference bias on the first-tick cells of RH_R0Hb20).  If a control fails, cells are not computed.
Usage: python rh_strat.py --out DIR [--seeds 0,1] [--w OUT] [--tracks ..] [--join ..] [--tree ..]   Writes DIR/READ_STRAT_ROLLHIST.json"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE + "/code/integ10k")
import rh_common as C  # noqa: E402
import strat_roll_read as S  # noqa: E402


def read(a, log=print):
    seeds = tuple(int(x) for x in a.seeds.split(","))
    S.SEEDS = seeds
    runs = list(C.ARM_FORMS_ALL)
    res = {"card": "PREREG_V8F_ROLLHIST.md sha256 781eea87 section 3.2/3.3 (STRATEGIC family via the frozen reader of card ADD66, Amendment 1)", "tier": S.TIER, "seeds": list(seeds),
           "one_seed": len(seeds) == 1, "controls": {}, "controls_all_pass": False}
    Cn = res["controls"]
    Cn["s4"] = S.run_s4()
    Cn["s8"] = S.run_s8()
    ctx = S.Ctx(a)
    ctx.load_static()
    present, missing = {}, []
    for run in runs:
        for s in seeds:
            p = S.tick_path(a.w, run, s)
            if os.path.isfile(p):
                present[f"{run}_s{s}"] = p
            else:
                missing.append(f"{run}_s{s}")
    res["missing_tick_files"] = missing
    if missing:
        res["stopped"] = f"tick files missing: {missing[:6]}{' ...' if len(missing) > 6 else ''}"
        return res
    s1 = {"max_dxy_m": 0.0, "max_dpsi_rad": 0.0}
    s2 = {"max_first_tick_anchor_dist_m": 0.0, "n_rows_anchor_not_t0": 0}
    s6 = {"pass": True, "per_file": {}}
    inputs, ref, T0 = {}, None, None
    for run in runs:
        for s in seeds:
            nm = f"{run}_s{s}"
            T = S.load_ticks(present[nm])
            inputs[nm] = S.sha256_file(present[nm])
            rbt = S.window_order(T)
            if ref is None:
                ref = {"ci": T["ci"][rbt[0]], "t0": T["t0"][rbt[0]]}
                T0 = T
            ok6, _ = S.structure(T, nm, ref)
            s6["per_file"][nm] = ok6
            s6["pass"] &= ok6
            dxy, dps = S.s1_row_check(T)
            s1["max_dxy_m"], s1["max_dpsi_rad"] = max(s1["max_dxy_m"], dxy), max(s1["max_dpsi_rad"], dps)
            st, ds = S.anchors_for(ctx, T, rbt, ticks=(0,))[0]
            s2["n_rows_anchor_not_t0"] += int((st != T["t0"][rbt[0]]).sum())
            s2["max_first_tick_anchor_dist_m"] = max(s2["max_first_tick_anchor_dist_m"], float(ds.max()))
            if nm == f"{runs[0]}_s{seeds[0]}":
                pass
    s1["pass"] = bool(s1["max_dxy_m"] <= 1e-3 and s1["max_dpsi_rad"] <= 1e-3)
    s2["pass"] = bool(s2["max_first_tick_anchor_dist_m"] <= 1e-6 and s2["n_rows_anchor_not_t0"] == 0)
    Cn["s1"], Cn["s2"], Cn["s6"] = s1, s2, s6
    Cn["s3"] = S.s3_check(ctx, T0, S.window_order(T0), tree_ov=a.tree)
    Cn["s9"] = {"pass": len(inputs) == len(runs) * len(seeds), "tick_file_sha256": inputs}
    ok_a = all(Cn[k]["pass"] for k in ("s4", "s8", "s1", "s2", "s6", "s9")) and Cn["s3"].get("pass") is True
    if not ok_a:
        res["stopped"] = "stage-A controls failed; no cell computed"
        return res
    t = time.time()
    cells, anc = {}, {}
    for run in runs:
        for s in seeds:
            T = S.load_ticks(S.tick_path(a.w, run, s))
            c, st, ds = S.compute_cells(ctx, T)
            cells[(run, s)], anc[(run, s)] = c, ds
        log(f"cells {run} done {time.time() - t:.0f} s")
    rbt = S.window_order(T0)
    t0 = T0["t0"][rbt[0]]
    ref_run = C.run_name("RH", "R0Hb20")
    eid, n_map = S.episode_map(a.w, T0, ref_run=ref_run, seed=seeds[0])
    Cn["s6"]["episode_map"] = {"dump_windows": n_map, "all_mapped": bool((eid >= 0).all()), "n_episodes": int(len(np.unique(eid)))}
    Cn["s6"]["pass"] = bool(Cn["s6"]["pass"] and (eid >= 0).all() and len(np.unique(eid)) == S.N_EP)
    # s5: the first tick (logged history only) does not depend on the readout: equal across the six forms of one checkpoint, exactly 0.0
    d5 = 0.0
    for kind in C.KINDS:
        for s in seeds:
            base = cells[(C.run_name(kind, "R0Hb20"), s)]
            for f in C.FORM_ORDER:
                for cell in S.CELLS:
                    x, y = cells[(C.run_name(kind, f), s)][cell][0], base[cell][0]
                    d5 = float("inf") if not (np.isnan(x) == np.isnan(y)).all() else max(d5, float(np.abs(x[np.isfinite(x)] - y[np.isfinite(y)]).max()) if np.isfinite(x).any() else 0.0)
    Cn["s5"] = {"pass": bool(d5 == 0.0), "largest_difference": d5, "meaning": "T0 cells of one checkpoint are equal across the six forms"}
    # s7: reference bias on the first-tick pos of RH_R0Hb20 (as ADD66)
    i1, i2 = S.neighbour_index(t0, eid, 1), S.neighbour_index(t0, eid, 2)
    s7, ok7 = {"per_arc": []}, True
    for m, A in enumerate(S.ARCS):
        vals, eids, lvl = [], [], []
        for s in seeds:
            p = cells[(ref_run, s)]["pos"][0][:, m]
            w = np.nonzero((i1 >= 0) & (i2 >= 0))[0]
            d = 0.5 * (p[w] + p[i2[w]]) - p[i1[w]]
            g = np.isfinite(d) & np.isfinite(p[w])
            vals.append(d[g]); eids.append(eid[w][g]); lvl.append(p[w][g])
        bt = S.boot(vals, eids)
        level = float(np.concatenate(lvl).mean())
        cond = bool((bt["lo"] <= 0.0 <= bt["hi"]) or abs(bt["mean"]) <= 0.02 * level)
        ok7 &= cond
        s7["per_arc"].append({"arc_m": A, "mean": bt["mean"], "lo": bt["lo"], "hi": bt["hi"], "first_tick_level": level, "ok": cond})
    s7["pass"] = bool(ok7)
    Cn["s7"] = s7
    res["controls_all_pass"] = bool(all(Cn[k]["pass"] for k in ("s1", "s2", "s4", "s5", "s6", "s8", "s9")) and Cn["s3"].get("pass") is True)
    res["controls_note"] = "s7 gates the ROLL names only (as ADD66)"
    if not res["controls_all_pass"]:
        res["stopped"] = "controls failed at stage B"
        return res
    roll = {}
    for run in runs:
        roll[run] = {}
        for tick in (1, 2):
            arrs, firsts = [], []
            for s in seeds:
                c = cells[(run, s)]
                d = S.roll_diff(c["pos"][tick], c["pos"][0], t0, eid, tick)
                arrs.append(d)
                firsts.append(np.where(np.isfinite(d), c["pos"][0], np.nan))
            bt = S.boot_cols(arrs, eid)
            lvl = [float(np.nanmean(np.concatenate([f[:, m] for f in firsts]))) for m in range(3)]
            roll[run][f"ROLL{tick}"] = [dict(b, first_tick_level_same_windows=lv, arc_m=A) for b, lv, A in zip(bt, lvl, S.ARCS)]
    res["roll"] = roll
    names = {}
    for run in runs:
        r1, r2 = [c["verdict"] for c in roll[run]["ROLL1"]], [c["verdict"] for c in roll[run]["ROLL2"]]
        names[run] = ("DEGRADES" if sum(v == "WORSE" for v in r2) >= 2 else "HOLDS" if not any(v == "WORSE" for v in r1 + r2) else "MIXED") if s7["pass"] else "NOT-NAMED (s7 failed)"
    res["roll_name"] = names
    pairs = {}
    for nm, (A, B) in C.PAIRS.items():
        pr = S.pair_read(cells, A, B, eid, ticks=(1, 2))
        pairs[nm] = {"A": A, "B": B, **pr, "family_tick2_pos": S.fam_of(pr, 2)}
    res["pairs"] = pairs
    # per-arm per-seed level of tick-2 pos at the three arcs, for the floors F_c (no pooling across seeds)
    lv = {}
    for run in runs:
        for s in seeds:
            lv[f"{run}_s{s}"] = [float(np.nanmean(cells[(run, s)]["pos"][2][:, m])) for m in range(3)]
    res["levels_tick2_pos"] = lv
    res["strategic_decision_between_routes"] = "NOT COMPUTED (n = 0, no instrument; as row 120)"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--w", default=C.W)
    ap.add_argument("--tracks", default=C.INPUTS["--tracks"])
    ap.add_argument("--join", default=C.INPUTS["--eval-join"])
    ap.add_argument("--tree", default=C.TREE)
    a = ap.parse_args()
    res = read(a, lambda s: print(s, flush=True))
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "READ_STRAT_ROLLHIST.json")
    json.dump(S.to_json(res), open(p, "w", encoding="utf-8"), indent=1)
    print("wrote", p, "controls_all_pass", res.get("controls_all_pass"), res.get("stopped", ""), flush=True)
    print("ZZ RH-STRAT " + ("DONE" if res.get("controls_all_pass") else "STOP") + " ZZ", flush=True)
    return 0 if res.get("controls_all_pass") else 1


if __name__ == "__main__":
    sys.exit(main())
