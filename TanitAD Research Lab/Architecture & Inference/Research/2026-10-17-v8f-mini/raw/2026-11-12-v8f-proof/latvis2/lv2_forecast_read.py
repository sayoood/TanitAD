"""lv2_forecast_read.py -- note PREREG_V8F_PROOF_ADD38_N3: the token-level read at equal calibration.
Tier T0 (a world-model diagnostic on logged frames, never driving performance). LONGITUDINAL, LATERAL, TACTICAL: not measured. STRATEGIC: PAUSED (PI item 25).

Each state is the input of its own out-of-fold linear forecast of the true token coordinates of frame t + 2.0 s (pod side: lv2_pod_tokens.py --forecast).
C1 = E(P) - E(I)  (present state against imagined state);  C2 = E(PA) - E(PAI)  (present state + logged actions, without and with the imagined state).
The statistics, the moving cells, the bootstrap and the two-level draw are those of lv2_read.py (reused by import, on a view of the dump).

  python lv2_forecast_read.py read --dump D_s0.npz D_s1.npz --arm G --seeds 0,1 --out-dir OUT/G --ref-windows LATENTS_CTX_G_s0_R1.npz --k5-parts P0 P1
  python lv2_forecast_read.py pair --a-dir OUT/RR --b-dir OUT/G --out-dir OUT/pair
"""
import argparse
import json
import os
import sys

import numpy as np

import lv2_read as R

INPUTS = ("P", "PA", "I", "PAI", "T")
CONTRASTS = {"C1": ("P", "I"), "C2": ("PA", "PAI")}
NAMES = {"C1": ("TOKENS-FORECAST-FROM-IMAGINATION", "TOKENS-FORECAST-FLAT"), "C2": ("ROLLOUT-BEYOND-LINEAR", "ROLLOUT-WITHIN-LINEAR")}
F3_BAR = 0.02
NOTE = "PREREG_V8F_PROOF_ADD38_N3"


def view(d, base, cand):
    """The dump seen by lv2_read.seed_stats: 'present' = the base input, 'imagined' = the candidate input, 'true' = the forecast from the true future state."""
    v = dict(d)
    v["err_pres"], v["err_imag"], v["err_true"] = d["fc_" + base], d["fc_" + cand], d["fc_T"]
    return v


def above0(s):
    return bool(all(x > 0 for x in s["per_seed"]) and s["pooled_ci"][0] > 0)


def fc_controls(dumps, plan, B):
    v1 = [view(d, "P", "I") for d in dumps]
    f1 = R.control_k3(v1)
    f1.update(name="f1", rule="with the present state given as the imagined state C1 is exactly 0")
    f2 = R.control_k4(v1, plan, B)
    f2.update(name="f2", rule="known positive: E(P) minus E(T) above 0 at both seeds with the two-level interval above 0 (all cells and moving cells)")
    sts = [R.seed_stats(view(d, "P", "T")) for d in dumps]
    f2["per_seed_mov"] = [R.point(s, "D_mov") for s in sts]
    f2["pass"] = bool(f2["pass"] and all(x > 0 for x in f2["per_seed_mov"]))
    v3 = [float(np.max(d["fc_skill_heldout_permuted_I"])) for d in dumps]
    f3 = {"name": "f3", "rule": f"with the imagined states permuted across windows the held-out skill of the forecast from I is at most {F3_BAR} in every fold", "value_max_per_seed": v3,
          "pass": bool(all(x <= F3_BAR for x in v3))}
    tr = {nm: [bool(np.all(d["fc_skill_train_" + nm] > d["fc_skill_heldout_" + nm])) for d in dumps] for nm in INPUTS}
    p0 = [bool(np.all(d["fc_skill_heldout_P"] > 0)) for d in dumps]
    f4 = {"name": "f4", "rule": "for every input and fold the skill on the training windows is above the held-out skill, and the held-out skill of P is above 0", "train_gt_heldout": tr,
          "heldout_P_gt0": p0, "skill_heldout": {nm: [np.asarray(d["fc_skill_heldout_" + nm]).tolist() for d in dumps] for nm in INPUTS},
          "lam": {nm: [np.asarray(d["fc_lam_" + nm]).tolist() for d in dumps] for nm in INPUTS}, "pass": bool(all(all(v) for v in tr.values()) and all(p0))}
    return [f1, f2, f3, f4]


def read_fc(dumps, arm, seeds, B=R.B_DEFAULT, ref_windows=None, k5_parts=None):
    for d in dumps:
        if "fc_P" not in d:
            raise SystemExit("REFUSED: the dump holds no forecast arrays (pod run without --forecast)")
    n_ep = len(np.unique(dumps[0]["eid"]))
    plan = R.boot_plan(len(dumps), n_ep, B, 0)
    ctrl = fc_controls(dumps, plan, B)
    k6 = R.control_k6(dumps, ref_windows)
    k5p = R.control_k5_parts(dumps, k5_parts) if k5_parts else {"name": "k5p", "rule": "k5 in parts (note ADD38_N2)", "status": "not run", "pass": False}
    ctrl.append({"name": "f5", "rule": "k6 of the card and k5 in parts of note N2 on this dump", "k6": k6, "k5p": k5p, "pass": bool(k6["pass"] and k5p["pass"])})
    passed = bool(all(c["pass"] for c in ctrl))
    stats, names, arrs = {}, {}, {}
    for c, (b, a) in CONTRASTS.items():
        sts = [R.seed_stats(view(d, b, a)) for d in dumps]
        for s in sts[1:]:
            if not np.array_equal(s["ueid"], sts[0]["ueid"]):
                raise SystemExit("REFUSED: the seeds hold different episodes")
        summ, _ = R.summarize(sts, plan, B)
        stats[c] = summ
        ok = above0(summ["D_mov"])
        names[c] = (NAMES[c][0] if ok else NAMES[c][1]) if (passed and len(dumps) >= 2) else None
        for i, s in enumerate(sts):
            for k in ("D_mov", "D_all"):
                arrs[f"{c}_S{i}_{k}"] = s["S"][k]
        arrs["N"], arrs["ueid"] = sts[0]["N"], sts[0]["ueid"]
        arrs["mov_sha"] = np.array([R.hash_arr(s["mov"]) for s in sts])
    res = {"arm": arm, "seeds": seeds, "tier": R.TIER, "families": R.FAMILIES, "strategic": R.STRAT, "card": "PREREG_V8F_PROOF_ADD38", "note": NOTE, "B": B,
           "bootstrap": "per seed: episode-cluster; pooled: two-level (seeds, then episodes), rng seed 0", "n_windows": int(len(dumps[0]["eid"])), "n_episodes": int(n_ep),
           "contrasts": {c: f"E({b}) minus E({a})" for c, (b, a) in CONTRASTS.items()}, "controls": ctrl, "controls_pass": passed, "stats": stats, "names": names,
           "name_withheld_reason": (None if passed and len(dumps) >= 2 else ("needs both seeds" if passed else "a control failed: " + ",".join(c["name"] for c in ctrl if not c["pass"]))),
           "dumps": [{"path": d["path"], "ckpt_sha256": str(d["meta"].get("ckpt_sha256")), "sha256": (R.sha256_file(d["path"]) if os.path.exists(d["path"]) else None)} for d in dumps]}
    return res, arrs


def verdict(r):
    f = R.fmt
    L = [f"# VERDICT_LV2F_{r['arm']}: the token-level read at equal calibration (note {NOTE} to card PREREG_V8F_PROOF_ADD38)", "", f"Tier {r['tier']}. {r['families']}. {r['strategic']}.", "",
         "Each state is the input of its own out-of-fold linear forecast of the true tokens of t + 2.0 s. C1 = E(present) minus E(imagined); C2 = E(present + actions) minus E(present + actions + imagined).", "",
         "## Controls", ""]
    for c in r["controls"]:
        L.append(f"* **{c['name']}**: {'PASS' if c['pass'] else 'FAIL'} - {c['rule']}")
    L += ["", f"controls_pass: {r['controls_pass']}", "", "| statistic | per seed | pooled | pooled 95 % interval (two-level) |", "|---|---|---|---|"]
    for c in CONTRASTS:
        for k, lab in (("D_mov", "moving cells"), ("D_all", "all cells"), ("H_mov", "ceiling E(base) minus E(T), moving cells"), ("share_mov", "share of the ceiling, moving cells")):
            s = r["stats"][c][k]
            L.append(f"| {c} {lab} | {', '.join(f(x) for x in s['per_seed'])} | {f(s['pooled'])} | [{f(s['pooled_ci'][0])}, {f(s['pooled_ci'][1])}] |")
    L += ["", "C1 per row of the grid (all cells of the row):", "", "| row | per seed | pooled | pooled interval |", "|---|---|---|---|"]
    for i, nm in enumerate(R.ROW_NAMES):
        s = r["stats"]["C1"][f"D_row{i}"]
        L.append(f"| {nm} | {', '.join(f(x) for x in s['per_seed'])} | {f(s['pooled'])} | [{f(s['pooled_ci'][0])}, {f(s['pooled_ci'][1])}] |")
    f4 = [c for c in r["controls"] if c["name"] == "f4"][0]
    L += ["", "Held-out skill of each forecast (mean over folds, per seed) and the chosen penalty factors:", ""]
    for nm in INPUTS:
        L.append(f"* {nm}: skill " + ", ".join(f"{np.mean(x):.4f}" for x in f4["skill_heldout"][nm]) + "; penalty " + "; ".join(str(sorted(set(x))) for x in f4["lam"][nm]))
    L += ["", "## Names", ""]
    for c in CONTRASTS:
        L.append(f"* {c}: {r['names'][c] or 'NONE'}" + ("" if r["names"][c] else f" ({r['name_withheld_reason']})"))
    L += ["", "Limits: linear forecasts on about 1,560 training windows per fold; the input I is a function of the present state and the logged actions, so C1 includes what the actions tell and C2 is the stricter read; "
          "two seeds per arm; one corpus; 139 episodes. A separated interval over episodes is necessary, not sufficient (the bootstrap does not see training variance)."]
    return chr(10).join(L) + chr(10)


def cmd_read(a):
    dumps = [R.load_dump(p) for p in a.dump]
    seeds = [int(x) for x in a.seeds.split(",")]
    if len(seeds) != len(dumps):
        raise SystemExit("REFUSED: --seeds and --dump differ in length")
    res, arrs = read_fc(dumps, a.arm, seeds, a.B, a.ref_windows, a.k5_parts)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"RESULT_LV2F_{a.arm}.json"), "w", encoding="utf-8", newline=chr(10)) as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(a.out_dir, f"VERDICT_LV2F_{a.arm}.md"), "w", encoding="utf-8", newline=chr(10)) as fh:
        fh.write(verdict(res))
    np.savez_compressed(os.path.join(a.out_dir, f"SUFF_LV2F_{a.arm}.npz"), **arrs)
    print(f"lv2_forecast_read {a.arm}: controls_pass {res['controls_pass']}; names {res['names']}; C1_mov pooled {res['stats']['C1']['D_mov']['pooled']:+.5f}; C2_mov pooled {res['stats']['C2']['D_mov']['pooled']:+.5f}")
    return 0 if res["controls_pass"] else 2


def pair_fc(ra, rb, sa, sb, B=R.B_DEFAULT):
    if not (np.array_equal(sa["ueid"], sb["ueid"]) and np.array_equal(sa["mov_sha"], sb["mov_sha"]) and np.array_equal(sa["N"], sb["N"])):
        raise SystemExit("REFUSED: the two arms do not hold the same windows and moving cells")
    K = len(ra["seeds"])
    N = sa["N"]
    plan = R.boot_plan(K, len(N), B, 0)
    out = {}
    for c in CONTRASTS:
        for k in ("D_mov", "D_all"):
            S = np.stack([sa[f"{c}_S{i}_{k}"] - sb[f"{c}_S{i}_{k}"] for i in range(K)])
            out[f"{c}_{k}"] = {"per_seed": [float(S[i].sum() / N.sum()) for i in range(K)], "pooled": float(np.mean([S[i].sum() / N.sum() for i in range(K)])), "pooled_ci": R.ci(R.pool_draws(S, N, plan))}
    both = bool(ra["controls_pass"] and rb["controls_pass"])
    nm = None
    if both and K >= 2 and (ra["arm"], rb["arm"]) == ("RR", "G") and above0(out["C1_D_mov"]):
        nm = "RR-FORECASTS-TOKENS-BETTER"
    return {"a": ra["arm"], "b": rb["arm"], "note": NOTE, "tier": R.TIER, "controls_pass": {"a": ra["controls_pass"], "b": rb["controls_pass"]}, "stats": out, "name": nm,
            "name_rule": "C1 of RR minus C1 of G on the moving cells above 0 at both seeds with the paired two-level interval above 0; else nothing is named"}


def cmd_pair(a):
    def load(d):
        js = [f for f in os.listdir(d) if f.startswith("RESULT_LV2F_")][0]
        with open(os.path.join(d, js), encoding="utf-8") as fh:
            r = json.load(fh)
        z = np.load(os.path.join(d, f"SUFF_LV2F_{r['arm']}.npz"), allow_pickle=True)
        return r, {k: z[k] for k in z.files}
    (ra, sa), (rb, sb) = load(a.a_dir), load(a.b_dir)
    res = pair_fc(ra, rb, sa, sb, a.B)
    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"PAIR_LV2F_{res['a']}_minus_{res['b']}"
    with open(os.path.join(a.out_dir, tag + ".json"), "w", encoding="utf-8", newline=chr(10)) as fh:
        json.dump(res, fh, indent=1)
    L = [f"# {tag} (note {NOTE})", "", f"Tier {res['tier']}.", "", f"Controls pass: {res['controls_pass']}", "", "| statistic (A minus B) | per seed | pooled | pooled paired two-level interval |", "|---|---|---|---|"]
    for k, s in res["stats"].items():
        L.append(f"| {k} | {', '.join(R.fmt(x) for x in s['per_seed'])} | {R.fmt(s['pooled'])} | [{R.fmt(s['pooled_ci'][0])}, {R.fmt(s['pooled_ci'][1])}] |")
    L += ["", f"Name: {res['name'] or 'NONE'} ({res['name_rule']})"]
    with open(os.path.join(a.out_dir, tag + ".md"), "w", encoding="utf-8", newline=chr(10)) as fh:
        fh.write(chr(10).join(L) + chr(10))
    print(f"lv2_forecast_read pair {tag}: name {res['name']}; C1_mov {res['stats']['C1_D_mov']['pooled']:+.5f} {res['stats']['C1_D_mov']['pooled_ci']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--dump", nargs="+", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--seeds", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--ref-windows", default=None)
    r.add_argument("--k5-parts", nargs="+", default=None)
    r.add_argument("--B", type=int, default=R.B_DEFAULT)
    p = sub.add_parser("pair")
    p.add_argument("--a-dir", required=True)
    p.add_argument("--b-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--B", type=int, default=R.B_DEFAULT)
    a = ap.parse_args(argv)
    return cmd_read(a) if a.cmd == "read" else cmd_pair(a)


if __name__ == "__main__":
    sys.exit(main())
