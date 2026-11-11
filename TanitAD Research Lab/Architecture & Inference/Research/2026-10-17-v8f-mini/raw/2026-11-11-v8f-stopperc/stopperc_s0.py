"""stopperc_s0.py -- PREREG_V8F_STOPPERC stage 0, reads S0-C and S0-M (theory/stopperc/PREREG_V8F_STOPPERC.md §4 and §8, frozen; sha256 in
PREREG_V8F_STOPPERC.sha256).  CPU, zero training.

S0-C, coverage ceiling: the frozen STOPORC model pass (stopbal/oracle/stoporc_read.run_arm, imported, NOT edited; its MODES hook narrowed to
  (fgt, mutant)) on BAL-NULL_s0 / s1.  Per c in C_GRID: the F-GT plan on the top-k(c) stop-point windows ranked by T's own p(STOP_POINT), T's
  own plan elsewhere, paired against OWN on the 117 stop-point windows.  Clears at c: tactical ADE separated, negative, |point| > P_FLOOR, both seeds.
S0-M (card Q7): the same pass with `limit` in the eval's `off` format (tag_paths' "limit" entry replaced by "off"; MODES (mutant,)), then
  STOPCUE's statistic on T's p(STOP_POINT) without `limit`, paired against T's causal-limit p from the STOPBAL STOPDEC npz (AMENDMENT_1_S0.md).
Usage:  python stopperc_s0.py --stage run --read s0c|s0m [--smoke N]
        python stopperc_s0.py --stage analyse --out RESULT_S0.json
The pure logic (k_of, rank_select, mix_plans, ceiling_outcome) needs no torch: test_stopperc_s0.py calls it directly.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.path.dirname(HERE)
ORACLE = os.path.join(THEORY, "stopbal", "oracle")
sys.path.insert(0, ORACLE)
import stoporc_read as SO  # noqa: E402  (imported, not copied)

SCRATCH = os.environ.get("STPERC_SCRATCH", "F:/TanitAD-scratch/stopperc_s0")
CARD = os.path.join(HERE, "PREREG_V8F_STOPPERC.md")
CARD_SHA = "70c75176e04dfb9284176aa15aa104df1a9204a43090440b3303c65e958c8877"
AMEND = os.path.join(HERE, "AMENDMENT_1_S0.md")
AMEND_SHA = open(os.path.join(HERE, "AMENDMENT_1_S0.sha256")).read().split()[0]
TAGS = ("BAL-NULL_s0", "BAL-NULL_s1")
C_GRID = (0.3, 0.5, 0.7, 1.0)
C_FAR = 0.7
P_FLOOR = 0.702720009384298                       # stopbal/read/RESULT_STOPBAL.json levers.BAL-G.seeds.s0.P_cell.floor
STOPORC_POINT = {"BAL-NULL_s0": -1.5392291009873544, "BAL-NULL_s1": -1.476844845746562}   # RESULT_STOPORC.json arms.*.primary.point
N_STOP = 117
STOP_GI = SO.STOP_GI
NB, SEED = SO.NB, SO.SEED
READS = {"s0c": ("fgt", "mutant"), "s0m": ("mutant",)}


def sha256(p):
    return SO.sha256(p)


# ------------------------------------------------------------------------------------------------------------------ pure logic (numpy)
def k_of(c, n=N_STOP):
    """k(c) = floor(c * n + 0.5) (card §8)."""
    return int(np.floor(c * n + 0.5))


def rank_select(p, sp, k):
    """top-k of the windows in mask sp by p descending, ties by window order.  -> bool mask [N] (a subset of sp with k entries)."""
    p, sp = np.asarray(p, np.float64), np.asarray(sp, bool)
    idx = np.where(sp)[0]
    assert 0 <= k <= len(idx), (k, len(idx))
    order = idx[np.lexsort((idx, -p[idx]))]
    sel = np.zeros(len(p), bool)
    sel[order[:k]] = True
    return sel


def mix_plans(plan_own, plan_fgt, sel):
    out = np.array(plan_own, copy=True)
    out[sel] = np.asarray(plan_fgt)[sel]
    return out


def clears(d, floor=P_FLOOR):
    return SO.is_gain(d, floor)


def ceiling_outcome(clear_by_c):
    """clear_by_c {c: {tag: bool}} -> CEILING-FAR iff no c <= C_FAR clears at both seeds; else CEILING-NEAR with c* the smallest c that clears."""
    both = {c: bool(v) and all(v.values()) for c, v in clear_by_c.items()}
    ok = sorted(c for c, b in both.items() if b)
    c_star = ok[0] if ok else None
    far = c_star is None or c_star > C_FAR
    return {"outcome": "CEILING-FAR" if far else "CEILING-NEAR", "c_star": c_star, "clears_both_seeds_by_c": {str(c): both[c] for c in sorted(both)}}


# ------------------------------------------------------------------------------------------------------------------ model pass (CPU)
def run_stage(read, smoke):
    SO.MODES = READS[read]
    if read == "s0m":
        import stopdec_arm as SA
        SA.install()
        sys.path.insert(0, os.path.join(THEORY, "stopdec"))
        sys.path.insert(0, os.path.join(THEORY, "speedsrc"))
        import speedsrc_read as SS
        orig = SS.tag_paths

        def tag_paths_off(tag):
            A, ck, ddir = orig(tag)
            assert A["limit"] == "causal", A["limit"]                 # AMENDMENT_1_S0.md: the dumps ran --limit causal
            A = dict(A)
            A["limit"] = "off"
            return A, ck, ddir
        SS.tag_paths = tag_paths_off
    os.makedirs(SCRATCH, exist_ok=True)
    for tag in TAGS:
        print(f"{read} arm {tag}", flush=True)
        o = SO.run_arm(tag, smoke)
        o["read"], o["modes"] = read, ",".join(SO.MODES)
        np.savez(os.path.join(SCRATCH, f"{'smoke_' if smoke else ''}{read}_{tag}.npz"), **{k: np.asarray(v) for k, v in o.items()})
        print(f"banked {read} {tag}", flush=True)


# ------------------------------------------------------------------------------------------------------------------ analysis
def setup():
    import stopdec_arm as SA
    SA.install()
    sys.path.insert(0, os.path.join(THEORY, "stopdec"))
    sys.path.insert(0, os.path.join(THEORY, "speedsrc"))
    import speedsrc_read as SS
    return SS.BL


def band_cells(BL, plan, g):
    out = {}
    for bn, (a, b) in SO.BANDS.items():
        r = BL.band_rows(np.asarray(plan, np.float64), g, a, b)
        for m in SO.METRICS:
            out[f"{bn}:{m}"] = r[m]
    return out


def analyse_s0c(tag, z, BL):
    CI = BL.CI
    ok, sp, ctl, S = SO.window_sets(z)
    eid = z["eid"]
    sd = dict(np.load(os.path.join(SO.STOPDEC_OUT, f"stopdec_{tag}.npz")))
    P = []
    ctrl = {"n_stop_point": int(sp.sum()), "n_control_set": int(ctl.sum()), "ckpt_sha256": str(z["ckpt_sha256"]), "manifest_ok": bool(z["manifest_ok"]),
            "ident_vs_own_max_abs_m": float(np.abs(z["plan_ident"].astype(np.float64) - z["plan_own"]).max()),
            "own_plan_vs_dump_ol_max_abs_m": float(z["ol_vs_own_max_abs_m"]),
            "mutant_code_equals_own_bits": bool(np.array_equal(z["code_mutant"].view(np.uint32), z["own_code"].view(np.uint32))),
            "own_goal_prob_max_abs_vs_stopdec": float(np.abs(z["own_code"][:, :SO.N_G].astype(np.float64) - sd["goal_prob"]).max()),
            "windows_equal_stopdec": bool(np.array_equal(sd["clip_index"], z["ci"]) and np.array_equal(sd["ws"], z["ws"]))}
    if ctrl["ident_vs_own_max_abs_m"] != 0.0:
        P.append("IDENT vs OWN is not exactly 0.0")
    if not ctrl["own_plan_vs_dump_ol_max_abs_m"] <= 1e-4:
        P.append(f"own plan vs dump {ctrl['own_plan_vs_dump_ol_max_abs_m']}")
    if (ctrl["n_stop_point"], ctrl["n_control_set"]) != (SO.EXPECT["n_stop"], SO.EXPECT["n_ctl"]):
        P.append("window counts differ from 117 / 2,234")
    if not ctrl["mutant_code_equals_own_bits"]:
        P.append("mutant code differs from OWN")
    if not (ctrl["windows_equal_stopdec"] and ctrl["own_goal_prob_max_abs_vs_stopdec"] <= 1e-5):
        P.append("windows or own p differ from the STOPDEC npz")
    _, pr = SO.check_forced(z["own_code"], z["code_fgt"], z["lon_al"], "fgt", sp)
    P.extend(pr)
    p_own = z["own_code"][:, STOP_GI].astype(np.float64)
    c_own = band_cells(BL, z["plan_own"], z["g"])
    idx = np.where(sp)[0]
    per_c, scan = {}, []
    for c in (0.0,) + C_GRID:
        k = k_of(c)
        sel = rank_select(p_own, sp, k)
        c_mix = band_cells(BL, mix_plans(z["plan_own"], z["plan_fgt"], sel), z["g"])
        cells = {cn: SO.paired(CI, c_mix[cn][idx], c_own[cn][idx], eid[idx]) for cn in SO.CELLS}
        prim = cells[SO.PRIMARY_CELL]
        per_c[str(c)] = {"k": k, "n_selected_own_p_ge_0p5": int((sel & (p_own >= 0.5)).sum()), "min_own_p_selected": float(p_own[sel].min()) if k else None,
                         "primary": prim, "clears": clears(prim), "cells": cells}
    z0 = per_c["0.0"]["cells"]
    ctrl["k0_all_cells_exactly_zero"] = bool(all(v.get("point") == 0.0 and not v.get("separated") for v in z0.values()))
    if not ctrl["k0_all_cells_exactly_zero"]:
        P.append("k = 0 does not read exactly 0.0 on every cell")
    ctrl["c1_point_vs_stoporc_abs"] = float(abs(per_c["1.0"]["primary"]["point"] - STOPORC_POINT[tag]))
    if not ctrl["c1_point_vs_stoporc_abs"] <= 1e-9:
        P.append(f"c = 1.0 does not reproduce STOPORC ({ctrl['c1_point_vs_stoporc_abs']})")
    for k in range(0, N_STOP + 1):                                   # descriptive scan, point only (no CI): beside, decides nothing
        sel = rank_select(p_own, sp, k)
        a = band_cells(BL, mix_plans(z["plan_own"], z["plan_fgt"], sel), z["g"])[SO.PRIMARY_CELL][idx]
        scan.append(float(np.nanmean(a - c_own[SO.PRIMARY_CELL][idx])))
    ctrl["problems"], ctrl["pass"] = P, not P
    return {"controls": ctrl, "per_c": per_c, "scan_primary_point_by_k": scan,
            "first_k_with_point_below_minus_floor": next((k for k, v in enumerate(scan) if v < -P_FLOOR), None)}


def analyse_s0m(tag, z, BL):
    sys.path.insert(0, os.path.join(THEORY, "stopcue"))
    import stopcue_read as SC
    sd = dict(np.load(os.path.join(SO.STOPDEC_OUT, f"stopdec_{tag}.npz")))
    assert np.array_equal(sd["clip_index"], z["ci"]) and np.array_equal(sd["ws"], z["ws"]), "windows differ from the STOPDEC npz"
    has = sd["has"].astype(bool)
    idx = np.where(has)[0]
    pos = sd["stop_hit"][idx].astype(bool)
    s_cau = sd["goal_prob"][idx, STOP_GI].astype(np.float64)
    s_off = z["own_code"][idx, STOP_GI].astype(np.float64)
    eid = sd["eid"][idx]

    def stat(sel):
        sel = np.asarray(sel)
        a, b = SC.auroc(s_off[sel], pos[sel]), SC.auroc(s_cau[sel], pos[sel])
        return {"auroc_limit_off": a, "auroc_limit_causal": b, "d_off_minus_causal": a - b}
    r = BL.CI.stat_cluster_bootstrap(list(range(len(idx))), eid, stat, n_boot=NB, seed=SEED)
    d = r["d_off_minus_causal"]
    return {"n_windows": int(len(idx)), "n_pos": int(pos.sum()), "n_episodes": int(len(np.unique(eid))), "stats": r,
            "separated": bool("lo" in d and (d["lo"] > 0 or d["hi"] < 0)),
            "mean_p_pos": {"off": float(s_off[pos].mean()), "causal": float(s_cau[pos].mean())},
            "fpr_at_0p5": {"off": float((s_off[~pos] >= 0.5).mean()), "causal": float((s_cau[~pos] >= 0.5).mean())},
            "hit_at_0p5": {"off": float((s_off[pos] >= 0.5).mean()), "causal": float((s_cau[pos] >= 0.5).mean())},
            "own_plan_vs_dump_ol_max_abs_m": float(z["ol_vs_own_max_abs_m"]), "ckpt_sha256": str(z["ckpt_sha256"])}


def analyse_stage(out_json):
    assert sha256(CARD) == CARD_SHA and sha256(AMEND) == AMEND_SHA, "the card or its amendment changed after freeze"
    BL = setup()
    res = {"tool": "stopperc_s0.py", "card": "theory/stopperc/PREREG_V8F_STOPPERC.md", "card_sha256": CARD_SHA, "amendment_1_sha256": AMEND_SHA,
           "tier": "T1-kin FIRST TICK, eval-141, oracle / ablation diagnostics (zero training), never driving performance",
           "estimator": f"taniteval.ci paired_episode_cluster_bootstrap (S0-C) and stat_cluster_bootstrap (S0-M), B={NB}, seed {SEED}: 'another draw of episodes' only",
           "evidence_class": "MEASURED (this file; per-arm rows banked to scratch npz, sha256 in `banked`)", "p_floor_m": P_FLOOR,
           "families": {"PRIMARY/ADE-FDE, LONGITUDINAL, LATERAL": "S0-C per_c.*.cells", "TACTICAL": SO.TACTICAL_FAMILY, "STRATEGIC": SO.STRATEGIC_FAMILY},
           "banked": {}, "s0c": {}, "s0m": {}}
    for read in ("s0c", "s0m"):
        for tag in TAGS:
            p = os.path.join(SCRATCH, f"{read}_{tag}.npz")
            if not os.path.exists(p):
                res["banked"][f"{read}_{tag}"] = "absent"
                continue
            z = dict(np.load(p, allow_pickle=False))
            res["banked"][f"{read}_{tag}"] = sha256(p)
            res[read][tag] = (analyse_s0c if read == "s0c" else analyse_s0m)(tag, z, BL)
            print(f"analysed {read} {tag}", flush=True)
    if len(res["s0c"]) == len(TAGS):
        ctl_ok = all(a["controls"]["pass"] for a in res["s0c"].values())
        oc = ceiling_outcome({c: {t: res["s0c"][t]["per_c"][str(c)]["clears"] for t in TAGS} for c in C_GRID})
        res["s0c_outcome"] = oc if ctl_ok else {"outcome": "INVALID-CONTROLS", "rule_outcome_if_controls_held": oc}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({"s0c_outcome": res.get("s0c_outcome"),
                      "s0m": {t: res["s0m"][t]["stats"]["d_off_minus_causal"] for t in res["s0m"]}}, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("run", "analyse"), required=True)
    ap.add_argument("--read", choices=tuple(READS), default="s0c")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "RESULT_S0.json"))
    a = ap.parse_args()
    if a.stage == "run":
        assert sha256(CARD) == CARD_SHA, "the card changed after freeze"
        run_stage(a.read, a.smoke)
    else:
        analyse_stage(a.out)


if __name__ == "__main__":
    main()
