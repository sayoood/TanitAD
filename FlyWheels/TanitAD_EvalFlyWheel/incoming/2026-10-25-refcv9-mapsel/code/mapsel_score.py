"""mapsel_score.py -- refcv9 MAP-SELECT (PREREG_MAPSEL.md): the MECHANICAL reading.  CPU only.  Reads the roll's trace (mapsel_roll.py) + its rasters, the banked per-frame battery (GT, v0,
compliance, episodes, banked prior/stage), L1's un-hooked BASE-os bank (identity control).

    python mapsel_score.py --seed 0 --trace F:/refcv9_mapsel/trace_s0.npz --rast F:/refcv9_mapsel/rast --out result/s0 [--human-json result/K_HUMAN.json]
    python mapsel_score.py --seed 0 --trace <smoke trace> --rast <smoke rast> --out <dir> --controls-only      # hard controls on whatever windows the trace holds (the GPU chain's smoke gate)

TIER: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance); the GT-* arms are a T0-ORACLE BOUND (privileged map).  ESTIMATOR: paired episode-cluster bootstrap
(pair_read_r8.paired_cell -> taniteval.ci, B 2000, seed 0) = 'another draw of EPISODES' only: NOT another training run (one checkpoint), NOT another inference run (one seed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
import mapsel_data as D  # noqa: E402
import mapsel_lib as L  # noqa: E402

torch.set_num_threads(4)
PREREG, SEAL = ROOT / "PREREG_MAPSEL.md", ROOT / "PREREG_MAPSEL.sha256"
PREREG2, SEAL2 = ROOT / "PREREG_MAPSEL_AMENDMENT1.md", ROOT / "PREREG_MAPSEL_AMENDMENT1.sha256"
PREREG3, SEAL3 = ROOT / "PREREG_MAPSEL_AMENDMENT2.md", ROOT / "PREREG_MAPSEL_AMENDMENT2.sha256"
L1_BANK = "C:/Users/Admin/refcv8_pf/refcv9_l1/bank"
N_FAN, N_EXPECT = 149, 4754
# ---- constants frozen by PREREG_MAPSEL.md + PREREG_MAPSEL_AMENDMENT1.md
ARMS = (("BASE", None, 0), ("GT-T1", "gt", 1), ("GT-T12", "gt", 2), ("GT-T123", "gt", 3), ("OWN-T1", "own", 1), ("OWN-T12", "own", 2), ("OWN-T123", "own", 3),
        ("DER-T1", "der", 1), ("DER-T12", "der", 2), ("DER-T123", "der", 3),
        ("OWN-CAL-T1", "cal", 1), ("DER-CAL-T1", "dercal", 1))      # CAL = the own-map P(drivable) score thresholded at tau (fixed a priori from the HUMAN paths only; sec. 3.2 of the prereg)
OWN_ARMS = ("OWN-T1", "OWN-CAL-T1", "OWN-T12", "OWN-T123")
HEADLINE_ARMS = ("OWN-T1", "OWN-CAL-T1")
DER_OF = {"OWN-T1": "DER-T1", "OWN-CAL-T1": "DER-CAL-T1", "OWN-T12": "DER-T12", "OWN-T123": "DER-T123"}
HUMAN_PASS_FRAC = 0.99          # tau = the 99th-percentile (method='higher') HUMAN value of max P^(sidewalk): >= 99 % of the human paths have max P^(sidewalk) <= tau (AMENDMENT 1)
STOP_GUARD_ROWS = ("LON_along_mae_m", "LON_speed_mae_mps", "LON_along_abs_final_m", "LON_speed_within_0p5", "LON_accel_mae_mps2", "along_signed_final_m@GTcruise",
                   "speed_err_signed_mps@GTcruise", "false_decel", "missed_decel")
HUMAN_MAX_OFF = 0.02            # K_HUMAN
LIVE_MARGIN = 0.02              # K_LIVE: own-vs-GT drivable agreement must exceed the deranged-donor agreement by this
MAP_METRICS = (("off_drivable", "off-drivable rate (any FP5 point on a seen SIDEWALK cell, t 0.5-6 s)"), ("edge_contact", "edge-contact rate (0.3 m margin)"),
               ("lane_cross", "lane-marking crossing rate (centre line, outside lane-change windows)"))


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jdump(obj, p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


def load_trace(path, require_final=True):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    if require_final and meta.get("final") is not True:
        raise SystemExit(f"[mapsel_score] {path} is not a FINAL trace")
    d = {k: z[k] for k in z.files if k != "meta_json"}
    d["idx"] = {(int(e), int(w)): j for j, (e, w) in enumerate(zip(d["e_i"], d["ws"]))}
    d["meta"], d["n"] = meta, len(d["ws"])
    return d


def donor_table(tr):
    """window j -> donor window index: the CYCLIC NEXT episode (sorted clip index among the episodes in the trace), position-matched (min(p, n_donor - 1)). -1 iff one episode only."""
    eps = sorted(set(tr["e_i"].tolist()))
    by = {e: sorted(np.nonzero(tr["e_i"] == e)[0].tolist(), key=lambda j: int(tr["ws"][j])) for e in eps}
    don = np.full(tr["n"], -1, np.int64)
    if len(eps) < 2:
        return don
    for k, e in enumerate(eps):
        d = eps[(k + 1) % len(eps)]
        for p, j in enumerate(by[e]):
            don[j] = by[d][min(p, len(by[d]) - 1)]
    return don


def rast_file(rdir, e, w):
    return Path(rdir) / f"e{int(e):03d}_w{int(w):04d}.npz"


def load_pair(args):
    rdir, e, w, de, dw = args
    o = np.load(rast_file(rdir, e, w))
    own = {k: o[k] for k in o.files}
    der = None
    if de is not None:
        d = np.load(rast_file(rdir, de, dw))
        der = {k: d[k] for k in ("pdrv", "psw", "code")}
    return own, der


def pass1(tr, rdir, has_gt, hum=None, limit=None):
    """Per-candidate flags under GT / OWN / DER masks for every window: bool [n, 149] x (off, edge, lane) x 3 sources, + own continuous features + liveness agreement."""
    n = tr["n"] if not limit else min(limit, tr["n"])
    don = donor_table(tr)
    F = {f"{s}_{k}": np.zeros((n, N_FAN), bool) for s in ("gt", "own", "der") for k in ("off", "edge", "lane")}
    F.update(pswmax=np.zeros((n, N_FAN), np.float32), pswmean=np.zeros((n, N_FAN), np.float16), pemax=np.zeros((n, N_FAN), np.float16), dpswmax=np.zeros((n, N_FAN), np.float32))
    hum_pmin = np.full(n, np.nan)       # the human paths' max own P^(sidewalk)
    acc_own, acc_der = np.full(n, np.nan), np.full(n, np.nan)
    n_samples = np.zeros((n, N_FAN), np.int16)
    t0 = time.time()
    CH = 32
    with ThreadPoolExecutor(3) as ex:
        for a in range(0, n, CH):
            js = list(range(a, min(n, a + CH)))
            jobs = [(rdir, tr["e_i"][j], tr["ws"][j], (tr["e_i"][don[j]] if don[j] >= 0 else None), (tr["ws"][don[j]] if don[j] >= 0 else None)) for j in js]
            for j, (own, der) in zip(js, ex.map(load_pair, jobs)):
                g = L.Geom(torch.from_numpy(tr["fan"][j]))
                n_samples[j] = np.minimum(g.n_samples.numpy(), 32000)
                off, edge, lane = g.flags(L.masks_own(own["psw"], own["code"]))
                F["own_off"][j], F["own_edge"][j], F["own_lane"][j] = off.numpy(), edge.numpy(), lane.numpy()
                pmax_, pmean, pemax = g.own_cont(own["psw"], own["pedge"])
                F["pswmax"][j], F["pswmean"][j], F["pemax"][j] = pmax_.numpy(), pmean.numpy(), pemax.numpy()
                if hum is not None and hum[1][j]:
                    hum_pmin[j] = float(L.Geom(torch.from_numpy(hum[0][j:j + 1])).own_cont(own["psw"], own["pedge"])[0][0])
                if der is not None:
                    F["dpswmax"][j] = g.own_cont(der["psw"], np.zeros_like(der["psw"]))[0].numpy()
                    off, edge, lane = g.flags(L.masks_own(der["psw"], der["code"]))
                    F["der_off"][j], F["der_edge"][j], F["der_lane"][j] = off.numpy(), edge.numpy(), lane.numpy()
                if has_gt[j] and "gt" in own:
                    gt = own["gt"]
                    off, edge, lane = g.flags(L.masks_gt(gt))
                    F["gt_off"][j], F["gt_edge"][j], F["gt_lane"][j] = off.numpy(), edge.numpy(), lane.numpy()
                    seen = gt != 255
                    if seen.any():
                        gdrv = np.isin(gt, L.DRV_CODES)
                        acc_own[j] = float(((own["pdrv"] >= L.PDRV_MIN_U8) == gdrv)[seen].mean())
                        if der is not None:
                            acc_der[j] = float(((der["pdrv"] >= L.PDRV_MIN_U8) == gdrv)[seen].mean())
            if (a // CH) % 10 == 0:
                print(f"[mapsel_score] pass1 {js[-1] + 1}/{n} el={time.time() - t0:.0f}s", flush=True)
    return F, acc_own, acc_der, don[:n], n_samples, hum_pmin


def picks_for(tr, elig, F, lc, n):
    """-> {arm: pick [n]}, {arm: fallbacks [n, 3]}, {arm: kept-set size [n]}."""
    out, fbs, kept = {}, {}, {}
    sv3 = tr["sel_score_v3"]
    for name, src, tier in ARMS:
        p = np.zeros(n, np.int64)
        fb = np.zeros((n, 3), np.int8)
        kp = np.zeros(n, np.int32)
        for j in range(n):
            if src is None:
                off = edge = lane = np.zeros(N_FAN, bool)
            else:
                off, edge, lane = F[f"{src}_off"][j], F[f"{src}_edge"][j], F[f"{src}_lane"][j]
            p[j], fb_, kp[j] = L.tier_pick(sv3[j], elig[j], off, edge, lane, tier, bool(lc[j]))
            fb[j] = fb_
        out[name], fbs[name], kept[name] = p, fb, kp
    return out, fbs, kept


def cell_json(c):
    return {k: (float(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v)) for k, v in c.items()
            if k in ("delta", "lo", "hi", "n_windows", "n_episodes", "mean_new", "mean_ref", "separated", "status", "reason")}


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rast", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--human-json", default=None)
    ap.add_argument("--controls-only", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="DRY-RUN ONLY: windows missing from the trace are dropped from every population instead of refusing (a scored reading never uses it)")
    ap.add_argument("--l1-bank", default=L1_BANK)
    a = ap.parse_args()
    t_start = time.time()
    seal = SEAL.read_text(encoding="utf-8").split()[0].strip() if SEAL.exists() else ""
    got = sha256_file(PREREG) if PREREG.exists() else ""
    seal2 = SEAL2.read_text(encoding="utf-8").split()[0].strip() if SEAL2.exists() else ""
    got2 = sha256_file(PREREG2) if PREREG2.exists() else ""
    seal3 = SEAL3.read_text(encoding="utf-8").split()[0].strip() if SEAL3.exists() else ""
    got3 = sha256_file(PREREG3) if PREREG3.exists() else ""
    seals_ok = bool(len(seal) == 64 and seal == got and len(seal2) == 64 and seal2 == got2 and len(seal3) == 64 and seal3 == got3)
    if not a.controls_only and not a.allow_partial and not seals_ok:
        raise SystemExit("[mapsel_score] PREREG_MAPSEL.md / PREREG_MAPSEL_AMENDMENT1.md is missing / does not match its seal: refusing a scored reading")
    import l2_common as C
    import l2_oracle_s2_rows as S2R
    import l2_score as S
    st = C.stack()
    PR, RS = st["PR"], st["RS"]
    tr = load_trace(a.trace, require_final=not a.controls_only)
    n = tr["n"]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    ctl = {}
    # ---------------------------------------------------------------- hard controls that need no sets
    m = tr["meta"]
    cnt = m["counters"]
    ctl["K_G_fields"] = {"n_windows": n, "present_counts": m["present_counts"], "fan_n_seen": cnt["fan_n_seen"], "batch_gt1": cnt["batch_gt1"], "rule_seen": cnt["rule_seen"],
                         "logit_shapes": cnt["logit_shapes"], "decide_mismatch_cells": cnt["decide_mismatch_cells"], "decide_checked": cnt["decide_checked"], "n_stack": m.get("n_stack"),
                         "pass": bool(all(m["present_counts"].get(k, 0) == n for k in ("fan", "sel_score", "sel_score_v3", "prior_path")) and cnt["fan_n_seen"] == [N_FAN] and cnt["batch_gt1"] == 0
                                      and cnt["rule_seen"] == ["prior_corrected"] and cnt["logit_shapes"] == [[1, 8, L.H, L.W]] and cnt["decide_mismatch_cells"] == 0 and cnt["decide_checked"] >= 1
                                      and m.get("n_stack") == 3)}
    emitted = tr["fan"][np.arange(n), tr["sel"]]
    ctl["K_B_emitted_equals_fan_at_sel_idx"] = {"max_abs": float(np.abs(emitted - tr["plan8"]).max()), "pass": bool(np.array_equal(emitted, tr["plan8"]))}
    elig = L.eligible(tr["sel_score_v3"], tr.get("reach_keep"), tr.get("ceil_keep"), tr.get("emit_keep"))
    mx = np.where(elig, tr["sel_score_v3"], -np.inf).max(1)
    ctl["K_C_sel_idx_is_argmax_over_e9_eligible"] = {"share": float((tr["sel_score_v3"][np.arange(n), tr["sel"]] == mx).mean()),
                                                      "pass": bool((tr["sel_score_v3"][np.arange(n), tr["sel"]] == mx).all()),
                                                      "emitted_eligible_share": float(elig[np.arange(n), tr["sel"]].mean())}
    base = S.load_l1_base(a.l1_bank, a.seed)
    if base is not None and not base.get("incomplete"):
        ks = [k for k in tr["idx"] if k in base["idx"]]
        d = np.array([np.abs(tr["plan8"][tr["idx"][k]] - base["plan8"][base["idx"][k]]).max() for k in ks])
        ctl["K_A_roll_equals_L1_BASEos"] = {"n_compared": len(ks), "n_bit_equal": int((d == 0).sum()), "max_abs": float(d.max()) if len(d) else None, "pass": bool(len(d) > 0 and (d == 0).all())}
    else:
        ctl["K_A_roll_equals_L1_BASEos"] = {"pass": False, "note": "L1 BASE-os bank missing"}
    has_gt = np.asarray(tr["gt_label"], bool)
    miss = [(int(e), int(w)) for e, w in zip(tr["e_i"], tr["ws"]) if not rast_file(a.rast, e, w).exists()]
    ctl["K_RAST_all_windows_have_a_raster"] = {"missing": len(miss), "n": n, "pass": len(miss) == 0}
    if miss:
        jdump({"controls": ctl}, Path(a.out) / "MAPSEL_CONTROLS.json")
        raise SystemExit(f"[mapsel_score] {len(miss)} windows have no raster file")
    keys = list(zip(tr["e_i"].tolist(), tr["ws"].tolist()))
    lc, lcchk = D.lc_windows(keys)
    ctl["K_LC_JOIN"] = {**{k: v for k, v in lcchk.items() if k != "per_clip_agree"}, "n_lc_windows": int(lc.sum()), "pass": bool(lcchk["ok"])}
    lc_active = bool(lcchk["ok"])
    if not lc_active:
        lc = np.zeros(n, bool)
    # ---------------------------------------------------------------- pass 1: per-candidate flags
    limit = None
    hp, hvalid = D.human_plans(keys)
    F, acc_own, acc_der, don, n_samples, hum_pmin = pass1(tr, a.rast, has_gt, (hp, hvalid), limit)
    hv = hum_pmin[np.isfinite(hum_pmin)]
    if len(hv) == 0:
        raise SystemExit("[mapsel_score] no human path could be scored on an own raster: tau undefined")
    tau = float(np.quantile(hv, HUMAN_PASS_FRAC, method="higher"))
    zero = np.zeros((n, N_FAN), bool)
    F["cal_off"], F["cal_edge"], F["cal_lane"] = F["pswmax"] > tau, zero, zero
    F["dercal_off"], F["dercal_edge"], F["dercal_lane"] = F["dpswmax"] > tau, zero, zero
    hard_pass = float((hv < L.PSW_MIN_U8 / 255.0).mean())
    ctl["TAU_CAL"] = {"tau": tau, "rule": "AMENDMENT 1: tau = quantile_0.99(method=higher) of the HUMAN paths' MAX own-map P^(sidewalk) over the in-range FP5 points (S6-valid human windows with an own raster); a candidate is rejected iff its max P^(sidewalk) > tau; "
                      "privileged GT trajectories, NO outcome metric; calibrated ON THE EVAL human paths (a train-side calibration is not available in this roll)",
                      "n_human_paths": int(len(hv)), "human_pass_frac_at_tau": float((hv <= tau).mean()), "human_pass_frac_at_hard_rule_0p502": hard_pass,
                      "hard_rule_plan_level_false_alarm_rate_on_human_paths": 1.0 - hard_pass,
                      "human_max_psw_quantiles_0.5_0.9_0.95_0.99": [float(np.quantile(hv, q)) for q in (0.5, 0.9, 0.95, 0.99)], "pass": bool(np.isfinite(tau) and (hv <= tau).mean() >= HUMAN_PASS_FRAC)}
    own_ok, der_ok = np.nanmean(acc_own) if np.isfinite(acc_own).any() else float("nan"), np.nanmean(acc_der) if np.isfinite(acc_der).any() else float("nan")
    ctl["K_LIVE_own_map_beats_deranged_donor"] = {"acc_own_vs_GT_drivable": float(own_ok), "acc_deranged_donor_vs_GT_drivable": float(der_ok), "n_gt_windows": int(np.isfinite(acc_own).sum()),
                                                  "margin_required": LIVE_MARGIN, "pass": bool(np.isfinite(own_ok) and np.isfinite(der_ok) and own_ok > der_ok + LIVE_MARGIN)}
    ctl["K_DERANGED_donor_is_another_episode"] = {"pass": bool(len(set(tr["e_i"].tolist())) < 2 or all(tr["e_i"][don[j]] != tr["e_i"][j] for j in range(n) if don[j] >= 0)), "n_no_donor": int((don < 0).sum())}
    picks, fbs, kept = picks_for(tr, elig, F, lc, n)
    ctl["K_BASE_arm_equals_emitted_pick"] = {"share": float((picks["BASE"] == tr["sel"]).mean()), "pass": bool((picks["BASE"] == tr["sel"]).all())}
    ar = np.arange(n)
    sel_flags = {s: {k: F[f"{s}_{k}"][ar, picks["BASE"]].mean() for k in ("off", "edge", "lane")} for s in ("gt", "own")}
    cand = {nm: {"changed_pick_share": float((picks[nm] != picks["BASE"]).mean()),
                 "fallback_share_t1_t2_t3": [float(x) for x in fbs[nm].mean(0)], "kept_set_size_median": float(np.median(kept[nm]))} for nm, _s, _t in ARMS}
    hard_keys = ("K_G_fields", "K_B_emitted_equals_fan_at_sel_idx", "K_C_sel_idx_is_argmax_over_e9_eligible", "K_A_roll_equals_L1_BASEos", "K_RAST_all_windows_have_a_raster",
                 "K_LIVE_own_map_beats_deranged_donor", "K_DERANGED_donor_is_another_episode", "K_BASE_arm_equals_emitted_pick")
    hard = all(ctl[k]["pass"] for k in hard_keys)
    out = Path(a.out)
    base_res = {"tool": "mapsel_score.py", "seed": a.seed, "prereg_sha256": got, "amendment1_sha256": got2, "amendment2_sha256": got3, "prereg_seal_ok": seals_ok, "trace": str(a.trace), "n_windows": n,
                "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance); GT-* arms = T0-ORACLE BOUND (privileged map)",
                "controls": ctl, "hard_controls_pass": bool(hard), "candidate_level": {"arms": cand, "base_pick_candidate_flag_rate": sel_flags}, "lc_exception_active": lc_active}
    np.savez_compressed(out / "features_s0.npz" if not a.controls_only else out / "features_smoke.npz", **{k: v for k, v in F.items()}, pick_BASE=picks["BASE"], don=don)
    if a.controls_only:
        base_res["wall_s"] = round(time.time() - t_start, 1)
        jdump(base_res, out / "MAPSEL_CONTROLS.json")
        print(f"ZZMSCONTROLSZZ hard_controls_pass={int(hard)} " + json.dumps({k: ctl[k]["pass"] for k in ctl}), flush=True)
        return 0 if hard else 5
    # ---------------------------------------------------------------- full reading
    E = C.load_eval_pf(a.seed)
    P = S2R.load_panel(a.seed)
    jS6 = np.array([tr["idx"].get(k, -1) for k in zip(E["ex_clip"][E["rows"]].tolist(), E["ex_ws"][E["rows"]].tolist())])
    jS2 = np.array([tr["idx"].get(k, -1) for k in zip(P["clip"].tolist(), P["ws"].tolist())])
    pres6, pres2 = jS6 >= 0, jS2 >= 0
    if (not pres6.all() or not pres2.all()) and not a.allow_partial:
        raise SystemExit(f"[mapsel_score] windows missing from the trace: S6 {(~pres6).sum()} S2 {(~pres2).sum()}")
    jS6, jS2 = np.where(pres6, jS6, 0), np.where(pres2, jS2, 0)
    E = dict(E, compliant=np.asarray(E["compliant"], bool) & pres6)
    P = dict(P, compliant=np.asarray(P["compliant"], bool) & pres2)
    table = C.deploy_table()
    prior_t = np.zeros((n, 8, 2), np.float32)
    prior_t[jS2[pres2]] = np.asarray(P["prior8"])[pres2]
    plan8 = {nm: tr["fan"][ar, picks[nm]].astype(np.float32) for nm, _s, _t in ARMS}
    full = {nm: RS.retime_plan(plan8[nm], prior_t, C.ALL_T, table, cast=True) for nm in plan8}          # [n, 8, 2] the SHIPPED stage, every slot
    # AMENDMENT 2: the banked S2 stage was built from the BATTERY draw (S2_plan8), not from the os-only BASE -> stage THAT plan through the arms' own call
    bat_t = np.zeros((n, 8, 2), np.float32)
    bat_t[jS2[pres2]] = np.asarray(P["plan8"], np.float32)[pres2]
    full_bat = RS.retime_plan(bat_t, prior_t, C.ALL_T, table, cast=True)
    ctl["K_STAGE_BASE_staged_equals_banked_S2_pos"] = {"pass": bool(np.array_equal(full_bat[jS2][pres2][:, :4], np.asarray(P["stage_banked"])[pres2])),
                                                       "amended": "A2: battery S2_plan8 staged through the arms' call == S2_pos",
                                                       "descriptive_BASEos_staged_differs_from_banked_n": int((np.abs(full["BASE"][jS2][pres2][:, :4] - np.asarray(P["stage_banked"])[pres2]).reshape(int(pres2.sum()), -1).max(1) > 0).sum())}
    ctl["K_S6_alignment"] = {"checks": E["checks"], "pass": bool(E["checks"]["ws_equal_deploy"] and E["checks"]["clip_equal_deploy"] and E["checks"]["os_equal_deploy"])}
    # ---- pass 2: the compliance of the EMITTED (staged) plans, and of the human, under the GT masks
    comp = {nm: {k: np.full(n, np.nan) for k in [k_ for k_, _ in MAP_METRICS] + ["cov_nocls", "cov_unseen"]} for nm in list(plan8) + ["HUMAN"]}
    CH = 32
    t2 = time.time()
    with ThreadPoolExecutor(3) as ex:
        for a0 in range(0, n, CH):
            js = [j for j in range(a0, min(n, a0 + CH)) if has_gt[j]]
            gts = list(ex.map(lambda j: np.load(rast_file(a.rast, tr["e_i"][j], tr["ws"][j]))["gt"], js))
            for j, gt in zip(js, gts):
                names = list(plan8) + ["HUMAN"]
                plans = torch.from_numpy(np.stack([full[nm][j] for nm in plan8] + [hp[j]]).astype(np.float32))
                g = L.Geom(plans)
                mg = L.masks_gt(gt)
                off, edge, lane = (x.numpy() for x in g.flags(mg))
                cn, cu = (x.numpy() for x in g.coverage(mg))
                for i, nm in enumerate(names):
                    if nm == "HUMAN" and not hvalid[j]:
                        continue
                    comp[nm]["off_drivable"][j], comp[nm]["edge_contact"][j] = float(off[i]), float(edge[i])
                    comp[nm]["lane_cross"][j] = np.nan if lc[j] else float(lane[i])
                    comp[nm]["cov_nocls"][j], comp[nm]["cov_unseen"][j] = float(cn[i]), float(cu[i])
            if (a0 // CH) % 20 == 0:
                print(f"[mapsel_score] pass2 {a0 + CH}/{n} el={time.time() - t2:.0f}s", flush=True)
    # ---- populations
    sets = {"S6c": (np.asarray(E["compliant"], bool), E["eid"], jS6), "S2c": (np.asarray(P["compliant"], bool), P["eid"], jS2)}
    mapcells = {}
    for sk, (cm, eid, jj) in sets.items():
        mapcells[sk] = {}
        for nm in comp:
            mapcells[sk][nm] = {k: np.where(cm & has_gt[jj], comp[nm][k][jj], np.nan) for k in comp[nm]}
    res = dict(base_res)
    res["map_compliance"] = {}
    for sk, (cm, eid, jj) in sets.items():
        blk = {"n_windows_compliant": int(cm.sum()), "n_windows_with_gt": int((cm & has_gt[jj]).sum()), "arms": {}}
        for nm in comp:
            row = {}
            for k, _lab in MAP_METRICS:
                x = mapcells[sk][nm][k]
                row[k] = {"rate": float(np.nanmean(x)) if np.isfinite(x).any() else None, "n": int(np.isfinite(x).sum())}
                if nm not in ("BASE", "HUMAN"):
                    row[k]["vs_BASE"] = cell_json(PR.paired_cell(x, mapcells[sk]["BASE"][k], list(eid)))
            row["COVERAGE"] = {k: {"mean": (float(np.nanmean(mapcells[sk][nm][k])) if np.isfinite(mapcells[sk][nm][k]).any() else None)} for k in ("cov_nocls", "cov_unseen")}
            blk["arms"][nm] = row
        hum_rates = {k: blk["arms"]["HUMAN"][k]["rate"] for k in ("off_drivable", "edge_contact", "lane_cross")}
        for nm, row in blk["arms"].items():
            for k in ("off_drivable", "edge_contact", "lane_cross"):
                row[k]["minus_HUMAN"] = (row[k]["rate"] - hum_rates[k]) if (row[k]["rate"] is not None and hum_rates[k] is not None) else None
        res["map_compliance"][sk] = blk
    # ---- STOP-trap guard (literature: a compliance filter can 'win' by choosing slow / short candidates): path length + stop share of the EMITTED staged plan
    def arclen(X):
        Z = np.concatenate([np.zeros_like(X[:, :1]), X], axis=1)
        return np.linalg.norm(np.diff(Z.astype(np.float64), axis=1), axis=-1).sum(1)
    prog = {nm: {"path_len_m": arclen(full[nm]), "stop_share": (np.linalg.norm(full[nm][:, -1], axis=-1) < 1.0).astype(np.float64)} for nm in plan8}
    prog["HUMAN"] = {"path_len_m": np.where(hvalid, arclen(hp), np.nan), "stop_share": np.where(hvalid, (np.linalg.norm(hp[:, -1], axis=-1) < 1.0).astype(np.float64), np.nan)}
    res["progress_guard"] = {}
    for sk, (cm, eid, jj) in sets.items():
        blk = {}
        for nm in prog:
            blk[nm] = {}
            for k in ("path_len_m", "stop_share"):
                x = np.where(cm, prog[nm][k][jj], np.nan)
                blk[nm][k] = {"mean": float(np.nanmean(x)) if np.isfinite(x).any() else None, "n": int(np.isfinite(x).sum())}
                if nm not in ("BASE", "HUMAN"):
                    blk[nm][k]["vs_BASE"] = cell_json(PR.paired_cell(x, np.where(cm, prog["BASE"][k][jj], np.nan), list(eid)))
        res["progress_guard"][sk] = blk
    hum = res["map_compliance"]["S6c"]["arms"]["HUMAN"]["off_drivable"]
    ctl["K_HUMAN_off_drivable_le_2pct"] = {"rate_S6c": hum["rate"], "n": hum["n"], "max": HUMAN_MAX_OFF, "pass": bool(hum["rate"] is not None and hum["rate"] <= HUMAN_MAX_OFF)}
    if a.human_json and Path(a.human_json).exists():
        hj = json.loads(Path(a.human_json).read_text(encoding="utf-8"))
        ctl["K_HUMAN_prerolled"] = hj.get("S6_compliant")
        ctl["K_SENS_blind_lateral_shift"] = {**hj.get("sensitivity", {"ok": False}), "pass": bool(hj.get("sensitivity", {}).get("ok"))}
    else:
        ctl["K_SENS_blind_lateral_shift"] = {"pass": False, "note": "result/K_HUMAN.json missing"}
    g1 = res["map_compliance"]["S6c"]["arms"]["GT-T1"]["off_drivable"]["vs_BASE"]
    ctl["K_GTLIVE_GT_T1_lowers_off_drivable"] = {"delta": g1.get("delta"), "hi": g1.get("hi"), "pass": bool("hi" in g1 and g1["hi"] < 0)}
    # ---- four families
    rows6 = S.metric_rows(PR)
    rows2 = [r for r in rows6 if r[1] in S2R.KEYS]
    cells6 = {nm: S.arm_cells(E, plan8[nm][jS6]) for nm in plan8}
    res["four_families"] = {"S6c_stage": {}, "S2c_stage": {}}
    mask2 = np.asarray(P["compliant"], bool)
    Em2 = dict(P, compliant=mask2)

    def stage4(p8):
        return RS.retime_plan(np.asarray(p8), np.asarray(P["prior8"]), C.ALL_T, table, cast=True)[:, S2R.S2_SLOTS]
    cells2 = {nm: S2R.cells(Em2, stage4(plan8[nm][jS2]), mask2) for nm in plan8}
    for nm in plan8:
        if nm == "BASE":
            continue
        res["four_families"]["S6c_stage"][nm] = S.compare(PR, E, cells6[nm]["stage"], cells6["BASE"]["stage"], rows6)
        res["four_families"]["S2c_stage"][nm] = S2R_compare(S, PR, Em2, cells2[nm], cells2["BASE"], rows2)
    res["four_families"]["base_means"] = {"S6c": {k: float(np.nanmean(v)) for k, v in cells6["BASE"]["stage"].items() if np.isfinite(v).any()},
                                          "S2c": {k: float(np.nanmean(v)) for k, v in cells2["BASE"].items() if np.isfinite(v).any()}}
    # ---- the inference-draw FLOOR for context (L1 BASE-os seed 1 vs seed 0, shipped stage), same cells as BASE
    try:
        b1 = S.load_l1_base(a.l1_bank, 1)
        E1, P1 = C.load_eval_pf(1), S2R.load_panel(1)
        if b1 is not None and not b1.get("incomplete"):
            j6 = np.array([b1["idx"][k] for k in zip(E1["ex_clip"][E1["rows"]].tolist(), E1["ex_ws"][E1["rows"]].tolist())])
            j2 = np.array([b1["idx"][k] for k in zip(P1["clip"].tolist(), P1["ws"].tolist())])
            A1 = S.arm_cells(E1, b1["plan8"][j6])["stage"]
            res["four_families"]["FLOOR_S6c_BASEos_s1_vs_s0"] = S.compare(PR, E, A1, cells6["BASE"]["stage"], rows6) if np.array_equal(E1["ws"], E["ws"]) else "S6 windows differ between seeds"
            Em1 = dict(P1, compliant=np.asarray(P1["compliant"], bool))
            A12 = S2R.cells(Em1, RS.retime_plan(np.asarray(b1["plan8"][j2]), np.asarray(P1["prior8"]), C.ALL_T, table, cast=True)[:, S2R.S2_SLOTS], np.asarray(P1["compliant"], bool))
            res["four_families"]["FLOOR_S2c_BASEos_s1_vs_s0"] = S2R_compare(S, PR, Em2, A12, cells2["BASE"], rows2) if np.array_equal(P1["ws"], P["ws"]) else "S2 windows differ between seeds"
    except Exception as exc:      # noqa: BLE001  (context only; never fatal)
        res["four_families"]["FLOOR_error"] = repr(exc)
    # ---- GT-vs-OWN gap
    gap = {}
    for tname in ("T1", "T12", "T123"):
        r = {}
        for k, _ in MAP_METRICS:
            b = res["map_compliance"]["S6c"]["arms"]["BASE"][k]["rate"]
            gv, ov, dv = (res["map_compliance"]["S6c"]["arms"][f"{s}-{tname}"][k]["rate"] for s in ("GT", "OWN", "DER"))
            r[k] = {"BASE": b, "GT": gv, "OWN": ov, "DER": dv, "gt_gain": (b - gv) if None not in (b, gv) else None, "own_gain": (b - ov) if None not in (b, ov) else None,
                    "own_captures_share_of_gt_gain": ((b - ov) / (b - gv)) if None not in (b, gv, ov) and abs(b - gv) > 1e-9 else None}
        gap[tname] = r
    r = {}
    for k, _ in MAP_METRICS:
        b = res["map_compliance"]["S6c"]["arms"]["BASE"][k]["rate"]
        gv, ov, dv = (res["map_compliance"]["S6c"]["arms"][x][k]["rate"] for x in ("GT-T1", "OWN-CAL-T1", "DER-CAL-T1"))
        r[k] = {"BASE": b, "GT": gv, "OWN": ov, "DER": dv, "gt_gain": (b - gv) if None not in (b, gv) else None, "own_gain": (b - ov) if None not in (b, ov) else None,
                "own_captures_share_of_gt_gain": ((b - ov) / (b - gv)) if None not in (b, gv, ov) and abs(b - gv) > 1e-9 else None}
    gap["T1_CAL"] = r
    res["gt_vs_own_gap_S6c"] = gap
    res["tier_increments_own"] = {"OWN-T12_vs_OWN-T1_changed_pick_share": float((picks["OWN-T12"] != picks["OWN-T1"]).mean()), "OWN-T123_vs_OWN-T12_changed_pick_share": float((picks["OWN-T123"] != picks["OWN-T12"]).mean()),
                                  "GT-T12_vs_GT-T1_changed_pick_share": float((picks["GT-T12"] != picks["GT-T1"]).mean()), "GT-T123_vs_GT-T12_changed_pick_share": float((picks["GT-T123"] != picks["GT-T12"]).mean()),
                                  "note": "PREDICTED before the roll (prereg sec. 3.4): the own edge channel is empty (IoU 0.001) and lane IoU is 0.2 -> OWN-T12 ~ OWN-T1 (< 2 % of windows differ)"}
    # availability of a tier-compliant eligible candidate (fan existence), per source
    res["availability"] = {s: {"T1_set_nonempty": float(1.0 - fbs[f"{s.upper()}-T1"][:, 0].mean()), "T12_after_T1": float(1.0 - fbs[f"{s.upper()}-T12"][:, 1].mean()),
                               "T123_after_T12": float(1.0 - fbs[f"{s.upper()}-T123"][:, 2].mean())} for s in ("gt", "own", "der")}
    res["availability"]["cal"] = {"T1_set_nonempty": float(1.0 - fbs["OWN-CAL-T1"][:, 0].mean())}
    # ---- verdict
    verdict = {"arms": {}}
    void = [k for k in ("K_G_fields", "K_B_emitted_equals_fan_at_sel_idx", "K_C_sel_idx_is_argmax_over_e9_eligible", "K_A_roll_equals_L1_BASEos", "K_BASE_arm_equals_emitted_pick",
                        "K_STAGE_BASE_staged_equals_banked_S2_pos", "K_HUMAN_off_drivable_le_2pct", "K_GTLIVE_GT_T1_lowers_off_drivable", "K_LIVE_own_map_beats_deranged_donor",
                        "K_DERANGED_donor_is_another_episode", "TAU_CAL", "K_S6_alignment", "K_RAST_all_windows_have_a_raster", "K_SENS_blind_lateral_shift") if not ctl[k]["pass"]]
    verdict["void_reasons"] = void
    for nm in OWN_ARMS:
        c1d = res["map_compliance"]["S6c"]["arms"][nm]["off_drivable"]["vs_BASE"]
        c1 = bool("hi" in c1d and c1d["hi"] < 0)
        worse, guard_worse = [], []
        for surf in ("S6c_stage", "S2c_stage"):
            for k, c in res["four_families"][surf][nm].items():
                if c.get("worse"):
                    worse.append(f"{surf}:{k}")
                    if k in STOP_GUARD_ROWS:
                        guard_worse.append(f"{surf}:{k}")
        c2 = len(worse) == 0
        derd = res["map_compliance"]["S6c"]["arms"][DER_OF[nm]]["off_drivable"]["vs_BASE"]
        der123 = res["map_compliance"]["S6c"]["arms"]["DER-T123"]["off_drivable"]["vs_BASE"]
        c3 = bool(not ("hi" in derd and derd["hi"] < 0) and not ("hi" in der123 and der123["hi"] < 0))
        pl = res["progress_guard"]["S6c"][nm]["path_len_m"]["vs_BASE"]
        ss = res["progress_guard"]["S6c"][nm]["stop_share"]["vs_BASE"]
        shorter = bool("hi" in pl and pl["hi"] < 0)
        stopsmore = bool("lo" in ss and ss["lo"] > 0)
        c4 = bool(not guard_worse and not shorter and not stopsmore)
        verdict["arms"][nm] = {"C1_off_drivable_separated_below_BASE": c1, "C1_delta": c1d, "C2_no_four_family_row_separated_worse": c2, "C2_worse_rows": worse,
                               "C3_deranged_does_not_beat_BASE": c3, "C3_DER_matched": derd, "C3_DER_T123": der123,
                               "C4_stop_guard": c4, "C4_guard_rows_worse": guard_worse, "C4_path_len_separated_shorter": shorter, "C4_stop_share_separated_higher": stopsmore,
                               "C4_path_len_delta": pl, "C4_stop_share_delta": ss, "PASS": bool(c1 and c2 and c3 and c4)}
    heads = [x for x in HEADLINE_ARMS if verdict["arms"][x]["PASS"]]
    others = [x for x in OWN_ARMS if x not in HEADLINE_ARMS and verdict["arms"][x]["PASS"]]
    drv_only = [x for x in OWN_ARMS if all(verdict["arms"][x][k] for k in ("C1_off_drivable_separated_below_BASE", "C2_no_four_family_row_separated_worse", "C4_stop_guard"))
                and not verdict["arms"][x]["C3_deranged_does_not_beat_BASE"]]
    if void:
        word = "VOID"
    elif heads:
        word = "PASS (" + ", ".join(heads) + (" | secondary also: " + ", ".join(others) if others else "") + ")"
    elif others:
        word = "PASS-SECONDARY (" + ", ".join(others) + "; neither headline arm passed; 4 pre-declared OWN arms)"
    elif drv_only:
        word = "INCONCLUSIVE-DER (" + ", ".join(drv_only) + " clear C1, C2, C4 but the DERANGED control also beats BASE: the pick change alone improves compliance)"
    else:
        word = "FAIL"
    verdict["word"] = word
    verdict["gt_bound_T1"] = arm_clauses(res, "GT-T1")
    verdict["next_lever"] = next_lever(res, verdict, word)
    res["verdict"] = verdict
    res["controls"] = ctl
    res["hard_controls_pass"] = bool(hard)
    res["wall_s"] = round(time.time() - t_start, 1)
    jdump(res, out / "MAPSEL_RESULT.json")
    write_md(res, out)
    print(f"ZZMSVERDICTZZ {word}", flush=True)
    return 0


def arm_clauses(res, nm):
    """C1 (off-drivable separated below BASE), C2 (no four-family row separated worse), C4 (STOP guard) for ANY arm (used for the GT bound)."""
    c1d = res["map_compliance"]["S6c"]["arms"][nm]["off_drivable"]["vs_BASE"]
    worse = [f"{surf}:{k}" for surf in ("S6c_stage", "S2c_stage") for k, c in res["four_families"][surf][nm].items() if c.get("worse")]
    gw = [w for w in worse if w.split(":", 1)[1] in STOP_GUARD_ROWS]
    pl = res["progress_guard"]["S6c"][nm]["path_len_m"]["vs_BASE"]
    ss = res["progress_guard"]["S6c"][nm]["stop_share"]["vs_BASE"]
    return {"C1": bool("hi" in c1d and c1d["hi"] < 0), "C1_delta": c1d, "C2": not worse, "C2_worse_rows": worse,
            "C4": bool(not gw and not ("hi" in pl and pl["hi"] < 0) and not ("lo" in ss and ss["lo"] > 0))}


def next_lever(res, verdict, word):
    """RULE ZERO: the mechanical next lever, committed in PREREG_MAPSEL.md sec. 6.3 (a template over measured quantities, never a free-text judgement)."""
    gt1 = verdict["gt_bound_T1"]
    cap = res["gt_vs_own_gap_S6c"]["T1"]["off_drivable"]["own_captures_share_of_gt_gain"]
    capc = res["gt_vs_own_gap_S6c"]["T1_CAL"]["off_drivable"]["own_captures_share_of_gt_gain"]
    base = {"gt_T1_clears_C1_C2_C4": bool(gt1["C1"] and gt1["C2"] and gt1["C4"]), "own_T1_captures_share_of_gt_T1_gain": cap, "own_cal_T1_captures_share": capc}
    if word.startswith("VOID"):
        txt = "VOID: repair the named instrument defect (void_reasons) and re-run; no lever is read."
    elif word.startswith("PASS"):
        txt = ("PASS: the head's own map delivers a zero-training compliance gain. Next: (a) register it as an opt-in inference row beside SHIP-R8 (T1 stamp), (b) train the DrivableCritic on GT-map labels "
               "(GT-T1 is its bound; the remaining GT-minus-OWN gap is its headroom), (c) re-read on NavSim navtest zero-shot.")
    elif not base["gt_T1_clears_C1_C2_C4"]:
        txt = ("GT-T1 itself does not clear C1/C2/C4: even with the true map, compliance-first selection costs accuracy or progress (or does not lower off-drivable). The fan/selector, not the map "
               "channel, is the bottleneck -> next: a LEARNED listwise selector with map-derived labels over the same fan (not a rule), and a fan-availability audit (availability block).")
    elif (cap is not None and cap < 0.5) or (capc is not None and capc < 0.5):
        txt = ("GT-T1 clears C1/C2/C4 but the own map recovers < 50 % of its gain: PERCEPTION is the bottleneck. Next: (a) a learned critic on own-map softmax + map_hires_bev features with GT-map labels "
               "(the literature's strongest effect; GT-T1 is its bound), (b) P3 thin-class / drivable false-alarm work, (c) report the plan-level false-alarm rate of the hard rule (TAU_CAL block).")
    else:
        txt = ("GT-T1 clears C1/C2/C4 and the own map recovers >= 50 % of its gain, yet no OWN arm passed: the shortfall is in C2/C3/C4, not in map content. Read the failing clause in the verdict table "
               "and fix that one (a stop-gaming or accuracy regression -> add the progress term; the deranged control beating BASE -> the fan's non-top members are cleaner than the pick: a selector problem).")
    return {"text": txt, **base}


def S2R_compare(S, PR, Em, A, B, rows):
    return S.compare(PR, Em, A, B, rows)


def fmt(c):
    if not c or "delta" not in c:
        return "-"
    return f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"


def write_md(res, out):
    v = res["verdict"]
    L_ = [f"# refcv9 MAP-SELECT -- VERDICT: {v['word']}", "",
          f"tier: {res['tier']}", "estimator: paired episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci, B 2000, seed 0, over EPISODES) = 'another draw of episodes' ONLY; one checkpoint (EMA-48,852), one inference seed (0): "
          "NOT another training run, NOT another inference run.", f"prereg sha256: {res['prereg_sha256']} (seal ok: {res['prereg_seal_ok']})", f"windows: {res['n_windows']}; lane-change exception active: {res['lc_exception_active']}", ""]
    if v["void_reasons"]:
        L_ += ["## VOID reasons (the instrument is broken; nothing below is a result)", ""] + [f"* {r}" for r in v["void_reasons"]] + [""]
    L_ += ["## Controls", "", "| control | pass | detail |", "|---|---|---|"]
    for k, c in res["controls"].items():
        det = {kk: vv for kk, vv in c.items() if kk not in ("pass", "checks", "present_counts")} if isinstance(c, dict) else c
        L_.append(f"| {k} | {c.get('pass') if isinstance(c, dict) else '-'} | {json.dumps(det, default=str)[:300]} |")
    for sk, lab in (("S6c", "S6-compliant (6 instants, 1-6 s)"), ("S2c", "S2-compliant (4 instants, 0.5-2 s)")):
        b = res["map_compliance"][sk]
        L_ += ["", f"## MAP-COMPLIANCE vs the GT map, {lab}: n compliant {b['n_windows_compliant']}, with GT {b['n_windows_with_gt']} (EMITTED plan after the shipped stage; paired delta vs BASE)", "",
               "| arm | off-drivable (sidewalk) rate | delta vs BASE [CI] | edge-contact rate | delta vs BASE [CI] | minus HUMAN | lane-cross rate | delta vs BASE [CI] | minus HUMAN | COVERAGE nocls / not-seen (share of in-range check points) |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for nm, row in b["arms"].items():
            def cell(k, row=row):
                r = row[k]["rate"]
                mh = row[k].get("minus_HUMAN")
                return (f"{r:.4f}" if r is not None else "-"), fmt(row[k].get("vs_BASE")), (f"{mh:+.4f}" if mh is not None else "-")
            (o, od, _), (e, ed, em), (l, ld, lm) = cell("off_drivable"), cell("edge_contact"), cell("lane_cross")
            cv = row["COVERAGE"]
            cvs = " / ".join("-" if cv[k]["mean"] is None else f"{cv[k]['mean']:.4f}" for k in ("cov_nocls", "cov_unseen"))
            L_.append(f"| {nm} | {o} | {od} | {e} | {ed} | {em} | {l} | {ld} | {lm} | {cvs} |")
    L_ += ["", "T2 / T3 are read RELATIVE to the HUMAN floor (the 'minus HUMAN' columns), never as an absolute bar. nocls (code 0) is UNKNOWN, not a violation (AMENDMENT 1); its share is the COVERAGE of the check."]
    L_ += ["", "## GT vs OWN gap per tier (S6-compliant; share of the GT-oracle gain the deployable map recovers)", "", "| tier | metric | BASE | GT | OWN | DER | GT gain | OWN gain | OWN share of GT gain |", "|---|---|---|---|---|---|---|---|---|"]
    for t, r in res["gt_vs_own_gap_S6c"].items():
        for k, x in r.items():
            f_ = lambda y: "-" if y is None else f"{y:.4f}"  # noqa: E731
            L_.append(f"| {t} | {k} | {f_(x['BASE'])} | {f_(x['GT'])} | {f_(x['OWN'])} | {f_(x['DER'])} | {f_(x['gt_gain'])} | {f_(x['own_gain'])} | {f_(x['own_captures_share_of_gt_gain'])} |")
    L_ += ["", "## Verdict clauses (OWN arms)", "", "| arm | C1 off-drivable separated below BASE | C2 no four-family row separated worse | C3 deranged does not beat BASE | C4 STOP guard | PASS |", "|---|---|---|---|---|---|"]
    for nm, x in v["arms"].items():
        L_.append(f"| {nm} | {x['C1_off_drivable_separated_below_BASE']} ({fmt(x['C1_delta'])}) | {x['C2_no_four_family_row_separated_worse']} ({', '.join(x['C2_worse_rows'][:6]) or 'none'}) | {x['C3_deranged_does_not_beat_BASE']} | {x['C4_stop_guard']} (path-len {fmt(x['C4_path_len_delta'])} m; stop-share {fmt(x['C4_stop_share_delta'])}; guard rows worse: {', '.join(x['C4_guard_rows_worse']) or 'none'}) | **{x['PASS']}** |")
    L_ += ["", "## STOP-trap guard (S6-compliant, EMITTED staged plan; delta = arm - BASE)", "", "| arm | mean path length to 6 s (m) | delta vs BASE [CI] | stop share (6 s displacement < 1 m) | delta [CI] |", "|---|---|---|---|---|"]
    for nm, d in res["progress_guard"]["S6c"].items():
        pl, ss = d["path_len_m"], d["stop_share"]
        L_.append(f"| {nm} | {pl['mean']:.3f} | {fmt(pl.get('vs_BASE'))} | {ss['mean']:.4f} | {fmt(ss.get('vs_BASE'))} |")
    t = res["controls"].get("TAU_CAL", {})
    L_ += ["", f"## CALIBRATED own-map arm (OWN-CAL-T1): tau = {t.get('tau')}; human paths passing at tau {t.get('human_pass_frac_at_tau')}, at the hard rule (P >= 0.502) {t.get('human_pass_frac_at_hard_rule_0p502')} "
           f"(= 1 - the plan-level FALSE-ALARM rate of the hard own-map rule on human paths); n human paths {t.get('n_human_paths')}. Calibrated ON the eval human paths (privileged GT trajectories, no outcome metric); a train-side calibration is not available in this roll.",
           "", "## GT-T1 is the T0-ORACLE BOUND for the learned-selector lever (a DrivableCritic / Hydra-MDP++ / DDv2-style selector with map-derived labels); OWN-* is what the head's own map delivers with NO training.", "",
           "## Own-map tier increments (PREDICTED before the roll: OWN-T12 ~ OWN-T1, the own edge channel is empty)", "", json.dumps(res["tier_increments_own"], indent=1), ""]
    L_ += ["", "## Four families -- worst rows per arm (S6c and S2c, +stage, delta = arm - BASE; 'worse' = CI excludes 0 on the bad side)", ""]
    for surf in ("S6c_stage", "S2c_stage"):
        for nm, d in res["four_families"][surf].items():
            ws_ = [k for k, c in d.items() if c.get("worse")]
            sep = [k for k, c in d.items() if c.get("separated") and not c.get("worse")]
            L_.append(f"* {surf} {nm}: worse rows {ws_ or 'none'}; separated (other direction or unsigned) {sep[:8] or 'none'}")
    L_ += ["", "## Availability of a tier-compliant eligible candidate in the 149-fan (share of windows whose set did not need the fallback)", "", json.dumps(res["availability"], indent=1)]
    L_ += ["", "## Candidate-level (all windows): share of windows whose pick changed; fallback shares (T1, T2, T3)", ""]
    for nm, x in res["candidate_level"]["arms"].items():
        L_.append(f"* {nm}: changed {x['changed_pick_share']:.4f}; fallbacks {x['fallback_share_t1_t2_t3']}; kept-set median {x['kept_set_size_median']}")
    nl = res["verdict"]["next_lever"]
    L_ += ["", "## NEXT LEVER (mechanical, PREREG sec. 6.3)", "", nl["text"], "", json.dumps({k: v for k, v in nl.items() if k != "text"}, indent=1), ""]
    L_ += ["", "## Evidence class: MEASURED (raw artifacts: MAPSEL_RESULT.json, features_s0.npz). Not computed: distance-keeping and STRATEGIC rows (no lead track in the banked dumps / --no-strategic; n = 0)."]
    (Path(out) / "VERDICT.md").write_text("\n".join(L_), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
