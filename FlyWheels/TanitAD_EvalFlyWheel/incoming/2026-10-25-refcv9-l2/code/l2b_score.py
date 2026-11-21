"""l2b_score.py -- refcv9 L2-B (PREREG_L2B.md): the MECHANICAL reading of the brake-locus roll: SELECTION vs CANDIDATE vs MIXED.  CPU only; reads
`locus_s{seed}.npz` (l2b_locus_roll.py), the banked per-frame battery (GT, v0, compliance, episodes, banked prior), and L1's un-hooked BASE-os bank (identity control).

    python l2b_score.py --seed 0 --trace F:/refcv9_l2_locus/locus_s0.npz --out result/l2b_s0           # scored reading (refuses unless PREREG_L2B.md matches its seal)
    python l2b_score.py --seed 0 --trace <smoke npz> --out <dir> --controls-only                        # K-B/K-C/K-G/K-A on whatever windows the trace holds

TIER: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  ESTIMATOR: episode-cluster bootstrap (pair_read_r8.paired_cell ->
taniteval.ci, B 2000, seed 0) = 'another draw of EPISODES' only; NOT another training run, NOT another inference run (seed 1 is the replicate, never pooled).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_common as C  # noqa: E402
import l2_score as S2  # noqa: E402  (load_l1_base)

L1_BANK = "C:/Users/Admin/refcv8_pf/refcv9_l1/bank"
# ---- constants frozen by PREREG_L2B.md (changing any of them changes the experiment: the seal check refuses first)
THR_SEL, THR_CAND = 0.67, 0.33          # R1/R2: SELECTION iff CI lower > 0.67; CANDIDATE iff CI upper < 0.33; else MIXED
SATURATED = 0.95                        # the informativeness guard: existence is non-discriminating iff both FB and DR point estimates >= 0.95
TAU_Q = 0.90                            # GOODCRUISE tolerance: q90 of the CT windows' emitted staged ADE_6s
N_EXPECT_WINDOWS, N_BASE, N_FAN = 4754, 117, 149
FB_LITERAL_S0 = 259                     # = 0.1395 * 1856 (L2_RESULT BASE-os|stage false-decel x GT-cruise n), re-derived from L1 BASE-os below


def load_locus(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    if meta.get("final") is not True:
        raise SystemExit(f"[l2b_score] {path} is not a FINAL trace")
    d = {k: z[k] for k in z.files if k != "meta_json"}
    d["idx"] = {(int(e), int(w)): j for j, (e, w) in enumerate(zip(d["e_i"], d["ws"]))}
    d["meta"], d["n"] = meta, len(d["ws"])
    return d


def eligible(score, reach, ceil, emit):
    """EXACT numpy replica of refc_v3.e9_rank's finite mask. score [n,N]; masks [n,N] int8 with -1 = ABSENT (None semantics)."""
    fin = np.isfinite(score)

    def has(m):
        return m is not None and not (np.asarray(m) == -1).all()
    f0 = fin & (reach.astype(bool) if has(reach) else True)
    f1 = f0 & ceil.astype(bool) if has(ceil) else f0
    dead = ~f1.any(1)
    f1 = np.where(dead[:, None], f0, f1)
    if has(emit):
        eb = emit.astype(bool)
        f2 = f1 & eb
        dead2 = ~f2.any(1)
        return np.where(dead2[:, None], fin & eb, f2)
    return f1


def stage_cands(fan, prior, table, chunk=200):
    """fan [n,N,8,2], prior [n,8,2] -> staged S6 paths [n,N,6,2] (the shipped DEPLOY8 stage on every candidate with the window's prior)."""
    RS = C.stack()["RS"]
    n, N = fan.shape[:2]
    out = np.empty((n, N, 6, 2), np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        flat = fan[a:b].reshape(-1, 8, 2)
        pr = np.repeat(prior[a:b], N, axis=0)
        out[a:b] = RS.retime_plan(flat, pr, C.ALL_T, table, cast=True)[:, C.S6_SLOTS].reshape(b - a, N, 6, 2)
    return out


def cls_of(P, v0, dt):
    """P [n,N,6,2] -> class codes [n,N] (0 decel, 1 cruise, 2 accel) by the battery's plan_lon_class."""
    n, N = P.shape[:2]
    c = C.plan_class(P.reshape(-1, 6, 2), np.repeat(v0, N), dt).reshape(n, N)
    return np.where(c == "decel", 0, np.where(c == "cruise", 1, 2))


def ci_mean(PR, ind, eid):
    """Episode-cluster bootstrap CI of the mean of an indicator that is NaN outside its set (the registered estimator, paired against a zero arm)."""
    zero = np.where(np.isnan(ind), np.nan, 0.0)
    c = PR.paired_cell(ind, zero, list(eid))
    return {"mean": float(c["delta"]), "lo": float(c["lo"]), "hi": float(c["hi"]), "n_windows": int(c["n_windows"]), "n_episodes": int(c["n_episodes"])}


def label_from(ci, thr_sel=THR_SEL, thr_cand=THR_CAND):
    if ci["lo"] > thr_sel:
        return "SELECTION"
    if ci["hi"] < thr_cand:
        return "CANDIDATE"
    return "MIXED"


def read_set(PR, name, setmask, target, A, eid, tau):
    """Per-window locus reading on `setmask` for the TARGET class (cruise for FB/CT/DR, decel for MB). A: dict of per-window arrays over ALL S6 windows."""
    ok = setmask
    ex = np.where(ok, A["has"][target].astype(np.float64), np.nan)                       # a TARGET-class eligible candidate exists
    good = np.where(ok, A["good"][target].astype(np.float64), np.nan)                    # ... AND is as good as the model's own typical right answer (ADE <= tau)
    base_t = np.where(ok, (A["cls_base"] == target).astype(np.float64), np.nan)          # the refcv6 (pre-blend) pick is TARGET class
    out = {"n": int(ok.sum()), "EXIST": ci_mean(PR, ex, eid), "GOOD": ci_mean(PR, good, eid), "BASE_PICK_is_target": ci_mean(PR, base_t, eid),
           "n_target_cands_median": float(np.median(A["n_target"][target][ok])) if ok.any() else None}
    gap = A["gap"][target][ok & A["has"][target]]
    out["score_gap_emitted_minus_best_target(sel_score_v3)"] = ({"q10_50_90": [float(x) for x in np.quantile(gap, [.1, .5, .9])], "n": int(len(gap))} if len(gap) else None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-only", action="store_true")
    ap.add_argument("--l1-bank", default=L1_BANK)
    a = ap.parse_args()
    t0 = time.time()
    root = Path(__file__).resolve().parent.parent
    seal = (root / "PREREG_L2B.sha256").read_text().split()[0]
    got = C.sha256_file(root / "PREREG_L2B.md")
    if seal != got and not a.controls_only:
        raise SystemExit(f"[l2b_score] PREREG_L2B.md does not match its seal ({got[:12]} vs {seal[:12]}): refusing a scored reading")
    st = C.stack()
    PR = st["PR"]
    tr = load_locus(a.trace)
    E = C.load_eval_pf(a.seed)
    base = S2.load_l1_base(a.l1_bank, a.seed)
    table = C.deploy_table()
    ctl = {}
    # ------------------------------------------------------------------ controls that need no sets (all trace windows)
    n = tr["n"]
    fan, sc, sv3 = tr["fan"], tr["sel_score"], tr["sel_score_v3"]
    ctl["K_G_fields"] = {"n_windows": n, "present_counts": tr["meta"]["present_counts"], "fan_n_seen": tr["meta"]["counters"]["fan_n_seen"], "batch_gt1": tr["meta"]["counters"]["batch_gt1"],
                         "pass": bool(all(tr["meta"]["present_counts"].get(k, 0) == n for k in ("fan", "sel_score", "sel_score_v3", "prior_path", "kind", "lon6")) and
                                      tr["meta"]["counters"]["fan_n_seen"] == [N_FAN] and tr["meta"]["counters"]["batch_gt1"] == 0)}
    emitted = fan[np.arange(n), tr["sel"]]
    ctl["K_B_emitted_equals_fan_at_sel_idx"] = {"max_abs": float(np.abs(emitted - tr["plan8"]).max()), "pass": bool(np.array_equal(emitted, tr["plan8"]))}
    elig = eligible(sv3, tr.get("reach_keep"), tr.get("ceil_keep"), tr.get("emit_keep"))
    mx = np.where(elig, sv3, -np.inf).max(1)
    ctl["K_C_sel_idx_is_argmax_over_e9_eligible"] = {"share": float((sv3[np.arange(n), tr["sel"]] == mx).mean()), "pass": bool((sv3[np.arange(n), tr["sel"]] == mx).all()),
                                                     "emitted_eligible_share": float(elig[np.arange(n), tr["sel"]].mean()), "mask_present": {k: bool(tr[k].max() >= 0) if k in tr else False for k in ("reach_keep", "ceil_keep", "emit_keep")}}
    if base is not None and not base.get("incomplete"):
        ks = [k for k in tr["idx"] if k in base["idx"]]
        d = np.array([np.abs(tr["plan8"][tr["idx"][k]] - base["plan8"][base["idx"][k]]).max() for k in ks])
        ctl["K_A_reroll_equals_L1_BASEos"] = {"n_compared": len(ks), "n_bit_equal": int((d == 0).sum()), "max_abs": float(d.max()), "pass": bool((d == 0).all())}
    else:
        ctl["K_A_reroll_equals_L1_BASEos"] = {"pass": False, "note": "L1 BASE-os bank missing"}
    hard = ctl["K_G_fields"]["pass"] and ctl["K_B_emitted_equals_fan_at_sel_idx"]["pass"] and ctl["K_C_sel_idx_is_argmax_over_e9_eligible"]["pass"]
    res = {"tool": "l2b_score.py", "seed": a.seed, "prereg_sha256": got, "prereg_seal_ok": seal == got, "trace": str(a.trace), "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)",
           "estimator": "episode-cluster bootstrap (pair_read_r8.paired_cell -> taniteval.ci), B 2000, seed 0, over EPISODES: 'another draw of episodes' only; NOT another training run, NOT another inference run (seed 1 = replicate, never pooled)",
           "controls": ctl, "controls_hard_pass": bool(hard)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.controls_only:
        C.jdump(res, out / "L2B_CONTROLS.json")
        print(json.dumps(ctl, indent=1, default=str))
        return 0 if hard else 6
    # ------------------------------------------------------------------ S6 surface
    keys = list(zip(E["ex_clip"][E["rows"]].tolist(), E["ex_ws"][E["rows"]].tolist()))
    j = np.array([tr["idx"].get(k, -1) for k in keys])
    if (j < 0).any():
        raise SystemExit(f"[l2b_score] {int((j < 0).sum())} S6 windows have no entry in the trace")
    comp, g, v0, dt, eid = E["compliant"], E["g"], E["v0"], E["dt"], E["eid"]
    prior_roll = tr["prior_path"][j]
    ctl["K_D_prior_path_equals_banked_prior8"] = {"max_abs": float(np.abs(prior_roll - E["prior8"]).max()), "pass": bool(np.abs(prior_roll - E["prior8"]).max() < 1e-4)}
    plan8 = tr["plan8"][j]
    emit_staged = C.apply_stage(plan8, prior_roll)                       # [n6,6,2]
    ce = C.plan_class(emit_staged, v0, dt)
    gt = C.plan_class(g, v0, dt)
    ade_emit = np.linalg.norm(emit_staged.astype(np.float64) - g.astype(np.float64), axis=-1).mean(-1)
    comps = C.arm_metrics(E, emit_staged)
    ctl["K_H_ade_definition_matches_battery"] = {"max_abs": float(np.abs(ade_emit - np.asarray(comps["ade_m"])).max()), "pass": bool(np.abs(ade_emit - np.asarray(comps["ade_m"])).max() < 1e-4)}
    FB = comp & (gt == "cruise") & (ce == "decel")
    cruise_pool = comp & (gt == "cruise") & (ce == "cruise")
    DR = comp & (gt == "decel") & (ce == "decel")
    MB = comp & (gt == "decel") & (ce != "decel")
    # CT: 1:1 greedy match to FB on (nav class exact, nearest |v0|), deterministic, without replacement
    D = C.ARCH / C.EVAL_TREE / f"dump_s{a.seed}" / "decisions"
    import glob
    nav = np.concatenate([np.load(f)["nav_cmd"] for f in sorted(glob.glob(str(D / "ep*.npz")))])[E["rows"]]
    CT = np.zeros(len(comp), bool)
    used = np.zeros(len(comp), bool)
    order = sorted(np.nonzero(FB)[0].tolist(), key=lambda i: (int(eid[i]), int(E["ws"][i])))
    for i in order:
        pool = np.nonzero(cruise_pool & ~used & (nav == nav[i]))[0]
        if len(pool) == 0:
            continue
        k = pool[np.argmin(np.abs(v0[pool] - v0[i]) + 1e-9 * pool)]
        CT[k] = used[k] = True
    ctl["K_F_set_sizes"] = {"FB": int(FB.sum()), "CT": int(CT.sum()), "DR": int(DR.sum()), "MB": int(MB.sum()), "GT_cruise_compliant": int((comp & (gt == "cruise")).sum()),
                            "FB_literal_s0": FB_LITERAL_S0 if a.seed == 0 else None, "pass": bool(a.seed != 0 or int(FB.sum()) == FB_LITERAL_S0)}
    ctl["K_I_CT_identity"] = {"emitted_is_cruise_share": float((ce[CT] == "cruise").mean()) if CT.any() else None, "pass": bool(CT.any() and (ce[CT] == "cruise").all()),
                              "balance": {"v0_mean_FB": float(v0[FB].mean()), "v0_mean_CT": float(v0[CT].mean()), "nav_share_FB": [float((nav[FB] == k).mean()) for k in (0, 1, 2)],
                                          "nav_share_CT": [float((nav[CT] == k).mean()) for k in (0, 1, 2)]}}
    # ------------------------------------------------------------------ candidate-level arrays over the windows we read
    need = np.nonzero(FB | CT | DR | MB)[0]
    fan_s = np.full((len(comp), N_FAN, 6, 2), np.nan, np.float32)
    fan_s[need] = stage_cands(tr["fan"][j[need]], prior_roll[need], table)
    cls = np.full((len(comp), N_FAN), -1, np.int8)
    cls[need] = cls_of(fan_s[need], v0[need], dt)
    cade = np.full((len(comp), N_FAN), np.nan)
    cade[need] = np.linalg.norm(fan_s[need].astype(np.float64) - g[need][:, None].astype(np.float64), axis=-1).mean(-1)
    el = eligible(tr["sel_score_v3"][j], tr["reach_keep"][j] if "reach_keep" in tr else None, tr["ceil_keep"][j] if "ceil_keep" in tr else None,
                  tr["emit_keep"][j] if "emit_keep" in tr else None)
    s3 = tr["sel_score_v3"][j]
    sel = tr["sel"][j]
    tau = float(np.quantile(ade_emit[CT], TAU_Q))
    ctl["K_J_emitted_candidate_class_matches"] = {"share": float((cls[np.arange(len(comp)), sel][need] == np.where(ce[need] == "decel", 0, np.where(ce[need] == "cruise", 1, 2))).mean()),
                                                  "pass": bool((cls[np.arange(len(comp)), sel][need] == np.where(ce[need] == "decel", 0, np.where(ce[need] == "cruise", 1, 2))).all())}
    A = {"has": {}, "good": {}, "n_target": {}, "gap": {}, "cls_base": np.full(len(comp), -1, np.int8)}
    for tgt in (1, 0):
        isT = (cls == tgt) & el
        A["has"][tgt] = isT.any(1)
        A["n_target"][tgt] = isT.sum(1)
        A["good"][tgt] = (isT & (cade <= tau)).any(1)
        best = np.where(isT, s3, -np.inf).max(1)
        A["gap"][tgt] = np.where(np.isfinite(best), s3[np.arange(len(comp)), sel] - best, np.nan)
    sb = tr["sel_base"][j]
    A["cls_base"][need] = cls[np.arange(len(comp)), np.clip(sb, 0, N_FAN - 1)][need]
    ctl["K_J_gap_nonnegative_when_emitted_is_argmax"] = {"min_gap_all_sets": float(np.nanmin(np.concatenate([A["gap"][1][need], A["gap"][0][need]]))), "pass": bool(np.nanmin(np.concatenate([A["gap"][1][need], A["gap"][0][need]])) >= -1e-6)}
    # ------------------------------------------------------------------ readings
    R = {"tau_good_cruise_m": tau, "tau_rule": f"q{int(TAU_Q * 100)} of the CT windows' emitted staged ADE_6s"}
    R["FB"] = read_set(PR, "FB", FB, 1, A, eid, tau)
    R["CT"] = read_set(PR, "CT", CT, 1, A, eid, tau)
    R["DR"] = read_set(PR, "DR", DR, 1, A, eid, tau)
    R["MB"] = read_set(PR, "MB", MB, 0, A, eid, tau)
    s1, s2 = R["FB"]["EXIST"], R["FB"]["GOOD"]
    guard = bool(s1["mean"] >= SATURATED and R["DR"]["EXIST"]["mean"] >= SATURATED)
    r1, r2 = label_from(s1), label_from(s2)
    final = r2 if guard else r1
    ngc = int((comp & (gt == "cruise")).sum())
    fd_now = float(FB.sum()) / ngc
    fd_floor_exist = float((FB & ~A["has"][1]).sum()) / ngc
    fd_floor_good = float((FB & ~A["good"][1]).sum()) / ngc
    # descriptive headroom: the best (by sel_score_v3) eligible cruise-class candidate vs the emitted plan and vs the GT-oracle candidate, on FB windows that have one
    bestj = np.where(((cls == 1) & el), s3, -np.inf).argmax(1)
    ade_bestc = np.where(A["has"][1], cade[np.arange(len(comp)), bestj], np.nan)
    ade_orac = np.where(el, cade, np.inf).min(1)
    hd = {"ADE6_best_cruise_cand_minus_emitted": ci_mean(PR, np.where(FB, ade_bestc - ade_emit, np.nan), eid),
          "ADE6_gt_oracle_cand_minus_emitted": ci_mean(PR, np.where(FB, ade_orac - ade_emit, np.nan), eid),
          "ADE6_emitted_FB_mean": float(ade_emit[FB].mean()), "ADE6_emitted_CT_mean": float(ade_emit[CT].mean())}
    base_pick = {"FB_base_pick_class_shares(decel,cruise,accel)": [float((A["cls_base"][FB] == k).mean()) for k in (0, 1, 2)]}
    res.update(controls=ctl, controls_hard_pass=bool(hard and ctl["K_D_prior_path_equals_banked_prior8"]["pass"] and ctl["K_H_ade_definition_matches_battery"]["pass"] and
                                                     ctl["K_I_CT_identity"]["pass"] and ctl["K_J_emitted_candidate_class_matches"]["pass"] and ctl["K_F_set_sizes"]["pass"]),
               readings=R, R1_label=r1, R2_label=r2, guard_existence_saturated=guard, label=final,
               label_basis=("R2 (GOOD-cruise existence; the class-level existence is saturated in FB and DR)" if guard else "R1 (cruise-class existence)"),
               FD={"now": fd_now, "floor_if_every_SELECTION_window_fixed(class existence)": fd_floor_exist, "floor_if_every_SELECTION_window_fixed(GOOD existence)": fd_floor_good,
                   "n_GT_cruise_compliant": ngc}, headroom_descriptive=hd, base_pick=base_pick,
               scope={"windows": "S6 compliant", "uncomputed": "distance-keeping (no lead track) and STRATEGIC (--no-strategic): n = 0"}, wall_s=round(time.time() - t0, 1))
    if not res["controls_hard_pass"]:
        res["label_stamped"] = "INCONCLUSIVE (a hard control failed: " + ", ".join(k for k, v in ctl.items() if isinstance(v, dict) and v.get("pass") is False) + ")"
    C.jdump(res, out / "L2B_RESULT.json")
    md = [f"# L2-B brake locus (seed {a.seed}) -- {res.get('label_stamped', final)}", "", f"PREREG_L2B sha256 {got[:12]}  |  tier {res['tier']}", f"estimator: {res['estimator']}", "",
          f"**label {final}** (basis: {res['label_basis']}); R1 (class existence) {r1}, R2 (good-cruise existence) {r2}; existence saturated: {guard}", "",
          "| set | n | EXIST target-class cand [CI] | GOOD (ADE<=tau) [CI] | refcv6 pre-blend pick is target class | median #target cands |", "|---|---|---|---|---|---|"]
    for nm in ("FB", "CT", "DR", "MB"):
        r = R[nm]
        md.append(f"| {nm}{' (target=decel)' if nm == 'MB' else ''} | {r['n']} | {r['EXIST']['mean']:.3f} [{r['EXIST']['lo']:.3f}, {r['EXIST']['hi']:.3f}] | {r['GOOD']['mean']:.3f} [{r['GOOD']['lo']:.3f}, {r['GOOD']['hi']:.3f}] | "
                  f"{r['BASE_PICK_is_target']['mean']:.3f} | {r['n_target_cands_median']} |")
    md += ["", f"FD now {fd_now:.4f}; floor if every SELECTION window were fixed: class-existence {fd_floor_exist:.4f}, good-existence {fd_floor_good:.4f} (n GT-cruise {ngc})",
           f"tau (GOOD tolerance) {tau:.3f} m; headroom (descriptive): {json.dumps(hd, default=str)}", f"FB score gap quantiles: {R['FB']['score_gap_emitted_minus_best_target(sel_score_v3)']}",
           f"base pick: {base_pick}", "", "controls: " + json.dumps({k: (v.get('pass') if isinstance(v, dict) else v) for k, v in ctl.items()})]
    (out / "L2B_RESULT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"ZZL2BLABELZZ seed={a.seed} label={res.get('label_stamped', final)} R1={r1} R2={r2} guard={guard} hard={res['controls_hard_pass']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
