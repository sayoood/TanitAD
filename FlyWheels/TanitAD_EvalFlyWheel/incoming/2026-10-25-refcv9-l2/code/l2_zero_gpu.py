"""l2_zero_gpu.py -- refcv9 L2 (PREREG_L2.md section 5 + controls K1/K2): the 0-GPU readings on the banked EMA-48,852 per-frame battery.
CPU only. Reads only banked arrays. Z1 head confusion (raw and calibrated), Z2 head-level frontier (ORACLE-TUNED on eval: an upper bound, never a
result), Z3 association projection (ESTIMATED), K1 stage bit-reproduction, K2 population + register literals.

    python l2_zero_gpu.py --out <dir> [--calibrator <json>] [--tag NAME] [--dev]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_common as C  # noqa: E402
import l2_fit as F  # noqa: E402

LITERALS_ALL_S0 = {"os_fd": 0.150, "os_md": 0.428, "r7_fd": 0.0965, "r7_md": 0.548}      # register R8-CONS-1 (printed precision)
LITERAL_TOL = {"os_fd": 5e-4, "os_md": 5e-4, "r7_fd": 5e-5, "r7_md": 5e-4}
POP_LITERAL = {"compliant_n": 3251, "compliant_ep": 138, "gt_cruise_n": 1856, "gt_decel_n": 639}
B_GRID = [round(x, 2) for x in np.arange(-4.0, 2.01, 0.25)]


def head_group(L6):
    return F.GROUP_OF6[np.argmax(L6, axis=1)]            # 0 decel, 1 cruise, 2 accel


def auc(score, pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie), rank-based."""
    s = np.concatenate([score[pos], score[neg]])
    r = s.argsort().argsort().astype(np.float64) + 1.0
    # average ranks for ties
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt) - cnt
    avg = (csum + (cnt + 1) / 2.0)[inv]
    n1, n0 = int(pos.sum()), int(neg.sum())
    return float((avg[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def per_seed(seed, cal):
    E = C.load_eval_pf(seed)
    comp = E["compliant"]
    allm = np.ones_like(comp)
    stg = C.apply_stage(E["plan8"], E["prior8"])
    out = {"seed": seed, "checks_alignment": E["checks"]}
    out["K1_stage_bit_equal_banked"] = bool(np.array_equal(stg, E["stage_banked"]))
    gt = C.plan_class(E["g"], E["v0"], E["dt"])
    out["populations"] = {"compliant_n": int(comp.sum()), "compliant_ep": int(len(set(E["eid"][comp].tolist()))),
                          "gt_cruise_n": int((comp & (gt == "cruise")).sum()), "gt_decel_n": int((comp & (gt == "decel")).sum()),
                          "gt_accel_n": int((comp & (gt == "accel")).sum())}
    # ---- K2 (all-window literals, seed 0 only) + the compliant cells that everything else is read against
    cells = {}
    for surf, m in (("compliant", comp), ("all", allm)):
        for arm, P in (("os", E["os"]), ("stage", stg), ("r7", E["r7"])):
            fd, md, _g, pc = C.lon_indicators(P, E, m)
            cells[f"{surf}|{arm}"] = {"fd": float(np.nanmean(fd)), "n_fd": int(np.isfinite(fd).sum()), "md": float(np.nanmean(md)), "n_md": int(np.isfinite(md).sum())}
    out["rate_cells"] = cells
    if seed == 0:
        got = {"os_fd": cells["all|os"]["fd"], "os_md": cells["all|os"]["md"], "r7_fd": cells["all|r7"]["fd"], "r7_md": cells["all|r7"]["md"]}
        out["K2_literals"] = {k: {"value": got[k], "literal": LITERALS_ALL_S0[k], "tol": LITERAL_TOL[k], "pass": bool(abs(got[k] - LITERALS_ALL_S0[k]) <= LITERAL_TOL[k])}
                              for k in got}
    out["K2_populations_pass"] = bool(out["populations"]["compliant_n"] == POP_LITERAL["compliant_n"] and out["populations"]["compliant_ep"] == POP_LITERAL["compliant_ep"]
                                      and out["populations"]["gt_cruise_n"] == POP_LITERAL["gt_cruise_n"] and out["populations"]["gt_decel_n"] == POP_LITERAL["gt_decel_n"])
    # ---- Z1 head confusion on compliant windows
    L6 = C.lon6(E["logits"])
    hg_raw = head_group(L6)
    names = ("decel", "cruise", "accel")

    def conf(hg):
        return {g: {names[k]: int(((gt == g) & comp & (hg == k)).sum()) for k in range(3)} for g in names}

    z1 = {"raw": conf(hg_raw)}
    cruise_m, decel_m = comp & (gt == "cruise"), comp & (gt == "decel")
    z1["raw_head_fd"] = float((hg_raw[cruise_m] == 0).mean())
    z1["raw_head_md"] = float((hg_raw[decel_m] != 0).mean())
    if cal is not None:
        L6c = C.lon6(F.apply8(E["logits"], b8=cal["b8"], T=cal["T"]))
        hg_cal = head_group(L6c)
        z1["calibrated"] = conf(hg_cal)
        z1["cal_head_fd"] = float((hg_cal[cruise_m] == 0).mean())
        z1["cal_head_md"] = float((hg_cal[decel_m] != 0).mean())
        z1["flips"] = {
            "gt_cruise_decel_to_nondecel": int((cruise_m & (hg_raw == 0) & (hg_cal != 0)).sum()),
            "gt_cruise_nondecel_to_decel": int((cruise_m & (hg_raw != 0) & (hg_cal == 0)).sum()),
            "gt_decel_decel_to_nondecel": int((decel_m & (hg_raw == 0) & (hg_cal != 0)).sum()),
            "gt_decel_nondecel_to_decel": int((decel_m & (hg_raw != 0) & (hg_cal == 0)).sum()),
            "gt_cruise_raw_decel_n": int((cruise_m & (hg_raw == 0)).sum()), "gt_decel_raw_decel_n": int((decel_m & (hg_raw == 0)).sum()),
            "argmax_class_changed_all_compliant": int((comp & (np.argmax(L6, 1) != np.argmax(L6c, 1))).sum())}
        z1["cal_mean_decel_group_prob_gt_cruise"] = {
            "raw": float(np.exp(logsumexp(L6[cruise_m][:, F.GROUP_OF6 == 0] - logsumexp(L6[cruise_m], 1, keepdims=True), axis=1)).mean()),
            "cal": float(np.exp(logsumexp(L6c[cruise_m][:, F.GROUP_OF6 == 0] - logsumexp(L6c[cruise_m], 1, keepdims=True), axis=1)).mean())}
    out["Z1"] = z1
    # ---- Z2 head-level frontier (ORACLE-TUNED on eval; T=1)
    ld = logsumexp(L6[:, F.GROUP_OF6 == 0], axis=1) - logsumexp(L6[:, F.GROUP_OF6 == 1], axis=1)
    z2 = {"auc_decel_vs_cruise_logodds_GTdecel_over_GTcruise": auc(ld, decel_m, cruise_m), "grid": []}
    for b in B_GRID:
        Lb = L6.copy()
        Lb[:, F.GROUP_OF6 == 0] += b
        hg = head_group(Lb)
        z2["grid"].append({"bD": b, "head_fd": float((hg[cruise_m] == 0).mean()), "head_md": float((hg[decel_m] != 0).mean())})
    out["Z2"] = z2
    # ---- Z3 association: P(plan decel | GT class, head group) from the banked battery os and stage plans; projection under head-group changes
    z3 = {}
    for arm, P in (("os", E["os"]), ("stage", stg)):
        pc = C.plan_class(P, E["v0"], E["dt"])
        t = {}
        for k, nm in enumerate(names):
            m_c = cruise_m & (hg_raw == k)
            m_d = decel_m & (hg_raw == k)
            t[nm] = {"n_gt_cruise": int(m_c.sum()), "p_plan_decel_given_gt_cruise": (float((pc[m_c] == "decel").mean()) if m_c.any() else None),
                     "n_gt_decel": int(m_d.sum()), "p_plan_nondecel_given_gt_decel": (float((pc[m_d] != "decel").mean()) if m_d.any() else None)}
        z3[arm] = {"by_head_group": t}
        if cal is not None:
            hg_cal = head_group(C.lon6(F.apply8(E["logits"], b8=cal["b8"], T=cal["T"])))
            # projection: each window keeps its raw-head-group conditional rate replaced by the calibrated head group's rate
            def proj(mask, key_p, hgx):
                v = [t[names[k]][key_p] for k in range(3)]
                return float(np.mean([v[g] if v[g] is not None else np.nan for g in hgx[mask]]))
            z3[arm]["projected_fd_calibrated"] = proj(cruise_m, "p_plan_decel_given_gt_cruise", hg_cal)
            z3[arm]["projected_md_calibrated"] = proj(decel_m, "p_plan_nondecel_given_gt_decel", hg_cal)
            z3[arm]["projected_fd_raw_check"] = proj(cruise_m, "p_plan_decel_given_gt_cruise", hg_raw)
            z3[arm]["projected_md_raw_check"] = proj(decel_m, "p_plan_nondecel_given_gt_decel", hg_raw)
        # the CONST-limit: every window's head says cruise
        v_c = t["cruise"]["p_plan_decel_given_gt_cruise"]
        v_m = t["cruise"]["p_plan_nondecel_given_gt_decel"]
        z3[arm]["const_cruise_limit_projection"] = {"fd": v_c, "md": v_m, "note": "P(plan | head says cruise-group) from the banked windows: ESTIMATED (association)"}
    out["Z3"] = z3
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibrator", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dev", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    seal = Path(__file__).resolve().parent.parent
    want = (seal / "PREREG_L2.sha256").read_text().split()[0]
    got = C.sha256_file(seal / "PREREG_L2.md")
    if want != got:
        raise SystemExit(f"[l2_zero_gpu] PREREG_L2.md does not match its seal ({got[:12]} vs {want[:12]}): refusing")
    cal = C.jload(a.calibrator) if a.calibrator else None
    res = {"tool": "l2_zero_gpu.py", "prereg_sha256": got, "tag": a.tag, "DEV_NOT_A_RESULT": bool(a.dev),
           "calibrator": (None if cal is None else {"file": a.calibrator, "sha256": C.sha256_file(a.calibrator), "tag": cal.get("tag"), "theta": cal.get("theta"), "T": cal.get("T")}),
           "tier": "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)",
           "estimator": "descriptive counts / rates (Z1-Z3 carry no interval); the bars use paired_cell in l2_score.py",
           "seeds": {str(s): per_seed(s, cal) for s in (0, 1)}, "script_sha256": C.sha256_file(__file__), "wall_s": None}
    res["controls"] = {"K1_all": all(res["seeds"][s]["K1_stage_bit_equal_banked"] for s in ("0", "1")),
                       "K2_populations_all": all(res["seeds"][s]["K2_populations_pass"] for s in ("0", "1")),
                       "K2_literals_all": all(v["pass"] for v in res["seeds"]["0"]["K2_literals"].values())}
    res["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    C.jdump(res, Path(a.out) / "ZERO_GPU.json")
    print(json.dumps({"controls": res["controls"], "wall_s": res["wall_s"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
