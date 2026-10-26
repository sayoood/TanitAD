"""l1c_roc.py -- PREREG_L1C bar G (braking AUC): DERIVED from refcv9_l1/code/l1_roc.py (PREREG_L1 AMENDMENT 2) with the arm prefix L1_ -> L1C_ and the
floor arm L1C_H2cs; adds the paired AUC vs H0 (the arm the prereg names).  The L2 ROC definition is reused, not re-derived.

Definition imported from `code/l2roc_ref/l2_roc.py` (byte-identical to 726f996 `…/2026-10-25-refcv9-l2/roc/l2_roc.py`, sha256
5f7cc047…): score dv = |P[6 s] − P[5 s]| / dt − v0; positives = GT-decel (dv_GT ≤ −1.5), negatives = GT-cruise (|dv_GT| < 1.5),
ceiling-compliant S6 windows of the 48,852 per-frame battery; AUC = P(score_pos < score_neg) + ½ ties.  The population, v0, GT, the
banked stage (`DEPLOY8_POS … S6_pos` = refcv8 + stage) and refcv7 come from the L2 loader `l2_common.load_eval_pf` UNCHANGED; the L1 arms
come from `result/arms_S6_s{seed}.npz` (l1_fit_score.py), aligned by (clip, ws).
Paired episode-cluster bootstrap over the episodes of the population, B 2000, numpy seed 0, ONE draw sequence shared by every arm (the L2
loop, extended): answers *another draw of EPISODES* only.

    python -X utf8 l1c_roc.py <result_dir>        -> <result_dir>/L1C_ROC.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
L1CODE = Path("C:/Users/Admin/refcv8_pf/refcv9_l1/code")
sys.path.insert(0, str(L1CODE / "l2roc_ref"))
import l2_roc as LR  # noqa: E402  (dv_of / cls_of / auc; imports l2_common from refcv9_l2/code)

C = LR.C
L2_ROC_BANKED = L1CODE / "l2roc_ref" / "L2_ROC.json"
ZERO_GPU = Path("C:/Users/Admin/refcv8_pf/refcv9_l2/result/zero_gpu_nocal/ZERO_GPU.json")
B = 2000


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def reading(lo_hi_by_seed):
    if all(lo > 0 for lo, _ in lo_hi_by_seed):
        return "AUC-UP"
    if all(hi < 0 for _, hi in lo_hi_by_seed):
        return "AUC-DOWN"
    return "AUC-SAME"


def main() -> int:
    d = Path(sys.argv[1])
    banked = json.loads(L2_ROC_BANKED.read_text(encoding="utf-8"))
    zg = json.loads(ZERO_GPU.read_text(encoding="utf-8"))
    out = {"tool": "l1c_roc.py", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "l2_roc_sha256": hashlib.sha256((L1CODE / "l2roc_ref" / "l2_roc.py").read_bytes()).hexdigest(), "seeds": {}}
    for seed in (0, 1):
        E = C.load_eval_pf(seed)
        A = np.load(d / f"arms_S6_s{seed}.npz")
        key = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(E["clip"], E["ws"]))}
        jr = np.array([key.get((int(c), int(w)), -1) for c, w in zip(A["clip"], A["ws"])])
        if (jr < 0).any():
            raise SystemExit(f"[l1-roc] {int((jr < 0).sum())} L1 rows are not in the L2 S6 population")
        m = np.zeros(len(E["ws"]), bool)
        m[jr] = True
        dt, v0 = E["dt"], E["v0"]
        dg = LR.dv_of(E["g"], v0, dt)
        gt = LR.cls_of(dg)
        pos, neg = E["compliant"] & (gt == "decel"), E["compliant"] & (gt == "cruise")
        arms = {"stage": E["stage_banked"], "os": E["os"], "r7": E["r7"]}
        for k in A.files:
            if k.startswith("arm_"):
                P = np.full_like(E["os"], np.nan, dtype=np.float64)
                P[jr] = A[k]
                arms["L1C_" + k[4:]] = P
        dvs = {a: LR.dv_of(P, v0, dt) for a, P in arms.items()}
        dvs["GT_C1"], dvs["CONST_C2"] = dg, np.zeros_like(dg)
        chk = {"population_equals_L1_compliant_rows": bool(np.array_equal(m, E["compliant"])),
               "C5_L1C_OS_equals_panel_os_maxabs_m": float(np.nanmax(np.abs(arms["L1C_OS"][jr] - E["os"][jr].astype(np.float64)))),
               "loader_checks": E["checks"]}
        res = {"n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "checks": chk}
        res["auc"] = {a: LR.auc(x[pos], x[neg]) for a, x in dvs.items()}
        fdmd = {}
        for a in ("stage", "os", "r7"):
            pc = LR.cls_of(dvs[a])
            ref = zg["seeds"][str(seed)]["rate_cells"][f"compliant|{a}"]
            f_, m_ = float((pc[neg] == "decel").mean()), float((pc[pos] != "decel").mean())
            fdmd[a] = {"fd": f_, "md": m_, "C3_match": bool(abs(f_ - ref["fd"]) <= 1e-4 and abs(m_ - ref["md"]) <= 1e-4)}
        for a in arms:
            if a.startswith("L1C_"):
                pc = LR.cls_of(dvs[a])
                fdmd[a] = {"fd": float((pc[neg] == "decel").mean()), "md": float((pc[pos] != "decel").mean())}
        res["fd_md"] = fdmd
        bk = banked["seeds"][str(seed)]
        res["controls"] = {"C1_gt_auc_is_1": res["auc"]["GT_C1"] == 1.0, "C2_const_auc_is_half": res["auc"]["CONST_C2"] == 0.5,
                           "C3_fdmd_reproduce": all(fdmd[a]["C3_match"] for a in ("stage", "os", "r7")),
                           "C4_banked_auc_reproduce": all(res["auc"][a] == bk["auc"][a] for a in ("stage", "os", "r7")),
                           "C5_L1_os_is_panel_os": chk["C5_L1C_OS_equals_panel_os_maxabs_m"] == 0.0,
                           "population": chk["population_equals_L1_compliant_rows"]}
        # paired episode-cluster bootstrap: the L2 loop's draw sequence (rng seed 0), every arm on the same draws
        eid = E["eid"]
        eps = np.unique(eid[pos | neg])
        rng = np.random.default_rng(0)
        idx_by_ep = {e: np.flatnonzero((eid == e) & (pos | neg)) for e in eps}
        names = [a for a in arms]
        D7 = {a: np.empty(B) for a in names}
        DS = {a: np.empty(B) for a in names}
        DH = {a: np.empty(B) for a in names}
        D0 = {a: np.empty(B) for a in names}
        for b in range(B):
            draw = rng.choice(eps, size=len(eps), replace=True)
            ii = np.concatenate([idx_by_ep[e] for e in draw])
            p, n = ii[pos[ii]], ii[neg[ii]]
            au = {a: LR.auc(dvs[a][p], dvs[a][n]) for a in names}
            for a in names:
                D7[a][b] = au[a] - au["r7"]
                DS[a][b] = au[a] - au["stage"]
                DH[a][b] = au[a] - au["L1C_H2cs"]
                D0[a][b] = au[a] - au["L1C_H0"]
        res["paired_vs_r7"] = {a: {"delta": res["auc"][a] - res["auc"]["r7"], "ci95": ci(D7[a])} for a in names if a != "r7"}
        res["paired_vs_stage"] = {a: {"delta": res["auc"][a] - res["auc"]["stage"], "ci95": ci(DS[a])} for a in names if a != "stage"}
        res["paired_vs_L1C_H2cs"] = {a: {"delta": res["auc"][a] - res["auc"]["L1C_H2cs"], "ci95": ci(DH[a])} for a in names
                                     if a.startswith("L1C_") and a != "L1C_H2cs"}
        res["paired_vs_L1C_H0"] = {a: {"delta": res["auc"][a] - res["auc"]["L1C_H0"], "ci95": ci(D0[a])} for a in names
                                   if a.startswith("L1C_") and a != "L1C_H0"}
        res["controls"]["C4_banked_stage_vs_r7_ci_reproduce"] = bool(
            max(abs(x - y) for x, y in zip(res["paired_vs_r7"]["stage"]["ci95"], bk["paired_vs_r7"]["stage"]["ci95"])) <= 1e-12)
        res["n_episodes"] = int(len(eps))
        out["seeds"][str(seed)] = res
        print(f"ZZL1ROCSEED{seed}ZZ " + json.dumps({k: res[k] for k in ("n_pos", "n_neg", "controls")}), flush=True)
    rd = {}
    for a in out["seeds"]["0"]["paired_vs_stage"]:
        if a.startswith("L1C_"):
            rd[a] = {"vs_stage": reading([tuple(out["seeds"][s]["paired_vs_stage"][a]["ci95"]) for s in ("0", "1")]),
                     "vs_r7": reading([tuple(out["seeds"][s]["paired_vs_r7"][a]["ci95"]) for s in ("0", "1")]),
                     "vs_H2cs": (reading([tuple(out["seeds"][s]["paired_vs_L1C_H2cs"][a]["ci95"]) for s in ("0", "1")]) if a != "L1C_H2cs" else None),
                     "vs_H0": (reading([tuple(out["seeds"][s]["paired_vs_L1C_H0"][a]["ci95"]) for s in ("0", "1")]) if a != "L1C_H0" else None)}
    out["readings"] = rd
    out["controls_all_pass"] = all(all(v for v in out["seeds"][s]["controls"].values()) for s in ("0", "1"))
    (d / "L1C_ROC.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"[l1-roc] controls_all_pass={out['controls_all_pass']} H2c={rd.get('L1C_H2c')} H2cs={rd.get('L1C_H2cs')} ZZL1ROCDONEZZ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
