#!/usr/bin/env python3
"""sm5_diag.py -- SM5's zero-GPU diagnostic (PREREG_S1ZSM_P6 section 5): which scorer head (NC, DAC, EP, TTC) produces the extra CRUISE slow picks of the S1Z-family arms?
Descriptive, decides nothing, no arm, no K-ledger key. CPU only.

Metric: E2 CRUISE slow-pick share = share of CRUISE tokens of E2 whose pick has plan v4/v0 < 0.85 (read_s1z.py line 312 definition).
Attribution: read_scorer1p6.shapley (the reader's own exact Shapley over {NC,DAC,EP,TTC}, coalition heads' probabilities replaced by the proposal's PDM truth sub-score, pick re-taken
by the same rule) is called UNCHANGED; the metric is swapped by handing it the slow indicator in place of PDMS (value = 100 * mean over tokens of the indicator of the re-taken pick).
Per-token Shapley values are obtained by calling it on one token at a time (Shapley is linear in the per-token values), so the reader's log-cluster bootstrap (BootSet, 10,000 draws,
seed 20260927, same draws for every arm) gives CIs for arm - P6.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

P6BASE = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base"
for _p in (P6BASE + "/s1z/read", P6BASE + "/eval"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import read_s1z as S  # noqa: E402
R = S.R

HEADS = ("NC", "DAC", "EP", "TTC")
SLOW_RATIO = S.SLOW_RATIO


def token_values(L, sub, valid, slow_mat, tokens):
    """per token i in `tokens`: v_empty, v_all, shapley[h], single-swap[h] (slow indicator x100 of the re-taken pick), via R.shapley unchanged"""
    out = {k: np.zeros(len(L)) for k in ("v_empty", "v_all")}
    sh = {h: np.zeros(len(L)) for h in HEADS}
    one = {h: np.zeros(len(L)) for h in HEADS}
    for i in tokens:
        s = slice(i, i + 1)
        r = R.shapley(L[s], sub[s], slow_mat[s], valid[s])
        out["v_empty"][i], out["v_all"][i] = r["v_empty"], r["v_all"]
        for h in HEADS:
            sh[h][i] = r["shapley_points"][h]
            one[h][i] = R.shapley(L[s], sub[s], slow_mat[s], valid[s], heads=(R.ALL_HEADS[h],))["v_all"] - r["v_empty"]
    return out, sh, one


def diag(L2_by_arm, sub, valid, ratio2, cru, log, pdms_truth, ref="P6"):
    """core, I/O free. L2_by_arm: name -> [n2,64,6] logits of the E2 tokens. -> dict"""
    n2 = len(cru)
    slow_mat = (ratio2 < SLOW_RATIO).astype(np.float64)
    B = R.BootSet(log)
    tok = np.where(cru)[0]
    res = {"n_E2": int(n2), "n_cruise": int(cru.sum()), "arms": {}}
    per = {}
    for nm, L in L2_by_arm.items():
        L = np.asarray(L)
        pk = R.pick_logits(L)
        own = slow_mat[np.arange(n2), pk]
        v, sh, one = token_values(L, sub, valid, slow_mat, tok)
        share = B.stat(own, cru)
        a = {"slow_share_cruise": R.rec(share[0], share[1], share[2]), "slow_share_point": float(own[cru].mean()),
             "v_empty_x100_point": float(v["v_empty"][cru].mean()), "v_all_x100_point": float(v["v_all"][cru].mean())}
        per[nm] = {"own": own, "v_empty": v["v_empty"], "v_all": v["v_all"], "sh": sh, "one": one}
        a["shapley_x100"] = {h: float(sh[h][cru].mean()) for h in HEADS}
        a["single_swap_x100"] = {h: float(one[h][cru].mean()) for h in HEADS}
        a["shapley_sum"] = float(sum(a["shapley_x100"].values()))
        a["efficiency_abs_err"] = abs(a["shapley_sum"] - (a["v_all_x100_point"] - a["v_empty_x100_point"]))
        a["C2_empty_vs_own_abs_err_x100"] = abs(a["v_empty_x100_point"] - 100.0 * a["slow_share_point"])
        a["pdms_shapley_E2"] = R.shapley(L, sub, pdms_truth, valid)
        res["arms"][nm] = a
    res["arm_minus_ref"] = {}
    for nm in per:
        if nm == ref:
            continue
        pa, pr = per[nm], per[ref]
        res["arm_minus_ref"][nm] = {
            "slow_share_diff": R.rec(*B.diff(pa["own"], pr["own"], cru)),
            "shapley_x100": {h: R.rec(*B.diff(pa["sh"][h], pr["sh"][h], cru)) for h in HEADS},
            "single_swap_x100": {h: R.rec(*B.diff(pa["one"][h], pr["one"][h], cru)) for h in HEADS},
            "v_empty_x100": R.rec(*B.diff(pa["v_empty"], pr["v_empty"], cru)),
            "v_all_x100": R.rec(*B.diff(pa["v_all"], pr["v_all"], cru))}
    return res


def all_zero(d):
    flat = [d["slow_share_diff"], d["v_empty_x100"], d["v_all_x100"]] + list(d["shapley_x100"].values()) + list(d["single_swap_x100"].values())
    return bool(all(f["point"] == 0.0 and f["lo"] == 0.0 and f["hi"] == 0.0 for f in flat))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-free-ram-gb", type=float, default=3.0)
    a = ap.parse_args(argv)
    S.wait_free_ram(a.min_free_ram_gb, timeout_s=600)
    t0 = time.time()
    armdirs = {"S1ZSM": "F:/tanitad_scratch/p6arms/pass_s1zsm/S1ZSM", "S1Z": "F:/tanitad_scratch/p6arms/pass_s1z/S1Z",
               "S1Z1": "F:/tanitad_scratch/p6arms/pass_rep_c0r2_s1e1_s1z1/S1Z1"}
    ns = argparse.Namespace(p6_proposals=R.DEFAULT_P6_PROPOSALS, p6_csv=R.DEFAULT_P6_CSV, e2_tokens=S.DEFAULT_E2_TOKENS, e2_table=S.DEFAULT_E2_TABLE, export=R.DEFAULT_EXPORT,
                            world_dump=None, arm=[f"{k}={v}" for k, v in armdirs.items()], ens3_csv=None, c3_tokens=None, no_tactical=True)
    inp = R.inputs_from_files(ns)
    W = inp.world
    wi2 = np.array([W.index[t] for t in inp.e2_tokens])
    tpos = {t: i for i, t in enumerate(inp.e2_table["token"])}
    ti = np.array([tpos[t] for t in inp.e2_tokens])
    T_pd, T_sub, T_val = inp.e2_table["pdms"][ti], inp.e2_table["sub"][ti], inp.e2_table["valid"][ti]
    ratio2 = R.plan_v4(inp.p6_proposals[wi2]) / np.maximum(W.v0[wi2], 1e-6)[:, None]
    cru = W.cruise()[wi2]
    L2 = {"P6": inp.p6_logits[wi2]}
    prop_ok = {}
    for nm, arm in inp.arms.items():
        ix, err = R._align(arm.e3_token, list(W.token), nm)
        assert err is None, err
        L2[nm] = np.asarray(arm.e3_logits)[ix][wi2]
        prop_ok[nm] = R.bits_equal(np.asarray(arm.e3_proposals)[ix][wi2], inp.p6_proposals[wi2])
    assert all(prop_ok.values()), prop_ok
    L_all = dict(L2)
    L_all["MUT_P6_in_S1ZSM_slot"] = L2["P6"].copy()
    res = diag(L_all, T_sub, T_val, ratio2, cru, W.log[wi2], T_pd)
    res["inputs"] = dict(inp.provenance)
    res["proposals_bit_identical_to_P6_on_E2"] = prop_ok
    arms = res["arms"]
    ctl = {}
    ctl["C1_reads"] = {"S1ZSM": {"got": arms["S1ZSM"]["slow_share_point"], "read": 0.2132}, "P6": {"got": arms["P6"]["slow_share_point"], "read": 0.1783}}
    ctl["C1_PASS"] = bool(round(arms["S1ZSM"]["slow_share_point"], 4) == 0.2132 and round(arms["P6"]["slow_share_point"], 4) == 0.1783)
    ctl["C2_empty_vs_own_max_abs_err_x100"] = max(x["C2_empty_vs_own_abs_err_x100"] for x in arms.values())
    ctl["C2_v_all_by_arm_x100"] = {k: x["v_all_x100_point"] for k, x in arms.items()}
    ctl["C2_v_all_spread_x100"] = float(max(ctl["C2_v_all_by_arm_x100"].values()) - min(ctl["C2_v_all_by_arm_x100"].values()))
    ctl["C3_max_efficiency_abs_err"] = max(x["efficiency_abs_err"] for x in arms.values())
    ctl["C3_PASS"] = bool(ctl["C3_max_efficiency_abs_err"] < 1e-9)
    ctl["C4_all_fields_exactly_zero"] = all_zero(res["arm_minus_ref"]["MUT_P6_in_S1ZSM_slot"])
    ref = {"v_empty": 92.26824903961263, "v_all": 98.12279047169775,
           "shapley_points": {"NC": 0.6663822803567498, "DAC": 1.1587245971888813, "EP": 1.7094648720465149, "TTC": 2.3199696824929816}}
    ctl["item4_read_shapley_E2"] = ref
    ctl["item4_recomputed_PDMS_shapley_E2"] = {k: arms[k]["pdms_shapley_E2"] for k in ("P6", "S1ZSM", "S1Z", "S1Z1")}
    ctl["item4_belongs_to"] = [k for k in ("P6", "S1ZSM", "S1Z", "S1Z1") if abs(arms[k]["pdms_shapley_E2"]["v_empty"] - ref["v_empty"]) < 1e-9
                               and all(abs(arms[k]["pdms_shapley_E2"]["shapley_points"][h] - ref["shapley_points"][h]) < 1e-9 for h in HEADS)]
    try:
        Z = np.load("F:/tanitad_scratch/p6arms/pass_s1zsm/P6_control/e3_logits.npz", allow_pickle=False)
        ix, err = R._align(Z["token"], list(W.token), "P6_control")
        ctl["P6_control_logits_bit_equal_to_P6"] = None if err else R.bits_equal(np.asarray(Z["logits"])[ix], inp.p6_logits)
    except Exception as e:  # noqa: BLE001
        ctl["P6_control_logits_bit_equal_to_P6"] = f"not checked: {e}"
    res["controls"] = ctl
    res["meta"] = {"tier": "E2 selection tier (2,000 tokens, CRUISE subset)", "evidence_class": "MEASURED (ours; artifact paths in inputs)",
                   "estimator": "paired log-cluster bootstrap over E2's logs, 10,000 draws, seed 20260927, 95 % percentile (read_scorer1p6.BootSet); same draws for every arm; Shapley = read_scorer1p6.shapley unchanged",
                   "status": "descriptive, decides nothing, adds no arm and no K-ledger key", "seconds": round(time.time() - t0, 1)}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "sm5_diag.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=R.jdefault)
    print(json.dumps({"controls": ctl, "seconds": res["meta"]["seconds"]}, indent=1, default=R.jdefault))
    return 0


if __name__ == "__main__":
    sys.exit(main())
