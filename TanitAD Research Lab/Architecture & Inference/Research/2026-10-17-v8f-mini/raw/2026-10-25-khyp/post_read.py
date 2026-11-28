"""post_read.py -- POST-HOC, DESCRIPTIVE, runs AFTER the PREREG's outcome was fixed by `khyp.py read`. It changes no class set, feature, regularisation, selector or bar and
it cannot change the outcome; nothing here is promoted without its own pre-registration. Three things, all from the already-banked hyps/*.npz (no model, no checkpoint):

  1. the distance-keeping family (headway / time-gap / min-TTC / LEAD-VIOL) through the programme's own reader, the one beside row `khyp.py read` could not produce:
     the reader REFUSED on its 1e-5 m `TOL_ADE` (float32 vs the banked ade6: 1.35e-5 m on seed 0, while the mean ADE6 agrees to 8e-8 m). Here TOL_ADE is raised to 1e-4 m
     (DECLARED DEVIATION in a frame/units sanity check only; every number the reader computes is unchanged). Written to dk_read_tol1e4.json.
  2. CONTROL 3 FAILED on seed 0 (SEL-LEARN fitted on PERMUTED labels is indistinguishable from SEL-LEARN). The cheapest discriminating look: ADE6 / speed MAE / accel MAE of
     the seven CONSTANT selectors (always class k) vs B, paired, and SEL-LEARN vs SEL-LEARN_PERM. If a constant class explains SEL-LEARN's seed-0 gain, it is not selection.
  3. reproduction check that this script rebuilds the same SEL-LEARN picks as `khyp.py read` (pick counts must equal READ.json's).
Output: posthoc.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import khyp as K  # noqa: E402
import numpy as np  # noqa: E402

D = K.D
EV, BE = K._eval_modules()
NB = 2000
a = argparse.Namespace(no_banked=False, eval_grid=str(OUT / "raw" / "eval_grid_windows.json"), eval_kin=str(OUT / "raw" / "eval141_kin.npz"),
                       eval_join=str(OUT / "raw" / "join_eval141.npz"), kin_mode="yaw")
grid = json.load(open(a.eval_grid))
J = D.load_join(a.eval_join)
X = D.eval_context(grid, a.eval_kin, J, kin_mode="yaw")
gt, eid = X["gt"].astype(np.float64), X["eid"]
READ = json.load(open(OUT / "READ.json"))

P = {"reproduction_vs_READ_json": {}, "constant_selectors_minus_B": {}, "LEARN_minus_LEARN_PERM": {}, "kstar_vs_constant": {}}
PLANS = {}
for s in K.SEEDS:
    ev = K.load_hyps(OUT / "hyps" / f"M2b_s{s}_eval.npz")
    tr = K.load_hyps(OUT / "hyps" / f"M2b_s{s}_train.npz")
    ktr = K.oracle_k(K.ade6_per_hyp(tr["tau"], tr["gt"]))
    kev = K.oracle_k(K.ade6_per_hyp(ev["tau"], gt))
    Xtr, Xev = K.hyp_feature_matrix(tr), K.hyp_feature_matrix(ev)
    khat = K.predict_learn(K.fit_learn(Xtr, ktr), Xev)
    kperm = K.predict_learn(K.fit_learn(Xtr, K.permute_labels(ktr, 0)), Xev)
    kart = K.argmax_k(ev["p_lon"])
    want = READ["per_seed"][f"s{s}"]["selector_accuracy"]
    got = {"pick_counts_LEARN": np.bincount(khat, minlength=K.K).tolist(), "pick_counts_ARGMAX": np.bincount(kart, minlength=K.K).tolist(),
           "kstar_eval_counts": np.bincount(kev, minlength=K.K).tolist()}
    P["reproduction_vs_READ_json"][f"s{s}"] = {**{k: {"mine": v, "read": want[k], "equal": v == want[k]} for k, v in got.items()}, "pick_counts_LEARN_PERM": np.bincount(kperm, minlength=K.K).tolist()}
    B = ev["own"].astype(np.float64)
    PLANS[s] = {"B": B, "ORACLE": K.pick(ev["tau"], kev), "ARGMAX": K.pick(ev["tau"], kart), "MIX": K.plan_mix(ev["tau"], ev["p_lon"]), "LEARN": K.pick(ev["tau"], khat)}
    pwB = K.arm_pw(B, gt, BE)
    rows = {}
    for k in range(K.K):
        pw = K.arm_pw(ev["tau"][:, k].astype(np.float64), gt, BE)
        rows[f"always_{k}"] = {m: K.prow(pw[m], pwB[m], eid, BE, NB, m) for m in ("ade6", "fw_lon_speed_mae", "fw_lon_accel_mae")}
        rows[f"always_{k}"]["share_of_windows_with_kstar_eq_k"] = float((kev == k).mean())
    P["constant_selectors_minus_B"][f"s{s}"] = rows
    pwL, pwP = K.arm_pw(PLANS[s]["LEARN"], gt, BE), K.arm_pw(K.pick(ev["tau"], kperm), gt, BE)
    P["LEARN_minus_LEARN_PERM"][f"s{s}"] = {m: K.prow(pwL[m], pwP[m], eid, BE, NB, m) for m in ("ade6", "ade2", "fw_lon_speed_mae", "fw_lon_accel_mae")}
    P["kstar_vs_constant"][f"s{s}"] = {"majority_kstar_class_on_train": int(np.bincount(ktr, minlength=K.K).argmax()), "LEARN_pick_equals_majority_share": float((khat == np.bincount(ktr, minlength=K.K).argmax()).mean()),
                                       "LEARN_acc": float((khat == kev).mean()), "majority_acc": float((kev == np.bincount(ktr, minlength=K.K).argmax()).mean())}


def dk_tol1e4():
    import importlib.util
    sp = importlib.util.spec_from_file_location("v8f_dk_read", str(OUT / "raw" / "v8f_dk_read.py"))
    dk = importlib.util.module_from_spec(sp)
    sys.modules["v8f_dk_read"] = dk
    sp.loader.exec_module(dk)
    dk.TOL_ADE = 1e-4                                                          # the declared deviation (frame / units sanity check only)
    order = list(grid["order"])
    preds, saved = {}, {}
    for s in K.SEEDS:
        for arm in ("B", "ORACLE", "ARGMAX", "MIX", "LEARN"):
            preds[f"{arm}_s{s}"] = PLANS[s][arm]
        saved[f"B_s{s}"] = np.load(f"{K.SCR}/hblend/inputs/banked/eval_M2b_s{s}-kind-own.npz")["ade6"]
    preds["P10"] = np.load(f"{K.SCR}/sfresh/inputs/p10/p10_eval_pred6.npy").astype(np.float64)
    fl = np.load(f"{K.SCR}/hblend/inputs/full_eval_floors_perwindow.npz")["P10_refit__ade6"]
    blk = dk.load_block(f"{K.SCR}/hblend/inputs/b1_eval_lead_block_k30.npz")
    pairs = [(f"{arm}_s{s}", f"B_s{s}") for s in K.SEEDS for arm in ("ORACLE", "ARGMAX", "MIX", "LEARN")]
    res = dk.read_dk(preds=preds, gt=gt, eid=X["eid"], clip=np.array([order[e] for e in X["eid"]]), frame=2 * np.asarray(X["t"], np.int64),
                     v0=np.asarray(X["kin"][:, 0], np.float64), block=blk, k=30, dt=K.DT, n_boot=NB, seed=0, saved_ade6=saved, p10_floor_ade6=fl, pairs=pairs)
    K.jdump(dk._clean(res["read"]), OUT / "dk_read_tol1e4.json")
    r = res["read"]
    out = {"status": "OK", "TOL_ADE_used_m": 1e-4, "artifact": str(OUT / "dk_read_tol1e4.json"), "arms": {}, "pairs": {}}
    for nm, ab in r["arms"].items():
        m = ab["metrics"]
        out["arms"][nm] = {"headway_min_m": m["headway_min_m"].get("mean"), "time_gap_min_s": m["time_gap_min_s"].get("mean"), "min_ttc_s": m["min_ttc_s"].get("mean"),
                           "lead_viol_share_of_at_risk": ab["lead_viol"].get("share_of_at_risk")}
    out["pairs_raw"] = dk._clean(r.get("pairs", {}))
    return out


try:
    P["distance_keeping_tol1e4"] = dk_tol1e4()
except Exception as e:                                                               # recorded, not swallowed into a pass
    import traceback
    P["distance_keeping_tol1e4"] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
K.jdump(P, OUT / "posthoc.json")
print("posthoc.json written; dk status:", P["distance_keeping_tol1e4"].get("status"), P["distance_keeping_tol1e4"].get("error", ""))
for s in K.SEEDS:
    print(f"s{s} repro:", {k: v["equal"] for k, v in P["reproduction_vs_READ_json"][f"s{s}"].items() if isinstance(v, dict)})
    for k in range(K.K):
        r = P["constant_selectors_minus_B"][f"s{s}"][f"always_{k}"]["ade6"]
        print(f"s{s} always_{k}: ADE6-B {K.fmt(r)}")
    print(f"s{s} LEARN-LEARN_PERM ade6:", K.fmt(P["LEARN_minus_LEARN_PERM"][f"s{s}"]["ade6"]))
