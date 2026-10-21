#!/usr/bin/env python3
"""PAIRED four-family deltas between two NavSim seams on the SAME tokens (B - A), the binding rule's estimator form.

Nothing here is re-derived: the per-window components are ``refav1_arm._components`` (four_families' own ``_seq_geometry`` +
``maneuver_kinematics`` + the programme's ``factor_from_kinematics`` trajectory labeller), the interval is ``refav1_arm.
_paired_families`` (``taniteval.ci.paired_episode_cluster_bootstrap``), and the win dict is the NavSim adapter's ``scenes_to_win``
with ``log_names`` so the cluster unit is the OpenScene LOG (the settled NavSim unit).  dt 0.5 s, frame ego, origin excluded
(the NavSim agent output contract, exactly as families7.py).

Families: ADE (ade/fde), LONGITUDINAL (speed / along-track / accel MAE), LATERAL (cross-track / heading / yaw-rate MAE),
TACTICAL (trajectory-labelled lat + lon manoeuvre agreement with the human).  STRATEGIC: NavSim scores none and refcv8 has
no strategic output head on this bridge -> UNAVAILABLE with reason and n.  Distance-keeping: no lead track on the NavSim
bridge -> UNAVAILABLE with reason and n (the adapter's own refusal).

Controls (all must pass or the file says INCOMPLETE):
  P1 self-pair: A - A is exactly 0 on every component, interval [0, 0];
  P2 analytic: the HUMAN future fed as the prediction scores 0 on every error component and 1 on both agreements (literal);
  P3 the two seams' human futures are the same array (same tokens, same GT) -- checked bit-exact.

    python paired_families_navsim.py --inputs navtest_inputs.json.gz --seam-a seam_A.npz --seam-b seam_B.npz \
        --label "..." --out paired.json
"""
import argparse
import gzip
import importlib.util
import json
import os
import sys

import numpy as np

D = "D:/Projects/TanitAD"
for p in (f"{D}/stack", f"{D}/taniteval", f"{D}/taniteval/tools"):
    if p not in sys.path:
        sys.path.append(p)
ADAPTER = f"{D}/taniteval/adapters/navsim.py"


def load_seam(path):
    z = np.load(path, allow_pickle=False)
    src = [str(s) for s in z["source"]] if "source" in z.files else ["precomputed"] * len(z["token"])
    idx = {str(t): i for i, t in enumerate(z["token"]) if src[i] != "cv_standin"}
    return str(z["arm"]), z["poses"], idx


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True); ap.add_argument("--seam-a", required=True); ap.add_argument("--seam-b", required=True)
    ap.add_argument("--name-a"); ap.add_argument("--name-b")
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    spec = importlib.util.spec_from_file_location("navsim_adapter_pf", ADAPTER)
    ad = importlib.util.module_from_spec(spec); sys.modules["navsim_adapter_pf"] = ad; spec.loader.exec_module(ad)
    import refav1_arm as ra
    doc = json.load(gzip.open(a.inputs, "rt", encoding="utf-8"))["tokens"]
    arm_a, Pa, ia = load_seam(a.seam_a)
    arm_b, Pb, ib = load_seam(a.seam_b)
    na, nb = a.name_a or arm_a, a.name_b or arm_b
    if na == nb:
        nb = nb + "_B"
    toks = sorted(t for t in set(ia) & set(ib) if int(doc[t].get("stage", 1)) == 1 and doc[t].get("human_future_poses") is not None)
    human = np.asarray([doc[t]["human_future_poses"] for t in toks], np.float64)
    v0 = [float(np.hypot(*doc[t]["ego_statuses"][-1]["ego_velocity"])) for t in toks]
    logs = [doc[t]["log_name"] for t in toks]
    kw = dict(frame="ego", origin_included=False, dt_s=0.5, scene_tokens=toks, ego_speed_mps=v0, log_names=logs)
    wa = ad.scenes_to_win(np.asarray([Pa[ia[t]] for t in toks], np.float64), human, **kw)
    wb = ad.scenes_to_win(np.asarray([Pb[ib[t]] for t in toks], np.float64), human, **kw)
    wh = ad.scenes_to_win(human.copy(), human, **kw)
    dt = float(wa["dt_s"])
    G = np.asarray(wa["gt_dense"], np.float64)
    comps = {na: ra._components(np.asarray(wa["pred_dense"], np.float64), G, dt),
             nb: ra._components(np.asarray(wb["pred_dense"], np.float64), G, dt),
             "HUMAN": ra._components(np.asarray(wh["pred_dense"], np.float64), G, dt)}
    # CURVATURE (binding: LATERAL names it) + a PAIR-MASKED yaw rate, both on four_families' own _seq_geometry and its own mask
    # (both_pair = pred pair_valid & gt pair_valid, four_families.py lateral()); a window with no valid pair is NaN and the paired
    # bootstrap drops it for BOTH arms.  Reducer: per-window mean, then mean over windows (four_families pools steps; stated).
    import torch
    from taniteval import four_families as ff
    Gg = ff._seq_geometry(torch.as_tensor(G).float(), dt)
    for k, w in ((na, wa), (nb, wb), ("HUMAN", wh)):
        Pg = ff._seq_geometry(torch.as_tensor(np.asarray(w["pred_dense"], np.float64)).float(), dt)
        bp = Pg["pair_valid"] & Gg["pair_valid"]
        nv = bp.sum(1)
        for nm, key in (("LAT_curvature_mae_1pm", "curvature"), ("LAT_yaw_rate_mae_masked_radps", "yaw_rate")):
            comps[k][nm] = torch.where(nv > 0, ((Pg[key] - Gg[key]).abs() * bp).sum(1) / nv.clamp_min(1), torch.nan).numpy()
    ra._FAMILY_OF = {**ra._FAMILY_OF, "LAT_curvature_mae_1pm": "lateral", "LAT_yaw_rate_mae_masked_radps": "lateral"}
    tiers = {na: "T1", nb: "T1", "HUMAN": "T1"}
    eid = list(wa["eid"])
    res = {"_label": a.label, "direction": f"{nb} - {na}", "n_tokens": len(toks), "n_log_clusters": len(set(eid)),
           "tier": "NavSim v1 open loop, single query per scene (the families' T1 stamp is the harness tier of a model-own plan)",
           "dt_s": dt, "seam_a": os.path.abspath(a.seam_a), "seam_b": os.path.abspath(a.seam_b),
           "means": {k: {m: round(float(np.nanmean(v)), 6) for m, v in comps[k].items()} for k in (na, nb)},
           "paired": ra._paired_families(comps, na, nb, eid, tiers, a.n_boot, a.seed),
           "unavailable": {
               "strategic": {"status": "UNAVAILABLE", "n": len(toks), "reason": "NavSim scores no strategic decision and the refcv8 NavSim "
                             "bridge exports no strategic output; the PhysicalAI battery carries the strategic family"},
               "longitudinal.distance_keeping": {"status": "UNAVAILABLE", "n": len(toks), "reason": "no lead-agent track on the NavSim "
                                                 "bridge (taniteval/adapters/navsim.py lead_block); a WORK ITEM, not a pass"}},
           "controls": {}}
    c = res["controls"]
    comps[na + "_self"] = comps[na]
    sp = ra._paired_families(comps, na, na + "_self", eid, {**tiers, na + "_self": "T1"}, 200, 0)
    flat = [r for fam in sp["families"].values() for r in fam.values()]
    c["P1_self_pair_zero"] = all(abs(float(r.get("delta", r.get("mean_diff", 1.0)))) == 0.0 for r in flat)
    hm = {m: float(np.nanmean(v)) for m, v in comps["HUMAN"].items()}
    err = [m for m in hm if not m.startswith("TAC_")]
    c["P2_human_analytic"] = {"errors": {m: round(hm[m], 9) for m in err}, "agreements": {m: hm[m] for m in hm if m.startswith("TAC_")},
                              "pass": all(abs(hm[m]) < 1e-6 for m in err) and all(hm[m] == 1.0 for m in hm if m.startswith("TAC_"))}
    c["P3_same_gt"] = bool(np.array_equal(np.asarray(wa["gt_dense"]), np.asarray(wb["gt_dense"])))
    c["all_pass"] = bool(c["P1_self_pair_zero"] and c["P2_human_analytic"]["pass"] and c["P3_same_gt"])
    res["status"] = "OK" if c["all_pass"] else "INCOMPLETE (a control failed)"
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    rows = []
    for fam, blk in res["paired"]["families"].items():
        for m, r in blk.items():
            rows.append(f"{fam}.{m} d={r.get('delta')} [{r.get('lo')}, {r.get('hi')}]")
    print("QQPF " + json.dumps({"status": res["status"], "n": len(toks), "logs": len(set(eid)), "controls": {k: (v if not isinstance(v, dict) else v.get("pass")) for k, v in c.items()}}) + " QQ")
    print("\n".join(rows))
    return 0 if c["all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
