#!/usr/bin/env python3
"""A2 real-data stage 2 (venv driverl-eval): on the scenes built by build_scenes.py
  (1) OFF-IDENTITY  : pre-A2 `score_plans` (pinned tip d5ec25ec) == overlay `score_plans(behaviour=False)` == overlay `score_plans(behaviour=True)` minus its extra key,
                      byte for byte, for every PDM key on every scene; run twice (seeded fan => deterministic).  Deliberate-regression arm: a perturbed copy must NOT match.
  (2) ACTIVATION    : per-term applicability / fired rate over proposals, per-scene discrimination (q varies across the 64 proposals), rates on the PDM-Closed row and on the
                      ground-truth-driving row, and the comfort-detector comparison (NAVSIM-filtered vs the pilot's raw-difference form) on the same states.
  (3) COST          : ms per scene of behaviour_terms vs the PDM scoring it rides on.
  python measure_a2.py --scenes <dir> --out <dir>
"""
import argparse
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
NAVSIM = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
TIP = os.environ.get("A2_TIP_REFE", "F:/a2_scratch/tip_refe")
A2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OV = os.path.join(A2, "overlay", "refe")
sys.path[:0] = [NAVSIM, OV, TIP]                      # overlay before tip: `import behaviour_terms` and the overlay modules win
import numpy as np  # noqa: E402


def load_as(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tip_pt = load_as("pt_tip", os.path.join(TIP, "navsim_pdm_targets.py"))
    ov_pt = load_as("pt_overlay", os.path.join(OV, "navsim_pdm_targets.py"))
    tip_pt._setup(); ov_pt._S.update(tip_pt._S)    # one NAVSIM stack for both modules (a second _setup would trip over the first one's stub)
                          # as the labeller's worker initialiser does (aliases the vendored transform_utils for the DriveRL fork)
    import behaviour_terms as BT
    import online_labeller as OL
    import navsim_dac  # noqa: F401
    states = json.load(open(os.path.join(a.scenes, "states.json")))
    KEYS = ["nc", "dac", "ep", "ttc", "c", "ddc", "pdms"]
    ident = []
    rows_on, rows_ref, rows_gt, cost, per_scene = [], [], [], [], []
    comf_cmp = dict(navsim_all=0, pilot_all=0, n_all=0, navsim_ref=0, pilot_ref=0, n_ref=0, navsim_gt=0, pilot_gt=0, n_gt=0)
    chan = dict(navsim=np.zeros(6, int), pilot=np.zeros(6, int))
    for k, st in enumerate(states):
        pth = os.path.join(a.scenes, "parts", st["log"], f"{st['token']}.pkl.xz")
        if not os.path.isfile(pth):
            continue
        parts = OL.load_parts(pth)
        fan = np.load(os.path.join(a.scenes, f"fan_{k:02d}.npy"))
        plans8 = tip_pt.executed_plans8(fan, repair=True)
        t0 = time.perf_counter()
        r_tip = tip_pt.score_plans(parts, plans8)
        t_base = time.perf_counter() - t0
        r_tip2 = tip_pt.score_plans(parts, plans8)
        r_off = ov_pt.score_plans(parts, plans8)
        t0 = time.perf_counter()
        r_on = ov_pt.score_plans(parts, plans8, behaviour=True)
        t_on = time.perf_counter() - t0
        same_keys = sorted(r_off) == sorted(r_tip) == KEYS or sorted(r_off) == sorted(KEYS)
        eq = {key: bool(np.asarray(r_tip[key]).tobytes() == np.asarray(r_off[key]).tobytes() == np.asarray(r_on[key]).tobytes() == np.asarray(r_tip2[key]).tobytes()) for key in KEYS}
        bad = {kk: np.asarray(r_off[kk]).copy() for kk in KEYS}
        bad["pdms"][0] += 1e-9                                                         # deliberate-regression arm: one ulp-scale perturbation must break equality
        reg_detected = np.asarray(bad["pdms"]).tobytes() != np.asarray(r_tip["pdms"]).tobytes()
        ident.append(dict(scene=k, log=st["log"], keys_equal=eq, key_sets_equal=bool(sorted(r_off) == sorted(KEYS) and sorted(r_tip) == sorted(KEYS)),
                          on_has_behaviour="behaviour" in r_on and "behaviour" not in r_off, regression_arm_detected=bool(reg_detected)))
        b = r_on["behaviour"]
        cost.append(dict(scene=k, base_s=t_base, on_s=t_on, extra_s=t_on - t_base))
        # ---- full sim incl. PDM-Closed (front half of score_plans, as rl_terms_label.executed_states)
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
        S = tip_pt._setup()
        pdm_traj, ego = parts[0], parts[1]
        sts = [get_trajectory_as_array(pdm_traj, S["samp"], ego.time_point)]
        for p in np.asarray(plans8, dtype=np.float32):
            sts.append(get_trajectory_as_array(transform_trajectory(Trajectory(p, S["samp8"]), ego), S["samp"], ego.time_point))
        sim = S["sim"].simulate_proposals(np.stack(sts, 0), ego)
        # ref row as a "proposal" (its own reference row = itself) and the GT-speed/GT-lateral row (index 4*8+2 = 34)
        bt_ref = BT.behaviour_terms(np.concatenate([sim, sim[:1]], 0)[[0, -1]], parts)
        gt_idx = 4 * 8 + 2
        # NAVSIM c identity: q_comf == 1 <=> c == 1 for every proposal
        c_ident = bool(np.array_equal(b["q_comf"] == 1.0, r_on["c"] == 1.0))
        viol_n = BT.comfort_violations(sim[1:])
        viol_p = BT.pilot_style_comfort(sim[1:])
        vr_n, vr_p = BT.comfort_violations(sim[:1]), BT.pilot_style_comfort(sim[:1])
        comf_cmp["navsim_all"] += int(viol_n.any(1).sum()); comf_cmp["pilot_all"] += int(viol_p.any(1).sum()); comf_cmp["n_all"] += len(viol_n)
        comf_cmp["navsim_ref"] += int(vr_n.any()); comf_cmp["pilot_ref"] += int(vr_p.any()); comf_cmp["n_ref"] += 1
        comf_cmp["navsim_gt"] += int(viol_n[gt_idx].any()); comf_cmp["pilot_gt"] += int(viol_p[gt_idx].any()); comf_cmp["n_gt"] += 1
        chan["navsim"] += viol_n.sum(0); chan["pilot"] += viol_p.sum(0)
        ent = dict(scene=k, log=st["log"], loc=st["location"], v0=float(np.hypot(sim[0, 0, 3], sim[0, 0, 4])), c_identity=c_ident,
                   n_free_steps_mean=float(b["n_free_steps"].mean()), n_lead_steps_mean=float(b["n_lead_steps"].mean()))
        for t in BT.ORDER:
            q = b[t]; ac = b["act_" + t[2:]]
            ent[t] = dict(applicable=float(ac.mean()), fired=float((q < 1.0).mean()), discriminates=bool(np.ptp(q) > 1e-9), q_min=float(q.min()), q_ref=float(bt_ref[t][0]),
                          q_gt=float(q[gt_idx]), act_gt=bool(ac[gt_idx]), act_ref=bool(bt_ref["act_" + t[2:]][0]))
        per_scene.append(ent)
        print(f"scene {k} ok  base {t_base:.2f}s  +behaviour {t_on - t_base:.2f}s  identity {all(eq.values())}  c_identity {c_ident}", flush=True)
    # ---- aggregate
    agg = {}
    n = len(per_scene)
    for t in BT.ORDER:
        agg[t] = dict(
            applicable_prop_rate=float(np.mean([e[t]["applicable"] for e in per_scene])),
            fired_prop_rate=float(np.mean([e[t]["fired"] for e in per_scene])),
            scenes_discriminating=int(sum(e[t]["discriminates"] for e in per_scene)), n_scenes=n,
            ref_row_applicable=int(sum(e[t]["act_ref"] for e in per_scene)), ref_row_fired=int(sum(e[t]["q_ref"] < 1.0 for e in per_scene)),
            gt_row_applicable=int(sum(e[t]["act_gt"] for e in per_scene)), gt_row_fired=int(sum(e[t]["q_gt"] < 1.0 for e in per_scene)))
    out = dict(n_scenes=n, identity=ident, identity_all_pass=bool(all(all(i["keys_equal"].values()) and i["key_sets_equal"] and i["on_has_behaviour"] and i["regression_arm_detected"] for i in ident)),
               c_identity_all=bool(all(e["c_identity"] for e in per_scene)), per_scene=per_scene, aggregate=agg, comfort_detector_comparison=comf_cmp,
               comfort_channel_violation_counts={k2: v2.tolist() for k2, v2 in chan.items()}, comfort_channel_order=list(BT.COMFORT_CHANNELS), cost=cost,
               cost_mean=dict(base_s=float(np.mean([c["base_s"] for c in cost])), extra_s=float(np.mean([c["extra_s"] for c in cost]))), constants=BT.C, version=BT.VERSION)
    json.dump(out, open(os.path.join(a.out, "a2_real_results.json"), "w"), indent=1, default=str)
    print(json.dumps({"identity_all_pass": out["identity_all_pass"], "c_identity_all": out["c_identity_all"], "aggregate": agg, "comfort": comf_cmp, "cost_mean": out["cost_mean"]}, indent=1))


if __name__ == "__main__":
    main()
