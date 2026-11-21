"""N4-DAC Q1: anatomy of the tracked-DAC failures on navtrain (469 tokens / 13 logs). Per failing (arm.form, token): where, when, how far, why."""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
import n4_geom as G, n4_common as N
from shapely.geometry import Point
from shapely.ops import nearest_points

Rl, S = N.load_rolls(), N.load_scores()
common = sorted(set.intersection(*[set(v) for v in S.values()]))
RUNS = N.RUNS
fail_tok = sorted({t for r in RUNS for f in "RC" for t in common if S[r][t]["arms"][f]["tracked"]["dac"] == 0})
print("failing tokens (any R or C label, any arm):", len(fail_tok))
by = {}
for t in fail_tok:
    by.setdefault(S["RJ5"][t]["log"], []).append(t)
out, ctl = {}, {"banked_dac_reproduced": 0, "banked_dac_total": 0, "mismatch": []}
t0 = time.time()
for lg, tk, cx in G.contexts(by):
    rec = {"log": lg, "arms": {}}
    for run in RUNS:
        t20 = np.asarray(Rl[run][tk]["traj20"], float)
        L = cx.labels(t20)
        res, sim = cx.score([L["T"], L["C"], L["R"]])
        tr_T = cx.transparent(L["T"])
        d_teacher_untracked = cx.ext_dist(G.TS.transparent_states(L["T"], cx.ego, cx.s0, 0.2))
        for i, f in enumerate("TCR"):
            ctl["banked_dac_total"] += 1
            b = S[run][tk]["arms"][f]["tracked"]["dac"]
            if abs(res[i]["dac"] - b) < 1e-9 and abs(res[i]["pdms"] - S[run][tk]["arms"][f]["tracked"]["pdms"]) < 1e-6:
                ctl["banked_dac_reproduced"] += 1
            else:
                ctl["mismatch"].append([run, f, tk, res[i]["dac"], b])
            if res[i]["dac"] == 1.0:
                continue
            d = cx.ext_dist(sim[i])
            viol = np.where(d > 0)[0]
            k = int(viol[0]); kmax = int(np.argmax(d))
            # direction back inside from the worst corner, in the ego heading frame at that step
            from navsim_pdm.pdm_array_representation import state_array_to_coords_array
            cs = state_array_to_coords_array(sim[i][None], cx.vp)[0][kmax, :4]
            far = max(cs, key=lambda p: cx.drivable.distance(Point(*p)))
            near = nearest_points(cx.drivable, Point(*far))[0]
            v = np.array([near.x - far[0], near.y - far[1]])
            h = sim[i][kmax, 2]
            vlon, vlat = v[0] * np.cos(h) + v[1] * np.sin(h), -v[0] * np.sin(h) + v[1] * np.cos(h)
            tw = cx.transparent(L[f])
            r = {"run": run, "form": f, "first_violation_step": k, "first_violation_t_s": round(k * 0.1, 1), "max_ext_m": round(float(d.max()), 3),
                 "max_ext_t_s": round(kmax * 0.1, 1), "n_violating_steps": int(len(viol)),
                 "return_vec_in_heading_frame_m": [round(float(vlon), 3), round(float(vlat), 3)],
                 "label_untracked_dac": tw["dac"],
                 "teacher_T_tracked_dac": res[0]["dac"], "teacher_T_untracked_dac": tr_T["dac"],
                 "teacher_T_untracked_max_ext_m": round(float(d_teacher_untracked.max()), 3),
                 "tracked_vs_label_xy_dev_at_viol_m": None}
            # tracker deviation at the worst step: simulated executed pose vs the label's own pose at the same time
            lab8 = G.ND.to_navsim(L[f]); trk8 = G.tracked8(G.ND, lab8[None], cx.ego)[0]
            r["tracker_dev8_max_m"] = round(float(np.linalg.norm(trk8[:, :2] - lab8[:, :2], axis=1).max()), 3)
            r["label_minus_teacher_xy_max_m"] = round(float(np.linalg.norm(L[f][:, :2] - L["T"][:, :2], axis=1).max()), 3)
            r["passes_elsewhere"] = None
            rec["arms"][f"{run}.{f}"] = r
    out[tk] = rec
    print(tk, "done", round(time.time() - t0, 1), flush=True)
# where do alternatives pass on each failing instance
for tk, rec in out.items():
    for key, r in rec["arms"].items():
        r["passes_elsewhere"] = sorted(f"{a}.{b}" for a in RUNS for b in "TCR" if S[a][tk]["arms"][b]["tracked"]["dac"] == 1.0 and f"{a}.{b}" != key)
json.dump({"control": ctl, "tokens": out}, open("Q1_anatomy.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(ctl)[:400])
