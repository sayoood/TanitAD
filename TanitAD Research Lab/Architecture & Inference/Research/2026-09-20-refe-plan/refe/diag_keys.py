"""Dump EVERY key the calculators write, for good vs veer. The readout may be the inert part."""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import scorer_gate as G, score_proposals as SP, augment_routes as AR

log = sys.argv[1]
cfg, _ = G.build_engine_config()
calcs = SP.build_calculators(cfg)
sd, log_sd, sc, smps, step, init = G.scenario_data_from_log(log, 40)
sd, st = SP.enrich_lane_graph(sd, sc.map_api, AR._anchor_from_ego(smps[step].ego_state),
                              init.route_roadblock_ids)
good_xy, good_yaw = G.teacher_future(smps, step)
veer_xy = good_xy.clone(); veer_xy[:, 1] += torch.linspace(0, 25.0, veer_xy.shape[0])

def run(xy, yw):
    cand = SP.inject_proposal(sd, xy, yw)
    shared = {}
    for n in SP.CALC_ORDER:
        try: calcs[n].forward(cand, log_sd, shared)
        except Exception as e: shared[f"{n}!ERR"] = str(e)[:80]
    return shared

A, B = run(good_xy, good_yaw), run(veer_xy, good_yaw)

def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        p = f"{pre}{k}"
        if isinstance(v, dict): out.update(flat(v, p + "."))
        else: out[p] = v
    return out

fa, fb = flat(A), flat(B)
print(f"{'key':44s} {'good[0,0]':>12s} {'veer[0,0]':>12s}  DIFFERS  (ego)   any-agent-differs")
ndiff = 0
for k in sorted(set(fa) | set(fb)):
    va, vb = fa.get(k), fb.get(k)
    if torch.is_tensor(va) and torch.is_tensor(vb):
        ea, eb = SP._ego_scalar(va), SP._ego_scalar(vb)
        de = abs(ea - eb) > 1e-9
        da = bool((va.float() - vb.float()).abs().max() > 1e-9) if va.shape == vb.shape else True
        ndiff += int(de)
        print(f"{k:44s} {ea:12.4f} {eb:12.4f}  {'YES' if de else ' no':>7s}       {'YES' if da else 'no'}"
              f"   {tuple(va.shape)}")
    else:
        print(f"{k:44s} {str(va)[:24]:>12s} {str(vb)[:24]:>12s}  (non-tensor)")
print(f"\nEGO-LEVEL DIFFERING KEYS: {ndiff}")
