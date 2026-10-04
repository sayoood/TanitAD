"""D6 -- does the PDM-Closed reference proposal of a scene pass the drivable-area test?   (map-only; NAVSIM VENV; CPU)

    PYTHONPATH=... python d6_ref_dac.py <hooks.json (token -> log_name)> <tokens file> <out.json>

The reference proposal is ``metric_cache.trajectory`` simulated by the devkit tracker and tested with the same ``PDMScorer`` DAC test as any plan --
exactly the ``ref_dac`` that ``d6_rescore.py`` read from proposal index 0 of the official two-proposal call (validated against it on the 30k DAC-zero scenes).
"""
import json
import lzma
import pickle
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import d6_rescore as R                                              # noqa: E402
from navsim.evaluate.pdm_score import get_trajectory_as_array        # noqa: E402

hooks_path, tokens_path, out_path = sys.argv[1:4]
hooks = {c["token"]: c for c in json.load(open(hooks_path, encoding="utf-8"))["pdm_score_calls"]}
toks = [l.strip() for l in open(tokens_path) if l.strip()]
ps, simulator, scorer, policy = R.build_objects()
out = {}
for t in toks:
    with lzma.open(R.cache_path(hooks[t]["log_name"], t), "rb") as f:
        mc = pickle.load(f)
    ini = mc.ego_state
    arr = get_trajectory_as_array(mc.trajectory, ps, ini.time_point)
    sim = simulator.simulate_proposals(arr[None], ini)
    r = scorer.score_proposals(sim, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map, mc.map_parameters, None, mc.past_human_trajectory)[0]
    out[t] = float(r["drivable_area_compliance"].iloc[0])
json.dump(out, open(out_path, "w"))
print(len(out), "scenes; ref DAC-clean share", float(np.mean(list(out.values()))))
