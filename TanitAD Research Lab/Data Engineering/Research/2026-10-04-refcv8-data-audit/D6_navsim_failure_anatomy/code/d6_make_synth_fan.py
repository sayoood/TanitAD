"""D6 -- build a SYNTHETIC fan sidecar around REAL banked plans, to unit-test d6_fan_score.py / the P1 analysis without a GPU.
Candidate sel_idx = the banked emitted plan; the others = the banked plan shifted laterally by a ladder of offsets and scaled in speed.
Never a result: its only use is the controls (the pick re-scored in-batch must reproduce the banked sub-scores)."""
import base64
import json
import sys

import numpy as np

hooks_path, tokens_path, out_path = sys.argv[1:4]
nt = int(sys.argv[4]) if len(sys.argv) > 4 else 12
NC = int(sys.argv[5]) if len(sys.argv) > 5 else 181          # 117 = the as-launched universe (no r7 heads)
hooks = {c["token"]: c for c in json.load(open(hooks_path, encoding="utf-8"))["pdm_score_calls"]}
toks = [l.strip() for l in open(tokens_path) if l.strip()][:nt]
rng = np.random.default_rng(1)


def b64(x):
    return base64.b64encode(np.ascontiguousarray(x.astype(np.float32)).tobytes()).decode()


with open(out_path, "w", encoding="utf-8") as fo:
    for tok in toks:
        h = hooks[tok]
        plan = np.asarray(h["agent_poses"], dtype=np.float32)
        cands = []
        for i in range(NC):
            c = plan.copy()
            sh = rng.uniform(-3.0, 3.0)
            sc = rng.uniform(0.5, 1.2)
            c[:, 0] *= sc
            c[:, 1] = c[:, 1] * sc + sh * np.linspace(0.1, 1.0, 8)
            cands.append(c)
        sel = 30
        cands[sel] = plan
        cands = np.stack(cands)
        fo.write(json.dumps({"token": tok, "stage": 1 if h["scene_type"].endswith("ORIGINAL") else 2, "arm": "SYN", "seed": 0, "n_fan": 117,
                             "n_cands": NC, "sel_idx": sel, "sel_idx_base": sel + 1, "r7_sel_idx": (150 if NC > 117 else None), "r7_pick_is_wta": (NC > 117), "universe": ("FAN117+WTA64" if NC > 117 else "FAN117"), "dtype": "float32",
                             "cands_knots_b64": b64(np.zeros((NC, 8, 2))), "cands_poses_b64": b64(cands),
                             "sel_score_v3_b64": b64(rng.normal(size=117)), "reach_keep_b64": b64(np.ones(117)),
                             "r7_score_b64": (b64(rng.normal(size=181)) if NC > 117 else None), "traj_r7_poses_b64": (b64(cands[150]) if NC > 117 else None),
                             "poses_emitted": plan.tolist()}) + "\n")
print("wrote", len(toks), "->", out_path)
