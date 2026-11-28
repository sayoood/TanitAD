"""PREREG section 3.1: do RL-1's V1 rows apply to S0's `feat`?  Structural comparison of an RL scorer (R0d u1000, arms A and B, read only) against P2's scorer modules.  No label, no pick, no data.
    python v1_applicability_check.py [out.json]
Reads: C:/Users/Admin/par18/rl/r0d_dl/r0d/scorer_states/scorer_states_u1000.pt and D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt (mmap).  CPU only.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import torch

RL = "C:/Users/Admin/par18/rl/r0d_dl/r0d/scorer_states/scorer_states_u1000.pt"
P2 = "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"


def flat(prefix, d):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(prefix + k + ".", v)
        else:
            yield prefix + k, v


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main(out):
    ck = torch.load(RL, map_location="cpu", weights_only=False)
    sd = torch.load(P2, map_location="cpu", mmap=True, weights_only=False)["model"]
    res = {"rl_states": RL, "rl_states_sha256": sha(RL), "p2_ckpt": P2, "p2_score_head_shape": list(sd["score_head.weight"].shape), "arms": {}}
    for arm in ("A", "B"):
        n = same = 0
        mx, shape_diff, by_module = 0.0, [], {}
        for k, v in flat("", ck[arm]):
            if k not in sd:
                shape_diff.append([k, "missing in P2"])
                continue
            w = sd[k]
            if tuple(w.shape) != tuple(v.shape):
                shape_diff.append([k, list(w.shape), list(v.shape)])
                continue
            d = float((w.float() - v.float()).abs().max())
            n += 1
            same += int(d == 0.0)
            mx = max(mx, d)
            mod = k.split(".")[0]
            by_module.setdefault(mod, {"tensors": 0, "bit_equal": 0, "max_abs_diff": 0.0})
            by_module[mod]["tensors"] += 1
            by_module[mod]["bit_equal"] += int(d == 0.0)
            by_module[mod]["max_abs_diff"] = max(by_module[mod]["max_abs_diff"], d)
        res["arms"][arm] = {"score_head_shape": list(ck[arm]["score_head"]["weight"].shape), "tensors_compared": n, "bit_equal_to_p2": same, "max_abs_diff": mx, "shape_mismatches": shape_diff, "by_module": by_module}
    res["conclusion"] = ("V1's rows do NOT apply to S0's feat: the RL scorer is a 7-output head on a re-trained scorer decoder; "
                         "none of the shared score_q_mlp / score_dec tensors equals P2's, so the 256-d pre-head feature differs from S0's `feat`.") \
        if all(a["bit_equal_to_p2"] == 0 and a["shape_mismatches"] for a in res["arms"].values()) else "UNEXPECTED: see arms"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps({k: res[k] for k in ("p2_score_head_shape", "conclusion")}), {a: (v["tensors_compared"], v["bit_equal_to_p2"], v["max_abs_diff"], v["score_head_shape"]) for a, v in res["arms"].items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/par18/pickrule/run/results/v1_applicability.json")
