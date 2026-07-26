"""Does each goal mode beat the constant-velocity baseline? PAIRED, same windows."""
import json
import sys

import torch

sys.path.insert(0, "/root/taniteval")
from taniteval import ci as _ci  # noqa: E402

R = "/root/v4eval/results_goalmode"
W = {m: torch.load(f"{R}/windows_v4-15k-goal-{m}.pt", map_location="cpu",
                   weights_only=False) for m in ("oracle", "produced", "neutral")}
o = W["oracle"]
eid = [str(x) for x in o["eid"]]
ade = lambda p, g: (p - g).norm(dim=-1).mean(dim=-1).numpy()   # noqa: E731
cv = ade(o["cv"], o["gt"])

out = {"_read": ("paired_episode_cluster_bootstrap of MODEL - CV on the same 881 "
                 "windows. NEGATIVE delta = the model beats constant velocity. "
                 "`separated` = the CI excludes zero."),
       "estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py), B=2000, "
                    "40 episodes, alpha=0.05",
       "cv_ade_0_2s": _ci.episode_cluster_bootstrap(cv, eid, n_boot=2000)}
for m in W:
    d = _ci.paired_episode_cluster_bootstrap(ade(W[m]["pred"], W[m]["gt"]), cv,
                                             eid, n_boot=2000, seed=0)
    d["beats_cv"] = bool(d["delta"] < 0)
    d["beats_cv_SEPARATED"] = bool(d["delta"] < 0 and d["separated"])
    out[f"{m}_minus_cv"] = d
    print(f"{m:9s} delta={d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] "
          f"sep={d['separated']} -> beats_cv_separated={d['beats_cv_SEPARATED']}")
# and against flagship v1's corrected full-set 0.4271 (point ref, NOT paired --
# different arm, different windows dump; stated as a reference not a test)
out["_v1_reference"] = {
    "flagship_v1_ade_0_2s_full_set": 0.4271,
    "source": "MODEL_REGISTRY.md 1.2 (flagship4b-speedjerk-30k @ 29999), the "
              "CORRECTED full-set value -- NOT the trainer's dense-20 reading",
    "note": "a POINT reference. v1 is a different arm evaluated through the "
            "intent-free operative rollout; it is not paired with these windows "
            "and no interval on the difference is claimed here."}
json.dump(out, open(f"{R}/CV_BASELINE_PAIRED.json", "w"), indent=2, default=str)
print("->", f"{R}/CV_BASELINE_PAIRED.json")
