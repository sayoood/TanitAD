"""Writes example/rollout.npz, example/human.npz, example/score.json (tailgating at 0.8 s behind a steady lead, 20 s)."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synth_rollouts as S
import behaviour_score as B

t = S.tgrid(20.0)
xl0 = S.EGO_L / 2 + 30.0 + 2.25
lead = S.const_agent(10, t, xl0, 0.0, 15.0)
v = np.full_like(t, 15.0)
ego = S.make(t, v, x0=18.0, agents=[lead], scenario_id="example-tailgate-0.8s", category="steady_lead", epdms=0.8)
hum = S.make(t, v, x0=0.0, agents=[lead], scenario_id="example-tailgate-0.8s", category="steady_lead")
ct = np.arange(0.0, 20.0, 0.5)
ego.update(call_t=ct, call_ax_fed=np.zeros_like(ct), call_pick=np.zeros(len(ct), np.int32), call_plan_v4=np.full_like(ct, 15.0),
           call_plan_v0=np.full_like(ct, 15.0), call_agg_score=np.zeros_like(ct))
out = HERE / "example"
out.mkdir(exist_ok=True)
np.savez(out / "rollout.npz", **ego)
np.savez(out / "human.npz", **{k: hum[k] for k in hum if not k.startswith(("checkpoint", "seed", "suite", "dtype"))})
B.score_files(out / "rollout.npz", out / "human.npz", out / "score.json")
print("wrote", out)
