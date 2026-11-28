"""Cost of the scorer, MEASURED. MIN over repeats: the machine is under load and the mean lies
(a first pass gave stride-4 SLOWER than stride-2, which is arithmetically impossible)."""
import sys, time, copy, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import scorer_gate as G, score_proposals as SP
cfg, _ = G.build_engine_config(); calcs = SP.build_calculators(cfg)
t0 = time.time(); sd, log_sd, sc, smps, step, init = G.scenario_data_from_log(sys.argv[1], 40)
t_build = time.time() - t0
gxy, gyw = G.teacher_future(smps, step)
def best(fn, n=5):
    ts = []
    for _ in range(n):
        t = time.time(); fn(); ts.append(time.time() - t)
    return min(ts)
t_copy = best(lambda: copy.deepcopy(sd))
t_pass = best(lambda: SP.score_proposal(sd, log_sd, calcs, gxy, gyw))
print(f"  ScenarioData build+log load (once per LOG): {t_build:6.2f} s")
print(f"  deepcopy of ScenarioData                  : {t_copy*1000:7.1f} ms")
print(f"  one endpoint pass (copy + 6 calculators)  : {t_pass*1000:7.1f} ms"
      f"   => calculators alone ~{(t_pass-t_copy)*1000:.1f} ms")
print(f"\n  {'stride':>7s} {'steps':>6s} {'ms/proposal':>12s} {'s / 64 proposals':>18s}")
for stride in (1, 2, 4, 5, 10):
    steps = len(range(1, 21, stride)) + (0 if (20 - 1) % stride == 0 else 1)
    t = best(lambda s=stride: SP.score_proposal_rollout(sd, log_sd, calcs, gxy, gyw, stride=s), 3)
    print(f"  {stride:7d} {steps:6d} {t*1000:12.1f} {t*64:18.1f}")
