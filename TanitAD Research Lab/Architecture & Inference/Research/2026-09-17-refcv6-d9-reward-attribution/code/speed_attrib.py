"""D9 attribution, part 2: does the RL policy OVER-SPEED relative to the human?

Part 1 (`dac_attrib.py`) showed rl-s0 nearly doubles the off-road waypoint rate
(SEPARATED) while rl-s1 does NOT -- so DAC-blindness cannot be seed 1's cause. The
four-family paired read says s1's damage sits in LONGITUDINAL (`LON_speed_mae_mps`,
`LON_accel_mae_mps2`) where s0's sits in LATERAL (`LAT_heading_mae_deg`).

HYPOTHESIS for s1: the proxy's EP term rewards PROGRESS along the human's own future
path. Nothing in the reward penalises exceeding the speed the situation warrants, so
maximising EP pushes speed up.
⇒ PREDICTION: rl-s1's mean speed is biased HIGH against the human, and rl-s0's is not.

Speed is read from the plan itself -- consecutive waypoint separations over dt = 0.5 s --
so it needs no extra model state. The human's future is transformed into the ego frame at
t0 first (MEASURED: `gt_future_ext` is in the EPISODE frame; skipping this transform is
what made part 1's control read 0.1741 instead of 0.0171).

Estimator: paired_episode_cluster_bootstrap on the SIGNED bias, clustered by episode.
⚠️ n = 2 RL seeds. A separated CI answers "would another draw of EPISODES say this?",
never "would another TRAINING RUN say this?".
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Admin\tanitad-wt-bevtac\taniteval")
from taniteval.ci import paired_episode_cluster_bootstrap as PB      # noqa: E402

RUN = pathlib.Path("C:/Users/Admin/tanitad-caches/ddv2rl-l1-20260917")
ARMS = {"base": "t1_base_dump", "rl-s0": "t1_l1-rl-s0_dump", "rl-s1": "t1_l1-rl-s1_dump"}
DT = 0.5
TICKS = np.array([4, 9, 14, 19])           # the gt grid is 0.1 s; the plan is 0.5 s x 4

acc: dict[str, list[float]] = {a: [] for a in ARMS}
hum: list[float] = []
eids: list[int] = []

for ei in range(41):
    ps = {a: RUN / d / f"ep{ei:03d}.npz" for a, d in ARMS.items()}
    pdec = RUN / "t1_base_dump" / "decisions" / f"ep{ei:03d}.npz"
    if not all(p.exists() for p in ps.values()) or not pdec.exists():
        continue
    z = {a: np.load(p) for a, p in ps.items()}
    zd = np.load(pdec)
    pl, gt, ok = zd["pose_last"], zd["gt_future_ext"], zd["gt_future_valid_ext"]
    for wi in range(len(zd["ws"])):
        if not (ok[wi][TICKS] > 0).all():
            continue
        d = gt[wi][TICKS, :2] - pl[wi][None, :2]
        c, s_ = np.cos(-pl[wi][2]), np.sin(-pl[wi][2])
        g = np.stack([d[:, 0] * c - d[:, 1] * s_, d[:, 0] * s_ + d[:, 1] * c], 1)
        hum.append(float(np.linalg.norm(np.diff(np.vstack([[0, 0], g]), axis=0), axis=1).mean() / DT))
        eids.append(ei)
        for a in ARMS:
            p = z[a]["os"][wi]
            acc[a].append(float(np.linalg.norm(
                np.diff(np.vstack([[0, 0], p]), axis=0), axis=1).mean() / DT))

h = np.asarray(hum)
eid = np.asarray(eids)
out = {"_what": "D9 attribution part 2: signed speed bias of the T1 plans vs the human",
       "_tier": "T1 (self-action open loop)", "_evidence_class": "MEASURED (ours)",
       "_estimator": "paired_episode_cluster_bootstrap, n_boot 2000, seed 0, by episode",
       "_n": {"windows": len(h), "episodes": int(len(set(eids)))},
       "human_mean_speed_mps": float(h.mean()), "arms": {}, "paired_vs_base": {}}

print(f"windows {len(h)}   episodes {len(set(eids))}   human mean speed {h.mean():.3f} m/s\n")
print(f"{'arm':8s} {'mean v':>9s} {'signed bias':>13s} {'|err|':>9s}")
for a in ARMS:
    v = np.asarray(acc[a])
    out["arms"][a] = {"mean_speed_mps": float(v.mean()),
                      "signed_bias_vs_human_mps": float((v - h).mean()),
                      "abs_err_mps": float(np.abs(v - h).mean())}
    print(f"{a:8s} {v.mean():>9.3f} {(v - h).mean():>+13.4f} {np.abs(v - h).mean():>9.4f}")

print()
for a in ("rl-s0", "rl-s1"):
    r = PB(np.asarray(acc[a]) - h, np.asarray(acc["base"]) - h, eid, n_boot=2000, seed=0)
    out["paired_vs_base"][a] = {k: r.get(k) for k in ("delta", "lo", "hi", "separated")}
    flag = "  <== SEPARATED" if r["separated"] else "  (not separated)"
    print(f"   signed speed bias, {a} - base = {r['delta']:+.4f} "
          f"[{r['lo']:+.4f}, {r['hi']:+.4f}]{flag}")

pathlib.Path("C:/Users/Admin/qland/speed_attrib.json").write_text(
    json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
print("\nwrote speed_attrib.json")
