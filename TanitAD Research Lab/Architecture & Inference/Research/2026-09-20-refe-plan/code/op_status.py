#!/usr/bin/env python3
"""The on-policy pipeline's status as ONE JSON line, for the report (pod-side, read-only).

Counts the labelled samples per label version (a key's newest (ckpt_step, label_version) wins, as in
train.OnPolicyBank), the NAVSIM vs teacher drivable-area violation rates over version-2 proposals, the
labelling rate over the last hour, and whether the trainer runs --scorer-mode onpolicy (the switch).
Label version 3 (2026-09-26, PI option 2) replaced the teacher's comfort label with NAVSIM's own
ego_is_comfortable on the simulated states; over version-3 proposals it also reports how many NAVSIM
calls comfortable, the teacher's mean comfort value on the same proposals, and how often the two agree.
  python3 op_status.py > onpolicy.json
  ssh <pod> 'python3 -' < op_status.py > onpolicy.json     # the report's route: nothing is written on the pod
"""
import glob
import json
import os
import subprocess
import time

SETS = "/workspace/data/refe_navtrain/onpolicy/sets"
RUN = "/workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3"
best, recent = {}, 0
first_v3 = None
now = time.time()
for f in glob.glob(os.path.join(SETS, "onpolicy_*.jsonl")):
    with open(f) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") != "onpolicy_set":
                continue
            k = (r["log_name"], r["token"], int(r["step"]), int(r["rank"]))
            rk = (int(r["ckpt_step"]), int(r.get("label_version", 1)))
            if k not in best or best[k][0] < rk:
                best[k] = (rk, r)
            if not r.get("backfilled"):
                if rk[1] >= 3 and r.get("at") and (first_v3 is None or r["at"] < first_v3):
                    first_v3 = r["at"]
                try:
                    t = time.mktime(time.strptime(r["at"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
                    recent += (now - t) < 3600
                except (KeyError, ValueError):
                    pass
by_version = {}
for (rk, r) in best.values():
    by_version[str(rk[1])] = by_version.get(str(rk[1]), 0) + 1
v2 = [r for (rk, r) in best.values() if rk[1] >= 2]
n = sum(len(r["targets"]) for r in v2)
nd = sum(t.get("navsim_dac.violation", 0) > 0 for r in v2 for t in r["targets"])
td = sum((t.get("dac.violation") or 0) > 0 for r in v2 for t in r["targets"])
v3 = [r for (rk, r) in best.values() if rk[1] >= 3]
t3 = [t for r in v3 for t in r["targets"]]
nc = [t["navsim_comfort"] for t in t3 if t.get("navsim_comfort") is not None]
tc = [(t["teacher_comfort.Comfort.reward"], t["navsim_comfort"]) for t in t3
      if t.get("teacher_comfort.Comfort.reward") is not None and t.get("navsim_comfort") is not None]
ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
switched = any(l.startswith("/workspace/venv-teacher/bin/python train.py") and "--scorer-mode onpolicy" in l
               for l in ps.splitlines())
sw_step, sw_at = None, None
mp = os.path.join(RUN, "metrics.jsonl")
if os.path.exists(mp):
    for line in open(mp):
        if '"declared_change"' in line:
            e = json.loads(line)
            sw_step, sw_at = e.get("step"), e.get("at")
print(json.dumps({"sets": len(best), "v2": len(v2), "proposals_v2": n,
                  "navsim_dac_viol": nd / max(n, 1), "teacher_dac_viol": td / max(n, 1),
                  "by_version": by_version, "v3": len(v3), "proposals_v3": len(t3),
                  "v3_backfilled": sum(bool(r.get("backfilled")) for r in v3), "v3_first_live_at": first_v3,
                  "navsim_comfortable": sum(c >= 0.5 for c in nc) / max(len(nc), 1),
                  "teacher_comfort_mean": sum(a for a, _ in tc) / max(len(tc), 1),
                  "teacher_comfort_values": sorted({round(a, 3) for a, _ in tc})[:8],
                  "comfort_agree": sum((a >= 0.5) == (b >= 0.5) for a, b in tc) / max(len(tc), 1),
                  "rate_per_h": recent, "switched": switched, "switch_step": sw_step, "switch_at_utc": sw_at,
                  "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
