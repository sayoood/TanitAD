"""THE 3-D JOIN'S FRAME OFFSET, MEASURED BY A SWEEP -- not derived, not trusted.

⛔ THE TRAP. The trainer's 2-D agent seam looks up `JoinFileReader` on the
EPISODE index (`frame_idx`, post-n_stack-trim); `AgentJoin3D` keys on the RAW
v2ep index (`frame`). They differ by `n_stack - 1`. Join on the wrong key and
every `cz`/`h` the box head sees belongs to a moment ~0.2 s away from the frame
the trunk saw. The loss stays finite, non-zero and entirely plausible. NOTHING
RAISES.

⭐ THE INDEPENDENT REFERENCE. The 3-D join is an ANNOTATION of the 2-D join:
*"every line, key, field, agent and byte of the 2-D join is carried; cz and h
are APPENDED"* (its own meta.json). So at the CORRECT offset the two readers
resolve THE SAME ROWS, and the 2-D reader's `[A, 6]` array must match the 3-D
line's `cx/cy/yaw/l/w` EXACTLY -- and at any other offset it must not. That is
an ANALYTIC target (exact equality, not a fitted minimum), computed from the
two files' own contents, with no formula of mine in the loop.

Swept over offsets -3..+3 on real (clip, frame) pairs, reporting for each:
  n_agents_equal   -- exact float equality on cx, cy, yaw, l, w
  mean_abs_dx      -- metres, as a continuous separation measure
"""
import io
import json
import sys
from pathlib import Path

import numpy as np

WT = r"C:/Users/Admin/tanitad-wt-perctrain"
for p in (WT + "/stack/scripts", WT + "/stack"):
    sys.path.insert(0, p)

import lzma                                                       # noqa: E402

from train_p8_occupancy import JoinFileReader                     # noqa: E402
from tanitad.data.v2_dataset import (load_or_build_manifest,      # noqa: E402
                                     stable_episode_id)

CACHE = r"D:/Projects/TanitAD-artifacts/v2ep-eval139-256x1024cyl"
JOIN = (r"D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/"
        r"b1eval_agents_3d.jsonl.xz")
N_CLIPS = int(sys.argv[2]) if len(sys.argv) > 2 else 6

man = load_or_build_manifest(CACHE, verbose=False)
clips = list(man["clip_id"])[:N_CLIPS]
n_stack = int(list(man["n_stack"])[0])
eids = {int(stable_episode_id(c)) for c in clips}
rd = JoinFileReader(JOIN, episode_ids=eids, with_track_ids=True)

# the 3-D join's own lines, keyed by (clip, RAW frame)
raw3d: dict = {}
with lzma.open(JOIN, "rt", encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        r = json.loads(line)
        if r["clip_id"] in set(clips):
            raw3d[(r["clip_id"], int(r["frame"]))] = r["agents"]

rep = {"n_stack_from_manifest": n_stack, "n_clips": len(clips),
       "n_3d_lines_for_those_clips": len(raw3d), "offsets": {}}

# the EPISODE-index frames the trainer actually looks up (t + w - 1)
pairs = []
for c in clips:
    eid = int(stable_episode_id(c))
    for f in range(7, 180, 17):                  # a spread of episode indices
        if rd.lookup(eid, f) is not None:
            pairs.append((c, eid, f))
rep["n_pairs"] = len(pairs)

for off in range(-3, 4):
    n_lines = n_ag = n_eq = 0
    dx = []
    for c, eid, f in pairs:
        two = rd.lookup(eid, f)
        three = raw3d.get((c, f + off))
        if two is None or three is None or len(three) != int(two.shape[0]):
            continue
        n_lines += 1
        for k, a in enumerate(three):
            n_ag += 1
            # JoinFileReader's [A, 6] row order is (cx, cy, yaw, l, w, occ)
            same = (float(a["cx"]) == float(two[k, 0])
                    and float(a["cy"]) == float(two[k, 1])
                    and float(a["yaw"]) == float(two[k, 2])
                    and float(a["l"]) == float(two[k, 3])
                    and float(a["w"]) == float(two[k, 4]))
            n_eq += bool(same)
            dx.append(abs(float(a["cx"]) - float(two[k, 0])))
    rep["offsets"][str(off)] = {
        "n_lines_comparable": n_lines, "n_agents": n_ag,
        "n_agents_exactly_equal": n_eq,
        "frac_exactly_equal": (n_eq / n_ag) if n_ag else float("nan"),
        "mean_abs_dx_m": float(np.mean(dx)) if dx else float("nan")}

best = max(rep["offsets"], key=lambda k: rep["offsets"][k]["frac_exactly_equal"]
           if rep["offsets"][k]["frac_exactly_equal"] ==
           rep["offsets"][k]["frac_exactly_equal"] else -1)
others = [v["mean_abs_dx_m"] for k, v in rep["offsets"].items()
          if k != best and v["mean_abs_dx_m"] == v["mean_abs_dx_m"]]
rep["verdict"] = {
    "best_offset": int(best),
    "expected_offset_n_stack_minus_1": n_stack - 1,
    "frac_exactly_equal_at_best": rep["offsets"][best]["frac_exactly_equal"],
    "mean_abs_dx_at_best_m": rep["offsets"][best]["mean_abs_dx_m"],
    "min_mean_abs_dx_elsewhere_m": min(others) if others else float("nan"),
    # ⭐ the SEPARATION: the neighbouring offsets must be clearly worse, or the
    # sweep has not identified anything -- the same discriminator
    # `check_pose_alignment` applies to its own shifted pairings.
    "separation_x": ((min(others) / rep["offsets"][best]["mean_abs_dx_m"])
                     if others and rep["offsets"][best]["mean_abs_dx_m"] > 0
                     else float("inf")),
    "trainer_uses": "f + n_stack - 1",
    "AGREES_WITH_TRAINER": int(best) == n_stack - 1,
}
io.open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(rep, indent=1))
print(json.dumps(rep["verdict"], indent=1))
for k, v in rep["offsets"].items():
    print(" offset %+d: %6d agents, exact %5.1f %%, mean|dx| %8.4f m"
          % (int(k), v["n_agents"], 100 * v["frac_exactly_equal"],
             v["mean_abs_dx_m"]))
