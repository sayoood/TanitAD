#!/usr/bin/env python3
"""merge_arm_dumps.py -- merge refav1_arm dumps of DISJOINT clip sets into one.

⛔ WHY IT REFUSES RATHER THAN CONCATENATES. Splitting a render roll across two
boxes to halve the wall-clock is fine; splitting it across two ARMS is a
fabricated result, and a merged dump is exactly the artifact in which that would
be invisible. So this asserts -- never assumes -- that the two dumps carry the
same checkpoint STEP and strict-load report, the same plan config, the same
grid, and the same action units, and it refuses on any clip present in both.
The ckpt PATH is deliberately excluded from the comparison: it differs between
boxes by construction, while the step + strict-load report identify the weights.

Each source dump must be COMPLETE (its `manifest.json` is written only after the
last episode), and `_merged_from` records which clips came from which dump.

    python merge_dumps.py <out_dir> <dump_a> <dump_b> [<dump_c> ...]
"""
import glob
import io
import json
import os
import shutil
import sys

out = sys.argv[1]
dumps = sys.argv[2:]
if len(dumps) < 2:
    raise SystemExit("need >= 2 dumps")

mans = []
for d in dumps:
    with io.open(os.path.join(d, "manifest.json"), encoding="utf-8") as fh:
        mans.append(json.load(fh))

ref = mans[0]


def key(m):
    return dict(
        ckpt=m["model"].get("ckpt"), step=m["model"].get("step"),
        load=m["model"].get("state_dict_load"),
        plan={k: m["plan_cfg"].get(k) for k in
              ("n_samples", "n_iters", "n_elites", "seed", "horizon", "dt",
               "a_max", "kappa_max", "beta", "decay", "inject_baselines")},
        grid={k: m["grid"].get(k) for k in
              ("dt_s", "horizon_k", "wm_k", "window", "window_stride")},
        units=m.get("action_units"), arms=m.get("arms"), tiers=m.get("tiers"),
    )


k0 = key(ref)
# ⛔ the ckpt PATH differs between boxes on purpose; everything that defines the
# ARM must not. The step and the strict-load report are the identity check.
k0.pop("ckpt")
for d, m in zip(dumps[1:], mans[1:]):
    k = key(m)
    k.pop("ckpt")
    if k != k0:
        diff = {a: (k0[a], k[a]) for a in k0 if k0[a] != k.get(a)}
        raise SystemExit(f"REFUSING: {d} is a different arm/grid: {diff}")

seen, entries = {}, []
os.makedirs(os.path.join(out, "decisions"), exist_ok=True)
for old in glob.glob(os.path.join(out, "ep*.npz")) + \
        glob.glob(os.path.join(out, "decisions", "ep*.npz")):
    os.remove(old)

n = 0
for d, m in zip(dumps, mans):
    for e in m["episodes"]:
        name = e["name"]
        if name in seen:
            raise SystemExit(f"REFUSING: clip {name} is in both {seen[name]} and {d}")
        seen[name] = d
        src = os.path.join(d, "ep%03d.npz" % int(e["file_index"]))
        dsrc = os.path.join(d, "decisions", "ep%03d.npz" % int(e["file_index"]))
        if not (os.path.exists(src) and os.path.exists(dsrc)):
            print(f"  skip {name}: not yet written in {d}")
            del seen[name]
            continue
        shutil.copy(src, os.path.join(out, "ep%03d.npz" % n))
        shutil.copy(dsrc, os.path.join(out, "decisions", "ep%03d.npz" % n))
        e2 = dict(e)
        e2["file_index"] = n
        e2["episode_index"] = n
        e2["_source_dump"] = os.path.abspath(d)
        entries.append(e2)
        n += 1

man = dict(ref)
man["episodes"] = entries
man["grid"] = dict(ref["grid"])
man["grid"]["n_episodes"] = len(entries)
man["grid"]["n_windows"] = sum(int(e.get("n_windows", 0)) for e in entries)
man["_merged_from"] = [{"dump": os.path.abspath(d),
                        "ckpt": m["model"].get("ckpt"),
                        "step": m["model"].get("step"),
                        "clips": [e["name"] for e in m["episodes"]]}
                       for d, m in zip(dumps, mans)]
man["_merge_rule"] = ("disjoint clip sets, identical step / strict-load report / "
                      "plan config / grid / action units — asserted, not assumed")
with io.open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
    json.dump(man, fh, indent=1)
print(f"merged {n} clips into {out} from {len(dumps)} dumps "
      f"({man['grid']['n_windows']} windows)")
for e in entries:
    print(f"  ep{e['file_index']:03d}  {e['name']}  n_windows={e.get('n_windows')}  "
          f"<- {os.path.basename(e['_source_dump'])}")
