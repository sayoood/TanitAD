#!/usr/bin/env python3
"""Falsify the OPEN-LOOP run's own claims against the banked rollout dumps.

⛔ Every claim this checks is one the panel would still LOOK fine without. That is exactly
why they are checked mechanically rather than asserted in a README:

1. **`ego_pinned_to_log`** — the ego pose on every window equals the logged pose at
   `i_gt` to 1e-9. If this were false the run would not be open loop at all, and the whole
   page would be mislabelled. Nothing else in the pipeline would notice.
2. **`arms_share_pixels`** — the per-tick frame md5 is IDENTICAL between the two arms.
   The paired estimator is invalid the moment the arms are not scored on the same
   observations, and the renderer is a step function of pose (a 0.1 px camera rotation has
   been measured to move the 2 s waypoint 6.65 m), so "we rendered it the same way twice"
   is not good enough — it has to be the same bytes.
3. **`md5_consistent`** — each step's recorded `frame_md5` matches the sweep-level digest
   at that tick, i.e. the step really was decided on the frame the manifest names.
4. **`objects_vs_empty_frames_differ`** — the A/B condition actually changed the pixels.
   ⚠️ A silently no-op ablation reads as "the model ignores the agents", which is the
   single most tempting wrong conclusion available on this panel.

Exit code is non-zero if any invariant fails.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TOL = 1e-9


def check(dirpath: Path, cond: str, arms=("flagship-v1", "refc-base")):
    ds = [json.loads((dirpath / f"{cond}_{a}.json").read_text()) for a in arms]
    m = [d["frame_md5"] for d in ds]
    steps = [[s for r in d["rollouts"] for s in r["steps"]] for d in ds]
    out = {
        "condition": cond,
        "mode_declared": ds[0].get("mode"),
        "n_windows": len(steps[0]),
        "n_clusters": len(ds[0]["rollouts"]),
        "arms_share_pixels": bool(m[0] == m[1]),
        "md5_consistent": all(s["frame_md5"] == d["frame_md5"][str(s["k"])]
                              for d, ss in zip(ds, steps) for s in ss),
        "ego_pinned_to_log": all(
            abs(s["ego"][0] - d["gt"][s["i_gt"]]["x"]) < TOL
            and abs(s["ego"][1] - d["gt"][s["i_gt"]]["y"]) < TOL
            and abs(s["ego"][3] - d["gt"][s["i_gt"]]["yaw"]) < TOL
            for d, ss in zip(ds, steps) for s in ss),
        "windows_aligned_across_arms": [s["k"] for s in steps[0]] == [s["k"] for s in steps[1]],
        "f_eff": [d.get("f_eff") for d in ds],
    }
    out["all_hold"] = all(out[k] for k in ("arms_share_pixels", "md5_consistent",
                                           "ego_pinned_to_log",
                                           "windows_aligned_across_arms"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="dir with <scene>/<cond>_<arm>.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.rollouts)
    res, ok = {}, True
    for scene in sorted(p.name for p in root.iterdir() if p.is_dir()):
        d = root / scene
        blocks = [check(d, c) for c in ("objects", "empty")
                  if (d / "objects_flagship-v1.json").exists()]
        a = json.loads((d / "objects_flagship-v1.json").read_text())["frame_md5"]
        b = json.loads((d / "empty_flagship-v1.json").read_text())["frame_md5"]
        ndiff = sum(1 for k in a if a[k] != b[k])
        res[scene] = {
            "conditions": blocks,
            "objects_vs_empty_frames_differ": {
                "n_ticks_differing": ndiff, "n_ticks": len(a),
                "all_differ": ndiff == len(a),
                "why_it_matters": "a silently no-op ablation reads as 'the model ignores "
                                  "the agents' — the most tempting wrong conclusion here"},
        }
        ok &= all(x["all_hold"] for x in blocks) and ndiff == len(a)
    res["ALL_INVARIANTS_HOLD"] = bool(ok)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
