"""Full per-frame label run (PI 2026-11-08: "We need the labels for each frame. Optimize results without me and start scaling").
One stage per launch, in 30-minute jobs under the shared dev-box GPU lock, resumable, pilot parts reused:
  chain_full.py --stage A                                image statistics + SigLIP on every 0.2 s frame of clips_all.json
  chain_full.py --stage B --clips splits/qwen_clips.json Qwen at 1 Hz on the listed clips
Markers: F:/strat_out/ZZ_<stage>_DONE, ZZ_<stage>_FAILED, and F:/strat_out/STOP stops at the next job boundary.
"""
import argparse
import json
import os
import shutil
import sys
import time

import chain_gpu as CG
from extract_context import replace_retry

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = CG.OUT


def reuse_pilot(stage, ids):
    """Pilot parts come from the same code, prompts and model; a part is copied only when the full run has none."""
    os.makedirs(f"{OUT}/ctx{stage}", exist_ok=True)
    n = 0
    for c in ids:
        src, dst = f"{CG.POUT}/ctx{stage}/{c}.npz", f"{OUT}/ctx{stage}/{c}.npz"
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst + ".tmp")
            replace_retry(dst + ".tmp", dst)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B"])
    ap.add_argument("--clips", default=HERE + "/splits/clips_all.json")
    ap.add_argument("--budget", type=int, default=30, help="minutes per lock-held job; Qwen loads in about 11 minutes, so B uses 60")
    a = ap.parse_args()
    clips = a.clips if os.path.isabs(a.clips) else HERE + "/" + a.clips
    ids = json.load(open(clips))["clips"]
    os.makedirs(OUT + "/logs", exist_ok=True)
    CG.log(f"stage {a.stage}: {len(ids)} clips, reused from the pilot {reuse_pilot(a.stage, ids) if a.stage == 'A' else 0}")      # B's questions differ from the pilot's, so its parts are not reused
    stalls = 0
    while CG.parts(OUT, a.stage, ids) < len(ids):
        if os.path.exists(OUT + "/STOP"):
            CG.log("STOP marker seen")
            return 2
        before = CG.parts(OUT, a.stage, ids)
        t0 = time.time()
        rc, lf = CG.job_retry(a.stage, clips, OUT, budget=a.budget)
        after = CG.parts(OUT, a.stage, ids)
        CG.log(f"{a.stage} rc={rc} {before}->{after}/{len(ids)} {round(time.time() - t0)} s {lf}")
        stalls = stalls + 1 if after == before else 0
        if stalls >= 3:
            open(f"{OUT}/ZZ_{a.stage}_FAILED", "w").write(f"no progress in 3 jobs, last log {lf}\n")
            return 1
    open(f"{OUT}/ZZ_{a.stage}_DONE", "w").write(f"{len(ids)} clips\n")
    CG.log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
