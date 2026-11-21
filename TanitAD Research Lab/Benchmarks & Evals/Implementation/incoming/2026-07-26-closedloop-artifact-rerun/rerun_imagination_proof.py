#!/usr/bin/env python3
"""RE-RUN the 2026-07-22 imagination-in-the-loop proof on its own 12-episode surface.

WHAT IS AND IS NOT ALREADY CORRECT IN THAT ARTIFACT
---------------------------------------------------
`closedloop_flagship-30k_imagination-proof.json` is the program's only MIXED
closed-loop artifact:

  * the HEADLINE it was built to produce — the paired A/B verdict
    `IMAGINATION_HELPS`, delta -0.213 [-0.341, -0.053] — was ALREADY
    `paired_episode_cluster_bootstrap`. It is **not** affected by this task.
  * every OTHER scalar in the same file — `closed_bike_ade@2s 1.7315 +- 0.2396`,
    `closed_grnd 2.6277`, `open_grnd 0.3177`, `divergence 0.2314`, both
    compounding blocks — came out of `_agg`/`_jack`, and the file says so:
    `protocol.ci = "overlapping_holdout_se, 8 random 20% holdouts (DEPRECATED,
    not a jackknife)"`.

So this run corrects the second set WITHOUT touching the first, and checks that
the already-correct A/B block reproduces — which is the strongest available
evidence that the re-run is faithful, because that block is bit-comparable.

SURFACE. 12 episodes (`ep_00000..ep_00011`), stride 8 -> 265 windows. The
original ran on a local RTX 4060; this runs the identical episode slice on the
eval pod A40. The closed loop is deterministic (no CEM, no sampling), so unlike
the G4 re-drive this one IS expected to reproduce.

Run:  OMP_NUM_THREADS=8 PYTHONPATH=/root/taniteval:/root/TanitAD/stack \
      python3 rerun_imagination_proof.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/root/taniteval")

from taniteval import closedloop as CL              # noqa: E402
from taniteval import data, loaders                 # noqa: E402
from taniteval.registry import MODELS               # noqa: E402

OUT = Path("/root/cl_rerun_20260726")
PUB = OUT / "published" / "closedloop_flagship-30k_imagination-proof.json"
VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"
EPISODES = 12                    # ep_00000..ep_00011, per provenance.json


def main():
    t0 = time.time()
    entry = [m for m in MODELS if m["key"] == "flagship-30k"][0]
    L = loaders.load(entry, "cuda")
    files = data.list_val_episodes(VAL, EPISODES)
    eps = (data.load_frames(files) if L["feed"] == "frames"
           else data.load_features(files, L["feed"], "cuda"))
    win = CL.collect(L["model"], L["step_readout"], eps, "cuda",
                     speed_input=bool(entry.get("speed_input")))
    torch.save({k: (v.cpu() if torch.is_tensor(v) else v)
                for k, v in win.items()}, OUT / "clwin_imagination-proof-12ep.pt")
    res = CL.analyze(win)
    res["model"] = {k: entry.get(k) for k in
                    ("key", "name", "arch", "encoder", "speed_input")}
    res["ckpt_step"] = L["step"]
    res["wall_s"] = round(time.time() - t0, 1)
    res["_rerun"] = {"date": "2026-07-26", "episodes": EPISODES,
                     "surface": "ep_00000..ep_00011, stride 8",
                     "note": "the paired A/B block was ALREADY decision-grade; "
                             "the per-path scalars were not"}
    (OUT / "closedloop_imagination-proof.CORRECTED.json").write_text(
        json.dumps(res, indent=2, default=str), encoding="utf-8")

    pub = json.loads(PUB.read_text(encoding="utf-8")) if PUB.exists() else None
    cmp = {"_surface": f"{res['n_windows']} windows / {res['n_episodes']} episodes",
           "already_decision_grade_AB_block": {}, "corrected_scalars": {}}
    if pub:
        # (a) the A/B block that was ALREADY correct — must reproduce
        for k in ("A_open_plan_bike_ade@2s", "B_closed_bike_ade@2s",
                  "paired_delta_B_minus_A_ade@2s"):
            p, n = pub["imagination_comparison"][k], res["imagination_comparison"][k]
            pm = p.get("mean", p.get("delta"))
            nm = n.get("mean", n.get("delta"))
            cmp["already_decision_grade_AB_block"][k] = {
                "published": pm, "rerun": nm,
                "published_ci": [p["lo"], p["hi"]], "rerun_ci": [n["lo"], n["hi"]],
                "estimator": n["estimator"],
                "reproduced": bool(abs(pm - nm) < 5e-3)}
        cmp["already_decision_grade_AB_block"]["verdict"] = {
            "published": pub["imagination_comparison"]["verdict"].split(":")[0],
            "rerun": res["imagination_comparison"]["verdict"].split(":")[0],
            "separated_published":
                pub["imagination_comparison"]["paired_delta_B_minus_A_ade@2s"]["separated"],
            "separated_rerun":
                res["imagination_comparison"]["paired_delta_B_minus_A_ade@2s"]["separated"]}
        # (b) the scalars that were LEGACY — the actual correction
        for blk, mk in (("closed_bike", "ade_0_2s"), ("closed_bike", "fde@2s"),
                        ("closed_grnd", "ade_0_2s"), ("open_grnd", "ade_0_2s"),
                        ("open_bike", "ade_0_2s"), ("cv", "ade_0_2s")):
            p = pub["closedloop_ade_fde"]["heldout"][blk][mk]
            n = res["closedloop_ade_fde"]["heldout"][blk][mk]
            cmp["corrected_scalars"][f"{blk}.{mk}"] = {
                "published_mean": p["mean"], "published_ci95": p["ci95"],
                "published_estimator": "overlapping_holdout_se",
                "corrected_mean": n["mean"], "corrected_ci95": n["ci95"],
                "corrected_lo": n["lo"], "corrected_hi": n["hi"],
                "corrected_estimator": n["estimator"],
                "mean_shift_pct": round(
                    100.0 * (p["mean"] - n["mean"]) / max(1e-9, n["mean"]), 3),
                "ci_width_ratio": (round(n["ci95"] / p["ci95"], 3)
                                   if p["ci95"] else None)}
        pd_ = pub["stability"]["divergence_rate_gt5m@2s"]
        nd = res["stability"]["divergence_rate_gt5m@2s"]
        cmp["corrected_scalars"]["stability.divergence_rate_gt5m@2s"] = {
            "published_mean": pd_["mean"], "published_ci95": pd_["ci95"],
            "corrected_mean": nd["mean"], "corrected_ci95": nd["ci95"],
            "corrected_lo": nd["lo"], "corrected_hi": nd["hi"],
            "corrected_estimator": nd["estimator"],
            "mean_shift_pct": round(
                100.0 * (pd_["mean"] - nd["mean"]) / max(1e-9, nd["mean"]), 3)}
    (OUT / "imagination_proof_published_vs_corrected.json").write_text(
        json.dumps(cmp, indent=2), encoding="utf-8")
    print(json.dumps(cmp, indent=2))


if __name__ == "__main__":
    main()
