"""BAR A, step 3 — the PAIRED test against deployed v1 on the identical windows.

Bar A is stated as "recover >= 70.8 % of the waste, i.e. reach <= 0.4271, TYING
v1". Comparing two point estimates is not a test of tying. v1's own per-window
dump is on this pod, on the SAME 40 episodes / 881 windows, so the comparison can
be PAIRED -- which is the only form in which "ties v1" means anything.

NAME IS NOT PROVENANCE (CLAUDE.md's standing inversion warning: the no-speed
ablation control and the deployed v1 have near-identical repo names). This script
therefore identifies v1 BY ITS NUMBER: the dump must reproduce the registry's
full-set 0.4271 before it is used, and the alternative dumps are printed next to
it so the choice is visible rather than implied.

Estimator: paired episode-cluster bootstrap (taniteval/ci.py), B=2000, unit =
episode. NEVER overlapping_holdout_se.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/taniteval")
from taniteval.ci import (episode_cluster_bootstrap,  # noqa: E402
                          paired_episode_cluster_bootstrap)

V1 = "/root/taniteval/results/windows_flagship-30k.pt"
REPRO = "/root/bara/repro_windows_30k_produced.pt"   # the alignment witness
ALTS = {
    "flagship-nospeed (the ABLATION CONTROL, must NOT be used as v1)":
        "/root/taniteval/results/windows_flagship-nospeed.pt",
    "flagship-speed (the isolated speed-fix proof)":
        "/root/taniteval/results/windows_flagship-speed.pt",
}
REGISTRY_V1_FULLSET = 0.4271
OUT = Path("/root/bara/v1_paired.json")


def ade4(d):
    return (d["pred"] - d["gt"]).norm(dim=-1).mean(1).numpy()


def miss2m(d):
    return ((d["pred"] - d["gt"]).norm(dim=-1)[:, -1] > 2.0).float().numpy()


def main(tag="produced"):
    L = lambda p: torch.load(p, map_location="cpu", weights_only=False)  # noqa: E731
    d1 = L(V1)
    W = torch.load(f"/root/bara/bar_a_{tag}_windows.pt", map_location="cpu",
                   weights_only=False)
    R = {"_experiment": "paired v4 (as-trained / CE-control / regret) vs deployed "
                        "v1 on the identical 881 windows",
         "_evidence_class": "MEASURED (ours)",
         "_estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py), "
                       "B=2000, unit = episode. NEVER overlapping_holdout_se.",
         "_v1_dump": V1, "_bar_a_tag": tag}

    a1 = ade4(d1)
    R["v1_identification"] = {
        "ade_0_2s_recomputed": round(float(a1.mean()), 5),
        "registry_full_set": REGISTRY_V1_FULLSET,
        "abs_diff": round(abs(float(a1.mean()) - REGISTRY_V1_FULLSET), 5),
        "n_windows": int(a1.shape[0]),
        "n_episodes": len(set(str(x) for x in d1["eid"])),
        "_rule": "identified BY THE NUMBER, not by the filename",
    }
    R["v1_identification"]["PASS"] = (
        R["v1_identification"]["abs_diff"] <= 1e-3)
    for name, p in ALTS.items():
        try:
            R.setdefault("_alternatives_printed_not_used", {})[name] = round(
                float(ade4(L(p)).mean()), 5)
        except Exception as e:
            R.setdefault("_alternatives_printed_not_used", {})[name] = repr(e)

    # ALIGNMENT BY TENSOR IDENTITY, NOT BY LABEL.
    # The two dumps label episodes differently -- `taniteval.rollout.collect`
    # writes the episode INDEX (0..39) while `collect_planner` writes
    # `int(episode_id)` (a large integer: the raw id reinterpreted). Comparing
    # the label strings therefore says "not aligned" for windows that are in fact
    # bit-identical. What actually proves alignment is the ground truth itself:
    # `gt`, `cv`, `speed` and `head_deg` are properties of the WINDOW, not of the
    # model, so if all four match position-wise the two dumps hold the same
    # windows in the same order. Verified here rather than assumed.
    rep = L(REPRO)
    e1 = [str(x) for x in d1["eid"]]
    gt_ok = torch.allclose(d1["gt"].float(), rep["gt"].float(), atol=1e-6)
    cv_ok = torch.allclose(d1["cv"].float(), rep["cv"].float(), atol=1e-6)
    sp_ok = torch.allclose(d1["speed"].float(), rep["speed"].float(), atol=1e-5)
    hd_ok = torch.allclose(d1["head_deg"].float(), rep["head_deg"].float(),
                           atol=1e-4)
    bnd = lambda e: [i for i, (a, b) in enumerate(zip(e[:-1], e[1:])) if a != b]
    bnd_ok = bnd(e1) == bnd([str(x) for x in rep["eid"]])
    R["window_alignment"] = {
        "n_v1": len(e1), "n_v4": int(np.asarray(W["ade_regret"]).shape[0]),
        "eid_LABELS_identical": e1 == [str(x) for x in W["eid"]],
        "_label_note": "labels DIFFER by construction (episode index vs raw id) "
                       "and that is not evidence of misalignment",
        "gt_identical": bool(gt_ok), "cv_identical": bool(cv_ok),
        "speed_identical": bool(sp_ok), "head_deg_identical": bool(hd_ok),
        "episode_boundaries_identical": bool(bnd_ok),
        "gt_max_abs_diff": float((d1["gt"].float()
                                  - rep["gt"].float()).abs().max()),
        "ALIGNED": bool(gt_ok and cv_ok and sp_ok and hd_ok and bnd_ok),
        "_why": "a paired bootstrap is only valid on the SAME windows in the "
                "SAME order",
    }
    if not (R["v1_identification"]["PASS"]
            and R["window_alignment"]["ALIGNED"]):
        R["ABORTED"] = ("v1 dump did not identify by number, or the windows are "
                        "not aligned -- no paired delta is quotable")
        OUT.write_text(json.dumps(R, indent=2, default=str))
        print(json.dumps(R, indent=2, default=str))
        return

    m1 = miss2m(d1)
    arms = {"as_trained": W["ade_as_trained"], "ce_control": W["ade_ce"],
            "regret": W["ade_regret"]}
    R["point_estimates"] = {"v1": round(float(a1.mean()), 4),
                            **{k: round(float(np.asarray(v).mean()), 4)
                               for k, v in arms.items()}}
    R["paired_vs_v1"] = {}
    for k, v in arms.items():
        v = np.asarray(v, dtype=float)
        R["paired_vs_v1"][k] = {
            "ade_0_2s": paired_episode_cluster_bootstrap(v, a1, e1, n_boot=2000,
                                                        seed=0),
            "_orientation": "arm - v1; NEGATIVE = the v4 arm BEATS deployed v1; "
                            "a CI containing 0 = statistically TIED",
        }
    R["singles"] = {"v1": episode_cluster_bootstrap(a1, e1, n_boot=2000, seed=0)}
    R["v1_miss_at_2m"] = round(float(m1.mean()), 4)
    OUT.write_text(json.dumps(R, indent=2, default=str))
    print(json.dumps(R, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "produced")
