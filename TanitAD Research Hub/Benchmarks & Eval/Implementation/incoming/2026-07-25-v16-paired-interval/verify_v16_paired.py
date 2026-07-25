"""Independently re-derive the v1.6-vs-v1 PAIRED episode-cluster bootstrap.

WHY THIS EXISTS
---------------
`MODEL_REGISTRY.md` §1.4b quotes a paired interval

    Delta(v1.6 - v1) = +0.0104 m  CI95 [-0.0888, +0.1147]  separated = FALSE

but the run that produced it left **no raw JSON in the repo** (only
`paired_v3enc10k_vs_flagship30k.json` exists), and the block does not state B.
Re-deriving it here turns an INHERITED number into a MEASURED one and emits the
missing artifact.

COST: CPU-only, no pod, no GPU, no checkpoint load. Both arms' per-window
predictions were persisted by their canonical eval runs; the paired bootstrap is
a pure function of those. Re-scoring the checkpoints would change nothing and
would put load on a pod for no information.

Run:  <venv>/python verify_v16_paired.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "taniteval"))

from taniteval.ci import (  # noqa: E402
    DEFAULT_N_BOOT,
    episode_cluster_bootstrap,
    overlapping_holdout_se,
    paired_episode_cluster_bootstrap,
)

RES = REPO / "taniteval" / "results"
A_KEY = "flagship-v16-ab-ft"      # v1.6 -- LP-FT, step 5999
B_KEY = "flagship-30k"            # THE DEPLOYED v1 = flagship4b-speedjerk-30k
N_BOOT = DEFAULT_N_BOOT           # 2000, the program default
SEED = 0


def ade_0_2s(pred, gt):
    """taniteval.driving.per_window['ade_0_2s'] verbatim: mean over the 4 wps."""
    return torch.linalg.norm(pred - gt, dim=-1).mean(1).numpy().astype(np.float64)


def main():
    a = torch.load(RES / f"windows_{A_KEY}.pt", map_location="cpu", weights_only=False)
    b = torch.load(RES / f"windows_{B_KEY}.pt", map_location="cpu", weights_only=False)

    # ---- ALIGNMENT PROOF ------------------------------------------------- #
    # A paired test is valid iff the two arms were scored on the SAME windows in
    # the SAME order AND clustered on the SAME episode partition.
    #
    # The eid LABELS deliberately differ between the two families: `bench.py`
    # labels episodes by file index 0..39, `eval_flagship_v16.py` by the real
    # `episode_id` (e.g. 808464434). Requiring identical labels would refuse a
    # valid pairing. The load-bearing invariant is the PARTITION -- the set of
    # window-index groups -- which is checked directly below, not inherited from
    # the registry's "consistent 1-to-1 relabel" note.
    ea = [str(x) for x in a["eid"]]
    eb = [str(x) for x in b["eid"]]
    pa, pb = defaultdict(list), defaultdict(list)
    for i, x in enumerate(ea):
        pa[x].append(i)
    for i, x in enumerate(eb):
        pb[x].append(i)
    part_a = sorted(tuple(v) for v in pa.values())
    part_b = sorted(tuple(v) for v in pb.values())
    fwd = dict(zip(ea, eb))

    align = {
        "eid_labels_identical": ea == eb,
        "eid_relabel_is_bijection": bool(
            len(fwd) == len(set(fwd.values())) == len(set(ea)) == len(set(eb))
            and all(fwd[x] == y for x, y in zip(ea, eb))),
        "EPISODE_PARTITION_IDENTICAL": part_a == part_b,
        "gt_max_abs_diff": float((a["gt"] - b["gt"]).abs().max()),
        "cv_max_abs_diff": float((a["cv"] - b["cv"]).abs().max()),
        "speed_max_abs_diff": float((a["speed"] - b["speed"]).abs().max()),
        "wp_steps_identical": list(a["wp_steps"]) == list(b["wp_steps"]),
        "shape_a": list(a["pred"].shape),
        "shape_b": list(b["pred"].shape),
        "n_episodes": len(part_a),
    }
    align["ALIGNED"] = bool(
        align["EPISODE_PARTITION_IDENTICAL"]
        and align["eid_relabel_is_bijection"]
        and align["gt_max_abs_diff"] == 0.0
        and align["cv_max_abs_diff"] == 0.0
        and align["wp_steps_identical"]
        and align["shape_a"] == align["shape_b"]
    )
    if not align["ALIGNED"]:
        raise SystemExit(f"REFUSING to pair unaligned arms: {align}")

    pw_a = ade_0_2s(a["pred"], a["gt"])
    pw_b = ade_0_2s(b["pred"], b["gt"])

    # The two eid families give the SAME estimand (identical partition) but a
    # different Monte-Carlo realisation, because `_draws` resamples
    # `np.unique(eid)` and the relabel changes the sort order. Reporting both is
    # the honest form: the spread between them IS the MC noise at B = 2000.
    single_a = episode_cluster_bootstrap(pw_a, ea, n_boot=N_BOOT, seed=SEED)
    single_b = episode_cluster_bootstrap(pw_b, eb, n_boot=N_BOOT, seed=SEED)
    paired = paired_episode_cluster_bootstrap(pw_a, pw_b, eb, n_boot=N_BOOT, seed=SEED)
    paired_realeid = paired_episode_cluster_bootstrap(pw_a, pw_b, ea, n_boot=N_BOOT, seed=SEED)

    # The DEPRECATED estimator on the same per-window vector, so the width
    # difference is visible side by side rather than asserted.
    old_a = overlapping_holdout_se(pw_a)
    old_b = overlapping_holdout_se(pw_b)

    # ---- SECONDARY: G1, on the decision-grade estimator -------------------- #
    # §1.4b states G1 (beat REF-C 0.458) as FAIL at line 474 and PASS at line
    # 520 -- a heldout-vs-full-set basis mismatch. The paired test settles it
    # without picking a basis. Same alignment discipline as above.
    xl = torch.load(RES / "windows_refc-xl-30k.pt", map_location="cpu", weights_only=False)
    exl = [str(x) for x in xl["eid"]]
    pxl = defaultdict(list)
    for i, x in enumerate(exl):
        pxl[x].append(i)
    xl_aligned = (sorted(tuple(v) for v in pxl.values()) == part_b
                  and float((xl["gt"] - b["gt"]).abs().max()) == 0.0)
    vs_xl = None
    if xl_aligned:
        pw_xl = ade_0_2s(xl["pred"], xl["gt"])
        vs_xl = paired_episode_cluster_bootstrap(pw_a, pw_xl, exl,
                                                 n_boot=N_BOOT, seed=SEED)
        vs_xl["arm_b"] = "refc-xl-30k"
        vs_xl["full_set_refc_xl"] = round(float(pw_xl.mean()), 5)

    out = {
        "task": "v1.6 vs deployed v1 -- paired episode-cluster bootstrap on ADE@2s",
        "generated": "2026-07-25",
        "evidence_class": "MEASURED (ours; re-derived from persisted per-window artifacts)",
        "corpus": "physicalai-val-0c5f7dac3b11 (clean held-out split)",
        "arms": {
            "a": {"key": A_KEY, "registry": "MODEL_REGISTRY.md §1.4b flagship-v16-ab-ft",
                  "step": 5999, "artifact": str(RES / f"windows_{A_KEY}.pt")},
            "b": {"key": B_KEY, "registry": "MODEL_REGISTRY.md §1.2 flagship4b-speedjerk-30k",
                  "step": 29999, "artifact": str(RES / f"windows_{B_KEY}.pt")},
        },
        "alignment_proof": align,
        "metric": "ade_0_2s (metric-BEV ego-frame, mean over wps @0.5/1/1.5/2 s, metres)",
        "point_estimates_full_set": {
            "v1_6": round(float(pw_a.mean()), 5),
            "v1": round(float(pw_b.mean()), 5),
        },
        "single_arm_episode_cluster_bootstrap": {"v1_6": single_a, "v1": single_b},
        "PAIRED": paired,
        "PAIRED_real_episode_id_family": paired_realeid,
        "SECONDARY_v16_vs_refc_xl_G1": vs_xl,
        "per_window_corr": round(float(np.corrcoef(pw_a, pw_b)[0, 1]), 4),
        "deprecated_overlapping_holdout_se_on_same_windows": {
            "v1_6_halfwidth": round(float(old_a), 4),
            "v1_halfwidth": round(float(old_b), 4),
            "note": "NOT the published heldout +-; that came from 8 random 20% "
                    "holdouts inside the eval script. Shown only to make the "
                    "estimator distinction concrete.",
        },
        "published_heldout_for_reference": {
            "v1_6": "0.4886 +- 0.0800 (eval_v16_flagship-v16-ab-ft.json, overlapping_holdout_se)",
            "v1": "0.4522 +- 0.0312 (MODEL_REGISTRY §1.2, overlapping_holdout_se)",
            "WARNING": "cross-eid-FAMILY comparison, not admissible -- see §1.4b",
        },
    }
    dst = Path(__file__).with_name("v16_vs_v1_paired_bootstrap.json")
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"ALIGNED: {align['ALIGNED']}  (gt max diff {align['gt_max_abs_diff']})")
    print(f"v1.6 full-set  {pw_a.mean():.5f}   {single_a['lo']:.4f}..{single_a['hi']:.4f}")
    print(f"v1   full-set  {pw_b.mean():.5f}   {single_b['lo']:.4f}..{single_b['hi']:.4f}")
    print(f"PAIRED delta   {paired['delta']:+.4f}  CI95 [{paired['lo']:+.4f}, {paired['hi']:+.4f}]"
          f"  separated={paired['separated']}  B={paired['n_boot']}"
          f"  eps={paired['n_episodes']}  win={paired['n_windows']}")
    print(f"  (real-eid fam) {paired_realeid['delta']:+.4f}  "
          f"CI95 [{paired_realeid['lo']:+.4f}, {paired_realeid['hi']:+.4f}]"
          f"  separated={paired_realeid['separated']}")
    print(f"p(delta>0)     {paired['p_delta_gt0']}   per-window corr {out['per_window_corr']}")
    if vs_xl:
        print(f"G1 vs REF-C-XL {vs_xl['delta']:+.4f}  "
              f"CI95 [{vs_xl['lo']:+.4f}, {vs_xl['hi']:+.4f}]  "
              f"separated={vs_xl['separated']}  (XL full-set {vs_xl['full_set_refc_xl']})")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
