#!/usr/bin/env python3
"""Is the reachability band a SELECTION lever, or only a COMPUTE lever?

⭐ THE QUESTION. `be2da04` measured that pre-decode anchor filtering leaves the
selection index identical on 881/881 windows at 2.78x fewer decodes. That is a
**compute** result. It is silent on the thing actually asked for — *a better
selector without retraining* — because "the winner always survives" and "the
winner is the best choice" are different claims, and the first one being true is
exactly what makes the second one untestable at that band.

So this sweeps the band. At `accel_max` wide enough that nothing is pruned the
delta is 0.0000 BY CONSTRUCTION; the interesting region is where the band starts
biting. For each band it reports, on the SAME banked windows:

  * how much is pruned (`survivors_mean`, `rows_full_fan_frac`, `rows_empty`)
  * ⭐ `oracle_survives_frac` — does the MIN-ADE anchor survive? This is the
    honest risk metric. `winner_survives_frac` cannot fall below it and cannot
    tell you whether pruning cost you anything, because the winner is chosen by
    the same logits either way.
  * ADE of the restricted argmax vs the unrestricted argmax, with a **paired**
    episode-cluster bootstrap over the fan's own episodes.

⛔ NO RETRAINING, NO NEW DECODES. Every number comes off an already-banked fan,
so this is a pure post-hoc selection experiment and cannot be confounded by a
different training run.

⚠️ Reachability is a property of the ANCHOR SET and `v0`, not of the model. These
survivor counts belong to the anchors that produced this fan and must be
re-measured, never inherited, when the anchors change.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch


def _ade(paths: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """[B,S,2] vs [B,S,2] -> [B] mean displacement over the waypoints."""
    return torch.linalg.norm(paths - gt, dim=-1).mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan", required=True, help="banked fan_*.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bands", default="0.5,1.0,1.5,2.0,2.5,3.0,4.0,6.0",
                    help="accel_max values (m/s^2) to sweep")
    ap.add_argument("--horizon-s", type=float, default=2.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--logits-key", default="logits",
                    help="'refined_logits' scores the S1-refined ranking instead")
    a = ap.parse_args()
    if os.path.isdir(a.out):
        sys.exit(f"--out must be a FILE, got a directory: {a.out}")

    sys.path.insert(0, os.environ.get("TANITAD_STACK", "/home/user/TanitAD/stack"))
    from tanitad.refs.refc_select import reachability_mask
    from taniteval import ci

    d = torch.load(a.fan, map_location="cpu", weights_only=False)
    fan, gt, v0 = d["fan"].float(), d["gt"].float(), d["v0"].float()
    logits = d[a.logits_key].float()
    eid = list(d["eid"])
    B, N = logits.shape
    print(f"[fan] {a.fan}\n[fan] {B} windows x {N} anchors  ckpt_step={d.get('ckpt_step')}  "
          f"logits_key={a.logits_key}", flush=True)

    per_anchor_ade = torch.linalg.norm(fan - gt[:, None], dim=-1).mean(-1)   # [B,N]
    base_idx = logits.argmax(-1)                                            # S1
    base_ade = per_anchor_ade.gather(1, base_idx[:, None]).squeeze(1)
    oracle_idx = per_anchor_ade.argmin(-1)
    oracle_ade = per_anchor_ade.gather(1, oracle_idx[:, None]).squeeze(1)

    head = {
        "fan": a.fan, "logits_key": a.logits_key,
        "ckpt": d.get("ckpt"), "ckpt_step": d.get("ckpt_step"),
        "n_windows": B, "n_anchors": N, "n_episodes": len(set(eid)),
        "wp_steps": d.get("wp_steps"), "horizon_s": a.horizon_s,
        "S1_ade": round(float(base_ade.mean()), 4),
        "oracle_ade": round(float(oracle_ade.mean()), 4),
        "selection_gap": round(float(base_ade.mean() - oracle_ade.mean()), 4),
        "_selection_gap_note": (
            "the headroom ANY selector improvement must come out of. A band that "
            "prunes nothing cannot touch it; a band that prunes the oracle can "
            "only make it worse."),
    }
    print(f"[base] S1 {head['S1_ade']}  oracle {head['oracle_ade']}  "
          f"gap {head['selection_gap']}", flush=True)

    rows = []
    for am in [float(x) for x in a.bands.split(",")]:
        keep = reachability_mask(fan, v0, accel_max=am, horizon_s=a.horizon_s)
        n_keep = keep.sum(-1)
        empty = n_keep == 0
        # ⛔ THE GUARD, not a silent fallback: a row the band empties keeps its
        # FULL fan. Dropping such a row instead would quietly change the
        # denominator between bands and make the sweep non-comparable.
        keep_eff = keep.clone()
        keep_eff[empty] = True
        masked = logits.masked_fill(~keep_eff, float("-inf"))
        sel = masked.argmax(-1)
        sel_ade = per_anchor_ade.gather(1, sel[:, None]).squeeze(1)

        changed = (sel != base_idx)
        paired = ci.paired_episode_cluster_bootstrap(
            sel_ade.numpy(), base_ade.numpy(), eid, n_boot=a.n_boot)
        rows.append({
            "accel_max": am,
            "survivors_mean": round(float(n_keep.float().mean()), 2),
            "survivors_min": int(n_keep.min()), "survivors_max": int(n_keep.max()),
            "decode_saving_x": round(N / max(float(n_keep.float().mean()), 1e-9), 3),
            "rows_full_fan_frac": round(float((n_keep == N).float().mean()), 4),
            "rows_empty_guard_fired": int(empty.sum()),
            # ⭐ the honest risk metric — winner_survives_frac cannot see this
            "oracle_survives_frac": round(
                float(keep.gather(1, oracle_idx[:, None]).squeeze(1).float().mean()), 4),
            "winner_survives_frac": round(
                float(keep.gather(1, base_idx[:, None]).squeeze(1).float().mean()), 4),
            "rows_selection_changed": int(changed.sum()),
            "ade": round(float(sel_ade.mean()), 4),
            "delta_vs_S1": paired,
        })
        r = rows[-1]
        print(f"  a={am:<4} keep {r['survivors_mean']:>6} ({r['decode_saving_x']}x)  "
              f"oracle_surv {r['oracle_survives_frac']}  changed {r['rows_selection_changed']:>4}  "
              f"ade {r['ade']}  d {paired['delta']} [{paired['lo']}, {paired['hi']}] "
              f"sep={paired['separated']}", flush=True)

    helped = [r for r in rows if r["delta_vs_S1"]["separated"]
              and r["delta_vs_S1"]["delta"] < 0]
    head["bands"] = rows
    head["_verdict"] = (
        f"the band separates FAVOURABLY on THIS ranking at accel_max="
        f"{[r['accel_max'] for r in helped]}, best post-band ADE "
        f"{min(r['ade'] for r in helped)} vs this ranking's {head['S1_ade']}. "
        f"⚠️ REPAIR IS NOT IMPROVEMENT: a band that rescues a badly-calibrated "
        f"ranking has not beaten the BEST available ranking. Before quoting "
        f"this as a selection win, score the same fan's other ranking keys "
        f"(--logits-key) and check the post-band ADE beats the best of them."
        if helped else
        "⛔ reachability is NOT a selection lever on this ranking: no band gives "
        "a separated improvement. Where it prunes nothing the delta is 0.0000 BY "
        "CONSTRUCTION; where it prunes, it does not beat the unrestricted "
        "argmax. It remains a COMPUTE lever (see decode_saving_x), which is a "
        "different claim and must not be reported as the other one.")
    head["_safety_boundary"] = (
        "⭐ oracle_survives_frac is what bounds how far the band may be tightened. "
        "Once it drops below 1.0 the band is deleting reachable-optimal "
        "candidates, and any ADE gain at that point is being bought against a "
        "damaged ceiling.")
    head["_stage_warning"] = (
        "⛔ decode_saving_x here is measured on the DECODED fan (S2). The "
        "published pre-decode anchor figure (S2b, `be2da04`: 2.78x) is a "
        "DIFFERENT STAGE with different survivor counts. Do not quote one for "
        "the other.")
    head["_estimator"] = ("paired episode-cluster bootstrap over the fan's own "
                          "episodes (taniteval.ci). Paired because both arms are "
                          "scored on the SAME windows.")
    head["_no_retrain"] = ("every number is post-hoc on a banked fan — no "
                           "training, no new decodes, so it cannot be confounded "
                           "by a different run.")
    with open(a.out, "w") as f:
        json.dump(head, f, indent=2, default=str)
    print(f"\n{head['_verdict']}\n[out] {a.out}", flush=True)


if __name__ == "__main__":
    main()
