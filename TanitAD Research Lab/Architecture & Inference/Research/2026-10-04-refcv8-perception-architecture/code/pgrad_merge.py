"""WP-D P-GRAD merge (CPU): the parts written by `pgrad.py --batch-count` -> pgrad.json, with the SAME summary
arithmetic as the single-job path (summed per-term gradients over all 8 batches = one 64-window batch).

Refuses unless the parts cover batches 0..7 exactly once and share one window draw. Controls: L (linearity, per batch
and group, <= 1e-4) and Z (weight-0 term exactly 0) are read from every part; K (aux share vs the in-run detector,
NOT pre-registered) is recomputed on the merged sums.
Run (Thor CPU): python pgrad_merge.py --dir <out/pgrad> [--n-batches 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

TERMS = ["traj", "box3d", "agent", "map_hires", "tac_v6", "rest"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n-batches", type=int, default=8)
    a = ap.parse_args()
    d = Path(a.dir)
    parts = [torch.load(p, map_location="cpu", weights_only=False) for p in sorted(d.glob("part_*.pt"))]
    cover = sorted(b for p in parts for b in range(*p["batches"]))
    if cover != list(range(a.n_batches)):
        raise SystemExit(f"[pgrad-merge] parts cover batches {cover}, need 0..{a.n_batches - 1} exactly once")
    if len({p["window_draw_sha256"] for p in parts}) != 1:
        raise SystemExit("[pgrad-merge] parts come from different window draws")
    groups = list(parts[0]["group_sizes"])
    sums = {k: {g: sum(p["sums"][k][g].double() for p in parts) for g in groups} for k in TERMS + ["total"]}
    per_batch = sorted([r for p in parts for r in p["per_batch"]], key=lambda r: r["batch"])
    L = max(p["ctl"]["L_max_rel_err"] for p in parts)
    Z = max(p["ctl"]["Z_max_abs"] for p in parts)
    summ = {"norm": {}, "share_of_term_norm_sum": {}, "cos": {}, "aux_share_conflict_def": {},
            "proj_share_on_total": {}}
    for g in groups:
        nrm = {k: float(sums[k][g].norm()) for k in TERMS + ["total"]}
        summ["norm"][g] = nrm
        tsum = sum(nrm[k] for k in TERMS)
        summ["share_of_term_norm_sum"][g] = {k: (nrm[k] / tsum if tsum > 0 else None) for k in TERMS}
        summ["cos"][g] = {}
        ks = TERMS + ["total"]
        for i, k1 in enumerate(ks):
            for k2 in ks[i + 1:]:
                den = nrm[k1] * nrm[k2]
                summ["cos"][g][f"{k1}|{k2}"] = float(torch.dot(sums[k1][g], sums[k2][g]) / den) if den > 0 else None
        aux = sums["box3d"][g] + sums["map_hires"][g] + sums["tac_v6"][g]
        na, nt = float(aux.norm()), nrm["traj"]
        summ["aux_share_conflict_def"][g] = na / (na + nt) if (na + nt) > 0 else None
        tot = sums["total"][g]
        tn2 = float(torch.dot(tot, tot))
        summ["proj_share_on_total"][g] = ({k: float(torch.dot(sums[k][g], tot) / tn2) for k in TERMS}
                                          if tn2 > 0 else None)
    trunk = [g for g in groups if not g.startswith("x_")]
    whole = {k: torch.cat([sums[k][g] for g in trunk]) for k in TERMS + ["total"]}
    tot = whole["total"]
    summ["trunk_whole"] = {
        "norm": {k: float(whole[k].norm()) for k in whole},
        "cos_to_total": {k: float(torch.dot(whole[k], tot) / (whole[k].norm() * tot.norm()))
                         if float(whole[k].norm()) > 0 else None for k in TERMS},
        "proj_share_on_total": {k: float(torch.dot(whole[k], tot) / torch.dot(tot, tot)) for k in TERMS},
        "aux_share_conflict_def": (lambda na, nt: na / (na + nt))(
            float((whole["box3d"] + whole["map_hires"] + whole["tac_v6"]).norm()), float(whole["traj"].norm())),
        "cos_pairs": {f"{k1}|{k2}": float(torch.dot(whole[k1], whole[k2]) / (whole[k1].norm() * whole[k2].norm()))
                      for i, k1 in enumerate(TERMS) for k2 in TERMS[i + 1:]
                      if float(whole[k1].norm()) > 0 and float(whole[k2].norm()) > 0}}
    controls = {"L_linearity_max_rel_err": L, "L_pass": L <= 1e-4, "Z_zero_weight_max_abs": Z, "Z_pass": Z == 0.0,
                "K_aux_share_trunk": summ["trunk_whole"]["aux_share_conflict_def"],
                "K_expected_band_in_run": [0.992, 0.995], "K_is_preregistered": False}
    p0 = parts[0]
    rec = {"probe": "P-GRAD", "prereg_sha256": "c054190bee4df5f266c99318d5374fbad9821a17ae7e44830c4b5f8feffc6f87",
           "tier": "OPEN-LOOP PERCEPTION DIAGNOSTIC (gradient census; no plan scored; four families N/A)",
           "ckpt": p0["ckpt"], "step": p0["step"], "strict": p0["strict"], "loader_departures": p0["loader_departures"],
           "measurement_choices": p0["measurement_choices"], "weights": p0["weights"],
           "group_sizes": p0["group_sizes"], "n_batches": len(per_batch), "batch": 8,
           "parts": [{"batches": p["batches"], "wall_s": p["wall_s"], "cuda_max_mem_gb": p["cuda_max_mem_gb"]}
                     for p in parts],
           "window_draw_sha256": p0["window_draw_sha256"], "controls": controls, "summary": summ,
           "per_batch": per_batch}
    (d / "pgrad.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"[pgrad-merge] L {L:.2e} pass={controls['L_pass']} Z {Z} pass={controls['Z_pass']} "
          f"K {controls['K_aux_share_trunk']:.4f}", flush=True)
    return 0 if (controls["L_pass"] and controls["Z_pass"]) else 3


if __name__ == "__main__":
    sys.exit(main())
