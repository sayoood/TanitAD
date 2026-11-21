"""Q3 — is the RANKED score computed from the CANDIDATE TRAJECTORY or from
something upstream of it?  (REFe's class-C second defect: the scoring branch was
fed the proposal queries instead of the path it is supposed to score.)

Method: un-zero `control_head` so the sampler's output actually moves, then
perturb ONLY the sampler (`control_head`) and ask whether `sel_score` /
`sel_idx` move.  A ranker that scores the emitted path must move; one that
scores the anchor bank cannot.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch
from diag_f1_f9_liveness import build, fwd
from diag_f3_f4_zeroinit import unzero_
from tanitad.models import refcv6_diffusion as rv6

def arm(flags, *, sel_refined=False, sel_score_emitted=False, tag=""):
    d = build(flags)
    d.sel.refined = sel_refined
    d.sel.score_emitted = sel_score_emitted
    unzero_(d, scale=0.3)
    base = fwd(d, seed=3)
    b_score, b_idx = base["sel_score"].clone(), base["sel_idx"].clone()
    b_traj = base["traj"].clone()
    with torch.no_grad():
        d.control_head.bias.add_(1.5) if d.cascade is None else \
            d.cascade.control_heads[-1].bias.add_(1.5)
    o = fwd(d, seed=3)
    return {"tag": tag,
            "traj_moved": round(float((o["traj"] - b_traj).abs().max()), 5),
            "sel_score_moved": round(float((o["sel_score"] - b_score).abs().max()), 6),
            "sel_idx_before": b_idx.tolist(), "sel_idx_after": o["sel_idx"].tolist(),
            "ranked_surface_sees_the_sample":
                bool(float((o["sel_score"] - b_score).abs().max()) > 0)}

def main():
    out = {"arms": []}
    out["arms"].append(arm(rv6.DiffusionFlags(f2_dd_step=True), tag="DEFAULT (sel.refined=False, no F5)"))
    out["arms"].append(arm(rv6.DiffusionFlags(f5_emitting_conf=True), tag="F5 emitting_conf"))
    out["arms"].append(arm(rv6.DiffusionFlags(f2_dd_step=True), sel_refined=True, tag="--sel-refined"))
    out["arms"].append(arm(rv6.DiffusionFlags(f3_per_layer=True, f5_emitting_conf=True), tag="F3+F5"))
    # what does the DEFAULT config actually carry?
    from tanitad.refs import refc as _r
    sc = _r.SelectionConfig()
    out["SelectionConfig_defaults"] = {"refined": sc.refined,
                                       "score_emitted": sc.score_emitted}
    d = build(rv6.DiffusionFlags(f2_dd_step=True))
    out["tele_sampler_ranks_the_fan_default"] = (fwd(d).get("sel_tele") or {}).get("sampler_ranks_the_fan")
    txt = json.dumps(out, indent=2, default=str); print(txt)
    (Path(__file__).resolve().parents[1] / "raw" / "q3_scoring_surface.json").write_text(txt, encoding="utf-8")

if __name__ == "__main__":
    sys.exit(main())
