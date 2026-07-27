#!/usr/bin/env python3
"""E-H2 cache builder — THE DUMP DELTA. One forward pass, then the sweep is CPU.

WHY THIS FILE EXISTS
--------------------
``HIERARCHY_PRIOR_RESEARCH.md`` section 9.4 states the gap as a measured fact: the
arms that close the commitment-vs-coverage question need tensors **no staged
artifact contains** -- ``prior``, ``refined_pre``, ``sel_score`` and the three
class logit vectors.  E-V5-2 dumped only per-window ADEs (100 kB) and
per-candidate ERRORS (``fan_err4``), never the SCORE INGREDIENTS.  So the 42-cell
(lambda, tau) surface -- which is otherwise seconds of CPU -- cannot be computed
from anything currently in the repo.

This script is the ~10-line delta section 5.6 asked for, written as a standalone so it
needs no edit to a staged harness.  It caches everything (lambda, tau) can vary
over, for BOTH the produced and the neutral goal mode (the neutral pass is what
makes GoalFlow's shadow branch measurable -- section 5.4 / E-H3).

WHERE IT MUST RUN
-----------------
Wherever the v4 checkpoint AND the PARITY val corpus live -- i.e. the eval pod,
NOT the dev box (whose episode cache is keyed ``14231cd29c74``, not the parity
key ``e438721ae894``; running here would silently change the window set and no
committed bar would reproduce).  It adds no training load: a single
``torch.no_grad`` pass over 881 windows, ~3 minutes on an A40 per goal mode
(E-V5-2 measured 165.9 s / 159.8 s for the same pass WITH imagination, which this
one does not do).

    PYTHONPATH=/workspace/TanitAD/stack \
    python eh2_build_cache.py --out /workspace/_eh2/eh2_cache.pt

Then, anywhere with a CPU:

    python eh2_lambda_tau_sweep.py --cache eh2_cache.pt --out eh2_sweep.json
    python eh3_shadow_branch.py   --cache eh2_cache.pt --out eh3_shadow.json

WHAT IT DUMPS, AND WHY EACH KEY IS LOAD-BEARING
-----------------------------------------------
``refined_pre``  [W,N]  the PRE-graft base score -- the lambda=0 arm, exactly
``lat/lon/dist`` [W,K]  the class logits -- tau acts on these and nothing else
``W_lat/lon/dist``[N,K] the graft weights -- lambda scales the map, not the logits
``sel_pen``      [W,N]  ``sel_gate * pen * vt_keep``, ALREADY assembled, so the
                        re-scorer reproduces ``sel_score`` without re-deriving the
                        longitudinal term (whose v_goal == v0 quirk is a separate
                        finding and must not be silently re-implemented here)
``sel_score``    [W,N]  the forward pass's own score -> the FIDELITY GATE
``ref_sel_idx``  [W]    the forward pass's own pick -> the fidelity gate's pick half
``fan_err``      [W,N]  per-candidate ade_0_2s -> the metric for every cell
``ep``, ``t``    [W]    episode-cluster bootstrap units
``prior``        [W,N]  the tau->0 argmax-class prior, for E-H1's deferred
                        ``O_graft(q)`` / ``H_rand(q)`` arms (section 9.4)

NOTHING IS TRAINED.  The graft weights are READ, never written.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

STACK = os.environ.get("V5_STACK", "/root/v4eval/stack")
sys.path.insert(0, STACK)
sys.path.insert(0, str(Path(STACK) / "scripts"))
sys.path.insert(0, os.environ.get("V5_TANITEVAL", "/root/taniteval"))

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
WP_STEPS = (5, 10, 15, 20)          # the 4 waypoints ade_0_2s averages over


@torch.no_grad()
def build_one(head, world4, goal_head, ds_val, sel, gm, batch, horizons):
    """One forward pass under goal mode ``gm``; cache every score ingredient."""
    import goal_modes
    import refb_labels
    from torch.utils.data import default_collate
    from train_flagship_v4 import _to_device

    keys = ("fan", "tgt", "refined_pre", "sel_score", "lat", "lon", "dist",
            "ref_sel_idx", "v0", "vt_speed", "vt_keep", "ep", "t")
    C: dict = {k: [] for k in keys}
    t0 = time.time()
    for b0 in range(0, len(sel), batch):
        idx = sel[b0:b0 + batch]
        b = _to_device(default_collate([ds_val[i] for i in idx]), DEV)
        pl = b["pose_last"].float()
        v0 = pl[:, 3]
        tgt = refb_labels.waypoint_targets(
            pl, b["future_poses"].float()[:, :max(horizons)], horizons)
        st4 = world4.encode_window(b["frames"])
        goal_kw, _rec = goal_modes.resolve_goal(
            gm, head=head, batch=b, v0=v0, states=st4, goal_head=goal_head,
            allow_fallback=False)
        out = head(st4, v0, lambda_plan=1.0, **goal_kw)
        # the PRE-GRAFT base score: head.forward OVERWRITES out["refined_logits"]
        # with the grafted score, so it has to be re-derived from the decoder.
        tokens = head.build_tokens(st4, None)
        m, _tele, vt_keep = head.condition(
            v0, goal_kw.get("vt_band"), goal_kw.get("route"),
            goal_kw.get("route_graded"))
        dec = head.decoder(tokens, m, steps=head.cfg.decoder.diffusion_steps)
        C["refined_pre"].append(dec["refined_logits"].float().cpu())
        for k in ("lat", "lon", "dist"):
            C[k].append(out[f"{k}_logits"].float().cpu())
        C["sel_score"].append(out["sel_score"].float().cpu())
        C["fan"].append(out["anchor_traj"].float().cpu())
        C["tgt"].append(tgt.float().cpu())
        C["ref_sel_idx"].append(out["sel_idx"].cpu())
        C["v0"].append(v0.cpu())
        vts = goal_kw.get("vt_speed")
        C["vt_speed"].append((vts.float() if vts is not None else v0).cpu())
        C["vt_keep"].append(vt_keep.cpu() if vt_keep is not None
                            else torch.ones_like(v0, dtype=torch.bool).cpu())
        for i in idx:
            e_i, tt = ds_val.index[i]
            C["ep"].append(int(e_i))
            C["t"].append(int(tt))
        if b0 % (batch * 10) == 0:
            print(f"  [{gm}] {b0 + len(idx)}/{len(sel)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    out_c = {k: (torch.cat(v) if v and isinstance(v[0], torch.Tensor)
                 else torch.tensor(v)) for k, v in C.items() if v}
    out_c["_wallclock_s"] = round(time.time() - t0, 1)
    return out_c


def assemble(C, head, horizons):
    """Turn one raw goal-mode cache into the sweep's input contract."""
    import v5_imagination_select as V5
    from tanitad.models.flagship_v15 import SPEED_SCALE                # noqa: F401

    fan, tgt = C["fan"], C["tgt"]
    nW, nC = fan.shape[0], fan.shape[1]
    fan_err = (V5.wp4(fan.reshape(nW * nC, fan.shape[-2], 2), horizons)
               .reshape(nW, nC, len(WP_STEPS), 2)
               - V5.wp4(tgt, horizons)[:, None]).norm(dim=-1).mean(dim=-1)

    Wl = head.lat_to_anchor.weight.detach().float().cpu()             # [N, K_lat]
    Wn = head.lon_to_anchor.weight.detach().float().cpu()
    Wd = head.dist_to_anchor.weight.detach().float().cpu()

    # sel_pen := sel_score - grafted_score, i.e. everything the longitudinal term
    # contributed. Taken as a RESIDUAL rather than re-implemented, so a change in
    # flagship_v15's selection term can never silently desynchronise the cache.
    lsm = torch.log_softmax
    graft = (lsm(C["lat"], dim=-1) @ Wl.T + lsm(C["lon"], dim=-1) @ Wn.T
             + lsm(C["dist"], dim=-1) @ Wd.T)
    base = C["refined_pre"].norm(dim=-1).clamp_min(1e-9)
    ratio = graft.norm(dim=-1) / base
    sc = head.cfg.seam_clamp / ratio.clamp_min(head.cfg.seam_clamp)
    grafted = C["refined_pre"] + graft * sc[:, None]
    sel_pen = C["sel_score"] - grafted

    prior = (Wl[:, C["lat"].argmax(1)] + Wn[:, C["lon"].argmax(1)]
             + Wd[:, C["dist"].argmax(1)]).T                          # [W, N]
    return {
        "refined_pre": C["refined_pre"], "lat": C["lat"], "lon": C["lon"],
        "dist": C["dist"], "W_lat": Wl, "W_lon": Wn, "W_dist": Wd,
        "sel_pen": sel_pen, "sel_score": C["sel_score"],
        "ref_sel_idx": C["ref_sel_idx"], "fan_err": fan_err, "prior": prior,
        "ep": C["ep"], "t": C["t"], "v0": C["v0"], "vt_keep": C["vt_keep"],
        # the emitted trajectory of the deployed pick — the shadow branch needs
        # the GEOMETRY of the two picks, not just their errors (section 5.4)
        "pick_traj": fan.gather(
            1, C["ref_sel_idx"][:, None, None, None]
            .expand(-1, 1, fan.shape[-2], 2)).squeeze(1),
        "_max_abs_selpen_check": float(sel_pen.abs().max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal-modes", default="produced,neutral")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--v4-ckpt",
                    default="/workspace/_v4gate/flagship-v4-fromscratch-30k/ckpt.pt")
    ap.add_argument("--v4-hcfg",
                    default="/workspace/_v4gate/flagship-v4-fromscratch-30k/config.json")
    ap.add_argument("--anchors", default="/root/models/flagship-v4-fromscratch-15k"
                                         "/flagship_v4_anchors_dense.pt")
    ap.add_argument("--val", default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--out", default="/workspace/_eh2/eh2_cache.pt")
    a = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    import eval_flagship_v4 as E
    cfg_e = E._eval_cfg()
    plan = E._plan(cfg_e)
    ds_val = E.build_val_dataset_v4(a.val, cfg_e, plan)
    ck4 = torch.load(a.v4_ckpt, map_location="cpu", weights_only=False)
    world4, _g4, head, _s4, _h, goal_head = E.load_v4_from_ck(
        ck4, DEV, head_config_path=a.v4_hcfg, anchors_dense_path=a.anchors)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    step = int(ck4.get("step", -1))
    del ck4
    horizons = head.cfg.horizons

    # MECHANISM CHECK FIRST: zero-init grafts would make every (lambda, tau) cell
    # identical and the sweep VACUOUS. Measure before assuming (E-V5-2 section 0.1).
    gw = {nm: {"frobenius_norm": round(float(g.weight.detach().norm()), 6),
               "max_abs": round(float(g.weight.detach().abs().max()), 6),
               "is_still_zero_init": bool(float(g.weight.detach().abs().max()) < 1e-8)}
          for nm, g in (("lat_to_anchor", head.lat_to_anchor),
                        ("lon_to_anchor", head.lon_to_anchor),
                        ("dist_to_anchor", head.dist_to_anchor))}
    if all(v["is_still_zero_init"] for v in gw.values()):
        raise SystemExit("grafts are STILL zero-init — the (lambda, tau) sweep "
                         "would be vacuous; report that, do not sweep it")

    sel = [i for i, (e, t) in enumerate(ds_val.index)
           if e < a.episodes and t % a.stride == 0]
    out: dict = {}
    for gm in [m.strip() for m in a.goal_modes.split(",")]:
        print(f"[eh2-cache] === goal mode {gm} ===", flush=True)
        C = build_one(head, world4, goal_head, ds_val, sel, gm, a.batch, horizons)
        A = assemble(C, head, horizons)
        if gm == "produced":
            out.update(A)
        else:
            out.update({f"{gm}|{k}": v for k, v in A.items()})
    out["_meta"] = json.dumps({
        "_experiment": "E-H2 / E-H3 cache — score ingredients for the (lambda, "
                       "tau) sweep and the shadow branch",
        "_evidence_class": "MEASURED (ours)",
        "_ckpt": {"path": a.v4_ckpt, "step": step},
        "_host": platform.node(), "_torch": torch.__version__,
        "_gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
        "_n_windows": len(sel), "_seam_clamp": head.cfg.seam_clamp,
        "_seam_fail": head.cfg.seam_fail,
        "_graft_mechanism_check": gw,
        "_goal_modes": a.goal_modes,
        "_parity": "val corpus must be the PARITY set; a different episode cache "
                   "key silently changes the window set and no committed bar "
                   "(0.8563 / 0.2505 / 0.6423) will reproduce.",
    }, indent=2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    print(f"[eh2-cache] -> {a.out}  ({len(sel)} windows)")


if __name__ == "__main__":
    main()
