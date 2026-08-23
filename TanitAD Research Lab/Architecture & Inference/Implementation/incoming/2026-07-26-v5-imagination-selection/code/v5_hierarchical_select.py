#!/usr/bin/env python3
"""E-V5-2 — HIERARCHICAL selection vs FLAT, as a CONDITIONING x STRUCTURE factorial.

Pre-registration: ``V5_IMAGINATION_SELECTION.md`` §0 (E-V5-2 block).

WHY A FACTORIAL AND NOT JUST "hierarchical vs flat"
---------------------------------------------------
Bar A's follow-up measured that the in-sample re-scoring ceiling moves
**0.4907 -> 0.4138 when the SCORE IS GIVEN GOAL INFORMATION**, while changing the
*objective* moved it 0.0317 / 0.0089 and in the wrong direction — information beat
objective by 2.4x-8.6x (INHERITED from the Bar-A agent; re-derived here on the
`oracle_in_fan` / selected-ADE decomposition, not quoted bare).  A "hierarchical
vs flat" test that does not hold conditioning fixed would confound the two.  So:

  AXIS 1 -- CONDITIONING (what the selector KNOWS)
      neutral   : no goal.  The `nav_cmd=None` analogue -- the documented confound
                  behind the retracted "strategic choice is a ~2% lever" claim.
      produced  : goal_head's own route / route_graded / vt_band.  DEPLOYABLE.
      oracle    : goal minted from the ego's OWN future poses.  NEVER deployable;
                  reported only as the paired upper bound (3 channels are minted;
                  `vt_speed` is overwritten with the observed v0).

  AXIS 2 -- STRUCTURE (how the selection is ORGANISED over the same 256 candidates)
      F_flat        : argmax of the full as-trained score.  1-of-256.
      F_base_only   : argmax of the pre-graft refined score (grafts + vt penalty off).
      H_graft(q)    : COMMIT to the tactical class first (argmax lat/lon/dist),
                      restrict to the top-q anchors that class favours, then rank
                      within by the base score.  The hierarchy the program claims.
      H_imag(q)     : identical commit, but the OPERATIVE ranker is E-V5-1's
                      imagination cost.  Hierarchy + imagination.

THE CONFOUND THAT MUST BE HELD, AND IS
--------------------------------------
The goal conditions the FAN as well as the selector, so a goal-mode difference in
selected ADE mixes proposal quality with selection quality.  `oracle_in_fan` is
therefore reported per goal mode next to every selected number, and the
decomposition (fan effect vs selection effect) is computed explicitly.

NOTHING IS TRAINED.  The graft weights are READ, never written, so the head's own
`seam_fail = 1.5` guard cannot fire — this scorer lives entirely outside the graft
path (it consumes `anchor_traj` and the world model only).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import v5_imagination_select as V5  # noqa: E402  (single source of truth)

DEV = V5.DEV
K_MAX, WP_STEPS, SEED = V5.K_MAX, V5.WP_STEPS, V5.SEED


@torch.no_grad()
def build(head, world4, world_s, grounding_s, goal_head, ds_val, sel, gm,
          batch, do_imag, imag_windows):
    """Cache the fan + every selection ingredient under goal mode ``gm``."""
    import goal_modes
    import refb_labels
    from torch.utils.data import default_collate
    from train_flagship_v4 import _to_device
    from tanitad.models.flagship_v15 import SPEED_SCALE

    horizons = head.cfg.horizons
    step_readout = grounding_s.step["op"]
    keys = ("fan", "tgt", "refined_pre", "sel_score", "lat", "lon", "dist",
            "ref_sel_idx", "v0", "vt_speed", "vt_keep", "ep", "t", "imag")
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
        # the PRE-GRAFT base score: re-derive it from the decoder, since
        # head.forward overwrites out["refined_logits"] with the grafted score.
        tokens = head.build_tokens(st4, None)
        m, _tele, vt_keep = head.condition(
            v0, goal_kw.get("vt_band"), goal_kw.get("route"),
            goal_kw.get("route_graded"))
        dec = head.decoder(tokens, m, steps=head.cfg.decoder.diffusion_steps)
        C["refined_pre"].append(dec["refined_logits"].float().cpu())
        C["lat"].append(out["lat_logits"].float().cpu())
        C["lon"].append(out["lon_logits"].float().cpu())
        C["dist"].append(out["dist_logits"].float().cpu())
        C["sel_score"].append(out["sel_score"].float().cpu())
        C["fan"].append(out["anchor_traj"].float().cpu())        # fp32 -- fp16
        C["tgt"].append(tgt.float().cpu())                       # flips argmax
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
        if do_imag:
            st_s = (st4 if world_s is world4
                    else world_s.encode_window(b["frames"]))
            aw2 = b["actions"].float()
            vch = (v0 / SPEED_SCALE)[:, None, None]
            aw = torch.cat([aw2, vch.expand(-1, aw2.shape[1], -1)], dim=-1)
            fanb = out["anchor_traj"].float()
            nC = fanb.shape[1]
            ims = []
            for s0 in range(0, fanb.shape[0], imag_windows):
                s = slice(s0, min(s0 + imag_windows, fanb.shape[0]))
                mm = s.stop - s.start
                acts = V5.traj_to_actions(
                    fanb[s], v0[s][:, None].expand(mm, nC))
                wp = V5.imagine(
                    world_s.predictor, step_readout,
                    st_s[s].repeat_interleave(nC, 0),
                    aw[s].repeat_interleave(nC, 0),
                    acts.reshape(mm * nC, K_MAX, 3), K_MAX)
                ims.append(wp.reshape(mm, nC, K_MAX, 2).cpu())
            C["imag"].append(torch.cat(ims))
        if b0 % (batch * 10) == 0:
            print(f"  [{gm}] {b0 + len(idx)}/{len(sel)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    out_c = {k: (torch.cat(v) if v and isinstance(v[0], torch.Tensor)
                 else torch.tensor(v)) for k, v in C.items() if v}
    out_c["_wallclock_s"] = round(time.time() - t0, 1)
    return out_c


def hierarchical_pick(C, head, q, operative_cost=None):
    """COMMIT to the tactical class, then rank the OPERATIVE candidates inside it.

    Tactical commit: argmax of each factorised head (lat / lon / dist).  The
    admissible operative set is the top-``q`` anchors by the SUM of the three
    class-conditional anchor priors the head itself learned
    (``lat_to_anchor.weight[:, cls]`` etc.).  Then rank inside that set by the
    base score (or by ``operative_cost``, lower = better).

    This is a genuine hierarchy: a high base score OUTSIDE the committed class can
    no longer win, which is exactly what the flat 1-of-256 argmax allows.
    """
    lat = C["lat"].argmax(dim=1)                                 # [W]
    lon = C["lon"].argmax(dim=1)
    dist = C["dist"].argmax(dim=1)
    Wl = head.lat_to_anchor.weight.detach().float().cpu()        # [N_anchor, N_lat]
    Wn = head.lon_to_anchor.weight.detach().float().cpu()
    Wd = head.dist_to_anchor.weight.detach().float().cpu()
    prior = (Wl[:, lat] + Wn[:, lon] + Wd[:, dist]).T            # [W, N_anchor]
    n = prior.shape[1]
    q = min(q, n)
    adm = prior.topk(q, dim=1).indices                           # [W, q]
    mask = torch.zeros_like(prior, dtype=torch.bool)
    mask.scatter_(1, adm, True)
    cost = (-C["refined_pre"] if operative_cost is None else operative_cost)
    cost = cost.clone()
    cost[~mask] = float("inf")
    return cost.argmin(dim=1), prior


_EPCACHE: dict = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal-modes", default="produced,neutral,oracle")
    ap.add_argument("--q", default="8,16,32,64")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imag-windows", type=int, default=8)
    ap.add_argument("--no-imag", action="store_true")
    ap.add_argument("--imag-dump",
                    default="/workspace/_v5/v5_v4_windows_reduced.pt",
                    help="reuse E-V5-1's produced-mode imagination cost "
                         "(identity-checked before use)")
    ap.add_argument("--v4-ckpt",
                    default="/workspace/_v4gate/flagship-v4-fromscratch-30k/ckpt.pt")
    ap.add_argument("--v4-hcfg",
                    default="/workspace/_v4gate/flagship-v4-fromscratch-30k/config.json")
    ap.add_argument("--anchors", default="/root/models/flagship-v4-fromscratch-15k"
                                         "/flagship_v4_anchors_dense.pt")
    ap.add_argument("--val", default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--out", default="/workspace/_v5/v5_hier")
    a = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_start = time.time()

    import eval_flagship_v4 as E
    R: dict = {
        "_experiment": "E-V5-2 -- HIERARCHICAL vs FLAT selection as a "
                       "CONDITIONING x STRUCTURE factorial over the frozen v4 fan",
        "_evidence_class": "MEASURED (ours)",
        "_estimator": "paired_episode_cluster_bootstrap (B=2000, unit = episode "
                      "cluster). NEVER overlapping_holdout_se.",
        "_host": platform.node(), "_python": platform.python_version(),
        "_torch": torch.__version__, "_stack_root": STACK,
        "_gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
        "_goal_oracle_honesty": "route / route_graded / vt_band are minted from "
                                "the ego's OWN future poses (3 channels; vt_speed "
                                "is overwritten with the observed v0). Oracle rows "
                                "are an upper bound, NEVER deployed capability.",
        "_committed_bars": V5.COMMITTED,
    }
    cfg_e = E._eval_cfg()
    plan = E._plan(cfg_e)
    ds_val = E.build_val_dataset_v4(a.val, cfg_e, plan)
    ck4 = torch.load(a.v4_ckpt, map_location="cpu", weights_only=False)
    R["_ckpt_v4"] = {"path": a.v4_ckpt, "md5": V5.md5(a.v4_ckpt),
                     "step": int(ck4.get("step", -1))}
    world4, grounding4, head, step4, hcfg, goal_head = E.load_v4_from_ck(
        ck4, DEV, head_config_path=a.v4_hcfg, anchors_dense_path=a.anchors)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    del ck4
    horizons = head.cfg.horizons
    R["_module_provenance"] = V5.mod_prov([
        "eval_flagship_v4", "train_flagship_v4", "goal_modes",
        "tanitad.models.flagship_v4", "tanitad.models.flagship_v15",
        "tanitad.models.metric_dynamics", "flagship_v4_data"])

    # ---- THE MECHANISM CHECK, first: are the grafts even non-zero? ---------
    # They are ZERO-INIT by construction (ReZero discipline). If training never
    # moved them, "hierarchical selection" over them is VACUOUS and any
    # hierarchical result is a statement about noise. Measure before assuming.
    gw = {}
    for nm, g in (("lat_to_anchor", head.lat_to_anchor),
                  ("lon_to_anchor", head.lon_to_anchor),
                  ("dist_to_anchor", head.dist_to_anchor)):
        w = g.weight.detach().float().cpu()
        gw[nm] = {"frobenius_norm": round(float(w.norm()), 6),
                  "max_abs": round(float(w.abs().max()), 6),
                  "mean_abs": round(float(w.abs().mean()), 6),
                  "shape": list(w.shape),
                  "is_still_zero_init": bool(float(w.abs().max()) < 1e-8)}
    R["graft_mechanism_check"] = {
        **gw,
        "_read": "the three grafts are the ONLY learned class->anchor mapping in "
                 "the head. If they are still at their zero init, a hierarchical "
                 "selector built on them is vacuous and must be reported as such.",
        "seam_note": "this experiment READS the grafts and never writes them, so "
                     "the head's seam_fail=1.5 guard cannot fire here.",
    }
    print(f"[hier] graft norms: "
          f"{ {k: v['frobenius_norm'] for k, v in gw.items()} }", flush=True)

    sel = [i for i, (e, t) in enumerate(ds_val.index)
           if e < a.episodes and t % a.stride == 0]
    qs = [int(x) for x in a.q.split(",")]
    modes = [m.strip() for m in a.goal_modes.split(",")]
    R["_design"] = {"n_windows": len(sel), "goal_modes": modes, "q_grid": qs,
                    "structures": ["F_flat", "F_base_only",
                                   *[f"H_graft_q{q}" for q in qs],
                                   *[f"H_imag_q{q}" for q in qs]]}

    results: dict = {}
    ade_store: dict = {}
    for gm in modes:
        print(f"[hier] === goal mode {gm} ===", flush=True)
        C = build(head, world4, world4, grounding4, goal_head, ds_val, sel, gm,
                  a.batch, not a.no_imag, a.imag_windows)
        ep = C["ep"].numpy()
        _EPCACHE["ep"] = ep
        fan = C["fan"].to(DEV)
        tgt = C["tgt"].to(DEV)
        nW, nC = fan.shape[0], fan.shape[1]
        fan_err4 = (V5.wp4(fan.reshape(nW * nC, K_MAX, 2), horizons)
                    .reshape(nW, nC, len(WP_STEPS), 2)
                    - V5.wp4(tgt, horizons)[:, None]).norm(dim=-1).mean(dim=-1)

        picks = {"F_flat": C["sel_score"].argmax(dim=1),
                 "F_base_only": C["refined_pre"].argmax(dim=1),
                 "O_oracle_in_fan": fan_err4.argmin(dim=1).cpu()}
        icost = None
        if "imag" in C:
            icost = (C["imag"].to(DEV) - fan).norm(dim=-1).mean(dim=-1).cpu()
        elif gm == "produced" and a.imag_dump and Path(a.imag_dump).exists():
            # EXACT reuse: E-V5-1's produced-mode run used the SAME checkpoint,
            # windows and goal mode, so its fan -- and therefore its per-candidate
            # imagination cost -- is the identical object. Verified, not assumed:
            # the window keys and the as-trained pick must match element-wise.
            Dm = torch.load(a.imag_dump, map_location="cpu", weights_only=False)
            same = bool(torch.equal(Dm["ep"], C["ep"])
                        and torch.equal(Dm["t"], C["t"])
                        and torch.equal(Dm["ref_sel_idx"], C["ref_sel_idx"]))
            R.setdefault("_imag_reuse", {})[gm] = {
                "source": a.imag_dump, "window_and_pick_identity": same}
            if same:
                icost = Dm["costs"]["A1_imag_consistency"].float()
            else:
                print("[hier] REFUSING to reuse imagination: window/pick "
                      "identity failed", flush=True)
            del Dm
        if icost is not None:
            picks["A1_imag_flat"] = icost.argmin(dim=1)
        for q in qs:
            picks[f"H_graft_q{q}"], prior = hierarchical_pick(C, head, q)
            if icost is not None:
                picks[f"H_imag_q{q}"], _ = hierarchical_pick(
                    C, head, q, operative_cost=icost)
        # fidelity: F_flat must reproduce the head's own forward-pass pick
        agree = float((picks["F_flat"] == C["ref_sel_idx"]).float().mean())

        rows: dict = {}
        for nm, idx in picks.items():
            ade, miss, along, cross = V5.pick_metrics(
                fan, idx.to(DEV), tgt, horizons)
            ade_store[(gm, nm)] = ade
            rows[nm] = {
                "ade_0_2s": round(float(ade.mean()), 4),
                "miss_at_2m": round(float(miss.mean()), 4),
                "along_abs_dense_LONGITUDINAL": round(float(along.mean()), 4),
                "cross_abs_dense_LATERAL": round(float(cross.mean()), 4),
                "n_distinct_picks": int(len(torch.unique(idx))),
                "frac_equals_flat": round(
                    float((idx == picks["F_flat"]).float().mean()), 4),
                "ci": V5.single_ci(ade, ep),
            }
        base = ade_store[(gm, "F_flat")]
        results[gm] = {
            "_selection_fidelity_vs_forward_pass": round(agree, 6),
            "_oracle_in_fan_4wp": round(float(fan_err4.min(dim=1).values.mean()), 4),
            "_cache_wallclock_s": C["_wallclock_s"],
            "arms": rows,
            "paired_vs_F_flat": {
                nm: V5.paired_ci(v, base, ep) for nm, v in
                ((n, ade_store[(gm, n)]) for n in picks) if nm != "F_flat"},
        }
        print(f"[hier] {gm}: " + json.dumps(
            {k: v["ade_0_2s"] for k, v in rows.items()}), flush=True)
        del fan, tgt, fan_err4, C
        torch.cuda.empty_cache()
    R["by_goal_mode"] = results

    # ---- AXIS 1 isolated: CONDITIONING, holding structure fixed ------------
    # and the fan-vs-selection decomposition that keeps it honest.
    cond: dict = {}
    if "produced" in results:
        ep_all = _EPCACHE["ep"]                 # identical windows in every mode
        for other in ("neutral", "oracle"):
            if other not in results:
                continue
            row = {}
            for struct in results["produced"]["arms"]:
                if struct not in results[other]["arms"]:
                    continue
                row[struct] = V5.paired_ci(ade_store[(other, struct)],
                                           ade_store[("produced", struct)],
                                           ep_all)
            cond[f"{other}_minus_produced"] = row
            cond[f"{other}_fan_effect"] = {
                "oracle_in_fan_delta": round(
                    results[other]["_oracle_in_fan_4wp"]
                    - results["produced"]["_oracle_in_fan_4wp"], 4),
                "selected_flat_delta": round(
                    results[other]["arms"]["F_flat"]["ade_0_2s"]
                    - results["produced"]["arms"]["F_flat"]["ade_0_2s"], 4),
                "_read": "the goal conditions the FAN as well as the selector. A "
                         "goal-mode difference in selected ADE that is not matched "
                         "by an oracle_in_fan difference is a SELECTION effect.",
            }
    R["axis1_conditioning"] = cond
    R["_wallclock_total_s"] = round(time.time() - t_start, 1)
    Path(f"{a.out}.json").write_text(json.dumps(R, indent=2))
    torch.save({k: torch.tensor(v) for k, v in
                {f"{g}|{n}": val for (g, n), val in ade_store.items()}.items()},
               f"{a.out}_windows.pt")
    print(json.dumps({g: {k: v["ade_0_2s"] for k, v in r["arms"].items()}
                      for g, r in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
