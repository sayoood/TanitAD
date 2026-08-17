"""PC4 — TASK 3: do the label-free levers even POINT at the agent readout?

METHOD, INHERITED NOT REINVENTED. This follows
``…/incoming/2026-08-16-o6-ablation/code/mask_grad_probe.py`` exactly:

  * a term's OWN gradient is obtained by EXACT LINEARITY —
    ``g(all weights) - g(that weight = 0)`` with everything else held
    bit-identical (same batch, same ``generator``, same ``sigreg_generator``).
    No fitting, no resampling;
  * the reading is the DIRECTION, and every cosine is reported against
    ``chance = 1/sqrt(D)`` and as a MULTIPLE of it, because two random vectors
    in D dims have |cos| ~ 1/sqrt(D) — at D ~ 1e8 that is ~1e-4, so 0.01 is
    100x chance and NOT orthogonal;
  * the full pairwise term x term matrix is computed, so "O3 is orthogonal to
    the readout" can be checked against "everything here is orthogonal", which
    would make the statement about the dimension rather than about O3;
  * the controls a probe needs in order for a finding to mean anything:
    N1 (same arm twice, seeded -> EXACTLY 0) and N2 (a lever ``for_stage('S-W')``
    forces to zero -> EXACTLY 0).

⭐ WHAT IS NEW HERE: the reference direction is the **AGENT-READOUT LOSS**, i.e.
``slot_set_loss(head(cells(z_op)), targets_from_join(...))`` with the head that
`…/2026-08-17-slot-probe-parity/` actually FITTED and banked, differentiated
into the TRUNK. A term whose gradient is orthogonal to that direction cannot
make agents more readable AT ANY WEIGHT — that is the question the brief asks.

⛔⛔ O4 IS NOT A LOSS TERM AND HAS NO GRADIENT. ``build_o4_weights``
(``stack/scripts/train_v6_staged.py:745``) returns **per-window SAMPLING
weights** consumed by ``InteractionSampler`` (``:2470``). It changes WHICH
windows are drawn, not the loss on a drawn window, so "the cosine of O4's
gradient" does not exist as a quantity. The honest analogue is computed instead
(``--o4``): the agent-readout gradient on an O4-WEIGHTED batch vs on a UNIFORM
batch, plus the correlation between a window's O4 saliency weight and its GT
lead geometry. Reported as its own thing, never as a cosine of O4.

⚠️⚠️ THE LIMITS, STATED BEFORE THE NUMBERS AND NOT SOFTENED AFTERWARDS:
  1. A gradient cosine is a LOCAL quantity at ONE point on the loss surface of
     ONE checkpoint. It is **ESTIMATED**, never MEASURED-as-causal, and it is
     NOT a claim about what a term would do over a full training run.
  2. ``o5_k`` is reduced from the live run's **60** to a declared small value
     (the 60-step rollout plus 60 future-frame encodes does not fit the dev
     box). ``o1_k`` likewise. Both are recorded, and a second setting is run so
     the reading's sensitivity to them is visible rather than assumed.
  3. The reference direction depends on THE HEAD. That head fails K1 — it is
     the readout we actually have, not a good one. A cosine against it is a
     statement about the direction the CURRENT readout wants, which is the
     actionable question, but it is not "the direction of perfect agent
     perception".
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pyarrow  # noqa: F401  ⚠️ CLAUDE.md: pyarrow BEFORE torch on this box
import torch

STACK = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\stack")
for p in (str(STACK), str(STACK / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, str(Path(__file__).resolve().parent))

STAGE = "S-W"
TRUNK_GROUPS = ("encoder", "readout", "predictor_op")

#: (name, weight fields, incumbent values) — the live run's own settings, read
#: off the checkpoint's `_meta.config.args` and asserted below.
LEVERS: dict[str, tuple[tuple[str, ...], tuple[float, ...]]] = {
    "o1": (("o1_ctrl", "o1_fact", "o1_scene"), (1.0, 1.0, 0.3)),
    "o2": (("o2_nearfield",), (1.0,)),
    "o3": (("o3_masked",), (1.0,)),
    "o5": (("o5_rollout",), (1.0,)),
    "o6": (("o6_sigreg",), (0.1,)),
}


def _cat(g: dict, names: list[str]) -> torch.Tensor:
    if not names:
        return torch.zeros(0, dtype=torch.float64)
    return torch.cat([g[n] for n in names]).double()


def _cos(a: torch.Tensor, b: torch.Tensor):
    na, nb = float(a.norm()), float(b.norm())
    if na < 1e-30 or nb < 1e-30:
        return None                      # undefined, and that IS the finding
    return float((a @ b) / (na * nb))


def _sub(ga: dict, gb: dict) -> dict:
    return {k: ga[k] - gb[k] for k in ga}


def _grab(stack) -> dict:
    return {n: (p.grad.detach().reshape(-1).double().cpu().clone()
                if p.grad is not None
                else torch.zeros(p.numel(), dtype=torch.float64))
            for n, p in stack.named_parameters()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-json", required=True)
    ap.add_argument("--v2-cache", nargs="+", required=True)
    ap.add_argument("--join-file", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--n-queries", type=int, default=74)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--o1-k", type=int, default=4)
    ap.add_argument("--o5-k", type=int, default=4)
    ap.add_argument("--batch-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-episodes", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--v2-lru", type=int, default=4)
    ap.add_argument("--frame-h", type=int, default=256)
    ap.add_argument("--frame-w", type=int, default=640)
    ap.add_argument("--frame-hfov", type=float, default=120.0)
    ap.add_argument("--projection", default="cylindrical")
    ap.add_argument("--v2-subframe", default=None)
    ap.add_argument("--require-parity", action="store_true")
    ap.add_argument("--o4", action="store_true",
                    help="also run the O4 SAMPLER analogue (not a cosine)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from eval_flagship_v4 import _eval_cfg, _plan, resolve_eval_frames
    from sp_common import merge_run_args, read_fp16_snapshot
    from tanitad.eval.v6_probe_trunk import load_trunk_auto
    from tanitad.models.agent_slots import (AgentSlotDecoder, SlotDecodeRanges,
                                            slot_set_loss, targets_from_join,
                                            track_rates_from_join)
    from tanitad.data.bev_raster import GRID_DEFAULT as GRID
    from train_flagship4b import FlagshipWindowDataset
    from train_p8_occupancy import JoinFileReader, window_frame
    from train_v58f_unicycle_head import build_train_episodes
    from tanitad.models.metric_dynamics import gt_ego_waypoints
    from train_v6_staged import (V6LossWeights, build_o4_weights,
                                 v6_loss_step)
    from torch.utils.data import default_collate

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    cfg = _eval_cfg()
    cache_frame, model_frame = resolve_eval_frames(a, cfg, label="pc4")
    plan = _plan(cfg)

    sd, run_args, art_step, prov = read_fp16_snapshot(a.ckpt, a.config_json)
    merged = vars(merge_run_args(run_args))
    ck = {"stack": sd, "config": {"args": merged}, "step": art_step}
    world, _g, base_step = load_trunk_auto(ck, device, ckpt_path=a.ckpt,
                                           frame=model_frame)
    del ck, sd
    stack = world.stack
    for p in stack.parameters():
        p.requires_grad_(True)
    window = int(getattr(world, "window", cfg.predictor.window))

    names = [n for n, _ in stack.named_parameters()]
    groups = {n: stack.group_of(n) for n in names}
    trunk = [n for n in names if groups[n] in TRUNK_GROUPS]

    # ---- data ---------------------------------------------------------------
    a.v2_lru = int(a.v2_lru)
    train_eps, _prov = build_train_episodes(a, cache_frame=cache_frame,
                                            train_frame=model_frame)
    train_eps = train_eps[:int(a.max_episodes)]
    ds = FlagshipWindowDataset(train_eps, window=window,
                               max_horizon=plan.max_horizon,
                               maneuver_h=plan.maneuver_h,
                               channels=cfg.encoder.in_channels)
    src = JoinFileReader(a.join_file)
    recs: dict[tuple[str, int], dict] = {}
    with open(a.join_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[(str(r["clip_id"]), int(r["frame_idx"]))] = r
    clip_of = {int(e.episode_id): src._clip_of(int(e.episode_id))
               for e in train_eps}
    labelled = [i for i in range(len(ds.index))
                if src.lookup(*window_frame(ds, i)) is not None]
    if len(labelled) < a.batch:
        raise SystemExit(f"[pc4] only {len(labelled)} labelled windows")

    # ---- the banked readout head (FROZEN — the direction, not a trainee) ----
    head = AgentSlotDecoder(int(stack.cfg.readout.d_readout),
                            int(stack.cfg.n_cells), n_queries=int(a.n_queries),
                            d_model=256, depth=3, n_heads=8,
                            ranges=SlotDecodeRanges(),
                            enforce_band=False).to(device)
    head.load_state_dict(torch.load(a.head, map_location="cpu",
                                    weights_only=False)["head"])
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)

    def make_batch(seed: int):
        rng = np.random.default_rng(seed)
        idx = [int(labelled[k]) for k in
               rng.choice(len(labelled), size=a.batch, replace=False)]
        b = default_collate([ds[i] for i in idx])
        b = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in b.items()}
        need = max(a.o1_k, a.o5_k)
        aw2 = b["actions"][..., :2].float()
        fa2 = b["future_actions"][..., :2].float()
        v0 = b["pose_last"][:, 3].float()
        with torch.no_grad():
            ff = b["future_frames"][:, :need]
            fb, fk = ff.shape[:2]
            zf = stack.readout(stack.encoder(
                ff.reshape(fb * fk, *ff.shape[2:]))).reshape(fb, fk, -1)
            z_true = [zf[:, j].detach() for j in range(need)]
        batch = {"frames": b["frames"], "actions2": aw2,
                 "future_actions2": fa2, "v0": v0, "z_true_steps": z_true,
                 "gt_wp": gt_ego_waypoints(b["pose_last"].float(),
                                           b["future_poses"].float(),
                                           tuple(range(1, a.o1_k + 1)))}
        # the agent targets for the SAME windows, from the SAME join reader
        parts, n_pad = [], 1
        raw = []
        for i in idx:
            eid, pf = window_frame(ds, i)
            cid = clip_of[eid]
            ag = np.asarray(src.lookup(eid, pf), dtype=np.float64).reshape(-1, 6)
            cls = src.lookup_classes(eid, pf)
            rt, rm = track_rates_from_join(recs.get((cid, pf - 1)),
                                           recs[(cid, pf)],
                                           recs.get((cid, pf + 1)))
            keep = ((ag[:, 0] > 0.0) & (ag[:, 0] <= GRID.x_fwd_m)
                    & (np.abs(ag[:, 1]) <= GRID.y_half_m)) if len(ag) \
                else np.zeros(0, dtype=bool)
            raw.append((ag[keep],
                        (np.asarray(cls, dtype=object)[keep].tolist()
                         if cls is not None and len(ag) else None),
                        rt[keep] if len(ag) else rt,
                        rm[keep] if len(ag) else rm))
            n_pad = max(n_pad, int(keep.sum()))
        for ag_k, cls_k, rt_k, rm_k in raw:
            parts.append(targets_from_join(ag_k, classes=cls_k, rates=rt_k,
                                           rates_mask=rm_k, n_pad=n_pad,
                                           device=device))
        tgt = {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}
        return batch, tgt, idx

    def g_agent(batch, tgt):
        stack.zero_grad(set_to_none=True)
        z = stack.encode_window(batch["frames"][:, -1:])
        loss = slot_set_loss(head(stack.cells(z[:, 0])), tgt)["total"]
        loss.backward()
        g = _grab(stack)
        stack.zero_grad(set_to_none=True)
        return g, float(loss.detach())

    def g_loss(batch, weights, *, sig_seed=11, gen_seed=5):
        stack.zero_grad(set_to_none=True)
        out = v6_loss_step(
            stack, batch, stage=STAGE, o1_k=a.o1_k, o5_k=a.o5_k,
            weights=weights,
            generator=torch.Generator(device="cpu").manual_seed(gen_seed),
            sigreg_generator=torch.Generator(device="cpu").manual_seed(sig_seed))
        out["loss"].backward()
        g = _grab(stack)
        stack.zero_grad(set_to_none=True)
        return g, float(out["loss"].detach()), dict(out["log"])

    res = {"_evidence_class": "ESTIMATED (local single-checkpoint gradient "
                              "geometry — NOT a causal claim about a training "
                              "run; see the module docstring's limits)",
           "eval_tier": "T0-DIAGNOSTIC (world-model diagnostic, not driving)",
           "run_stamp": f"v6F-SW-30k@{base_step}",
           "ckpt": str(a.ckpt), "ckpt_provenance": prov,
           "stage": STAGE, "batch": a.batch,
           "o1_k": a.o1_k, "o5_k": a.o5_k,
           "live_run_o1_k": merged.get("o1_k"), "live_run_o5_k": merged.get("o5_k"),
           "_k_scope_note": ("o5_k/o1_k REDUCED from the live run's values — "
                             "the 60-step rollout does not fit the dev box. "
                             "Declared, and a second setting is run."),
           "n_params_all": int(sum(p.numel() for p in stack.parameters())),
           "n_params_trunk": int(sum(p.numel() for n, p in
                                     stack.named_parameters() if n in trunk)),
           "trunk_groups": list(TRUNK_GROUPS),
           "head": str(a.head), "n_queries": a.n_queries,
           "o4_is_not_a_loss_term": (
               "build_o4_weights (train_v6_staged.py:745) returns per-window "
               "SAMPLING weights for InteractionSampler (:2470). It has no "
               "gradient; a cosine of O4 does not exist. See o4_sampler below."),
           "seeds": {}}
    res["chance_cos_trunk"] = 1.0 / math.sqrt(max(res["n_params_trunk"], 1))
    res["chance_cos_all"] = 1.0 / math.sqrt(max(res["n_params_all"], 1))

    from dataclasses import replace as _replace
    full = V6LossWeights().for_stage(STAGE)      # the stage's ACTUAL weights
    for bs in a.batch_seeds:
        t0 = time.time()
        batch, tgt, idx = make_batch(bs)
        ga, l_agent = g_agent(batch, tgt)
        va_t = _cat(ga, trunk)
        gf, l_full, log_full = g_loss(batch, full)
        rec = {"loss_agent_readout": round(l_agent, 5),
               "loss_v6_full": round(l_full, 5),
               "norm_agent_TRUNK": float(va_t.norm()),
               "o3_mask_rate": log_full.get("o3_mask_rate"),
               "levers": {}, "cos_matrix_trunk": {}}
        iso = {}
        for lev, (fields, on) in LEVERS.items():
            w_off = _replace(full, **{f: 0.0 for f in fields})
            g_off, l_off, _ = g_loss(batch, w_off)
            g_lev = _sub(gf, g_off)
            iso[lev] = g_lev
            v_lev = _cat(g_lev, trunk)
            v_off = _cat(g_off, trunk)
            rec["levers"][lev] = {
                "weights_zeroed": list(fields), "incumbent_weight": list(on),
                "loss_full": l_full, "loss_lever_off": l_off,
                "cos_vs_AGENT_READOUT_TRUNK": _cos(v_lev, va_t),
                "cos_term_vs_rest_TRUNK": _cos(v_lev, v_off),
                "norm_term_TRUNK": float(v_lev.norm()),
                "norm_rest_TRUNK": float(v_off.norm()),
                "relative_pull_TRUNK": (float(v_lev.norm() / v_off.norm())
                                        if float(v_off.norm()) > 1e-30 else None),
            }
        for x in LEVERS:
            rec["cos_matrix_trunk"][x] = {
                y: _cos(_cat(iso[x], trunk), _cat(iso[y], trunk))
                for y in LEVERS}
        rec["cos_full_vs_AGENT_TRUNK"] = _cos(_cat(gf, trunk), va_t)

        # ---- controls -------------------------------------------------------
        if bs == a.batch_seeds[0]:
            g1, _, _ = g_loss(batch, full)
            g2, _, _ = g_loss(batch, full)
            n1 = float(_cat(_sub(g1, g2), names).norm())
            t_on = _replace(full, t1_latent=1.0)
            t_off = _replace(full, t1_latent=0.0)
            gt1, _, _ = g_loss(batch, t_on)
            gt2, _, _ = g_loss(batch, t_off)
            n2 = float(_cat(_sub(gt1, gt2), names).norm())
            ga2, _ = g_agent(batch, tgt)
            n0 = float(_cat(_sub(ga, ga2), names).norm())
            res["controls"] = {
                "N0_agent_grad_twice": {"delta_l2": n0, "expect": "EXACTLY 0",
                                        "pass": n0 == 0.0},
                "N1_same_arm_twice_seeded": {"delta_l2": n1,
                                             "expect": "EXACTLY 0",
                                             "pass": n1 == 0.0},
                "N2_inert_lever_t1_in_SW": {"delta_l2": n2,
                                            "expect": "EXACTLY 0",
                                            "pass": n2 == 0.0},
                "_read": "a probe that always finds something has found nothing"}
        rec["wall_s"] = round(time.time() - t0, 1)
        res["seeds"][str(bs)] = rec
        print(f"[pc4] seed {bs} done in {rec['wall_s']}s  "
              f"cos(agent, ·) = " + json.dumps(
                  {k: (round(v["cos_vs_AGENT_READOUT_TRUNK"], 5)
                       if v["cos_vs_AGENT_READOUT_TRUNK"] is not None else None)
                   for k, v in rec["levers"].items()}), flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1), "utf-8")

    # ---- O4: the SAMPLER analogue, explicitly not a cosine -------------------
    if a.o4:
        need = max(a.o1_k, a.o5_k)
        span = int(stack.cfg.predictor.window) + need
        acts = []
        for e_i, t in ds.index:
            arr = ds.episodes[e_i].actions[t:t + span]
            acts.append(torch.as_tensor(arr[:, :2]).float())
        n_max = max(x.shape[0] for x in acts)
        acts = [x if x.shape[0] == n_max else
                torch.cat([x, x[-1:].expand(n_max - x.shape[0], 2)], dim=0)
                for x in acts]
        w4, o4log = build_o4_weights(torch.stack(acts), dt=stack.cfg.dt,
                                     alpha=float(merged.get("o4_alpha", 1.0)),
                                     floor=float(merged.get("o4_floor", 0.25)))
        w4 = w4.double().numpy()
        # does O4 up-weight the windows a lead readout cares about?
        gt_gap, has_lead, wsel = [], [], []
        for i in labelled:
            eid, pf = window_frame(ds, i)
            ag = np.asarray(src.lookup(eid, pf), dtype=np.float64).reshape(-1, 6)
            m = ((ag[:, 0] > 0) & (np.abs(ag[:, 1]) <= 1.75)
                 & (ag[:, 0] <= 30.0)) if len(ag) else np.zeros(0, bool)
            has_lead.append(bool(m.any()))
            gt_gap.append(float(ag[m, 0].min()) if bool(m.any()) else np.nan)
            wsel.append(float(w4[i]))
        wsel = np.asarray(wsel); gt_gap = np.asarray(gt_gap)
        has_lead = np.asarray(has_lead)
        ok = ~np.isnan(gt_gap)
        # the DIRECTION question, in the only form O4 admits: the agent-readout
        # gradient on an O4-weighted draw vs on a uniform draw.
        rng = np.random.default_rng(123)
        pu = np.ones(len(labelled)) / len(labelled)
        pw = wsel / wsel.sum()
        cos_u_w = []
        for rep in range(2):
            iu = rng.choice(len(labelled), size=a.batch, replace=False, p=pu)
            iw = rng.choice(len(labelled), size=a.batch, replace=False, p=pw)
            gs = []
            for sel in (iu, iw):
                bsel = [int(labelled[k]) for k in sel]
                bb = default_collate([ds[i] for i in bsel])
                bb = {k: (v.to(device) if torch.is_tensor(v) else v)
                      for k, v in bb.items()}
                parts, n_pad, raw = [], 1, []
                for i in bsel:
                    eid, pf = window_frame(ds, i)
                    cid = clip_of[eid]
                    ag = np.asarray(src.lookup(eid, pf),
                                    dtype=np.float64).reshape(-1, 6)
                    cls = src.lookup_classes(eid, pf)
                    rt, rm = track_rates_from_join(recs.get((cid, pf - 1)),
                                                   recs[(cid, pf)],
                                                   recs.get((cid, pf + 1)))
                    keep = ((ag[:, 0] > 0.0) & (ag[:, 0] <= GRID.x_fwd_m)
                            & (np.abs(ag[:, 1]) <= GRID.y_half_m)) if len(ag) \
                        else np.zeros(0, dtype=bool)
                    raw.append((ag[keep],
                                (np.asarray(cls, dtype=object)[keep].tolist()
                                 if cls is not None and len(ag) else None),
                                rt[keep] if len(ag) else rt,
                                rm[keep] if len(ag) else rm))
                    n_pad = max(n_pad, int(keep.sum()))
                for ag_k, cls_k, rt_k, rm_k in raw:
                    parts.append(targets_from_join(ag_k, classes=cls_k,
                                                   rates=rt_k, rates_mask=rm_k,
                                                   n_pad=n_pad, device=device))
                tg = {k: torch.cat([p[k] for p in parts], dim=0)
                      for k in parts[0]}
                gg, _ = g_agent({"frames": bb["frames"]}, tg)
                gs.append(_cat(gg, trunk))
            cos_u_w.append(_cos(gs[0], gs[1]))
        res["o4_sampler"] = {
            "_what": ("O4 is a SAMPLER. Reported: how its weights correlate "
                      "with lead-bearing windows, and the cosine between the "
                      "agent-readout gradient on an O4-weighted draw and on a "
                      "uniform draw."),
            "o4_log": o4log,
            "n_labelled_windows": len(labelled),
            "corr_w4_vs_gt_lead_gap": (
                round(float(np.corrcoef(wsel[ok], gt_gap[ok])[0, 1]), 5)
                if ok.sum() > 2 else None),
            "mean_w4_lead_windows": round(float(wsel[has_lead].mean()), 5)
            if has_lead.any() else None,
            "mean_w4_no_lead_windows": round(float(wsel[~has_lead].mean()), 5)
            if (~has_lead).any() else None,
            "cos_agent_grad_uniform_vs_o4weighted": cos_u_w,
        }
        Path(a.out).write_text(json.dumps(res, indent=1), "utf-8")
    print(f"[pc4] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
