"""WP-D P-GRAD (PREREG_WPD_PROBES.md, registered sha256 c054190b…, 2026-10-04T12:31:35Z) -- who trains the trunk, per TERM.

refcv7-r101-s0 @ 50,400, built by the run's own eval loader (`tanitad.eval.refcv7_loader.build_model`, STRICT), put in
TRAIN mode (`model.train()`, exactly what the trainer does; BN is frozen + folded as trained). 8 TRAIN batches of 8
windows, a seeded draw (numpy default_rng(0)) from the TRAIN-DIAG grid (the diagnostics package's rule: 139 train
clips sorted by sha12, 8 evenly spaced windows each = 1,112). ONE forward per batch through the trainer's own
`compute_losses_v3`, then one `torch.autograd.grad` per term (retain_graph) -- `.grad` is never touched.

Terms, each AT ITS LOSS WEIGHT (the weights are read off the built model, never typed):
  traj      TRAJ_WEIGHT x losses["traj"]
  box3d     model._w_box3d x losses["box3d"]
  agent     model._w_agent x agent_losses(...)["total"]   (captured by wrapping the trainer's own call; the agent loss
            is not returned as one scalar)
  map_hires model._w_map_hires x losses["map_hires"]
  tac_v6    model._w_tac_v6 x losses["tac_v6"]
  rest      losses["loss"] - (the five above)            (every other planner / tactical auxiliary term)
  total     losses["loss"]
Groups: the trunk (`core.encoder.`) split by `grad_conflict.default_group_of` (stem, stage_layer1..4, fuse), plus two
NON-trunk shared modules reported for information: the shared 0.25 m BEV (`_map_hires.lift` + `_map_hires.encoder`,
read by the map decoder, the planner pool and the box memory) and the planner BEV pool (`_perception.bev_pool`).

Controls (pre-registered): (L) linearity -- per batch and group, |sum of the six term gradients - total gradient| /
|total gradient| <= 1e-4; (Z) a weight-0 term (0.0 x box3d) reads EXACTLY 0 in every group.
Additional control (NOT pre-registered, disclosed): (K) the aux share as the run's own conflict detector defines it
(box3d + map_hires + tac_v6 vs traj, agent excluded) must land near the in-run value 0.992-0.995 (RESULT §1.4).
Measurement choices (disclosed; not in the registered text): the trunk runs in FP32 here (training used bf16 autocast
for the backbone) and cuDNN is set deterministic -- a 1e-4 linearity tolerance is not meaningful through bf16
activation gradients (~4e-3 rounding per element). Output: <out>/pgrad.json. Prints sha12 only.
Env (caller, before import): REFCV6_REPO, REFCV6_KIT, PYTHONPATH=<tree>/stack, REFCV6_REMAP_OVERRIDES=<train remap>.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

TERMS = ["traj", "box3d", "agent", "map_hires", "tac_v6", "rest"]
ALL = TERMS + ["total", "zero"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-dir", default="/home/nvidia/refcv7_run/runs/refcv7-r101-s0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fp32-trunk", type=int, default=1)
    a = ap.parse_args()

    import numpy as np
    import torch
    from tanitad.eval import refcv7_loader as L
    t0 = time.time()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tr = L.trainer()
    gcf = tr._gcf
    config = L.load_config(Path(a.run_dir) / "config.json")
    model, cfg, args, mrec = L.build_model(config, a.ckpt, device="cuda", strict=True)
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args, config)
    sha_of_eid = dict(e_ds._vis1_sha12)
    enc = model.core.encoder
    levers_before = dict(enc.memory_levers)
    if a.fp32_trunk:
        enc.memory_levers["bf16"] = False
    print(f"[pgrad] built step {mrec['state_dict']['step']} windows {len(e_ds)} levers bf16 "
          f"{levers_before.get('bf16')} -> {enc.memory_levers.get('bf16')} ({time.time() - t0:.0f}s)", flush=True)

    # ---- the window draw ------------------------------------------------------------------------- #
    by_ep = {}
    for i, (e_i, t) in enumerate(e_ds.index):
        by_ep.setdefault(int(e_i), []).append((int(t), i))
    grid = []
    for e_i in sorted(by_ep, key=lambda e: sha_of_eid[int(e_ds.episodes[e].episode_id)]):
        ts = sorted(by_ep[e_i])
        for j in range(8):
            grid.append(ts[int((j + 0.5) * len(ts) / 8)][1])
    rng = np.random.default_rng(a.seed)
    pick = [int(grid[k]) for k in rng.choice(len(grid), a.n_batches * a.batch, replace=False)]
    win = [(sha_of_eid[int(e_ds.episodes[e_ds.index[i][0]].episode_id)], int(e_ds.index[i][1])) for i in pick]
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(e_ds, pick), batch_size=a.batch, shuffle=False,
                                     num_workers=a.workers, drop_last=False)

    # ---- parameter groups -------------------------------------------------------------------------- #
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups, params = {}, []
    for n, p in named:
        if n.startswith("core.encoder."):
            g = gcf.default_group_of(n[len("core.encoder."):])
        elif n.startswith("_map_hires.lift.") or n.startswith("_map_hires.encoder."):
            g = "x_shared_bev_025m"
        elif n.startswith("_perception.bev_pool."):
            g = "x_planner_bev_pool"
        else:
            continue
        groups.setdefault(g, []).append(len(params))
        params.append(p)
    gsizes = {g: int(sum(params[i].numel() for i in ix)) for g, ix in groups.items()}
    print(f"[pgrad] groups { {g: gsizes[g] for g in sorted(gsizes)} }", flush=True)

    # ---- capture the agent loss total (the trainer's own call, untouched) -------------------------- #
    CAP = {"agent": []}
    orig_agent = tr._refc_agents.agent_losses

    def agent_wrapped(*args_, **kw):
        out = orig_agent(*args_, **kw)
        CAP["agent"].append(out["total"])
        return out
    tr._refc_agents.agent_losses = agent_wrapped
    w = {"traj": float(tr.TRAJ_WEIGHT), "box3d": float(getattr(model, "_w_box3d", 0.0)),
         "agent": float(getattr(model, "_w_agent", 0.0)), "map_hires": float(getattr(model, "_w_map_hires", 0.0)),
         "tac_v6": float(getattr(model, "_w_tac_v6", 0.0))}
    print(f"[pgrad] weights {w}", flush=True)

    model.train()
    per_batch = []
    sums = {t: {g: None for g in groups} for t in TERMS + ["total"]}
    ctl = {"L_max_rel_err": 0.0, "Z_max_abs": 0.0, "n_batches": 0}
    for bi, b in enumerate(dl):
        tb = time.time()
        CAP["agent"].clear()
        torch.manual_seed(1000 + bi)
        kw = ({"log_metrics": False} if "log_metrics" in inspect.signature(tr.compute_losses_v3).parameters
              else {})                         # the launch tree fec3a0d has no log_metrics (MEASURED, CPU smoke)
        losses = tr.compute_losses_v3(model, b, "cuda", mode=args.mode, ablate_frames=args.ablate_frames, **kw)
        zero_t = losses["loss"].sum() * 0.0
        T = {"traj": w["traj"] * losses["traj"],
             "box3d": w["box3d"] * losses["box3d"],
             "agent": (w["agent"] * sum(CAP["agent"])) if CAP["agent"] else zero_t,
             "map_hires": w["map_hires"] * losses["map_hires"],
             "tac_v6": w["tac_v6"] * losses["tac_v6"]}
        T["rest"] = losses["loss"] - sum(T[k] for k in ("traj", "box3d", "agent", "map_hires", "tac_v6"))
        T["total"] = losses["loss"]
        T["zero"] = 0.0 * T["box3d"]
        vals = {k: float(v.detach()) for k, v in T.items()}
        G = {}
        for k in ALL:
            if not T[k].requires_grad:
                gr = [None] * len(params)
            else:
                gr = torch.autograd.grad(T[k], params, retain_graph=True, allow_unused=True)
            G[k] = {g: torch.cat([(gr[i] if gr[i] is not None else torch.zeros_like(params[i])).reshape(-1).float()
                                  for i in ix]) for g, ix in groups.items()}
            del gr
        row = {"batch": bi, "windows": win[bi * a.batch:(bi + 1) * a.batch], "loss_values": vals,
               "n_agent_calls": len(CAP["agent"]), "norm": {}, "cos": {}, "L_rel_err": {}, "Z_max_abs": {}}
        for g in groups:
            nrm = {k: float(G[k][g].double().norm()) for k in ALL}
            row["norm"][g] = nrm
            row["cos"][g] = {}
            for i, k1 in enumerate(TERMS + ["total"]):
                for k2 in (TERMS + ["total"])[i + 1:]:
                    d = nrm[k1] * nrm[k2]
                    row["cos"][g][f"{k1}|{k2}"] = (float(torch.dot(G[k1][g].double(), G[k2][g].double()) / d)
                                                   if d > 0 else None)
            s = sum(G[k][g].double() for k in TERMS)
            tot = G["total"][g].double()
            rel = float((s - tot).norm() / tot.norm()) if float(tot.norm()) > 0 else 0.0
            row["L_rel_err"][g] = rel
            row["Z_max_abs"][g] = float(G["zero"][g].abs().max())
            ctl["L_max_rel_err"] = max(ctl["L_max_rel_err"], rel)
            ctl["Z_max_abs"] = max(ctl["Z_max_abs"], row["Z_max_abs"][g])
            for k in TERMS + ["total"]:
                sums[k][g] = G[k][g].clone() if sums[k][g] is None else sums[k][g] + G[k][g]
        del G
        ctl["n_batches"] += 1
        per_batch.append(row)
        print(f"[pgrad] ZZ{bi + 1}-{a.n_batches}ZZ {time.time() - tb:.0f}s L_rel_err {max(row['L_rel_err'].values()):.2e} "
              f"Z {max(row['Z_max_abs'].values()):.1e} cuda_max {torch.cuda.max_memory_allocated() / 1e9:.1f} GB",
              flush=True)
    tr._refc_agents.agent_losses = orig_agent

    # ---- summary over the 8 batches: the SUMMED gradient (a 64-window batch) ------------------------ #
    summ = {"norm": {}, "share_of_term_norm_sum": {}, "cos": {}, "aux_share_conflict_def": {}}
    for g in groups:
        nrm = {k: float(sums[k][g].double().norm()) for k in TERMS + ["total"]}
        summ["norm"][g] = nrm
        tsum = sum(nrm[k] for k in TERMS)
        summ["share_of_term_norm_sum"][g] = {k: (nrm[k] / tsum if tsum > 0 else None) for k in TERMS}
        summ["cos"][g] = {}
        for i, k1 in enumerate(TERMS + ["total"]):
            for k2 in (TERMS + ["total"])[i + 1:]:
                d = nrm[k1] * nrm[k2]
                summ["cos"][g][f"{k1}|{k2}"] = (float(torch.dot(sums[k1][g].double(), sums[k2][g].double()) / d)
                                                if d > 0 else None)
        aux = (sums["box3d"][g] + sums["map_hires"][g] + sums["tac_v6"][g]).double()
        na, nt = float(aux.norm()), nrm["traj"]
        summ["aux_share_conflict_def"][g] = na / (na + nt) if (na + nt) > 0 else None
        # projection of each term on the total direction: the share of the update each term "pays for"
        tot = sums["total"][g].double()
        tn2 = float(torch.dot(tot, tot))
        summ.setdefault("proj_share_on_total", {})[g] = (
            {k: float(torch.dot(sums[k][g].double(), tot) / tn2) for k in TERMS} if tn2 > 0 else None)
    trunk_groups = [g for g in groups if not g.startswith("x_")]
    whole = {}
    for k in TERMS + ["total"]:
        whole[k] = torch.cat([sums[k][g] for g in trunk_groups]).double()
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
    controls = {"L_linearity_max_rel_err": ctl["L_max_rel_err"], "L_pass": ctl["L_max_rel_err"] <= 1e-4,
                "Z_zero_weight_max_abs": ctl["Z_max_abs"], "Z_pass": ctl["Z_max_abs"] == 0.0,
                "K_aux_share_trunk": summ["trunk_whole"]["aux_share_conflict_def"],
                "K_expected_band_in_run": [0.992, 0.995], "K_is_preregistered": False}
    rec = {"probe": "P-GRAD", "prereg_sha256": "c054190bee4df5f266c99318d5374fbad9821a17ae7e44830c4b5f8feffc6f87",
           "tier": "OPEN-LOOP PERCEPTION DIAGNOSTIC (gradient census; no plan scored; four families N/A)",
           "ckpt": a.ckpt, "step": mrec["state_dict"]["step"], "strict": mrec["state_dict"]["strict"],
           "loader_departures": mrec.get("departures"),
           "measurement_choices": {"fp32_trunk": bool(a.fp32_trunk), "trunk_levers_as_trained": levers_before,
                                   "cudnn_deterministic": True, "train_mode": True, "seed_draw": a.seed,
                                   "forward_seed": "torch.manual_seed(1000 + batch)"},
           "weights": w, "group_sizes": gsizes, "n_batches": ctl["n_batches"], "batch": a.batch,
           "window_draw_sha256": hashlib.sha256(json.dumps(win).encode()).hexdigest(),
           "controls": controls, "summary": summ, "per_batch": per_batch,
           "cuda_max_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "wall_s": time.time() - t0}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pgrad.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"[pgrad] DONE L {controls['L_linearity_max_rel_err']:.2e} pass={controls['L_pass']} "
          f"Z {controls['Z_zero_weight_max_abs']} pass={controls['Z_pass']} "
          f"K {controls['K_aux_share_trunk']:.4f} wall {rec['wall_s']:.0f}s", flush=True)
    return 0 if (controls["L_pass"] and controls["Z_pass"]) else 3


if __name__ == "__main__":
    sys.exit(main())
