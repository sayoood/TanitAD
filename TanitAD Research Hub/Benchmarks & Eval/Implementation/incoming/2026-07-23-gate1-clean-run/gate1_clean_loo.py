#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate-1 clean-run P0/P1 measurement: leave-group-out on the prototype's 15
junction scenes, measuring (i) MEMORIZATION (held-out recovery-L1 vs train) and
(ii) the HIGH-DEVIATION side-effect (plan-shift-from-base on held-out scenes),
and (P1) whether CAT-K target-filtering + a base-plan trust-region regularizer
tame the side-effect without giving back the recovery gain.

Reuses the prototype's exact frozen forward + decoder objective (gate1_finetune.py).
Caches the frozen forward ONCE for all 15 scenes; each fold re-inits the decoder
to base and trains decoder-only (the prototype's recipe). ~0 extra data.

Within-sim / ~3.2x NuRec OOD (prototype data) -- this measures the LEVER's
generalization + deviation behaviour at n=15, which transfers to the real-footage
scale (same decoder, same objective, ~= same #distinct scenes). It is NOT a clean
absolute Gate-1 number (see P0 report: the low-OOD source cannot emit offroad).
"""
from __future__ import annotations
import argparse, copy, json, os, time, importlib.util as U
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/root/TanitAD/stack/scripts")
sys.path.insert(0, "/root/TanitAD/stack")
from refc_v12_cache import load_frozen  # noqa

_spec = U.spec_from_file_location("g1ft", "/workspace/gate1_finetune.py")
g1 = U.module_from_spec(_spec); _spec.loader.exec_module(g1)
build_windows, frozen_forward = g1.build_windows, g1.frozen_forward


def cache_all(model, data, anchors, device):
    manifest = json.load(open(os.path.join(data, "manifest.json")))
    scene8s = [b["scene8"] for b in manifest["bundles"]]
    FMAP, M, CTX, MAN, TGT, ASTAR, SCENE = [], [], [], [], [], [], []
    for s8 in scene8s:
        bundle = torch.load(os.path.join(data, f"{s8}.pt"), weights_only=False)
        for (win_u8, v0, nav, tgt) in build_windows(bundle):
            win_f = win_u8[None].to(device).float().div_(255.0)
            fmap, m, ctx, man = frozen_forward(
                model, win_f, torch.tensor([v0], device=device),
                torch.tensor([nav], device=device, dtype=torch.long), device)
            tgt = tgt.to(device)
            d = ((tgt[None, None] - anchors[None]) ** 2).sum(dim=(-1, -2))
            FMAP.append(fmap); M.append(m)
            CTX.append(ctx if ctx is not None else torch.zeros(1, 0, device=device))
            MAN.append(man); TGT.append(tgt[None]); ASTAR.append(d.argmin(dim=1))
            SCENE.append(s8)
    C = dict(FMAP=torch.cat(FMAP), M=torch.cat(M), MAN=torch.cat(MAN),
             TGT=torch.cat(TGT), ASTAR=torch.cat(ASTAR), SCENE=np.array(SCENE))
    C["CTX"] = torch.cat(CTX) if CTX[0].numel() else None
    return C, scene8s


def dec_out(model, C, idx, device, train_mode=False):
    was = model.decoder.training
    model.decoder.train(train_mode)
    trajs, hits, errs = [], [], []
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            j = torch.as_tensor(idx[i:i + 256], device=device)
            ctx = C["CTX"][j] if C["CTX"] is not None else None
            dec = model.decoder(C["FMAP"][j], C["M"][j], ctx=ctx,
                                maneuver_logits=C["MAN"][j], steps=2)
            trajs.append(dec["traj"])
            errs.append((dec["traj"] - C["TGT"][j]).abs().mean(dim=(-1, -2)))
            hits.append((dec["anchor_logits"].argmax(1) == C["ASTAR"][j]).float())
    model.decoder.train(was)
    return torch.cat(trajs), float(torch.cat(errs).mean()), float(torch.cat(hits).mean())


def catk_keep_mask(C, envelope_lat=3.0, envelope_yaw_deg=12.0):
    """CAT-K / recovery-feasibility filter: DROP catastrophic-state labels whose
    recovery target leaves the P1-measured low-OOD envelope or points backward.
    tgt is [.,4,2] rig frame (fwd,left) = the expert recovery path 0.5..2s ahead.
    Catastrophic if: (a) the first recovery waypoint requires |left|>envelope_lat
    (beyond the validated lateral envelope) OR the implied heading correction
    exceeds envelope_yaw; or (b) the recovery points BACKWARD (fwd<=0) -- an
    infeasible reverse target. Keeps only near-manifold, feasible recovery labels."""
    tgt = C["TGT"]                                   # [N,4,2]
    fwd0 = tgt[:, 0, 0]; left0 = tgt[:, 0, 1]
    fwd_end = tgt[:, -1, 0]
    # heading correction proxy: atan2(dleft, dfwd) over the 2s path
    dleft = tgt[:, -1, 1] - tgt[:, 0, 1]
    dfwd = (tgt[:, -1, 0] - tgt[:, 0, 0]).clamp_min(1e-3)
    yaw_corr = torch.atan2(dleft.abs(), dfwd) * 180 / np.pi
    backward = fwd_end <= 0.0                         # target ends behind the ego
    too_lat = tgt[:, :, 1].abs().amax(1) > envelope_lat
    too_yaw = yaw_corr > envelope_yaw_deg
    drop = backward | too_lat | too_yaw
    return (~drop).cpu().numpy(), {
        "n": int(tgt.shape[0]), "dropped": int(drop.sum()),
        "drop_backward": int(backward.sum()), "drop_too_lat": int(too_lat.sum()),
        "drop_too_yaw": int(too_yaw.sum())}


def train_fold(model, C, base_dec, tr_idx, ho_idx, device, steps, batch, lr,
               lambda_dev=0.0, base_traj=None, seed=0):
    """Reset decoder to base, train decoder-only on tr_idx. Optional base-plan
    trust-region regularizer: lambda_dev * ||FT_traj - base_traj||_1 (keeps the
    recovery gentle -> tames the high-deviation side-effect)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model.decoder.load_state_dict(copy.deepcopy(base_dec))
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.decoder.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(model.decoder.parameters(), lr=lr)
    log = []
    for step in range(steps + 1):
        if step % 200 == 0 or step == steps:
            _, tr_l1, tr_acc = dec_out(model, C, tr_idx, device)
            row = {"step": step, "train_l1": round(tr_l1, 4), "train_acc": round(tr_acc, 3)}
            if len(ho_idx):
                ho_traj, ho_l1, ho_acc = dec_out(model, C, ho_idx, device)
                row["ho_l1"] = round(ho_l1, 4); row["ho_acc"] = round(ho_acc, 3)
                if base_traj is not None:
                    shift = (ho_traj - base_traj[torch.as_tensor(ho_idx, device=device)]
                             ).norm(dim=-1).mean()
                    row["ho_plan_shift_m"] = round(float(shift), 4)
            log.append(row)
        if step == steps:
            break
        model.decoder.train(True)
        bi = torch.as_tensor(np.random.choice(tr_idx, size=min(batch, len(tr_idx)),
                                              replace=False), device=device)
        ctx = C["CTX"][bi] if C["CTX"] is not None else None
        dec = model.decoder(C["FMAP"][bi], C["M"][bi], ctx=ctx,
                            maneuver_logits=C["MAN"][bi], steps=2)
        loss_cls = F.cross_entropy(dec["anchor_logits"], C["ASTAR"][bi])
        recon = dec["anchor_traj"][torch.arange(len(bi), device=device), C["ASTAR"][bi]]
        loss = (recon - C["TGT"][bi]).abs().mean() + loss_cls
        if lambda_dev > 0 and base_traj is not None:
            loss = loss + lambda_dev * (dec["traj"] - base_traj[bi]).abs().mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0); opt.step()
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/gate1_ft_data")
    ap.add_argument("--ckpt", default="/root/models/refc-base-30k/ckpt.pt")
    ap.add_argument("--preset", default="base")
    ap.add_argument("--out", default="/workspace/gate1_clean_loo.json")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--holdout-size", type=int, default=3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, cfg, base_step = load_frozen(args.ckpt, args.preset, None, device)
    model.eval()
    anchors = model.decoder.anchors.to(device).float()
    base_dec = copy.deepcopy(model.decoder.state_dict())

    t0 = time.time()
    C, scene8s = cache_all(model, args.data, anchors, device)
    n = C["FMAP"].shape[0]
    base_traj, base_tr_l1, base_tr_acc = dec_out(model, C, np.arange(n), device)
    print(f"[cache] {n} windows / {len(scene8s)} scenes in {time.time()-t0:.1f}s | "
          f"BASE in-sample recovery-L1={base_tr_l1:.4f} acc={base_tr_acc:.3f}", flush=True)

    keep, catk_stats = catk_keep_mask(C)
    print(f"[catk] {catk_stats}", flush=True)

    # 5 leave-3-out folds covering all 15 scenes as holdout exactly once
    order = list(scene8s)
    folds = [order[i:i + args.holdout_size] for i in range(0, len(order), args.holdout_size)]
    sc = C["SCENE"]

    def run_variant(name, lambda_dev, use_catk):
        fold_rows = []
        for fi, ho_scenes in enumerate(folds):
            ho_mask = np.isin(sc, ho_scenes)
            tr_mask = ~ho_mask
            if use_catk:
                tr_mask = tr_mask & keep
            tr_idx = np.where(tr_mask)[0]; ho_idx = np.where(ho_mask)[0]
            # per-fold BASE holdout metrics -- reset to PRISTINE base first
            # (else it reads the previous fold's trained decoder: contamination)
            model.decoder.load_state_dict(copy.deepcopy(base_dec))
            _, b_ho_l1, b_ho_acc = dec_out(model, C, ho_idx, device)
            log = train_fold(model, C, base_dec, tr_idx, ho_idx, device,
                             args.steps, args.batch, args.lr,
                             lambda_dev=lambda_dev, base_traj=base_traj, seed=fi)
            f = log[-1]
            fold_rows.append({"fold": fi, "ho_scenes": ho_scenes,
                              "n_train": int(len(tr_idx)), "n_ho": int(len(ho_idx)),
                              "base_ho_l1": round(b_ho_l1, 4),
                              "final_ho_l1": f.get("ho_l1"),
                              "final_train_l1": f["train_l1"],
                              "final_ho_plan_shift_m": f.get("ho_plan_shift_m"),
                              "traj": log})
            print(f"[{name} fold {fi}] ho={ho_scenes} base_ho_l1={b_ho_l1:.3f} "
                  f"-> final_ho_l1={f.get('ho_l1'):.3f} train_l1={f['train_l1']:.3f} "
                  f"ho_plan_shift={f.get('ho_plan_shift_m')}", flush=True)
        # aggregate
        bl = np.array([r["base_ho_l1"] for r in fold_rows])
        fl = np.array([r["final_ho_l1"] for r in fold_rows])
        tl = np.array([r["final_train_l1"] for r in fold_rows])
        ps = np.array([r["final_ho_plan_shift_m"] for r in fold_rows])
        agg = {"variant": name, "lambda_dev": lambda_dev, "use_catk": use_catk,
               "mean_base_ho_l1": round(float(bl.mean()), 4),
               "mean_final_ho_l1": round(float(fl.mean()), 4),
               "mean_ho_l1_improve": round(float((bl - fl).mean()), 4),
               "mean_ho_l1_improve_frac": round(float(((bl - fl) / bl).mean()), 4),
               "mean_final_train_l1": round(float(tl.mean()), 4),
               "mean_ho_plan_shift_m": round(float(ps.mean()), 4),
               "folds": fold_rows}
        print(f"[{name}] AGG base_ho_l1={agg['mean_base_ho_l1']} "
              f"final_ho_l1={agg['mean_final_ho_l1']} "
              f"(train_l1={agg['mean_final_train_l1']}) "
              f"ho_improve={agg['mean_ho_l1_improve']} "
              f"({agg['mean_ho_l1_improve_frac']*100:.0f}%) "
              f"ho_plan_shift={agg['mean_ho_plan_shift_m']}m", flush=True)
        return agg

    results = {
        "meta": {"n_windows": int(n), "n_scenes": len(scene8s), "steps": args.steps,
                 "batch": args.batch, "lr": args.lr, "base_step": base_step,
                 "base_in_sample_l1": round(base_tr_l1, 4),
                 "framing": "within-sim ~3.2x NuRec OOD; LEVER generalization at n=15, "
                            "transfers to real-footage scale; NOT a clean absolute number",
                 "catk_stats": catk_stats},
        "A_naive": run_variant("A_naive", 0.0, False),
        "B_catk": run_variant("B_catk", 0.0, True),
        "C_catk_dev": run_variant("C_catk_dev", 1.0, True),
    }
    json.dump(results, open(args.out, "w"), indent=2, default=str)
    print("\n=== SUMMARY (mean over 5 leave-3-out folds) ===")
    for k in ("A_naive", "B_catk", "C_catk_dev"):
        a = results[k]
        print(f"{k:12s} base_ho_l1={a['mean_base_ho_l1']:.3f} final_ho_l1={a['mean_final_ho_l1']:.3f} "
              f"train_l1={a['mean_final_train_l1']:.3f} ho_improve={a['mean_ho_l1_improve_frac']*100:5.1f}% "
              f"ho_plan_shift={a['mean_ho_plan_shift_m']:.3f}m")
    print("wrote", args.out)
    print("LOO_DONE")


if __name__ == "__main__":
    main()
