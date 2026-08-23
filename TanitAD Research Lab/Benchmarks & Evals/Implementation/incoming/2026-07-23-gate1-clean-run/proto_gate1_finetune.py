#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate-1 closed-loop-aware planner fine-tune (DAgger / CAT-K-style, NO reward).

Fine-tunes REF-C-base's ANCHORED-DIFFUSION DECODER (the planner) on the ON-POLICY
junction states (extracted from the closed-loop rollout.asl by gate1_extract.py)
to output the GT expert RECOVERY path (`ref_lookahead_rig`, rig frame = the
decoder's own output space). The ResNet encoder + all aux heads are FROZEN; only
the decoder moves. This directly targets the measured failure: on-policy PLAN
degradation under closed-loop covariate shift (GATE1_ROLLOUTS_NOTE.md).

Objective == the REF-C trainer's trajectory primaries (scripts/refc_train.py),
target swapped for the recovery path:
    anchor-cls CE   classify the recovery target's nearest anchor
    traj-recon L1   reconstruct the recovery path from that anchor
(diffusion mode, steps=2 — matches how REF-C-base is trained AND deployed.)

SPEED: the frozen encoder/strategic/measurement/maneuver forward is cached ONCE
per on-policy window (refc_v12_cache pattern) so the decoder trains at ~0 encoder
cost — enabling a gentle, well-monitored fine-tune. Eval-mode cache (no ego-
dropout, real v0) == the deployment condition the driver feeds.

⚠️ WITHIN-SIM RELATIVE, ~3.2x OOD (NuRec recon). Prototype / mechanism de-risk,
NOT a clean Gate-1 number. n=15 -> overfit-watch (train + holdout recovery-L1).

Run: bash /workspace/run_ft.sh  (wrapper sets venv + PYTHONPATH + GPU)
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from tanitad.data.comma2k19 import stack_frames

import sys
sys.path.insert(0, "/root/TanitAD/stack/scripts")
from refc_v12_cache import load_frozen   # noqa: E402

NAVN = 4
WINDOW = 8


def build_windows(bundle):
    """canon_u8 [T,3,256,256] + steps -> list of (win_u8 [8,9,256,256], v0, nav, tgt)."""
    canon = bundle["canon_u8"]                      # uint8 [T,3,256,256]
    out = []
    for st in bundle["steps"]:
        k = st["k"]
        sl = canon[k - (WINDOW + 2):k]              # last 10 frames
        win = stack_frames(sl, 3)                   # [8,9,256,256] uint8
        out.append((win, float(st["v0"]), int(st["nav"]),
                    torch.tensor(st["tgt"], dtype=torch.float32)))
    return out


@torch.no_grad()
def frozen_forward(model, win_f, v0, nav, device):
    """Replicate RefCModel.forward's FROZEN part (eval, no dropout) -> the tensors
    the decoder consumes: fmap, m (measurement cond), ctx, man_logits."""
    b, w = win_f.shape[:2]
    fmap_all, pooled_all = model.encoder(win_f.reshape(b * w, *win_f.shape[2:]))
    pooled_seq = pooled_all.reshape(b, w, -1)
    pooled = pooled_seq[:, -1]
    fmap = fmap_all.reshape(b, w, *fmap_all.shape[1:])[:, -1]
    ctx = model.strategic(pooled_seq) if model.cfg.hierarchy else None
    nav_oh = F.one_hot(nav, NAVN).to(pooled.dtype)
    v = (v0.to(pooled.dtype) / 10.0).reshape(b, 1)          # eval: NO ego-dropout
    m = model.measurement(torch.cat([v, nav_oh], dim=-1))
    man_logits = model.maneuver_head(pooled)
    return fmap, m, ctx, man_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/gate1_ft_data")
    ap.add_argument("--ckpt", default="/root/models/refc-base-30k/ckpt.pt")
    ap.add_argument("--preset", default="base")
    ap.add_argument("--out", default="/root/models/refc-gate1-ft")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--holdout-scenes", default="",
                    help="comma scene8 list held OUT of training (generalization diag)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    model, cfg, base_step = load_frozen(args.ckpt, args.preset, None, device)
    model.eval()
    anchors = model.decoder.anchors.to(device).float()        # [N,S,2]
    holdout = set(s for s in args.holdout_scenes.split(",") if s)

    manifest = json.load(open(os.path.join(args.data, "manifest.json")))
    scene8s = [b["scene8"] for b in manifest["bundles"]]
    # ---- cache the frozen forward for every on-policy window ----
    FMAP, M, CTX, MAN, TGT, ASTAR, SCENE = [], [], [], [], [], [], []
    t0 = time.time()
    for s8 in scene8s:
        bundle = torch.load(os.path.join(args.data, f"{s8}.pt"), weights_only=False)
        wins = build_windows(bundle)
        for (win_u8, v0, nav, tgt) in wins:
            win_f = win_u8[None].to(device).float().div_(255.0)   # [1,8,9,256,256]
            fmap, m, ctx, man = frozen_forward(
                model, win_f, torch.tensor([v0], device=device),
                torch.tensor([nav], device=device, dtype=torch.long), device)
            tgt = tgt.to(device)
            d = ((tgt[None, None] - anchors[None]) ** 2).sum(dim=(-1, -2))  # [1,N]
            astar = d.argmin(dim=1)
            FMAP.append(fmap); M.append(m); CTX.append(ctx if ctx is not None else torch.zeros(1, 0, device=device))
            MAN.append(man); TGT.append(tgt[None]); ASTAR.append(astar); SCENE.append(s8)
    FMAP = torch.cat(FMAP); M = torch.cat(M)
    CTX = torch.cat(CTX) if CTX[0].numel() else None
    MAN = torch.cat(MAN); TGT = torch.cat(TGT); ASTAR = torch.cat(ASTAR)
    SCENE = np.array(SCENE)
    n = FMAP.shape[0]
    tr_mask = np.array([s not in holdout for s in SCENE])
    tr_idx = np.where(tr_mask)[0]
    ho_idx = np.where(~tr_mask)[0]
    print(f"[cache] windows={n} cached in {time.time()-t0:.1f}s | fmap={tuple(FMAP.shape)} "
          f"train={len(tr_idx)} holdout={len(ho_idx)} ({sorted(holdout)}) device={device}", flush=True)

    def eval_sel_l1(idx, train_mode=False):
        """Selected-traj L1 vs recovery target on window subset idx (deployment-
        relevant: the argmax-selected plan, steps=2, eval/deterministic)."""
        if len(idx) == 0:
            return None
        was = model.decoder.training
        model.decoder.train(train_mode)
        errs, cls_hit = [], []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                j = torch.as_tensor(idx[i:i + 256], device=device)
                ctx = CTX[j] if CTX is not None else None
                dec = model.decoder(FMAP[j], M[j], ctx=ctx,
                                    maneuver_logits=MAN[j], steps=2)
                errs.append((dec["traj"] - TGT[j]).abs().mean(dim=(-1, -2)))
                cls_hit.append((dec["anchor_logits"].argmax(1) == ASTAR[j]).float())
        model.decoder.train(was)
        return (float(torch.cat(errs).mean()), float(torch.cat(cls_hit).mean()))

    base_tr = eval_sel_l1(tr_idx)
    base_ho = eval_sel_l1(ho_idx) if len(ho_idx) else None
    print(f"[baseline] sel_L1 train={base_tr[0]:.4f} (cls_acc {base_tr[1]:.3f})"
          + (f" | holdout={base_ho[0]:.4f}" if base_ho else ""), flush=True)

    # ---- unfreeze the DECODER only (the planner) ----
    for p in model.parameters():
        p.requires_grad_(False)
    dec_params = list(model.decoder.parameters())
    for p in dec_params:
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in dec_params)
    print(f"[unfreeze] decoder-only: {n_train:,} trainable params", flush=True)
    opt = torch.optim.Adam(dec_params, lr=args.lr)

    log = []
    best = {"step": -1, "val": float("inf")}
    for step in range(args.steps + 1):
        if step % args.save_every == 0 or step == args.steps:
            tr = eval_sel_l1(tr_idx)
            ho = eval_sel_l1(ho_idx) if len(ho_idx) else None
            row = {"step": step, "sel_l1_train": round(tr[0], 4),
                   "cls_acc_train": round(tr[1], 3)}
            if ho:
                row["sel_l1_holdout"] = round(ho[0], 4)
                row["cls_acc_holdout"] = round(ho[1], 3)
            log.append(row)
            print(f"[step {step:4d}] {json.dumps(row)}", flush=True)
            # save intermediate
            torch.save({"model": model.state_dict(), "step": base_step,
                        "gate1_ft_step": step},
                       os.path.join(args.out, f"ckpt_step{step}.pt"))
            valm = ho[0] if ho else tr[0]
            if valm < best["val"]:
                best = {"step": step, "val": valm}
        if step == args.steps:
            break
        # ---- train step (decoder, diffusion mode) ----
        model.decoder.train(True)
        bi = torch.as_tensor(np.random.choice(tr_idx, size=min(args.batch, len(tr_idx)),
                                              replace=False), device=device)
        ctx = CTX[bi] if CTX is not None else None
        dec = model.decoder(FMAP[bi], M[bi], ctx=ctx, maneuver_logits=MAN[bi], steps=2)
        loss_cls = F.cross_entropy(dec["anchor_logits"], ASTAR[bi])
        recon = dec["anchor_traj"][torch.arange(len(bi), device=device), ASTAR[bi]]
        loss_traj = (recon - TGT[bi]).abs().mean()
        loss = loss_traj + loss_cls
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec_params, 1.0)
        opt.step()

    # ---- final ckpt.pt (the driver loads this via load_frozen + config.json) ----
    torch.save({"model": model.state_dict(), "step": base_step,
                "gate1_ft_final_step": args.steps}, os.path.join(args.out, "ckpt.pt"))
    # copy base config.json so load_frozen(preset) applies the trained shape
    import shutil
    base_cfg = os.path.join(os.path.dirname(args.ckpt), "config.json")
    if os.path.exists(base_cfg):
        shutil.copy(base_cfg, os.path.join(args.out, "config.json"))
    result = {"base_ckpt": args.ckpt, "base_step": base_step,
              "n_windows": int(n), "n_train": int(len(tr_idx)),
              "holdout_scenes": sorted(holdout), "n_holdout": int(len(ho_idx)),
              "lr": args.lr, "steps": args.steps, "batch": args.batch,
              "baseline_sel_l1_train": round(base_tr[0], 4),
              "baseline_sel_l1_holdout": round(base_ho[0], 4) if base_ho else None,
              "best_ckpt": best, "log": log,
              "trainable_params_decoder": int(n_train)}
    json.dump(result, open(os.path.join(args.out, "gate1_ft_result.json"), "w"), indent=2)
    print("\n[done] " + json.dumps({"out": args.out, "best": best,
          "final_train_sel_l1": log[-1]["sel_l1_train"],
          "final_holdout_sel_l1": log[-1].get("sel_l1_holdout")}), flush=True)


if __name__ == "__main__":
    main()
