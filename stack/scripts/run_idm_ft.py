"""Light-FT ablation for the IDM re-gate (PRE_REGISTRATION §R2).

The baseline gate FAILED with a FROZEN encoder. This runs the pre-registered
light-FT lever: unfreeze the last `ft_blocks` ViT blocks + final norm + readout,
joint-train with the IDM head on the TRAINING split, then eval CROSS + in-domain.
A paired FROZEN arm on the IDENTICAL reduced train/eval sets is run in the same
process so the frozen→light-FT delta is apples-to-apples.

Efficiency: the frozen prefix (patch + pos + blocks[0:k_frozen]) never changes
during FT, so its activations are precomputed ONCE per train frame (no_grad,
fp16, in RAM) and the FT step runs only the trainable suffix — cheap.

One invocation = one experiment (`rig` or `comma`), both arms. Same gate as the
baseline: PASS iff cross speed R²>0.9 AND yaw R²>0.9 AND ADE@2s < 1.5× in-domain.

Usage (pod3, gpu_lock idm-regate):
  PYTHONPATH=/workspace/TanitAD/stack /workspace/venv/bin/python scripts/run_idm_ft.py \
    --experiment comma --ckpt /workspace/tmp/idm/ckpt.pt \
    --pai-cache /workspace/pai_epcache/physicalai-train-e438721ae894 \
    --pai-val-cache /workspace/pai_epcache/physicalai-val-f1b378f295ae \
    --comma-cache /workspace/data/comma2k19-val-61c46fca8f7f \
    --rig-table /workspace/tmp/idm/rig_table.json \
    --out /workspace/tmp/idm/regate_comma.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import idm_head as ih  # noqa: E402
import run_idm_proof as R  # noqa: E402


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# encoder with a trainable suffix                                             #
# --------------------------------------------------------------------------- #
def build_encoder(ckpt: str, device: str):
    enc, readout, meta = R.load_encoder(ckpt, device)      # frozen by default
    return enc, readout, meta


def set_trainable(enc, readout, ft_blocks: int):
    """Freeze everything, then unfreeze the last ``ft_blocks`` blocks + norm +
    readout. Returns k_frozen (index of first trainable block)."""
    for p in enc.parameters():
        p.requires_grad_(False)
    for p in readout.parameters():
        p.requires_grad_(False)
    depth = len(enc.blocks)
    k_frozen = depth - ft_blocks
    if ft_blocks > 0:
        for blk in enc.blocks[k_frozen:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for p in enc.norm.parameters():
            p.requires_grad_(True)
        for p in readout.parameters():
            p.requires_grad_(True)
        enc.train()
        readout.train()
    return k_frozen


@torch.no_grad()
def prefix_activations(enc, frames_u8: torch.Tensor, k_frozen: int, device: str,
                       batch: int = 64) -> torch.Tensor:
    """frames_u8 [T,9,256,256] -> prefix act [T, N, D] fp16 (blocks[0:k_frozen])."""
    outs = []
    for i in range(0, frames_u8.shape[0], batch):
        fb = frames_u8[i:i + batch].to(device).float().div_(255.0)
        t = enc.patch(fb).flatten(2).transpose(1, 2) + enc.pos
        for blk in enc.blocks[:k_frozen]:
            t = blk(t)
        outs.append(t.half().cpu())
    return torch.cat(outs)


def suffix_forward(enc, readout, t: torch.Tensor, k_frozen: int) -> torch.Tensor:
    """prefix act [M, N, D] -> z [M, state_dim], WITH grad on blocks[k_frozen:]."""
    for blk in enc.blocks[k_frozen:]:
        t = blk(t)
    return readout(enc.norm(t))


@torch.no_grad()
def full_encode(enc, readout, frames_u8: torch.Tensor, device: str,
                batch: int = 64) -> torch.Tensor:
    """Full (FT'd) encoder+readout, frozen forward: frames -> z [T, state_dim] fp16."""
    outs = []
    for i in range(0, frames_u8.shape[0], batch):
        fb = frames_u8[i:i + batch].to(device).float().div_(255.0)
        t = enc.patch(fb).flatten(2).transpose(1, 2) + enc.pos
        for blk in enc.blocks:
            t = blk(t)
        outs.append(readout(enc.norm(t)).half().cpu())
    return torch.cat(outs)


# --------------------------------------------------------------------------- #
# data: window index over episodes held in RAM                                #
# --------------------------------------------------------------------------- #
def load_eps(paths: list[str]) -> list[dict]:
    eps = []
    for p in paths:
        d = R._load_ep(p)
        eps.append({"frames": d["frames_u8"], "poses": d["poses"].float(),
                    "actions": d["actions"].float()})
    return eps


def window_index(eps: list[dict], k: int, stride: int) -> list[tuple[int, int]]:
    idx = []
    for e_i, ep in enumerate(eps):
        for t in ih.valid_centers(ep["frames"].shape[0], k, ih.DEFAULT_HORIZONS,
                                  stride).tolist():
            idx.append((e_i, t))
    return idx


def fit_targets(eps: list[dict], index: list[tuple[int, int]]) -> tuple:
    """Precompute scalar+traj targets for every window (cheap; poses/actions)."""
    S, T = [], []
    for e_i, t in index:
        ep = eps[e_i]
        tt = torch.tensor([t])
        S.append(ih.scalar_targets_at(ep["poses"], ep["actions"], tt))
        T.append(ih.traj_targets_at(ep["poses"], tt))
    return torch.cat(S), torch.cat(T)


# --------------------------------------------------------------------------- #
# arms                                                                        #
# --------------------------------------------------------------------------- #
def eval_ft(enc, readout, head, eval_sets: dict, device: str) -> dict:
    """Encode each eval set with the (FT'd) encoder -> latents -> windows ->
    metrics. Streams per-episode so eval frames are never all resident."""
    out = {}
    for name, paths in eval_sets.items():
        Z, S, T = [], [], []
        for p in paths:
            d = R._load_ep(p)
            z = full_encode(enc, readout, d["frames_u8"], device).float()
            zw, sc, tj = ih.build_windows(z, d["poses"].float(),
                                          d["actions"].float(), k=4, stride=2)
            if zw.shape[0]:
                Z.append(zw)
                S.append(sc)
                T.append(tj)
        Z, S, T = torch.cat(Z), torch.cat(S), torch.cat(T)
        out[name] = ih.evaluate(head, Z, S, T, device=device)
    return out


def run_experiment(enc, readout, train_paths: list[str], eval_sets: dict,
                   state_dim: int, device: str, *, ft_blocks: int, ft_steps: int,
                   batch: int, seed: int, enc_lr: float = 5e-5) -> dict:
    """One arm: (re)load fresh weights via the passed enc/readout state, train,
    eval. Returns metrics. NOTE: caller resets enc/readout weights between arms."""
    torch.manual_seed(seed)
    k_frozen = set_trainable(enc, readout, ft_blocks)
    eps = load_eps(train_paths)
    index = window_index(eps, k=4, stride=2)
    Str_all, Ttr_all = fit_targets(eps, index)
    std = ih.Standardizer.fit(Str_all)
    n = len(index)
    log(f"  arm ft_blocks={ft_blocks}: {len(eps)} train eps, {n} windows, "
        f"k_frozen={k_frozen}")

    # precompute frozen-prefix activations for all train frames (once)
    prefix = [prefix_activations(enc, ep["frames"], k_frozen, device)
              for ep in eps]   # list of [T, N, D] fp16

    head = ih.IDMHead(state_dim=state_dim, horizons=ih.DEFAULT_HORIZONS).to(device)
    groups = [{"params": head.parameters(), "lr": 3e-4}]
    if ft_blocks > 0:
        enc_params = [p for blk in enc.blocks[k_frozen:] for p in blk.parameters()]
        enc_params += list(enc.norm.parameters()) + list(readout.parameters())
        groups.append({"params": enc_params, "lr": enc_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    warmup = max(10, ft_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warmup) * (0.5 * (1 + torch.cos(
            torch.tensor(min(s, ft_steps) / ft_steps * 3.14159)).item())
            if s >= warmup else 1.0))
    head.train()
    g = torch.Generator().manual_seed(seed)
    K = 4
    for step in range(ft_steps):
        sel = torch.randint(n, (batch,), generator=g).tolist()
        hs, scs, tjs = [], [], []
        for si in sel:
            e_i, t = index[si]
            hs.append(prefix[e_i][t - K:t + K + 1])          # [9, N, D] fp16
            tt = torch.tensor([t])
            scs.append(ih.scalar_targets_at(eps[e_i]["poses"],
                                            eps[e_i]["actions"], tt))
            tjs.append(ih.traj_targets_at(eps[e_i]["poses"], tt))
        hb = torch.stack(hs).to(device).float()              # [B,9,N,D]
        B, W = hb.shape[:2]
        if ft_blocks > 0:
            z = suffix_forward(enc, readout, hb.reshape(B * W, *hb.shape[2:]),
                               k_frozen).reshape(B, W, -1)
        else:
            with torch.no_grad():
                z = suffix_forward(enc, readout,
                                   hb.reshape(B * W, *hb.shape[2:]),
                                   k_frozen).reshape(B, W, -1)
        out = head(z)
        sc = torch.cat(scs).to(device)
        tj = torch.cat(tjs).to(device)
        ld = ih.idm_loss(out, sc, tj, std)
        opt.zero_grad(set_to_none=True)
        ld["loss"].backward()
        opt.step()
        sched.step()
        if step % max(1, ft_steps // 8) == 0 or step == ft_steps - 1:
            log(f"    step {step}/{ft_steps} loss {float(ld['loss'].detach()):.4f}")
    del prefix
    enc.eval()
    readout.eval()
    val = eval_ft(enc, readout, head, eval_sets, device)
    return {"ft_blocks": ft_blocks, "train_windows": n, "train_eps": len(eps),
            "head_params": ih.count_params(head), "val": val}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment",
                    choices=["rig", "comma", "multirig", "multirig_symm"],
                    required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pai-cache", required=True)
    ap.add_argument("--pai-val-cache", required=True)
    ap.add_argument("--comma-cache", required=True)
    ap.add_argument("--rig-table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ft-blocks", type=int, default=4)
    ap.add_argument("--ft-steps", type=int, default=1000)
    ap.add_argument("--enc-lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=70)
    ap.add_argument("--n-eval-cross", type=int, default=120)
    ap.add_argument("--n-eval-indomain", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--git-hash", default="unknown")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rig_table = json.loads(Path(args.rig_table).read_text())
    enc, readout, meta = build_encoder(args.ckpt, device)
    state_dim = meta["state_dim"]
    # snapshot pristine weights so each arm starts from the same encoder
    enc_sd0 = {k: v.detach().clone() for k, v in enc.state_dict().items()}
    ro_sd0 = {k: v.detach().clone() for k, v in readout.state_dict().items()}

    a_eps, b_eps = R.select_episodes(rig_table, args.pai_cache, 400, 400)
    a_paths = [p for _t, p in a_eps]
    b_paths = [p for _t, p in b_eps]
    val_paths = [str(p) for p in
                 sorted(Path(args.pai_val_cache).glob("ep_*.pt"))]
    comma_paths = [str(p) for p in
                   sorted(Path(args.comma_cache).glob("ep_*.pt"))]

    if args.experiment == "rig":
        n_tr = min(args.n_train, len(a_paths))
        train_paths = a_paths[:n_tr]
        held_a = a_paths[n_tr:n_tr + args.n_eval_indomain]
        eval_sets = {"in_rig_heldout_rigA": held_a,
                     "cross_rig_rigB": b_paths[:args.n_eval_cross]}
        indomain_key = "in_rig_heldout_rigA"
        cross_key = "cross_rig_rigB"
    elif args.experiment == "comma":
        pai_pool = (a_paths[:args.n_train // 2] + b_paths[:args.n_train // 2])
        train_paths = pai_pool
        eval_sets = {"in_corpus_heldout_paival": val_paths[:args.n_eval_indomain],
                     "cross_domain_comma": comma_paths[:args.n_eval_cross]}
        indomain_key = "in_corpus_heldout_paival"
        cross_key = "cross_domain_comma"
    elif args.experiment == "multirig":
        # PRIMARY: co-train {rig-A + comma2k19}, HELD-OUT rig-B (never in train).
        # ~window-balanced (comma clips are 300-frame vs PhysicalAI 199), so 60
        # rig-A + 40 comma ~ matched window counts; rig-A stays well-represented
        # for the direct comparison to the single-domain rig-A->rig-B re-gate.
        train_paths = a_paths[:60] + comma_paths[:40]
        eval_sets = {"in_rigA_heldout": a_paths[60:100],
                     "in_comma_heldout": comma_paths[40:80],
                     "cross_heldout_rigB": b_paths[:args.n_eval_cross]}
        indomain_key = "in_rigA_heldout"       # same-corpus reference for rig-B
        cross_key = "cross_heldout_rigB"
    else:  # multirig_symm
        # SYMMETRIC: co-train {rig-A + rig-B}, HELD-OUT comma2k19 (fully unseen).
        train_paths = a_paths[:45] + b_paths[:45]
        eval_sets = {"in_rigA_heldout": a_paths[45:85],
                     "in_rigB_heldout": b_paths[45:85],
                     "cross_heldout_comma": comma_paths[:90]}
        indomain_key = "in_rigA_heldout"
        cross_key = "cross_heldout_comma"

    log(f"experiment {args.experiment}: {len(train_paths)} train, "
        f"eval {{{', '.join(f'{k}:{len(v)}' for k,v in eval_sets.items())}}}")

    results = {"meta": {"experiment": f"idm_regate_{args.experiment}",
                        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "device": device, "git_hash": args.git_hash,
                        "ckpt_md5": R.md5_of(args.ckpt), "ckpt_step": meta["ckpt_step"],
                        "state_dim": state_dim, "ft_blocks": args.ft_blocks,
                        "ft_steps": args.ft_steps, "enc_lr": args.enc_lr,
                        "head_lr": 3e-4, "n_train": len(train_paths),
                        "pass_rule": "cross speed R2>0.9 AND yaw R2>0.9 AND "
                                     "ADE@2s < 1.5x in-domain heldout ADE@2s"},
               "arms": {}}

    for arm, ftb in (("frozen", 0), ("light_ft", args.ft_blocks)):
        log(f"=== arm {arm} (ft_blocks={ftb}) ===")
        enc.load_state_dict(enc_sd0)                    # reset to pristine
        readout.load_state_dict(ro_sd0)
        res = run_experiment(enc, readout, train_paths, eval_sets, state_dim,
                             device, ft_blocks=ftb, ft_steps=args.ft_steps,
                             batch=args.batch, seed=args.seed, enc_lr=args.enc_lr)
        cross = res["val"][cross_key]
        indom = res["val"][indomain_key]
        res["verdict"] = {
            "cross_speed_r2": cross["r2"]["speed"],
            "cross_yaw_r2": cross["r2"]["yaw_rate"],
            "cross_ade_2s": cross["ade_2s"],
            "in_domain_ade_2s": indom["ade_2s"],
            "ade_ratio": cross["ade_2s"] / max(indom["ade_2s"], 1e-9),
            "PASS": bool(cross["r2"]["speed"] > 0.9 and cross["r2"]["yaw_rate"] > 0.9
                         and cross["ade_2s"] < 1.5 * indom["ade_2s"])}
        results["arms"][arm] = res
        log(f"  {arm} VERDICT {json.dumps(res['verdict'])}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    log(f"WROTE {args.out}")
    log("IDM_REGATE_DONE")


if __name__ == "__main__":
    main()
