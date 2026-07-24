"""Camera-conditioning ABLATION (pre-registered go/no-go for our own dynamics
encoder) — `…/incoming/2026-07-22-own-dynamics-encoder/PRE_REGISTRATION.md`.

Question: does GAIA-2-style explicit camera conditioning (arXiv:2503.20523) recover
cross-rig ego-motion recovery where multi-domain co-training did NOT
(`results_multirig.json`: held-out rig-B light-FT speed R² -1.61)?

Design: identical to the re-gate light-FT arm (`run_idm_ft.py`), warm-started from
flagship-v1, EXCEPT the trainable suffix blocks get per-block camera conditioning
(intrinsics/extrinsics/distortion embeds summed, zero-init injection). Two arms,
capacity-matched:
  OFF : the conditioning modules exist but are fed a CONSTANT all-unknown vector
        (no geometry info) — reproduces the re-gate light-FT baseline.
  ON  : fed the true per-clip camera params (cy from the rig table + fisheye flag).
Gate (unchanged): cross speed R²>0.9 AND yaw R²>0.9 AND ADE@2s < 1.5x in-domain.

Memory-safe (pod cgroup ~46 GB, clips 117 MB each): the FROZEN prefix
(blocks[0:k_frozen], cam-independent, invariant across arms) is banked ONCE with
frames freed; eval streams one clip at a time. Run on pod3 (A40) under gpu_lock.
Self-contained: imports run_idm_proof (R), run_idm_ft (FT), idm_head (ih).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import idm_head as ih          # noqa: E402
import run_idm_proof as R      # noqa: E402
import run_idm_ft as FT        # noqa: E402


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# GAIA-2 camera conditioning (intrinsics/extrinsics/distortion embeds summed)  #
# --------------------------------------------------------------------------- #
CAM_GROUPS = (("intrinsics", 3), ("extrinsics", 3), ("distortion", 2))
N_CAM = 8
_SCALE = torch.tensor([1/256, 1/256, 1/256, 1.0, 1/2.0, 1.0, 1.0, 1.0])
_DEFAULT = torch.tensor([266.0, 128.0, 128.0, 0.0, 1.4, 0.0, 0.0, 0.0])


def normalize_cam(raw: Tensor, known: Tensor) -> Tensor:
    raw = torch.where(known > 0.5, raw.float(), _DEFAULT)
    return torch.cat([raw * _SCALE, known.float()], dim=-1)      # [16]


def clip_cam(cy: float, is_fisheye: float) -> Tensor:
    """ON: true per-clip geometry — principal point cy (the rig signal: rigA~542 /
    rigB~753) + the fisheye flag (PhysicalAI f-theta vs comma rectilinear)."""
    raw = torch.tensor([266.0, 128.0, float(cy), 0.0, 1.4, 0.0, 0.0, float(is_fisheye)])
    return normalize_cam(raw, torch.ones(N_CAM))


OFF_CAM = normalize_cam(_DEFAULT.clone(), torch.zeros(N_CAM))    # constant, no info


class CamEncoding(nn.Module):
    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.embeds = nn.ModuleList()
        self.slices = []
        off = 0
        for _n, g in CAM_GROUPS:
            self.slices.append((off, off + g))
            off += g
            self.embeds.append(nn.Sequential(
                nn.Linear(2 * g, hidden), nn.GELU(), nn.Linear(hidden, d_model)))

    def forward(self, cam16: Tensor) -> Tensor:
        sc, kn = cam16[..., :N_CAM], cam16[..., N_CAM:]
        enc = None
        for (lo, hi), emb in zip(self.slices, self.embeds):
            e = emb(torch.cat([sc[..., lo:hi], kn[..., lo:hi]], dim=-1))
            enc = e if enc is None else enc + e
        return enc


def make_inject(d_model: int, n: int) -> nn.ModuleList:
    inj = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(n))
    for lin in inj:
        nn.init.zeros_(lin.weight)          # zero-init => identity at step 0
        nn.init.zeros_(lin.bias)
    return inj


def suffix_cond(enc, readout, cam_enc, inject, prefix_t: Tensor, cam16: Tensor,
                k_frozen: int) -> Tensor:
    """prefix act [M,N,D] + per-row cam [M,16] -> z [M,S], grad on suffix blocks +
    cond modules + readout."""
    cam = cam_enc(cam16)                                 # [M,D]
    t = prefix_t
    for j, blk in enumerate(enc.blocks[k_frozen:]):
        t = t + inject[j](cam).unsqueeze(1)              # GAIA-2 per-suffix-block
        t = blk(t)
    return readout(enc.norm(t))


@torch.no_grad()
def full_encode_cond(enc, readout, cam_enc, inject, frames_u8: Tensor,
                     cam16_row: Tensor, k_frozen: int, device: str,
                     batch: int = 64) -> Tensor:
    """Eval: encode a whole clip through ALL blocks, conditioning the suffix with
    the clip's single cam vector. -> z [T,S] fp16."""
    cam_e = cam_enc(cam16_row.to(device).unsqueeze(0))   # [1,D]
    outs = []
    for i in range(0, frames_u8.shape[0], batch):
        fb = frames_u8[i:i + batch].to(device).float().div_(255.0)
        t = enc.patch(fb).flatten(2).transpose(1, 2) + enc.pos
        for blk in enc.blocks[:k_frozen]:
            t = blk(t)
        for j, blk in enumerate(enc.blocks[k_frozen:]):
            t = t + inject[j](cam_e).unsqueeze(1)
            t = blk(t)
        outs.append(readout(enc.norm(t)).half().cpu())
    return torch.cat(outs)


# --------------------------------------------------------------------------- #
# memory-safe data: frozen-prefix BANK (frames freed) + streamed eval          #
# --------------------------------------------------------------------------- #
def build_prefix_bank(enc, specs: list[tuple[str, Tensor]], k_frozen: int,
                      device: str) -> list[dict]:
    """One entry per clip: {prefix [T,N,D] fp16 (cam-independent), poses, actions,
    cam}. Frames are loaded, prefixed, and FREED (never all resident)."""
    bank = []
    for j, (p, cam) in enumerate(specs):
        d = R._load_ep(p)
        pref = FT.prefix_activations(enc, d["frames_u8"], k_frozen, device)
        bank.append({"prefix": pref, "poses": d["poses"].float(),
                     "actions": d["actions"].float(), "cam": cam})
        del d
        if j % 25 == 0:
            log(f"    prefix {j}/{len(specs)} banked")
    return bank


def window_index(bank: list[dict], k: int, stride: int) -> list[tuple[int, int]]:
    idx = []
    for ci, c in enumerate(bank):
        for t in ih.valid_centers(c["prefix"].shape[0], k, ih.DEFAULT_HORIZONS,
                                  stride).tolist():
            idx.append((ci, t))
    return idx


# --------------------------------------------------------------------------- #
# one arm (OFF or ON)                                                         #
# --------------------------------------------------------------------------- #
def run_arm(cond_on: bool, enc, readout, enc_sd0, ro_sd0, state_dim: int,
            bank: list[dict], eval_specs: dict, device: str, *, ft_blocks: int,
            ft_steps: int, batch: int, enc_lr: float, seed: int, k: int = 4) -> dict:
    torch.manual_seed(seed)
    enc.load_state_dict(enc_sd0)
    readout.load_state_dict(ro_sd0)
    k_frozen = FT.set_trainable(enc, readout, ft_blocks)
    d_model = enc.norm.normalized_shape[0]
    cam_enc = CamEncoding(d_model).to(device).train()
    inject = make_inject(d_model, ft_blocks).to(device).train()

    index = window_index(bank, k, stride=2)
    n = len(index)
    log(f"  arm {'ON' if cond_on else 'OFF'}: {len(bank)} clips, {n} windows,"
        f" k_frozen={k_frozen}, ft_blocks={ft_blocks}")

    head = ih.IDMHead(state_dim=state_dim, horizons=ih.DEFAULT_HORIZONS).to(device)
    groups = [{"params": head.parameters(), "lr": 3e-4},
              {"params": list(cam_enc.parameters()) + list(inject.parameters()),
               "lr": 3e-4}]
    if ft_blocks > 0:
        ep = [p for blk in enc.blocks[k_frozen:] for p in blk.parameters()]
        ep += list(enc.norm.parameters()) + list(readout.parameters())
        groups.append({"params": ep, "lr": enc_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    warm = max(10, ft_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warm) * (0.5 * (1 + math.cos(
            min(s, ft_steps) / ft_steps * math.pi)) if s >= warm else 1.0))

    Sall = [ih.scalar_targets_at(bank[ci]["poses"], bank[ci]["actions"],
                                 torch.tensor([t])) for ci, t in index]
    std = ih.Standardizer.fit(torch.cat(Sall))

    g = torch.Generator().manual_seed(seed)
    for step in range(ft_steps):
        sel = torch.randint(n, (batch,), generator=g).tolist()
        hs, scs, tjs, cams = [], [], [], []
        for si in sel:
            ci, t = index[si]
            hs.append(bank[ci]["prefix"][t - k:t + k + 1])    # [9,N,D] fp16
            tt = torch.tensor([t])
            scs.append(ih.scalar_targets_at(bank[ci]["poses"],
                                            bank[ci]["actions"], tt))
            tjs.append(ih.traj_targets_at(bank[ci]["poses"], tt))
            cams.append(bank[ci]["cam"] if cond_on else OFF_CAM)
        hb = torch.stack(hs).to(device).float()               # [B,9,N,D]
        B, W = hb.shape[:2]
        cam16 = torch.stack(cams).to(device).unsqueeze(1).expand(B, W, 16) \
            .reshape(B * W, 16)
        z = suffix_cond(enc, readout, cam_enc, inject,
                        hb.reshape(B * W, *hb.shape[2:]), cam16,
                        k_frozen).reshape(B, W, -1)
        ld = ih.idm_loss(head(z), torch.cat(scs).to(device),
                         torch.cat(tjs).to(device), std)
        opt.zero_grad(set_to_none=True)
        ld["loss"].backward()
        opt.step()
        sched.step()
        if step % max(1, ft_steps // 8) == 0 or step == ft_steps - 1:
            log(f"    step {step}/{ft_steps} loss {float(ld['loss'].detach()):.4f}")

    enc.eval(); readout.eval(); cam_enc.eval(); inject.eval()
    val = {}
    for name, specs in eval_specs.items():          # STREAM one clip at a time
        Z, S, T = [], [], []
        for p, cam in specs:
            d = R._load_ep(p)
            c = cam if cond_on else OFF_CAM
            z = full_encode_cond(enc, readout, cam_enc, inject, d["frames_u8"],
                                 c, k_frozen, device).float()
            zw, sc, tj = ih.build_windows(z, d["poses"].float(),
                                          d["actions"].float(), k=k, stride=2)
            if zw.shape[0]:
                Z.append(zw); S.append(sc); T.append(tj)
            del d, z
        val[name] = ih.evaluate(head, torch.cat(Z), torch.cat(S), torch.cat(T),
                                device=device)
        log(f"    eval {name}: n={val[name]['n']} "
            f"speedR2={val[name]['r2']['speed']:.3f} "
            f"yawR2={val[name]['r2']['yaw_rate']:.3f} ade={val[name]['ade_2s']:.3f}")
    return {"arm": "ON" if cond_on else "OFF", "train_windows": n, "val": val}


def verdict(cross, indom_ade):
    r2 = cross["r2"]
    return {"cross_speed_r2": r2["speed"], "cross_yaw_r2": r2["yaw_rate"],
            "cross_steer_r2": r2["steer"], "cross_ade_2s": cross["ade_2s"],
            "in_domain_ade_2s": indom_ade,
            "ade_ratio": cross["ade_2s"] / max(indom_ade, 1e-9),
            "PASS": bool(r2["speed"] > 0.9 and r2["yaw_rate"] > 0.9
                         and cross["ade_2s"] < 1.5 * indom_ade)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["rig", "multirig"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pai-cache", required=True)
    ap.add_argument("--comma-cache", required=True)
    ap.add_argument("--rig-table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ft-blocks", type=int, default=4)
    ap.add_argument("--ft-steps", type=int, default=1000)
    ap.add_argument("--enc-lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-eval-cross", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--git-hash", default="camcond-ablation")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rig_table = json.loads(Path(args.rig_table).read_text())
    enc, readout, meta = R.load_encoder(args.ckpt, device)
    state_dim = meta["state_dim"]
    enc_sd0 = {k: v.detach().clone() for k, v in enc.state_dict().items()}
    ro_sd0 = {k: v.detach().clone() for k, v in readout.state_dict().items()}
    k_frozen = len(enc.blocks) - args.ft_blocks

    a_eps, b_eps = R.select_episodes(rig_table, args.pai_cache, 400, 400)

    def pc(tag_paths, rig):
        out = []
        for tag, p in tag_paths:
            idx = int(tag.split("_")[-1])
            cy = rig_table[str(idx)]["cy"] or (542.0 if rig == "a" else 753.0)
            out.append((p, clip_cam(cy, 1.0)))
        return out

    a_pc = pc(a_eps, "a")               # rig-A (cy~542, fisheye)
    b_pc = pc(b_eps, "b")               # rig-B (cy~753, fisheye) — held-out rig
    comma_paths = sorted(Path(args.comma_cache).glob("ep_*.pt"))
    comma_pc = [(str(p), clip_cam(128.0, 0.0)) for p in comma_paths]  # rectilinear

    if args.experiment == "rig":
        train_specs = a_pc[:60]
        eval_specs = {"in_rig_heldout_rigA": a_pc[60:100],
                      "cross_rig_rigB": b_pc[:args.n_eval_cross]}
        cross_key, indom_key = "cross_rig_rigB", "in_rig_heldout_rigA"
        baseline = "single-domain rigA->rigB light-FT speed R2 = -1.65 (results_regate)"
    else:  # multirig {rig-A + comma} -> held-out rig-B
        train_specs = a_pc[:60] + comma_pc[:40]
        eval_specs = {"in_rigA_heldout": a_pc[60:100],
                      "in_comma_heldout": comma_pc[40:80],
                      "cross_heldout_rigB": b_pc[:args.n_eval_cross]}
        cross_key, indom_key = "cross_heldout_rigB", "in_rigA_heldout"
        baseline = "multirig {rigA+comma}->rigB light-FT speed R2 = -1.61 (results_multirig)"

    log(f"experiment {args.experiment}: {len(train_specs)} train clips; baseline {baseline}")
    log(f"banking frozen prefix (k_frozen={k_frozen}) once for both arms...")
    bank = build_prefix_bank(enc, train_specs, k_frozen, device)

    res = {"meta": {"experiment": f"camcond_ablation_{args.experiment}",
                    "design": "PRE_REGISTRATION.md (own-dynamics-encoder)",
                    "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "ckpt_md5": R.md5_of(args.ckpt), "ckpt_step": meta["ckpt_step"],
                    "device": device, "git_hash": args.git_hash,
                    "ft_blocks": args.ft_blocks, "ft_steps": args.ft_steps,
                    "enc_lr": args.enc_lr, "n_train_clips": len(train_specs),
                    "baseline": baseline,
                    "pass_rule": "cross speed R2>0.9 AND yaw R2>0.9 AND ADE<1.5x in-domain"},
           "arms": {}}
    for cond_on in (False, True):
        arm = run_arm(cond_on, enc, readout, enc_sd0, ro_sd0, state_dim, bank,
                      eval_specs, device, ft_blocks=args.ft_blocks,
                      ft_steps=args.ft_steps, batch=args.batch,
                      enc_lr=args.enc_lr, seed=args.seed)
        arm["verdict"] = verdict(arm["val"][cross_key],
                                 arm["val"][indom_key]["ade_2s"])
        res["arms"]["ON" if cond_on else "OFF"] = arm
        log(f"  {'ON' if cond_on else 'OFF'} VERDICT {json.dumps(arm['verdict'])}")

    off, on = res["arms"]["OFF"]["verdict"], res["arms"]["ON"]["verdict"]
    res["comparison"] = {
        "cross_speed_r2_OFF": off["cross_speed_r2"],
        "cross_speed_r2_ON": on["cross_speed_r2"],
        "delta_speed_r2_ON_minus_OFF": on["cross_speed_r2"] - off["cross_speed_r2"],
        "cross_yaw_r2_OFF": off["cross_yaw_r2"], "cross_yaw_r2_ON": on["cross_yaw_r2"],
        "cross_ade_OFF": off["cross_ade_2s"], "cross_ade_ON": on["cross_ade_2s"],
        "ON_PASS": on["PASS"], "OFF_PASS": off["PASS"]}
    Path(args.out).write_text(json.dumps(res, indent=2))
    log(f"WROTE {args.out}")
    log(f"COMPARISON {json.dumps(res['comparison'])}")
    log("CAMCOND_ABLATION_DONE")


if __name__ == "__main__":
    main()
