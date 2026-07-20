"""flagship v1.6 trainer — LP-FT: UNFREEZE the v1 trunk under the trained v1.5 head.

WHY (read the v1.5 ladder first). v1.5 established that the imagination
conditioning is load-bearing (`a->ab` = -0.1355 m, CI [0.038, 0.233], SIGNIFICANT)
while goal conditioning is null. Best arm `ab` = 0.5437 heldout, still 0.086 m
short of G1. The bottleneck is localised and is NOT ranking: `ab`'s oracle-in-fan
is 0.3073 m vs REF-C-XL's canonical 0.1640 m — the *proposal set* is ~2x worse
because the trunk is FROZEN at imitation-optimal features (and the head saw only
8 k steps). REF-C-XL reaches 0.1640 with a decoder SMALLER than v1.5's head
(22.70 M vs 29.76 M), so proposal quality comes from the trunk being trained
end-to-end, not from decoder size. v1.6 tests exactly that: unfreeze the trunk.

LP-FT, IN THE RIGHT ORDER (Kumar et al., ICLR'22 — naive FT distorts good
pretrained features; linear-probe THEN fine-tune preserves them). v1.5 already
did the linear-probe phase (8 k head-only steps). v1.6 is the fine-tune phase:

  * WARM START from the trained `ab` head (never re-init the head): --warm-head.
  * UNFREEZE the operative predictor (--unfreeze-predictor) — the proven
    mechanism; its imagined latents adapt from imitation- to planning-optimal.
  * UNFREEZE the last N encoder blocks + final norm + readout
    (--unfreeze-enc-blocks N, start N=4 of 12); early blocks stay frozen.
  * DISCRIMINATIVE LR: head ~1e-4, trunk ~1e-5 (~1/10) — --lr-head / --lr-trunk.
  * GRADUAL UNFREEZING: trunk LR held at 0 for the first --trunk-warmup steps so
    the head re-settles on the (still-frozen) features before the trunk moves.

  Unlike v1.5, the encoder is UNFROZEN, so the cached `states_*.pt` are stale —
  v1.6 re-encodes RAW FRAMES from the epcache every step. This is the "days not
  hours" cost v1.5 avoided on purpose; it is the price of moving the trunk.

CANARY (world-model collapse guard). The frozen trunk's own operative rollout
(predictor under TRUE actions -> grounding.step['op'] -> SE(2)) scores ~0.452 m
ADE@2s. If fine-tuning destroys the world model that number blows up. v1.6
establishes its OWN frozen-trunk baseline at step 0 (harness-consistent) and
re-checks it every eval; a rising canary means back off --lr-trunk. The full
external check is scripts/eval_grounded_rollout_4b_speed.py on the saved trunk
checkpoint (same code that produced the 0.452 reference).

PRIMARY READ = oracle-in-fan (proposal quality — the thing this experiment moves)
and frac_sel_2x_worse, reported against `ab`'s 0.3073 / 0.318 every eval. If
ADE improves but oracle does not, we bought ranking, not proposals, and the
experiment did NOT do what it was designed to do.

Loss: the v1.5 head loss verbatim (v15_losses = DiffusionDrive anchor-cls CE +
L1 traj recon + refined-rank CE). There is no separate trunk loss — the trunk
adapts to minimise the SAME planning objective (the LP-FT contract). The decoder
is kept exactly as `ab` (d512 / 8 layers / 256 anchors) so the warm start loads
strict; the capacity review found it already exceeds XL's decoder, so spending
budget on the trunk (this script), not the decoder, is the justified move.

Usage (pod2, the host that holds /workspace/v15 + the ab ckpt + the frames):
  PYTHONPATH=/workspace/TanitAD/stack python3 train_flagship_v16.py \
    --poses-train /workspace/v15/poses_train.pt \
    --poses-val   /workspace/v15/poses_val.pt \
    --labels-train /workspace/v15/labels_train.pt \
    --labels-val   /workspace/v15/labels_val.pt \
    --train-cache /workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894 \
    --val-cache   /workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11 \
    --trunk /workspace/experiments/flagship4b-speedjerk-30k/ckpt.pt \
    --warm-head /workspace/experiments/flagship-v15-ab/ckpt_best.pt \
    --anchors /workspace/v15/anchors256.pt --probes /workspace/v15/probes8.pt \
    --unfreeze-enc-blocks 4 --unfreeze-predictor \
    --lr-head 1e-4 --lr-trunk 1e-5 --trunk-warmup 500 \
    --steps 6000 --batch 16 --out /workspace/experiments/flagship-v16-ab-ft
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint

from tanitad.models.flagship_v15 import (SPEED_SCALE, FlagshipV15Head, V15Config,
                                         imagine_probes, param_breakdown,
                                         v15_ablation_config, v15_losses)
from tanitad.models.metric_dynamics import rollout_decode
from tanitad.refs.refc import DecoderConfig
from v15_prep import HORIZONS, K_MAX, WINDOW, load_frozen_v1


def _ego(dxy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * dxy[:, 0] + s * dxy[:, 1],
                        -s * dxy[:, 0] + c * dxy[:, 1]], dim=-1)


# --------------------------------------------------------------------------- #
# dataset — RAW FRAMES windows (encoder is unfrozen, so states can't be cached) #
# --------------------------------------------------------------------------- #
class V16FramesDataset(Dataset):
    """Windows over the epcache FRAMES + the precomputed v1.5 label artifact.

    Same pose-derived targets and v2.1 label handling as
    ``train_flagship_v15.V15Dataset`` (proven), but the per-item state source is
    RAW FRAMES read from the epcache — the encoder now trains, so states cannot
    be precomputed. ``eids`` (poses/labels) are epcache filenames (`ep_00000.pt`),
    so the frames come from ``<cache>/<eid>``.

    Item: frames [W, 9, 256, 256] uint8 · actions [W, 3] · v0 [] ·
    future_actions [K_MAX, 2] (for the canary) · traj_tgt [4, 2] · vt/route labels.
    """

    def __init__(self, cache: str, poses_pt: str, labels_pt: str,
                 stride: int = 1, episodes: int = 0, label_set: str = "v21"):
        pd = torch.load(poses_pt, weights_only=False)
        ld = torch.load(labels_pt, weights_only=False)
        n = min(len(pd["eids"]), len(ld["eids"]))
        if not (pd["eids"][:n] == ld["eids"][:n]):
            raise SystemExit("pose / label caches disagree on episode order — "
                             "refusing to train on misaligned labels")
        self.cache = cache
        self.eids = pd["eids"][:n]
        # sanity: the frames files the eids point at must exist
        miss = [e for e in self.eids[:5] if not os.path.exists(os.path.join(cache, e))]
        if miss:
            raise SystemExit(f"epcache {cache} is missing frames for {miss} — "
                             "does --train-cache point at the split dir?")
        self.label_set = label_set
        n_ep = n if not episodes else min(episodes, n)
        self.index: list[tuple[int, int]] = []
        self.traj: list[torch.Tensor] = []
        self.acts: list[torch.Tensor] = []
        self.v0: list[torch.Tensor] = []
        vt_key = "vt_band_v2" if label_set == "v21" else "vt_band_raw"
        self.vband = ld[vt_key]
        self.vspeed = ld["vt_v2" if label_set == "v21" else "vt_raw"]
        if label_set == "v21":
            self.route = ld["route_v21"]
            self.rgraded = ld["route_graded"]
        else:
            self.route = ld["route_legacy"]
            self.rgraded = [torch.zeros_like(x, dtype=torch.float32)
                            for x in ld["route_legacy"]]
        self.label_stats = ld.get("stats", {})
        for e in range(n_ep):
            po = torch.as_tensor(pd["poses"][e], dtype=torch.float32)
            ac = torch.as_tensor(pd["actions"][e], dtype=torch.float32)
            n_w = po.shape[0] - WINDOW - K_MAX
            self.acts.append(ac)
            self.v0.append(po[:, 3])
            if n_w <= 0:
                self.traj.append(torch.zeros(0, len(HORIZONS), 2))
                continue
            last = torch.arange(n_w) + WINDOW - 1
            yaw = po[last, 2]
            wps = [_ego(po[last + k, :2] - po[last, :2], yaw) for k in HORIZONS]
            self.traj.append(torch.stack(wps, dim=1))
            if self.vband[e].shape[0] != n_w:
                raise SystemExit(
                    f"label/window count mismatch on episode {e}: "
                    f"{self.vband[e].shape[0]} labels vs {n_w} windows")
            self.index.extend((e, int(x)) for x in range(0, n_w, stride))

    def __len__(self) -> int:
        return len(self.index)

    def _frames(self, e: int, t: int) -> torch.Tensor:
        """mmap the ep file, slice the window, CLONE (MooseFS mmap-slice safety —
        the bus-error fix, memory D-026 / refbpatch: a raw mmap slice crossing the
        DataLoader worker boundary bus-errors).

        Keeps a small per-worker cache of open mmap handles so a shuffled epoch
        does not re-parse the zip header of a 117 MB episode on every item.
        """
        cache = getattr(self, "_mm", None)
        if cache is None:
            cache = self._mm = {}
        d = cache.get(e)
        if d is None:
            if len(cache) >= 8:                     # bounded: keep FDs/VMA sane
                cache.pop(next(iter(cache)))
            d = torch.load(os.path.join(self.cache, self.eids[e]),
                           map_location="cpu", weights_only=True, mmap=True)
            cache[e] = d
        return d["frames_u8"][t:t + WINDOW].clone()          # [W, 9, 256, 256] u8

    def __getitem__(self, i: int) -> dict:
        e, t = self.index[i]
        last = t + WINDOW - 1
        v0 = self.v0[e][last]
        fr = self._frames(e, t)
        a = self.acts[e][t:t + WINDOW]                        # [W, 2]
        a3 = torch.cat([a, (v0 / SPEED_SCALE).expand(WINDOW, 1)], dim=-1)
        fa = self.acts[e][t + WINDOW:t + WINDOW + K_MAX]      # [K_MAX, 2] canary
        return {"frames": fr, "actions": a3, "v0": v0,
                "future_actions": fa,
                "traj_tgt": self.traj[e][t], "vt_band": self.vband[e][t],
                "vt_speed": self.vspeed[e][t],
                "route": self.route[e][t], "route_graded": self.rgraded[e][t],
                "ep": e, "last": last}


# --------------------------------------------------------------------------- #
# encoder forward with a FREEZE BOUNDARY (frozen prefix no_grad, trainable tail) #
# --------------------------------------------------------------------------- #
def encode_window_ft(world, frames_u8: torch.Tensor, n_unfrozen: int,
                     use_ckpt: bool) -> torch.Tensor:
    """frames_u8 [B, W, 9, 256, 256] uint8 -> states [B, W, S].

    Runs patch + pos + the FROZEN prefix blocks under no_grad (no graph, no
    activation retention), then the last ``n_unfrozen`` blocks + final norm +
    readout WITH grad (optionally checkpointed to bound memory). ``n_unfrozen==0``
    reproduces the frozen ``WorldModel.encode`` numerically (whole pass no_grad).
    """
    enc = world.encoder
    b, w = frames_u8.shape[:2]
    x = frames_u8.reshape(b * w, *frames_u8.shape[2:]).float().div_(255.0)
    depth = len(enc.blocks)
    n_frozen = depth - n_unfrozen
    trainable = n_unfrozen > 0
    with torch.no_grad():
        t = enc.patch(x).flatten(2).transpose(1, 2) + enc.pos
        for blk in enc.blocks[:n_frozen]:
            t = blk(t)
    if not trainable:                                    # predictor-only mode
        with torch.no_grad():
            t = enc.norm(t)
            states = world.readout(t)
        return states.reshape(b, w, -1)
    for blk in enc.blocks[n_frozen:]:                    # trainable tail (grad on)
        if use_ckpt:
            t = checkpoint(blk, t, use_reentrant=False)
        else:
            t = blk(t)
    t = enc.norm(t)
    states = world.readout(t)
    return states.reshape(b, w, -1)


# --------------------------------------------------------------------------- #
# grad-enabled imagination (predictor unfrozen) — per-step checkpointed BPTT     #
# --------------------------------------------------------------------------- #
def imagine_probes_grad(predictor, states, actions, probes, read, v0n,
                        use_ckpt=True):
    """Same roll as flagship_v15.imagine_probes but WITH gradient into the
    predictor (used when the predictor is unfrozen). Each 1-step predictor call
    is checkpointed so the 20-step BPTT costs ~1 step of activation memory."""
    b, w, s = states.shape
    m, k, _ = probes.shape
    a_dim = actions.shape[-1]
    ws = states.unsqueeze(1).expand(b, m, w, s).reshape(b * m, w, s)
    wa = actions.unsqueeze(1).expand(b, m, w, a_dim).reshape(b * m, w, a_dim)
    pr = probes.unsqueeze(0).expand(b, m, k, 2).reshape(b * m, k, 2)
    v_col = v0n.reshape(b, 1, 1).expand(b, m, 1).reshape(b * m, 1)

    def _one(s_in, a_in):
        return predictor(s_in, a_in)[1]                  # 1-step head -> dict[1]

    reads, k_max = [], max(read)
    for j in range(k_max):
        z = checkpoint(_one, ws, wa, use_reentrant=False) if use_ckpt else _one(ws, wa)
        if (j + 1) in read:
            reads.append(z)
        if j < k_max - 1:
            a_next = (torch.cat([pr[:, min(j, k - 1)], v_col], dim=-1)
                      if a_dim == 3 else pr[:, min(j, k - 1)])
            ws = torch.cat([ws[:, 1:], z.unsqueeze(1)], dim=1)
            wa = torch.cat([wa[:, 1:], a_next.unsqueeze(1)], dim=1)
    out = torch.stack(reads, dim=1)
    return out.reshape(b, m * len(read), s)


# --------------------------------------------------------------------------- #
# canary — the frozen operative rollout (world-model collapse detector)         #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def canary_rollout(world, grounding, ds_val, device, episodes=40, stride=8,
                   batch=16, n_unfrozen=0, amp=True) -> dict:
    """Operative predictor rollout under TRUE actions -> grounding.step['op'] ->
    SE(2) accumulate -> ADE@2s. The SAME method as
    eval_grounded_rollout_4b_speed.py (which produced the ~0.452 reference).
    Recomputes states through the CURRENT (partially unfrozen) encoder so the
    canary reflects the trunk as it is being fine-tuned."""
    step_readout = grounding.step["op"]
    sel = [i for i, (e, t) in enumerate(ds_val.index)
           if e < episodes and t % stride == 0]
    errs = []
    wp_idx = torch.tensor([k - 1 for k in HORIZONS], device=device)
    for b0 in range(0, len(sel), batch):
        idx = sel[b0:b0 + batch]
        items = [ds_val[i] for i in idx]
        fr = torch.stack([x["frames"] for x in items]).to(device)
        aw = torch.stack([x["actions"] for x in items]).to(device)      # [B,W,3]
        v0 = torch.stack([x["v0"] for x in items]).to(device)
        fa2 = torch.stack([x["future_actions"] for x in items]).to(device)  # [B,K,2]
        gt = torch.stack([x["traj_tgt"] for x in items]).to(device)     # [B,4,2]
        fa = torch.cat([fa2, (v0 / SPEED_SCALE)[:, None, None]
                        .expand(-1, fa2.shape[1], -1)], dim=-1)          # [B,K,3]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            states = encode_window_ft(world, fr, n_unfrozen=0, use_ckpt=False)
            wp_full, _ = rollout_decode(world.predictor, states, aw, fa,
                                        step_readout, K_MAX)             # [B,K,2]
        pred = wp_full.index_select(1, wp_idx).float()
        errs.append((pred - gt).norm(dim=-1).mean(dim=1).cpu())         # [B]
    e = torch.cat(errs)
    return {"canary_ade@2s": float(e.mean()), "n": int(e.shape[0])}


# --------------------------------------------------------------------------- #
# planner eval — re-encode val frames through the CURRENT trunk                 #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(head, world, predictor_unfrozen, ds_val, probes, cfg, device,
             stride=8, batch=16, episodes=40, steps=None, n_unfrozen=0, amp=True):
    """ADE/FDE/miss + oracle-in-fan at the TanitEval waypoints, re-encoding val
    frames through the CURRENT (unfrozen) trunk. Mirrors
    train_flagship_v15.evaluate but with the live encoder."""
    head.eval()
    sel = [i for i, (e, t) in enumerate(ds_val.index)
           if e < episodes and t % stride == 0]
    preds, gts, fans = [], [], []
    for b0 in range(0, len(sel), batch):
        idx = sel[b0:b0 + batch]
        items = [ds_val[i] for i in idx]
        fr = torch.stack([x["frames"] for x in items]).to(device)
        ac = torch.stack([x["actions"] for x in items]).to(device)
        v0 = torch.stack([x["v0"] for x in items]).to(device)
        vb = torch.stack([x["vt_band"] for x in items]).to(device)
        rt = torch.stack([x["route"] for x in items]).to(device)
        rg = torch.stack([x["route_graded"] for x in items]).to(device)
        vs = torch.stack([x["vt_speed"] for x in items]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            st = encode_window_ft(world, fr, n_unfrozen=0, use_ckpt=False)
            imag = None
            if cfg.cond_imagination:
                imag = imagine_probes(world.predictor, st, ac, probes,
                                      cfg.imag_read, v0 / SPEED_SCALE)
            out = head(st, v0, imagined=imag, vt_band=vb, route=rt,
                       route_graded=rg, vt_speed=vs, steps=steps)
        preds.append(out["traj"].float().cpu())
        fans.append(out["anchor_traj"].float().cpu())
        gts.append(torch.stack([x["traj_tgt"] for x in items]))
    head.train()
    p, g = torch.cat(preds), torch.cat(gts)
    err = (p - g).norm(dim=-1)
    fan = torch.cat(fans)
    fan_err = (fan - g[:, None]).norm(dim=-1).mean(dim=-1)
    oracle = fan_err.min(dim=1).values
    sel_e = err.mean(dim=1)
    return {"n": int(p.shape[0]),
            "oracle_ade@2s": float(oracle.mean()),
            "sel_gap@2s": float((sel_e - oracle).mean()),
            "frac_sel_2x_worse": float((sel_e > 2.0 * oracle).float().mean()),
            "ade@0.5s": float(err[:, 0].mean()),
            "ade@1s": float(err[:, :2].mean()),
            "ade@1.5s": float(err[:, :3].mean()),
            "ade@2s": float(err.mean()),
            "fde@2s": float(err[:, -1].mean()),
            "miss@2m": float((err[:, -1] > 2.0).float().mean())}


# --------------------------------------------------------------------------- #
# selective unfreeze + discriminative-LR param groups                          #
# --------------------------------------------------------------------------- #
def configure_trainable(world, head, n_unfrozen: int, unfreeze_predictor: bool):
    """Freeze the whole trunk, then re-enable the last ``n_unfrozen`` encoder
    blocks + final norm + readout, and (optionally) the operative predictor.
    Returns (trunk_params, head_params, manifest)."""
    for p in world.parameters():
        p.requires_grad_(False)
    trunk, manifest = [], {"enc_blocks_unfrozen": n_unfrozen,
                           "enc_depth": len(world.encoder.blocks),
                           "predictor_unfrozen": bool(unfreeze_predictor),
                           "modules": []}
    depth = len(world.encoder.blocks)
    if n_unfrozen > 0:
        for bi in range(depth - n_unfrozen, depth):
            for p in world.encoder.blocks[bi].parameters():
                p.requires_grad_(True); trunk.append(p)
        for p in world.encoder.norm.parameters():
            p.requires_grad_(True); trunk.append(p)
        for p in world.readout.parameters():
            p.requires_grad_(True); trunk.append(p)
        manifest["modules"] += [f"encoder.blocks[{depth-n_unfrozen}:{depth}]",
                                "encoder.norm", "readout"]
    if unfreeze_predictor:
        for p in world.predictor.parameters():
            p.requires_grad_(True); trunk.append(p)
        manifest["modules"].append("predictor")
    head_params = [p for p in head.parameters() if p.requires_grad]
    manifest["n_trunk_trainable"] = sum(p.numel() for p in trunk)
    manifest["n_head_trainable"] = sum(p.numel() for p in head_params)
    return trunk, head_params, manifest


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


def trunk_lr(step, total, hold, ramp, base):
    """0 until ``hold`` (gradual unfreezing — head re-settles first), linear ramp
    over ``ramp`` steps, then cosine decay to 0 over the rest."""
    if step < hold:
        return 0.0
    s = step - hold
    if s < ramp:
        return base * (s + 1) / max(ramp, 1)
    p = (s - ramp) / max(total - hold - ramp, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses-train", required=True)
    ap.add_argument("--poses-val", required=True)
    ap.add_argument("--labels-train", required=True)
    ap.add_argument("--labels-val", required=True)
    ap.add_argument("--train-cache", required=True,
                    help="epcache SPLIT dir with ep_*.pt frames (train)")
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--trunk", required=True, help="frozen v1 ckpt to unfreeze")
    ap.add_argument("--warm-head", required=True,
                    help="the trained `ab` head ckpt (ckpt_best.pt) — LP phase")
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cond", choices=("a", "ab", "abc"), default="ab")
    ap.add_argument("--unfreeze-enc-blocks", type=int, default=4,
                    help="unfreeze the LAST N of the 12 encoder blocks (+norm+readout)")
    ap.add_argument("--unfreeze-predictor", action="store_true")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-trunk", type=float, default=1e-5)
    ap.add_argument("--head-warmup", type=int, default=200)
    ap.add_argument("--trunk-warmup", type=int, default=500,
                    help="hold trunk LR at 0 for this many steps (gradual unfreeze)")
    ap.add_argument("--trunk-ramp", type=int, default=200)
    ap.add_argument("--label-set", choices=("v21", "legacy"), default="v21")
    ap.add_argument("--no-enc-ckpt", action="store_true",
                    help="disable grad-checkpoint on the trainable encoder tail")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--episodes", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=40,
                    help="val episodes for the in-training eval AND the canary "
                         "(40 = the TanitEval window set; lower only to smoke)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    dev = a.device
    amp = (not a.no_amp) and dev == "cuda"
    use_enc_ckpt = not a.no_enc_ckpt
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    cfg = v15_ablation_config(states=True,
                              imagination=a.cond in ("ab", "abc"),
                              vtarget=a.cond == "abc")

    # --- trunk: load the frozen v1, KEEP the encoder (v1.6 re-encodes frames) --
    world, grounding, trunk_step = load_frozen_v1(a.trunk, dev)
    object.__setattr__(world.encoder.cfg, "grad_checkpoint", use_enc_ckpt)
    world.train()

    anc = torch.load(a.anchors, weights_only=False)
    anchors = anc["anchors"] if isinstance(anc, dict) else anc
    prb = torch.load(a.probes, weights_only=False)
    probes = (prb["probes"] if isinstance(prb, dict) else prb).to(dev)

    # --- head: WARM START from the trained ab checkpoint (never re-init) -------
    head = FlagshipV15Head(cfg).to(dev)
    head.load_anchors(anchors.to(dev))
    hck = torch.load(a.warm_head, map_location=dev, weights_only=False)
    head.load_state_dict(hck["head"])                    # STRICT — same geometry
    print(f"[v16] warm-started head from {a.warm_head} "
          f"(ab step {hck.get('step')})", flush=True)

    trunk_params, head_params, unfreeze = configure_trainable(
        world, head, a.unfreeze_enc_blocks, a.unfreeze_predictor)
    pb = param_breakdown(head)
    print(f"[v16] cond={a.cond} head params={pb['total']:,} | UNFREEZE {unfreeze}",
          flush=True)

    ds = V16FramesDataset(a.train_cache, a.poses_train, a.labels_train,
                          episodes=a.episodes, label_set=a.label_set)
    ds_val = V16FramesDataset(a.val_cache, a.poses_val, a.labels_val,
                              episodes=a.eval_episodes, label_set=a.label_set)
    print(f"[data] train windows={len(ds)} val windows={len(ds_val)}", flush=True)

    # frames are ~4.7 MB/item, so prefetch is deliberately shallow (a batch of 16
    # is ~75 MB uint8; deep prefetch pins GBs for no throughput gain).
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                    num_workers=a.workers, persistent_workers=a.workers > 0,
                    pin_memory=True, prefetch_factor=2 if a.workers else None)
    opt = torch.optim.AdamW(
        [{"params": head_params, "lr": a.lr_head, "name": "head"},
         {"params": trunk_params, "lr": a.lr_trunk, "name": "trunk"}],
        weight_decay=0.01)

    (out / "config.json").write_text(json.dumps({
        "arch": "flagship-v1.6 (LP-FT: unfrozen v1 trunk + warm ab head)",
        "cond": a.cond, "cfg": dataclasses.asdict(cfg), "args": vars(a),
        "trunk": {"ckpt": a.trunk, "step": trunk_step},
        "warm_head": a.warm_head, "unfreeze": unfreeze,
        "param_breakdown_head": pb,
        "optimizer": {"kind": "AdamW", "lr_head": a.lr_head,
                      "lr_trunk": a.lr_trunk, "wd": 0.01,
                      "head_warmup": a.head_warmup,
                      "trunk_warmup_hold": a.trunk_warmup,
                      "trunk_ramp": a.trunk_ramp, "schedule": "cosine"},
        "lp_ft": "Kumar et al. ICLR'22 — LP(v1.5 8k head-only) then FT(this).",
        "baselines": {"ab_oracle@2s": 0.3073, "ab_frac_2x": 0.318,
                      "ab_ade@2s_heldout": 0.5437,
                      "canary_frozen_ref": 0.452},
    }, indent=2, default=str), encoding="utf-8")

    log_f = (out / "train_log.jsonl").open("a")
    ckpt_p = out / "ckpt.pt"
    step = 0
    if ckpt_p.exists():
        ck = torch.load(ckpt_p, map_location=dev, weights_only=False)
        head.load_state_dict(ck["head"]); world.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"]); step = int(ck["step"]) + 1
        print(f"[resume] step {step}", flush=True)

    def save_ckpt(path, with_opt=True):
        obj = {"head": head.state_dict(), "model": world.state_dict(),
               "grounding": grounding.state_dict(), "step": step,
               "cfg": dataclasses.asdict(cfg), "unfreeze": unfreeze}
        if with_opt:
            obj["opt"] = opt.state_dict()
        tmp = Path(path).with_suffix(".tmp")
        torch.save(obj, tmp); tmp.replace(path)

    anchors_d = head.decoder.anchors
    n_unf = a.unfreeze_enc_blocks

    # --- canary BASELINE on the (still-frozen, step-0) trunk -------------------
    base_canary = canary_rollout(world, grounding, ds_val, dev,
                                 episodes=a.eval_episodes, batch=a.batch, amp=amp)
    print(json.dumps({"step": step, "canary_baseline": base_canary,
                      "ref_0.452": 0.452}), flush=True)
    log_f.write(json.dumps({"step": step, "canary_baseline": base_canary}) + "\n")
    log_f.flush()

    it = iter(dl); t0 = time.time()
    best = float("inf"); best_p = out / "ckpt_best.pt"
    while step < a.steps:
        lr_h = cosine_lr(step, a.steps, a.head_warmup, a.lr_head)
        lr_t = trunk_lr(step, a.steps, a.trunk_warmup, a.trunk_ramp, a.lr_trunk)
        for pg in opt.param_groups:
            pg["lr"] = lr_h if pg.get("name") == "head" else lr_t
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl); b = next(it)
        fr = b["frames"].to(dev, non_blocking=True)
        ac = b["actions"].to(dev, non_blocking=True)
        v0 = b["v0"].to(dev, non_blocking=True)
        vb = b["vt_band"].to(dev, non_blocking=True)
        rt = b["route"].to(dev, non_blocking=True)
        rg = b["route_graded"].to(dev, non_blocking=True)
        vs = b["vt_speed"].to(dev, non_blocking=True)
        tgt = b["traj_tgt"].to(dev, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            st = encode_window_ft(world, fr, n_unf, use_enc_ckpt)
            imag = None
            if cfg.cond_imagination:
                imag = (imagine_probes_grad(world.predictor, st, ac, probes,
                                            cfg.imag_read, v0 / SPEED_SCALE)
                        if a.unfreeze_predictor else
                        imagine_probes(world.predictor, st, ac, probes,
                                       cfg.imag_read, v0 / SPEED_SCALE))
            o = head(st, v0, imagined=imag, vt_band=vb, route=rt,
                     route_graded=rg, vt_speed=vs)
            L = v15_losses(o, anchors_d, tgt)
        opt.zero_grad(set_to_none=True)
        L["loss"].backward()
        gn_h = float(torch.nn.utils.clip_grad_norm_(head_params, 1.0))
        gn_t = (float(torch.nn.utils.clip_grad_norm_(trunk_params, 1.0))
                if trunk_params else 0.0)
        opt.step()

        if step % a.log_every == 0 or step == a.steps - 1:
            row = {"step": step, "lr_head": round(lr_h, 8),
                   "lr_trunk": round(lr_t, 8), "amp": amp,
                   "loss": round(float(L["loss"]), 5),
                   "traj": round(float(L["traj"]), 5),
                   "cls": round(float(L["cls"]), 5),
                   "cls_refined": round(float(L["cls_refined"]), 5),
                   "anchor_acc": round(float(L["anchor_acc"]), 4),
                   "rank_acc": round(float(L["rank_acc"]), 4),
                   "train_ade": round(float(L["ade"]), 4),
                   "oracle_ade": round(float(L["oracle_ade"]), 4),
                   "sel_gap": round(float(L["sel_gap"]), 4),
                   "sel_2x_worse": round(
                       float(L["frac_sel_2x_worse_than_oracle"]), 4),
                   "gnorm_head": round(gn_h, 3), "gnorm_trunk": round(gn_t, 3),
                   "elapsed_s": round(time.time() - t0, 1)}
            print(json.dumps(row), flush=True)
            log_f.write(json.dumps(row) + "\n"); log_f.flush()

        if step > 0 and step % a.eval_every == 0:
            ev = evaluate(head, world, a.unfreeze_predictor, ds_val, probes, cfg,
                          dev, batch=a.batch, episodes=a.eval_episodes,
                          n_unfrozen=n_unf, amp=amp)
            can = canary_rollout(world, grounding, ds_val, dev, batch=a.batch,
                                 episodes=a.eval_episodes, amp=amp)
            row = {"step": step,
                   "val": {k: round(v, 5) if isinstance(v, float) else v
                           for k, v in ev.items()},
                   "canary_ade@2s": round(can["canary_ade@2s"], 5),
                   "canary_vs_base": round(
                       can["canary_ade@2s"] - base_canary["canary_ade@2s"], 5)}
            print(json.dumps(row), flush=True)
            log_f.write(json.dumps(row) + "\n"); log_f.flush()
            if ev["ade@2s"] < best:
                best = ev["ade@2s"]
                save_ckpt(best_p, with_opt=False)

        if step > 0 and step % a.save_every == 0:
            save_ckpt(ckpt_p)
        step += 1

    save_ckpt(ckpt_p)
    ev = evaluate(head, world, a.unfreeze_predictor, ds_val, probes, cfg, dev,
                  batch=a.batch, episodes=a.eval_episodes, n_unfrozen=n_unf,
                  amp=amp)
    can = canary_rollout(world, grounding, ds_val, dev, batch=a.batch,
                         episodes=a.eval_episodes, amp=amp)
    (out / "metrics.json").write_text(json.dumps(
        {"final_step": step - 1, "val": ev, "best_val_ade2s": best,
         "canary_ade@2s": can["canary_ade@2s"],
         "canary_baseline": base_canary["canary_ade@2s"],
         "unfreeze": unfreeze, "cond": a.cond,
         "wallclock_s": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "cond": a.cond, "val": ev,
                      "canary": can}), flush=True)


if __name__ == "__main__":
    main()
