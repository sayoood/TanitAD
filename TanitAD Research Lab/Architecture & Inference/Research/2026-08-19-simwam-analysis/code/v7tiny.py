"""v7-tiny G2 — does the predictor beat HOLD? The gate no model has ever passed.

PI settings, 2026-08-22: keep resolution (256x640), stick to PhysicalAI data, do
G2 first, shorter horizon is allowed.

G2, from `V7_TINY_DESIGN.md`:

    explained_movement = 1 - ||zhat_{t+1} - z_{t+1}||^2 / ||z_t - z_{t+1}||^2

    > 0   the predictor explains some of the latent's real movement
    = 0   exactly as good as HOLD (predict no change) -- no dynamics learned
    < 0   WORSE than hold

MEASURED on v6F @20k: 580.5x WORSE than hold at the true dt=0.1s tick. No arm in
the programme has ever cleared this, and it needs NO LABELS -- only the model's
own latents -- which is why it runs first and gates everything else.

⭐ IT USES THE REAL `OperativePredictor`. A proxy predictor would validate the
proxy. This trains v6/v7's actual class so a pass is a statement about the
component v7 will ship.

⛔⛔ THE ARMS.

    fixed     RESIDUAL_HEAD_INIT_SCALE applied (v7's design), 2-term core
    regress   the DEFECT RE-INTRODUCED on purpose -- default residual init.
              If G2 does not FAIL this, the gate cannot detect the bug it
              exists to catch and a pass means nothing (the `oracle` discipline).

⭐ THE O-TERM LADDER (PI, 2026-08-22): "the o terms were introduced to teach the
model the physical world and dynamics, so were measures taken from the failed v5
results. So let run the current test and reintroduce them in v7 tiny to
understand their effects."

That is the right correction: the 2-term core is LeWM's recipe, but it SILENTLY
DISCARDS five remedies for MEASURED v5 failures. Each is added back one at a
time, so its effect on G2 is attributable. From `train_v6_staged.py`'s header:

    O1  action-conditioned prediction with L_ctrl in RESPONSE FORM from step 0
        -- the ANTI-ACTION-ECHO measure (imported from stage A: counterfactual
        arms + physical envelope clamp)
    O2  near-field latent loss weighted by TIME-TO-REACH, not a fixed 40 m band
        (PI: "a fixed 40 m band cannot cover a 6 s horizon")
    O3  masked SPATIAL-latent prediction over the readout grid (I-JEPA on cells)
    O4  interaction-weighted SAMPLING from actions only -- jerk, decel, steering
        reversals. Label-free; reweights the draw, never removes a window
    O5  multi-step rollout consistency, error at EVERY step -- "the P5
        compounding lesson trained in"
    O6  SIGReg (LeJEPA) -- anti-collapse. Always on; it is the core's 2nd term

⚠️ THE REAL IMPLEMENTATIONS ARE IMPORTED from `train_v6_staged.py`, not
reimplemented -- a reimplementation would ablate my copy of the term, not v6's.

⚠️ O1 IS TIER-2 AND NOT YET WIRED. `stage_a_losses` needs a STEP READOUT
(latent transition -> delta-pose) and ground-truth waypoints, which this rig does
not build. It is the anti-action-echo term and therefore the most interesting
one, so it is called out as MISSING rather than quietly dropped.

⚠️ WHAT THIS DOES NOT TEST. G2 is label-free and says only that the predictor
models the latent's own dynamics. It does NOT say the latent carries the world
(that is G3, E-DETECT-1) and says nothing about driving (T1). A pass licenses
proceeding, not a capability claim.

TIER: T0-DIAGNOSTIC. Dev-box only; Thor untouched.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
REPO = Path("G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(SP))
sys.path.insert(0, str(REPO / "stack"))

EPS = SP / "sp2/cache/slotprobe-lead130-w120-256x640cyl"
BANK = SP / "v7tiny"
#: PI: KEEP RESOLUTION. 256x640 patch 16 -> 16x40 = 640 tokens, so a vehicle
#: still spans 1.22 tokens at 30 m (MEASURED) and the perception task stays
#: possible. Halving this would put vehicles sub-token beyond ~18 m.
IMG_H, IMG_W, PATCH = 256, 640, 16
GRID_H, GRID_W = IMG_H // PATCH, IMG_W // PATCH
N_TOK = GRID_H * GRID_W
N_STACK = 3                      # v6's 9-channel input: 3 frames as channels
WINDOW = 6
#: PI allowed a SHORTER horizon. v6 rolls o5_k=60 (6.0 s); 8 ticks (0.8 s) still
#: exposes compounding error at a fraction of the cost.
K_ROLL = 8
D_OP = 2048                      # 16 cells x 128, v6's operative width


def build_bank(n_clips: int) -> tuple[np.ndarray, list[dict]]:
    """256x640 RGB frames for `n_clips` PhysicalAI clips, banked once."""
    BANK.mkdir(parents=True, exist_ok=True)
    fp, mp = BANK / f"frames{n_clips}.npy", BANK / f"meta{n_clips}.json"
    if fp.exists() and mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
        return np.load(fp, mmap_mode="r"), meta["clips"]
    cids = sorted(p.name.split(".")[0] for p in EPS.glob("*.v2ep.pt"))[:n_clips]
    clips, total = [], 0
    per = []
    for cid in cids:
        d = torch.load(EPS / f"{cid}.v2ep.pt", map_location="cpu",
                       weights_only=False)
        n = len(d["jpeg_len"])
        clips.append({"clip_id": cid, "start": total, "n": n})
        per.append((cid, d, n))
        total += n
    m = np.lib.format.open_memmap(fp, mode="w+", dtype=np.uint8,
                                  shape=(total, 3, IMG_H, IMG_W))
    for cid, d, n in per:
        raw = d["jpeg_buf"].numpy().tobytes()
        off = np.concatenate([[0], np.cumsum(d["jpeg_len"].tolist())]).astype(
            np.int64)
        s = next(c["start"] for c in clips if c["clip_id"] == cid)
        for i in range(n):
            im = Image.open(io.BytesIO(raw[off[i]:off[i + 1]])).convert("RGB")
            a = np.asarray(im, dtype=np.uint8)
            if a.shape[:2] != (IMG_H, IMG_W):
                raise SystemExit(f"[FATAL] {cid} f{i}: {a.shape[:2]}")
            m[s + i] = a.transpose(2, 0, 1)
        print(f"    banked {cid[:8]} {n} frames", flush=True)
    m.flush()
    # CONTENT CHECK -- a pre-allocated memmap that never got written is a
    # full-size file of zeros that still passes `ls`.
    smp = np.asarray(m[::97])
    if (smp.reshape(len(smp), -1).max(1) == 0).any():
        raise SystemExit("[FATAL] frame bank has all-zero rows")
    acts, spds = [], []
    for cid, d, n in per:
        acts.append(d["actions"].float()[:n])
        spds.append(d["poses"].float()[:n, 3])
    np.save(BANK / f"act{n_clips}.npy", torch.cat(acts).numpy())
    np.save(BANK / f"spd{n_clips}.npy", torch.cat(spds).numpy())
    mp.write_text(json.dumps({"clips": clips, "n": total}), encoding="utf-8")
    print(f"  bank OK: {total} frames, mean {float(smp.mean()):.1f}")
    return np.load(fp, mmap_mode="r"), clips


class TinyEncoder(nn.Module):
    """Small ViT over the 9-channel (3-frame) stack -> v6's 4x4 readout."""

    def __init__(self, d: int = 192, depth: int = 4, heads: int = 3):
        super().__init__()
        self.patch = nn.Conv2d(3 * N_STACK, d, PATCH, PATCH)
        self.pos = nn.Parameter(torch.randn(1, N_TOK, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True,
                                           norm_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 128)
        self.d_latent = D_OP

    def forward(self, x):                       # [B, 9, H, W]
        t = self.norm(self.blocks(self.patch(x).flatten(2).transpose(1, 2)
                                  + self.pos))
        g = t.transpose(1, 2).reshape(-1, t.shape[-1], GRID_H, GRID_W)
        g = F.adaptive_avg_pool2d(g, (4, 4)).flatten(2).transpose(1, 2)
        return self.proj(g).flatten(1)          # [B, 2048]


def explained_movement(zhat, z_fut, z_now) -> dict:
    """G2. Inputs are numpy arrays -- `np.abs`, not Tensor `.abs()`."""
    zhat, z_fut, z_now = (np.asarray(x, dtype=np.float64)
                          for x in (zhat, z_fut, z_now))
    err = ((zhat - z_fut) ** 2).sum(-1)
    mov = ((z_now - z_fut) ** 2).sum(-1)
    ok = mov > 0
    mae_p = float(np.abs(zhat - z_fut).mean())
    mae_h = float(np.abs(z_now - z_fut).mean())
    return {"explained_movement":
                round(float(1 - err[ok].sum() / mov[ok].sum()), 4),
            "mae_pred": round(mae_p, 8), "mae_hold": round(mae_h, 8),
            "ratio_vs_hold": round(mae_p / max(mae_h, 1e-12), 4),
            "n": int(len(zhat))}


#: v6's own weights, so an arm reproduces v6's balance rather than inventing one.
O_WEIGHTS = {"o2": 1.0, "o3": 1.0, "o5": 1.0}
LADDER = {
    "core":     [],                       # LeWM's two terms only
    "o5":       ["o5"],                   # rollout consistency (P5)
    "o3":       ["o3"],                   # masked spatial cells (I-JEPA)
    "o2":       ["o2"],                   # near-field, time-to-reach
    "o5o3":     ["o5", "o3"],
    "all":      ["o5", "o3", "o2"],       # v6's stack minus the tier-2 O1
}


def cell_ranges(n_cells: int = 16, x_max: float = 60.0) -> torch.Tensor:
    """Forward range of each 4x4 readout cell, for O2's time-to-reach weight.

    ⚠️ Row-major over the 4x4 grid, so cell i's row index i//4 sets its range --
    the same convention `stack.cells` uses. A transposed guess here would weight
    the wrong cells and O2 would look ineffective for the wrong reason."""
    rows = torch.arange(n_cells) // 4
    return (rows.float() + 0.5) * (x_max / 4.0)


def run(arm: str, frames, clips, acts, spds, *, steps, batch, dev, lam=0.09,
        lr=3e-4, seed=0, terms=None) -> dict:
    from tanitad.config import PredictorConfig
    from tanitad.models.predictor import (RESIDUAL_HEAD_INIT_SCALE,
                                          OperativePredictor)
    from tanitad.models.flagship_v15 import SPEED_SCALE
    from tanitad.models.sigreg import SigReg
    from tanitad.models.v6 import MaskedCellPredictor
    from tanitad.models.metric_dynamics import rollout_transitions
    # ⚠️ THE REAL IMPLEMENTATIONS, imported from v6's trainer. Reimplementing
    # them here would ablate MY copy of each term, not v6's.
    # ⚠️ ABSOLUTE, not Path.cwd(): this script is run from the scratchpad, so a
    # cwd-relative path silently misses the repo and the o-terms vanish.
    sys.path.insert(0, str(REPO / "stack" / "scripts"))
    from train_v6_staged import (o2_near_field_loss, o3_masked_cell_loss,
                                 o5_rollout_consistency_loss,
                                 rollout_step_weights)
    terms = list(terms or [])
    torch.manual_seed(seed)
    enc = TinyEncoder().to(dev)
    pcfg = PredictorConfig(d_model=256, depth=4, n_heads=4, window=WINDOW,
                           horizons=(1, 2, 4), action_dim=3, residual=True)
    pred = OperativePredictor(pcfg, D_OP).to(dev)
    if arm == "regress":
        # THE DELIBERATE REGRESSION: undo the fix by restoring default init.
        for h in pred.heads.values():
            h.weight.data.div_(RESIDUAL_HEAD_INIT_SCALE)
            h.bias.data.div_(RESIDUAL_HEAD_INIT_SCALE)
    mods = [enc, pred]
    masked = None
    if "o3" in terms:
        # O3 predicts MASKED readout cells from visible context: [B, C, d_r].
        masked = MaskedCellPredictor(16, 128, hidden=128, depth=2).to(dev)
        mods.append(masked)
    n_par = sum(p.numel() for m in mods for p in m.parameters())
    sig = SigReg(n_slices=256)
    params = [q for m in mods for q in m.parameters()]
    rng_o = torch.Generator(device="cpu").manual_seed(seed + 1)
    w_roll = rollout_step_weights(K_ROLL, device=dev)
    c_rng = cell_ranges().to(dev)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    # EPISODE-DISJOINT: last 4 clips are held out, never trained on.
    hold = {c["clip_id"] for c in clips[-4:]}
    tr_c = [c for c in clips if c["clip_id"] not in hold]
    te_c = [c for c in clips if c["clip_id"] in hold]
    span = WINDOW + K_ROLL
    starts = np.concatenate([np.arange(c["start"], c["start"] + c["n"] - span)
                             for c in tr_c if c["n"] > span])
    print(f"    [{arm}] {n_par/1e6:.2f}M · {len(starts):,} train windows · "
          f"{len(te_c)} held-out clips · window {WINDOW} · k {K_ROLL} · "
          f"terms {['core'] + terms}", flush=True)

    def stack9(idx):
        """9-channel input: frames (i-2, i-1, i), clamped at clip starts."""
        out = np.empty((len(idx), 3 * N_STACK, IMG_H, IMG_W), np.uint8)
        for j, i in enumerate(idx):
            out[j] = np.concatenate([frames[max(i - k, 0)]
                                     for k in range(N_STACK - 1, -1, -1)], 0)
        return torch.from_numpy(out).to(dev).float() / 255.0

    g = torch.Generator().manual_seed(seed)
    hist, t0 = [], time.time()
    for step in range(steps):
        idx = starts[torch.randint(0, len(starts), (batch,),
                                   generator=g).numpy()]
        flat = np.concatenate([idx + j for j in range(WINDOW + 1)])
        z = enc(stack9(flat)).reshape(WINDOW + 1, batch, -1).transpose(0, 1)
        a2 = torch.from_numpy(np.stack([acts[idx + j] for j in range(WINDOW)],
                                       1)).to(dev).float()
        v = (torch.from_numpy(spds[idx]).to(dev).float()
             / SPEED_SCALE)[:, None, None].expand(-1, WINDOW, -1)
        aw3 = torch.cat([a2, v], -1)
        zhat = pred(z[:, :WINDOW], aw3)[1]
        l_pred = F.mse_loss(zhat, z[:, WINDOW].detach())
        l_sig = sig(z.reshape(-1, z.shape[-1]))
        loss = l_pred + lam * l_sig
        extra = {}

        if terms:
            # the FACTUAL roll, computed ONCE and shared by O2/O3/O5 -- exactly
            # as train_v6_staged does (`need_roll`).
            fa3 = aw3[:, -1:].expand(-1, K_ROLL, -1)
            trans = rollout_transitions(pred, z[:, :WINDOW], aw3, fa3, K_ROLL)
            zh_steps = [t[1] for t in trans]
            # true future latents: encode the k frames after the window
            ztrue = [z[:, WINDOW]] if K_ROLL == 1 else None
            if ztrue is None:
                fut = np.concatenate([idx + WINDOW + j for j in range(K_ROLL)])
                zf = enc(stack9(fut)).reshape(K_ROLL, batch, -1)
                ztrue = [zf[j].detach() for j in range(K_ROLL)]
            if "o5" in terms:
                l5, g5 = o5_rollout_consistency_loss(zh_steps, ztrue, w_roll)
                loss = loss + O_WEIGHTS["o5"] * l5
                extra["o5"] = round(float(l5.detach()), 5)
            if "o2" in terms:
                pc = zhat.reshape(-1, 16, 128)
                tc = z[:, WINDOW].detach().reshape(-1, 16, 128)
                l2, g2d = o2_near_field_loss(pc, tc, c_rng,
                                             v[:, 0, 0] * SPEED_SCALE)
                loss = loss + O_WEIGHTS["o2"] * l2
                extra["o2"] = round(float(l2.detach()), 5)
            if "o3" in terms:
                ctx = zhat.reshape(-1, 16, 128)
                tc = z[:, WINDOW].detach().reshape(-1, 16, 128)
                m = torch.rand(len(ctx), 16, generator=rng_o) < 0.4
                l3, g3 = o3_masked_cell_loss(masked, ctx, tc, m.to(dev))
                loss = loss + O_WEIGHTS["o3"] * l3
                extra["o3"] = round(float(l3.detach()), 5)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        if step % 250 == 0 or step == steps - 1:
            rec = {"step": step, "pred": float(l_pred.detach()),
                   "sigreg": float(l_sig.detach()), **extra,
                   "s": round(time.time() - t0, 1)}
            hist.append(rec)
            print(f"    [{arm}] {rec}", flush=True)

    # ---- G2 on the HELD-OUT clips -------------------------------------- #
    enc.eval(); pred.eval()
    te_starts = np.concatenate([np.arange(c["start"], c["start"] + c["n"] - span)
                                for c in te_c if c["n"] > span])
    zh, zf, zn = [], [], []
    with torch.no_grad():
        for s in range(0, len(te_starts), batch):
            idx = te_starts[s:s + batch]
            flat = np.concatenate([idx + j for j in range(WINDOW + 1)])
            z = enc(stack9(flat)).reshape(WINDOW + 1, len(idx), -1).transpose(0, 1)
            a2 = torch.from_numpy(np.stack([acts[idx + j]
                                            for j in range(WINDOW)], 1)).to(dev).float()
            v = (torch.from_numpy(spds[idx]).to(dev).float()
                 / SPEED_SCALE)[:, None, None].expand(-1, WINDOW, -1)
            zh.append(pred(z[:, :WINDOW], torch.cat([a2, v], -1))[1].cpu())
            zf.append(z[:, WINDOW].cpu())
            zn.append(z[:, WINDOW - 1].cpu())
    g2 = explained_movement(torch.cat(zh).numpy(), torch.cat(zf).numpy(),
                            torch.cat(zn).numpy())
    verdict = ("PASS (beats hold)" if g2["explained_movement"] > 0.02 else
               "FAIL == hold" if g2["explained_movement"] > -0.02 else
               "FAIL worse than hold")
    print(f"  [{arm}] G2 explained_movement {g2['explained_movement']:+.4f} "
          f"· ratio_vs_hold {g2['ratio_vs_hold']:.3f}x · {verdict}", flush=True)
    return {"arm": arm, "terms": ["core"] + terms,
            "n_params": n_par, "steps": steps,
            "held_out_clips": sorted(hold), "g2": g2, "verdict": verdict,
            "history": hist, "wall_s": round(time.time() - t0)}


def main() -> int:
    ap = argparse.ArgumentParser(description="v7-tiny G2")
    ap.add_argument("--clips", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--arms", nargs="*", default=["fixed", "regress"])
    ap.add_argument("--out", default=str(SP / "v7tiny_g2.json"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames, clips = build_bank(a.clips)
    acts = np.load(BANK / f"act{a.clips}.npy")
    spds = np.load(BANK / f"spd{a.clips}.npy")
    out = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC",
           "gate": "G2: explained_movement > 0 on HELD-OUT clips",
           "design": "V7_TINY_DESIGN.md", "resolution": [IMG_H, IMG_W],
           "n_tok": N_TOK, "window": WINDOW, "k_roll": K_ROLL,
           "n_clips": a.clips, "steps": a.steps, "arms": {}}
    for arm in a.arms:
        out["arms"][arm] = run(arm, frames, clips, acts, spds,
                               steps=a.steps, batch=a.batch, dev=dev,
                               terms=LADDER.get(arm.replace("fixed_", ""), []))
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    if {"fixed", "regress"} <= set(out["arms"]):
        f = out["arms"]["fixed"]["g2"]["explained_movement"]
        r = out["arms"]["regress"]["g2"]["explained_movement"]
        out["gate_can_fail"] = bool(r < f)
        print(f"\n  GATE CAN FAIL (regress {r:+.4f} < fixed {f:+.4f}): "
              f"{out['gate_can_fail']}")
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
