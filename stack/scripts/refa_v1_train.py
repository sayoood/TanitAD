#!/usr/bin/env python3
"""REF-A v1 trainer — feature prediction on cached frozen DINOv3 fields.

⭐ WHAT IS DIFFERENT FROM `refa_train_plus.py`, IN ONE LINE: the loss is the
FUTURE PATCH FIELD, not a trajectory label. Everything else that made REF-A
stable is kept verbatim (feature-cache training, fit-once standardizer, no
BatchNorm/dropout, adapter-vs-predictor LR groups, the adapter-collapse
monitor).

CACHE CONTRACT (stage 1, separate job — this trainer never touches an image):
    <cache>/<episode_id>.pt  ->  fp16 tensor [T, 640, 1024]
      * DINOv3 ViT-L/16 patch tokens, CLS DISCARDED
      * 256x640 crop at 120 deg HFOV (grid 16x40)
    <cache>/index.json       ->  {"episodes": [...], "parity_key": "...",
                                  "skip_hash": "...", "geometry": {...}}
⛔ The trainer REFUSES a cache whose geometry disagrees with
``refa_v1.DINOV3_GEOMETRY`` — a silently narrower interface is exactly the
defect v1 exists to remove, and it would still train.

Run (smoke, CPU, no cache needed):
    python stack/scripts/refa_v1_train.py --smoke
Run (real):
    python stack/scripts/refa_v1_train.py --cache /path/dinov3_w120 \
        --steps 30000 --bs 8 --out ~/experiments/refa-v1
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.refs.refa_v1 import DINOV3_GEOMETRY, RefAV1, RefAV1Config


def build_model(args) -> RefAV1:
    cfg = RefAV1Config(
        strategic_cfg=None if args.no_hierarchy else StrategicPolicyConfig(),
        tactical_cfg=None if args.no_hierarchy else TacticalPolicyConfig(),
    )
    if args.smoke:
        cfg.d_enc, cfg.n_tokens, cfg.d_state = 32, 8, 32
        cfg.op_layers, cfg.op_heads, cfg.tac_layers = 1, 2, 1
        cfg.tac_queries, cfg.str_dim, cfg.str_layers = 4, 16, 1
        if not args.no_hierarchy:
            cfg.strategic_cfg = StrategicPolicyConfig(d_model=32, depth=1,
                                                      n_heads=2, d_ctx=16,
                                                      d_cmd=8)
            cfg.tactical_cfg = TacticalPolicyConfig(d_model=32, depth=1,
                                                    n_heads=2, d_intent=16)
    return RefAV1(cfg)


def verify_cache(cache: Path) -> dict:
    """⛔ Geometry is a CONTRACT, not a hint (see module docstring)."""
    idx = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    geo = idx.get("geometry", {})
    for k in ("n_tokens", "d_enc", "hfov_deg"):
        want, got = DINOV3_GEOMETRY[k], geo.get(k)
        if got != want:
            raise SystemExit(
                f"REFUSING this cache: geometry.{k} = {got!r}, v1 requires "
                f"{want!r}. A narrowed visual interface trains happily and is "
                "the defect v1 was built to remove.")
    return idx


class SmokeData:
    """Random fields with the right shapes — proves the loop, never a number."""

    def __init__(self, cfg: RefAV1Config, bs: int):
        self.cfg, self.bs = cfg, bs

    def batch(self):
        c = self.cfg
        return (torch.randn(self.bs, c.op_window, c.n_tokens, c.d_enc),
                torch.randn(self.bs, c.op_steps, c.a_dim) * 0.1,
                torch.randn(self.bs, c.op_steps, c.n_tokens, c.d_enc))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--out", type=Path, default=Path("./refa_v1_run"))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--adapter-lr-mult", type=float, default=0.1,
                    help="REF-A stability item 4: the adapter warms up SLOWER "
                         "than the predictor (10x longer warmup == 0.1x LR).")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--no-hierarchy", action="store_true",
                    help="ablation arm: drop both brains (they are a matched "
                         "set) — the change-#7 control")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    a = ap.parse_args(argv)

    if not a.smoke and a.cache is None:
        raise SystemExit("--cache is required unless --smoke")
    if a.cache is not None:
        verify_cache(a.cache)

    torch.manual_seed(0)
    model = build_model(a).to(a.device)
    cfg = model.cfg
    a.out.mkdir(parents=True, exist_ok=True)

    # Stability item 4: adapter and predictor are SEPARATE param groups.
    adapter_p = list(model.adapter.parameters())
    ids = {id(p) for p in adapter_p}
    rest_p = [p for p in model.parameters() if id(p) not in ids]
    opt = torch.optim.AdamW(
        [{"params": adapter_p, "lr": a.lr * a.adapter_lr_mult},
         {"params": rest_p, "lr": a.lr}], weight_decay=0.01)

    data = SmokeData(cfg, a.bs) if a.smoke else None
    if data is None:
        raise SystemExit(
            "the real DataLoader binds to the stage-1 cache and is wired in "
            "the launch job; --smoke exercises the full loop offline. This is "
            "deliberate: the trainer must not invent a loader for a cache that "
            "does not exist yet, or the first real run would debug two things "
            "at once.")

    # Fit the standardizer ONCE, before step 0 (stability item 1).
    with torch.no_grad():
        feats, _, _ = data.batch()
        model.std.fit(feats.to(a.device))

    log = (a.out / "train_log.jsonl").open("a", encoding="utf-8")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        feats, actions, future = (x.to(a.device) for x in data.batch())
        out = model(feats, actions, future_feats=future)
        loss = out["loss"] + cfg.w_aux_head * torch.zeros((), device=a.device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % a.log_every == 0 or step == 1:
            with torch.no_grad():
                # THE COLLAPSE MONITOR, carried over from REF-A: adapter output
                # per-dim std. Part 2 measured the trained REF-A adapter at
                # 0.8011 vs 0.220 random-init — i.e. NOT collapsed. If v1 ever
                # drives this toward 0 the run is dead regardless of the loss.
                adapter_std = float(model.encode(feats).std(dim=(0, 1, 2)).mean())
            row = {"step": step, "loss": float(loss),
                   "loss_feat_op": float(out["loss_feat_op"]),
                   "loss_feat_tac": float(out["loss_feat_tac"]),
                   "loss_feat_str": float(out["loss_feat_str"]),
                   "grad_norm": float(gnorm), "adapter_std": adapter_std,
                   "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row))
            if not math.isfinite(row["loss"]):
                raise SystemExit("non-finite loss — refusing to continue")

        if step % a.save_every == 0 or step == a.steps:
            torch.save({"step": step, "model": model.state_dict(),
                        "opt": opt.state_dict(), "cfg": vars(cfg)},
                       a.out / "ckpt.pt")
    log.close()
    print(f"done: {a.steps} steps, {model.trainable_parameters():,} trainable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
