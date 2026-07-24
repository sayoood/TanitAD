"""RefcCL encoder-integrity CANARY (plan-free, label-free) — the WM-degradation gate.

Unfreezing the encoder is the v4 hazard: a fine-tune can silently pull the encoder
off the manifold the rest of the model (WM aux / measurement / decoder) was fit to.
This canary measures, on HELD-OUT windows, how far the RefcCL encoder's readout has
drifted from the FROZEN base encoder, WITHOUT touching the planner:

  feat_cos     mean cosine(pooled_ft, pooled_base)   -> readout alignment (1 = intact)
  rel_l2       mean ||pooled_ft - pooled_base|| / ||pooled_base||
  man_agree    argmax match of the BASE maneuver-head on ft-vs-base pooled features
  route_agree  argmax match of the BASE route-head on ft-vs-base pooled features

man/route agreement probe whether the encoder still supports the aux tasks it was
trained for (a label-free proxy for "the representation the WM depends on is intact").

Pre-registered gate (SPEED_TERM_PREREG successor / RefcCL pre-reg):
  HOLDS    feat_cos >= 0.90 AND man_agree >= 0.80 AND route_agree >= 0.80
  DEGRADED feat_cos <  0.85 OR  man_agree <  0.70 OR route_agree <  0.70  -> ABORT unfreeze
  (between = WATCH)

Run (eval pod): PYTHONPATH=...:/root/taniteval python3 encoder_canary.py \
  --base-ckpt /root/models/refc-base-30k/ckpt.pt --ft-ckpt <refccl>/ckpt.pt \
  --val-dir /root/valdata/physicalai-val-0c5f7dac3b11 --slice 28:40 --out canary.json
CPU smoke: python3 encoder_canary.py --smoke --out /tmp/canary_smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
_ROOTS = ["/root/TanitAD/stack", "/root/TanitAD/stack/scripts",
          "/workspace/TanitAD/stack", "/workspace/TanitAD/stack/scripts"]
for _up in (5, 6):
    try:
        _ROOTS.append(str(_HERE.parents[_up] / "stack"))
        _ROOTS.append(str(_HERE.parents[_up] / "stack" / "scripts"))
    except IndexError:
        pass
for _p in _ROOTS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

HOLD = {"feat_cos": 0.90, "man_agree": 0.80, "route_agree": 0.80}
DEGRADE = {"feat_cos": 0.85, "man_agree": 0.70, "route_agree": 0.70}


def _load(ckpt, preset, device):
    import dataclasses
    from tanitad.refs.refc import (RefCModel, refc_config, refc_small_config,
                                   refc_smoke_config, refc_xl_config)
    P = {"base": refc_config, "small": refc_small_config, "xl": refc_xl_config,
         "smoke": refc_smoke_config}
    cfg = P[preset]()
    cj = Path(ckpt).parent / "config.json"
    if cj.exists():
        d = json.loads(cj.read_text()).get("cfg", {})
        _apply(cfg, d)
    m = RefCModel(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(ck["model"])
    return m.to(device).eval()


def _apply(cfg, d):
    for k, v in d.items():
        if not hasattr(cfg, k):
            continue
        cur = getattr(cfg, k)
        if isinstance(v, dict) and hasattr(cur, "__dataclass_fields__"):
            _apply(cur, v)
        elif isinstance(cur, tuple) and isinstance(v, list):
            setattr(cfg, k, tuple(v))
        else:
            setattr(cfg, k, v)


@torch.no_grad()
def canary(base, ft, frames_list, poses_list, device, W=8, stride=8, batch=16):
    """base = frozen REF-C (also supplies the aux heads); ft = RefcCL model."""
    cos_s, l2_s, man_s, route_s, n = 0.0, 0.0, 0.0, 0.0, 0
    for fr, poses in zip(frames_list, poses_list):
        T = poses.shape[0]
        starts = list(range(0, T - W, stride))
        for bi in range(0, len(starts), batch):
            ch = starts[bi:bi + batch]
            last = torch.stack([fr[t0 + W - 1] for t0 in ch]).to(device)  # [b,C,H,W]
            last = last.float().div(255.0) if last.dtype == torch.uint8 else last.float()
            _, pb = base.encoder(last)
            _, pf = ft.encoder(last)
            cos = torch.nn.functional.cosine_similarity(pf, pb, dim=-1)     # [b]
            l2 = (pf - pb).norm(dim=-1) / pb.norm(dim=-1).clamp_min(1e-6)
            mb = base.maneuver_head(pb).argmax(-1); mf = base.maneuver_head(pf).argmax(-1)
            rb = base.route_head(pb).argmax(-1); rf = base.route_head(pf).argmax(-1)
            k = last.shape[0]
            cos_s += float(cos.sum()); l2_s += float(l2.sum())
            man_s += float((mb == mf).sum()); route_s += float((rb == rf).sum()); n += k
    r = {"feat_cos": round(cos_s / n, 4), "rel_l2": round(l2_s / n, 4),
         "man_agree": round(man_s / n, 4), "route_agree": round(route_s / n, 4),
         "n_windows": n}
    holds = (r["feat_cos"] >= HOLD["feat_cos"] and r["man_agree"] >= HOLD["man_agree"]
             and r["route_agree"] >= HOLD["route_agree"])
    degraded = (r["feat_cos"] < DEGRADE["feat_cos"]
                or r["man_agree"] < DEGRADE["man_agree"]
                or r["route_agree"] < DEGRADE["route_agree"])
    r["gate"] = "HOLDS" if holds else ("DEGRADED" if degraded else "WATCH")
    r["thresholds"] = {"HOLD": HOLD, "DEGRADE": DEGRADE}
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="/root/models/refc-base-30k/ckpt.pt")
    ap.add_argument("--ft-ckpt", default="")
    ap.add_argument("--preset", default="base")
    ap.add_argument("--val-dir", default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--slice", default="28:40")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        from tanitad.refs.refc import RefCModel, refc_smoke_config
        cfg = refc_smoke_config()
        base = RefCModel(cfg).to(device).eval()
        ft = RefCModel(cfg).to(device).eval()
        ft.load_state_dict(base.state_dict())            # identical -> cos ~1
        with torch.no_grad():                            # perturb ft encoder a hair
            for p in list(ft.encoder.stages)[-1].parameters():
                p.add_(torch.randn_like(p) * 0.01)
        fl = [torch.rand(40, cfg.encoder.in_channels, cfg.encoder.image_size,
                         cfg.encoder.image_size) for _ in range(2)]
        pl = [torch.randn(40, 4) for _ in range(2)]
        r = canary(base, ft, fl, pl, device, W=cfg.window)
        print(json.dumps(r, indent=2))
        Path(args.out).write_text(json.dumps({"smoke": True, **r}, indent=2))
        print("ENCODER_CANARY_SMOKE_OK")
        return

    from tanitad.data.mixing import load_episode
    eps = sorted(Path(args.val_dir).glob("ep_*.pt"))
    a, b = (int(x) if x else None for x in args.slice.split(":"))
    eps = eps[a:b]
    E = [load_episode(str(p), mmap=True) for p in eps]
    fl = [e.frames for e in E]; pl = [e.poses.float() for e in E]
    base = _load(args.base_ckpt, args.preset, device)
    ft = _load(args.ft_ckpt, args.preset, device)
    r = canary(base, ft, fl, pl, device)
    r["base_ckpt"] = args.base_ckpt; r["ft_ckpt"] = args.ft_ckpt
    r["evidence_class"] = "MEASURED (this canary run)"
    Path(args.out).write_text(json.dumps(r, indent=2))
    print(f"[canary] feat_cos={r['feat_cos']} rel_l2={r['rel_l2']} "
          f"man_agree={r['man_agree']} route_agree={r['route_agree']} "
          f"-> GATE={r['gate']}", flush=True)
    print(f"[canary] wrote {args.out}")
    print("ENCODER_CANARY_DONE", flush=True)


if __name__ == "__main__":
    main()
