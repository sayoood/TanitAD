"""R3 — what a wider field actually costs: params, FLOPs and MEASURED throughput.

Three questions the v5 geometry decision needs answered in numbers:

  Q1  PARAMS. What fraction of the sub-300M budget is the encoder, and how much
      does widening the field cost in PARAMETERS? (Pre-registered expectation:
      almost nothing — a ViT's parameters are independent of token count except
      for the positional embedding. If that holds, the sub-300M invariant is NOT
      the binding constraint and the real cost is compute + activation memory.)

  Q2  FLOPs. Analytic per-block split between the LINEAR term (12 N d^2) and the
      QUADRATIC attention term (2 N^2 d). The brief asserts "at 640-1600 tokens
      attention becomes the dominant term". That is testable arithmetic and it is
      checked here rather than inherited.

  Q3  MEASURED throughput + memory on the dev box, with a SPILL FILTER.
      Windows/WDDM spills to host RAM instead of OOMing; an unfiltered number is
      fiction (a batch-8 run once reported "11.36 GiB peak" while running 22x
      slower). SPILL TEST: every config is run at batch B and batch B/2. If
      per-sample time at B exceeds 2x per-sample time at B/2, the config SPILLED
      and its throughput is reported as SPILLED, not as a capacity.

Geometries costed (token counts are EXACT, from the config's own token_grid):
    today      256x256 p16 -> 16x16 =  256 tokens,  51.4 deg
    wide-uni   640x256 p16 -> 16x40 =  640 tokens, 100.6 deg at today's px/deg
    square-uni 640x640 p16 -> 40x40 = 1600 tokens,  the naive "keep it square"
    foveated   640x256, 16px centre + 32px wings   ->  352 tokens (see below)

The FOVEATED layout is a fixed, static-shape, STT-style concentric-band scheme
(Schmidt & Newcombe 2506.11131 use 5 rings at stride 1->8; ours is the 1-D
horizontal analogue because driving's acuity gradient is horizontal):
    centre 256 px wide  @ patch 16 -> 16 x 16 = 256 tokens (4.64 px/deg)
    two wings 192 px    @ patch 32 ->  8 x  6 =  48 each   (2.32 px/deg)
                                       total  =  352 tokens
It is implemented here for real (two strided convs + a shared width projection)
so the measurement is of the actual module, not of an N-token stand-in.

NOTHING IN tanitad/ IS MODIFIED. The wide/non-square path is the geometry
sibling's already-landed `EncoderConfig.image_width`; this script only reads it.

Run:  python encoder_shape_cost.py --out ../artifacts/encoder_shape_cost.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *([".."] * 6)))
sys.path.insert(0, os.path.join(REPO, "stack"))

from tanitad.config import EncoderConfig                    # noqa: E402
from tanitad.models.encoder import ViTEncoder               # noqa: E402

# flagship-v1 camera-encoder backbone (DynEncConfig defaults, dynamics_encoder.py:221-229)
D_MODEL, DEPTH, N_HEADS, PATCH, IN_CH = 768, 12, 12, 16, 9
F_REF = 266.0            # canonical effective focal at 256 px (calib.py:38)


def hfov_deg(width_px: int, f_eff: float) -> float:
    return 2.0 * math.degrees(math.atan(0.5 * width_px / f_eff))


class FoveatedPatchEmbed(nn.Module):
    """Static two-scale horizontal foveation -> ONE token sequence [B, N, D].

    Centre band at `patch_c` (full acuity), left+right wings at `patch_w`
    (coarse). Token count is FIXED and content-independent => TensorRT-safe.
    Separate positional embedding per band, so the ViT can still read geometry.

    TWO WING VARIANTS, and the difference is a real params-budget decision:
      `sep`    a second Conv2d at kernel `patch_w`. Costs
               in_ch * patch_w^2 * d_model extra params -- at 32px that is
               9*1024*768 = 7.08M, i.e. +8.1% on the encoder. MEASURED.
      `shared` (DEFAULT) avg-pool the wings by patch_w/patch_c, then apply the
               SAME centre conv. Identical token geometry, **ZERO extra
               parameters**, and the wing tokens land in the centre's embedding
               space rather than a separate one.
    """

    def __init__(self, in_ch: int, d_model: int, height: int, width: int,
                 centre_w: int, patch_c: int = 16, patch_w: int = 32,
                 wing: str = "shared"):
        super().__init__()
        assert (width - centre_w) % 2 == 0, "centre must be symmetric in the frame"
        assert patch_w % patch_c == 0, "wing patch must be a multiple of centre patch"
        self.height, self.width, self.centre_w = height, width, centre_w
        self.wing_w = (width - centre_w) // 2
        self.patch_c, self.patch_w, self.wing = patch_c, patch_w, wing
        self.proj_c = nn.Conv2d(in_ch, d_model, patch_c, patch_c)
        if wing == "sep":
            self.proj_w = nn.Conv2d(in_ch, d_model, patch_w, patch_w)
            self.pool = None
        else:                                    # zero-param shared-weight wings
            self.proj_w = None
            self.pool = nn.AvgPool2d(patch_w // patch_c)
        self.n_c = (height // patch_c) * (centre_w // patch_c)
        self.n_w = (height // patch_w) * (self.wing_w // patch_w)
        self.n_tokens = self.n_c + 2 * self.n_w
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def _wing(self, y):
        return self.proj_w(y) if self.wing == "sep" else self.proj_c(self.pool(y))

    def forward(self, x):
        L = x[..., :self.wing_w]
        C = x[..., self.wing_w:self.wing_w + self.centre_w]
        R = x[..., self.wing_w + self.centre_w:]
        t = [self._wing(L), self.proj_c(C), self._wing(R)]
        t = [y.flatten(2).transpose(1, 2) for y in t]
        return torch.cat(t, dim=1) + self.pos


class FoveatedViT(nn.Module):
    """The real encoder blocks, fed by the foveated embedder."""

    def __init__(self, embed: FoveatedPatchEmbed, d_model, depth, n_heads):
        super().__init__()
        from tanitad.models.encoder import Block
        self.embed = embed
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(depth))
        self.norm = nn.LayerNorm(d_model)
        self.n_tokens = embed.n_tokens

    def forward(self, x):
        t = self.embed(x)
        for b in self.blocks:
            t = b(t)
        return self.norm(t)


def flops_per_image(n_tokens: int, d=D_MODEL, depth=DEPTH, mlp_ratio=4.0):
    """MACs for the transformer blocks. qkv+proj = 4Nd^2, mlp = 2*mlp_ratio*Nd^2,
    attention scores+apply = 2N^2 d.  (Patch embed excluded; it is <1%.)"""
    lin = depth * (4 * n_tokens * d * d + 2 * mlp_ratio * n_tokens * d * d)
    quad = depth * (2 * n_tokens * n_tokens * d)
    return {"linear_macs": lin, "attn_quad_macs": quad, "total_macs": lin + quad,
            "attn_share": quad / (lin + quad)}


def count_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def _noop():
    pass


def bench(model, x, iters=12, warmup=4, train=False):
    """Returns (min_ms, median_ms, peak_alloc_MiB, peak_reserved_MiB).

    ⚠️ The dev box is SHARED and was measured at 100 % foreign GPU utilisation
    during this run. Under contention the MEDIAN is biased upward by an unknown,
    time-varying amount, but the MINIMUM over repeats is the closest available
    estimate of the uncontended cost -- contention can only ever ADD time. The
    headline ratios therefore use `min`; `median` is kept so the spread is
    visible rather than hidden."""
    dev = x.device
    model.train(train)
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)

    def one():
        if train:
            y = model(x)
            loss = y.float().pow(2).mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                model(x)

    for _ in range(warmup):
        one()
    torch.cuda.synchronize(dev)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        one()
        torch.cuda.synchronize(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    return (min(ts), statistics.median(ts),
            torch.cuda.max_memory_allocated(dev) / 2**20,
            torch.cuda.max_memory_reserved(dev) / 2**20)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "artifacts",
                                                  "encoder_shape_cost.json"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    out = {"device": str(dev), "batch": args.batch,
           "backbone": {"d_model": D_MODEL, "depth": DEPTH, "n_heads": N_HEADS,
                        "patch": PATCH, "in_channels": IN_CH},
           "spill_filter": ("per-sample time at batch B vs batch B/2; ratio > 2.0 "
                            "at constant per-sample work == SPILLED, not a capacity"),
           "geometries": {}}
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        out["gpu"] = {"name": p.name, "total_GiB": round(total / 2**30, 2),
                      "free_at_start_GiB": round(free / 2**30, 2)}

    # ---- Q1/Q2: params + analytic FLOPs for every candidate geometry ----
    GEOMS = [
        ("today_256x256", 256, 256, None, None),
        ("wide_640x256", 256, 640, None, None),
        ("square_640x640", 640, 640, None, None),
        ("foveated_640x256_shared", 256, 640, 256, "shared"),
        ("foveated_640x256_sep", 256, 640, 256, "sep"),
    ]
    models = {}
    for name, h, w, centre, wing in GEOMS:
        if centre is None:
            cfg = EncoderConfig(in_channels=IN_CH, image_size=h,
                                image_width=(None if w == h else w),
                                patch_size=PATCH, d_model=D_MODEL,
                                depth=DEPTH, n_heads=N_HEADS)
            m = ViTEncoder(cfg)
            n_tok = m.n_tokens
            grid = list(cfg.token_grid())
            px_per_deg_note = "uniform"
        else:
            emb = FoveatedPatchEmbed(IN_CH, D_MODEL, h, w, centre, 16, 32, wing)
            m = FoveatedViT(emb, D_MODEL, DEPTH, N_HEADS)
            n_tok = m.n_tokens
            grid = [f"centre {emb.height//emb.patch_c}x{centre//emb.patch_c}",
                    f"2 wings {emb.height//emb.patch_w}x{emb.wing_w//emb.patch_w}"]
            px_per_deg_note = f"two-scale (16px centre / 32px wings, wing={wing})"
        models[name] = (m, h, w)
        fl = flops_per_image(n_tok)
        base = flops_per_image(256)["total_macs"]
        # angular geometry: keep today's 4.64 px/deg -> f_eff scales with width
        f_keep = F_REF                       # px/deg preserved
        out["geometries"][name] = {
            "input_hw": [h, w], "token_grid": grid, "n_tokens": n_tok,
            "tokens_vs_today": round(n_tok / 256, 3),
            "params_total": count_params(m),
            "params_pos_embed": (m.pos.numel() if hasattr(m, "pos")
                                 else m.embed.pos.numel()),
            "hfov_deg_at_F_REF": round(hfov_deg(w, f_keep), 2),
            "px_per_deg_centre": round(math.pi * f_keep / 180.0, 3),
            "sampling": px_per_deg_note,
            "flops": {k: (round(v, 4) if k == "attn_share" else int(v))
                      for k, v in fl.items()},
            "flops_vs_today": round(fl["total_macs"] / base, 3),
        }

    # ---- the brief's O(N^2) assertion, checked ----
    out["attention_share_check"] = {
        "claim_in_brief": "at 640-1600 tokens attention becomes the dominant term",
        "formula": "attn_share = 2N^2 d / (12 N d^2 + 2 N^2 d) = N / (6d + N), d=768",
        "measured_analytically": {str(n): round(flops_per_image(n)["attn_share"], 4)
                                  for n in (256, 352, 640, 1600, 4608)},
        "N_at_which_attention_is_50pct": 6 * D_MODEL,
    }

    # ---- Q3: measured throughput with the spill filter + contention sentinel ----
    if dev.type == "cuda":
        def sentinel():
            """Reference workload run before AND after the sweep. If its cost
            drifts, the box was contended and the absolute timings are not
            quotable. This is the C13 guard on our own instrument."""
            mref, hh, ww = models["today_256x256"]
            mref = mref.to(dev)
            x = torch.randn(2, IN_CH, hh, ww, device=dev)
            mn, _md, _a, _r = bench(mref, x, 10, 3, False)
            del x
            mref.to("cpu")
            torch.cuda.empty_cache()
            return mn

        s0 = sentinel()
        for name, (m, h, w) in models.items():
            m = m.to(dev)
            rec = out["geometries"][name]
            for mode, train in (("infer", False), ("train_fwd_bwd", True)):
                res = {}
                for B in (args.batch, max(1, args.batch // 2)):
                    x = torch.randn(B, IN_CH, h, w, device=dev)
                    try:
                        mn, md, alloc, resv = bench(m, x, args.iters, 4, train)
                        res[B] = {"ms_min": round(mn, 3), "ms_median": round(md, 3),
                                  "ms_per_sample": round(mn / B, 3),
                                  "median_over_min": round(md / mn, 3),
                                  "peak_alloc_MiB": round(alloc, 1),
                                  "peak_reserved_MiB": round(resv, 1)}
                    except torch.cuda.OutOfMemoryError:
                        res[B] = {"OOM": True}
                    del x
                    torch.cuda.empty_cache()
                full, half = res.get(args.batch), res.get(max(1, args.batch // 2))
                spill = None
                if full and half and "ms_per_sample" in full and "ms_per_sample" in half:
                    ratio = full["ms_per_sample"] / half["ms_per_sample"]
                    spill = {"per_sample_ratio_B_over_Bhalf": round(ratio, 3),
                             "SPILLED": bool(ratio > 2.0),
                             "note": ("computed on min-times; under foreign GPU load "
                                      "this test itself is noisy -- read with the "
                                      "contention sentinel")}
                rec[mode] = {"by_batch": {str(k): v for k, v in res.items()},
                             "spill_filter": spill}
            m.to("cpu")
            torch.cuda.empty_cache()
        s1 = sentinel()
        drift = max(s0, s1) / max(min(s0, s1), 1e-9)
        out["contention_sentinel"] = {
            "reference_ms_before": round(s0, 3), "reference_ms_after": round(s1, 3),
            "drift_ratio": round(drift, 3),
            "CONTENDED": bool(drift > 1.2),
            "verdict": ("absolute latencies NOT quotable; params/tokens/FLOPs are "
                        "unaffected (deterministic)" if drift > 1.2 else
                        "reference stable within 20%"),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # ------------------------------------------------------------- console
    print(f"device={dev}  batch={args.batch}")
    print(f"\n{'geometry':26s} {'tok':>5s} {'xTok':>6s} {'params':>12s} "
          f"{'pos':>8s} {'xFLOP':>7s} {'attn%':>6s}")
    for k, v in out["geometries"].items():
        print(f"{k:26s} {v['n_tokens']:5d} {v['tokens_vs_today']:6.2f} "
              f"{v['params_total']:12,d} {v['params_pos_embed']:8,d} "
              f"{v['flops_vs_today']:7.2f} {100*v['flops']['attn_share']:5.1f}%")
    print("\nattention share of encoder FLOPs (analytic):")
    for n, s in out["attention_share_check"]["measured_analytically"].items():
        print(f"   N={n:>5s}  {100*s:5.1f}%")
    if dev.type == "cuda":
        print(f"\n{'geometry':26s} {'mode':14s} {'ms/sample':>10s} "
              f"{'peakMiB':>9s} {'spill?':>7s}")
        for k, v in out["geometries"].items():
            for mode in ("infer", "train_fwd_bwd"):
                r = v.get(mode, {})
                b = r.get("by_batch", {}).get(str(args.batch), {})
                sp = r.get("spill_filter") or {}
                if "ms_per_sample" in b:
                    print(f"{k:26s} {mode:14s} {b['ms_per_sample']:10.2f} "
                          f"{b['peak_alloc_MiB']:9.1f} "
                          f"{'SPILL' if sp.get('SPILLED') else 'ok':>7s}")
                else:
                    print(f"{k:26s} {mode:14s} {'OOM':>10s}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
