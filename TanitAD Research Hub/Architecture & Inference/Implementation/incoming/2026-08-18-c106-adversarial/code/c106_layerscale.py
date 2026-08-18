"""⛔ ATTACK 3 — VERIFY C106's MECHANISM FROM SOURCE AND FROM THE WEIGHTS.

C106's interpretation rests on one claim: `ViT5Encoder` uses LayerScale at
`ls_init = 1e-5`, so a RANDOM-INIT copy is "approximately `RMSNorm(patch_conv(x)
+ pos)` — a fixed random LINEAR map of raw patch pixels". That interpretation is
what gives the finding its force ("raw pixels beat our trained tokens").

⛔ BUT THE INTERPRETATION ONLY HOLDS IF THE **TRAINED** ARM'S LayerScale HAS
MOVED. If training left `ls` at ~1e-5, then BOTH arms are near-linear maps of
raw pixels and the comparison is not "random linear map vs trained deep net" —
it is "one near-linear map vs another", which is a different and much smaller
claim. C106 never checked.

WHAT THIS MEASURES, on every locally banked checkpoint plus a random init:
  1. `ls1`/`ls2` magnitude per block — did the trained values leave 1e-5?
  2. ⭐ THE DIRECT TEST, not the proxy: the RESIDUAL FRACTION. Run the encoder
     on real banked frames twice — once intact, once with EVERY `ls1`/`ls2`
     forced to ZERO (which reduces the network to exactly
     `RMSNorm(patch(x) + pos)`, registers stripped) — and report
     ||full − linear|| / ||linear|| and cos(full, linear). If the trained arm's
     tokens are still ~the linear map, C106's mechanism story is wrong.
  3. Token scale/rank diagnostics, so "the trained tokens are degenerate" is
     measured rather than inferred from a flat readout.

⛔ PARITY: SELECTS NOTHING — reads a banked window cache's own frames.
TIER: T0-DIAGNOSTIC. Device memory on this box is read with
`torch.cuda.max_memory_allocated()` only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[6]
for _p in (_REPO / "stack",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def build(ckpt: str, seed: int, trained: bool, device):
    from tanitad.config import EncoderConfig
    from tanitad.models.encoder import ViT5Encoder
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    vc = ck["_meta"]["config"]["v6_config"]
    enc_cfg = EncoderConfig(**vc["encoder"])
    torch.manual_seed(int(seed))
    enc = ViT5Encoder(enc_cfg, n_registers=int(vc.get("n_registers", 4)))
    step = int(ck["_meta"].get("step", -1))
    n_keys = None
    if trained:
        sd = {k[len("encoder."):]: v for k, v in ck["model"].items()
              if k.startswith("encoder.")}
        missing, unexpected = enc.load_state_dict(sd, strict=False)
        fatal = [k for k in missing if not k.startswith("rope_")]
        if fatal or unexpected:
            raise SystemExit(f"⛔ load mismatch missing={fatal[:5]} "
                             f"unexpected={sorted(unexpected)[:5]}")
        n_keys = len(sd)
    del ck
    return enc.to(device).eval(), step, n_keys


@torch.no_grad()
def probe(enc, x):
    """(full tokens, linear-only tokens) — linear-only = every LayerScale 0."""
    full = enc(x).float()
    saved = []
    for b in enc.blocks:
        saved.append((b.ls1.data.clone(), b.ls2.data.clone()))
        b.ls1.data.zero_()
        b.ls2.data.zero_()
    lin = enc(x).float()
    for b, (a1, a2) in zip(enc.blocks, saved):
        b.ls1.data.copy_(a1)
        b.ls2.data.copy_(a2)
    return full, lin


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--row-index", required=True)
    ap.add_argument("--episodes-dir", required=True)
    ap.add_argument("--rand-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-rows", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from tanitad.data.v2_dataset import _decode_stacked, _jpeg_offsets
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    idx = torch.load(a.row_index, map_location="cpu", weights_only=False)
    rows = idx["rows"]
    sel = rows[::max(1, len(rows) // int(a.n_rows))][:int(a.n_rows)]
    eps = Path(a.episodes_dir)
    frames = []
    cache: dict = {}
    for r in sel:
        cid = r["clip_id"]
        if cid not in cache:
            if len(cache) >= 4:
                cache.pop(next(iter(cache)))
            d = torch.load(eps / f"{cid}.v2ep.pt", map_location="cpu",
                           weights_only=False)
            cache[cid] = (d["jpeg_buf"], _jpeg_offsets(d["jpeg_len"]),
                          int(d["n_stack"]), str(d.get("codec", "jpeg")))
        buf, offs, n_stack, codec = cache[cid]
        pf = int(r["frame_idx"])
        frames.append(_decode_stacked(buf, offs, n_stack, pf, pf + 1, codec,
                                      None)[0])
    X = (torch.stack(frames).float() / 255.0).to(dev)
    del cache, frames
    print(f"[ls] {X.shape[0]} real banked frames {tuple(X.shape[1:])}",
          flush=True)

    out = {"_evidence_class": "MEASURED (ours; LayerScale + residual-fraction on "
                              "REAL banked frames through frozen encoders)",
           "eval_tier": "T0-DIAGNOSTIC",
           "source_of_ls_init": "stack/tanitad/models/encoder.py Block5 "
                                "ls_init=1e-5 (:302-312), residual x + ls*f(x) "
                                "(:315-316); ViT5Encoder default ls_init=1e-5 "
                                "(:331)",
           "n_frames": int(X.shape[0]), "arms": {}}

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    plan = ([("random_init_s%d" % s, a.ckpts[0], s, False) for s in a.rand_seeds]
            + [("trained_%s" % Path(c).stem, c, 0, True) for c in a.ckpts])
    for name, ck, seed, trained in plan:
        enc, step, n_keys = build(ck, seed, trained, dev)
        ls1 = torch.stack([b.ls1.data.abs().mean() for b in enc.blocks]).cpu()
        ls2 = torch.stack([b.ls2.data.abs().mean() for b in enc.blocks]).cpu()
        with torch.no_grad():
            full, lin = probe(enc, X)
        num = (full - lin).norm().item()
        den = lin.norm().item()
        fv, lv = full.reshape(-1), lin.reshape(-1)
        cos = float((fv @ lv / (fv.norm() * lv.norm())).item())
        # token-level conditioning: how many directions actually carry variance
        T = full.reshape(-1, full.shape[-1]).double()
        T = T - T.mean(0, keepdim=True)
        sv = torch.linalg.svdvals(T.cpu())
        p = (sv ** 2) / (sv ** 2).sum()
        eff_rank = float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()))
        out["arms"][name] = {
            "ckpt": Path(ck).name, "step_in_ckpt": step,
            "init": "TRAINED" if trained else "RANDOM",
            "n_encoder_keys_loaded": n_keys, "init_seed": None if trained else seed,
            "ls1_abs_mean_per_block": [round(float(v), 8) for v in ls1],
            "ls2_abs_mean_per_block": [round(float(v), 8) for v in ls2],
            "ls_abs_mean_all": round(float(torch.cat([ls1, ls2]).mean()), 8),
            "ls_abs_max_all": round(float(torch.cat([ls1, ls2]).max()), 8),
            "ls_over_init_1e-5_mean": round(
                float(torch.cat([ls1, ls2]).mean()) / 1e-5, 2),
            # ⭐ THE DIRECT TEST of C106's "approximately a random linear map"
            "residual_frac_vs_linear": round(num / max(den, 1e-30), 6),
            "cos_full_vs_linear": round(cos, 6),
            "token_sd": round(float(full.std()), 6),
            "token_abs_mean": round(float(full.abs().mean()), 6),
            "effective_rank_768": round(eff_rank, 2)}
        r = out["arms"][name]
        print("  %-26s ls_mean=%.3e (%.1fx init)  resid/lin=%.4f  cos=%.4f  "
              "eff_rank=%.1f  tok_sd=%.4f"
              % (name, r["ls_abs_mean_all"], r["ls_over_init_1e-5_mean"],
                 r["residual_frac_vs_linear"], r["cos_full_vs_linear"],
                 r["effective_rank_768"], r["token_sd"]), flush=True)
        del enc, full, lin, T
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    out["cuda_max_mem_gb"] = (round(float(torch.cuda.max_memory_allocated()) / 1e9, 3)
                              if dev.type == "cuda" else None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), "utf-8")
    print(f"[ls] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
