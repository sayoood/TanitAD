"""⭐ THE MECHANISM C106 DID NOT LOOK FOR — is the trained token field
RANK-COLLAPSED, and does that (not "subtraction") explain the readout gap?

C106 concludes *"the objective is SUBTRACTING geometry"*. There is a second
reading of the same numbers that C106 never tested and that implies a completely
different remedy: the geometry may still be present but pushed into
LOW-VARIANCE directions, so a ridge on 2048 random projections is starved of it.
"Subtracted" and "compressed into a tiny subspace" are not the same claim.

⛔ THE DISCRIMINATING QUANTITY IS THE RANK OF THE MATRIX THE RIDGE ACTUALLY
SEES. This measures, through er10's OWN `pool_tokens` / `make_projection` (the
DEPLOYED AvgPool2d((4,10)) and the ladder's fixed Gaussian RP), on the SAME
banked windows:

  1. token-channel effective rank (768-dim) — the encoder's own output space
  2. pooled-feature effective rank (12 288-dim) — what survives the pool
  3. ⭐ z-scored DESIGN-MATRIX effective rank (2 048-dim) — the ridge's actual
     input, standardised exactly as `er10_pool_ladder` standardises it
  4. the final RMSNorm gain's anisotropy — the cheapest candidate cause

Effective rank = exp(Shannon entropy of the normalised squared singular-value
spectrum) — the participation ratio in nats, so it is comparable across arms
with different scales and needs no threshold.

⛔ PARITY: SELECTS NOTHING. TIER: T0-DIAGNOSTIC.
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
_INC = _REPO / "TanitAD Research Hub/Architecture & Inference/Implementation/incoming"
for _p in (_REPO / "taniteval", _REPO / "stack",
           _INC / "2026-08-17-probe-positive-control/code",
           _INC / "2026-08-17-slot-probe-parity/code",
           _INC / "2026-08-17-latent-linear-ladder/code",
           _INC / "2026-08-18-pooling-ladder-ER10/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import er10_pool_ladder as ER                                    # noqa: E402


def eff_rank(M: np.ndarray) -> tuple[float, float]:
    """(effective rank, fraction of variance in the top direction)."""
    M = M - M.mean(0, keepdims=True)
    sv = np.linalg.svd(M, compute_uv=False)
    p = (sv ** 2) / max((sv ** 2).sum(), 1e-300)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum())), float(p.max())


def build_enc(ckpt: str, seed: int, trained: bool, device):
    from tanitad.config import EncoderConfig
    from tanitad.models.encoder import ViT5Encoder
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    vc = ck["_meta"]["config"]["v6_config"]
    enc = None
    torch.manual_seed(int(seed))
    enc = ViT5Encoder(EncoderConfig(**vc["encoder"]),
                      n_registers=int(vc.get("n_registers", 4)))
    step = int(ck["_meta"].get("step", -1))
    gain = None
    if trained:
        sd = {k[len("encoder."):]: v for k, v in ck["model"].items()
              if k.startswith("encoder.")}
        missing, unexpected = enc.load_state_dict(sd, strict=False)
        fatal = [k for k in missing if not k.startswith("rope_")]
        if fatal or unexpected:
            raise SystemExit(f"⛔ load mismatch {fatal[:5]} {sorted(unexpected)[:5]}")
    gain = enc.norm.weight.detach().float().clone()
    del ck
    return enc.to(device).eval(), step, gain


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--row-index", required=True)
    ap.add_argument("--episodes-dir", required=True)
    ap.add_argument("--arms", nargs="+",
                    default=["random_s0", "random_s1", "random_s2", "trained"])
    ap.add_argument("--n-rows", type=int, default=384)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--proj-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from tanitad.data.v2_dataset import _decode_stacked, _jpeg_offsets
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    idx = torch.load(a.row_index, map_location="cpu", weights_only=False)
    rows = idx["rows"]
    sel = rows[::max(1, len(rows) // int(a.n_rows))][:int(a.n_rows)]
    sel = sorted(sel, key=lambda r: (r["clip_id"], r["frame_idx"]))
    eps, cache = Path(a.episodes_dir), {}
    frames = []
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
    X = torch.stack(frames).float() / 255.0
    del cache, frames
    n, th, tw = X.shape[0], 16, 40
    print(f"[rank] {n} real banked frames {tuple(X.shape[1:])}", flush=True)

    out = {"_evidence_class": "MEASURED (ours; effective rank of the DEPLOYED "
                              "readout path on REAL banked frames)",
           "eval_tier": "T0-DIAGNOSTIC",
           "definition": "effective rank = exp(Shannon entropy of the "
                         "normalised squared singular-value spectrum) of the "
                         "COLUMN-CENTRED matrix; no threshold, comparable "
                         "across arms",
           "pool": "er10_pool_ladder.pool_tokens with POOL_ARMS['p40'] = "
                   "AvgPool2d((4,10)) — the DEPLOYED operator, imported",
           "projection": "er10_pool_ladder.make_projection (the ladder's own "
                         "fixed Gaussian RP, seed base 20260818)",
           "n_frames": n, "token_grid": [th, tw], "arms": {}}

    P = None
    for arm in a.arms:
        trained = arm == "trained"
        seed = 0 if trained else int(arm.split("_s")[-1])
        enc, step, gain = build_enc(a.ckpt, seed, trained, dev)
        toks = []
        with torch.no_grad():
            for s in range(0, n, int(a.batch)):
                xb = X[s:s + int(a.batch)].to(dev)
                with torch.autocast(dev.type, dtype=torch.float16,
                                    enabled=(dev.type == "cuda")):
                    hs = enc(xb)
                toks.append(hs.float().to(torch.float16).cpu())
        T = torch.cat(toks)                                  # [n, 640, 768]
        del toks, enc
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        d_model = T.shape[-1]
        if P is None:
            P = ER.make_projection((th // 4) * (tw // 10), d_model,
                                   [ER.PROJ_SEED_BASE, a.proj_seed,
                                    sorted(ER.POOL_ARMS).index("p40")], dev)
        pooled = []
        for s in range(0, n, 32):
            tk = T[s:s + 32].to(dev).half()
            pooled.append(ER.pool_tokens(tk, ER.POOL_ARMS["p40"], th, tw)
                          .float().cpu())
        Pool = torch.cat(pooled).double().numpy()            # [n, 12288]
        Proj = np.concatenate(
            [(torch.from_numpy(Pool[s:s + 64]).to(dev).half() @ P)
             .float().cpu().numpy().astype(np.float64) for s in range(0, n, 64)])
        mu, sd = Proj.mean(0), Proj.std(0)
        Z = (Proj - mu) / np.where(sd < 1e-12, 1.0, sd)

        # token-channel space: pool (frame, token) pairs, 768 columns
        Tc = T.reshape(-1, d_model)[::7].double().numpy()
        er_tok, top_tok = eff_rank(Tc)
        er_pool, top_pool = eff_rank(Pool)
        er_proj, top_proj = eff_rank(Z)
        g = gain.abs().numpy()
        out["arms"][arm] = {
            "init": "TRAINED" if trained else "RANDOM", "init_seed":
                None if trained else seed, "step_in_ckpt": step,
            "token_channel_eff_rank_of_768": round(er_tok, 3),
            "token_channel_top_dir_var_frac": round(top_tok, 5),
            "pooled_feature_eff_rank_of_%d" % min(n, Pool.shape[1]):
                round(er_pool, 3),
            "pooled_top_dir_var_frac": round(top_pool, 5),
            "zscored_design_matrix_eff_rank_of_%d" % min(n, Z.shape[1]):
                round(er_proj, 3),
            "design_top_dir_var_frac": round(top_proj, 5),
            "final_rmsnorm_gain_abs_max": round(float(g.max()), 4),
            "final_rmsnorm_gain_abs_mean": round(float(g.mean()), 4),
            "final_rmsnorm_gain_max_over_mean": round(
                float(g.max() / max(g.mean(), 1e-12)), 2),
            "final_rmsnorm_gain_eff_rank_proxy_top1_frac": round(
                float((g ** 2).max() / max((g ** 2).sum(), 1e-30)), 5)}
        r = out["arms"][arm]
        print("  %-12s tok_rank=%7.2f (top %.3f)  pooled_rank=%7.2f  "
              "design_rank=%7.2f  gain_max/mean=%.2f"
              % (arm, er_tok, top_tok, er_pool, er_proj,
                 r["final_rmsnorm_gain_max_over_mean"]), flush=True)
        del T, Pool, Proj, Z, Tc
    out["cuda_max_mem_gb"] = (round(float(torch.cuda.max_memory_allocated()) / 1e9, 3)
                              if dev.type == "cuda" else None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), "utf-8")
    print(f"[rank] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
