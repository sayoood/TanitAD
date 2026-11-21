"""G2 for v7-tiny — does the predictor beat HOLD, on clips it has NEVER seen?

⛔ THE QUESTION, and why ``o5_step1`` cannot answer it. The trainer logs
``o5_step1``, an ABSOLUTE MAE. A predictor that has collapsed to emitting a zero
delta scores an EXCELLENT absolute MAE, because the latent barely moves per
tick. It is also worthless. The only statistic that separates "beats hold" from
"collapsed to hold" normalises by the movement actually present:

    EM = 1 - sum||zhat - z+||^2 / sum||z - z+||^2

    EM  > 0   beats HOLD -- the predictor explains real movement
    EM == 0   IS hold    -- predicts no change, explains nothing
    EM  < 0   worse than HOLD

⭐ EM is also the ONLY cross-arm-comparable number here. The two arms trained
their OWN encoders, so their latent spaces and their hold baselines differ; a
raw MAE compared across arms compares two different rulers. EM is normalised
per arm by that arm's own movement, so it is not.

⛔ HELD-OUT BY CONSTRUCTION. The arms were trained with NO ``--v2-val-cache``, so
all 130 clips of ``slotprobe-lead130`` are IN-SAMPLE. This reads clips pulled
from the 2,270 episodes of the canonical split that neither arm has ever seen.

⚠️ THE TICK TRAP (same as E-RESCUE): the predictor's tick is dt=0.1 s = ONE
frame. Latents are therefore encoded at STRIDE 1 here, not read from a stride-4
cache, or a 6-row window would span 2.4 s instead of 0.6 s.

⚠️ The encoder takes NINE channels (n_stack=3, three frames stacked as channels).

⚠️ The frame buffer is named ``jpeg_buf`` but its ``codec`` field is what decides
the decoder -- on this corpus it says ``png``. Trusting the NAME is what filled a
2.76 GB memmap with zeros once already.

ESTIMATOR: episode-cluster bootstrap of the POOLED statistic over held-out
clips, and the PAIRED version for the arm difference -- never a mean of
per-episode EMs, never ``overlapping_holdout_se``.

TIER: T0-DIAGNOSTIC. A world-model diagnostic; NEVER a driving number.
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
from PIL import Image

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, str(SP))
sys.path.insert(0, str(Path(r"G:\Meine Ablage\SayBouBase\raw\Projects"
                            r"\TanitAD\stack")))

N_STACK = 3
ARMS = ("fixed", "regress")


def frames_of(path: Path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    raw = d["jpeg_buf"].numpy().tobytes()
    off = np.concatenate([[0], np.cumsum(d["jpeg_len"].tolist())]).astype(
        np.int64)
    return d, raw, off, len(off) - 1, d.get("codec", "jpeg")


def encode_clip(world, path: Path, dev, max_frames: int):
    """STRIDE-1 latents for one clip through THIS arm's own encoder."""
    d, raw, off, n, _codec = frames_of(path)
    n = min(n, max_frames)
    imgs = []
    for i in range(n):
        im = Image.open(io.BytesIO(raw[off[i]:off[i + 1]])).convert("RGB")
        imgs.append(torch.from_numpy(np.asarray(im).copy())
                    .permute(2, 0, 1).float() / 255.0)
    if not imgs or float(imgs[0].abs().mean()) == 0.0:
        raise SystemExit(f"[FATAL] {path.name} decoded to all-zero frames")
    Z, B = [], 16
    with torch.no_grad():
        for s in range(0, n, B):
            chunk = []
            for i in range(s, min(s + B, n)):
                idx = [max(i - j, 0) for j in range(N_STACK - 1, -1, -1)]
                chunk.append(torch.cat([imgs[k] for k in idx], 0))
            x = torch.stack(chunk)[:, None].to(dev)             # [b,1,9,H,W]
            Z.append(world.encode_window(x)[:, 0].float().cpu())
    return torch.cat(Z), d["actions"].float()[:n], d["poses"].float()[:n, 3]


def load_arm(arm: str, dev):
    from tanitad.eval.v6_probe_trunk import load_trunk_auto
    p = SP / f"v7tiny_{arm}" / "ckpt.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    world, _g, step = load_trunk_auto(ck, dev, ckpt_path=str(p))
    for q in world.parameters():
        q.requires_grad_(False)
    world.eval()
    return world, int(step)


def per_clip_terms(world, clips, dev, horizons, W, max_frames):
    """-> {h: (err_per_clip, mov_per_clip)}, summed WITHIN each clip.

    Kept as per-clip SUMS so the bootstrap can resample clips and recompute the
    POOLED ratio. Averaging per-clip EMs is a different (and biased) statistic.
    """
    from tanitad.models.flagship_v15 import SPEED_SCALE
    acc = {h: ([], []) for h in horizons}
    hmax = max(horizons)
    t0 = time.time()
    for n, cp in enumerate(clips, 1):
        z, act, spd = encode_clip(world, cp, dev, max_frames)
        idx = list(range(len(z) - W - hmax + 1))
        if not idx:
            continue
        zs = torch.stack([z[i:i + W] for i in idx])
        aa = torch.stack([act[i:i + W] for i in idx])
        vv = torch.stack([spd[i] for i in idx])
        e_sum = {h: 0.0 for h in horizons}
        m_sum = {h: 0.0 for h in horizons}
        with torch.no_grad():
            for s in range(0, len(idx), 64):
                b = slice(s, s + 64)
                zb = zs[b].to(dev)
                a3 = torch.cat([aa[b].to(dev),
                                (vv[b].to(dev) / SPEED_SCALE)[:, None, None]
                                .expand(-1, W, -1)], -1)
                out = world.predictor(zb, a3)
                now = zb[:, -1]
                for h in horizons:
                    tgt = torch.stack([z[i + W - 1 + h] for i in idx[b]]).to(dev)
                    e_sum[h] += float(((out[h] - tgt) ** 2).sum())
                    m_sum[h] += float(((now - tgt) ** 2).sum())
        for h in horizons:
            acc[h][0].append(e_sum[h])
            acc[h][1].append(m_sum[h])
        print(f"      [{n}/{len(clips)}] {cp.name[:10]} {len(idx)} win "
              f"({time.time() - t0:.0f}s)", flush=True)
    return {h: (np.array(a), np.array(b)) for h, (a, b) in acc.items()}


def em(err, mov):
    return 1.0 - err.sum() / mov.sum()


def boot(errs, movs, n=4000, seed=0):
    """Episode-cluster bootstrap of the POOLED statistic."""
    rng = np.random.default_rng(seed)
    k = len(errs)
    out = np.empty(n)
    for i in range(n):
        j = rng.integers(0, k, k)
        out[i] = em(errs[j], movs[j])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def boot_paired(eA, mA, eB, mB, n=4000, seed=0):
    """PAIRED over the SAME resampled clips -- never a quadrature combination."""
    rng = np.random.default_rng(seed)
    k = len(eA)
    out = np.empty(n)
    for i in range(n):
        j = rng.integers(0, k, k)
        out[i] = em(eA[j], mA[j]) - em(eB[j], mB[j])
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)),
            float((out <= 0).mean()))


def main() -> int:
    ap = argparse.ArgumentParser(description="v7-tiny G2")
    ap.add_argument("--cache", default=str(
        SP / "sp2/cache/v7tiny-heldout24-w120-256x640cyl"))
    ap.add_argument("--clips", type=int, default=24)
    ap.add_argument("--frames-per-clip", type=int, default=140)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", default=str(SP / "v7tiny_g2.json"))
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Path(a.cache)
    clips = sorted(cache.glob("*.v2ep.pt"))[:a.clips]
    if not clips:
        print(f"  [FATAL] no clips in {cache}")
        return 1
    print(f"  {len(clips)} HELD-OUT clips from {cache.name}")
    prov = cache / "_PROVENANCE.json"
    if prov.exists():
        pj = json.loads(prov.read_text(encoding="utf-8"))
        print(f"  disjoint from {pj['disjoint_from']} "
              f"({pj['n_excluded_in_sample']} in-sample excluded, "
              f"{pj['n_available']} available, stride {pj['stride']})")

    res = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC",
           "gate": "G2 -- explained movement vs HOLD, on HELD-OUT clips",
           "estimator": "episode-cluster bootstrap of the POOLED statistic; "
                        "PAIRED for the arm difference",
           "parity": False, "n_clips": len(clips), "arms": {}}
    terms = {}
    for arm in a.arms:
        world, step = load_arm(arm, dev)
        W = int(world.window)
        H = sorted(int(h) for h in world.stack.cfg.predictor.horizons)
        sc = json.loads((SP / f"v7tiny_{arm}" / "init_scale.json")
                        .read_text(encoding="utf-8"))
        print(f"\n  === arm {arm.upper()} @ step {step} · window {W} · "
              f"horizons {H} · init_scale "
              f"{sc['residual_head_init_scale']} ===", flush=True)
        terms[arm] = per_clip_terms(world, clips, dev, H, W, a.frames_per_clip)
        res["arms"][arm] = {"step": step, "window": W, "horizons": H,
                            "residual_head_init_scale":
                                sc["residual_head_init_scale"], "em": {}}
        for h in H:
            e, m = terms[arm][h]
            lo, hi = boot(e, m)
            v = em(e, m)
            res["arms"][arm]["em"][str(h)] = {
                "explained_movement": round(float(v), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "mse_pred_over_hold": round(float(e.sum() / m.sum()), 4),
                "beats_hold": bool(lo > 0)}
            tag = ("BEATS hold" if lo > 0 else
                   "== hold (collapsed)" if hi > -0.02 else "WORSE than hold")
            print(f"    h={h} ({h * 0.1:.1f}s)  EM {v:+.4f}  "
                  f"[{lo:+.4f}, {hi:+.4f}]  ratio {e.sum() / m.sum():.4f}  "
                  f"{tag}")
        del world
        torch.cuda.empty_cache()

    if len(a.arms) == 2 and all(x in terms for x in ARMS):
        print("\n  === PAIRED: fixed - regress (same resampled clips) ===")
        res["paired"] = {}
        for h in sorted(terms["fixed"]):
            eF, mF = terms["fixed"][h]
            eR, mR = terms["regress"][h]
            lo, hi, p = boot_paired(eF, mF, eR, mR)
            d = em(eF, mF) - em(eR, mR)
            res["paired"][str(h)] = {"delta_em": round(float(d), 4),
                                     "ci95": [round(lo, 4), round(hi, 4)],
                                     "p_fixed_not_better": round(p, 4),
                                     "significant": bool(lo > 0)}
            print(f"    h={h}  dEM {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
                  f"p(fixed<=regress) {p:.4f}  "
                  f"{'SIGNIFICANT' if lo > 0 else 'not significant'}")
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
