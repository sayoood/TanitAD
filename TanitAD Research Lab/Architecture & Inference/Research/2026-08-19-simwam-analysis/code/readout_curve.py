"""E-DEC-3: the decodability-vs-AZIMUTH-BINS curve, measured WITHOUT retraining.

E-DEC-1 showed the trunk HAS decodable content that the 4x4 readout discards
(enc - z_op = +0.2117, t 7.86, 12/12 episodes). The obvious next question is not
"does a wider readout help" -- it is "HOW WIDE does it need to be", and that is
answerable on the ALREADY-TRAINED encoder, in minutes, with no GPU-days.

Take champ30k's frozen encoder tokens (16 rows x 40 azimuth columns over a 120 deg
CYLINDRICAL FOV -- the column axis is LINEAR IN AZIMUTH, so 40 columns = 3 deg per
column) and re-pool them into grids of increasing azimuth resolution:

    4 x 4   -> 4 bins,  30.0 deg/bin   (THE INCUMBENT)
    4 x 8   -> 8 bins,  15.0 deg/bin
    4 x 10  -> 10 bins, 12.0 deg/bin
    4 x 20  -> 20 bins,  6.0 deg/bin
    4 x 40  -> 40 bins,  3.0 deg/bin   (no azimuth pooling at all)
    16 x 40 -> full token grid, flattened (no pooling whatsoever)

Every column is PCA'd to k=128 inside the probe, so this is NOT a
feature-dimension race: it isolates WHERE THE POOLING BOUNDARIES SIT.

Scored with the LOEO paired design that settled E-DEC-1 (fit on 11 episodes,
score the held-out one), against the raw-pixel floor and frozen DINOv3.

⛔ MEMORY: 12 clips x 120 frames x 640 tokens x 128 dims is 1.2 GB in float64 --
tokens are pooled PER CLIP on arrival and the full stack is never held.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
OUT = SP / "e_dec3_readout_curve.json"
FOV_DEG = 120.0

GRIDS = [(4, 4), (4, 8), (4, 10), (4, 20), (4, 40), (16, 40)]


def pool_tokens(tok: np.ndarray, rows: int, cols: int, gh: int, gw: int) -> np.ndarray:
    """[F, rows*cols, d] -> [F, gh*gw*d] by mean-pooling each cell."""
    f, n, d = tok.shape
    assert n == rows * cols, f"{n} tokens != {rows}x{cols}"
    assert rows % gh == 0 and cols % gw == 0, f"{rows}x{cols} not divisible by {gh}x{gw}"
    t = tok.reshape(f, gh, rows // gh, gw, cols // gw, d)
    return t.mean(axis=(2, 4)).reshape(f, gh * gw * d)


def main() -> int:
    import v7tiny_g2 as G
    import v7tiny_probe as P

    dev = torch.device("cuda")
    arm = sys.argv[1] if len(sys.argv) > 1 else "champ30k"
    clips = sorted((SP / "sp2/cache/physicalai-val-w120-256x640cyl").glob("*.v2ep.pt"))
    F = 120
    world, st = G.load_arm(arm, dev)

    # token grid geometry from the model itself, never assumed
    rows = 256 // world.stack.cfg.encoder.patch_size
    cols = 640 // world.stack.cfg.encoder.patch_size
    print(f"\n  E-DEC-3 · {arm}@{st} · token grid {rows}x{cols} over {FOV_DEG:.0f} deg "
          f"cylindrical = {FOV_DEG / cols:.2f} deg/column\n", flush=True)

    import io as _io
    from PIL import Image as _Image
    COLS = {f"{gh}x{gw}": [] for gh, gw in GRIDS}
    PIX, PO = [], []
    for ci, c in enumerate(clips, 1):
        d = torch.load(c, map_location="cpu", weights_only=False)
        raw = d["jpeg_buf"].numpy().tobytes()
        off = np.concatenate([[0], np.cumsum(d["jpeg_len"].tolist())]).astype(np.int64)
        m = min(F, len(off) - 1)
        # ⛔ the encoder takes a 3-FRAME STACK (9 channels), and the tokens must be
        # produced by `stack.encode_window` exactly as the incumbent readout sees
        # them -- feeding a bare 3-channel image raises "expected 9 channels".
        # This mirrors v7tiny_probe.encode_tokens_meanpooled, minus its pooling.
        imgs = []
        for i in range(m):
            im = _Image.open(_io.BytesIO(raw[off[i]:off[i + 1]])).convert("RGB")
            imgs.append(torch.from_numpy(np.asarray(im).copy())
                        .permute(2, 0, 1).float() / 255.0)
        toks = []
        n_stack = 3
        with torch.no_grad():
            for s in range(0, m, 8):
                chunk = []
                for i in range(s, min(s + 8, m)):
                    idx = [max(i - j, 0) for j in range(n_stack - 1, -1, -1)]
                    chunk.append(torch.cat([imgs[k] for k in idx], 0))
                x = torch.stack(chunk)[:, None].to(dev)
                _z, tk = world.stack.encode_window(x, return_tokens=True)
                toks.append(tk[:, 0].float().cpu().numpy())
        T = np.concatenate(toks).astype(np.float64)          # [m, rows*cols, d]
        for gh, gw in GRIDS:
            COLS[f"{gh}x{gw}"].append(pool_tokens(T, rows, cols, gh, gw))
        PO.append(d["poses"].numpy().astype(np.float64)[:m])
        PIX.append(P.pooled_frames(c, m))
        print(f"    [{ci}/{len(clips)}] {c.name[:10]} tokens={T.shape}", flush=True)
        del T, toks
    DN = P.dinov3_encode(clips, F, dev)
    del world
    torch.cuda.empty_cache()

    TGT = {"speed": [p[:, 3:4] for p in PO],
           "d_ego": [np.concatenate([np.diff(p[:, :2], axis=0), np.zeros((1, 2))]) for p in PO]}
    COLS["pixel (floor)"] = PIX
    COLS["frozen DINOv3"] = DN
    COLS["constant (control)"] = [np.ones((len(p), 1)) for p in PO]
    n = len(clips)

    def loeo(X, Y):
        out = []
        for e in range(n):
            idx = [i for i in range(n) if i != e]
            Xf = [np.asarray(X[i])[:len(Y[i])] for i in idx]
            Yf = [Y[i][:len(np.asarray(X[i]))] for i in idx]
            Xs = [np.asarray(X[e])[:len(Y[e])]]
            Ys = [Y[e][:len(np.asarray(X[e]))]]
            fn = getattr(P, "probe_fit_score", None)
            out.append(fn(Xf, Yf, Xs, Ys, 128) if fn else P.probe(Xf + Xs, Yf + Ys, 128)[0])
        return np.array(out, dtype=np.float64)

    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)", "eval_tier": "T0-DIAGNOSTIC",
           "arm": arm, "step": int(st), "n_episodes": n, "token_grid": [rows, cols],
           "deg_per_column": round(FOV_DEG / cols, 3),
           "estimator": "leave-one-episode-out, paired against the 4x4 incumbent",
           "targets": {}}
    print(f"\n  {'target':<9}{'readout':<16}{'az bins':>8}{'deg/bin':>9}{'R2':>9}"
          f"{'vs 4x4':>9}{'t':>7}{'12/12?':>8}")
    print("  " + "-" * 78)
    for tn, Y in TGT.items():
        R = {k: loeo(v, Y) for k, v in COLS.items()}
        base = R["4x4"]
        rep["targets"][tn] = {}
        for k in list(COLS):
            r = R[k]
            d = r - base
            m = float(d.mean())
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
            t = m / max(se, 1e-12)
            gw = int(k.split("x")[1]) if "x" in k and k[0].isdigit() else None
            rep["targets"][tn][k] = {"r2": round(float(r.mean()), 4),
                                     "delta_vs_4x4": round(m, 4), "t": round(t, 2),
                                     "n_favouring": int((d > 0).sum()),
                                     "az_bins": gw,
                                     "deg_per_bin": round(FOV_DEG / gw, 2) if gw else None}
            print(f"  {tn:<9}{k:<16}{str(gw or '-'):>8}"
                  f"{(f'{FOV_DEG / gw:.1f}' if gw else '-'):>9}"
                  f"{float(r.mean()):>+9.4f}{m:>+9.4f}{t:>7.2f}"
                  f"{int((d > 0).sum()):>6}/{len(d)}")
        print()

    ctrl = rep["targets"]["speed"]["constant (control)"]["r2"]
    if abs(ctrl) > 1e-6:
        rep["verdict"] = f"⛔ VOID — constant control reads {ctrl}, not 0."
    else:
        sp = rep["targets"]["speed"]
        gains = [(v["az_bins"], v["delta_vs_4x4"], v["t"]) for k, v in sp.items()
                 if v["az_bins"] and v["az_bins"] > 4]
        sig = [g for g in gains if g[2] > 2.2]
        if sig:
            best = max(sig, key=lambda g: g[1])
            knee = min(g[0] for g in sig)
            rep["verdict"] = (
                f"⭐ AZIMUTH RESOLUTION IS THE LEVER — decodability rises above the 4x4 incumbent "
                f"from {knee} bins ({FOV_DEG / knee:.0f} deg/bin) onward; best {best[0]} bins "
                f"(+{best[1]:.4f}, t {best[2]:.2f}). The readout geometry can be fixed WITHOUT "
                f"retraining the encoder.")
        else:
            rep["verdict"] = ("Widening the azimuth axis on the FROZEN encoder does NOT recover "
                              "decodability ⇒ the loss is not a pooling-boundary effect; the "
                              "readout must be RE-TRAINED, not merely re-shaped.")
    print(f"  VERDICT: {rep['verdict']}")
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
