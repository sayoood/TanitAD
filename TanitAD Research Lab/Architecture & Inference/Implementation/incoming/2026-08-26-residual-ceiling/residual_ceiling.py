"""E-DEC-63 — WHAT IS THE CEILING ON PREDICTING Δz BEYOND DRIFT?

Pre-registered: `TanitAD Research Lab/Architecture & Inference/Implementation/
incoming/2026-08-26-residual-ceiling/SPEC.md` (+ AMENDMENT A1, recorded before
any result was read). Both outcomes committed in advance:

  H-A  the ORACLE column's marginal over drift is inside the null
       -> the post-drift residual is NOT predictable from time-t observation;
          stop optimising the predictor, the work is representational.
  H-B  the ORACLE clears with headroom over OUR PREDICTOR's column
       -> the loss/architecture IS the lever.

RIG — identical to the banked E-DEC-40/59 construction so the positive control
has a KNOWN value (`latentmotion.json`: rdw8p30k drift r +0.6718, t 134.84):
corpus `physicalai-val130-heldout` (80 clips, F=100), k=4, Δz PCA band [0:8),
K-fold fit / per-clip score, RFF+ridge (NONLINEAR, convex), λ on a clip-disjoint
inner split of FIT only. The readable quantity of every arm is its per-clip
paired MARGINAL over the z_t (drift) column.

COLUMNS (SPEC arm -> column)
  A0  constant                          reads EXACTLY 0.0000
  A1  z_t  (DRIFT / POSITIVE CONTROL)   must read ~ +0.6718 / t ~ 135
  A6  z_t + noise3                      DELIBERATE REGRESSION - marginal MUST be null
  A2  z_t + zhat_k4                     our predictor (T0 true-action conditioning)
  A3  z_t + pixels 32x80               raw-input floor
  A4  z_t + tokens 4x10x768 + pixels    ORACLE-t (AMENDMENT A1: pooled field)
  A5  = each column's TIME-SHUFFLED read through the identical path

VERDICT TIERS (conservative, stated in advance): SURVIVES t>=4.2, MARGINAL
2.9<=t<4.2, INSIDE_NULL t<2.9 — 2.9 is the 104-draw measured-null bar and 4.2
covers the K-fold null's observed 4.15 excursion (E-DEC-59 notes). Every cell is
enumerated; there is no substantive default (C160).

T0-DIAGNOSTIC. MEASURED (ours; dev-box RTX 4060). NEVER a driving number.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

SP_OLD = pathlib.Path(
    r"C:\Users\Admin\AppData\Local\Temp\claude"
    r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
    r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
SP = pathlib.Path(__file__).resolve().parent
# ⛔ BIND tanitad FROM THE MIRROR BEFORE v7tiny_g2 CAN PUSH THE G: STACK ONTO
# sys.path — the namespace-shadow rule: once bound, later path edits are inert,
# which here is exactly what we want (G: cannot RUN the stack, Errno 22).
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
import tanitad  # noqa: E402  (eager bind, mirror)
sys.path.insert(0, str(SP_OLD))
sys.path.insert(0, str(SP_OLD / "sp2"))

ARM = os.environ.get("SPD_ARM", "rdw8p30k")
LEAD = pathlib.Path(os.environ.get(
    "SPD_CORPUS", str(SP_OLD / "sp2/cache/physicalai-val130-heldout")))
OUT = pathlib.Path(os.environ.get("SPD_OUT", str(SP / "residual_ceiling.json")))
N_CLIPS = int(os.environ.get("SPD_NCLIPS", "80"))
F, K, K_FOLDS = 100, 4, 10
_b = os.environ.get("SPD_BAND", "0:8").split(":")
BAND_LO, BAND_HI = int(_b[0]), int(_b[1])
PIX_H, PIX_W = 32, 80
TOK_PH, TOK_PW = 4, 10


def encode_clip_full(world, path, dev):
    """z [n,d_op], pooled tokens [n, 4*10*768] fp16, grey pixels [n, 32*80]."""
    import v7tiny_g2 as G
    d, raw, off, n, _codec = G.frames_of(path)
    n = min(n, F)
    imgs = []
    for i in range(n):
        im = Image.open(io.BytesIO(raw[off[i]:off[i + 1]])).convert("RGB")
        imgs.append(torch.from_numpy(np.asarray(im).copy())
                    .permute(2, 0, 1).float() / 255.0)
    if not imgs or float(imgs[0].abs().mean()) == 0.0:
        raise SystemExit(f"[FATAL] {path.name} decoded to all-zero frames")
    Z, TOK, PIX = [], [], []
    with torch.no_grad():
        for s in range(0, n, 16):
            chunk, pchunk = [], []
            for i in range(s, min(s + 16, n)):
                idx = [max(i - j, 0) for j in range(G.N_STACK - 1, -1, -1)]
                chunk.append(torch.cat([imgs[k] for k in idx], 0))
                pchunk.append(imgs[i].mean(0, keepdim=True))     # grey [1,H,W]
            x = torch.stack(chunk)[:, None].to(dev)              # [b,1,9,H,W]
            # the V6ProbeTrunk wrapper does not forward return_tokens; the real
            # encode_window (v6.py:4924) is on the wrapped stack
            z, tok = world.stack.encode_window(x, return_tokens=True)
            Z.append(z[:, 0].float().cpu())
            t = tok[:, 0]                                        # [b,640,768]
            b = t.shape[0]
            t = t.reshape(b, 16, 40, -1).permute(0, 3, 1, 2)     # [b,768,16,40]
            t = torch.nn.functional.adaptive_avg_pool2d(t, (TOK_PH, TOK_PW))
            TOK.append(t.permute(0, 2, 3, 1).reshape(b, -1).half().cpu())
            p = torch.nn.functional.adaptive_avg_pool2d(
                torch.stack(pchunk), (PIX_H, PIX_W))
            PIX.append(p.reshape(p.shape[0], -1))
    return (torch.cat(Z), torch.cat(TOK), torch.cat(PIX),
            d["actions"].float()[:n], d["poses"].float()[:n, 3])


def zhat_rows(world, z, act, spd, rows, dev):
    """Predictor output at horizon K for each anchor row (T0: TRUE actions)."""
    from tanitad.models.flagship_v15 import SPEED_SCALE
    W = int(world.window)
    H = sorted(int(h) for h in world.stack.cfg.predictor.horizons)
    if K not in H:
        raise SystemExit(f"[FATAL] horizon {K} not in predictor horizons {H}")
    out = torch.empty(len(rows), z.shape[1])
    with torch.no_grad():
        for s in range(0, len(rows), 64):
            rr = rows[s:s + 64]
            zb = torch.stack([z[i - W + 1:i + 1] for i in rr]).to(dev)
            aa = torch.stack([act[i - W + 1:i + 1] for i in rr]).to(dev)
            vv = torch.stack([spd[i - W + 1] for i in rr]).to(dev)
            a3 = torch.cat([aa, (vv / SPEED_SCALE)[:, None, None]
                            .expand(-1, W, -1)], -1)
            out[s:s + len(rr)] = world.predictor(zb, a3)[K].float().cpu()
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    import panel_kfold as PK
    import v7tiny_g2 as G
    from rangeprobe_rff import rff_fold, within_clip_r

    assert "tanitad-mirror" in tanitad.__file__, tanitad.__file__
    dev = torch.device("cuda")
    clips = sorted(LEAD.glob("*.v2ep.pt"))[:N_CLIPS]
    print("\n  E-DEC-63 — THE CEILING ON PREDICTING Δz BEYOND DRIFT")
    print(f"  arm {ARM} · {len(clips)} clips · k={K} · band [{BAND_LO}:{BAND_HI})"
          f" · folds {K_FOLDS}")
    print("  every arm reads as its per-clip paired MARGINAL over z_t (drift)\n",
          flush=True)

    world, st = G.load_arm(ARM, dev)
    W = int(world.window)
    ZT, TOKS, PIXS, ZH, DZ = [], [], [], [], []
    t0 = time.time()
    for nn, c in enumerate(clips, 1):
        z, tok, pix, act, spd = encode_clip_full(world, c, dev)
        m = min(len(z) - K, len(act), len(spd))
        rows = list(range(W - 1, m))
        if len(rows) < 30:
            continue
        zh = zhat_rows(world, z, act, spd, rows, dev)
        zt = z.numpy().astype(np.float64)
        i = np.asarray(rows)
        ZT.append(zt[i])
        TOKS.append(tok.numpy()[i].astype(np.float64))
        PIXS.append(pix.numpy()[i].astype(np.float64))
        ZH.append(zh.numpy().astype(np.float64))
        DZ.append(zt[i + K] - zt[i])
        if nn % 10 == 0 or nn == len(clips):
            print(f"      [{nn}/{len(clips)}] encoded ({time.time()-t0:.0f}s)",
                  flush=True)
    del world
    torch.cuda.empty_cache()
    if len(DZ) < 10:
        print("  [FATAL] too few clips")
        return 1

    ALL = np.concatenate(DZ)
    mu = ALL.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(ALL - mu, full_matrices=False)
    rng = np.random.default_rng(63)
    COL = {
        "constant (A0)": [np.ones((len(x), 1)) for x in ZT],
        "z_t (A1 DRIFT/POSITIVE CTRL)": ZT,
        "z_t + noise3 (A6 DELIB-REGRESS)":
            [np.concatenate([x, rng.standard_normal((len(x), 3))], 1)
             for x in ZT],
        "z_t + zhat_k4 (A2 OUR PREDICTOR)":
            [np.concatenate([x, h], 1) for x, h in zip(ZT, ZH)],
        "z_t + pixels (A3 RAW FLOOR)":
            [np.concatenate([x, p], 1) for x, p in zip(ZT, PIXS)],
        "z_t + tokens4x10 + pixels (A4 ORACLE-t)":
            [np.concatenate([x, tk, p], 1)
             for x, tk, p in zip(ZT, TOKS, PIXS)],
    }
    nrow = sum(len(x) for x in ZT)
    print(f"\n  === {ARM} (step {st}) — {len(ZT)} clips, {nrow} rows ===")
    print(f"  {'column':<40}{'d':>7}{'r':>9}{'shuf':>9}{'t':>8}")
    print("  " + "-" * 75, flush=True)

    cells = {}
    for cn, X in COL.items():
        tr, sh = [], []
        tcol = time.time()
        for j in range(BAND_LO, BAND_HI):
            Y = [(dz - mu) @ Vt[j][:, None] for dz in DZ]
            rngj = np.random.default_rng(100 + j)
            Ysh = [y.ravel()[rngj.permutation(len(y))][:, None] for y in Y]
            tr.append(PK.kfold_clip_scores(X, Y, rff_fold, within_clip_r,
                                           K_FOLDS))
            sh.append(PK.kfold_clip_scores(X, Ysh, rff_fold, within_clip_r,
                                           K_FOLDS))
        tr, sh = np.concatenate(tr), np.concatenate(sh)
        cells[cn] = (tr, sh)
        d = tr - sh
        t = float(d.mean()) / max(float(d.std(ddof=1) / np.sqrt(len(d))), 1e-12)
        print(f"  {cn:<40}{X[0].shape[1]:>7}{tr.mean():>+9.4f}"
              f"{sh.mean():>+9.4f}{t:>8.2f}   ({time.time()-tcol:.0f}s)",
              flush=True)

    def tt(x):
        return float(x.mean()) / max(float(x.std(ddof=1) / np.sqrt(len(x))),
                                     1e-12)

    drift_key = "z_t (A1 DRIFT/POSITIVE CTRL)"
    base = cells[drift_key][0]
    print(f"\n  {'MARGINAL over z_t (the readable quantity)':<46}"
          f"{'delta':>9}{'t':>8}  verdict")
    print("  " + "-" * 78)
    marg = {}
    for cn in COL:
        if cn == drift_key or cn.startswith("constant"):
            continue
        d = cells[cn][0] - base
        t = tt(d)
        v = ("SURVIVES" if t >= 4.2 else
             "MARGINAL" if t >= 2.9 else "INSIDE_NULL")
        marg[cn] = {"delta": round(float(d.mean()), 4), "t": round(t, 2),
                    "verdict": v}
        print(f"  {cn:<46}{d.mean():>+9.4f}{t:>8.2f}  {v}", flush=True)

    # ---- rig validity, every cell enumerated, no substantive default (C160) --
    a0 = float(cells["constant (A0)"][0].mean())
    a1r, a1t = float(base.mean()), tt(base - cells[drift_key][1])
    a6t = marg["z_t + noise3 (A6 DELIB-REGRESS)"]["t"]
    shuf_ok = all(abs(float(v[1].mean())) < 0.25 for v in cells.values())
    checks = {
        "A0_constant_reads_zero": a0 == 0.0,
        "A1_drift_in_known_band": 0.60 <= a1r <= 0.74,
        "A1_banked_value": {"expected_r": 0.6718, "expected_t": 134.84,
                            "got_r": round(a1r, 4), "got_t": round(a1t, 2)},
        "A6_deliberate_regression_failed_as_required": abs(a6t) < 2.9,
        "all_shuffle_controls_near_zero": shuf_ok,
    }
    rig_valid = (checks["A0_constant_reads_zero"]
                 and checks["A1_drift_in_known_band"]
                 and checks["A6_deliberate_regression_failed_as_required"]
                 and shuf_ok)
    print(f"\n  RIG {'VALID' if rig_valid else '*** INVALID — READ NOTHING ***'}"
          f"  (A0 {a0:+.4f} · A1 r {a1r:+.4f} t {a1t:.1f} · A6 marginal t "
          f"{a6t:+.2f} · shuffles {'ok' if shuf_ok else 'BAD'})")

    verdict = "RIG_INVALID"
    if rig_valid:
        a4 = marg["z_t + tokens4x10 + pixels (A4 ORACLE-t)"]
        a2 = marg["z_t + zhat_k4 (A2 OUR PREDICTOR)"]
        if a4["verdict"] == "INSIDE_NULL":
            verdict = ("H-A: the post-drift residual is NOT predictable from "
                       "4x10-pooled time-t observation — stop optimising the "
                       "predictor; the work is representational")
        elif a4["verdict"] in ("SURVIVES", "MARGINAL"):
            d = (cells["z_t + tokens4x10 + pixels (A4 ORACLE-t)"][0]
                 - cells["z_t + zhat_k4 (A2 OUR PREDICTOR)"][0])
            th = tt(d)
            if th >= 2.9:
                verdict = (f"H-B ({a4['verdict']}): residual predictable and "
                           f"OUR PREDICTOR IS BELOW THE CEILING (oracle-vs-"
                           f"predictor t {th:+.2f}) — the loss/architecture is "
                           f"the lever")
            else:
                verdict = (f"AT-CEILING ({a4['verdict']}): oracle clears but "
                           f"our predictor matches it (t {th:+.2f}) — same "
                           f"action as H-A, stronger statement")
            checks["oracle_vs_predictor_t"] = round(th, 2)
        print(f"\n  VERDICT: {verdict}\n", flush=True)

    OUT.write_text(json.dumps({
        "_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
        "eval_tier": "T0-DIAGNOSTIC",
        "prereg": "TanitAD Research Lab/Architecture & Inference/Implementation/"
                  "incoming/2026-08-26-residual-ceiling/SPEC.md (+AMENDMENT A1)",
        "arm": ARM, "step": int(st), "n_clips": len(ZT), "n_rows": nrow,
        "k": K, "pca_band": [BAND_LO, BAND_HI], "k_folds": K_FOLDS,
        "oracle_form": f"tokens {TOK_PH}x{TOK_PW}x"
                       f"{TOKS[0].shape[1] // (TOK_PH * TOK_PW)} + grey "
                       f"{PIX_H}x{PIX_W} + z_t (AMENDMENT A1 pooling; token "
                       f"d_model read from the bank, NOT assumed)",
        "columns": {cn: {"d": int(X[0].shape[1]),
                         "r": round(float(cells[cn][0].mean()), 4),
                         "shuf": round(float(cells[cn][1].mean()), 4)}
                    for cn, X in COL.items()},
        "marginals_over_drift": marg,
        "rig_checks": checks, "rig_valid": rig_valid, "verdict": verdict,
    }, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
