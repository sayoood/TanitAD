"""E-DEC-63 FOLLOW-UP — F1..F4: what IS the pixel-borne residual component?

Specified in `.../2026-08-26-residual-ceiling/RESULT.md` before this ran. The main
panel measured: raw pixels' marginal over z_t = +0.0096 (t 5.11, SURVIVES) while
the tokens+pixels ORACLE — a strict SUPERSET — read null. Two live explanations,
each with a column that kills it:

  F1  z_t + pixels + noise5120   width-matched to the oracle. If +0.0096 DIES,
                                 the oracle null was CAPACITY DILUTION and the
                                 token question REOPENS; if it survives, the
                                 token field genuinely lacks the component.
  F2  z_t + lum3                 3-dim photometric control (global/top/bottom
                                 luminance). If ~+0.01 REAPPEARS here, the
                                 "discovery" is auto-exposure/sun drift — real
                                 but boring — and it must not steer GPU.
  F3  z_t + tokens (no pixels)   the token side of the dilution question.
  F4  per-direction split        which Δz PCA directions carry A3's marginal —
                                 one direction (global appearance) vs spread
                                 (scene structure).

Rig: byte-identical machinery to the main panel (same corpus, 80 clips, k=4,
band [0:8), folds 10, same rff_fold/λ protocol). The z_t and z_t+pixels columns
are RE-RUN here rather than quoted, so F-columns and their baseline share every
fold draw. T0-DIAGNOSTIC. MEASURED (ours; dev-box RTX 4060).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import residual_ceiling as RC  # noqa: E402  (machinery; __main__-guarded)

SP = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path(os.environ.get("SPD_OUT", str(SP / "residual_ceiling_f.json")))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    import panel_kfold as PK
    import v7tiny_g2 as G
    from rangeprobe_rff import rff_fold, within_clip_r

    dev = torch.device("cuda")
    clips = sorted(RC.LEAD.glob("*.v2ep.pt"))[:RC.N_CLIPS]
    print("\n  E-DEC-63 FOLLOW-UP — F1 dilution / F2 photometric / F3 tokens / "
          "F4 per-direction")
    print(f"  arm {RC.ARM} · {len(clips)} clips · k={RC.K} · band "
          f"[{RC.BAND_LO}:{RC.BAND_HI}) · folds {RC.K_FOLDS}\n", flush=True)

    world, st = G.load_arm(RC.ARM, dev)
    W = int(world.window)
    ZT, TOKS, PIXS, DZ = [], [], [], []
    t0 = time.time()
    for nn, c in enumerate(clips, 1):
        z, tok, pix, act, spd = RC.encode_clip_full(world, c, dev)
        m = min(len(z) - RC.K, len(act), len(spd))
        rows = np.arange(W - 1, m)
        if len(rows) < 30:
            continue
        zt = z.numpy().astype(np.float64)
        ZT.append(zt[rows])
        TOKS.append(tok.numpy()[rows].astype(np.float64))
        PIXS.append(pix.numpy()[rows].astype(np.float64))
        DZ.append(zt[rows + RC.K] - zt[rows])
        if nn % 20 == 0 or nn == len(clips):
            print(f"      [{nn}/{len(clips)}] encoded ({time.time()-t0:.0f}s)",
                  flush=True)
    del world
    torch.cuda.empty_cache()

    ALL = np.concatenate(DZ)
    mu = ALL.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(ALL - mu, full_matrices=False)
    rng = np.random.default_rng(64)

    def lum3(p):
        # 32x80 grey, flattened row-major: rows 0:16 = top half (sky-ish),
        # 16:32 = bottom (road-ish)
        g = p.reshape(len(p), RC.PIX_H, RC.PIX_W)
        return np.stack([g.mean((1, 2)), g[:, :16].mean((1, 2)),
                         g[:, 16:].mean((1, 2))], 1)

    COL = {
        "z_t (baseline)": ZT,
        "z_t + pixels (A3 re-run)":
            [np.concatenate([x, p], 1) for x, p in zip(ZT, PIXS)],
        "z_t + pixels + noise5120 (F1 DILUTION)":
            [np.concatenate([x, p, rng.standard_normal((len(x), 5120))], 1)
             for x, p in zip(ZT, PIXS)],
        "z_t + lum3 (F2 PHOTOMETRIC)":
            [np.concatenate([x, lum3(p)], 1) for x, p in zip(ZT, PIXS)],
        "z_t + tokens (F3 NO PIXELS)":
            [np.concatenate([x, tk], 1) for x, tk in zip(ZT, TOKS)],
    }
    nrow = sum(len(x) for x in ZT)
    print(f"\n  === {RC.ARM} (step {st}) — {len(ZT)} clips, {nrow} rows ===")
    print(f"  {'column':<44}{'d':>7}{'r':>9}{'shuf':>9}{'t':>8}")
    print("  " + "-" * 79, flush=True)

    cells, perdir = {}, {}
    for cn, X in COL.items():
        tr_d, sh_d = [], []
        tcol = time.time()
        for j in range(RC.BAND_LO, RC.BAND_HI):
            Y = [(dz - mu) @ Vt[j][:, None] for dz in DZ]
            rngj = np.random.default_rng(100 + j)
            Ysh = [y.ravel()[rngj.permutation(len(y))][:, None] for y in Y]
            tr_d.append(PK.kfold_clip_scores(X, Y, rff_fold, within_clip_r,
                                             RC.K_FOLDS))
            sh_d.append(PK.kfold_clip_scores(X, Ysh, rff_fold, within_clip_r,
                                             RC.K_FOLDS))
        cells[cn] = (np.concatenate(tr_d), np.concatenate(sh_d))
        perdir[cn] = [float(np.mean(t)) for t in tr_d]      # F4: per-direction
        tr, sh = cells[cn]
        d = tr - sh
        t = float(d.mean()) / max(float(d.std(ddof=1) / np.sqrt(len(d))), 1e-12)
        print(f"  {cn:<44}{X[0].shape[1]:>7}{tr.mean():>+9.4f}"
              f"{sh.mean():>+9.4f}{t:>8.2f}   ({time.time()-tcol:.0f}s)",
              flush=True)

    def tt(x):
        return float(x.mean()) / max(float(x.std(ddof=1) / np.sqrt(len(x))),
                                     1e-12)

    base = cells["z_t (baseline)"][0]
    print(f"\n  {'MARGINAL over z_t':<50}{'delta':>9}{'t':>8}  verdict")
    print("  " + "-" * 78)
    marg = {}
    for cn in COL:
        if cn == "z_t (baseline)":
            continue
        d = cells[cn][0] - base
        t = tt(d)
        v = ("SURVIVES" if t >= 4.2 else "MARGINAL" if t >= 2.9
             else "INSIDE_NULL")
        marg[cn] = {"delta": round(float(d.mean()), 4), "t": round(t, 2),
                    "verdict": v}
        print(f"  {cn:<50}{d.mean():>+9.4f}{t:>8.2f}  {v}", flush=True)

    # F4 — where does the pixel marginal live?
    print("\n  F4 per-direction marginal of A3 over baseline (r per Δz PC):")
    f4 = [round(a - b, 4) for a, b in
          zip(perdir["z_t + pixels (A3 re-run)"], perdir["z_t (baseline)"])]
    print("     " + "  ".join(f"PC{j}:{v:+.4f}" for j, v in enumerate(f4)))

    # mechanical reading, every cell enumerated
    a3 = marg["z_t + pixels (A3 re-run)"]
    f1 = marg["z_t + pixels + noise5120 (F1 DILUTION)"]
    f2 = marg["z_t + lum3 (F2 PHOTOMETRIC)"]
    f3 = marg["z_t + tokens (F3 NO PIXELS)"]
    reads = {
        "a3_replicates": a3["verdict"] != "INSIDE_NULL",
        "dilution_explains_oracle_null": f1["verdict"] == "INSIDE_NULL",
        "photometric_explains_a3": (f2["verdict"] != "INSIDE_NULL"
                                    and abs(f2["delta"]) >= 0.7 * abs(a3["delta"])),
        "tokens_carry_it": f3["verdict"] != "INSIDE_NULL",
    }
    print("\n  READS: " + json.dumps(reads))

    OUT.write_text(json.dumps({
        "_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
        "eval_tier": "T0-DIAGNOSTIC",
        "arm": RC.ARM, "step": int(st), "n_clips": len(ZT), "n_rows": nrow,
        "k": RC.K, "pca_band": [RC.BAND_LO, RC.BAND_HI],
        "columns": {cn: {"d": int(X[0].shape[1]),
                         "r": round(float(cells[cn][0].mean()), 4),
                         "shuf": round(float(cells[cn][1].mean()), 4)}
                    for cn, X in COL.items()},
        "marginals_over_baseline": marg,
        "f4_per_direction_pixel_marginal": f4,
        "reads": reads,
    }, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
