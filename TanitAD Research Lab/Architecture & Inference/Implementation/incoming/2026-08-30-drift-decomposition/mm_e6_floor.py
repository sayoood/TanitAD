"""MM-E6 step 0c — THE RAW-INPUT FLOOR for the environment side.

⛔ WHY. CLAUDE.md's probe rule requires a RAW-INPUT floor: *"a learned
representation that does not beat raw input has added nothing"*. MM-E6's
environment side rests on frozen DINOv3, and the claim forming from
``mm_e6_pctrl.py`` is that the frozen scene predicts drift only weakly
(mean +0.037 against self-reference's +0.667). That claim is NOT quotable until we
know what RAW PIXELS score on the identical target, folds and estimator — if raw
pixels score the same, the number is about the probe's reach, not about DINOv3.

This runs the identical panel twice, changing ONLY the input representation:

    DINOv3 (frozen, 4x8x1024)   the environment representation MM-E6's P uses
    raw pixels (FLOOR, 32x80x3) the same frames, downsampled, no learning at all

Both are reduced by a PCA basis of the SAME rank, fit on the FIT clips only, and
both are scored by the drift instrument's own clip-disjoint 10-fold RFF+ridge with
within-clip Pearson r against a TIME-SHUFFLED null through the identical path.
``n`` and ``d`` are printed for every cell, per the same rule.

⚠️ The frame buffer is named ``jpeg_buf`` while its ``codec`` field says ``png``;
PIL sniffs the magic bytes, and the decoded bank is CONTENT-ASSERTED (finite,
non-zero mean) before use — an all-zero floor arm would make every representation
look like a winner, which is the E-DETECT-1 failure exactly.

TIER: T0-DIAGNOSTIC. MEASURED (ours; dev-box CPU).
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

WORK = pathlib.Path(__file__).resolve().parent
OLD = pathlib.Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
                   r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
                   r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
MIRROR = pathlib.Path(r"C:\Users\Admin\tanitad-wt\stack")
sys.path.insert(0, str(OLD / "sp2"))
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(MIRROR))

import mm_e6_drift_decompose as M          # noqa: E402

OUT = pathlib.Path(os.environ.get("FLOOR_OUT", str(WORK / "mm_e6_floor.json")))
PIXDIR = pathlib.Path(os.environ.get("FLOOR_PIX", str(WORK / "raw_pix_32x80")))
ARM = os.environ.get("MME6_ARMS", "postrain30k").split(",")[0]
R = int(os.environ.get("MME6_RANK", "96"))
NB = int(os.environ.get("MME6_BANDS", "8"))
PH, PW = 32, 80


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def build_pix(clips):
    PIXDIR.mkdir(parents=True, exist_ok=True)
    for c in clips:
        dst = PIXDIR / f"{c.stem}.npy"
        d = torch.load(c, map_location="cpu", weights_only=False)
        raw = d["jpeg_buf"].numpy().tobytes()
        off = np.concatenate([[0], np.cumsum(d["jpeg_len"].tolist())]).astype(np.int64)
        m = min(len(off) - 1, M.F)
        if dst.is_file() and int(np.load(dst, mmap_mode="r").shape[0]) >= m:
            continue
        A = np.stack([np.asarray(
            Image.open(io.BytesIO(raw[off[j]:off[j + 1]])).convert("RGB")
            .resize((PW, PH)), dtype=np.float32) / 255.0 for j in range(m)])
        # ⛔ CONTENT ASSERTION — an all-zero floor scores at chance and makes
        # every trunk look like a winner.
        if not np.isfinite(A).all() or float(np.abs(A).mean()) == 0.0:
            raise SystemExit(f"[FATAL] raw-pixel bank for {c.stem} is "
                             f"non-finite / all-zero — NOT written")
        np.save(dst, A.reshape(m, -1).astype(np.float16))


def load_pix(stem, m, _stack3=False):
    return np.load(PIXDIR / f"{stem}.npy")[:m].astype(np.float32)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    import rangeprobe_rff as RF
    import v7tiny_g2 as G
    from rangeprobe_rff import rff_fold, within_clip_r
    rff_multi = M.make_rff_multi(RF)

    dev = M.pick_device()
    clips = sorted(M.CORPUS.glob("*.v2ep.pt"))
    if M.N_CLIPS_ALL:
        clips = clips[:M.N_CLIPS_ALL]
    have = [c for c in clips if (M.DINO / f"{c.stem}.npy").is_file()]
    scored, fit = have[:M.N_SCORED], have[M.N_SCORED:]
    print(f"\n  MM-E6 RAW-INPUT FLOOR · arm {ARM} · SCORED {len(scored)} / "
          f"FIT {len(fit)} · rank {R} · device {dev}\n", flush=True)
    t0 = time.time()
    build_pix(have)
    print(f"    raw-pixel bank {PH}x{PW}x3 ready ({time.time() - t0:.0f}s)",
          flush=True)

    w, st = G.load_arm(ARM, dev)
    Z, SPD, OM, mlen = {}, {}, {}, {}
    with torch.no_grad():
        for c in have:
            d = torch.load(c, map_location="cpu", weights_only=False)
            yaw = np.asarray(d["poses"], dtype=np.float64)[:, 2]
            z, act, spd = G.encode_clip(w, c, dev, M.F)
            zt = z.float().numpy()
            v = spd.float().numpy().ravel().astype(np.float64)
            nsc = int(np.load(M.DINO / f"{c.stem}.npy", mmap_mode="r").shape[0])
            npx = int(np.load(PIXDIR / f"{c.stem}.npy", mmap_mode="r").shape[0])
            m = min(len(zt) - M.K, len(act) - M.K, len(v) - M.K,
                    len(yaw) - M.K - 1, nsc, npx)
            if m < 30:
                continue
            i = np.arange(m)
            Z[c.stem] = zt[:m + M.K].astype(np.float64)
            SPD[c.stem] = v[i][:, None]
            OM[c.stem] = (wrap(yaw[i + 1] - yaw[i]) / 0.1)[:, None]
            mlen[c.stem] = m
    del w
    sc_ids = [c.stem for c in scored if c.stem in Z]
    fi_ids = [c.stem for c in fit if c.stem in Z]
    Zsc = [Z[i][:mlen[i]] for i in sc_ids]
    DZ = [Z[i][M.K:M.K + mlen[i]] - Z[i][:mlen[i]] for i in sc_ids]

    def pca_targets(mats, nb):
        A = np.concatenate(mats)
        mu = A.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(A - mu, full_matrices=False)
        del A
        return [[(x - mu) @ Vt[j][:, None] for x in mats] for j in range(nb)]

    names, Ys = [], []
    names.append("speed  ⭐POSITIVE CONTROL")
    Ys.append([SPD[i] for i in sc_ids])
    names.append("omega (yaw rate)")
    Ys.append([OM[i] for i in sc_ids])
    for j, t in enumerate(pca_targets(DZ, NB)):
        names.append(f"dz_pca{j}  ⭐⭐DRIFT")
        Ys.append(t)
    names.append("constant (ctrl)")
    Ys.append([np.ones((len(x), 1)) for x in Zsc])
    Yall = list(Ys)
    for j, Y in enumerate(Ys):
        rj = np.random.default_rng(500 + j)
        Yall.append([y.ravel()[rj.permutation(len(y))][:, None] for y in Y])
    ny = len(Ys)

    rep = {"_evidence_class": "MEASURED (ours; dev-box CPU)",
           "eval_tier": "T0-DIAGNOSTIC", "arm": ARM, "step": int(st),
           "rank": R, "scene_bank": str(M.DINO),
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
           "n_scored_clips": len(sc_ids),
           "n_rows": int(sum(mlen[i] for i in sc_ids)),
           "estimator": "clip-disjoint 10-fold RFF+ridge, within-clip Pearson r, "
                        "TIME-SHUFFLED null through the identical path",
           "inputs": {}}

    # ⚠️ THE LABEL IS DERIVED FROM THE BANK, NEVER HARDCODED. MEASURED 2026-08-30:
    # a hardcoded "4x8x1024" printed over a run that had actually read the 16x40
    # bank — the computation was right (d_raw 655,360 in the same line) and the
    # LABEL was wrong. That is the shape of every stale-claim defect here: an
    # artifact that says something other than what it is.
    gsh = np.load(M.DINO / f"{sc_ids[0]}.npy", mmap_mode="r").shape
    dino_dim = f"{gsh[1]}x{gsh[2]}" + (f" (grid {M.DINO.name})")
    for tag, loader, dim in (("DINOv3 (frozen)", M.load_scene, dino_dim),
                             ("raw pixels (FLOOR)", load_pix, f"{PH}x{PW}x3")):
        saved = M.load_scene
        M.load_scene = loader                    # scene_basis streams via this
        try:
            mu_s, V_s = M.scene_basis(fi_ids, mlen, False, R)
            A_fit = [(loader(i, mlen[i], False) - mu_s) @ V_s for i in fi_ids]
            sd = np.concatenate(A_fit).std(0, keepdims=True) + 1e-8
            X = [((((loader(i, mlen[i], False) - mu_s) @ V_s) / sd)
                  ).astype(np.float64) for i in sc_ids]
            d_raw = int(loader(sc_ids[0], 1, False).shape[1])
            del A_fit, V_s
        finally:
            M.load_scene = saved
        worst = M.verify_multi(X, Yall, rff_multi, rff_fold)
        if worst >= 1e-8:
            print(f"    [FATAL] vectorisation gate FAIL {worst:.3e}")
            return 3
        S = M.kfold_clip_scores_multi(X, Yall, rff_multi, within_clip_r)
        TR, SH = S[:ny], S[ny:]
        cells = {}
        print(f"\n    === input: {tag} ({dim} -> PCA rank {R}) · "
              f"n {rep['n_rows']} rows / {len(sc_ids)} clips · d_raw {d_raw} "
              f"· gate {worst:.1e} ===")
        print(f"    {'target':<34}{'r':>9}{'shuf':>9}{'r-shuf':>9}{'t':>8}"
              f"{'n':>7}{'d':>7}")
        print("    " + "-" * 83)
        for j, nm in enumerate(names):
            dd = TR[j] - SH[j]
            t = float(dd.mean()) / max(
                float(dd.std(ddof=1) / np.sqrt(len(dd))), 1e-12)
            cells[nm] = {"r": round(float(TR[j].mean()), 4),
                         "r_time_shuffled": round(float(SH[j].mean()), 4),
                         "r_minus_shuffled": round(float(dd.mean()), 4),
                         "t": round(t, 2), "n_rows": rep["n_rows"],
                         "d_input_pcs": R, "d_raw": d_raw}
            print(f"    {nm:<34}{TR[j].mean():>+9.4f}{SH[j].mean():>+9.4f}"
                  f"{dd.mean():>+9.4f}{t:>8.2f}{rep['n_rows']:>7}{R:>7}",
                  flush=True)
        db = np.array([cells[n]["r_minus_shuffled"] for n in names
                       if n.startswith("dz_pca")])
        rep["inputs"][tag] = {"d_raw": d_raw, "targets": cells,
                              "dz_mean_r_minus_shuffled": round(float(db.mean()), 4)}
        print(f"    -> drift (top-{NB} PCA) mean r-shuf {db.mean():+.4f}")
        OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        del X, S

    a = rep["inputs"]["DINOv3 (frozen)"]["dz_mean_r_minus_shuffled"]
    b = rep["inputs"]["raw pixels (FLOOR)"]["dz_mean_r_minus_shuffled"]
    rep["verdict"] = {
        "dino_dz": a, "floor_dz": b, "dino_minus_floor": round(a - b, 4),
        "dino_beats_raw_pixels": bool(a > b)}
    print(f"\n    ⭐ DINOv3 {a:+.4f} vs RAW-PIXEL FLOOR {b:+.4f} "
          f"(margin {a - b:+.4f}) — "
          f"{'DINOv3 beats the floor' if a > b else 'DINOv3 DOES NOT beat the floor'}")
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
