"""E-DENSE-1 read-out — score the four arms with E-DETECT-1's validated head.

⚠️⚠️ THE NUMBERS HERE ARE NOT COMPARABLE TO E-DETECT-1's TABLE, and the floors
below exist so nobody tries. That table is 640 tokens x d 1024 on 256x640 frames;
this harness is 160 tokens x d 192 on 128x320 frames. A cross-table comparison
would be the derived-constant trap again — same metric, different experiment.

So every arm here is scored against floors built at THIS resolution:

  prior      closed form, per-fold train-mean occupancy. NO features.
  pixel160   ⛔ raw 16x16x3 patches from the 128x320 frames on the SAME 8x20
             grid the arms use. The resolution-matched floor.

and the four arms:

  pooled      A — v6's design: the loss sees only the pooled latent
  dense       B — the loss sees ALL 160 target tokens
  dense_deep  C — B plus an intermediate-layer dense target
  distill     D — POSITIVE CONTROL, regresses frozen DINOv3 tokens

⛔ THE COMMITTED DECISION RULE lives in `PREREG_E_DENSE_1.md` §5 and is not
restated here, so it cannot drift between the two documents.

TIER: T0-DIAGNOSTIC.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP))
import e_dense as D         # noqa: E402
import e_detect as E        # noqa: E402
import e_detect_prep as P   # noqa: E402
import e_trunk2_probe as T  # noqa: E402

CACHE = D.CACHE
PATCH_D = 16 * 16 * 3       # 768, the raw-patch dimension


def probe_rows() -> tuple[np.ndarray, list[tuple[str, int]]]:
    """The 5,617 probe rows, as positions into the 26,108-frame harness cache.

    ⚠️ Same two-index-space hazard as `e_dense.train_arm`'s distill branch:
    harness FRAME index vs probe ROW position. Kept explicit."""
    keys = [tuple(k) for k in
            json.loads((P.FEAT / "keys.json").read_text(encoding="utf-8"))]
    clips = json.loads((CACHE / "clips.json").read_text(encoding="utf-8"))
    fmap = {(c["clip_id"], f): c["start"] + f for c in clips
            for f in range(c["n"])}
    keep = [(k, fmap[k]) for k in keys if k in fmap]
    if len(keep) != len(keys):
        raise SystemExit(f"[FATAL] only {len(keep)} of {len(keys)} probe rows "
                         "map into the harness cache — the row sets differ and "
                         "no comparison is valid")
    return np.array([r for _, r in keep]), [k for k, _ in keep]


def build_pixel160(rows: np.ndarray) -> Path:
    """The resolution-matched floor: raw patches on the arms' own 8x20 grid."""
    out_p = P.OUT / "pixels160.npy"
    if out_p.exists():
        return out_p
    frames = np.load(CACHE / "frames.npy", mmap_mode="r")   # [N,3,128,320] u8
    m = np.lib.format.open_memmap(out_p, mode="w+", dtype=np.uint8,
                                  shape=(len(rows), D.N_TOK, PATCH_D))
    for i in range(0, len(rows), 512):
        blk = np.asarray(frames[rows[i:i + 512]])           # [b,3,H,W]
        b = blk.shape[0]
        a = (blk.transpose(0, 2, 3, 1)                      # [b,H,W,3]
                .reshape(b, D.GRID_H, 16, D.GRID_W, 16, 3)
                .transpose(0, 1, 3, 2, 4, 5)
                .reshape(b, D.N_TOK, PATCH_D))
        m[i:i + b] = a
    m.flush()
    s = np.asarray(m[::200])
    if (s.reshape(len(s), -1).max(1) == 0).any():
        raise SystemExit("[FATAL] pixels160 has all-zero rows")
    print(f"  pixel160 bank {m.shape} mean {float(s.mean()):.2f} — content OK")
    return out_p


def main() -> None:
    rows, _ = probe_rows()
    print(f"  {len(rows):,} probe rows located in the harness cache")
    build_pixel160(rows)

    occ = np.load(P.OUT / "occ.npy")
    ep = [k[0] for k in
          json.loads((P.FEAT / "keys.json").read_text(encoding="utf-8"))]
    folds = T.episode_folds(ep)
    tmp: dict[str, list[int]] = {}
    for i, e in enumerate(ep):
        tmp.setdefault(e, []).append(i)
    rows_by_ep = {k: np.array(v) for k, v in tmp.items()}
    base = float(occ.mean())
    pos_w = (1 - base) / base

    res_p = SP / "e_dense_score.json"
    out = (json.loads(res_p.read_text(encoding="utf-8"))
           if res_p.exists() and "--fresh" not in sys.argv else {
        "_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
        "eval_tier": "T0-DIAGNOSTIC",
        "prereg": "…/simwam-analysis/PREREG_E_DENSE_1.md",
        "warning": "NOT comparable to E_DETECT_1_RESULT.md's table — that is "
                   "640 tok x d1024 at 256x640; this is 160 tok x d192 at "
                   "128x320. Compare only within this table.",
        "n_tok": D.N_TOK, "grid": [D.GRID_H, D.GRID_W],
        "base_rate": round(base, 6), "arms": {}})

    def bank(name, rec):
        out["arms"][name] = rec
        res_p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"  {name:<14} AP {rec['ap']:.4f} {rec['ap_ci95']}  "
              f"AUC {rec['auc']:.4f}  loc {rec['topk_loc_err_m']} m", flush=True)

    # ---- floor: closed form, no features ---------------------------------- #
    if "prior" not in out["arms"]:
        pr = np.zeros_like(occ, np.float32)
        for k, te in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
            pr[te] = occ[tr].mean(0)[None, :]
        bank("prior", {"n_tok": 0, "d": 0,
                       **E.cluster_boot(pr, occ, rows_by_ep,
                                        np.random.default_rng(E.SEED)),
                       "topk_loc_err_m": round(E.topk_loc_err(pr, occ), 3),
                       "note": "closed form; NO features"})

    want = [a for a in sys.argv[1:] if not a.startswith("-")] or \
        (["pixel160"] + list(D.ARMS))
    for arm in want:
        if arm == "pixel160":
            X = np.load(P.OUT / "pixels160.npy", mmap_mode="r")
            d_in = PATCH_D
        else:
            f = CACHE / f"tok_{arm}_s0.npy"
            if not f.exists():
                print(f"  [skip] {arm}: {f.name} absent (not trained yet)",
                      flush=True)
                continue
            X = np.load(f, mmap_mode="r")[rows].reshape(-1, D.N_TOK, D.D_MODEL)
            d_in = D.D_MODEL
        print(f"  == {arm} ==", flush=True)
        t0 = time.time()
        pred = np.zeros_like(occ, np.float32)
        taps = []
        for k, te in enumerate(folds):
            tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
            pred[te], tap = E.run_fold(X, occ, tr, te, D.N_TOK, d_in, pos_w)
            taps.append(tap)
            print(f"    fold {k + 1}/{len(folds)} ({time.time() - t0:.0f}s)",
                  flush=True)
        bank(arm, {"n_tok": D.N_TOK, "d": d_in,
                   **E.cluster_boot(pred, occ, rows_by_ep,
                                    np.random.default_rng(E.SEED)),
                   "topk_loc_err_m": round(E.topk_loc_err(pred, occ), 3),
                   "train_ap_mean": round(float(np.mean(taps)), 4),
                   "train_s": round(time.time() - t0, 1)})
        np.save(SP / f"e_dense_pred_{arm}.npy", pred)
        del X
        gc.collect()
    print(f"\n-> {res_p}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
