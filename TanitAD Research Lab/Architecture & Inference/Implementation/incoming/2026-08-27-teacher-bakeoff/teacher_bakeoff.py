"""P2 — the fallback-teacher bake-off: DINOv3 vs V-JEPA2, paired on one rig.

Answers the PI's open decision 1 ("DINOv3 vs V-JEPA 2.1 as the distill/fallback
teacher; default measure both"). Provenance: official facebook/vjepa2-vitl
(no official 2.1 exists on HF — two probes, third-party re-uploads rejected).

⭐ EVERYTHING IS IMPORTED FROM `e_trunk2_probe` — folds, dual (Gram) ridge with
inner-split λ, bootstrap, target loading — so the two teachers are scored by the
IDENTICAL instrument that produced the banked DINOv3 numbers, on the identical
5,617 frames and episode-disjoint folds. The only new code is the V-JEPA2 arm
export (mirroring `export_dino`, including the row-order REFUSAL) and the paired
delta bootstrap (same episode resample applied to both arms' OOF predictions).

TIER: T0-DIAGNOSTIC. Decodability, never capability.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch

SP = pathlib.Path(__file__).resolve().parent
OLDSP = pathlib.Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
                     r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
                     r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(OLDSP))
import e_trunk2_probe as T  # noqa: E402

OUT = SP / "teacher_bakeoff.json"


def export_vjepa2():
    md = json.loads((SP / "vjepa2_fields/meta.json").read_text(encoding="utf-8"))
    keys = [tuple(k) for k in
            json.loads((T.FEAT / "keys.json").read_text(encoding="utf-8"))]
    order = [(c, f) for c, v in md["clips"].items() for f in v["frames"]]
    if order != keys:
        raise RuntimeError("V-JEPA2 row order != probe key order — refusing a "
                           "silently misaligned arm (the export_dino rule)")
    n = len(keys)
    d_grid, d_emb = md["clips"][order[0][0]]["shape"][1:]
    dst = T.FEAT / "vjepa2_tokens.npy"
    if dst.exists():
        print("vjepa2_tokens already exported")
        return
    tok = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float16,
                                    shape=(n, d_grid * d_emb))
    i = 0
    for cid in md["clips"]:
        a = np.load(SP / f"vjepa2_fields/{cid}.npy")
        k = a.shape[0]
        tok[i:i + k] = a.reshape(k, -1).astype(np.float16)
        i += k
    tok.flush()
    assert i == n
    print(f"vjepa2_tokens {tok.shape}")


def gram_for(name):
    gp = T.FEAT / f"gram_{name}.npy"
    if gp.exists():
        return np.load(gp)
    G = T.gram_memmap(T.FEAT / f"{name}.npy")
    np.save(gp, np.asarray(G))
    return np.asarray(G)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    export_vjepa2()
    keys = [tuple(k) for k in
            json.loads((T.FEAT / "keys.json").read_text(encoding="utf-8"))]
    ep = [k[0] for k in keys]
    folds = T.episode_folds(ep)

    tgt = {}
    for line in T.TARGETS.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            tgt[(r["clip_id"], int(r["frame_idx"]))] = r
    names = sorted({k for r in tgt.values() for k in r
                    if k not in ("clip_id", "frame_idx")
                    and k not in T.DEGENERATE})
    print(f"targets: {names}")

    arms = {}
    for arm, fn in (("dino_tokens", "dino_tokens.npy"),
                    ("vjepa2_tokens", "vjepa2_tokens.npy"),
                    ("C-PIXEL", "c_pixel.npy")):
        if not (T.FEAT / fn).exists():
            print(f"[skip] {arm} ({fn} absent)")
            continue
        print(f"gram {arm} … ({time.time()-t0:.0f}s)", flush=True)
        G = gram_for(fn[:-4]) if arm != "C-PIXEL" else None
        if G is None:
            X = np.load(T.FEAT / fn).astype(np.float64)
            G = T.gram(X)
        arms[arm] = G

    out = {"_evidence_class": "MEASURED (ours; dev-box, e_trunk2 instrument "
                              "imported verbatim)",
           "eval_tier": "T0-DIAGNOSTIC",
           "provenance": "facebook/vjepa2-vitl-fpc64-256 official; no official "
                         "2.1 on HF (2 probes, 2026-08-27)",
           "n_frames": len(keys), "n_folds": T.N_FOLDS, "targets": {}}
    preds = {}
    for tn in names:
        y, mask = [], []
        for k in keys:
            v = tgt.get(k, {}).get(tn)
            mask.append(v is not None)
            y.append(v if v is not None else 0.0)
        y = np.asarray(y, float)
        m = np.asarray(mask)
        if m.sum() < 500:
            out["targets"][tn] = {"skipped": f"n={int(m.sum())} too small"}
            continue
        binary = set(np.unique(y[m])) <= {0.0, 1.0}
        row = {"n": int(m.sum()), "binary": binary, "arms": {}}
        for arm, G in arms.items():
            Gm = G[np.ix_(m.nonzero()[0], m.nonzero()[0])]
            epm = [e for e, keep in zip(ep, m) if keep]
            fm = T.episode_folds(epm)
            p, _lam = T.dual_ridge_oof(Gm, y[m], epm, fm)
            stat = T.r2(p, y[m]) if not binary else T.auc(p, y[m])
            row["arms"][arm] = {"score": round(float(stat), 4)}
            preds[(tn, arm)] = (p, y[m], epm, binary)
        # paired dino - vjepa2 on the SAME episode resamples
        if ("dino_tokens" in row["arms"]) and ("vjepa2_tokens" in row["arms"]):
            pd_, yd, epd, binary = preds[(tn, "dino_tokens")]
            pv, yv, epv, _ = preds[(tn, "vjepa2_tokens")]
            assert epd == epv and np.array_equal(yd, yv)
            eps_u = sorted(set(epd))
            idx_of = {e: np.array([i for i, q in enumerate(epd) if q == e])
                      for e in eps_u}
            rng = np.random.default_rng(0)
            score = T.auc if binary else T.r2
            ds = np.empty(2000)
            for b in range(2000):
                pick = rng.integers(0, len(eps_u), len(eps_u))
                j = np.concatenate([idx_of[eps_u[q]] for q in pick])
                ds[b] = score(pd_[j], yd[j]) - score(pv[j], yv[j])
            lo, hi = np.percentile(ds, [2.5, 97.5])
            row["paired_dino_minus_vjepa2"] = {
                "delta": round(float(score(pd_, yd) - score(pv, yv)), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "separated": bool(lo > 0 or hi < 0)}
        out["targets"][tn] = row
        print(f"  {tn:<16} " + "  ".join(
            f"{a}:{row['arms'][a]['score']:+.4f}" for a in row["arms"])
            + ("  Δ(d−v):" + str(row.get("paired_dino_minus_vjepa2", {})
                                 .get("delta")) if "paired_dino_minus_vjepa2"
               in row else ""), flush=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
