"""MM-E6 step 0 — ⛔ IS ``P`` ABLE TO MEASURE THE SCENE AT ALL?

⭐ WHY THIS RUNS BEFORE ANY READ. MM-E6's whole result is
``share_self = r_res^2 / (r_env^2 + r_res^2)``. If ``P`` has no out-of-sample power
then ``z_env -> const``, ``z_res -> z_t``, and share_self -> **1.0000 for every arm,
by construction, whatever the truth is**. MEASURED in the 24-clip smoke: P's
out-of-sample R^2 was **-0.0003** and share_self read **0.9995** — a
SELF-DOMINATED verdict produced entirely by an impotent projection. That is the
CLAUDE.md probe rule in its exact form: *a probe panel carries a control that must
read a known value, and a linear oracle's failure is a statement about the linear
map, not about the representation.*

This script answers, at PRODUCTION scale and with ZERO tuning on the scored split:

  1. how z's variance splits into BETWEEN-clip and WITHIN-clip parts — the ceiling
     any cross-clip map can reach;
  2. what a linear ridge scene -> z reaches in-FIT and OUT-OF-SAMPLE, at several
     scene-basis ranks, with lambda always selected on a CLIP-DISJOINT inner split
     of FIT;
  3. the per-lambda out-of-sample curve — REPORTED, NEVER SELECTED ON (selecting
     lambda on the scored split is failure #3 of the four in CLAUDE.md, and it
     reads +0.0000 with a zero-width CI);
  4. the SCENE-SHUFFLED P through the identical path — the matched null. If the
     true P does not beat it, P is not measuring the scene;
  5. a NONLINEAR P (random Fourier features on the same frozen scene basis, then
     the same ridge) — because a negative from a LINEAR probe is not a negative
     about the representation, and the programme's own instrument for exactly this
     question is RFF+ridge.

TIER: T0-DIAGNOSTIC. MEASURED (ours; dev-box CPU).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

WORK = pathlib.Path(__file__).resolve().parent
OLD = pathlib.Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
                   r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
                   r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
MIRROR = pathlib.Path(r"C:\Users\Admin\tanitad-wt\stack")
sys.path.insert(0, str(OLD / "sp2"))
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(MIRROR))

import mm_e6_drift_decompose as M          # noqa: E402  (path set above)

RANKS = [int(x) for x in os.environ.get("PDIAG_RANKS", "32,96,256").split(",")]
OUT = pathlib.Path(os.environ.get("PDIAG_OUT", str(WORK / "mm_e6_pdiag.json")))
ARM = os.environ.get("MME6_ARMS", "postrain30k").split(",")[0]
D_RFF = int(os.environ.get("PDIAG_DRFF", "2048"))


def r2(pred_clips, Z_clips, zm):
    num = sum(float(((p - Z) ** 2).sum()) for p, Z in zip(pred_clips, Z_clips))
    den = sum(float(((Z - zm) ** 2).sum()) for Z in Z_clips)
    return 1.0 - num / max(den, 1e-12)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    import tanitad
    print(f"  tanitad from {tanitad.__file__}", flush=True)
    import rangeprobe_rff as RF
    import v7tiny_g2 as G

    dev = torch.device("cpu")
    clips = sorted(M.CORPUS.glob("*.v2ep.pt"))
    if M.N_CLIPS_ALL:
        clips = clips[:M.N_CLIPS_ALL]
    have = [c for c in clips if (M.DINO / f"{c.stem}.npy").is_file()]
    scored, fit = have[:M.N_SCORED], have[M.N_SCORED:]
    print(f"\n  MM-E6 P-DIAGNOSTIC · arm {ARM} · SCORED {len(scored)} / "
          f"FIT {len(fit)} clips · ranks {RANKS}\n", flush=True)

    w, st = G.load_arm(ARM, dev)
    Z, mlen = {}, {}
    with torch.no_grad():
        for c in have:
            d = torch.load(c, map_location="cpu", weights_only=False)
            yaw = np.asarray(d["poses"], dtype=np.float64)[:, 2]
            z, act, spd = G.encode_clip(w, c, dev, M.F)
            zt = z.float().numpy()
            nsc = int(np.load(M.DINO / f"{c.stem}.npy", mmap_mode="r").shape[0])
            m = min(len(zt) - M.K, len(act) - M.K, len(spd) - M.K,
                    len(yaw) - M.K - 1, nsc)
            if m < 30:
                continue
            Z[c.stem] = zt[:m].astype(np.float64)
            mlen[c.stem] = m
    del w
    sc_ids = [c.stem for c in scored if c.stem in Z]
    fi_ids = [c.stem for c in fit if c.stem in Z]
    Zsc = [Z[i] for i in sc_ids]
    Zfit = [Z[i] for i in fi_ids]

    # ---- 1. where z's variance lives -------------------------------------
    allz = np.concatenate(Zsc)
    gm = allz.mean(0, keepdims=True)
    tot = float(((allz - gm) ** 2).sum())
    within = sum(float(((x - x.mean(0, keepdims=True)) ** 2).sum()) for x in Zsc)
    rep = {"_evidence_class": "MEASURED (ours; dev-box CPU)",
           "eval_tier": "T0-DIAGNOSTIC", "arm": ARM, "step": int(st),
           "n_scored": len(sc_ids), "n_fit": len(fi_ids),
           "n_rows_scored": int(sum(mlen[i] for i in sc_ids)),
           "z_variance": {"between_clip_frac": round(1 - within / tot, 4),
                          "within_clip_frac": round(within / tot, 4)},
           "ranks": {}}
    print(f"  z variance: BETWEEN-clip {1 - within / tot:.4f} · "
          f"WITHIN-clip {within / tot:.4f}")
    print(f"  ⇒ a cross-clip map that transfers nothing scores at most "
          f"R2 {1 - within / tot:.4f} on the between part\n", flush=True)

    LAM = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
    for R in RANKS:
        t0 = time.time()
        mu_s, V_s = M.scene_basis(fi_ids, mlen, False, R)
        A_fit = [(M.load_scene(i, mlen[i], False) - mu_s) @ V_s for i in fi_ids]
        sd = np.concatenate(A_fit).std(0, keepdims=True) + 1e-8
        A_fit = [(x / sd).astype(np.float64) for x in A_fit]
        A_sc = [(((M.load_scene(i, mlen[i], False) - mu_s) @ V_s) / sd
                 ).astype(np.float64) for i in sc_ids]
        row = {}
        for tag, Af, Zf in (("P", A_fit, Zfit),):
            Wp, zm, lam, r2f = M.fit_P(Af, Zf, LAM, tag=f" rank{R}")
            row["lambda"] = lam
            row["in_fit_R2"] = round(r2f, 4)
            row["oos_R2"] = round(r2([A @ Wp + zm for A in A_sc], Zsc, zm), 4)
            # per-lambda OOS curve — REPORTED, NEVER SELECTED ON
            A = np.concatenate(Af)
            Zc = np.concatenate(Zf)
            zm2 = Zc.mean(0, keepdims=True)
            sc_l = float(np.trace(A.T @ A)) / max(A.shape[1], 1)
            curve = {}
            for l2 in LAM:
                Wl = M.ridge_multi(A, Zc - zm2, sc_l, l2)
                curve[f"{l2:g}"] = round(
                    r2([Aa @ Wl + zm2 for Aa in A_sc], Zsc, zm2), 4)
            row["oos_R2_per_lambda_REPORTED_NOT_SELECTED"] = curve
        # scene-shuffled matched null through the identical path
        rr = np.random.default_rng(11)
        n_f = len(fi_ids)
        perm = rr.permutation(n_f)
        for _ in range(200):
            if not any(perm == np.arange(n_f)):
                break
            perm = rr.permutation(n_f)
        A_sh, Z_sh = [], []
        for j in range(n_f):
            mm = min(len(A_fit[perm[j]]), len(Zfit[j]))
            A_sh.append(A_fit[perm[j]][:mm])
            Z_sh.append(Zfit[j][:mm])
        Wq, zmq, lamq, r2q = M.fit_P(A_sh, Z_sh, LAM, tag=f" rank{R} SHUFFLED")
        row["shuffled_in_fit_R2"] = round(r2q, 4)
        row["shuffled_oos_R2"] = round(r2([A @ Wq + zmq for A in A_sc], Zsc, zmq), 4)

        # ---- NONLINEAR P: RFF on the same frozen scene basis ---------------
        rng = np.random.default_rng(7)
        gam = None
        sub = np.concatenate(A_fit)
        ss = sub[rng.choice(len(sub), size=min(400, len(sub)), replace=False)]
        d2 = ((ss[:, None, :] - ss[None, :, :]) ** 2).sum(-1)
        gam = max(np.sqrt(np.median(d2[d2 > 0])), 1e-6)
        Wr = rng.standard_normal((R, D_RFF)) / gam
        br = rng.uniform(0, 2 * np.pi, size=D_RFF)
        phi = lambda X: np.sqrt(2.0 / D_RFF) * np.cos(X @ Wr + br)   # noqa: E731
        Pf = [phi(x) for x in A_fit]
        Ps = [phi(x) for x in A_sc]
        Wn, zmn, lamn, r2n = M.fit_P(Pf, Zfit, LAM, tag=f" rank{R} RFF-NONLINEAR")
        row["rff_lambda"] = lamn
        row["rff_in_fit_R2"] = round(r2n, 4)
        row["rff_oos_R2"] = round(r2([p @ Wn + zmn for p in Ps], Zsc, zmn), 4)
        Wns, zmns, _, _ = M.fit_P([phi(a) for a in A_sh], Z_sh, LAM,
                                  tag=f" rank{R} RFF SHUFFLED")
        row["rff_shuffled_oos_R2"] = round(
            r2([p @ Wns + zmns for p in Ps], Zsc, zmns), 4)
        row["elapsed_s"] = round(time.time() - t0, 1)
        rep["ranks"][str(R)] = row
        print(f"    rank {R}: LINEAR oos R2 {row['oos_R2']:+.4f} "
              f"(shuffled {row['shuffled_oos_R2']:+.4f}) · "
              f"RFF oos R2 {row['rff_oos_R2']:+.4f} "
              f"(shuffled {row['rff_shuffled_oos_R2']:+.4f}) · "
              f"{row['elapsed_s']}s\n", flush=True)
        OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        del A_fit, A_sc, Pf, Ps, V_s

    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
