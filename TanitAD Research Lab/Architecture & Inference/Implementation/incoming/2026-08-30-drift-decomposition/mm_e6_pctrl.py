"""MM-E6 step 0b — ⛔ IS THE SCENE->TARGET MACHINERY SOUND, OR IS MY PIPELINE BROKEN?

``mm_e6_pdiag.py`` measured P at production scale and found **no out-of-sample power
at any rank, linear or nonlinear** (oos R^2 <= +0.0010, and the SCENE-SHUFFLED P reads
the same). Two readings fit that observation and they have opposite consequences:

    (a) the pipeline is broken — my scene features, my splits or my ridge are wrong;
    (b) the pipeline is fine and z genuinely carries almost no scene information that
        this frozen representation can recover.

⛔ A CONTROL THAT MUST READ A KNOWN VALUE IS THE ONLY THING THAT SEPARATES THEM, and
declaring VOID without it would be an absence-claim from a single probe. So the SAME
scene features, the SAME clip-disjoint basis and the SAME RFF+ridge instrument are
pointed at targets whose answers are already banked:

    speed        ⭐ THE POSITIVE CONTROL. MODEL_REGISTRY 13.0d records frozen DINOv3
                 at **+0.4081** on speed and **+0.2754** on n_agents. If speed reads
                 near zero here, MY pipeline is broken and nothing else may be read.
    omega        yaw rate from the measured poses — a second scene-driven quantity.
    z_pca0..7    the latent's OWN top directions: how much of z does the scene explain?
    dz_pca0..7   ⭐⭐ DRIFT ITSELF, PREDICTED DIRECTLY FROM THE SCENE. This is MM-E6's
                 question asked without P: if the environment drives drift, the frozen
                 scene must predict dz. A near-zero here is an ENVIRONMENT-side
                 negative that does not depend on the projection working at all.
    constant     reads EXACTLY 0.0000.

Every read is within-clip Pearson r against a TIME-SHUFFLED null through the identical
code path, scored with the drift instrument's own clip-disjoint 10-fold. The scene PCA
basis is fit on the FIT clips only, so the SCORED clips are scored and never tuned on.

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

import mm_e6_drift_decompose as M          # noqa: E402

OUT = pathlib.Path(os.environ.get("PCTRL_OUT", str(WORK / "mm_e6_pctrl.json")))
ARM = os.environ.get("MME6_ARMS", "postrain30k").split(",")[0]
R = int(os.environ.get("MME6_RANK", "96"))
NB = int(os.environ.get("MME6_BANDS", "8"))


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


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
    print(f"\n  MM-E6 P-MACHINERY CONTROL · arm {ARM} · SCORED {len(scored)} / "
          f"FIT {len(fit)} clips · scene rank {R} · device {dev}\n", flush=True)

    w, st = G.load_arm(ARM, dev)
    Z, SP, OM, mlen = {}, {}, {}, {}
    with torch.no_grad():
        for c in have:
            d = torch.load(c, map_location="cpu", weights_only=False)
            yaw = np.asarray(d["poses"], dtype=np.float64)[:, 2]
            z, act, spd = G.encode_clip(w, c, dev, M.F)
            zt = z.float().numpy()
            v = spd.float().numpy().ravel().astype(np.float64)
            nsc = int(np.load(M.DINO / f"{c.stem}.npy", mmap_mode="r").shape[0])
            m = min(len(zt) - M.K, len(act) - M.K, len(v) - M.K,
                    len(yaw) - M.K - 1, nsc)
            if m < 30:
                continue
            i = np.arange(m)
            Z[c.stem] = zt[:m + M.K].astype(np.float64)
            SP[c.stem] = v[i][:, None]
            OM[c.stem] = (wrap(yaw[i + 1] - yaw[i]) / 0.1)[:, None]
            mlen[c.stem] = m
    del w
    sc_ids = [c.stem for c in scored if c.stem in Z]
    fi_ids = [c.stem for c in fit if c.stem in Z]

    # scene basis on FIT clips only -> the SCORED clips are never tuned on
    t0 = time.time()
    mu_s, V_s = M.scene_basis(fi_ids, mlen, False, R)
    A_fit = [(M.load_scene(i, mlen[i], False) - mu_s) @ V_s for i in fi_ids]
    sd = np.concatenate(A_fit).std(0, keepdims=True) + 1e-8
    X = [((((M.load_scene(i, mlen[i], False) - mu_s) @ V_s) / sd)
          ).astype(np.float64) for i in sc_ids]
    del A_fit
    print(f"    scene basis rank {R} fit on {len(fi_ids)} FIT clips "
          f"({time.time() - t0:.0f}s)", flush=True)

    Zsc = [Z[i][:mlen[i]] for i in sc_ids]
    DZ = [Z[i][M.K:M.K + mlen[i]] - Z[i][:mlen[i]] for i in sc_ids]

    def pca_targets(mats, nb):
        A = np.concatenate(mats)
        mu = A.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(A - mu, full_matrices=False)
        del A
        return [[(x - mu) @ Vt[j][:, None] for x in mats] for j in range(nb)]

    names, Ys = [], []
    names.append("speed  ⭐POSITIVE CONTROL (DINOv3 banked +0.4081)")
    Ys.append([SP[i] for i in sc_ids])
    names.append("omega (yaw rate)")
    Ys.append([OM[i] for i in sc_ids])
    for j, t in enumerate(pca_targets(Zsc, NB)):
        names.append(f"z_pca{j}   (latent content)")
        Ys.append(t)
    for j, t in enumerate(pca_targets(DZ, NB)):
        names.append(f"dz_pca{j}  ⭐⭐DRIFT FROM SCENE")
        Ys.append(t)
    names.append("constant (ctrl)")
    Ys.append([np.ones((len(x), 1)) for x in X])

    # time-shuffled null for every target, through the identical path
    Yall = list(Ys)
    for j, Y in enumerate(Ys):
        rj = np.random.default_rng(500 + j)
        Yall.append([y.ravel()[rj.permutation(len(y))][:, None] for y in Y])
    ny = len(Ys)

    worst = M.verify_multi(X, Yall, rff_multi, rff_fold)
    print(f"    ⛔ VECTORISATION GATE max|dev| {worst:.3e} "
          f"{'PASS' if worst < 1e-8 else 'FAIL'}", flush=True)
    if worst >= 1e-8:
        return 3

    # ⚠️ the CONSTANT target is a target of ones: within_clip_r must return 0.0
    S = M.kfold_clip_scores_multi(X, Yall, rff_multi, within_clip_r)
    TR, SH = S[:ny], S[ny:]
    rep = {"_evidence_class": "MEASURED (ours; dev-box CPU)",
           "eval_tier": "T0-DIAGNOSTIC", "arm": ARM, "step": int(st),
           "scene_rank": R, "n_scored_clips": len(sc_ids),
           "n_rows": int(sum(mlen[i] for i in sc_ids)),
           "input": "frozen DINOv3 4x8x1024 -> PCA rank R (basis fit on FIT clips)",
           "estimator": "clip-disjoint 10-fold RFF+ridge, within-clip Pearson r, "
                        "TIME-SHUFFLED null through the identical path",
           "targets": {}}
    print(f"\n    {'target':<48}{'r':>9}{'shuf':>9}{'r-shuf':>9}{'t':>8}")
    print("    " + "-" * 83)
    for j, nm in enumerate(names):
        d = TR[j] - SH[j]
        t = float(d.mean()) / max(float(d.std(ddof=1) / np.sqrt(len(d))), 1e-12)
        rep["targets"][nm] = {"r": round(float(TR[j].mean()), 4),
                              "r_time_shuffled": round(float(SH[j].mean()), 4),
                              "r_minus_shuffled": round(float(d.mean()), 4),
                              "t": round(t, 2)}
        print(f"    {nm:<48}{TR[j].mean():>+9.4f}{SH[j].mean():>+9.4f}"
              f"{d.mean():>+9.4f}{t:>8.2f}", flush=True)
    zb = np.array([rep["targets"][n]["r_minus_shuffled"] for n in names
                   if n.startswith("z_pca")])
    db = np.array([rep["targets"][n]["r_minus_shuffled"] for n in names
                   if n.startswith("dz_pca")])
    rep["summary"] = {"z_pca_mean_r_minus_shuffled": round(float(zb.mean()), 4),
                      "dz_pca_mean_r_minus_shuffled": round(float(db.mean()), 4),
                      "speed_r_minus_shuffled":
                          rep["targets"][names[0]]["r_minus_shuffled"]}
    print(f"\n    scene -> z (top-{NB} PCA) mean {zb.mean():+.4f} · "
          f"scene -> dz (top-{NB} PCA) mean {db.mean():+.4f}")
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
