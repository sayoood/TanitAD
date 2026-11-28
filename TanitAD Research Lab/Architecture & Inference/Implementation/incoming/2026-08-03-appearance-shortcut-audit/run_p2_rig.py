"""D-APPEAR P2 — APPEARANCE SHORTCUT vs CAMERA GEOMETRY across the two PhysicalAI rigs.

Pre-registration: ``Project Steering/PREREG_APPEARANCE_SHORTCUT.md`` §2 (the four discriminators
G1-G4 and their OPPOSITE predictions were written down before this ran).

THE PROBLEM
    The measured cross-rig collapse is frozen-v1 speed R2 **+0.930 -> -2.465**
    (``…/incoming/2026-07-22-idm-proof/results.json``). It is currently attributed to CAMERA
    GEOMETRY, and that explanation has independent support: PhysicalAI AV front-wide has TWO
    rigs (cy ~543 rig A / cy ~755 rig B) and a geometric-centre crop is ~215 px wrong for rig B.
    An APPEARANCE SHORTCUT predicts the same drop. Two explanations, one observation.

THE DISCRIMINATORS (each measures something the two hypotheses disagree about)
    G1  the HORIZON ROW of the cached frames, rig A vs rig B.
        geometry: an offset must be VISIBLE IN THIS SUBSTRATE.  appearance: no offset needed.
        ⚠️ Measured from the pixels, not read from the build config -- the programme's own
        rule (a code reading is not a measurement) applied to its own cache.
    G2  cross-rig drop of a MOTION-ENERGY arm vs a STILL-FRAME arm.
        geometry: both drop similarly (same grid, same shift).
        appearance: the still arm drops much more; optical-flow magnitude -> speed is physics
        and should transfer modulo scale.
    G3  a SYNTHETIC VERTICAL SHIFT of the held-out frames, swept.
        geometry: reproduces a large part of the collapse.  appearance: reproduces little.
    G4  WITHIN-rig (A->A, B->B) vs CROSS-rig (B->A, A->B), paired on the same held-out rows.

⚠️ If the discriminators point in different directions the registered answer is
   "CANNOT SEPARATE" and that is what gets reported.

usage:
  OMP_NUM_THREADS=6 python run_p2_rig.py --n-boot 2000 --out results_p2_rig.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))
sys.path.insert(0, str(REPO / "taniteval"))

RIG_A = Path(r"C:/Users/Admin/tanitad-data/eval/dappear_rigA.pt")
RIG_B = Path(r"C:/Users/Admin/tanitad-data/eval/dappear_rigB.pt")
EPC_TRAIN = Path(r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74")
EPC_VAL = Path(r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836")
SPEED_J = 0
SEL_TOL, MIN_SKILL = 0.005, 0.01
MEAN_KEY = (None, float("inf"))
RBF_GAMMA_MULTS = (0.25, 1.0, 4.0)
SHIFTS_PX = (0, 4, 8, 16, 24, 32, 48)          # 256-space rows; G3 sweep
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


ARMS = (
    ("pix32_centre_rbf", "pix32", "centre", "rbf"),     # STILL FRAME  (appearance)
    ("mot8_window_rbf",  "mot8",  "window", "rbf"),     # MOTION ENERGY (physics)
    ("v1_window",        "Z",     "window", "linear"),  # the learned latent
)


def split_episodes(eps, hold_every=3):
    return ([e for i, e in enumerate(eps) if i % hold_every != 0],
            [e for i, e in enumerate(eps) if i % hold_every == 0])


def stack_sub(eps, key):
    X = torch.cat([e[key] for e in eps]).float()
    if X.ndim == 2:
        X = X[:, None, :]
    S = torch.cat([e["S"] for e in eps]).float()
    eid = np.concatenate([np.full(e["n"], e["name"]) for e in eps])
    return X, S.numpy().astype(np.float64), eid


def fit_predict(AP, Xfit, yfit, Xsel, ysel, Xful, yful, tests, *, feat, kernel, device):
    """Fit ONE arm on (fit/sel/full) and predict SPEED on every tensor in ``tests``."""
    Xf = AP.window_features(Xfit, feat)
    Xs = AP.window_features(Xsel, feat)
    XF = AP.window_features(Xful, feat)
    Xt = [AP.window_features(t, feat) for t in tests]
    Xf, Xs = AP.standardize(Xf, Xs)
    XF, *Xt = AP.standardize(XF, *Xt)
    mu, sd = float(yful.mean()), float(max(yful.std(), 1e-6))
    kw = dict(kernel=kernel, matmul_device=device, matmul_dtype=torch.float32)
    alphas = AP.DualRidge.alpha_grid(2, -4, 10)
    gms = RBF_GAMMA_MULTS if kernel == "rbf" else (None,)
    sel_p, test_p = {}, {}
    for gm in gms:
        gamma = None
        if kernel == "rbf":
            g = torch.Generator().manual_seed(20260803)
            idx = torch.randperm(len(Xf), generator=g)[:1000]
            d2 = torch.cdist(Xf[idx].double(), Xf[idx].double()).pow(2)
            gamma = gm / max(float(d2[d2 > 0].median()), 1e-9)
        yv = torch.from_numpy(((yfit - mu) / sd)[:, None])
        inner = AP.DualRidge(Xf, yv.double(), gamma=gamma, **kw)
        for al in alphas:
            sel_p[(gm, al)] = inner.predict(Xs, al).numpy().ravel() * sd + mu
        del inner
        yV = torch.from_numpy(((yful - mu) / sd)[:, None])
        full = AP.DualRidge(XF, yV.double(), gamma=gamma, **kw)
        for al in alphas:
            test_p[(gm, al)] = [full.predict(t, al).numpy().ravel() * sd + mu for t in Xt]
        del full
    k0 = next(iter(sel_p))
    sel_p[MEAN_KEY] = np.full_like(sel_p[k0], mu)
    test_p[MEAN_KEY] = [np.full(len(t), mu) for t in Xt]
    scored = {k: AP.r2_score(v, ysel) for k, v in sel_p.items()}
    best = max(scored.values())
    k_shrunk = max(scored, key=lambda k: k[1])
    if best < MIN_SKILL:
        k_sel, rule = k_shrunk, "skill_gate_to_train_mean"
    else:
        ok = [k for k, s in scored.items() if s >= best - SEL_TOL]
        k_sel, rule = max(ok, key=lambda k: k[1]), "one_se_tiebreak"
    return test_p[k_sel], {"gamma_mult": k_sel[0],
                           "alpha": ("inf(exact_train_mean)" if np.isinf(k_sel[1])
                                     else k_sel[1]),
                           "sel_r2": round(float(scored[k_sel]), 5),
                           "best_sel_r2": round(float(best), 5), "rule": rule,
                           "n_features": int(Xf.shape[1])}


# --------------------------------------------------------------------------- #
# G1 — the horizon row, MEASURED from the pixels of the substrate actually used #
# --------------------------------------------------------------------------- #
def horizon_profile(eps_meta, n_eps=25, n_frames=40):
    """Mean |d(luma)/d(row)| profile over frames -> the row band where the horizon sits.

    The horizon is the strongest persistent HORIZONTAL edge in a forward-facing driving
    frame, so the row-gradient profile peaks there. Averaging over clips removes scene
    content and leaves the geometry.
    """
    LUMA = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    acc, n = None, 0
    for m in eps_meta[:n_eps]:
        base = EPC_TRAIN if m["name"].startswith("0_") else EPC_VAL
        idx = int(m["name"].split("_")[1])
        e = torch.load(base / f"ep_{idx:05d}.pt", map_location="cpu", weights_only=False)
        fr = e["frames_u8"][:n_frames, 6:9].float() / 255.0
        g = (fr * LUMA).sum(1)                                # [n, 256, 256]
        prof = (g[:, 1:] - g[:, :-1]).abs().mean(-1).mean(0)  # [255]
        acc = prof if acc is None else acc + prof
        n += 1
    prof = (acc / max(n, 1)).numpy()
    rows = np.arange(len(prof))
    return {"n_episodes": n, "n_frames_each": n_frames,
            "argmax_row": int(prof.argmax()),
            "centroid_row": round(float((prof * rows).sum() / prof.sum()), 3),
            "profile": [round(float(x), 6) for x in prof]}


def shifted_pix(eps_meta, shift_rows: int, side: int = 32, k: int = 4,
                basis: str = "pix"):
    """Rebuild the pix32 window substrate with the frames rolled DOWN by ``shift_rows``.

    ⚠️ ``torch.roll`` wraps rather than pads. On a 256-row frame a 48-row roll moves 19 % of
    the image from the bottom to the top, which is a HARSHER perturbation than a real rig
    offset (which would crop, not wrap). That direction of error is deliberate: G3 asks
    whether a geometric shift can reproduce the collapse, so an over-strong shift that still
    fails to reproduce it is the stronger evidence.
    """
    import idm_head as ih
    LUMA = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    offs = torch.arange(-k, k + 1)
    out = []
    for m in eps_meta:
        base = EPC_TRAIN if m["name"].startswith("0_") else EPC_VAL
        idx = int(m["name"].split("_")[1])
        e = torch.load(base / f"ep_{idx:05d}.pt", map_location="cpu", weights_only=False)
        x = e["frames_u8"][:, 6:9].float() / 255.0
        if shift_rows:
            x = torch.roll(x, shifts=int(shift_rows), dims=2)
        g = (x * LUMA).sum(1, keepdim=True)
        if basis == "mot":                   # motion energy: |dI| at FULL res, pool AFTER
            d = torch.zeros_like(g)
            d[:-1] = (g[1:] - g[:-1]).abs()
            g = d
        g = torch.nn.functional.avg_pool2d(g, g.shape[-1] // side).reshape(g.shape[0], -1)
        t = m["t"]
        out.append(g[t[:, None] + offs[None, :]])
    return torch.cat(out).float()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_p2_rig.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-g3", action="store_true")
    ap.add_argument("--stage", default="all", choices=("all", "g3"))
    a = ap.parse_args()

    import tanitad.eval.accel_probe as AP
    import tanitad.eval.ap_ci as APCI

    A = torch.load(RIG_A, map_location="cpu", weights_only=False)["episodes"]
    B = torch.load(RIG_B, map_location="cpu", weights_only=False)["episodes"]
    trA, hoA = split_episodes(A)
    trB, hoB = split_episodes(B)
    log(f"rig A {len(A)} eps ({len(trA)}/{len(hoA)}), rig B {len(B)} eps "
        f"({len(trB)}/{len(hoB)})")
    res = {"meta": {
        "prereg": "Project Steering/PREREG_APPEARANCE_SHORTCUT.md#2",
        "rig_A_episodes": len(A), "rig_B_episodes": len(B),
        "rig_A_cy_mean": round(float(np.mean([e["cy"] for e in A])), 2),
        "rig_B_cy_mean": round(float(np.mean([e["cy"] for e in B])), 2),
        "n_boot": a.n_boot,
        "estimator": "paired episode-cluster bootstrap (tanitad/eval/ap_ci.py)"}}

    if a.stage == "g3" and Path(a.out).exists():
        res = json.load(open(a.out))                  # merge into the existing verdicts
        log(f"stage=g3: merging into {a.out}")

    # ---------------- G1 ---------------- #
    if a.stage == "all":
        log("G1: measuring the horizon row from the pixels")
        pA, pB = horizon_profile(hoA), horizon_profile(hoB)
        res["G1_horizon_row"] = {
            "rig_a": {k: v for k, v in pA.items() if k != "profile"},
            "rig_b": {k: v for k, v in pB.items() if k != "profile"},
            "argmax_offset_rows_256space": pB["argmax_row"] - pA["argmax_row"],
            "centroid_offset_rows_256space": round(pB["centroid_row"] - pA["centroid_row"], 3),
            "interpretation": "a LEGACY geometric-centre crop would put rig B's horizon ~215 px "
                              "off in ORIGINAL 1920x1080 pixels; the D-016 R1 per-clip "
                              "principal-point crop puts both rigs' horizon at the same output "
                              "row. This measures which one this cache actually is.",
            "profiles": {"rig_a": pA["profile"], "rig_b": pB["profile"]}}
        log(f"G1 horizon argmax rows: A={pA['argmax_row']} B={pB['argmax_row']} "
            f"(offset {pB['argmax_row']-pA['argmax_row']}), centroids "
            f"{pA['centroid_row']:.1f} / {pB['centroid_row']:.1f}")

    # ---------------- G2 + G4 ---------------- #
    res.setdefault("G2_G4_transfer", {})
    for arm, key, feat, kernel in (ARMS if a.stage == "all" else ()):
        rec = {"feature": feat, "kernel": kernel, "substrate": key, "cells": {}}
        for src_name, src_tr in (("A", trA), ("B", trB)):
            fit_eps = [e for i, e in enumerate(src_tr) if i % 3 != 0]
            sel_eps = [e for i, e in enumerate(src_tr) if i % 3 == 0]
            Xf, yf, _ = stack_sub(fit_eps, key)
            Xs, ys, _ = stack_sub(sel_eps, key)
            XF, yF, _ = stack_sub(src_tr, key)
            XhA, yhA, eA = stack_sub(hoA, key)
            XhB, yhB, eB = stack_sub(hoB, key)
            preds, sel = fit_predict(
                AP, Xf, yf[:, SPEED_J], Xs, ys[:, SPEED_J], XF, yF[:, SPEED_J],
                [XhA, XhB], feat=feat, kernel=kernel, device=a.device)
            for tgt, p, yt, et in (("A", preds[0], yhA[:, SPEED_J], eA),
                                   ("B", preds[1], yhB[:, SPEED_J], eB)):
                r2 = AP.r2_score(p, yt)
                rec["cells"][f"{src_name}->{tgt}"] = {
                    "r2_speed": round(r2, 5),
                    "r2_speed_ci": APCI.stat_episode_cluster_bootstrap(
                        (lambda pp, gg: (lambda s: AP.r2_score(pp[s], gg[s])))(p, yt),
                        et, n_boot=a.n_boot, name=f"r2_speed_{src_name}_to_{tgt}"),
                    "mae": round(float(np.abs(p - yt).mean()), 5),
                    "corr": (round(float(np.corrcoef(p, yt)[0, 1]), 5)
                             if p.std() > 0 else None),
                    "within_rig": src_name == tgt, "selection": sel}
                rec.setdefault("_preds", {})[f"{src_name}->{tgt}"] = p
            log(f"G2 {arm:18s} src={src_name}  A:{rec['cells'][src_name+'->A']['r2_speed']:+.4f}"
                f"  B:{rec['cells'][src_name+'->B']['r2_speed']:+.4f}")
        # PAIRED drop on the SAME held-out rows: cross-rig minus within-rig
        for tgt, y_t, e_t in (("A", yhA[:, SPEED_J], eA), ("B", yhB[:, SPEED_J], eB)):
            src_x = "B" if tgt == "A" else "A"
            pw = rec["_preds"][f"{tgt}->{tgt}"]
            px = rec["_preds"][f"{src_x}->{tgt}"]
            rec[f"paired_cross_minus_within_on_{tgt}"] = \
                APCI.paired_stat_episode_cluster_bootstrap(
                    (lambda p, g: (lambda s: AP.r2_score(p[s], g[s])))(px, y_t),
                    (lambda p, g: (lambda s: AP.r2_score(p[s], g[s])))(pw, y_t),
                    e_t, n_boot=a.n_boot, name=f"{src_x}to{tgt}_minus_{tgt}to{tgt}")
        rec.pop("_preds")
        res["G2_G4_transfer"][arm] = rec
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))

    # ---------------- G3 ---------------- #
    if not a.skip_g3:
        log("G3: synthetic vertical-shift sweep on the held-out frames")
        fit_eps = [e for i, e in enumerate(trA) if i % 3 != 0]
        sel_eps = [e for i, e in enumerate(trA) if i % 3 == 0]
        Xf, yf, _ = stack_sub(fit_eps, "pix32")
        Xs, ys, _ = stack_sub(sel_eps, "pix32")
        XF, yF, _ = stack_sub(trA, "pix32")
        _, yhA, eA = stack_sub(hoA, "pix32")
        res["G3_vertical_shift"] = {
            "note": "torch.roll WRAPS; a real rig offset would crop. The perturbation is "
                    "therefore STRONGER than the geometric one it stands in for, so a shift "
                    "that still fails to reproduce the collapse is the stronger evidence.",
            "reference_scale": "a legacy geometric-centre crop is ~215 px off for rig B in "
                               "1920x1080; scaled into a ~533 px crop resized to 256 that is "
                               "~100 output ROWS -- beyond the top of this sweep.",
            "arms": {}}
        # RUN IT ON AN ARM THAT HAS SKILL TO LOSE. The still-frame arm is at the null on
        # PhysicalAI in BOTH rigs, so a degradation cannot be measured on it; that cell is
        # reported VOID rather than as "geometry does not matter".
        for aname, key_, basis, side, feat_ in (
                ("pix32_centre_rbf", "pix32", "pix", 32, "centre"),
                ("mot8_window_rbf",  "mot8",  "mot",  8, "window")):
            Xf, yf, _ = stack_sub(fit_eps, key_)
            Xs, ys, _ = stack_sub(sel_eps, key_)
            XF, yF, _ = stack_sub(trA, key_)
            _, yhA, eA = stack_sub(hoA, key_)
            tests = [shifted_pix(hoA, sh, side=side, basis=basis) for sh in SHIFTS_PX]
            preds, sel = fit_predict(AP, Xf, yf[:, SPEED_J], Xs, ys[:, SPEED_J],
                                     XF, yF[:, SPEED_J], tests,
                                     feat=feat_, kernel="rbf", device=a.device)
            base = AP.r2_score(preds[0], yhA[:, SPEED_J])
            sweep = []
            for sh, pr in zip(SHIFTS_PX, preds):
                r2 = AP.r2_score(pr, yhA[:, SPEED_J])
                sweep.append({"shift_rows": sh, "r2_speed": round(r2, 5),
                              "frac_of_baseline_lost": (None if base < 0.05 else
                                                        round(float((base - r2) / base), 4))})
            res["G3_vertical_shift"]["arms"][aname] = {
                "fitted_on": "rig A train, UNSHIFTED", "selection": sel,
                "baseline_r2_shift0": round(base, 5), "VOID": bool(base < 0.05),
                "void_reason": (None if base >= 0.05 else
                                "the arm has no skill at shift 0 on this corpus, so a "
                                "degradation cannot be measured on it"),
                "sweep": sweep}
            log(f"G3 {aname}: base {base:+.4f} -> "
                f"{[r['r2_speed'] for r in sweep]}")

    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
