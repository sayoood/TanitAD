"""RL2 — the C104 readout ladder on REF-C's OWN encoder latents (val40, banked).

⛔ EVAL TIER: T0-DIAGNOSTIC. Linear readability of a frozen latent is a
representation diagnostic — NEVER driving performance, and (C123) it does NOT
predict driving. This run answers exactly one question: what scene state does
REF-C's deployed pooled vision surface carry linearly, on the same canonical
881 val40 windows its driving numbers live on?

WHAT IS PROBED (banked, vision-only by construction — the dump's provenance
separates `pooled`/`pooled_seq`/`ctx` from the ego+nav `measurement` echo path):
  refc_base_pooled       [881,  704]  mean-pooled conv map, LAST window frame —
                                      the surface route_head / maneuver_head /
                                      goal_head read at inference.
  refc_xl_pooled         [881,  992]  the same for REF-C-XL.
  refc_base_pooled_seq   [881, 5632]  all 8 window frames, flattened — the
                                      temporal surface StrategicCtx consumes.
  refc_xl_pooled_seq     [881, 7936]  the same for XL.

PROTOCOL — the ER10 ladder's, applied verbatim where it applies:
  * every arm projected to EXACTLY RP_DIM=2048 features by the SAME fixed
    Gaussian RP (`er10_pool_ladder.make_projection`, seeded on
    [PROJ_SEED_BASE, seed, arm]), 5 seeds, spread reported (C103);
  * z-score by probe-train stats, ones column appended;
  * ridge via `pc6_linear_readout.ridge_fit(intercept_col=-1)` (C92 repair,
    gated at startup), alpha grid chosen on an inner episode split;
  * primary quantity r2_ceiling = corr(pred, truth)^2; C97 K1_DEGENERATE stamp
    whenever pred_sd/gt_sd < 0.10; every row also carries the v0-partial
    correlation (C92 — a headline was an ego-speed echo once already);
  * estimator: taniteval.ci episode-cluster bootstrap over the EVAL episodes;
    paired deltas via `er10_pool_ladder.paired_delta_r2c` (raw AND v0-partial).
    `overlapping_holdout_se` is never imported.

SPLIT: episode-disjoint, declared, fixed — EVEN eids probe-train, ODD eids
eval (20/20 episodes). No window is selected; the banked 881 set is read
verbatim; parity untouched.

CONTROLS, per arm (the brief's traps):
  PLANT   positive control — the standardized target written into the RAW
          features along a fixed random direction at amp × feature_sd BEFORE
          the RP. Proves the instrument can find a signal of known size in
          THIS feature matrix at this n/d/alpha (C119: evaluate the
          instrument where the answer is known).
  NOISE   matched-random null — features replaced by N(mu, sd) per column
          (the D1 "random vectors" floor).
  YPERM   episode-block permutation of the target (train AND eval) — the
          no-signal floor for the FIT machinery.
  V0PROXY the ego-speed scalar as the only feature — the trivial-proxy readout
          every vision number must clear (C92).

Window identity: rl1_targets.py gated it (G1..G5); this file re-asserts the
row count and eid partition against the targets npz and refuses on drift.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[5]
_INC = _REPO / "TanitAD Research Hub/Architecture & Inference/Implementation/incoming"
for _p in (_REPO / "taniteval", _REPO / "stack",
           _INC / "2026-08-18-pooling-ladder-ER10/code",
           _INC / "2026-08-17-probe-positive-control/code",
           _INC / "2026-08-17-slot-probe-parity/code",
           _INC / "2026-08-17-latent-linear-ladder/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import er10_pool_ladder as EP                                    # noqa: E402
import ll1_ladder as LL                                          # noqa: E402
from pc6_linear_readout import ridge_fit                         # noqa: E402

RP_DIM = EP.RP_DIM                       # 2048 — the ER10 dimension, unchanged
PLANT_SEED = [EP.PROJ_SEED_BASE, 424242]
NOISE_SEED = 12345
YPERM_SEED = 777
# target -> the scale it is divided by before planting (ER10 ORACLE_TARGETS
# convention, extended to the two rungs ER10 did not plant)
PLANT_SCALE = {"ego_v0": 10.0, "lead_gap": 15.0, "lead_closing": 2.0,
               "lead_present": 1.0, "n_agents_grid": 5.0, "n_agents_any": 8.0}
TARGETS = ("ego_v0", "lead_present", "lead_gap", "lead_closing",
           "n_agents_grid", "n_agents_any")


def load_arms(a) -> dict:
    base = torch.load(a.latents, map_location="cpu", weights_only=False)
    xl = torch.load(a.latents_xl, map_location="cpu", weights_only=False)
    arms = {
        "refc_base_pooled": base["pooled"].numpy().astype(np.float64),
        "refc_xl_pooled": xl["pooled"].numpy().astype(np.float64),
        "refc_base_pooled_seq":
            base["pooled_seq"].numpy().reshape(base["pooled_seq"].shape[0], -1)
            .astype(np.float64),
        "refc_xl_pooled_seq":
            xl["pooled_seq"].numpy().reshape(xl["pooled_seq"].shape[0], -1)
            .astype(np.float64),
    }
    meta = {"base_ckpt": str(base.get("ckpt")),
            "base_step": int(base.get("ckpt_step", -1)),
            "xl_ckpt": str(xl.get("ckpt")),
            "xl_step": int(xl.get("ckpt_step", -1)),
            "raster": list(base.get("raster", ())),
            "grid_shape": list(base.get("grid_shape", ())),
            "provenance": str(base.get("provenance", ""))[:400]}
    return arms, meta, np.asarray(base["eid"], dtype=np.int64)


def project(X: np.ndarray, seed: int, arm_idx: int, device) -> np.ndarray:
    """[n, D] -> [n, RP_DIM] through the ER10 fixed Gaussian RP (n_units=1)."""
    P = EP.make_projection(1, X.shape[1],
                           [EP.PROJ_SEED_BASE, seed, arm_idx], device)
    Xt = torch.from_numpy(X.astype(np.float32)).to(device).half()
    out = (Xt @ P).float().cpu().numpy().astype(np.float64)
    del P
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def run_fit(X, y, ok, eid, tr_mask_ep, alphas, inner_frac, seed, n_boot,
            v0, sd_floor, binary=False):
    """One arm x one target x one (already-projected) feature matrix."""
    tr = tr_mask_ep[eid] & ok
    ev = (~tr_mask_ep[eid]) & ok
    Xtr, Xev = X[tr], X[ev]
    ytr, yev = y[tr], y[ev]
    ctr, cev = eid[tr], eid[ev]
    eev = eid[ev]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Ztr = np.concatenate([(Xtr - mu) / sd, np.ones((Xtr.shape[0], 1))], 1)
    Zev = np.concatenate([(Xev - mu) / sd, np.ones((Xev.shape[0], 1))], 1)
    v0ev = v0[ev]
    out, pred = EP.fit_one(Ztr, ytr, ctr, Zev, yev, eev, cev, alphas,
                           inner_frac, seed, n_boot, v0ev, sd_floor, icol=-1)
    out["n_train"] = int(tr.sum())
    out["n_eval"] = int(ev.sum())
    if binary:
        auc = LL.auc_binary(pred, yev)
        out["auc"] = round(auc, 4) if auc is not None else None
    return out, pred, ev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--latents-xl", required=True)
    ap.add_argument("--targets-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--proj-seeds", type=int, nargs="+",
                    default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ap.add_argument("--inner-frac", type=float, default=0.25)
    ap.add_argument("--sd-ratio-floor", type=float, default=0.10)
    ap.add_argument("--plant-amp", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    t0 = time.time()

    # ⛔ C92 gate — verbatim ER10's: at alpha 1e9 the repaired solve predicts
    # the mean, the penalised one predicts ~0. Refuses to run otherwise.
    _Xg = np.concatenate([np.random.default_rng(7).normal(size=(40, 3)),
                          np.ones((40, 1))], 1)
    _yg = np.random.default_rng(8).normal(size=40) + 5.0
    _wp = ridge_fit(_Xg, _yg, 1e9, intercept_col=None)
    _wu = ridge_fit(_Xg, _yg, 1e9, intercept_col=-1)
    gate = {"alpha": 1e9, "y_mean": round(float(_yg.mean()), 6),
            "penalised_pred": round(float((_Xg @ _wp).mean()), 6),
            "unpenalised_pred": round(float((_Xg @ _wu).mean()), 6)}
    if abs(gate["unpenalised_pred"] - gate["y_mean"]) > 1e-4:
        raise SystemExit(f"[rl2] C92 intercept gate FAILED: {gate}")

    dev = torch.device(a.device if (a.device == "cpu"
                                    and True) or torch.cuda.is_available()
                       else "cpu")
    arms, arm_meta, eid_lat = load_arms(a)
    tz = np.load(a.targets_npz, allow_pickle=True)
    eid = tz["eid"].astype(np.int64)
    if not np.array_equal(eid, eid_lat):
        raise SystemExit("[rl2] eid drift between targets npz and latents")
    v0 = tz["v0"].astype(np.float64)
    n = eid.size
    if n != 881:
        raise SystemExit(f"[rl2] expected 881 rows, got {n}")
    n_eps = int(eid.max()) + 1
    tr_mask_ep = (np.arange(n_eps) % 2 == 0)          # EVEN train / ODD eval

    ys = {t: (tz[f"y_{t}"].astype(np.float64), tz[f"ok_{t}"].astype(bool))
          for t in TARGETS}

    res = {"_evidence_class":
           "MEASURED (ours; ladder over BANKED REF-C latents on the canonical "
           "881 val40 windows — a T0 representation diagnostic, never driving "
           "performance; C123: linear readability does NOT predict driving)",
           "eval_tier": "T0-DIAGNOSTIC",
           "experiment": "RL — REF-C encoder readout ladder "
                         "(2026-08-18-refc-architecture-review JOB 2)",
           "windows": "canonical val40 881 (identity gates: rl_gates.json "
                      "G1..G5; NO episode selected — parity untouched)",
           "arm_meta": arm_meta,
           "split": {"rule": "EVEN eid -> probe-train, ODD eid -> eval",
                     "n_train_eps": int(tr_mask_ep.sum()),
                     "n_eval_eps": int((~tr_mask_ep).sum())},
           "rp_dim": RP_DIM, "rp_seed_base": EP.PROJ_SEED_BASE,
           "proj_seeds": a.proj_seeds, "alphas": a.alphas,
           "inner_frac": a.inner_frac, "n_boot": a.n_boot,
           "sd_ratio_floor": a.sd_ratio_floor,
           "ridge_intercept_col": -1, "ridge_intercept_gate": gate,
           "solve_source": "pc6_linear_readout.ridge_fit (imported)",
           "fit_source": "er10_pool_ladder.fit_one / make_projection / "
                         "paired_delta_r2c (imported, not re-implemented)",
           "estimator": "taniteval.ci episode-cluster bootstrap "
                        "(paired for deltas)",
           "forbidden": "overlapping_holdout_se",
           "plant": {"amp_rel_feature_sd": a.plant_amp,
                     "scales": PLANT_SCALE, "seed": PLANT_SEED},
           "device": str(dev), "arms": {}}

    arm_names = sorted(arms)
    preds: dict = {}
    for ai, name in enumerate(arm_names):
        Xraw = arms[name]
        arec = {"d_raw": int(Xraw.shape[1]), "targets": {}}
        # feature sd for the plant, from the probe-train rows only
        f_sd = float(Xraw[tr_mask_ep[eid]].std())
        arec["feature_sd_train"] = round(f_sd, 6)
        for s in a.proj_seeds:
            Xp = project(Xraw, s, ai, dev)
            for t in TARGETS:
                y, ok = ys[t]
                row, pred, ev = run_fit(
                    Xp, y, ok, eid, tr_mask_ep, a.alphas, a.inner_frac, s,
                    a.n_boot, v0, a.sd_ratio_floor,
                    binary=(t == "lead_present"))
                arec["targets"].setdefault(t, {})[f"seed{s}"] = row
                preds[(name, t, s)] = (pred, ev)
            del Xp
        # ---- controls at seed 0 ------------------------------------------
        s = a.proj_seeds[0]
        # NOISE: matched-random features (D1 floor)
        Xp = project(Xraw, s, ai, dev)
        g = np.random.default_rng(NOISE_SEED)
        mu_r = Xp[tr_mask_ep[eid]].mean(0)
        sd_r = np.maximum(Xp[tr_mask_ep[eid]].std(0), 1e-12)
        Xn = g.normal(mu_r, sd_r, Xp.shape)
        for t in TARGETS:
            y, ok = ys[t]
            row, _, _ = run_fit(Xn, y, ok, eid, tr_mask_ep, a.alphas,
                                a.inner_frac, s, a.n_boot, v0,
                                a.sd_ratio_floor)
            arec["targets"][t]["NOISE"] = row
        del Xn
        # YPERM: episode-block permutation of the target
        gp = np.random.default_rng(YPERM_SEED)
        perm = gp.permutation(n_eps)
        for t in TARGETS:
            y, ok = ys[t]
            # permute episode blocks: window w of episode e reads episode
            # perm[e]'s value at the same within-episode rank when defined,
            # else the episode median — a no-signal floor that preserves the
            # per-episode autocorrelation structure.
            yp = np.empty_like(y)
            okp = np.zeros_like(ok)
            for e in range(n_eps):
                src, dst = perm[e], e
                iv_src = np.flatnonzero((eid == src) & ok)
                iv_dst = np.flatnonzero(eid == dst)
                if iv_src.size == 0:
                    continue
                take = iv_src[np.arange(iv_dst.size) % iv_src.size]
                yp[iv_dst] = y[take]
                okp[iv_dst] = True
            okp &= ok
            row, _, _ = run_fit(Xp, yp, okp, eid, tr_mask_ep, a.alphas,
                                a.inner_frac, s, a.n_boot, v0,
                                a.sd_ratio_floor)
            arec["targets"][t]["YPERM"] = row
        # PLANT: the standardized target written into the RAW features along a
        # fixed direction, then the SAME projection — instrument power.
        gd = np.random.default_rng(PLANT_SEED)
        u = gd.standard_normal(Xraw.shape[1])
        u /= np.linalg.norm(u)
        for t in TARGETS:
            y, ok = ys[t]
            code = np.where(ok, y / PLANT_SCALE[t], 0.0)
            Xpl = Xraw + (a.plant_amp * f_sd) * np.outer(code, u)
            Xq = project(Xpl, s, ai, dev)
            row, _, _ = run_fit(Xq, y, ok, eid, tr_mask_ep, a.alphas,
                                a.inner_frac, s, a.n_boot, v0,
                                a.sd_ratio_floor)
            arec["targets"][t]["PLANT"] = row
            del Xq, Xpl
        del Xp
        res["arms"][name] = arec
        print(f"[rl2] {name} done  {time.time()-t0:.0f} s", flush=True)

    # ---- V0PROXY (arm-independent) --------------------------------------
    prox = {"targets": {}}
    Xv = v0.reshape(-1, 1)
    for t in TARGETS:
        y, ok = ys[t]
        row, pred, ev = run_fit(Xv, y, ok, eid, tr_mask_ep, a.alphas,
                                a.inner_frac, a.proj_seeds[0], a.n_boot, v0,
                                a.sd_ratio_floor,
                                binary=(t == "lead_present"))
        prox["targets"][t] = row
    res["v0_proxy"] = prox

    # ---- paired deltas (r2_ceiling), raw AND v0-partial -------------------
    pairs = [("refc_base_pooled", "refc_xl_pooled"),
             ("refc_base_pooled", "refc_base_pooled_seq"),
             ("refc_xl_pooled", "refc_xl_pooled_seq")]
    res["paired_deltas"] = {}
    for A, B in pairs:
        for t in TARGETS:
            y, ok = ys[t]
            key = f"{A}__minus__{B}__{t}"
            per_seed = {}
            for s in a.proj_seeds:
                pa, eva = preds[(A, t, s)]
                pb, evb = preds[(B, t, s)]
                assert np.array_equal(eva, evb)
                yev = y[eva]
                eev = eid[eva]
                v0ev = v0[eva]
                per_seed[f"seed{s}"] = {
                    "raw": EP.paired_delta_r2c(pa, pb, yev, eev, a.n_boot),
                    "partial_v0": (None if t == "ego_v0" else
                                   EP.paired_delta_r2c(pa, pb, yev, eev,
                                                       a.n_boot, z=v0ev))}
            res["paired_deltas"][key] = per_seed

    res["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1), "utf-8")
    print(f"[rl2] -> {a.out}  ({res['wall_s']} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
