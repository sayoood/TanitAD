"""⛔ THE C106 ADVERSARIAL RE-READ — an attempt to KILL "random init beats the
trained encoder 3.6x", not to confirm it.

C106 (`Project Steering/RETRACTION_LOG.md`) claims our trained ViT5Encoder at
S-W step 11250 reads `ego_v0` and `lead_gap` ~3.6x WORSE than its own random
initialisation, and concludes the objective is SUBTRACTING linearly readable
geometry. That is the most consequential claim in the programme right now, so it
gets attacked before it is believed.

FOUR DEFECTS IN THE PRODUCING MEASUREMENT, EACH ADDRESSED BY A FLAG HERE
  1. ⛔ THE 3.6x CARRIES NO DECISION-GRADE INTERVAL. C106 quotes
     `0.1894 [0.1736, 0.2011]` — that bracket is the spread over three
     PROJECTION SEEDS, i.e. the instrument's own noise, NOT an episode-cluster
     bootstrap over the 70 eval clusters. CLAUDE.md: "never quote an interval
     without its estimator". ⇒ `--stage delta` computes the PRE-REGISTERED
     statistic `paired_delta_r2c` (er10's own, `taniteval.ci._draws`) on
     Δ r2_ceiling BETWEEN THE TWO CACHES, on the identical eval windows —
     the comparison C106 made by eye.
  2. ⛔ THE ALPHA WAS PINNED AT THE GRID EDGE for our arm on every seed and both
     rungs (`alpha_chosen = 1e7`, `alpha_at_grid_edge: true`). A sweep that
     never bracketed its optimum has not selected a readout. ⇒ `--alphas` is
     widened here; `alpha_at_grid_edge` is re-emitted so the reader can see
     whether it un-pinned.
  3. ⚠️ THE INNER SPLIT IS ONE DRAW. `fit_one` picks alpha on a single
     25%-of-clips inner split at `ridge_seed = 0`, and our arm's inner-MAE curve
     is NON-MONOTONE across four decades — a noisy selector. ⇒ `--ridge-seeds`
     re-draws the inner split; a finding that moves with the split is a
     selection artifact.
  4. ⛔ NEITHER ARM PASSES K1, so the 3.6x is a ratio between two failing arms.
     ⇒ `--randomise-features` runs the matched-random NULL through the identical
     path so both arms can be placed against the floor, and `--oracle local2`
     runs PC-2OBJ so the instrument must demonstrate it can SEE a signal on the
     very tokens it reports nothing on.

⛔ NOTHING IS RE-IMPLEMENTED. `build_features`, `fit_one`, `paired_delta_r2c`,
`POOL_ARMS` and the target loader are IMPORTED from `er10_pool_ladder`, which is
the producer of the numbers under attack. An adversarial re-read that swaps in
its own solver cannot separate "the claim is wrong" from "my code is different".

⛔ PARITY: SELECTS NOTHING. Reads banked window caches verbatim.
TIER: T0-DIAGNOSTIC on the 130-episode probe corpus — never driving performance.
ESTIMATOR: paired episode-cluster bootstrap. `overlapping_holdout_se` is never
imported.
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
_REPO = _HERE.parents[6]
_INC = _REPO / "TanitAD Research Hub/Architecture & Inference/Implementation/incoming"
for _p in (_REPO / "taniteval", _REPO / "stack",
           _INC / "2026-08-17-probe-positive-control/code",
           _INC / "2026-08-17-slot-probe-parity/code",
           _INC / "2026-08-17-latent-linear-ladder/code",
           _INC / "2026-08-18-pooling-ladder-ER10/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import er10_pool_ladder as ER                                    # noqa: E402
import ll1_ladder as LL                                          # noqa: E402

WIDE_ALPHAS = [1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7,
               1e8, 1e9, 1e10, 1e11, 1e12, 1e13]
BASE_ALPHAS = [1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]


# --------------------------------------------------------------------------- #
def stage_fit(a) -> int:
    dev = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available())
                       else "cpu")
    t0 = time.time()
    blob = torch.load(a.cache, map_location="cpu", weights_only=False)
    rows, meta = blob["rows"], blob["meta"]
    if rows[0].get("tokens") is None:
        raise SystemExit("[c106] ⛔ this cache banked no tokens")
    th, tw = int(meta["token_grid"][0]), int(meta["token_grid"][1])
    d_model = int(rows[0]["tokens"].shape[-1])
    print(f"[c106] {a.tag}: {len(rows)} rows grid {th}x{tw} d_model {d_model} "
          f"({time.time()-t0:.0f} s)", flush=True)

    decl = json.loads(Path(a.split_json).read_text("utf-8"))
    ev_c, tr_c = set(decl["eval_clips"]), set(decl["train_clips"])
    idx_tr = [i for i, r in enumerate(rows) if r["clip_id"] in tr_c]
    idx_ev = [i for i, r in enumerate(rows) if r["clip_id"] in ev_c]
    sub = [rows[i] for i in idx_tr + idx_ev]
    pos_tr = np.arange(len(idx_tr))
    pos_ev = np.arange(len(idx_tr), len(sub))

    ctr_all = np.array([sub[i]["clip_id"] for i in pos_tr])
    cev_all = np.array([sub[i]["clip_id"] for i in pos_ev])
    eev_all = np.array([sub[i]["episode_uid"] for i in pos_ev])
    v0_all = np.array([float(r["v0"]) for r in sub])
    # ⛔ the identity key: two caches may only be differenced if they carry the
    # SAME windows in the SAME order. Asserted in `--stage delta`, not assumed.
    rowkey = np.array([f"{r['clip_id']}#{int(r['frame_idx'])}" for r in sub])

    # PC-2OBJ needs the planted code; the oracle path also needs ego targets.
    oracle_u = oracle_R = token_sd = None
    amp = 0.0
    ego: dict = {}
    pose_off = 0
    if a.oracle is not None:
        clips_used = {r["clip_id"] for r in rows}
        ego = LL.load_ego(Path(a.episodes_dir), Path(a.join_file), clips_used)
        align = LL.bind_pose_grid(rows, ego)
        pose_off = align["pose_index_offset"]
        smp = np.concatenate([sub[i]["tokens"].float().numpy().reshape(-1)
                              for i in range(0, len(sub), max(1, len(sub) // 64))])
        token_sd = float(smp.std())
        oracle_R = (np.random.default_rng([ER.PROJ_SEED_BASE, 999])
                    .standard_normal((d_model, len(ER.ORACLE_TARGETS)))
                    .astype(np.float32))
        oracle_R /= np.sqrt(len(ER.ORACLE_TARGETS))
        oracle_u = np.zeros((len(sub), len(ER.ORACLE_TARGETS)), dtype=np.float32)
        for j, r in enumerate(sub):
            for q, (tname, sc) in enumerate(ER.ORACLE_TARGETS):
                v, ok = LL.target_of(r, ego, tname, pose_off)
                oracle_u[j, q] = (v / sc) if ok else 0.0
        amp = a.oracle_amp * token_sd
        print(f"[c106] ORACLE {a.oracle} token_sd={token_sd:.6f} amp={amp:.6f}",
              flush=True)

    tvals = {}
    for tname in a.targets:
        vv, ok = zip(*[LL.target_of(r, ego, tname, pose_off) for r in sub])
        tvals[tname] = (np.array(vv, dtype=np.float64), np.array(ok))

    feats, n_units = ER.build_features(sub, "p40", a.proj_seeds, th, tw,
                                       d_model, dev, a.oracle, amp,
                                       oracle_u, oracle_R)
    if a.randomise_features is not None:
        g = np.random.default_rng(a.randomise_features)
        for s in list(feats):
            Xr = feats[s]
            mu_r, sd_r = Xr[pos_tr].mean(0), Xr[pos_tr].std(0)
            feats[s] = g.normal(mu_r, np.maximum(sd_r, 1e-12), Xr.shape)

    res = {
        "_evidence_class": "MEASURED (ours; C106 ADVERSARIAL re-read of a frozen "
                           "banked token cache — T0 WM diagnostic, never driving "
                           "performance)",
        "eval_tier": "T0-DIAGNOSTIC",
        "posture": "REFUTATION — this run exists to try to kill C106",
        "tag": a.tag, "cache": str(a.cache),
        "arm": "p40 (the DEPLOYED AvgPool2d((4,10)))",
        "token_grid": [th, tw], "d_model": d_model, "n_units": n_units,
        "n_raw_features": n_units * d_model, "rp_dim": ER.RP_DIM,
        "alphas": a.alphas, "alpha_grid_name": a.alpha_grid_name,
        "proj_seeds": a.proj_seeds, "ridge_seeds": a.ridge_seeds,
        "inner_frac": a.inner_frac,
        "ridge_intercept_col": -1,
        "randomise_features_seed": a.randomise_features,
        "oracle": a.oracle, "oracle_amp_rel": a.oracle_amp,
        "oracle_token_sd": token_sd,
        "sd_ratio_floor": a.sd_ratio_floor,
        "estimator": "taniteval.ci.paired_episode_cluster_bootstrap",
        "forbidden": "overlapping_holdout_se",
        "solve_source": "er10_pool_ladder.fit_one / build_features (IMPORTED "
                        "from the producer, not re-implemented)",
        "parity": "SELECTS NOTHING — banked window set read verbatim",
        "n_train_windows_all": len(idx_tr), "n_eval_windows_all": len(idx_ev),
        "src_run_stamp": meta.get("run_stamp"),
        "src_encoder_meta": meta.get("encoder_meta"),
        "targets": {}}

    preds_out: dict = {}
    for tname in a.targets:
        y, ok = tvals[tname]
        mtr, mev = ok[pos_tr], ok[pos_ev]
        ytr, yev = y[pos_tr][mtr], y[pos_ev][mev]
        ctr, cev = ctr_all[mtr], cev_all[mev]
        eev = eev_all[mev]
        v0ev = None if tname == "ego_v0" else v0_all[pos_ev][mev]
        cells = {}
        for s in a.proj_seeds:
            X = feats[s]
            Xtr, Xev = X[pos_tr][mtr], X[pos_ev][mev]
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            Ztr = np.concatenate([(Xtr - mu) / sd,
                                  np.ones((Xtr.shape[0], 1))], 1)
            Zev = np.concatenate([(Xev - mu) / sd,
                                  np.ones((Xev.shape[0], 1))], 1)
            for rs in a.ridge_seeds:
                rec, pr = ER.fit_one(Ztr, ytr, ctr, Zev, yev, eev, cev,
                                     a.alphas, a.inner_frac, rs, a.n_boot,
                                     v0ev, a.sd_ratio_floor, -1)
                cells[f"p{s}|r{rs}"] = rec
                preds_out[f"{tname}|p{s}|r{rs}"] = pr.astype(np.float64)
        r2s = [c["r2_ceiling"] for c in cells.values()]
        edge = [bool(c["alpha_at_grid_edge"]) for c in cells.values()]
        res["targets"][tname] = {
            "unit": LL.UNITS[tname], "rung": LL.RUNG[tname],
            "n_train": int(mtr.sum()), "n_eval": int(mev.sum()),
            "n_eval_clusters": int(len(np.unique(eev))),
            "gt_mean": round(float(yev.mean()), 4),
            "gt_sd": round(float(yev.std()), 4),
            "r2_ceiling_mean": round(float(np.mean(r2s)), 5),
            "r2_ceiling_sd": round(float(np.std(r2s)), 5),
            "r2_ceiling_min": round(float(np.min(r2s)), 5),
            "r2_ceiling_max": round(float(np.max(r2s)), 5),
            "n_cells": len(cells),
            "ANY_ALPHA_AT_GRID_EDGE": bool(any(edge)),
            "n_alpha_at_grid_edge": int(sum(edge)),
            "alphas_chosen": sorted({float(c["alpha_chosen"])
                                     for c in cells.values()}),
            "K1_PASS_count": int(sum(bool(c["K1_PASSES"])
                                     for c in cells.values())),
            "pred_sd_over_gt_sd_range": [
                round(float(min(c["pred_sd_over_gt_sd"] for c in cells.values())), 4),
                round(float(max(c["pred_sd_over_gt_sd"] for c in cells.values())), 4)],
            "per_cell": cells}
        # the target vectors ride with the predictions so `delta` needs nothing else
        preds_out[f"__y__{tname}"] = yev
        preds_out[f"__eid__{tname}"] = eev.astype(str)
        preds_out[f"__key__{tname}"] = rowkey[pos_ev][mev]
        if v0ev is not None:
            preds_out[f"__v0__{tname}"] = v0ev
        t = res["targets"][tname]
        print("  %-14s n=%4d/%2d r2c=%.5f+-%.5f edge=%d/%d alphas=%s K1pass=%d "
              "psd/gsd=%s" % (tname, t["n_eval"], t["n_eval_clusters"],
                              t["r2_ceiling_mean"], t["r2_ceiling_sd"],
                              t["n_alpha_at_grid_edge"], t["n_cells"],
                              [f"{v:g}" for v in t["alphas_chosen"]],
                              t["K1_PASS_count"], t["pred_sd_over_gt_sd_range"]),
              flush=True)

    res["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str), "utf-8")
    if a.out_preds:
        Path(a.out_preds).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.out_preds, **preds_out)
    print(f"[c106] wrote {a.out} ({res['wall_s']} s)", flush=True)
    return 0


# --------------------------------------------------------------------------- #
def stage_delta(a) -> int:
    """⭐ THE MISSING INTERVAL. Δ r2_ceiling between two caches on the IDENTICAL
    eval windows, under the pre-registered paired episode-cluster bootstrap."""
    A = np.load(a.preds_a, allow_pickle=False)
    B = np.load(a.preds_b, allow_pickle=False)
    out = {"_evidence_class": "MEASURED (ours; paired episode-cluster bootstrap "
                              "on Δ r2_ceiling between two frozen encoders on "
                              "the SAME banked windows)",
           "eval_tier": "T0-DIAGNOSTIC",
           "estimator": "paired_episode_cluster_bootstrap (taniteval.ci "
                        "_draws/episode_index) on Δ corr² — er10's own "
                        "`paired_delta_r2c`, imported",
           "forbidden": "overlapping_holdout_se",
           "arm_a": a.name_a, "arm_b": a.name_b,
           "sign_convention": f"delta > 0 means {a.name_a} reads BETTER than "
                              f"{a.name_b}",
           "n_boot": a.n_boot, "targets": {}}
    tnames = sorted({k[len("__y__"):] for k in A.files if k.startswith("__y__")})
    for tname in tnames:
        y, eid = A[f"__y__{tname}"], A[f"__eid__{tname}"]
        # ⛔ IDENTITY GATE — a paired statistic on mismatched windows is void.
        if not np.array_equal(A[f"__key__{tname}"], B[f"__key__{tname}"]):
            raise SystemExit(f"[c106] ⛔ {tname}: the two caches do NOT carry "
                             f"the same eval windows — refusing to pair")
        if not np.allclose(y, B[f"__y__{tname}"]):
            raise SystemExit(f"[c106] ⛔ {tname}: target vectors differ")
        v0 = A[f"__v0__{tname}"] if f"__v0__{tname}" in A.files else None
        cells = sorted(k.split("|", 1)[1] for k in A.files
                       if k.startswith(f"{tname}|"))
        per, per_v0 = {}, {}
        for c in cells:
            ka, kb = f"{tname}|{c}", f"{tname}|{c}"
            if kb not in B.files:
                continue
            per[c] = ER.paired_delta_r2c(A[ka], B[kb], y, eid, a.n_boot)
            if v0 is not None:
                per_v0[c] = ER.paired_delta_r2c(A[ka], B[kb], y, eid, a.n_boot,
                                                z=v0)
        if not per:
            continue
        d = [v["delta"] for v in per.values()]
        out["targets"][tname] = {
            "n_windows": per[cells[0]]["n_windows"],
            "n_episode_clusters": per[cells[0]]["n_episodes"],
            "delta_r2c_mean": round(float(np.mean(d)), 5),
            "delta_r2c_min": round(float(np.min(d)), 5),
            "delta_r2c_max": round(float(np.max(d)), 5),
            "ALL_CELLS_SEPARATED": bool(all(v["separated"] for v in per.values())),
            "ALL_CELLS_SEPARATED_AND_POSITIVE": bool(
                all(v["separated"] and v["delta"] > 0 for v in per.values())),
            "ALL_CELLS_SEPARATED_PARTIAL_V0": (
                bool(per_v0) and bool(all(v["separated"]
                                          for v in per_v0.values()))),
            "per_cell": per, "per_cell_partial_v0": per_v0}
        t = out["targets"][tname]
        print("  %-14s Δr2c=%+.5f [%+.5f,%+.5f over cells]  allsep=%s  "
              "allsep+pos=%s  n=%d/%d" %
              (tname, t["delta_r2c_mean"], t["delta_r2c_min"],
               t["delta_r2c_max"], t["ALL_CELLS_SEPARATED"],
               t["ALL_CELLS_SEPARATED_AND_POSITIVE"], t["n_windows"],
               t["n_episode_clusters"]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str), "utf-8")
    print(f"[c106] wrote {a.out}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fit", "delta"], required=True)
    # fit
    ap.add_argument("--cache")
    ap.add_argument("--tag", default="arm")
    ap.add_argument("--split-json")
    ap.add_argument("--episodes-dir")
    ap.add_argument("--join-file")
    ap.add_argument("--targets", nargs="+",
                    default=["ego_v0", "lead_gap", "lead_closing"])
    ap.add_argument("--alpha-grid", choices=["base", "wide"], default="wide")
    ap.add_argument("--proj-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ridge-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--inner-frac", type=float, default=0.25)
    ap.add_argument("--sd-ratio-floor", type=float, default=0.10)
    ap.add_argument("--randomise-features", type=int, default=None)
    ap.add_argument("--oracle", choices=["dist", "local", "local2"], default=None)
    ap.add_argument("--oracle-amp", type=float, default=1.0)
    ap.add_argument("--out-preds", default=None)
    ap.add_argument("--device", default="cuda")
    # delta
    ap.add_argument("--preds-a")
    ap.add_argument("--preds-b")
    ap.add_argument("--name-a", default="a")
    ap.add_argument("--name-b", default="b")
    # both
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    a.alpha_grid_name = a.alpha_grid
    a.alphas = WIDE_ALPHAS if a.alpha_grid == "wide" else BASE_ALPHAS
    if a.stage == "fit":
        for req in ("cache", "split_json", "episodes_dir", "join_file"):
            if not getattr(a, req):
                raise SystemExit(f"[c106] --{req.replace('_','-')} is required")
        return stage_fit(a)
    return stage_delta(a)


if __name__ == "__main__":
    sys.exit(main())
