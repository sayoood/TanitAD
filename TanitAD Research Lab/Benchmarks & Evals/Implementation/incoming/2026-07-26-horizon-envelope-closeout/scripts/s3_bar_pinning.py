#!/usr/bin/env python3
"""s3_bar_pinning.py — T3 (§9.9): PIN THE S3 SKILL BARS, or say they cannot be.

THE PROBLEM (MEASURED, `POD2_EVAL_HOST.md` §3.3.1)
--------------------------------------------------
S3's *labels* reproduce digit-for-digit across hosts — 12/12 strata, every window
and every cluster count. Its **skill bars do not**: on md5-identical code, an
identical corpus and the same seed, `S3 lateral` read **0.6534** on pod3 and
**0.6493** on pod2; `S3 longitudinal` **0.5323 → 0.5420**. `skill = QWK(model) −
bar`, and the bar is quoted to **four decimals**, so an arm scoring 0.6510 clears
one pod's bar and fails the other's. The standup labelled the mechanism a
`HYPOTHESIS` ("floating-point non-determinism in the BLAS/threading path") and
explicitly did not instrument it. This does.

THE BAR, EXACTLY
----------------
`run_s3_characterisation.run_firewall` fits a 2-layer MLP scene-blind
(`s3_blind_baseline._fit_mlp`: full-batch Adam, 400 epochs, `torch.manual_seed(0)`,
no dropout, no data shuffling) for each conditioning arm, then

    S3   bar = max(QWK(B1), QWK(B2), QWK(B3))     `operative_blind_floor`
    S3-W bar = QWK(B1)                            `B1_sensor_only`

Nothing in that path is *deliberately* stochastic once the seed is fixed. So a
moving bar has exactly three candidate causes, and they are separable:

  E-A  THREADS      — refit at OMP_NUM_THREADS in {1,2,4,8,16}, seed fixed.
                      CPU GEMM reduction order depends on thread count; 400 Adam
                      steps amplify a 1e-7 difference into a different argmax on
                      borderline rows. If the bar moves with thread count, the
                      mechanism is threading and PINNING THREADS PINS THE BAR.
  E-B  REPEATABILITY— refit 5x at ONE thread count, same seed, same process
                      lifetime and across processes. Tests whether the fit is
                      deterministic GIVEN the thread count.
  E-C  SEED         — refit at seeds 0..9, threads fixed. This is the bar's own
                      ESTIMATOR VARIANCE: even a perfectly reproducible pipeline
                      has it, and it is the honest precision the bar may be
                      quoted at. A bar pinned only by freezing seed 0 is pinned
                      by fiat, not measured.

The mining is done ONCE and cached (it costs ~41 min of MooseFS reads and is
provably deterministic — the standup reproduced every count across two hosts on
two different data surfaces), so every condition below re-runs only the FIT.

OUTPUT
------
Per condition: QWK for B1/B2/B3 on both axes, the two bars, and the exact float.
Then the pinning verdict: the number of decimal places at which every condition
agrees, computed rather than asserted.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ARMS = ("B1_sensor_only", "B2_plus_route", "B3_FULL_CONDITIONING")


def _import(s3dir, stack):
    for p in (s3dir, stack, str(Path(stack) / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import run_s3_characterisation as R
    import s3_labels as S3
    from s3_blind_baseline import _fit_mlp
    return R, S3, _fit_mlp


# --------------------------------------------------------------------------- #
# phase 1 — mine once, cache the feature matrices                              #
# --------------------------------------------------------------------------- #
def phase_mine(a):
    R, S3, _ = _import(a.s3dir, a.stack)
    t0 = time.time()
    tr_rows = R.mine_cache(a.train_cache, a.horizon_s, a.min_ttm_s, 0)
    te_rows = R.mine_cache(a.test_cache, a.horizon_s, a.min_ttm_s, 0)
    mine_s = round(time.time() - t0, 1)
    groups = ["sensor", "route", "vtarget"]
    blob = {"_mine_seconds": mine_s, "_horizon_s": a.horizon_s,
            "_min_ttm_s": a.min_ttm_s, "_n_bands": int(S3.N_BANDS),
            "_train_cache": a.train_cache, "_test_cache": a.test_cache}
    for axis in ("lat", "lon"):
        tr = [r for r in tr_rows if r.get(f"{axis}_admissible")]
        te = [r for r in te_rows if r.get(f"{axis}_admissible")]
        Xtr, names = R.build_X(tr, groups)
        Xte, _ = R.build_X(te, groups)
        blob[f"{axis}_Xtr"] = Xtr
        blob[f"{axis}_Xte"] = Xte
        blob[f"{axis}_ytr"] = np.array([r[f"band_{axis}"] for r in tr], np.int64)
        blob[f"{axis}_yte"] = np.array([r[f"band_{axis}"] for r in te], np.int64)
        blob[f"{axis}_eid_te"] = np.array([str(r["eid"]) for r in te])
        blob[f"{axis}_names"] = np.array(names)
    np.savez_compressed(a.cache, **blob)
    print(f"[mine] {mine_s}s -> {a.cache}", flush=True)
    for axis in ("lat", "lon"):
        print(f"   {axis}: n_train={blob[axis + '_ytr'].size} "
              f"n_test={blob[axis + '_yte'].size} "
              f"n_test_eps={len(set(blob[axis + '_eid_te'].tolist()))} "
              f"n_features={blob[axis + '_Xtr'].shape[1]}", flush=True)


# --------------------------------------------------------------------------- #
# phase 2 — ONE condition (one process, so OMP_NUM_THREADS is honoured)        #
# --------------------------------------------------------------------------- #
def phase_fit(a):
    R, S3, _fit_mlp = _import(a.s3dir, a.stack)
    import torch
    z = np.load(a.cache, allow_pickle=False)
    slices = {"B1_sensor_only": len(R.COND_GROUPS["sensor"]),
              "B2_plus_route": len(R.COND_GROUPS["sensor"])
              + len(R.COND_GROUPS["route"]),
              "B3_FULL_CONDITIONING": (len(R.COND_GROUPS["sensor"])
                                       + len(R.COND_GROUPS["route"])
                                       + len(R.COND_GROUPS["vtarget"]))}
    out = {"seed": a.seed,
           "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
           "torch_num_threads": int(torch.get_num_threads()),
           "torch_version": torch.__version__,
           "torch_num_interop": int(torch.get_num_interop_threads()),
           "host": os.uname().nodename, "axes": {}}
    for axis in ("lat", "lon"):
        Xtr, Xte = z[f"{axis}_Xtr"], z[f"{axis}_Xte"]
        ytr, yte = z[f"{axis}_ytr"], z[f"{axis}_yte"]
        q = {}
        for arm in ARMS:
            k = slices[arm]
            t = time.time()
            yhat, final_ce = _fit_mlp(Xtr[:, :k], ytr, Xte[:, :k],
                                      int(z["_n_bands"]), seed=a.seed)
            # ⚠️ S3.band_metrics ROUNDS qwk to 4 dp — which is exactly the
            # precision under test. Take the unrounded kappa.
            qk = float(S3.quadratic_weighted_kappa(np.asarray(yte, np.int64),
                                                   np.asarray(yhat, np.int64),
                                                   int(z["_n_bands"])))
            m = S3.band_metrics(yte, yhat)
            q[arm] = {"qwk": qk, "qwk_rounded_4dp": float(m["qwk"]),
                      "band_acc": float(m["band_acc"]),
                      "train_final_ce": float(final_ce),
                      "fit_s": round(time.time() - t, 1),
                      "pred_sha1": __import__("hashlib").sha1(
                          np.asarray(yhat, np.int64).tobytes()).hexdigest()[:16]}
        floor_arm = max(ARMS, key=lambda k: q[k]["qwk"])
        out["axes"][axis] = {
            "arms": q,
            "S3_bar": q[floor_arm]["qwk"], "S3_bar_arm": floor_arm,
            "S3_W_bar": q["B1_sensor_only"]["qwk"]}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[fit] seed={a.seed} OMP={out['omp_num_threads_env']} "
          f"torch_threads={out['torch_num_threads']} | "
          f"lat S3={out['axes']['lat']['S3_bar']:.6f} "
          f"S3W={out['axes']['lat']['S3_W_bar']:.6f} | "
          f"lon S3={out['axes']['lon']['S3_bar']:.6f} "
          f"S3W={out['axes']['lon']['S3_W_bar']:.6f} -> {a.out}", flush=True)


# --------------------------------------------------------------------------- #
# phase 3 — the design: spawn one SUBPROCESS per condition                     #
# --------------------------------------------------------------------------- #
def phase_sweep(a):
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    conds = []
    for t in (1, 2, 4, 8, 16):                       # E-A threads
        conds.append(("E-A_threads", t, 0, 0))
    for rep in range(5):                             # E-B repeatability
        conds.append(("E-B_repeat", 8, 0, rep))
    for s in range(10):                              # E-C seed
        conds.append(("E-C_seed", 8, s, 0))
    rows = []
    for i, (fam, thr, seed, rep) in enumerate(conds):
        tag = f"{fam}_t{thr}_s{seed}_r{rep}"
        o = outdir / f"cond_{tag}.json"
        env = dict(os.environ)
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[k] = str(thr)
        cmd = [sys.executable, os.path.abspath(__file__), "fit",
               "--cache", a.cache, "--seed", str(seed), "--out", str(o),
               "--s3dir", a.s3dir, "--stack", a.stack]
        t0 = time.time()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[sweep] {tag} FAILED:\n{r.stdout[-800:]}\n{r.stderr[-1500:]}",
                  flush=True)
            continue
        d = json.loads(o.read_text())
        rows.append({"family": fam, "threads": thr, "seed": seed, "rep": rep,
                     "wall_s": round(time.time() - t0, 1),
                     "torch_num_threads": d["torch_num_threads"],
                     "lat_S3": d["axes"]["lat"]["S3_bar"],
                     "lat_S3_arm": d["axes"]["lat"]["S3_bar_arm"],
                     "lat_S3W": d["axes"]["lat"]["S3_W_bar"],
                     "lon_S3": d["axes"]["lon"]["S3_bar"],
                     "lon_S3_arm": d["axes"]["lon"]["S3_bar_arm"],
                     "lon_S3W": d["axes"]["lon"]["S3_W_bar"],
                     "lat_pred_sha1": {k: v["pred_sha1"]
                                       for k, v in d["axes"]["lat"]["arms"].items()},
                     "lon_pred_sha1": {k: v["pred_sha1"]
                                       for k, v in d["axes"]["lon"]["arms"].items()}})
        print(f"[sweep] {i + 1}/{len(conds)} {tag}: lat {rows[-1]['lat_S3']:.6f} "
              f"lon {rows[-1]['lon_S3']:.6f} ({rows[-1]['wall_s']}s)", flush=True)
        (outdir / "s3_bar_pinning.json").write_text(
            json.dumps({"rows": rows}, indent=1), encoding="utf-8")

    res = {"_experiment": "S3 skill-bar reproducibility — T3 / POD2_EVAL_HOST §9.9",
           "_evidence_class": "MEASURED (ours; artifact = this JSON)",
           "_bar_definition": ("S3 bar = max(QWK) over B1/B2/B3 blind arms "
                               "(run_s3_characterisation.operative_blind_floor); "
                               "S3-W bar = QWK(B1_sensor_only). "
                               "skill = QWK(model) - bar."),
           "_fit": ("s3_blind_baseline._fit_mlp: 2-layer MLP hidden=64, "
                    "full-batch Adam 400 epochs, lr 3e-3, wd 1e-4, "
                    "class-weighted CE, torch.manual_seed(seed). No dropout, "
                    "no shuffling, no data augmentation."),
           "_cross_host_reference": {
               "_class": "PUBLISHED (POD2_EVAL_HOST.md §3.3.1)",
               "pod3": {"lat_S3": 0.6534, "lon_S3": 0.5323,
                        "lat_S3W": 0.2566, "lon_S3W": 0.2881},
               "pod2": {"lat_S3": 0.6493, "lon_S3": 0.5420,
                        "lat_S3W": 0.2591, "lon_S3W": 0.2881}},
           "host": os.uname().nodename, "cache": a.cache, "rows": rows}

    def stats(sel, field):
        v = np.array([r[field] for r in rows if r["family"] == sel], float)
        if v.size == 0:
            return None
        return {"n": int(v.size), "min": float(v.min()), "max": float(v.max()),
                "mean": float(v.mean()), "sd": float(v.std(ddof=1))
                if v.size > 1 else 0.0, "spread": float(v.max() - v.min()),
                "values": [float(x) for x in v]}

    res["summary"] = {fam: {f: stats(fam, f) for f in
                            ("lat_S3", "lat_S3W", "lon_S3", "lon_S3W")}
                      for fam in ("E-A_threads", "E-B_repeat", "E-C_seed")}

    def stable_dp(spread):
        """Largest d such that a difference of `spread` cannot change the d-th
        decimal. A value is quotable to d dp iff spread < 0.5 * 10^-d."""
        for d in range(6, -1, -1):
            if spread < 0.5 * 10 ** (-d):
                return d
        return 0

    verdict = {}
    for f in ("lat_S3", "lat_S3W", "lon_S3", "lon_S3W"):
        allv = np.array([r[f] for r in rows], float)
        fixed = np.array([r[f] for r in rows
                          if r["family"] in ("E-A_threads", "E-B_repeat")], float)
        seedv = np.array([r[f] for r in rows if r["family"] == "E-C_seed"], float)
        verdict[f] = {
            "spread_all": float(allv.max() - allv.min()) if allv.size else None,
            "spread_fixed_seed_any_threads":
                float(fixed.max() - fixed.min()) if fixed.size else None,
            "spread_across_seeds":
                float(seedv.max() - seedv.min()) if seedv.size else None,
            "dp_quotable_fixed_seed":
                stable_dp(float(fixed.max() - fixed.min())) if fixed.size else None,
            "dp_quotable_across_seeds":
                stable_dp(float(seedv.max() - seedv.min())) if seedv.size else None,
            "mean_across_seeds": float(seedv.mean()) if seedv.size else None,
            "sd_across_seeds": float(seedv.std(ddof=1)) if seedv.size > 1 else None,
        }
    res["pinning_verdict"] = verdict
    (Path(a.outdir) / "s3_bar_pinning.json").write_text(
        json.dumps(res, indent=1), encoding="utf-8")
    print("S3_PINNING_DONE", flush=True)
    print(json.dumps(verdict, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser("s3_bar_pinning")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("mine", "fit", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("--cache", default="/root/s3pin/s3_features.npz")
        p.add_argument("--s3dir", default="/root/TanitAD/s3")
        p.add_argument("--stack", default="/root/TanitAD/stack")
        if name == "mine":
            p.add_argument("--train-cache", required=True)
            p.add_argument("--test-cache", required=True)
            p.add_argument("--horizon-s", type=float, default=12.0)
            p.add_argument("--min-ttm-s", type=float, default=1.0)
        if name == "fit":
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--out", required=True)
        if name == "sweep":
            p.add_argument("--outdir", default="/root/s3pin")
    a = ap.parse_args()
    {"mine": phase_mine, "fit": phase_fit, "sweep": phase_sweep}[a.cmd](a)


if __name__ == "__main__":
    main()
