#!/usr/bin/env python3
"""RE-DRIVE the G4 gate (P2 planner closed-loop) on the decision-grade estimator.

WHY THIS IS THE HIGHEST-CONSEQUENCE ITEM
----------------------------------------
`planner_p2_flagship-30k.json` is the artifact behind

    G4 closed-loop | ADE@2s 1.038 +- 0.202 vs v1 head 1.685 +- 0.098 | PASS

quoted in MODEL_REGISTRY.md :1456-1458 and :1683, in the adjudicating research
note, in V3_HIERARCHICAL_PLANNING_DESIGN.md (where 1.69 m IS the gate
THRESHOLD), in V35_DESIGN.md, in TANITAD_PAPER.md and in the 4-brain dominance
program. **Both sides of that comparison are `overlapping_holdout_se`**: the
artifact stamps `"ci": "8-split episode jackknife"`, and `planner_p2.py` was
never migrated — it still aggregates through `_jack_scalar` / `_jack_paired`
(`:373` / `:381`), which the module's own docstring already flags as *"random
holdouts; NOT a jackknife"*.

⚠️ THE CEM IS UNSEEDED. `planner_p2.py:249` draws `torch.randn(B,N,K,2)` with no
seed, so a re-drive is NOT bit-reproducible against 2026-07-19. This script
therefore seeds the re-drive and reports THREE numbers, never conflating them:

  published_legacy  — 2026-07-19, unseeded CEM, overlapping_holdout_se
  rerun_legacy      — this run,   seeded CEM,   overlapping_holdout_se
  rerun_corrected   — this run,   seeded CEM,   episode_cluster_bootstrap

`rerun_corrected` vs `rerun_legacy` is the **clean estimator effect** (identical
windows, identical CEM draw). `rerun_legacy` vs `published_legacy` is the CEM
sampling drift, reported separately so it is never mistaken for the correction.

This script does NOT modify `planner_p2.py` — it calls that module's collection
functions unchanged (the physics) and re-aggregates (the statistic).

Run:  OMP_NUM_THREADS=8 PYTHONPATH=/root/taniteval:/root/TanitAD/stack \
      python3 rerun_planner_p2_g4.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/taniteval")

from taniteval import ci as _ci                      # noqa: E402
from taniteval import planner_p2 as P2               # noqa: E402
from tanitad.eval.gates import split_by_episode      # noqa: E402

OUT = Path("/root/cl_rerun_20260726")
PUB = Path("/root/taniteval/results/planner_p2_flagship-30k.json")
SEED = 0
CL_EPISODES = 20        # matches the published artifact (n_episodes 20, n=221)
CL_STRIDE = 16          # matches the published artifact


def legacy(vals, eids, n_splits=8, val_frac=0.2):
    """`planner_p2._jack_scalar` — the deprecated statistic, by its real name."""
    v = np.asarray(vals, dtype=float)
    splits = [split_by_episode(eids, val_frac, s) for s in range(n_splits)]
    sm = np.asarray([float(np.nanmean(v[va])) for _tr, va in splits if len(va)])
    m = float(np.mean(sm))
    c = float(_ci.overlapping_holdout_se(sm))
    return {"mean": round(m, 4), "ci95": round(c, 4), "n": int(v.size),
            "estimator": "overlapping_holdout_se", "deprecated": True}


def corrected(vals, eids):
    return _ci.episode_cluster_bootstrap(np.asarray(vals, dtype=float),
                                         [str(x) for x in eids],
                                         n_boot=2000, seed=SEED)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    t0 = time.time()
    entry, L, files, data = P2._load("flagship-30k", "cuda")
    speed_input = bool(entry.get("speed_input"))
    eps_cl = data.load_frames(files[:CL_EPISODES])
    col = P2.collect_closedloop(L["model"], L["step_readout"], eps_cl, "cuda",
                                w=P2.W, cfg=P2.CEM_CL, stride=CL_STRIDE,
                                speed_input=speed_input, replan_every=1)
    torch.save({k: (v.cpu() if torch.is_tensor(v) else v)
                for k, v in col.items()}, OUT / "p2win_flagship-30k.pt")
    eids = col["eid"]
    gt = col["gt"]
    de = torch.linalg.norm(col["closed_bike"] - gt, dim=-1)      # [N,4]
    ade = de.mean(dim=1).numpy()
    fde = de[:, -1].numpy()
    div = (fde > 5.0).astype(float)
    ade_o = torch.linalg.norm(col["open_grnd"] - gt, dim=-1).mean(dim=1).numpy()
    ade_cv = torch.linalg.norm(col["cv"] - gt, dim=-1).mean(dim=1).numpy()

    pub = json.loads(PUB.read_text(encoding="utf-8"))["closed_loop"]
    rows = {}
    for name, vals, pk in (("closed_bike_ade2s", ade, "closed_bike_ade2s"),
                           ("closed_bike_fde2s", fde, "closed_bike_fde2s"),
                           ("open_grnd_ade2s", ade_o, "open_grnd_ade2s"),
                           ("cv_ade2s", ade_cv, "cv_ade2s"),
                           ("divergence_rate_gt5m", div, "divergence_rate_gt5m")):
        lg, cr = legacy(vals, eids), corrected(vals, eids)
        rows[name] = {
            "published_legacy_mean": pub[pk]["mean"],
            "published_legacy_ci95": pub[pk]["ci95"],
            "published_estimator": "overlapping_holdout_se "
                                   "(stamped '8-split episode jackknife')",
            "rerun_legacy_mean": lg["mean"], "rerun_legacy_ci95": lg["ci95"],
            "rerun_corrected_mean": cr["mean"], "rerun_corrected_ci95": cr["ci95"],
            "rerun_corrected_lo": cr["lo"], "rerun_corrected_hi": cr["hi"],
            "rerun_corrected_estimator": cr["estimator"],
            "estimator_effect_pct": round(
                100.0 * (lg["mean"] - cr["mean"]) / max(1e-9, cr["mean"]), 3),
            "cem_sampling_drift_pct": round(
                100.0 * (pub[pk]["mean"] - lg["mean"]) / max(1e-9, lg["mean"]), 3),
            "ci_width_ratio_corrected_over_rerun_legacy": (
                round(cr["ci95"] / lg["ci95"], 3) if lg["ci95"] else None),
            "n_windows": cr["n_windows"], "n_episodes": cr["n_episodes"]}

    res = {
        "_what": "G4 gate (P2 CEM planner closed-loop) re-driven on the "
                 "decision-grade estimator",
        "_published_artifact": str(PUB),
        "_seeded": {"seed": SEED,
                    "why": "planner_p2.py:249 draws torch.randn UNSEEDED, so "
                           "2026-07-19 is not bit-reproducible. Seeding makes "
                           "the estimator effect (rerun_corrected vs "
                           "rerun_legacy) exact on identical windows; the CEM "
                           "sampling drift is reported separately."},
        "_estimator": {"corrected": "episode_cluster_bootstrap (B=2000, val episodes)",
                       "deprecated": "overlapping_holdout_se"},
        "protocol": {"cl_episodes": CL_EPISODES, "stride": CL_STRIDE,
                     "cem": P2.CEM_CL},
        "rows": rows,
        "G4_verdict": {},
        "wall_s": round(time.time() - t0, 1)}

    # ---- the gate itself -------------------------------------------------- #
    # NOTE the gate compares the planner (n=221 / 20 ep / stride 16) against the
    # v1 head closed-loop baseline (n=881 / 40 ep / stride 8). DIFFERENT WINDOW
    # SETS -> it can never be a paired test, and neither side's interval was
    # decision-grade. Both facts are recorded rather than papered over.
    cb = rows["closed_bike_ade2s"]
    res["G4_verdict"] = {
        "published": {"planner": pub["closed_bike_ade2s"]["mean"],
                      "planner_ci95": pub["closed_bike_ade2s"]["ci95"],
                      "head_baseline": 1.6852,
                      "head_baseline_source": "closedloop_flagship-30k.json "
                                              "(legacy) closed_bike ade_0_2s",
                      "pass": True},
        "corrected": {"planner": cb["rerun_corrected_mean"],
                      "planner_lo": cb["rerun_corrected_lo"],
                      "planner_hi": cb["rerun_corrected_hi"],
                      "head_baseline": 1.7318,
                      "head_baseline_source": "closedloop_flagship-30k."
                                              "CORRECTED.json closed_bike "
                                              "ade_0_2s (episode_cluster_"
                                              "bootstrap, this task)"},
        "_unpaired_warning": "planner n=221 / 20 ep / stride 16 vs head "
                             "baseline n=881 / 40 ep / stride 8 — DIFFERENT "
                             "window sets. This comparison cannot be paired "
                             "and its 'CI-separated' claim rests on two "
                             "independent intervals, both of which were "
                             "computed with the deprecated estimator."}
    (OUT / "planner_p2_G4.CORRECTED.json").write_text(
        json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res["rows"], indent=2))
    print(json.dumps(res["G4_verdict"], indent=2))


if __name__ == "__main__":
    main()
