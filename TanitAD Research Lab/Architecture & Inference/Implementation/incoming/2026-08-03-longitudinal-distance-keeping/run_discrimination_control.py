#!/usr/bin/env python3
"""D-LEAD-1 — the pre-registered GT-vs-CV discrimination control. Run PRE_REGISTRATION.md.

Executes the control that decides whether `distance_keeping` is admissible into
`four_families.longitudinal` at all. 0 GPU. Read-only over the local PhysicalAI labels.

    python run_discrimination_control.py [--out raw/dlead1.json] [--chunks 0036,0170]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "taniteval"))

from build_lead_tracks import build_windows, local_chunks, read_chunk  # noqa: E402
from lead_metrics import TTC_CAP_S, distance_keeping  # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap  # noqa: E402

HORIZON_S, DT_S, STRIDE_S = 2.0, 0.5, 1.0
N_BOOT, SEED = 2000, 0
#: PRE_REGISTRATION.md outcome 3 — the INSTRUMENT-FAIL thresholds, fixed in advance.
MIN_WINDOWS, MIN_CLUSTERS, MAX_CENSORED_FRAC = 100, 10, 0.50


def collect(chunks: list[str], max_clips_per_chunk: int | None) -> tuple[list[dict], list[dict]]:
    wins, diags = [], []
    for ch in chunks:
        zo, ze = read_chunk(ch)
        clips = sorted({n.split("/")[-1].split(".")[0]
                        for n in zo.namelist() if n.endswith(".parquet")})
        if max_clips_per_chunk:
            clips = clips[:max_clips_per_chunk]
        for clip in clips:
            w, d = build_windows(clip, zo, ze, HORIZON_S, DT_S, STRIDE_S)
            d["chunk"] = ch
            diags.append(d)
            wins.extend(w)
        zo.close(); ze.close()
    return wins, diags


def score(wins: list[dict], key: str) -> dict:
    return distance_keeping(
        np.stack([w[key] for w in wins]),
        np.stack([w["lead"] for w in wins]),
        np.array([w["lead_len"] for w in wins]),
        np.array([w["v0"] for w in wins]),
        DT_S,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "raw" / "dlead1_discrimination.json"))
    ap.add_argument("--chunks", default="")
    ap.add_argument("--max-clips-per-chunk", type=int, default=0)
    a = ap.parse_args()

    t_start = time.time()
    chunks = a.chunks.split(",") if a.chunks else local_chunks()
    print(f"[D-LEAD-1] chunks: {chunks}")
    wins, diags = collect(chunks, a.max_clips_per_chunk or None)
    print(f"[D-LEAD-1] {len(wins)} windows with a causal lead, "
          f"{len({w['clip'] for w in wins})} clips, {time.time() - t_start:.1f}s")
    if not wins:
        print("[D-LEAD-1] no windows — nothing to test")
        return 2

    gt, cv = score(wins, "gt_path"), score(wins, "cv_path")
    eid = [w["clip"] for w in wins]

    # ⭐ the paired test runs on windows where BOTH arms have a lead in their corridor. A window
    # where only one arm keeps the lead is not a paired observation, and dropping it is reported.
    both = np.isfinite(gt["headway_min_m"]) & np.isfinite(cv["headway_min_m"])
    n_both, n_gt_only = int(both.sum()), int((np.isfinite(gt["headway_min_m"]) & ~both).sum())
    n_cv_only = int((np.isfinite(cv["headway_min_m"]) & ~both).sum())
    eidp = [e for e, m in zip(eid, both) if m]

    boot = {}
    for k in ("min_ttc_s", "headway_min_m", "time_gap_min_s"):
        ga, ca = gt[k][both], cv[k][both]
        m = np.isfinite(ga) & np.isfinite(ca)
        if m.sum() < 2:
            boot[k] = {"status": "insufficient-n", "n": int(m.sum())}
            continue
        r = paired_episode_cluster_bootstrap(ga[m], ca[m],
                                             [e for e, x in zip(eidp, m) if x],
                                             n_boot=N_BOOT, seed=SEED)
        r["n_paired"] = int(m.sum())
        boot[k] = r

    # ---- the pre-registered verdict, decided mechanically -------------------------------------
    prim = boot["min_ttc_s"]
    cens_gt = 1.0 - gt["n_closing"] / max(gt["n"], 1)
    cens_cv = 1.0 - cv["n_closing"] / max(cv["n"], 1)
    fails = []
    if n_both < MIN_WINDOWS:
        fails.append(f"n_paired {n_both} < {MIN_WINDOWS}")
    if len(set(eidp)) < MIN_CLUSTERS:
        fails.append(f"clusters {len(set(eidp))} < {MIN_CLUSTERS}")
    if cens_gt > MAX_CENSORED_FRAC and cens_cv > MAX_CENSORED_FRAC:
        fails.append(f"both arms >{MAX_CENSORED_FRAC:.0%} censored at TTC_CAP_S "
                     f"({cens_gt:.0%}/{cens_cv:.0%})")
    if prim.get("separated") and prim.get("delta", 0) < 0:
        fails.append("separated with the WRONG SIGN — CV safer than the human")

    if fails:
        verdict, branch = "INSTRUMENT-FAIL", 3
    elif prim.get("separated"):
        verdict, branch = "PASS — ADMISSIBLE", 1
    else:
        verdict, branch = "FAIL — NOT-APPLICABLE", 2

    res = {
        "experiment": "D-LEAD-1 distance-keeping discrimination control",
        "prereg": "PRE_REGISTRATION.md",
        "date": "2026-08-03",
        "config": {"horizon_s": HORIZON_S, "dt_s": DT_S, "stride_s": STRIDE_S,
                   "n_boot": N_BOOT, "seed": SEED, "ttc_cap_s": TTC_CAP_S,
                   "chunks": chunks},
        "coverage": {
            "n_windows_with_lead": len(wins),
            "n_clips_with_lead": len({w["clip"] for w in wins}),
            "n_paired": n_both, "n_gt_only": n_gt_only, "n_cv_only": n_cv_only,
            "n_clusters_paired": len(set(eidp)),
            "windows_scanned": sum(d["n_windows"] for d in diags),
            "dropped_no_lead": sum(d["dropped_no_lead"] for d in diags),
            "dropped_span": sum(d["dropped_span"] for d in diags),
            "clips_scanned": len(diags),
        },
        "arms": {
            "GT": {k: gt[k] for k in ("status", "n", "n_windows", "mean_headway_min_m",
                                      "mean_time_gap_min_s", "mean_min_ttc_s", "n_closing",
                                      "n_time_gap") if k in gt},
            "CV": {k: cv[k] for k in ("status", "n", "n_windows", "mean_headway_min_m",
                                      "mean_time_gap_min_s", "mean_min_ttc_s", "n_closing",
                                      "n_time_gap") if k in cv},
        },
        "censored_frac": {"GT": round(cens_gt, 4), "CV": round(cens_cv, 4)},
        "paired_bootstrap": boot,
        "verdict": verdict,
        "prereg_branch": branch,
        "instrument_fail_reasons": fails,
        "wall_clock_s": round(time.time() - t_start, 1),
        "hardware": "dev box CPU (no GPU)", "cost_usd": 0.0,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in
                      ("coverage", "arms", "censored_frac", "paired_bootstrap",
                       "verdict", "instrument_fail_reasons")}, indent=2))
    print(f"[D-LEAD-1] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
