"""DZ-10: value-guided test-time search N-sweep on the released DriveRL teacher, nuPlan mini.

Inputs are produced by run_mini_teacher.sh run once per N (tts=0 for the plain Beta mode, then
8/16/32/64). Two sources, both written by THEIR tooling:
  (a) score_summary.tsv from their runner (closed-loop score per task)            -> the headline
  (b) the per-step TTS candidate table persisted in every SimulationLog sample     -> the mechanism
      (trajectory.debug_info["model_input"]["test_time_scaling"], verified 2026-09-20)

What this reports, and what it refuses to report:
  * PAIRED within-scenario deltas vs the tts=0 run on the SAME scenario tokens (the runner uses
    scenario_filter.shuffle=false and a fixed limit, so the token set is identical across N; the
    script CHECKS that and refuses to compare otherwise).
  * switch rate = fraction of policy steps where selected_candidate != 0, and the score margin of
    the winner over the mode when it switched -- their conservative rule guarantees a sampled
    candidate is executed only if it beats the mode by >= margin, so a switch with margin below
    the configured value is a HARNESS ERROR, not a finding.
  * It never emits a number comparable to DriveZero Table 3 (mini tokens != val14/test14 tokens).

Usage: python tts_sweep_analysis.py <out_root_mini> <driverl_task_logs_root>
  e.g. C:/dzo  C:/Users/Admin/dz/DriveZero/DriveRL/output/task_logs
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from nuplan.planning.simulation.simulation_log import SimulationLog


def read_scores(task_logs_root: Path) -> dict[str, dict]:
    out = {}
    for tsv in task_logs_root.glob("m-*/score_summary.tsv"):
        suite = tsv.parent.name  # m-<proto>-<n|N>   (short: Windows MAX_PATH)
        for line in tsv.read_text(encoding="utf-8").splitlines()[1:]:
            f = line.split("\t")
            if len(f) >= 6:
                out[suite] = {"task": f[0], "score": float(f[2]), "ok": f[3], "fail": f[4], "total": f[5]}
    return out


def per_scenario_scores(out_dir: Path) -> dict[str, float]:
    """scenario token -> score from the aggregator parquet of one run (their emit_score reads the same file)."""
    import pandas as pd

    files = sorted((out_dir / "aggregator_metric").rglob("*weighted_average_metrics*.parquet"))
    if not files:
        return {}
    df = pd.read_parquet(files[-1])
    if "scenario" not in df or "score" not in df:
        return {}
    # MEASURED 2026-09-20: this parquet holds THREE row kinds -- 8 per-SCENARIO rows (log_name set),
    # 8 per-TYPE aggregate rows (log_name null), and final_score. Taking everything that is not
    # final_score yielded n_scen=16 for an 8-scenario run. On mini the per-type rows happen to equal
    # the per-scenario ones (one scenario per type), so the mean delta was still +0.10 and the bug was
    # INVISIBLE in the value -- only the printed n exposed it. On the six-task suite (many scenarios
    # per type) the per-type rows are AVERAGES and mixing them in would silently corrupt the pairing.
    # => key on the per-scenario rows ONLY, identified by a non-null log_name.
    rows = df[(df["scenario"].astype(str) != "final_score") & df["log_name"].notna()]
    key = "scenario" if "scenario_token" not in df else "scenario_token"
    out = {str(r[key]): float(r["score"]) for _, r in rows.iterrows()}
    if len(out) != len(rows):
        raise ValueError(f"duplicate scenario keys in {files[-1]}: {len(rows)} rows -> {len(out)} keys")
    return out


def tts_step_stats(out_dir: Path) -> dict:
    n_steps = n_switch = n_bad_margin = 0
    margins, winners_over_mode, valid_frac = [], [], []
    for log in sorted((out_dir).rglob("*.msgpack.xz")):
        try:
            data = SimulationLog.load_data(file_path=log)
        except Exception:
            continue
        for smp in data.simulation_history.data:
            dbg = getattr(smp.trajectory, "debug_info", None) or {}
            mi = dbg.get("model_input") if isinstance(dbg.get("model_input"), dict) else {}
            tts = mi.get("test_time_scaling") if isinstance(mi, dict) else None
            if not tts:
                continue
            n_steps += 1
            sel = int(tts.get("selected_candidate", 0))
            scs = np.asarray(tts.get("candidate_scores")).astype(float).ravel()
            valid = np.asarray(tts.get("candidate_valid")).astype(bool).ravel()
            margin_cfg = float(tts.get("total_return_switch_margin", 0.03))
            valid_frac.append(float(valid.mean()) if valid.size else float("nan"))
            if scs.size:
                best = float(np.nanmax(np.where(valid, scs, -np.inf))) if valid.size == scs.size else float(np.nanmax(scs))
                winners_over_mode.append(best - float(scs[0]))
            if sel != 0:
                n_switch += 1
                m = float(scs[sel] - scs[0]) if scs.size > sel else float("nan")
                margins.append(m)
                if np.isfinite(m) and m < margin_cfg - 1e-9:
                    n_bad_margin += 1
    return {
        "policy_steps_with_tts": n_steps,
        "switch_rate": (n_switch / n_steps) if n_steps else None,
        "mean_margin_when_switched": float(np.mean(margins)) if margins else None,
        "mean_best_over_mode_all_steps": float(np.mean(winners_over_mode)) if winners_over_mode else None,
        "mean_candidate_valid_frac": float(np.nanmean(valid_frac)) if valid_frac else None,
        "switches_below_configured_margin_HARNESS_ERROR": n_bad_margin,
    }


def main(out_root: Path, task_logs_root: Path) -> None:
    scores = read_scores(task_logs_root)
    print("== closed-loop scores per suite (their score_summary.tsv) ==")
    for suite, r in sorted(scores.items()):
        print(f"  {suite:28s} task={r['task']:10s} score={r['score']:7.2f}  ok/fail/total={r['ok']}/{r['fail']}/{r['total']}")
    # paired per-scenario comparison vs the notts baseline, per protocol
    for proto in ("nr", "r"):
        runs = {}
        for d in out_root.glob(f"m-{proto}-*"):
            m = re.fullmatch(r"m-" + proto + r"-(n|\d+)", d.name)
            if not m:
                continue
            n = 1 if m.group(1) == "n" else int(m.group(1))
            leaf = next(iter(d.rglob("closed_loop_*")), None)
            if leaf is None:
                continue
            runs[n] = (leaf, per_scenario_scores(leaf))
        if 1 not in runs or len(runs) < 2:
            continue
        base_leaf, base = runs[1]
        print(f"\n== PAIRED vs N=1 ({proto}) on identical scenario tokens ==")
        for n in sorted(k for k in runs if k != 1):
            leaf, sc = runs[n]
            common = sorted(set(base) & set(sc))
            if set(base) != set(sc):
                print(f"  N={n:3d}: REFUSED -- token sets differ (base {len(base)}, run {len(sc)}, common {len(common)})")
                continue
            d = np.array([sc[t] - base[t] for t in common])
            stats = tts_step_stats(leaf)
            print(f"  N={n:3d}: n_scen={len(common):3d}  mean delta={d.mean()*100:+.2f}  "
                  f"[paired, not a CI: {len(common)} scenarios]  switch_rate={stats['switch_rate']}  "
                  f"mean_margin_when_switched={stats['mean_margin_when_switched']}  "
                  f"below-margin switches={stats['switches_below_configured_margin_HARNESS_ERROR']}")
    print("\nNOTE: mini tokens != val14/test14 tokens; nothing above is comparable to DriveZero Table 3.")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
