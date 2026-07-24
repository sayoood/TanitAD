"""``speed_benefit_recovered_frac`` — the v4 gate's quiet-plateau KILL secondary.

WHY THIS EXISTS (V4_FLAGSHIP_DESIGN.md §7.5 / P8, 2026-07-23)
------------------------------------------------------------
The WM canary (``wm_canary_ade_2s``) catches a WM that *blows up* (v1.6:
0.452 -> 1.10). It does NOT catch the quieter failure [PM] found in v3enc: a
trunk that stays superficially healthy while the operative rollout **quietly
plateaus far short of the speed-channel benefit** — v3enc's canary looked fine,
yet it recovered only **18.6 %** of the speed benefit at 8-10 k while v1 recovered
**81.8 %**. ``speed_benefit_recovered_frac >= 0.70`` is the KILL secondary that
fires on that plateau (V4_FLAGSHIP_DESIGN.md §9 split-card table: **KILL** — "the
quiet plateau [PM] found in v3enc; the canary alone does not catch it").

THE PINNED CONVENTION (do NOT re-derive — CLAUDE.md §"source of truth")
-----------------------------------------------------------------------
The reducer is promoted verbatim from ``postmortem_a_analyze.py`` (the file that
first published the 81.8 % / 18.6 % rows), so the gate reads the SAME quantity
the post-mortem did. Three pinned facts, each of which changes the number if got
wrong, so each is a test:

  1. **Metric** = ``g_op_fwd_ade_m`` — the operative forward-rollout ADE logged
     every ``--log-every`` step in ``train_log.jsonl``. This is a TRAIN-log
     descriptive summary (never a held-out eval number); it is admissible here
     ONLY because the statistic is a matched-step RATIO against the arm's own
     no-speed control, so the shared train-time descent cancels.
  2. **Bucket** = ``lo < step <= hi`` (left-OPEN, right-CLOSED). This is the one
     of four candidate conventions that reproduces the post-mortem's published
     rows; ``postmortem_a_analyze.py`` verified it against SS1.1 (0-2 k) and
     Appendix B (8-10 k). The gate bucket is ``(8000, 10000]``.
  3. **Formula** = ``(nospeed - arm) / nospeed`` — the fraction of the no-speed
     control's error the arm removes. 1.0 = perfect (zero error); 0.0 = no better
     than a trunk with no speed channel at all. This is
     ``postmortem_a_analyze.py`` line 205 verbatim
     (``row[f"speed_benefit_recovered_{nm}"] = round((ns - val) / ns, 4)``).

Costs ZERO GPU: both reference logs are git-tracked in
``taniteval/results/trainlogs/`` (V4_FLAGSHIP_DESIGN.md §12.2 P4/O-04 — "computable
off a pod entirely; this was v4's most dangerous single-disk dependency and it is
gone"). The candidate arm's own ``train_log.jsonl`` is the only per-run input.

Evidence class: **MEASURED** (the arm's + the no-speed control's train logs).
Reproduction pin (this module's test): v1 -> 0.8184, v3enc -> 0.1859 at (8000,
10000] — the design's 81.8 % / 18.6 % headline, to the quoted precision.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

# The gate metric, bucket and threshold — all pinned by V4_FLAGSHIP_DESIGN.md.
GATE_METRIC = "g_op_fwd_ade_m"
GATE_BUCKET = (8000, 10000)          # (lo, hi]; the step-10 000 gate's 8-10 k window
GATE_THRESHOLD = 0.70                # >= ; §9 KILL secondary
# The no-speed ablation CONTROL is the 0 %-recovery floor. It is the SAME arm the
# post-mortem used: flagship4b-phase0 (no speed channel, action_dim 2). Git-tracked.
DEFAULT_NOSPEED_LOG = "taniteval/results/trainlogs/nospeed-phase0_train_log.jsonl"
DEFAULT_V1_LOG = "taniteval/results/trainlogs/v1-speedjerk_train_log.jsonl"


def load_log(path) -> dict[int, dict]:
    """``{step: row}``, deduped on step keeping the **LAST** occurrence.

    Identical to ``postmortem_a_analyze.load`` / experiment B: v1 and the no-speed
    arm replay steps after a resume, and the last write is the live one."""
    bystep: dict[int, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue                                    # tolerate [guard] noise
        if isinstance(r, dict) and "step" in r:
            bystep[int(r["step"])] = r
    return bystep


def bucket_mean(rows: dict[int, dict], key: str, lo: int, hi: int):
    """Arithmetic mean of ``row[key]`` over ``lo < step <= hi`` (left-open,
    right-closed — the post-mortem's pinned convention). ``None`` if empty."""
    v = [r[key] for s, r in rows.items() if lo < s <= hi and key in r]
    return round(st.fmean(v), 4) if v else None


def recovered_frac(arm_rows: dict[int, dict], nospeed_rows: dict[int, dict],
                   bucket=GATE_BUCKET, metric=GATE_METRIC) -> dict:
    """``(nospeed - arm) / nospeed`` on the bucket-mean ``metric``.

    Returns a self-describing dict (never a bare float — CLAUDE.md: an interval /
    ratio carries its construction). ``value`` is ``None`` iff either arm has no
    logged rows in the bucket, so the caller can render ``NOT SUPPLIED`` rather
    than fabricate a pass."""
    lo, hi = bucket
    arm = bucket_mean(arm_rows, metric, lo, hi)
    ns = bucket_mean(nospeed_rows, metric, lo, hi)
    n_arm = sum(1 for s, r in arm_rows.items() if lo < s <= hi and metric in r)
    n_ns = sum(1 for s, r in nospeed_rows.items() if lo < s <= hi and metric in r)
    frac = (round((ns - arm) / ns, 4)
            if (arm is not None and ns not in (None, 0)) else None)
    return {
        "metric": metric,
        "bucket": [lo, hi],
        "bucket_convention": "lo < step <= hi (left-open, right-closed)",
        "arm_bucket_mean": arm,
        "nospeed_control_bucket_mean": ns,
        "n_arm_rows": n_arm,
        "n_nospeed_rows": n_ns,
        "value": frac,
        "formula": "(nospeed - arm) / nospeed",
        "reading": ("1.0 = zero operative-rollout error (full speed benefit); "
                    "0.0 = no better than a trunk with no speed channel"),
        "estimator": "train_log_bucket_mean_ratio (DESCRIPTIVE train-log summary, "
                     "NOT a held-out eval; admissible only as a matched-step "
                     "ratio vs the arm's own no-speed control)",
    }


def emit(arm_log, nospeed_log=DEFAULT_NOSPEED_LOG, bucket=GATE_BUCKET,
         metric=GATE_METRIC, threshold=GATE_THRESHOLD, repo_root=None,
         per_2k_trajectory=True) -> dict:
    """Gate-ready emission for ``speed_benefit_recovered_frac``.

    ``arm_log`` is the candidate arm's ``train_log.jsonl`` (the v4 arm at the 10 k
    gate; or flagship v1's for the §17.1b dry-run). Paths may be repo-relative;
    ``repo_root`` (default: three parents up = the repo) resolves them.

    Also emits the per-2 k-bucket trajectory so the "flat below 0.70 for three
    consecutive 2 k buckets" early-warning (V4_FLAGSHIP_DESIGN.md §7.1 falsifier
    1) is one read away — the gate value is the (8000, 10000] bucket only."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (root / p)

    arm_rows = load_log(_resolve(arm_log))
    ns_rows = load_log(_resolve(nospeed_log))
    core = recovered_frac(arm_rows, ns_rows, bucket, metric)
    val = core["value"]
    out = {
        "gate_metric": "speed_benefit_recovered_frac",
        "value": val,
        "threshold": threshold,
        "direction": ">=",
        "pass": (bool(val >= threshold) if val is not None else None),
        "evidence_class": "MEASURED (arm train_log + no-speed control train_log)",
        "provenance": {
            "arm_log": str(arm_log),
            "nospeed_control_log": str(nospeed_log),
            "reducer": "tanitad.eval.speed_benefit.recovered_frac "
                       "(pinned from postmortem_a_analyze.py:205)",
        },
        **core,
    }
    if per_2k_trajectory:
        traj = {}
        for lo in range(0, bucket[1], 2000):
            hi = lo + 2000
            traj[f"{lo}-{hi}"] = recovered_frac(arm_rows, ns_rows, (lo, hi),
                                                metric)["value"]
        out["per_2k_bucket_trajectory"] = traj
        below = [k for k, v in traj.items() if v is not None and v < threshold]
        out["three_consecutive_2k_below_threshold"] = _has_3_consecutive(
            traj, threshold)
        out["buckets_below_threshold"] = below
    return out


def _has_3_consecutive(traj: dict, threshold: float) -> bool:
    """The §7.1 falsifier-1 early warning: three consecutive 2 k buckets whose
    recovered-fraction is below ``threshold`` (ordered by bucket start)."""
    ordered = [traj[k] for k in sorted(traj, key=lambda s: int(s.split("-")[0]))]
    run = 0
    for v in ordered:
        run = run + 1 if (v is not None and v < threshold) else 0
        if run >= 3:
            return True
    return False


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        "speed_benefit", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-log", default=DEFAULT_V1_LOG,
                    help="candidate arm train_log.jsonl (default: flagship v1, "
                         "the §17.1b dry-run fixture)")
    ap.add_argument("--nospeed-log", default=DEFAULT_NOSPEED_LOG)
    ap.add_argument("--bucket", nargs=2, type=int, default=list(GATE_BUCKET))
    ap.add_argument("--metric", default=GATE_METRIC)
    ap.add_argument("--threshold", type=float, default=GATE_THRESHOLD)
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = emit(a.arm_log, a.nospeed_log, tuple(a.bucket), a.metric, a.threshold,
               a.repo_root)
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
