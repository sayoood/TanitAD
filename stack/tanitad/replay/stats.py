"""Replay aggregation + baseline regression compare (CI-hookable).

``aggregate`` turns a stream of :class:`~tanitad.replay.engine.TimestepRecord`
into one stats dict (JSON-serializable): per-arm/per-horizon ADE/FDE, action
MAE, maneuver accuracy, imag_rel means, latency p50/p95, per-episode ADE and
the worst-K windows (the "scrub here" list for the viz).

``compare`` diffs a stats dict against a baseline stats.json with per-metric
tolerances and returns the regressions — the CLI exits 1 on any, so a
checkpoint that quietly got worse fails CI loudly (pairs with the future
ci.ps1 gate, Tools&DevEnv 2026-07-09 note).

Direction handling: metrics containing an entry of :data:`HIGHER_BETTER_TOKENS`
count a DROP as regression; everything else counts a RISE. ``n_windows`` and
``fit_windows`` are informational (compared for visibility, never a verdict —
different replay configs legitimately differ). Latency defaults to a loose
tolerance because wall-clock is machine-dependent; tighten it explicitly when
baseline and candidate run on the same box.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from tanitad.replay.engine import WAYPOINT_STEPS, TimestepRecord

# Metric-name tokens where HIGHER is better (everything else: lower better).
HIGHER_BETTER_TOKENS = ("maneuver_acc", "r2")
# Informational metrics: reported in the delta table, never a verdict.
INFO_ONLY = ("n_windows", "fit_windows")

# Default relative tolerances by substring match (first match wins, ordered);
# "" is the catch-all. Override per metric via compare(tolerances=...).
DEFAULT_TOLERANCES: tuple[tuple[str, float], ...] = (
    ("latency", 0.50),        # wall-clock: machine-dependent, loose by default
    ("ood", 0.25),            # monitor signals: drift-y, loose
    ("conf", 0.25),
    ("sigma", 0.25),
    ("imag_rel", 0.10),
    ("", 0.05),               # quality metrics: ADE/FDE/MAE/accuracy
)
ABS_FLOOR = 1e-3              # relative tolerance floor for near-zero baselines


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def aggregate(records: Iterable[TimestepRecord], meta: dict | None = None,
              worst_k: int = 10) -> dict:
    """Aggregate replay records into the stats dict (see module docstring).

    Raises on an empty record stream and on arms that produced no waypoint
    output at all — an all-``None`` arm is a wiring bug, not a statistic.
    """
    per_arm: dict[str, dict] = {}
    n_records = 0
    for rec in records:
        n_records += 1
        for name, out in rec.arms.items():
            a = per_arm.setdefault(name, {
                "err_h": {k: [] for k in WAYPOINT_STEPS}, "ade": [],
                "steer_ae": [], "accel_ae": [], "man_hit": [],
                "imag_rel": {}, "conf": [], "ood": [], "sigma": [],
                "latency": [], "per_ep": {}, "worst": []})
            a["latency"].append(out.latency_ms)
            if out.waypoints is not None:
                if tuple(out.waypoint_steps) != WAYPOINT_STEPS:
                    raise ValueError(
                        f"arm {name!r} emitted waypoint steps "
                        f"{out.waypoint_steps} != {WAYPOINT_STEPS}")
                err = np.linalg.norm(
                    np.asarray(out.waypoints) - rec.gt_waypoints, axis=-1)
                for i, k in enumerate(WAYPOINT_STEPS):
                    a["err_h"][k].append(float(err[i]))
                ade = float(err.mean())
                a["ade"].append(ade)
                ep_key = f"{rec.corpus}/ep{rec.episode_id}"
                a["per_ep"].setdefault(ep_key, []).append(ade)
                a["worst"].append({"corpus": rec.corpus,
                                   "episode_id": rec.episode_id,
                                   "t": rec.t, "step": rec.step,
                                   "ade": round(ade, 4)})
            if out.action is not None:
                d = np.abs(np.asarray(out.action) - rec.gt_action)
                a["steer_ae"].append(float(d[0]))
                a["accel_ae"].append(float(d[1]))
            if out.maneuver_probs is not None and out.maneuver_gt is not None:
                a["man_hit"].append(
                    int(int(np.argmax(out.maneuver_probs)) == out.maneuver_gt))
            if out.imag_rel is not None:
                for k, v in out.imag_rel.items():
                    a["imag_rel"].setdefault(int(k), []).append(float(v))
            for key in ("conf", "ood", "sigma"):
                v = getattr(out, key)
                if v is not None:
                    a[key].append(float(v))
    if n_records == 0:
        raise ValueError("aggregate() got zero records — nothing was replayed")

    arms_out: dict[str, dict] = {}
    for name, a in per_arm.items():
        if not a["ade"]:
            raise ValueError(
                f"arm {name!r} produced no waypoint outputs over "
                f"{n_records} records — wiring bug, not a statistic")
        m: dict = {"n_windows": len(a["latency"])}
        for k in WAYPOINT_STEPS:
            m[f"ade@{k}"] = round(float(np.mean(a["err_h"][k])), 4)
        m["ade"] = round(float(np.mean(a["ade"])), 4)
        m[f"fde@{max(WAYPOINT_STEPS)}"] = m[f"ade@{max(WAYPOINT_STEPS)}"]
        if a["steer_ae"]:
            m["steer_mae"] = round(float(np.mean(a["steer_ae"])), 4)
            m["accel_mae"] = round(float(np.mean(a["accel_ae"])), 4)
        if a["man_hit"]:
            m["maneuver_acc"] = round(float(np.mean(a["man_hit"])), 4)
        for k in sorted(a["imag_rel"]):
            m[f"imag_rel_k{k}"] = round(float(np.mean(a["imag_rel"][k])), 4)
        for key in ("conf", "ood", "sigma"):
            if a[key]:
                m[f"{key}_mean"] = round(float(np.mean(a[key])), 4)
        m["latency_p50_ms"] = round(_percentile(a["latency"], 50), 3)
        m["latency_p95_ms"] = round(_percentile(a["latency"], 95), 3)
        m["per_episode_ade"] = {k: round(float(np.mean(v)), 4)
                                for k, v in sorted(a["per_ep"].items())}
        m["worst_windows"] = sorted(a["worst"], key=lambda r: -r["ade"]
                                    )[:worst_k]
        arms_out[name] = m

    return {"meta": dict(meta or {}) | {"n_records": n_records,
                                        "waypoint_steps": list(WAYPOINT_STEPS)},
            "arms": arms_out}


# --------------------------------------------------------------------------
# Regression compare
# --------------------------------------------------------------------------

def flatten_metrics(stats: dict) -> dict[str, float]:
    """``{"arm.metric": value}`` over all scalar arm metrics."""
    out: dict[str, float] = {}
    for arm, metrics in stats.get("arms", {}).items():
        for k, v in metrics.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            out[f"{arm}.{k}"] = float(v)
    return out


def _tolerance_for(key: str, tolerances: dict[str, float] | None) -> float:
    """Exact key -> user substring -> default substring chain."""
    if tolerances:
        if key in tolerances:
            return tolerances[key]
        for pat, tol in tolerances.items():
            if pat and pat in key:
                return tol
    for pat, tol in DEFAULT_TOLERANCES:
        if pat in key:
            return tol
    raise AssertionError("unreachable: catch-all tolerance missing")


def compare(current: dict, baseline: dict,
            tolerances: dict[str, float] | None = None
            ) -> tuple[list[dict], list[dict]]:
    """Diff ``current`` stats against ``baseline`` -> (regressions, rows).

    A metric regresses when it moves in its BAD direction by more than
    ``tol * max(|baseline|, ABS_FLOOR)``. Metrics present on only one side
    are reported as INFO rows (arm/head sets may legitimately differ), never
    silently dropped. ``rows`` is the full human-readable delta table.
    """
    cur, base = flatten_metrics(current), flatten_metrics(baseline)
    rows: list[dict] = []
    regressions: list[dict] = []
    for key in sorted(set(cur) | set(base)):
        row: dict = {"metric": key,
                     "baseline": base.get(key), "current": cur.get(key)}
        if key not in cur or key not in base:
            row.update(status="INFO", note="only on one side")
            rows.append(row)
            continue
        b, c = base[key], cur[key]
        delta = c - b
        row["delta"] = round(delta, 4)
        leaf = key.split(".", 1)[-1]
        if leaf in INFO_ONLY:
            row["status"] = "INFO" if delta else "OK"
            rows.append(row)
            continue
        higher_better = any(tok in leaf for tok in HIGHER_BETTER_TOKENS)
        worse = -delta if higher_better else delta
        tol = _tolerance_for(leaf, tolerances)
        row["tol"] = tol
        allowed = tol * max(abs(b), ABS_FLOOR)
        if worse > allowed:
            row["status"] = "REGRESS"
            regressions.append(row)
        elif worse < -allowed:
            row["status"] = "BETTER"
        else:
            row["status"] = "OK"
        rows.append(row)
    return regressions, rows


def format_table(rows: Sequence[dict]) -> str:
    """Aligned, human-readable delta table for terminal/CI logs."""
    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    header = ("metric", "baseline", "current", "delta", "tol", "status")
    body = [(r["metric"], fmt(r.get("baseline")), fmt(r.get("current")),
             fmt(r.get("delta")), fmt(r.get("tol")), r["status"])
            for r in rows]
    widths = [max(len(h), *(len(b[i]) for b in body)) if body else len(h)
              for i, h in enumerate(header)]
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(header)),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(c.ljust(widths[i]) for i, c in enumerate(b))
              for b in body]
    return "\n".join(lines)


def load_stats(path: str | Path) -> dict:
    """Read a stats.json (fail-loud on shape: must carry an 'arms' dict)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "arms" not in d or not isinstance(d["arms"], dict):
        raise ValueError(f"{path} is not a replay stats.json (no 'arms' key)")
    return d
