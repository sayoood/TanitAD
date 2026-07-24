"""run_gate.py — the TanitAD restart/continue gate. Replaces the power-law exponent.

WHY THIS EXISTS (360-review W2/P1, 2026-07-20)
----------------------------------------------
Restart decisions on multi-GPU-day runs were being made from a log-log power-law
fit of a TRAIN metric (``g_op_fwd_ade_m`` out of ``train_log.jsonl``), with
R^2 0.13-0.57, no CI on the exponent, no stated fit window, and 4-6x
extrapolation. The same log on the same day gave -0.379 / -0.503 / -0.563 /
-0.774 / -0.820 depending on the window chosen, and v1's reference -0.84 was
quoted with **no stated fit range at all**. D-031 killed v2 on it; D-A7 made it
v3enc's pre-registered falsifier.

The v2 kill was probably right, but for the OTHER statistic in the same note:
the **matched-step ratio** (v2/v1 on identical metric at identical steps,
widening 1.51 -> 4.33) and "v1 reached v2's 7.5k value at step ~250". Those need
no power law, no extrapolation and no exponent. This tool promotes them and
demotes the exponent to a logged diagnostic that CANNOT be quoted bare.

  CAVEAT (2026-07-21): that "~step 250" was produced by the 3-point-median rule
  since RETIRED as not robust for this per-batch metric -- the same rule that
  put v1 at "step 450" for v3enc, ~5x too early (``reference_reached_at``
  docstring). It has NOT been recomputed under the k-consecutive rule and must
  not be re-quoted until it is. The v2 kill does not rest on it: the widening
  matched-step ratio and the failed probe carry that verdict.

WHAT THIS TOOL ENFORCES
-----------------------
1. **Held-out milestone metric is primary.** A train-log slope may never decide
   a gate. ``check`` refuses to pass/fail on train metrics alone (REF-A's train
   fwd-ADE was 0.65 against a held-out 2.92 -- a 4.5x dissociation).
2. **Matched-step ratio** r(s) = M_new(s)/M_ref(s) at IDENTICAL s is the
   comparative statistic, with a bootstrap CI, plus the assumption-free
   "reference reached the new run's current value at step X" -- where X requires
   k CONSECUTIVE crossings and ships with the rule that produced it, because a
   per-batch metric swings ~2x between adjacent rows.
3. **Every slope carries a bootstrap CI**, and its fit window, R^2 and n.
4. **R^2 < 0.8 => no exponent.** ``SlopeFit.exponent`` RAISES; the renderer
   prints ``UNSUPPORTED``. There is no code path that returns a bare float.
5. **Extrapolation capped at 2x** the fitted range. Beyond that it refuses.
6. **One pre-registered gate step**, written BEFORE launch (``register``).
   ``check`` refuses to judge a run with no card, and refuses at any step other
   than the registered one -- no garden of forking paths.
7. **GPU-hours, not steps.** Budget comparisons are normalised by measured
   wall-clock from the logs, because equal-step is not equal-cost.
8. **Restart cap per lever family** (default 2). A third failure refutes the
   lever family; it does not license more schedule tuning.

USAGE
-----
  # BEFORE launch -- writes the card; refuses to overwrite one
  python run_gate.py register --run flagship-v3enc --gate-step 10000 \
      --primary-metric ade_0_2s --primary-threshold 2.5 \
      --secondary "encoder_speed_probe_r2>=0.55" \
      --secondary "highspeed_long_overshoot_m<=8.0" \
      --reference-run flagship-v1 --reference-log v1_train_log.jsonl \
      --compare-metric g_op_fwd_ade_m --tau 1.5 \
      --lever-family encoder-grounding --restarts-used 1 \
      --card gates/flagship-v3enc.card.json

  # AT the gate step
  python run_gate.py check --card gates/flagship-v3enc.card.json \
      --log /tmp/flagship_v3enc.log --eval-json results/flagship-v3enc-10k.json

  # diagnostics (never a decision)
  python run_gate.py fit   --log <log> --metric g_op_fwd_ade_m --from 1500 --to 7500
  python run_gate.py ratio --log <new> --reference-log <ref> --metric g_op_fwd_ade_m
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

R2_FLOOR = 0.80          # below this an exponent may not be quoted at all
MAX_EXTRAP = 2.0         # never project more than 2x beyond the fitted range
DEFAULT_TAU = 1.5        # matched-step ratio continue-threshold
DEFAULT_RESTART_CAP = 2  # restarts per lever family
N_BOOT = 2000
REACHED_K = 3            # consecutive logged rows required to call a crossing
REACHED_BUCKET = 1000    # step width of the corroborating bucket-mean view

# The ONLY decision-grade interval estimator (taniteval/ci.py). A gate that
# reads anything else is invalid.
CLUSTER_BOOTSTRAP_ESTIMATOR = "episode_cluster_bootstrap"
# 1.28-2.06x too narrow across 10 arms (CLAUDE.md; CI_RECOMPUTE_2026-07-20.json).
# Forbidden for any decision; kept in eval JSON for historical reproducibility.
DEPRECATED_ESTIMATOR = "overlapping_holdout_se"

# --- metric-name aliasing — the 3-way miss-name drift ----------------------- #
# The SAME quantity (final-point miss@2m) is spelled three ways: the gate card
# writes ``miss_at_2m``, ``driving.py`` emits ``miss_2m``, ``bench.py`` emits
# ``miss_rate@2m``. V4_FLAGSHIP_DESIGN §17.3 requires all three to resolve to
# one another. Resolving here (rather than renaming the emitters) means no
# published JSON, report, or test has to change, and the card's canonical
# ``miss_at_2m`` always finds whichever spelling the eval JSON carries.
_METRIC_ALIAS_GROUPS = [
    ("miss_at_2m", "miss_2m", "miss_rate@2m"),
]
_ALIAS_OF = {name: grp for grp in _METRIC_ALIAS_GROUPS for name in grp}


def _metric_aliases(metric):
    """Every admissible spelling of ``metric`` (itself first). A non-aliased
    metric returns just itself, so this is safe to call unconditionally."""
    return _ALIAS_OF.get(metric, (metric,))


# =========================================================================== #
# Log reading                                                                 #
# =========================================================================== #
def read_log(path) -> list[dict]:
    """Every JSON object line carrying a ``step``, in order. Tolerates logs with
    interleaved non-JSON noise (``[guard] ...``) — the v3enc log is exactly that."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict) and "step" in d:
                rows.append(d)
    if not rows:
        raise SystemExit(f"[gate] REFUSING: no step-bearing JSON lines in {path}")
    rows.sort(key=lambda r: r["step"])
    return rows


def series(rows, metric) -> tuple[np.ndarray, np.ndarray]:
    s = np.array([r["step"] for r in rows if metric in r], dtype=float)
    v = np.array([r[metric] for r in rows if metric in r], dtype=float)
    ok = np.isfinite(s) & np.isfinite(v)
    return s[ok], v[ok]


def gpu_hours(rows, metric="step_s") -> float:
    """Total GPU-hours from the logged per-interval wall time.

    ``step_s`` is ACCUMULATED over the ``--log-every`` interval (verified: v1
    logs ~663 s per 50 steps = 13.3 s/step; v3enc ~515 s per 50 = 10.3 s/step).
    Summing the logged values therefore gives total seconds directly; the step-0
    row covers no interval and is dropped."""
    tot = sum(float(r[metric]) for r in rows[1:] if metric in r)
    return tot / 3600.0


def s_per_step(rows, metric="step_s") -> float:
    """Mean seconds/step, from accumulated interval times over the step deltas."""
    tot, steps = 0.0, 0
    prev = rows[0]["step"]
    for r in rows[1:]:
        if metric in r:
            tot += float(r[metric])
            steps += r["step"] - prev
        prev = r["step"]
    return tot / steps if steps else float("nan")


# =========================================================================== #
# Slope fitting — an exponent CANNOT escape this object without provenance     #
# =========================================================================== #
@dataclass
class SlopeFit:
    """A log-log slope that is impossible to quote without window / R^2 / n.

    ``exponent`` is a property that RAISES when ``r2 < R2_FLOOR``; the only
    string form (:meth:`render`) always carries the fit window, R^2, n and the
    bootstrap CI. ``asdict`` for JSON always carries them too. There is no
    accessor that returns a bare number."""
    metric: str
    log: str
    from_step: int
    to_step: int
    n: int
    r2: float
    _exponent: float
    ci_lo: float
    ci_hi: float
    intercept: float
    n_boot: int = N_BOOT
    r2_floor: float = R2_FLOOR

    @property
    def supported(self) -> bool:
        return self.r2 >= self.r2_floor

    @property
    def exponent(self) -> float:
        if not self.supported:
            raise ValueError(
                f"REFUSING to report an exponent for {self.metric}: R^2="
                f"{self.r2:.3f} < {self.r2_floor} over steps "
                f"{self.from_step}-{self.to_step} (n={self.n}). The power law "
                f"does not describe the data; use the matched-step ratio.")
        return self._exponent

    def render(self) -> str:
        span = f"steps {self.from_step}-{self.to_step}, n={self.n}, R^2={self.r2:.3f}"
        if not self.supported:
            return (f"{self.metric}: exponent UNSUPPORTED ({span}; R^2 < "
                    f"{self.r2_floor}) — power law does not describe the data; "
                    f"point slope would have been {self._exponent:+.3f}, "
                    f"bootstrap CI [{self.ci_lo:+.3f}, {self.ci_hi:+.3f}]")
        return (f"{self.metric}: exponent {self._exponent:+.3f} "
                f"CI [{self.ci_lo:+.3f}, {self.ci_hi:+.3f}] ({span}, "
                f"B={self.n_boot})")

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("_exponent")
        d["exponent"] = self._exponent if self.supported else None
        d["exponent_point_estimate_unsupported"] = (
            None if self.supported else self._exponent)
        d["supported"] = self.supported
        d["rendered"] = self.render()
        return d

    def project(self, at_step: int) -> float:
        """Extrapolated value at ``at_step``. Refuses beyond ``MAX_EXTRAP``x the
        fitted range, and refuses at all when the fit is unsupported."""
        _ = self.exponent                                   # raises if R^2 < floor
        if at_step > MAX_EXTRAP * self.to_step:
            raise ValueError(
                f"REFUSING to extrapolate {self.metric} to step {at_step}: "
                f"{at_step / self.to_step:.1f}x beyond the fitted range "
                f"(cap {MAX_EXTRAP}x, fit ended at {self.to_step}).")
        return float(math.exp(self.intercept + self._exponent * math.log(at_step)))


def fit_power_law(rows, metric, from_step, to_step, log_path="?",
                  n_boot=N_BOOT, seed=0) -> SlopeFit:
    """Log-log least squares over an EXPLICIT window, with a bootstrap CI.

    The window is mandatory: an exponent with no stated fit range is the exact
    defect this replaces (v1's -0.84 was quoted with none)."""
    s, v = series(rows, metric)
    m = (s >= from_step) & (s <= to_step) & (s > 0) & (v > 0)
    s, v = s[m], v[m]
    if s.size < 5:
        raise SystemExit(f"[gate] REFUSING: only {s.size} usable points for "
                         f"{metric} in [{from_step}, {to_step}] — too few to fit")
    x, y = np.log(s), np.log(v)
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else 0.0
    rng = np.random.default_rng(seed)
    slopes = np.empty(n_boot)
    idx = np.arange(x.size)
    for i in range(n_boot):
        pick = rng.choice(idx, size=idx.size, replace=True)
        if np.ptp(x[pick]) == 0:
            slopes[i] = np.nan
            continue
        slopes[i] = np.polyfit(x[pick], y[pick], 1)[0]
    lo, hi = np.nanpercentile(slopes, [2.5, 97.5])
    return SlopeFit(metric=metric, log=str(log_path), from_step=int(s.min()),
                    to_step=int(s.max()), n=int(s.size), r2=r2,
                    _exponent=float(b), ci_lo=float(lo), ci_hi=float(hi),
                    intercept=float(a), n_boot=n_boot)


# =========================================================================== #
# The comparative statistic — matched steps, no power law                     #
# =========================================================================== #
def _at(s, v, step, tol):
    m = np.abs(s - step) <= tol
    return float(v[m].mean()) if m.any() else float("nan")


def matched_step_ratio(new_rows, ref_rows, metric, at_steps=None, tol=100,
                       n_boot=N_BOOT, seed=0) -> dict:
    """r(s) = M_new(s) / M_ref(s) at IDENTICAL steps. No extrapolation, no fit.

    The CI resamples the matched step points (a bootstrap over the comparison's
    own units). It is honestly labelled ``log_point_bootstrap``: for a
    DECISION-grade interval the milestone path uses the episode-cluster
    bootstrap on held-out windows (``taniteval/ci.py``) instead."""
    sn, vn = series(new_rows, metric)
    sr, vr = series(ref_rows, metric)
    if at_steps is None:
        at_steps = [int(x) for x in sn if x > 0]
    pts = []
    for st in at_steps:
        a, b = _at(sn, vn, st, tol), _at(sr, vr, st, tol)
        if np.isfinite(a) and np.isfinite(b) and b > 0:
            pts.append((st, a, b, a / b))
    if not pts:
        raise SystemExit(f"[gate] REFUSING: no matched steps for {metric}")
    r = np.array([p[3] for p in pts])
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(r, size=r.size, replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    tail = pts[-max(1, len(pts) // 5):]                     # last 20 % of steps
    return {"metric": metric, "n_matched_steps": len(pts),
            "step_range": [pts[0][0], pts[-1][0]],
            "ratio_mean": round(float(r.mean()), 4),
            "ratio_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "ratio_first": round(float(pts[0][3]), 4),
            "ratio_last": round(float(pts[-1][3]), 4),
            "ratio_tail_mean": round(float(np.mean([p[3] for p in tail])), 4),
            "widening": bool(pts[-1][3] > pts[0][3]),
            "estimator": "log_point_bootstrap (diagnostic-grade; the "
                         "decision-grade interval is the episode-cluster "
                         "bootstrap on held-out windows)",
            "n_boot": n_boot,
            "per_step": [{"step": p[0], "new": round(p[1], 4),
                          "ref": round(p[2], 4), "ratio": round(p[3], 4)}
                         for p in pts]}


def reference_reached_at(ref_rows, metric, value, k=REACHED_K,
                         bucket=REACHED_BUCKET) -> dict:
    """The step at which the REFERENCE run first reached ``value``.

    Assumption-free: no power law, no extrapolation, no exponent.

    DEFECT FIXED 2026-07-21 -- the 3-point rolling median this used to apply was
    not robust enough for a PER-BATCH (B=16) train metric. On v1's raw
    ``g_op_fwd_ade_m`` the logged rows at steps 300-550 are
    ``0.758 0.616 0.404 0.687 0.384 0.816`` -- adjacent rows swing ~2x, so a
    3-point median still passes isolated dips straight through. It reported
    "v1 reached 0.4101 at step 450" when v1's BUCKET means only reach ~0.41 in
    the 2k-4k range: ~5x too early, and that fed a "~23x more step-efficient"
    claim into MODEL_REGISTRY.md 1.4 (retracted). See
    ``Research/2026-07-21-flagship-v3enc-postmortem.md`` 7.1.

    Two robust rules replace it, and BOTH are returned:

    * **primary** -- ``reached_at_step`` is the first step from which ``k``
      CONSECUTIVE logged rows all sit at/below the target. A run of k-1 lucky
      draws can no longer fabricate a crossing.
    * **corroborating** -- the first fixed-width ``bucket``-step window whose
      MEAN is at/below the target, reported as an interval, never a point.

    ``estimator`` states the rule in words and is non-optional, so the number
    cannot be quoted bare (the failure mode that produced the 23x claim and,
    before it, the retired exponent gate). Disagreement between the two rules is
    surfaced as ``rules_agree``: when they disagree the statistic is soft and the
    interval, not the point, is what may be quoted."""
    s, v = series(ref_rows, metric)
    k = max(1, int(k))
    rule = (f"first step from which {k} CONSECUTIVE logged rows are all "
            f"<= target; corroborated by the first {bucket}-step bucket whose "
            f"MEAN is <= target. NOT a rolling median and NOT a single row -- "
            f"this metric is per-batch and swings ~2x between adjacent rows, "
            f"so any single-row rule reads ~5x too early (see docstring).")
    if v.size < k:
        return {"target_value": round(float(value), 4), "reached_at_step": None,
                "reference_final": None, "smoothing": rule, "estimator": rule,
                "k_consecutive": k, "bucket_steps": int(bucket),
                "note": f"too few reference points ({v.size}) for k={k}"}

    # --- primary: k consecutive crossings ---------------------------------- #
    below = v <= value
    run = np.convolve(below.astype(int), np.ones(k, dtype=int), mode="valid")
    hit = np.flatnonzero(run == k)
    reached = int(s[hit[0]]) if hit.size else None

    # --- corroborating: fixed-width bucket means --------------------------- #
    b_lo = b_hi = b_mean = None
    if bucket and bucket > 0 and s.size:
        w = float(bucket)
        for lo in np.arange(0.0, float(s.max()) + w, w):
            m = (s >= lo) & (s < lo + w)
            if m.any() and float(v[m].mean()) <= value:
                b_lo, b_hi = int(lo), int(lo + w)
                b_mean = round(float(v[m].mean()), 4)
                break
        # final level = mean of the LAST bucket that actually holds rows, never
        # a single row (s.max() rarely lands on a bucket edge)
        last_lo = np.floor(float(s.max()) / w) * w
        tail = s >= last_lo
        final = float(v[tail].mean())
    else:
        final = float(np.mean(v[-min(v.size, k):]))

    agree = (reached is not None and b_lo is not None
             and b_lo <= reached <= b_hi)
    return {"target_value": round(float(value), 4),
            "reached_at_step": reached,
            "reached_in_bucket": None if b_lo is None else [b_lo, b_hi],
            "bucket_mean_at_crossing": b_mean,
            "reference_final": round(final, 4),
            "smoothing": rule,
            "estimator": rule,
            "k_consecutive": k,
            "bucket_steps": int(bucket),
            "rules_agree": bool(agree),
            "n_reference_points": int(v.size)}


# =========================================================================== #
# Pre-registration card                                                       #
# =========================================================================== #
@dataclass
class GateCard:
    run: str
    gate_step: int
    primary_metric: str
    primary_threshold: float
    primary_direction: str                 # "<=" or ">="
    primary_source: str                    # must be a HELD-OUT eval, never train
    secondary: list = field(default_factory=list)   # ["name>=0.55", ...]
    reference_run: str | None = None
    reference_log: str | None = None
    compare_metric: str | None = None
    tau: float = DEFAULT_TAU
    lever_family: str | None = None
    restarts_used: int = 0
    restart_cap: int = DEFAULT_RESTART_CAP
    registered_utc: str = ""
    note: str = ""

    def check_restart_budget(self) -> tuple[bool, str]:
        if self.restarts_used >= self.restart_cap:
            return False, (f"restart cap reached for lever family "
                           f"{self.lever_family!r}: {self.restarts_used}/"
                           f"{self.restart_cap}. A further failure REFUTES the "
                           f"lever family; it does not license more tuning.")
        return True, (f"restart budget {self.restarts_used}/{self.restart_cap} "
                      f"for lever family {self.lever_family!r}")


def _parse_secondary(spec: str) -> tuple[str, str, float]:
    for op in (">=", "<=", ">", "<"):
        if op in spec:
            name, val = spec.split(op, 1)
            return name.strip(), op, float(val)
    raise SystemExit(f"[gate] bad --secondary {spec!r}; use name>=value")


def _lookup_secondary_value(supplied, name):
    """``(value, matched_key)`` for a card secondary ``name`` from the supplied
    ``--secondary-value`` dict, trying every alias so a value passed as
    ``miss_2m`` still satisfies a card that names ``miss_at_2m``. ``matched_key``
    is the supplied key that was consumed (used to keep it out of the
    report-only channel) or ``None`` if nothing matched."""
    if not supplied:
        return None, None
    for alias in _metric_aliases(name):
        if alias in supplied:
            return supplied[alias], alias
    return None, None


# =========================================================================== #
# Sub-commands                                                                #
# =========================================================================== #
def cmd_register(a):
    card_path = Path(a.card)
    if card_path.exists() and not a.force:
        raise SystemExit(f"[gate] REFUSING to overwrite an existing gate card "
                         f"{card_path}. Pre-registration means it is written "
                         f"ONCE, before launch. Use --force only to correct a "
                         f"card before the run starts.")
    for s in a.secondary:
        _parse_secondary(s)
    card = GateCard(run=a.run, gate_step=a.gate_step,
                    primary_metric=a.primary_metric,
                    primary_threshold=a.primary_threshold,
                    primary_direction=a.primary_direction,
                    primary_source=a.primary_source, secondary=list(a.secondary),
                    reference_run=a.reference_run, reference_log=a.reference_log,
                    compare_metric=a.compare_metric, tau=a.tau,
                    lever_family=a.lever_family, restarts_used=a.restarts_used,
                    restart_cap=a.restart_cap,
                    registered_utc=datetime.now(timezone.utc).isoformat(),
                    note=a.note)
    ok, msg = card.check_restart_budget()
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(json.dumps(asdict(card), indent=2), encoding="utf-8")
    print(f"[gate] pre-registered {card_path}")
    print(json.dumps(asdict(card), indent=2))
    print(f"[gate] {'OK' if ok else 'BLOCKED'}: {msg}")
    return 0 if ok else 2


def cmd_check(a):
    card = GateCard(**json.loads(Path(a.card).read_text()))
    rows = read_log(a.log)
    cur = rows[-1]["step"]
    gh, sps = gpu_hours(rows), s_per_step(rows)
    out = {"run": card.run, "card": str(a.card), "current_step": cur,
           "gate_step": card.gate_step, "gpu_hours": round(gh, 2),
           "s_per_step": round(sps, 2)}

    print(f"\n=== GATE: {card.run} ===")
    print(f"pre-registered {card.registered_utc} — gate step {card.gate_step}, "
          f"primary {card.primary_metric} {card.primary_direction} "
          f"{card.primary_threshold} from {card.primary_source}")
    print(f"current step {cur} | {gh:.1f} GPU-h | {sps:.2f} s/step")

    ok_budget, budget_msg = card.check_restart_budget()
    print(f"[restart budget] {budget_msg}")
    out["restart_budget_ok"] = ok_budget

    # --- comparative diagnostics (always shown, never the decision) --------- #
    if card.reference_log and card.compare_metric and Path(
            a.reference_log or card.reference_log).exists():
        ref = read_log(a.reference_log or card.reference_log)
        ratio = matched_step_ratio(rows, ref, card.compare_metric)
        s, v = series(rows, card.compare_metric)
        reached = reference_reached_at(ref, card.compare_metric, float(v[-1]))
        ref_sps = s_per_step(ref)
        out["matched_step_ratio"] = ratio
        out["reference_reached_at"] = reached
        out["budget_normalization"] = {
            "new_s_per_step": round(sps, 2), "ref_s_per_step": round(ref_sps, 2),
            "steps_per_gpu_hour_new": round(3600.0 / sps, 1) if sps else None,
            "steps_per_gpu_hour_ref": round(3600.0 / ref_sps, 1) if ref_sps else None,
            "note": ("equal-step is NOT equal-cost; the run with the smaller "
                     "s/step gets more steps per GPU-hour")}
        print(f"\n[comparative] matched-step ratio vs {card.reference_run} on "
              f"{card.compare_metric} over steps {ratio['step_range']}:")
        print(f"  mean {ratio['ratio_mean']} CI {ratio['ratio_ci95']} | "
              f"first {ratio['ratio_first']} -> last {ratio['ratio_last']} "
              f"({'WIDENING' if ratio['widening'] else 'closing/flat'})")
        print(f"  reference reached the new run's current value "
              f"({reached['target_value']}) at step {reached['reached_at_step']} "
              f"[{reached['k_consecutive']} consecutive rows; bucket-mean "
              f"crossing {reached['reached_in_bucket']}"
              f"{'' if reached['rules_agree'] else ' -- RULES DISAGREE, quote the bucket'}]")
        print(f"  budget: new {sps:.2f} s/step vs ref {ref_sps:.2f} s/step "
              f"-> {3600.0/sps:.0f} vs {3600.0/ref_sps:.0f} steps/GPU-h")

    if a.fit_metric:
        fit = fit_power_law(rows, a.fit_metric, a.fit_from, a.fit_to, a.log)
        out["exponent_diagnostic"] = fit.to_json()
        print(f"\n[diagnostic only — NOT a gate] {fit.render()}")

    # --- the PRIMARY gate --------------------------------------------------- #
    if cur < card.gate_step:
        out["verdict"] = "NOT_YET"
        out["reason"] = (f"step {cur} < pre-registered gate step "
                         f"{card.gate_step}. No kill/continue call is admissible "
                         f"before the registered step — that is the whole point "
                         f"of pre-registration.")
        print(f"\nVERDICT: NOT_YET — {out['reason']}")
        _emit(a, out)
        return 0

    if not a.eval_json:
        out["verdict"] = "BLOCKED"
        out["reason"] = ("no --eval-json. The primary gate is a HELD-OUT "
                         "milestone metric; this tool refuses to decide from a "
                         "train-log slope (REF-A: train fwd-ADE 0.65 vs held-out "
                         "2.92 — a 4.5x dissociation).")
        print(f"\nVERDICT: BLOCKED — {out['reason']}")
        _emit(a, out)
        return 3

    ev = json.loads(Path(a.eval_json).read_text())
    val, prov = _read_eval_metric(ev, card.primary_metric)
    passed = (val <= card.primary_threshold if card.primary_direction == "<="
              else val >= card.primary_threshold)
    out["primary"] = {"metric": card.primary_metric, "value": val,
                      "threshold": card.primary_threshold,
                      "direction": card.primary_direction, "pass": bool(passed),
                      "provenance": prov}
    print(f"\n[primary] {card.primary_metric} = {val} (from {prov}) "
          f"{card.primary_direction} {card.primary_threshold} -> "
          f"{'PASS' if passed else 'FAIL'}")

    sec_ok = True
    out["secondary"] = []
    matched_supplied = set()          # supplied keys a card (KILL) secondary ate
    for spec in card.secondary:
        name, op, thr = _parse_secondary(spec)
        sv, sv_key = _lookup_secondary_value(a.secondary_value, name)
        if sv_key is not None:
            matched_supplied.add(sv_key)
        if sv is None:
            sec_ok = False
            out["secondary"].append({"metric": name, "value": None,
                                     "pass": None, "note": "NOT SUPPLIED"})
            print(f"[secondary] {name}: NOT SUPPLIED -> gate cannot complete "
                  f"(pass --secondary-value {name}=<v>)")
            continue
        p = (sv <= thr if op in ("<=", "<") else sv >= thr)
        sec_ok &= p
        row = {"metric": name, "value": sv, "op": op, "threshold": thr,
               "pass": bool(p)}
        if sv_key != name:
            row["supplied_as"] = sv_key
        out["secondary"].append(row)
        print(f"[secondary] {name} = {sv} {op} {thr} -> {'PASS' if p else 'FAIL'}")

    # --- report-only falsifiers (§9 split card) ---------------------------- #
    # Any --secondary-value NOT consumed by a card (KILL) secondary is a
    # REPORT-ONLY read: it is emitted into the gate JSON and printed, but it does
    # NOT adjudicate the verdict. Design §9 requires the 5 v4 falsifiers
    # (imag_win_at_5s, strat_subspace_{sufficiency,compression},
    # longh_5s_beats_persistence, cruise_delta_vs_holdv0) to "land in the gate
    # JSON ... read at the gate, they just do not adjudicate it". The old code
    # discarded them silently. They can never move the KILL verdict.
    out["report_only"] = []
    for key, val in (a.secondary_value or {}).items():
        if key in matched_supplied:
            continue
        out["report_only"].append(
            {"metric": key, "value": val, "adjudicated": False,
             "note": "report-only (off-card): read + emitted, does NOT affect "
                     "the verdict (§9 split card)"})
        print(f"[report-only] {key} = {val} (off-card; recorded, NOT adjudicated)")

    if any(s["pass"] is None for s in out["secondary"]):
        out["verdict"] = "INCOMPLETE"
        out["reason"] = "a pre-registered secondary gate was not measured"
    elif passed and sec_ok:
        out["verdict"] = "CONTINUE"
        out["reason"] = "all pre-registered gates pass"
    else:
        out["verdict"] = "RESTART" if ok_budget else "REFUTE_LEVER_FAMILY"
        out["reason"] = ("pre-registered gate failed" if ok_budget else
                         "pre-registered gate failed AND the restart cap for "
                         "this lever family is exhausted — the lever family is "
                         "refuted, not the schedule")
    print(f"\nVERDICT: {out['verdict']} — {out['reason']}")
    _emit(a, out)
    return 0


def _dig(ev, path):
    """Follow a key path through nested dicts, returning {} on any miss."""
    node = ev
    for key in path:
        node = node.get(key, {}) if isinstance(node, dict) else {}
    return node


def _cluster_node(ev, aliases):
    """First cluster-bootstrap-grade metric node for any alias, and where it was
    found. Searches bench's ``cluster_bootstrap.model`` block, the merged
    driving block, and either module's ``headline``. A node qualifies ONLY if it
    is an interval dict that NAMES ``episode_cluster_bootstrap`` — so the
    deprecated interval can never masquerade as the primary."""
    for path, where in ((("cluster_bootstrap", "model"), "cluster_bootstrap"),
                        (("driving", "cluster_bootstrap", "model"),
                         "driving.cluster_bootstrap"),
                        (("headline",), "headline"),
                        (("driving", "headline"), "driving.headline")):
        node = _dig(ev, path)
        for alias in aliases:
            m = node.get(alias) if isinstance(node, dict) else None
            if (isinstance(m, dict) and "mean" in m
                    and m.get("estimator") == CLUSTER_BOOTSTRAP_ESTIMATOR):
                return m, where, alias
    return None, None, None


def _deprecated_present(ev, aliases):
    """True iff the JSON carries the forbidden ``overlapping_holdout_se``
    (``heldout``) interval for the metric. Used ONLY when no cluster bootstrap
    exists, to decide between fail-loud and a clean point-estimate fallback."""
    for base in (("heldout", "model"), ("driving", "heldout", "model")):
        model = _dig(ev, base)
        for alias in aliases:
            n = model.get(alias) if isinstance(model, dict) else None
            if isinstance(n, dict) and (
                    "mean" in n or n.get("estimator") == DEPRECATED_ESTIMATOR):
                return True
    return False


def _read_eval_metric(ev, metric):
    """Pull ``metric`` from a taniteval result JSON for a DECISION-grade gate.

    Contract (⭐ the most dangerous of the three v4 gate bugs, fixed 2026-07-22):

    1. **PREFER the episode-cluster bootstrap.** Any block whose node names
       ``episode_cluster_bootstrap`` is admissible (bench's
       ``cluster_bootstrap.model``, the merged driving block, or either module's
       ``headline``).
    2. **FAIL LOUD** — raise, never warn — if the cluster bootstrap is absent but
       the DEPRECATED ``overlapping_holdout_se`` (``heldout``) block is present.
       That estimator is 1.28-2.06x too narrow (CLAUDE.md;
       CI_RECOMPUTE_2026-07-20.json) and MUST NOT silently decide a gate. The old
       code fell back to it, so EVERY v4 gate would have adjudicated on the
       forbidden statistic — and on a different (anti-conservative) mean.
    3. Only a NON-deprecated point estimate (``full_set`` / top-level) is an
       admissible fallback, and it is labelled as carrying NO interval.

    Metric names are alias-resolved (``miss_at_2m`` <-> ``miss_2m`` <->
    ``miss_rate@2m``) so the card's canonical name always finds the emitter's.
    """
    aliases = _metric_aliases(metric)

    node, where, alias = _cluster_node(ev, aliases)
    if node is not None:
        lo, hi = node.get("lo"), node.get("hi")
        extra = (f" CI [{lo}, {hi}]" if lo is not None else
                 f" +-{node['ci95']}" if node.get("ci95") is not None else "")
        as_note = "" if alias == metric else f" (as {alias!r})"
        return float(node["mean"]), (
            f"cluster_bootstrap (primary, {CLUSTER_BOOTSTRAP_ESTIMATOR}; "
            f"{where}){as_note}{extra}")

    if _deprecated_present(ev, aliases):
        raise SystemExit(
            f"[gate] REFUSING: {metric!r} has ONLY the DEPRECATED "
            f"{DEPRECATED_ESTIMATOR!r} interval (the 'heldout' block) and no "
            f"{CLUSTER_BOOTSTRAP_ESTIMATOR!r} block. That estimator is "
            f"1.28-2.06x too narrow (CLAUDE.md; CI_RECOMPUTE_2026-07-20.json) "
            f"and may not decide a gate. Re-run the eval so it emits a "
            f"cluster_bootstrap block (bench.run / driving.tier0 both do), then "
            f"re-check.")

    # No interval at all: a NON-deprecated point estimate is admissible, but is
    # labelled as carrying no interval so nobody mistakes it for one.
    for path, where in ((("full_set", "model"), "full_set"),
                        (("driving", "full_set", "model"), "driving.full_set")):
        node = _dig(ev, path)
        for alias in aliases:
            v = node.get(alias) if isinstance(node, dict) else None
            if v is not None and not isinstance(v, dict):
                as_note = "" if alias == metric else f" (as {alias!r})"
                return float(v), (f"{where} (point estimate, NO interval — "
                                  f"cluster_bootstrap preferred){as_note}")
    for alias in aliases:
        if alias in ev and not isinstance(ev[alias], dict):
            return float(ev[alias]), "top-level (point estimate, NO interval)"
    raise SystemExit(f"[gate] REFUSING: {metric!r} (aliases {list(aliases)}) "
                     f"not found in the eval JSON")


def _emit(a, out):
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"-> {a.json}")


def cmd_fit(a):
    rows = read_log(a.log)
    fit = fit_power_law(rows, a.metric, getattr(a, "from"), a.to, a.log,
                        n_boot=a.n_boot)
    print(fit.render())
    print(f"  [reminder] this is a DIAGNOSTIC. It may not decide a restart.")
    if a.project:
        try:
            print(f"  projection @ {a.project}: {fit.project(a.project):.4f}")
        except ValueError as e:
            print(f"  projection @ {a.project}: {e}")
    if a.json:
        Path(a.json).write_text(json.dumps(fit.to_json(), indent=2), encoding="utf-8")
    return 0 if fit.supported else 1


def cmd_ratio(a):
    new, ref = read_log(a.log), read_log(a.reference_log)
    r = matched_step_ratio(new, ref, a.metric, n_boot=a.n_boot)
    s, v = series(new, a.metric)
    reached = reference_reached_at(ref, a.metric, float(v[-1]))
    print(json.dumps({"matched_step_ratio": {k: r[k] for k in r if k != "per_step"},
                      "reference_reached_at": reached,
                      "new_s_per_step": round(s_per_step(new), 2),
                      "ref_s_per_step": round(s_per_step(ref), 2),
                      "new_gpu_hours": round(gpu_hours(new), 2),
                      "ref_gpu_hours_at_same_steps": None}, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"matched_step_ratio": r, "reference_reached_at": reached}, indent=2),
            encoding="utf-8")
    return 0


class _KV(argparse.Action):
    def __call__(self, parser, ns, values, option_string=None):
        d = getattr(ns, self.dest) or {}
        for v in values:
            k, _, val = v.partition("=")
            d[k.strip()] = float(val)
        setattr(ns, self.dest, d)


def main(argv=None):
    ap = argparse.ArgumentParser("run_gate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register", help="pre-register the gate BEFORE launch")
    r.add_argument("--run", required=True)
    r.add_argument("--gate-step", type=int, required=True)
    r.add_argument("--primary-metric", default="ade_0_2s")
    r.add_argument("--primary-threshold", type=float, required=True)
    r.add_argument("--primary-direction", default="<=", choices=["<=", ">="])
    r.add_argument("--primary-source", default="held-out taniteval eval JSON")
    r.add_argument("--secondary", action="append", default=[])
    r.add_argument("--reference-run", default=None)
    r.add_argument("--reference-log", default=None)
    r.add_argument("--compare-metric", default=None)
    r.add_argument("--tau", type=float, default=DEFAULT_TAU)
    r.add_argument("--lever-family", default=None)
    r.add_argument("--restarts-used", type=int, default=0)
    r.add_argument("--restart-cap", type=int, default=DEFAULT_RESTART_CAP)
    r.add_argument("--note", default="")
    r.add_argument("--card", required=True)
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_register)

    c = sub.add_parser("check", help="evaluate a run against its registered card")
    c.add_argument("--card", required=True)
    c.add_argument("--log", required=True)
    c.add_argument("--reference-log", default=None)
    c.add_argument("--eval-json", default=None)
    c.add_argument("--secondary-value", nargs="*", action=_KV, default={})
    c.add_argument("--fit-metric", default=None, help="log an exponent DIAGNOSTIC")
    c.add_argument("--fit-from", type=int, default=1500)
    c.add_argument("--fit-to", type=int, default=7500)
    c.add_argument("--json", default=None)
    c.set_defaults(fn=cmd_check)

    f = sub.add_parser("fit", help="DIAGNOSTIC power-law fit (never a decision)")
    f.add_argument("--log", required=True)
    f.add_argument("--metric", required=True)
    f.add_argument("--from", type=int, required=True, dest="from")
    f.add_argument("--to", type=int, required=True)
    f.add_argument("--project", type=int, default=None)
    f.add_argument("--n-boot", type=int, default=N_BOOT)
    f.add_argument("--json", default=None)
    f.set_defaults(fn=cmd_fit)

    q = sub.add_parser("ratio", help="matched-step ratio vs a reference run")
    q.add_argument("--log", required=True)
    q.add_argument("--reference-log", required=True)
    q.add_argument("--metric", required=True)
    q.add_argument("--n-boot", type=int, default=N_BOOT)
    q.add_argument("--json", default=None)
    q.set_defaults(fn=cmd_ratio)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
