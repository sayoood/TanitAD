"""Route-counterfactual sensitivity — does the planner USE its route input?

THE MISSING INSTRUMENT
----------------------
``route_skill_vs_chance = 0.0`` and ``nonav_route_beats_majority = 0`` say the
route HEAD is a command echo. Neither says anything about the direction that
actually matters for the hierarchy thesis: **does the route INPUT change the
decoded trajectory at all?** Every published REF-C number is decoded with
``nav_cmd=None`` (constant ``follow``), so that question has never been asked of
any arm — the C6 confound logged twice (RETRACTION_LOG 2026-07-21, 2026-07-25).

This module asks it, with the discipline CLAUDE.md requires of a new metric:

* **Three arms, not two.** ``base`` (the route the window really has),
  ``treatment`` (a counterfactual route), and ``control`` (a bit-identical copy
  of ``base``). The control MUST return exactly 0 — it is the proof the harness
  is deterministic, so any nonzero treatment number is signal and not decode
  noise. A metric is not quoted until its negative control has fired.
* **Two treatments, because "it moved" is not "it complied."** ``mirror`` flips
  left/right while preserving ‖features‖ (topology changed, energy constant);
  ``straight`` is the majority-class route. On a 74 %-straight corpus a model
  that merely reacts to signal energy separates on neither.
* **Signed compliance, not just magnitude.** ``lat_compliance`` is the fraction
  of windows whose predicted endpoint moves in the SAME lateral direction as the
  commanded change. Magnitude alone cannot distinguish "follows the route" from
  "is destabilised by a changed input".

Estimator: per-window components are returned so the caller runs the paired
episode-cluster bootstrap (``taniteval.ci``) over episodes. This module never
computes an interval from a split-mean.

Model-agnostic: the caller supplies ``predict(cond) -> [B, S, 2]``. It works on
the DEPLOYED REF-C checkpoint through :func:`nav_cmd_sensitivity` (4 discrete
commands, forward passes only, no retraining) and on a LAN-conditioned model
through :func:`lan_sensitivity`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A control displacement above this is a harness fault, not a finding: the two
# arms were bit-identical inputs, so anything nonzero is nondeterminism (train
# -mode diffusion noise, dropout left on, a non-seeded sampler).
CONTROL_TOL_M = 1e-6


@dataclass(frozen=True)
class RouteCFResult:
    """Per-window components + the aggregate. Components are the quotable unit."""

    name: str
    disp_m: np.ndarray          # [N] mean-over-horizons L2 shift, base vs treat
    lat_delta_m: np.ndarray     # [N] signed lateral shift of the endpoint
    commanded_m: np.ndarray     # [N] signed lateral change the route commanded
    control_disp_m: np.ndarray  # [N] base vs an identical copy — must be ~0
    n_windows: int

    @property
    def sensitivity_m(self) -> float:
        return float(self.disp_m.mean()) if self.n_windows else float("nan")

    @property
    def control_m(self) -> float:
        return (float(np.abs(self.control_disp_m).max())
                if self.n_windows else float("nan"))

    @property
    def discriminative(self) -> bool:
        """The negative control fired: the harness is deterministic."""
        return bool(self.n_windows) and self.control_m <= CONTROL_TOL_M

    @property
    def lat_compliance(self) -> float:
        """Fraction of windows moving the way the route asked (ties excluded).

        Ties (commanded change exactly 0 — e.g. an already-straight route under
        ``mirror``) are EXCLUDED rather than counted as agreement; counting them
        would inflate compliance toward the base rate, which is precisely how
        the route head's 0.0 skill hid behind a 74 %-straight corpus.
        """
        m = np.abs(self.commanded_m) > 1e-9
        if not m.any():
            return float("nan")
        return float((np.sign(self.lat_delta_m[m])
                      == np.sign(self.commanded_m[m])).mean())

    @property
    def n_decided(self) -> int:
        return int((np.abs(self.commanded_m) > 1e-9).sum())

    def summary(self) -> dict:
        return {"treatment": self.name,
                "n_windows": self.n_windows,
                "n_decided": self.n_decided,
                "route_sensitivity_m": round(self.sensitivity_m, 6),
                "lat_compliance": (None if np.isnan(self.lat_compliance)
                                   else round(self.lat_compliance, 4)),
                "control_max_disp_m": round(self.control_m, 12),
                "discriminative": self.discriminative,
                "control_tol_m": CONTROL_TOL_M,
                "verdict": self._verdict()}

    def _verdict(self) -> str:
        if not self.n_windows:
            return "NO-DATA"
        if not self.discriminative:
            return ("INSTRUMENT-FAIL — the identical-input control moved by "
                    f"{self.control_m:.3e} m > {CONTROL_TOL_M:.0e}; decode is "
                    "nondeterministic, so no sensitivity number is admissible")
        if self.sensitivity_m <= CONTROL_TOL_M:
            return ("INERT — the route input does not change the decoded "
                    "trajectory at all. The route pathway is structurally "
                    "unused; a route metric computed on this arm reports the "
                    "marginal (this is the C6 confound, measured)")
        return "RESPONSIVE — pair with the episode-cluster bootstrap to decide"


def _mean_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[N, S, 2] pair -> [N] mean-over-horizons L2 (the program's ADE reducer)."""
    return np.linalg.norm(a - b, axis=-1).mean(axis=-1)


def route_counterfactual(predict, base_cond, treat_cond, name: str,
                         commanded_lat_m=None) -> RouteCFResult:
    """Run the three arms and return per-window components.

    ``predict(cond) -> [N, S, 2]`` ego-frame trajectories. ``base_cond`` /
    ``treat_cond`` are whatever the caller's ``predict`` consumes. The control
    arm re-runs ``predict(base_cond)`` — the SAME object, so a difference can
    only come from the model or the harness, never from the input.
    """
    y0 = np.asarray(predict(base_cond), dtype=np.float64)
    y1 = np.asarray(predict(treat_cond), dtype=np.float64)
    y0b = np.asarray(predict(base_cond), dtype=np.float64)
    if y0.ndim != 3 or y0.shape[-1] != 2:
        raise ValueError(f"predict must return [N, S, 2], got {y0.shape}")
    if y1.shape != y0.shape or y0b.shape != y0.shape:
        raise ValueError(f"arms must be aligned: {y0.shape} / {y1.shape} / "
                         f"{y0b.shape}")
    n = y0.shape[0]
    lat_delta = y1[:, -1, 1] - y0[:, -1, 1]
    if commanded_lat_m is None:
        commanded = np.zeros((n,), dtype=np.float64)
    else:
        commanded = np.asarray(commanded_lat_m, dtype=np.float64).reshape(-1)
        if commanded.shape[0] != n:
            raise ValueError(f"commanded_lat_m must be [N]={n}, got "
                             f"{commanded.shape[0]}")
    return RouteCFResult(name=name,
                         disp_m=_mean_l2(y1, y0),
                         lat_delta_m=lat_delta,
                         commanded_m=commanded,
                         control_disp_m=_mean_l2(y0b, y0),
                         n_windows=n)


def commanded_lateral(base_feats, treat_feats, arclengths_m) -> np.ndarray:
    """Signed lateral change a LAN counterfactual asks for, in metres.

    Read off the FARTHEST anchor valid in both arms (the strategic end of the
    corridor), reconstructed as ``lat_norm * arc_length``. Windows with no
    commonly-valid anchor return 0 and are excluded from compliance by the tie
    rule in :attr:`RouteCFResult.lat_compliance`.
    """
    from tanitad.data.lan import LAN_FEATS_PER_ANCHOR

    a = np.asarray(base_feats, dtype=np.float64).reshape(
        -1, len(arclengths_m), LAN_FEATS_PER_ANCHOR)
    b = np.asarray(treat_feats, dtype=np.float64).reshape(a.shape)
    arc = np.asarray(arclengths_m, dtype=np.float64)
    both = (a[..., 3] > 0.5) & (b[..., 3] > 0.5)                 # [N, K]
    out = np.zeros((a.shape[0],), dtype=np.float64)
    for i in range(a.shape[0]):
        idx = np.flatnonzero(both[i])
        if idx.size:
            j = int(idx[-1])
            out[i] = (b[i, j, 2] - a[i, j, 2]) * arc[j]
    return out


def lan_sensitivity(predict, base_routes, treat_routes, arclengths_m,
                    name: str = "mirror") -> RouteCFResult:
    """Convenience wrapper: two lists of :class:`~tanitad.data.lan.LanRoute`."""
    base = np.stack([r.features for r in base_routes]).astype(np.float64)
    treat = np.stack([r.features for r in treat_routes]).astype(np.float64)
    return route_counterfactual(predict, base, treat, name,
                                commanded_lateral(base, treat, arclengths_m))


def nav_cmd_sensitivity(predict, n_windows: int, n_commands: int = 4) -> dict:
    """THE CHEAPEST DISCRIMINATING EXPERIMENT — runnable on a DEPLOYED arm.

    Sweep the existing 4-way ``nav_cmd`` over the same windows and measure how
    far the decoded trajectory moves. Forward passes only: no training, no new
    labels, no new data. ``predict(nav_idx) -> [N, S, 2]``.

    Pre-registered reading (both outcomes committed in PREREG_lan_refc.md):

    * ``max_pairwise_m ≈ 0`` ⇒ the route input is INERT in the deployed model.
      The C6 confound is then MEASURED rather than argued, and the 4-way command
      is confirmed as the defect LAN exists to fix.
    * ``max_pairwise_m`` materially > 0 ⇒ the input IS consumed, and evaluating
      at ``nav_cmd=None`` was discarding a live signal. The cheap fix is then to
      SUPPLY the label at eval, and LAN's marginal value drops accordingly.

    Either way the next step is decided by the number, not by a preference.
    """
    trajs = [np.asarray(predict(i), dtype=np.float64)
             for i in range(int(n_commands))]
    if any(t.shape != trajs[0].shape for t in trajs):
        raise ValueError("nav_cmd arms are not aligned across commands")
    if trajs[0].shape[0] != n_windows:
        raise ValueError(f"expected {n_windows} windows, got {trajs[0].shape[0]}")
    control = _mean_l2(np.asarray(predict(0), dtype=np.float64), trajs[0])
    pairs = {}
    per_window_max = np.zeros((n_windows,), dtype=np.float64)
    for i in range(len(trajs)):
        for j in range(i + 1, len(trajs)):
            d = _mean_l2(trajs[i], trajs[j])
            pairs[f"{i}v{j}"] = round(float(d.mean()), 6)
            per_window_max = np.maximum(per_window_max, d)
    ctrl = float(np.abs(control).max())
    discriminative = ctrl <= CONTROL_TOL_M
    mx = float(per_window_max.mean())
    return {"n_windows": int(n_windows),
            "n_commands": int(n_commands),
            "pairwise_mean_m": pairs,
            "max_pairwise_mean_m": round(mx, 6),
            "control_max_disp_m": round(ctrl, 12),
            "discriminative": discriminative,
            "per_window_max_m": per_window_max,
            "verdict": ("INSTRUMENT-FAIL — nondeterministic decode"
                        if not discriminative else
                        "INERT — nav_cmd does not move the trajectory"
                        if mx <= CONTROL_TOL_M else
                        "RESPONSIVE — nav_cmd moves the trajectory")}
