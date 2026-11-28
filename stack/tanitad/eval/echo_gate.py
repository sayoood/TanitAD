"""THE ANTI-ECHO GATE — can this arm beat its own dynamics?

Pre-registration: ``TanitAD Research Lab/Architecture & Inference/Research/
2026-09-03-refc-v4-design/PREREG_REFC_V4.md``. Hypotheses ``H-ECHO-1..3``.

WHY THIS EXISTS
============================================================================
REF-C v4 admits the MEASURED ego state at t0 — speed, longitudinal
acceleration, yaw rate, curvature — into the goal path (edge E11', PI
2026-09-03). That makes a trivial cheat available: integrate a
constant-acceleration, constant-curvature path and ignore the image entirely.

⛔⛔ THE CHEAT IS NOT SMALL, AND THAT IS THE WHOLE POINT. MEASURED 2026-09-04
on the val epcache (``physicalai-val-bb543bdf7836``, 40 episodes, 7,963
frames, stride 7; :func:`corpus_echo_report` reproduces it; banked at
``.../2026-09-03-refc-v4-design/raw/census_val40.json``):

    horizon   ha0 (const v)   ha0_ext (const v, a, kappa)   ext beats ha0 by
      2.0 s      0.7040 m            0.4449 m                   +36.80 %
      6.0 s      5.3547 m            4.4652 m                   +16.61 %

**0.4449 m at 2 s, from ZERO PIXELS**, is the same league as flagship v1's
deployed 0.452 m (``MODEL_REGISTRY.md``, ``flagship4b-speedjerk-30k``). So an
arm handed these channels can reproduce the programme's headline 2 s number
without ever reading the image — and every 2 s comparison made against the old
``ha0`` control has been scored against a bar that is 36 % too low.

⇒ **The bar is ``ha0_ext``, not ``ha0``.** This module computes it, pairs it
against the arm on the SAME windows, and refuses to certify an arm that cannot
clear it.

⚠️ CORRECTION, LOGGED RATHER THAN OVERWRITTEN. This block first recorded
**0.4476 / 4.5912 m (+36.42 % / +14.26 %)**. Re-running the CURRENT instrument
on the SAME 40 episodes and the SAME window counts (1,041 @2 s / 801 @6 s)
reproduces ``ha0`` exactly (0.70402 / 5.35474) and reads ``ha0_ext``
**0.44492 / 4.46515**. The mechanism is the stop clamp (tau clamped at -v0/a0
under deceleration, so the vehicle STOPS rather than reversing) landing after
that first measurement -- it can only lower the error, and it lowered both
cells. The banked JSON is the quotable number. Class: a derived constant
re-measured after its input changed, which is why the artifact and not the
docstring is the source.

⚠️ SCOPE OF THE TWO NUMBERS ABOVE, STATED. They are a corpus-kinematic property
of THOSE windows (stride-7 over 40 val episodes) — the design input and the
gate's construction. They are **NOT** decision numbers. :func:`echo_gate`
recomputes both controls on the same windows as the model it scores. Quoting
0.4449 against a differently-windowed arm is the ``df``/``step_s`` scope error
in a new costume, and this file will not do it for you.

WHAT THE GATE IS, IN THREE PARTS
============================================================================
1. :func:`echo_gate` — beat the controls, PER METRIC FAMILY, paired
   episode-cluster bootstrap. Five references, two of which must read a KNOWN
   value exactly or the panel is broken rather than passed.
2. :func:`ego_intervention_test` — BOTH directions. Perturbing ego must move
   the output (else the channels are dead: an UNPOWERED pass, C109) AND
   varying the scene must move the output (else the arm is ECHOING). An arm
   that passes only the first is the failure this module exists to name.
3. :func:`assert_not_echoing` — the gate raises. A rule with no mechanism is
   C108, and ``goal_admissibility`` sat with zero call sites for 12 days
   precisely because nothing ever raised.

⛔ AND A DELIBERATE-REGRESSION ARM IS MANDATORY (skill §2). An arm *designed*
to echo — v4 with the image ablated — must be FAILED by this gate. If it is
not, a PASS on the real arm means nothing, and that outcome (PREREG §7,
"OUTCOME IV") voids the panel rather than the model.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

__all__ = [
    "EchoViolation", "DT", "ha0", "ha0_ext", "ha_finite_diff_accel",
    "constant_only_reference", "raw_pixel_floor", "corpus_echo_report",
    "trajectory_families", "echo_gate", "ego_intervention_test",
    "source_ablation_test", "assert_not_echoing", "REQUIRED_REFERENCES",
]

#: ⛔ THE REFERENCES AN ARM MUST CLEAR — not one, and `ha0` is not among the
#: deciders. MEASURED on refcv3: `ha` 0.2996 BEATS the model's 0.4799 @30k
#: while `ha0` 0.6723 loses to it, so an `ha0`-only panel certifies an arm the
#: trivial control already beats by 2x. Both deciders are ego-extrapolations,
#: which is the point: `ha0_ext` is the sharpened `ha`, and an arm that clears
#: `ha` while TYING `ha0_ext` gained by echoing harder, not by seeing.
REQUIRED_REFERENCES: tuple[str, ...] = ("ha", "ha0_ext")

#: 10 Hz — BINDING (``V3_HORIZONS``, PLAN_STEPS=60).
DT: float = 0.1


class EchoViolation(RuntimeError):
    """Raised by :func:`assert_not_echoing`: the arm did not clear its own
    kinematic self-extrapolation, or it responds to ego and not to the scene."""


# --------------------------------------------------------------------------- #
# the controls                                                                #
# --------------------------------------------------------------------------- #
def ha0(v0, taus_s: Sequence[float]):
    """Constant velocity at the measured ``v0`` (a = 0, kappa = 0). ``[B,K,4]``.

    The programme's existing hold-action control. ⚠️ It is now known to be
    **too weak a bar** for any arm that reads acceleration or yaw rate: on the
    val epcache ``ha0_ext`` beats it by 36.80 % at 2 s. Kept because the
    ha0 -> ha0_ext gap is itself the quantity that says how much of an arm's
    margin is echo.
    """
    import torch
    from tanitad.refs.refc_v3 import kinematic_goal_extrapolation
    v0 = torch.as_tensor(v0)
    z = torch.zeros_like(v0)
    return kinematic_goal_extrapolation(v0, z, z, taus_s)


def ha_finite_diff_accel(v_t0, v_prev, steer_prev, taus_s: Sequence[float], *,
                         dt: float = DT, wheelbase: float = 2.9):
    """``ha`` — the HARNESS's hold-action control, reproduced exactly.

    ⭐⭐ ``ha`` IS ``ha0_ext``. Read at source (``taniteval/tools/refcv3_arm.py::
    hold_controls``), the harness's hold-action arm holds
    ``a = (v[t0] - v[t0-1]) / DT`` and ``kappa = actions[t0-1, 0]`` for the
    whole horizon — i.e. **constant longitudinal acceleration and constant
    curvature from the measured t0 state**, which is this module's echo,
    arrived at independently. The two differ in exactly two details, both of
    which make ``ha`` the WEAKER form:

      1. it derives the accel by FINITE DIFFERENCE of speed, where the corpus
         carries its own measured ``ax``. MEASURED: those two correlate only
         **+0.5624** (``corpus_echo_report``), and ``physicalai.py:13`` says
         the finite difference *"differentiates interpolation noise and lags
         the true signal"*;
      2. it reads the steer at ``t0-1``, not ``t0``.

    ⛔⛔ AND IT ALREADY BEATS THE SHIPPED MODEL, AT BOTH CHECKPOINTS WE HAVE.
    MEASURED on refcv3, 4,823 windows / 141 episodes, paired episode-cluster
    bootstrap (`taniteval/results/openloop-suite-refcv3-30k-ckpt30000.json`,
    `taniteval/results/RESULT-refcv3-40284-openloop.md`):

        step 30,000   ha 0.2996 m  |  os 0.4799 m  |  ha0 0.6723 m
            os - ha0 = -0.1924 [-0.2525, -0.1379]  separated (beats the floor)
            os - ha  = +0.1803 [+0.1563, +0.2047]  separated THE WRONG WAY
        step 40,284   ha 0.2996 m  |  os 0.4419 m  |  ha0 0.6723 m
            os - ha0 = -0.2304                     (floor margin improved)
            os - ha  = +0.1423 [+0.1187, +0.1658]  STILL the wrong way

    ⚠ BOTH are quoted deliberately. Only the 30k number would be a stale
    source; only the 40,284 number would hide that 10,284 further steps closed
    just 21 % of the gap. `ha` and `ha0` are checkpoint-independent and read
    BIT-IDENTICALLY at both steps on the same 4,823 windows -- the internal
    control that the two reads sit on one surface.

    ⇒ the trivial control captures **twice** the model's margin, and a refcv4
    that beats ``ha0`` has proven nothing refcv3 does not already do. ⚠️ This
    RETRACTS ``refcv3_arm.py``'s line calling ``ha0`` *"the echo test's real
    bar"*: ``ha`` is 2.2x harder and it is the bar.

    ⚠️ AND THE TRAP THAT COMES WITH IT: ``ha`` is itself an ego-dynamics
    extrapolation, so an arm that beats ``ha`` **by echoing harder** has not
    succeeded either. That is precisely why :func:`ha0_ext` — the same control
    built on the corpus's own ``ax``, and therefore the STRONGER form — must
    sit beside it. Beating ``ha`` while TYING ``ha0_ext`` means the gain is
    echo, and the gate reads it that way.
    """
    import torch
    v_t0 = torch.as_tensor(v_t0)
    a = (v_t0 - torch.as_tensor(v_prev)) / dt
    k = torch.tan(torch.as_tensor(steer_prev)) / wheelbase
    return ha0_ext(v_t0, a, k, taus_s)


def ha0_ext(v0, a0, k0, taus_s: Sequence[float]):
    """⭐⭐ THE ECHO: constant ``a0`` AND constant curvature ``k0``. ``[B,K,4]``.

    The SHARPENED form of the harness's ``ha`` (see
    :func:`ha_finite_diff_accel`): same construction, but built on the corpus's
    OWN measured ``ax`` rather than a finite difference of speed, so it is the
    stronger control of the two and the one an arm must clear to claim it read
    the scene.

    ONE implementation, shared with the model's own E14 base
    (:func:`tanitad.refs.refc_v3.kinematic_goal_extrapolation`) — deliberately.
    A control re-implemented beside the thing it controls is a control that can
    drift away from it, and then the gate measures the drift instead of the
    model. Pinned by ``test_control_and_base_are_one_implementation``.
    """
    from tanitad.refs.refc_v3 import kinematic_goal_extrapolation
    return kinematic_goal_extrapolation(v0, a0, k0, taus_s)


def constant_only_reference(target):
    """⭐ THE CONTROL THAT MUST READ A KNOWN VALUE: predict the corpus mean.

    Returns the mean target broadcast over the batch. It explains **exactly
    zero** variance by construction, so ``r2(constant_only) == 0.0`` to machine
    precision. If a panel reports anything else for this row, the panel is
    broken and NO other row in it is admissible.

    (Skill §4, and the 2026-08-22 lesson: three of four estimator failures were
    caught ONLY because a control read the same value as the thing being
    measured.)
    """
    import torch
    t = torch.as_tensor(target)
    return t.mean(dim=0, keepdim=True).expand_as(t).clone()


def raw_pixel_floor(frames, target, *, fit_idx, score_idx, n_pix: int = 16,
                    lambdas=(1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)):
    """⭐ THE RAW-INPUT FLOOR: ridge from DOWNSAMPLED PIXELS -> target.

    A learned representation that does not beat raw input has added nothing.
    Returns ``(prediction_on_score_idx, meta)`` with ``n`` and ``d`` PRINTED in
    the meta, because ``n << d`` is underpowered BY CONSTRUCTION and reads as a
    negative if you do not look (2026-08-22 failure #4).

    ⛔ ``lambda`` is selected on a validation split carved from ``fit_idx``
    ONLY — never on ``score_idx``. Selecting on the scored split picks maximal
    regularisation, which shrinks the ridge to the constant predictor and
    scores a beautiful, meaningless 0.0000 with a zero-width CI (failure #3).
    """
    import torch
    import torch.nn.functional as F
    x = torch.as_tensor(frames).float()
    if x.dim() == 5:                       # [B, W, C, H, W'] -> last frame
        x = x[:, -1]
    x = F.adaptive_avg_pool2d(x, (n_pix, n_pix)).reshape(x.shape[0], -1)
    y = torch.as_tensor(target).float().reshape(x.shape[0], -1)
    fit = torch.as_tensor(fit_idx, dtype=torch.long)
    sc = torch.as_tensor(score_idx, dtype=torch.long)
    cut = max(1, int(0.8 * len(fit)))
    inner_fit, inner_val = fit[:cut], fit[cut:]
    if len(inner_val) == 0:                # degenerate tiny rig
        inner_val = inner_fit
    d = x.shape[1]
    xm, ym = x[inner_fit].mean(0, keepdim=True), y[inner_fit].mean(0, keepdim=True)

    def _solve(idx, lam):
        a = x[idx] - xm
        bt = y[idx] - ym
        g = a.T @ a + lam * torch.eye(d, dtype=a.dtype)
        return torch.linalg.solve(g, a.T @ bt)

    best, best_lam = None, None
    for lam in lambdas:
        w = _solve(inner_fit, float(lam))
        pv = (x[inner_val] - xm) @ w + ym
        err = float(((pv - y[inner_val]) ** 2).mean())
        if best is None or err < best:
            best, best_lam = err, float(lam)
    w = _solve(fit, best_lam)              # refit on all of FIT at the chosen lam
    pred = (x[sc] - xm) @ w + ym
    return pred, {"n_fit": int(len(fit)), "n_score": int(len(sc)), "d": int(d),
                  "lambda": best_lam, "lambda_selected_on": "inner val of FIT",
                  "underpowered_n_lt_d": bool(len(fit) < d),
                  "_reads": ("n < d means the validation CORRECTLY chooses "
                             "maximal regularisation and everything reads "
                             "0.0000 — that is underpowered by construction, "
                             "not an absence of signal")}


# --------------------------------------------------------------------------- #
# the corpus measurement (the prereg's cited artifact)                        #
# --------------------------------------------------------------------------- #
def corpus_echo_report(episodes: Sequence, *, horizons_s=(2.0, 6.0),
                       stride: int = 7, wheelbase: float = 2.9) -> dict:
    """Reproduce the PREREG §2.3/§2.4 measurement from episode objects.

    ``episodes`` carry ``.poses`` ``[T, 4]`` and ``.actions`` ``[T, 2]``. Emits
    the channel statistics, the finite-difference cross-check, and the
    ``ha0``/``ha0_ext`` ADEs — the numbers the pre-registration cites, so the
    claim is reproducible from an artifact rather than from a report.

    ⭐ The finite-difference row is the INDEPENDENT CROSS-CHECK the derived
    channels need: ``r0 = v0 * tan(steer)/L`` must correlate strongly and
    POSITIVELY with ``d/dt(unwrap(yaw))``, which comes from an entirely
    different corpus field (the orientation quaternion). MEASURED +0.9430. A
    sign or wheelbase-inversion error would read ~-0.94 or ~0 and would
    otherwise look exactly like a working channel.
    """
    import numpy as np
    rows = []
    for ep in episodes:
        a = np.asarray(getattr(ep, "actions", None), dtype=np.float64)
        p = np.asarray(getattr(ep, "poses", None), dtype=np.float64)
        if a.ndim == 2 and p.ndim == 2 and len(p) > 3:
            rows.append((a, p))
    if not rows:
        raise ValueError("no episodes carried usable poses/actions")
    A = np.concatenate([r[0] for r in rows], 0)
    P = np.concatenate([r[1] for r in rows], 0)
    steer, accel, v = A[:, 0], A[:, 1], P[:, 3]
    curv = np.tan(steer) / wheelbase
    yr = v * curv
    out = {
        "n_episodes": len(rows), "n_frames": int(len(A)),
        "wheelbase_used": wheelbase,
        "channels": {
            "a_long": {"nonzero_frac": float((accel != 0).mean()),
                       "mean": float(accel.mean()), "std": float(accel.std()),
                       "min": float(accel.min()), "max": float(accel.max())},
            "curvature": {"nonzero_frac": float((steer != 0).mean()),
                          "std": float(curv.std())},
            "yaw_rate": {"std": float(yr.std()),
                         "absmax": float(np.abs(yr).max())},
            "v0": {"mean": float(v.mean()), "std": float(v.std()),
                   "frac_below_0p5": float((v < 0.5).mean())},
        },
    }
    ca, cy = [], []
    for a_, p_ in rows:
        vv = p_[:, 3]
        fd_a = np.diff(vv) / DT
        fd_y = np.diff(np.unwrap(p_[:, 2])) / DT
        m_a = a_[:-1, 1]
        m_y = (vv * np.tan(a_[:, 0]) / wheelbase)[:-1]
        if m_a.std() > 0 and fd_a.std() > 0:
            ca.append(float(np.corrcoef(m_a, fd_a)[0, 1]))
        if m_y.std() > 0 and fd_y.std() > 0:
            cy.append(float(np.corrcoef(m_y, fd_y)[0, 1]))
    out["finite_diff_crosscheck"] = {
        "corr_a_long": float(np.mean(ca)) if ca else None,
        "corr_yaw_rate": float(np.mean(cy)) if cy else None,
        "n_episodes": len(cy),
        "_reads": ("corr_yaw_rate near +1 confirms the SIGN and WHEELBASE "
                   "inversion against an independent field (quaternion yaw). "
                   "corr_a_long well below 1 is EXPECTED and is why `ax` is "
                   "used and d/dt(v) refused — physicalai.py says so at :13."),
    }
    out["echo"] = {}
    for hs in horizons_s:
        H = int(round(hs / DT))
        e0, e1 = [], []
        for a_, p_ in rows:
            for t in range(0, len(p_) - H - 1, stride):
                x0, y0, th0, v0 = p_[t]
                base = _np_extrapolate(v0, 0.0, 0.0, H)
                ext = _np_extrapolate(v0, a_[t, 1],
                                      float(np.tan(a_[t, 0]) / wheelbase), H)
                gt = p_[t + 1:t + 1 + H, :2] - p_[t, :2]
                c, s = np.cos(-th0), np.sin(-th0)
                gt = np.stack([gt[:, 0] * c - gt[:, 1] * s,
                               gt[:, 0] * s + gt[:, 1] * c], 1)
                e0.append(float(np.linalg.norm(base - gt, axis=1).mean()))
                e1.append(float(np.linalg.norm(ext - gt, axis=1).mean()))
        out["echo"][f"{hs:.1f}s"] = {
            "n_windows": len(e0), "ha0_ade_m": float(np.mean(e0)),
            "ha0_ext_ade_m": float(np.mean(e1)),
            "ext_beats_ha0_pct": float(100 * (np.mean(e0) - np.mean(e1))
                                       / np.mean(e0)),
            "tier": "corpus-kinematic (NOT a driving number, NOT T1)",
        }
    return out


def _np_extrapolate(v0, a0, k0, n_steps):
    """numpy twin of ``kinematic_goal_extrapolation`` xy, for the corpus report.

    ⚠️ It exists only so the corpus report has no torch dependency in its inner
    loop; ``test_numpy_twin_matches_torch`` pins the two AGREE, because a twin
    that silently drifts would make the pre-registration's cited numbers
    describe a function nobody runs.
    """
    import numpy as np
    ts = np.arange(1, n_steps + 1) * DT
    if a0 < 0:
        ts = np.minimum(ts, -v0 / a0)
    ts = np.maximum(ts, 0.0)
    s = np.maximum(v0 * ts + 0.5 * a0 * ts * ts, 0.0)
    th = k0 * s
    if abs(k0) < 1e-6:
        return np.stack([s, 0.5 * k0 * s * s], 1)
    return np.stack([np.sin(th) / k0, (1.0 - np.cos(th)) / k0], 1)


# --------------------------------------------------------------------------- #
# the four metric families over a goal/trajectory prediction                  #
# --------------------------------------------------------------------------- #
def trajectory_families(pred, target, *, taus_s: Sequence[float]) -> dict:
    """Per-window components for the BINDING four families (PI 2026-08-02).

    ``pred``/``target`` are ``[B, K, 4]`` = (x, y, heading, speed) in the ego
    frame of t0 — the ``refb_labels.goal_tac_targets`` layout verbatim.

    ⛔ Returns PER-WINDOW components, never a pooled scalar: a single composite
    hides exactly the trade-off the four families exist to expose, and the CI
    machinery needs per-window values to cluster by episode anyway.

    Families that cannot be computed from a goal row alone — headway / TTC
    (needs a lead agent) and the tactical/strategic decision families (need the
    heads' logits) — are NOT silently dropped: they are returned as ``None``
    with the reason, per the PI's rule 5.
    """
    import torch
    p = torch.as_tensor(pred).float()
    t = torch.as_tensor(target).float()
    if p.shape != t.shape or p.shape[-1] != 4:
        raise ValueError(f"pred/target must match and be [B,K,4]; got "
                         f"{tuple(p.shape)} vs {tuple(t.shape)}")
    disp = torch.linalg.vector_norm(p[..., :2] - t[..., :2], dim=-1)  # [B,K]
    dhead = torch.atan2(torch.sin(p[..., 2] - t[..., 2]),
                        torch.cos(p[..., 2] - t[..., 2])).abs()
    dspeed = (p[..., 3] - t[..., 3]).abs()
    tau = torch.as_tensor(list(taus_s), dtype=p.dtype).reshape(1, -1)
    # curvature implied by each goal row: kappa ~ heading / arclength, with the
    # arclength taken from the row's own (x, y) so the two arms are compared on
    # the same construction.
    arc = torch.linalg.vector_norm(p[..., :2], dim=-1).clamp_min(1e-3)
    arc_t = torch.linalg.vector_norm(t[..., :2], dim=-1).clamp_min(1e-3)
    dcurv = (p[..., 2] / arc - t[..., 2] / arc_t).abs()
    dyaw = (p[..., 2] / tau - t[..., 2] / tau).abs()
    # cross-track = lateral component in the TARGET's heading frame
    ct = (p[..., 1] - t[..., 1]).abs()
    return {
        "ade_m": disp,
        "longitudinal": {"speed_err_mps": dspeed,
                         "headway_ttc": None,
                         "_headway_reason": ("needs a lead agent in frame; "
                                             "supply obstacle.offline joins "
                                             "to compute it — n reported "
                                             "per-window when present")},
        "lateral": {"heading_err_rad": dhead, "curvature_err_invm": dcurv,
                    "yaw_rate_err_radps": dyaw, "cross_track_m": ct},
        "tactical": None,
        "_tactical_reason": ("decision quality needs the lat/lon head logits, "
                             "not a goal row — computed by the caller from "
                             "`lat_logits_tac`/`lon_logits_tac`"),
        "strategic": None,
        "_strategic_reason": ("g_str bearing error needs the strategic head "
                             "output and the LAN label; computed by the "
                             "caller"),
        "taus_s": list(taus_s),
    }


def _ci():
    """The canonical paired estimator, imported DEFENSIVELY.

    ⛔ ``overlapping_holdout_se`` is NOT admissible and is not reachable from
    here: it biases the POINT ESTIMATE (mean-of-split-means, not full_set) by
    -6.67 % to +11.69 % bidirectionally over 27 dumps, and has flipped a paired
    delta's SIGN. Only ``paired_episode_cluster_bootstrap`` decides anything.

    ⚠️ ``taniteval`` lives OUTSIDE ``stack/`` and the outer directory has no
    ``__init__.py``, so a bare import can resolve to a NAMESPACE PACKAGE with
    no ``ci`` submodule — which raises ``No module named 'taniteval.ci'`` and
    looks like a missing dependency rather than a shadow. The error below says
    which it is, because that distinction cost a debugging round before.
    """
    try:
        from taniteval.ci import paired_episode_cluster_bootstrap
        return paired_episode_cluster_bootstrap
    except ImportError as e:                                  # pragma: no cover
        raise ImportError(
            f"echo_gate needs taniteval.ci.paired_episode_cluster_bootstrap "
            f"(the ONLY decision-grade estimator; see CLAUDE.md). Got: {e}. "
            f"If this is `No module named 'taniteval.ci'` it is a NAMESPACE "
            f"SHADOW, not a missing package: put the INNER `<repo>/taniteval` "
            f"directory on sys.path, not the outer one.") from e


# --------------------------------------------------------------------------- #
# GATE 1 — beat the controls                                                  #
# --------------------------------------------------------------------------- #
def echo_gate(*, arm_ade, references: Mapping[str, "object"], eid,
              margins: Mapping[str, float] | None = None,
              n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired comparison of one arm against every reference, per horizon slot.

    ``arm_ade`` ``[N, K]`` per-window ADE of the arm; ``references`` maps a
    control name to its own ``[N, K]``; ``eid`` ``[N]`` episode ids. Every
    array is on the SAME windows — that is what makes the pairing valid and
    strictly more powerful than combining two intervals in quadrature.

    ``margins`` maps a reference name to the minimum RELATIVE point margin the
    prereg committed in advance (e.g. ``{"ha0_ext": 0.10}``). Both the CI
    separation AND the margin must hold; either alone is not the criterion the
    pre-registration committed.
    """
    import numpy as np
    paired = _ci()
    missing = [r for r in REQUIRED_REFERENCES if r not in references]
    if missing:
        raise ValueError(
            f"echo_gate refuses a panel missing {missing}. MEASURED on refcv3 "
            f"@30k: `ha` 0.2996 BEATS the 107 M model's 0.4799 (+0.1803 "
            f"[+0.1563, +0.2047], separated the WRONG way) while `ha0` 0.6723 "
            f"loses to it. A panel scored only against `ha0` certifies an arm "
            f"the trivial control already beats twice over — which is the "
            f"false positive this module exists to prevent. Supply "
            f"{REQUIRED_REFERENCES}.")
    a = np.asarray(arm_ade, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    eid = np.asarray(eid)
    margins = dict(margins or {})
    slots: list[dict] = []
    for k in range(a.shape[1]):
        row: dict = {"slot": k, "arm_mean": float(a[:, k].mean()), "vs": {}}
        for name, ref in references.items():
            r = np.asarray(ref, dtype=np.float64)
            if r.ndim == 1:
                r = r[:, None]
            res = paired(r[:, k], a[:, k], eid, n_boot=n_boot, seed=seed)
            rm = float(r[:, k].mean())
            point = rm - float(a[:, k].mean())      # >0 = the arm is better
            rel = point / rm if rm > 0 else float("nan")
            need = margins.get(name)
            row["vs"][name] = {
                "ref_mean": rm, "delta_ref_minus_arm": point,
                "relative_margin": rel,
                "separated": bool(res.get("separated")),
                "ci": [res.get("lo"), res.get("hi")],
                "estimator": "paired_episode_cluster_bootstrap",
                "required_relative_margin": need,
                "passes": bool(res.get("separated")) and point > 0 and (
                    need is None or rel >= need),
            }
        slots.append(row)
    return {"slots": slots, "n_windows": int(a.shape[0]),
            "n_episodes": int(len(set(eid.tolist()))),
            "references": sorted(references),
            "_reads": ("`passes` requires BOTH the CI to exclude zero AND the "
                       "point margin committed in advance. A separated CI on a "
                       "0.3 % margin is a real but useless difference, and the "
                       "prereg committed to margins for that reason.")}


# --------------------------------------------------------------------------- #
# GATE 2 — the ego-intervention test, BOTH directions                         #
# --------------------------------------------------------------------------- #
def ego_intervention_test(model, batch: Mapping, *,
                          goal_nodes: Sequence[str] = ("g_str", "g_tac",
                                                       "goal_point_tac"),
                          ego_key: str = "ego_state",
                          scene_key: str = "frames",
                          future_keys: Sequence[str] = ()) -> dict:
    """⭐⭐ BOTH DIRECTIONS. This is the heart of the anti-echo deliverable.

    ============ ================================================ ============
    direction    requirement                                       failing means
    ============ ================================================ ============
    EGO->OUT     perturbing ``ego_state`` MOVES every goal node    channels dead
    SCENE->OUT   holding ego, replacing frames MOVES every node    **ECHOING**
    FUTURE->OUT  perturbing future poses leaves nodes IDENTICAL    future leak
    ============ ================================================ ============

    ⛔ An arm that fires on EGO but not on SCENE is *exactly* the failure this
    module is named for: it responds to its own dynamics and not to the world.
    An arm that fires on neither is UNPOWERED, not clean (C109) — a dead node
    reads as independent of everything, including of a signal it is genuinely
    wired to.

    Built on :mod:`tanitad.eval.goal_provenance`, whose intervention primitive
    is **detach-transparent** — it measures INFORMATION, not gradient. That
    matters here because every downward goal edge in this codebase runs through
    ``.detach()``, so a gradient probe would certify a fully-leaking wire clean.
    """
    from tanitad.eval import goal_provenance as gp
    inputs = {ego_key: ego_key, scene_key: scene_key}
    for fk in future_keys:
        inputs[fk] = fk
    run = gp.module_runner(
        model, batch, nodes={},
        output_nodes={n: (lambda o, _n=n: o.get(_n)) for n in goal_nodes},
        input_nodes=inputs)
    det = gp.determinism_check(run)
    if not det["deterministic"]:
        return {"status": "UNPOWERED", "verdict": None, "determinism": det,
                "reason": ("forward is non-deterministic (max drift "
                           f"{det['max_drift']} at {det['worst_node']!r}); "
                           "every node would read as dependent — put the "
                           "model in eval() and disable dropout first")}
    ego = gp.probe_dependency(run, ego_key, list(goal_nodes))
    scene = gp.probe_dependency(run, scene_key, list(goal_nodes))
    fut = {fk: gp.probe_dependency(run, fk, list(goal_nodes))
           for fk in future_keys}

    ego_ok = bool(ego["any_dependence"])
    scene_ok = bool(scene["any_dependence"])
    fut_leak = sorted(k for k, v in fut.items() if v["any_dependence"])
    perturbable = ego["source_was_perturbable"] and scene["source_was_perturbable"]

    if not perturbable:
        status, verdict = "UNPOWERED", None
        reason = ("a source could not be perturbed on this batch, so nothing "
                  "it fails to move is evidence of anything")
    elif fut_leak:
        status, verdict = "MEASURED", "FUTURE_LEAK"
        reason = (f"⛔ perturbing {fut_leak} moved a goal node — the goal path "
                  f"reads FUTURE ego, which the PI's 'not the future one' "
                  f"forbids. This is the edge E11' pins in place of the old "
                  f"v0 refusal.")
    elif ego_ok and scene_ok:
        status, verdict = "MEASURED", "READS_BOTH"
        reason = ("the goal path reads the measured ego state AND the scene — "
                  "the admissible v4 state")
    elif ego_ok and not scene_ok:
        status, verdict = "MEASURED", "ECHOING"
        reason = ("⛔⛔ the goal path moves with ego and NOT with the scene: "
                  "the arm is reproducing its own dynamics. This is the "
                  "failure the anti-echo gate exists to catch.")
    elif scene_ok and not ego_ok:
        status, verdict = "MEASURED", "EGO_DEAD"
        reason = ("the ego channels do not reach any goal node — this is v3 "
                  "wearing a v4 config, and a 'no improvement' result from it "
                  "would be about the wiring, not the hypothesis")
    else:
        status, verdict = "UNPOWERED", None
        reason = ("neither ego nor scene moved any goal node; every node is "
                  "constant on this batch (typically a zero-init head on an "
                  "untrained model) — an artefact, not a finding")
    return {"status": status, "verdict": verdict, "reason": reason,
            "ego_moves_goal": ego_ok, "scene_moves_goal": scene_ok,
            "future_leak": fut_leak, "determinism": det,
            "detail": {"ego": ego, "scene": scene, "future": fut},
            "_note": ("REQUIRED: both EGO and SCENE. An arm passing only EGO "
                      "is the echo; an arm passing only SCENE never got the "
                      "channels.")}


# --------------------------------------------------------------------------- #
# GATE 2b — ⭐⭐ THE FUNCTIONAL PROBE (this one actually catches the echo)      #
# --------------------------------------------------------------------------- #
def source_ablation_test(predict: Callable, *, frames, ego_state, target,
                         eid=None, n_boot: int = 500, seed: int = 0,
                         min_degradation: float = 0.05) -> dict:
    """⭐⭐ DOES THE ARM *USE* THE SCENE, OR ONLY CARRY A WIRE TO IT?

    ⛔⛔ WHY THIS EXISTS — MEASURED 2026-09-03, AND IT INVALIDATED THE FIRST
    VERSION OF THIS MODULE'S GATE 2. The deliberate-regression arm (v4 trained
    with the image ablated, i.e. a pure echo BY CONSTRUCTION) **PASSED** the
    interventional ``scene -> goal`` probe: perturbing ``frames`` moved every
    goal node. Of course it did. The encoder is a live function of its input,
    so ``frames -> goal`` is a live forward path in ANY model with a
    non-degenerate trunk — **including one that has learned to ignore the scene
    completely.** The intervention measures WIRING; echoing is about USE, and
    the two come apart exactly where this module lives.

    That is the same family as the finding ``goal_provenance`` was built on —
    *"a detached wire carries the full signal and zero gradient"* — read in the
    converse direction: **a live wire can carry zero useful signal.** An
    interventional probe is necessary (it catches a severed wire) and it is NOT
    sufficient (it cannot catch an ignored one).

    THE FIX, and it is functional rather than structural: hand the model the
    **WRONG** scene while holding ego correct, and ask whether its ERROR gets
    worse. A model that reads the scene is hurt by the wrong scene. A model
    that echoes is not. The mirror asks the same of ego. Both are paired on the
    same windows, with the episode-cluster bootstrap.

    ``predict(frames, ego_state) -> [B, K, 4]``. Returns per-source
    ``degradation`` = relative rise in ADE under a deranged input.

    ⭐ THE CONTROL THAT MUST READ A KNOWN VALUE: a constant predictor is
    unaffected by either derangement and MUST read exactly 0.0 on both — that
    is what says the probe is measuring use rather than noise.
    """
    import numpy as np
    import torch
    frames = torch.as_tensor(frames)
    ego_state = torch.as_tensor(ego_state)
    target = torch.as_tensor(target)
    b = frames.shape[0]
    if b < 2:
        raise ValueError("source_ablation_test needs batch >= 2 to derange")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(b, generator=g)
    if bool((perm == torch.arange(b)).any()):        # a DERANGEMENT, not a
        perm = torch.roll(torch.arange(b), 1)        # permutation: a fixed
                                                     # point leaves that row's
                                                     # input correct and dilutes
                                                     # the measured degradation.

    def _ade(p):
        return torch.linalg.vector_norm(
            p[..., :2] - target[..., :2], dim=-1).mean(dim=-1)   # [B]

    with torch.no_grad():
        base = _ade(predict(frames, ego_state))
        mis_scene = _ade(predict(frames[perm], ego_state))
        mis_ego = _ade(predict(frames, ego_state[perm]))
    out: dict = {"n": int(b), "base_ade": float(base.mean()),
                 "sources": {}}
    eid = np.arange(b) if eid is None else np.asarray(eid)
    paired = _ci()
    for name, worse in (("scene", mis_scene), ("ego", mis_ego)):
        d = float(worse.mean() - base.mean())
        rel = d / float(base.mean()) if float(base.mean()) > 0 else float("nan")
        res = paired(worse.numpy().astype("float64"),
                     base.numpy().astype("float64"), eid,
                     n_boot=n_boot, seed=seed)
        out["sources"][name] = {
            "ade_deranged": float(worse.mean()), "degradation_abs": d,
            "degradation_rel": rel,
            "separated": bool(res.get("separated")), "ci": [res.get("lo"),
                                                            res.get("hi")],
            "estimator": "paired_episode_cluster_bootstrap",
            "required_rel": min_degradation,
            "is_used": bool(res.get("separated")) and rel >= min_degradation,
        }
    sc, ego_u = out["sources"]["scene"]["is_used"], out["sources"]["ego"]["is_used"]
    out["verdict"] = ("READS_BOTH" if sc and ego_u else
                      "ECHOING" if ego_u and not sc else
                      "IGNORES_EGO" if sc and not ego_u else
                      "READS_NEITHER")
    out["_reads"] = {
        "READS_BOTH": "the admissible v4 state: wrong scene hurts, wrong ego hurts",
        "ECHOING": ("⛔⛔ the wrong SCENE does not hurt — the arm is "
                    "reproducing its own dynamics and the image is decoration"),
        "IGNORES_EGO": ("the ego channels are present and unused; a 'no "
                        "improvement' result from this arm is about the "
                        "wiring, not the hypothesis"),
        "READS_NEITHER": ("neither input matters — typically an untrained or "
                          "collapsed arm; UNPOWERED, not clean"),
    }[out["verdict"]]
    return out


# --------------------------------------------------------------------------- #
# THE GATE                                                                    #
# --------------------------------------------------------------------------- #
def assert_not_echoing(gate1: Mapping | None, gate2: Mapping, *,
                       gate2b: Mapping | None = None,
                       primary_slot: int | None = None,
                       primary_reference: str = "ha0_ext",
                       allow_unpowered: bool = False) -> dict:
    """⛔ THE GATE. Raise unless the arm clears the echo in EVERY gate.

    ``gate1``  — :func:`echo_gate` (``None`` at tiny-rig scale, where no
                 trained arm exists yet).
    ``gate2``  — :func:`ego_intervention_test`: **structural**. Catches a
                 SEVERED wire and a FUTURE leak.
    ``gate2b`` — :func:`source_ablation_test`: **functional**. Catches an
                 IGNORED wire, i.e. the echo itself.

    ⛔⛔ ``gate2b`` IS NOT OPTIONAL FOR AN ECHO VERDICT, and omitting it is the
    mistake this module made first. MEASURED 2026-09-03: the deliberate-
    regression arm PASSED ``gate2``'s scene direction, because a live encoder
    moves the output for any model — including one that ignores the scene
    entirely. ⇒ passing ``gate2b=None`` yields a verdict that is explicitly
    marked ``STRUCTURAL_ONLY``; it is **not** an anti-echo pass and must not be
    reported as one.

    ``allow_unpowered=False`` by default and that is the load-bearing choice:
    an UNPOWERED reading is *"we did not establish it"*, and letting that pass
    is how a guard becomes decoration.
    """
    bad: list[str] = []
    if gate2b is not None:
        v = gate2b.get("verdict")
        if v in ("ECHOING", "IGNORES_EGO", "READS_NEITHER"):
            bad.append(f"GATE 2b ({v}): {gate2b.get('_reads')} — scene "
                       f"degradation {gate2b['sources']['scene']['degradation_rel']:+.4f} "
                       f"(required {gate2b['sources']['scene']['required_rel']}), "
                       f"ego degradation "
                       f"{gate2b['sources']['ego']['degradation_rel']:+.4f}")
    if gate2.get("verdict") == "ECHOING":
        bad.append("GATE 2: " + str(gate2.get("reason")))
    if gate2.get("verdict") == "FUTURE_LEAK":
        bad.append("GATE 2: " + str(gate2.get("reason")))
    if gate2.get("verdict") == "EGO_DEAD":
        bad.append("GATE 2: " + str(gate2.get("reason")))
    if gate2.get("status") == "UNPOWERED" and not allow_unpowered:
        bad.append("GATE 2 UNPOWERED: " + str(gate2.get("reason")))
    if gate1 is not None:
        slots = gate1.get("slots", [])
        if primary_slot is None:
            primary_slot = len(slots) - 1        # the LONGEST horizon
        refs = (primary_reference,) if primary_reference not in (
            None, "ha0_ext") else REQUIRED_REFERENCES
        for row in slots:
            if row["slot"] != primary_slot:
                continue
            for ref_name in refs:
                cell = row["vs"].get(ref_name)
                if cell is None:
                    bad.append(f"GATE 1: reference {ref_name!r} absent — an "
                               f"arm scored without it is not scored")
                elif not cell["passes"]:
                    bad.append(
                        f"GATE 1 slot {primary_slot}: did NOT clear "
                        f"{ref_name} — ref {cell['ref_mean']:.4f} vs arm "
                        f"{row['arm_mean']:.4f} (relative margin "
                        f"{cell['relative_margin']:+.4f}, required "
                        f"{cell['required_relative_margin']}, CI "
                        f"{cell['ci']}, separated={cell['separated']})")
        # ⚠️ THE ECHO-HARDER TRAP, made mechanical. Clearing `ha` while TYING
        # `ha0_ext` means the margin came from a BETTER extrapolation of the
        # ego state, not from reading the scene — both controls are ego
        # extrapolations and `ha0_ext` is the sharper one.
        for row in slots:
            if row["slot"] != primary_slot:
                continue
            ha_c, ext_c = row["vs"].get("ha"), row["vs"].get("ha0_ext")
            if ha_c and ext_c and ha_c["passes"] and not ext_c["separated"]:
                bad.append(
                    "GATE 1: cleared `ha` but TIED `ha0_ext` — the margin is a "
                    "sharper ego extrapolation, not scene reading. `ha` uses a "
                    "finite-difference accel (corr +0.5624 with the corpus's "
                    "own ax); beating it that way is echoing harder.")
    if bad:
        raise EchoViolation(
            "ANTI-ECHO GATE FAILED — this arm has not been shown to beat its "
            "own measured dynamics:\n  " + "\n  ".join(bad)
            + "\n(MEASURED 2026-09-03: a constant-a, constant-kappa "
              "extrapolation of the measured t0 ego state reaches ADE 0.4449 m "
              "at 2 s from ZERO PIXELS — the same league as flagship v1's "
              "deployed 0.452 m. Clearing `ha0` is not evidence of anything.)")
    return {"ok": True, "gate2_verdict": gate2.get("verdict"),
            "gate2b_verdict": (None if gate2b is None
                               else gate2b.get("verdict")),
            "class": ("ANTI_ECHO" if gate2b is not None else "STRUCTURAL_ONLY"),
            "_reads": ("STRUCTURAL_ONLY means gate2b was not supplied: the "
                       "wire was shown to exist, NOT to be used. That is not "
                       "an anti-echo pass and must not be reported as one."),
            "primary_reference": primary_reference,
            "primary_slot": primary_slot}
