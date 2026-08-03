"""SITUATION DETECTORS — lane change / intersection (roundabout kept, but UNPOWERED).

⭐ PROMOTED INTO stack/ 2026-07-29 on PI direction ("implement the situation classification
including the lane change and intersection labels as I specified, skip roundabouts for later").
It previously lived only under
…/incoming/2026-07-26-situation-classifier/scripts/sc_situations.py — i.e. it was real,
measured code that no training or eval path could import. This file is that code, unchanged in
behaviour, moved to where the rest of the stack can use it.

⚠️ THRESHOLDS ARE FROZEN by …/2026-07-26-situation-classifier/PRE_REGISTRATION.md §2 and MUST
NOT be swept. They were chosen on the TRAIN chunks before any held-out number existed; sweeping
them here would retro-fit a pre-registered study.

WHAT THE MEASURED STUDY FOUND (so nobody re-derives it):
  lane change  153 held-out clusters · image arm above chance ΔAP +0.01987 [+0.01141, +0.02901],
               AUROC 0.703 · median anticipation lead 1.4 s · verdict A−
  intersection 264 clusters · ΔAP +0.04894 [+0.03735, +0.06277], AUROC 0.769 · lead 2.0 s · A−
  roundabout    26 clusters · UNPOWERED — which is why the PI deferred it
⛔ **RETRACTED 2026-08-03 — "VISION ADDS NOTHING OVER EGO STATE" was the wrong test.**
The retired claim read the ego arm against the image arms (head_ego CV-AP 0.0697 vs img+ego 0.0525,
img-only 0.0376) and concluded the front camera "has no measured signal to stand on". Two defects:
  (a) the img+ego arm was produced by the SCALE-MISMATCH early concat at `sc_train.py:143` — a
      16-dim PCA image block normalised by its own mean-abs against a 3-dim ego block on a hand-set
      `EGO_SCALE`, one shared Linear. A broken fusion is not evidence about vision.
  (b) **ego is not a legal inference input.** PI ruling 2026-08-03: *"for ground truth data of
      scenario classification you can use both ego and other label, for inference only vision."*
      So "is vision better than ego" cannot decide anything deployable — the deployable question is
      whether vision beats CHANCE, and that comparison was never run.
✅ **The right test — vision against its OWN null — SEPARATES.** MEASURED on the banked held-out
bundle, paired episode-cluster bootstrap B=2000 over 1,610 clip clusters
(`…/incoming/2026-08-03-sitclf-fusion-wired/results_sitclf_vision_only.json`): on `lane_change`,
`head_img` AP 0.03741 vs its permuted-feature null `head_img_shuf` 0.01715 —
ΔAP-lift **+1.1749 [+0.7930, +1.6890]**, i.e. the front camera carries **2.18x its own null**.
⚠️ **LABEL PROVENANCE — why the old comparison was structurally unfair (2 probes).** Every situation
label is a pure deterministic function of the ego pose track [x, y, yaw, v]:
`scripts/emit_situation_labels.py:54-62` reads only `d["poses"]`, and every detector below
(`:161`, `:210`, `:244`, `:284`) takes only `K = kinematics(P)` — the emitter passes `cross=None`, so
even `intersection` is the turn half alone. An ego-input head therefore observes the label's
GENERATING PROCESS directly, while the camera must infer it from pixels.
   It is **NOT** a future-information leak: the head's window is [t-0.7 s, t] (`sc_train.py:37`,
   offsets -7..0) and the label's evidence window is [onset, onset+4 s] with onset > t — disjoint.
   But `omega_pre`/`alon_pre` are built on `np.gradient`, a CENTRED difference, so they do read
   **one frame (0.1 s) past t**; a small boundary leak that bites only for onsets at exactly t+1.

The THREE SITUATION DETECTORS — lane change / roundabout / intersection.

This module is the single definition of the PI's three situations. It is imported by the label
builder, by the validation scripts and by the evaluator; it is never re-implemented anywhere.

INPUT is the ego trajectory ONLY: ``P = [x, y, yaw, v]`` at 10 Hz, which is exactly what
`physicalai.build_episode` stores in every cached episode (`poses`), index-for-index with the
frames the encoder sees.  The cross-traffic half of the intersection detector takes a separately
computed per-frame boolean (`sc_cross.py`) because it needs `obstacle.offline` + calibration.

⚠ Nothing in here is a model input. The model receives the 256 px / 51.4 deg front crop and its own
speed. `x`, `y`, `yaw` and every derived curvature are PRIVILEGED GEOMETRY — that is what makes the
supervision non-circular.

Thresholds are frozen in `PRE_REGISTRATION.md` Sec 2 and must not be swept.
"""
from __future__ import annotations

import numpy as np

HZ = 10.0
DT = 1.0 / HZ

# ---------------------------------------------------------------- frozen constants (PRE-REG Sec 2)
V_FLOOR_KAPPA = 1.5          # m/s   speed floor in kappa = omega / max(v, .)  (standstill guard)
SMOOTH_S = 0.5               # s     centred moving average for omega and a_lon
SMOOTH_KAPPA_S = 1.0         # s     centred moving average for kappa

# --- lane change
LC_W_S = 4.0                 # s     manoeuvre window
LC_V_MIN = 8.0               # m/s   speed at window start
LC_V_MIN_ANY = 6.0           # m/s   speed floor throughout the window
LC_DPSI_MAX = 8.0            # deg   NET heading change (this is what separates it from a turn)
LC_LAT_MIN = 2.4             # m     one lane width, low tolerance
LC_LAT_MAX = 5.5             # m     one lane width, high tolerance
LC_MONO = 0.85               # -     |lat(end)| >= LC_MONO * max|lat|   (net offset, not a wobble)
LC_LOBE_DEG = 1.5            # deg   both yaw-rate lobes must reach this (the S-shape)

# --- roundabout
RB_KAPPA_MIN = 0.020         # 1/m   R <= 50 m
RB_GAP_S = 0.3               # s     tolerated dropout below the deadband inside a run
RB_DUR_MIN_S = 3.0           # s     SUSTAINED  (the duration half of the discriminator)
RB_DPSI_MIN = 90.0           # deg   substantial arc
RB_CV_MAX = 0.5              # -     std(kappa)/mean|kappa|  (the CONSTANCY half)
RB_V_LO, RB_V_HI = 2.0, 14.0  # m/s  lower speed
RB_BRACKET_S = 3.0           # s     window before/after in which the entry/exit deflection is sought
RB_BRACKET_DEG = 3.0         # deg   opposite-sign deflection on AT LEAST ONE side (entry OR exit)
# ^ RB_* selected on TRAIN chunks only by the rule "maximise events subject to DEV
#   counter-clockwise purity >= 0.90" (PRE_REGISTRATION Sec 2.2). The corpus contains ZERO
#   left-hand-traffic clips (MEASURED), so a true roundabout label must be ~100 % ccw.
#   Selected point: 22 DEV events at 0.909 ccw. Sweep: artifacts/round_sweep.json.

# --- intersection (turn half)
IX_DPSI_LO, IX_DPSI_HI = 45.0, 135.0   # deg  a quantised quarter-turn
IX_DUR_MAX_S = 6.0           # s     a junction turn is short; a long arc is a curve or a roundabout
IX_R_MAX = 25.0              # m     tight radius -> a junction, not a road curve
IX_V_MIN = 1.0               # m/s

# --- the ANTICIPATION target
LEAD_S = 3.0                 # s     Y(t) = 1 iff an onset falls in (t, t + LEAD_S]
MIN_USEFUL_LEAD_S = 1.0      # s     registered BEFORE measurement (PRE-REG Sec 4)


# --------------------------------------------------------------------------------- small utilities
def movavg(x: np.ndarray, n: int) -> np.ndarray:
    """centred moving average, edge-padded (identical convention to l2_build._movavg)."""
    n = int(n)
    if n < 2:
        return x
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def kinematics(P: np.ndarray) -> dict:
    """P [T,4] = (x, y, yaw, v) at 10 Hz -> the derived signals every detector reads."""
    x, y = P[:, 0].astype(np.float64), P[:, 1].astype(np.float64)
    psi = np.unwrap(P[:, 2].astype(np.float64))
    v = P[:, 3].astype(np.float64)
    T = len(v)
    ns = max(2, int(round(SMOOTH_S * HZ)))
    nk = max(2, int(round(SMOOTH_KAPPA_S * HZ)))
    omega = np.gradient(psi, DT)
    omega_s = movavg(omega, ns)
    kappa = movavg(omega_s / np.maximum(v, V_FLOOR_KAPPA), nk)
    alon = movavg(np.gradient(v, DT), ns)
    # trailing 0.5 s means (STRICTLY CAUSAL -- these are the only ego channels a head may receive)
    alon_pre = np.convolve(np.pad(np.gradient(v, DT), (ns - 1, 0), mode="edge"),
                           np.ones(ns) / ns, mode="valid")
    omega_pre = np.convolve(np.pad(omega, (ns - 1, 0), mode="edge"),
                            np.ones(ns) / ns, mode="valid")
    return dict(T=T, x=x, y=y, psi=psi, v=v, omega=omega_s, kappa=kappa,
                alon=alon, alon_pre=alon_pre, omega_pre=omega_pre)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """maximal True runs of `mask` as inclusive [a, b] index pairs."""
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.nonzero(d == 1)[0] + 1)
    ends = list(np.nonzero(d == -1)[0])
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask) - 1]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def curvature_runs(K: dict, kappa_min: float = RB_KAPPA_MIN) -> list[tuple[int, int, int]]:
    """Maximal same-sign, above-deadband curvature runs -> (a, b, sign).

    A dropout below the deadband shorter than RB_GAP_S does not break a run (a roundabout has a
    brief straightening between entry and circulation).

    ⚠️ `kappa_min` is a parameter because the DEFAULT deadband (R <= 50 m) makes a LARGE-RADIUS road
    curve invisible — with it, `detect_curves` returned **3 events on the whole corpus** and the
    Sec 6.2 control population was empty by construction. A control that cannot be populated is the
    C13 failure mode (a guard that cannot fire), so the curve population uses its own, lower
    deadband; the roundabout and turn detectors are untouched."""
    kap = K["kappa"]
    out = []
    gap = int(round(RB_GAP_S * HZ))
    for s in (+1, -1):
        m = (np.sign(kap) == s) & (np.abs(kap) >= kappa_min)
        # close short gaps
        segs = _runs(m)
        merged = []
        for a, b in segs:
            if merged and a - merged[-1][1] - 1 <= gap and not (np.sign(kap[merged[-1][1]+1:a]) == -s).any():
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        out += [(a, b, s) for a, b in merged]
    return sorted(out)


# ------------------------------------------------------------------------------------ LANE CHANGE
def detect_lane_change(K: dict) -> list[tuple[int, int]]:
    """-> list of (onset_index, end_index).  Overlapping windows merged; onset = earliest start."""
    T, W = K["T"], int(round(LC_W_S * HZ))
    if T < W + 2:
        return []
    x, y, psi, v = K["x"], K["y"], K["psi"], K["v"]
    hits = []
    for i in range(0, T - W):
        j = i + W
        if v[i] < LC_V_MIN or v[i:j + 1].min() < LC_V_MIN_ANY:
            continue
        dpsi = np.degrees(psi[j] - psi[i])
        if abs(dpsi) > LC_DPSI_MAX:
            continue
        c, s = np.cos(psi[i]), np.sin(psi[i])
        lat = -s * (x[i:j + 1] - x[i]) + c * (y[i:j + 1] - y[i])
        end = lat[-1]
        if not (LC_LAT_MIN <= abs(end) <= LC_LAT_MAX):
            continue
        if abs(end) < LC_MONO * np.abs(lat).max():
            continue
        d = np.degrees(np.diff(psi[i:j + 1]))
        pos, neg = d[d > 0].sum(), -d[d < 0].sum()
        if min(pos, neg) < LC_LOBE_DEG:
            continue
        # the first substantial lobe must point TOWARD the target lane
        cum = np.cumsum(d)
        k1 = int(np.argmax(np.abs(cum) >= LC_LOBE_DEG)) if (np.abs(cum) >= LC_LOBE_DEG).any() else 0
        if np.sign(cum[k1]) != np.sign(end):
            continue
        hits.append((i, j))
    return _merge(hits)


def _merge(hits: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """merge overlapping (a, b) into events; the event's onset is the earliest a."""
    if not hits:
        return []
    hits = sorted(hits)
    out = [list(hits[0])]
    for a, b in hits[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(int(a), int(b)) for a, b in out]


# ------------------------------------------------------------------------------------- ROUNDABOUT
def detect_roundabout(K: dict, bracket: bool = True) -> list[tuple[int, int]]:
    """Sustained, CONSTANT, one-sign curvature over a long arc at lower speed.

    `bracket=False` is the pre-registered `ROUND_core` sensitivity (R0-R4 only)."""
    psi, v, kap = K["psi"], K["v"], K["kappa"]
    nb = int(round(RB_BRACKET_S * HZ))
    dmin = int(round(RB_DUR_MIN_S * HZ))
    out = []
    for a, b, s in curvature_runs(K):
        if b - a + 1 < dmin:
            continue
        dpsi = abs(np.degrees(psi[b] - psi[a]))
        if dpsi < RB_DPSI_MIN:
            continue
        seg = kap[a:b + 1]
        m = np.abs(seg).mean()
        if m <= 0 or seg.std() / m > RB_CV_MAX:
            continue
        vm = v[a:b + 1].mean()
        if not (RB_V_LO <= vm <= RB_V_HI):
            continue
        if bracket:
            # entry/exit deflection: OPPOSITE-sign heading change on AT LEAST ONE side.
            # (-s * d) is the opposite-sign magnitude, so >= RB_BRACKET_DEG encodes both the
            # sign test and the magnitude test in one expression.
            pre = -s * np.degrees(psi[a] - psi[max(0, a - nb)])
            post = -s * np.degrees(psi[min(len(psi) - 1, b + nb)] - psi[b])
            if max(pre, post) < RB_BRACKET_DEG:
                continue
        out.append((a, b))
    return _merge(out)


# ----------------------------------------------------------------------------------- INTERSECTION
def detect_turns(K: dict) -> list[tuple[int, int]]:
    """The TURN half of the intersection detector: a short, tight, quantised quarter-turn."""
    psi, v, kap = K["psi"], K["v"], K["kappa"]
    dmax = int(round(IX_DUR_MAX_S * HZ))
    out = []
    for a, b, _s in curvature_runs(K):
        if b - a + 1 > dmax:
            continue
        dpsi = abs(np.degrees(psi[b] - psi[a]))
        if not (IX_DPSI_LO <= dpsi <= IX_DPSI_HI):
            continue
        seg = np.abs(kap[a:b + 1])
        seg = seg[seg > 0]
        if not len(seg) or (1.0 / np.median(seg)) > IX_R_MAX:
            continue
        if v[a:b + 1].mean() < IX_V_MIN:
            continue
        out.append((a, b))
    return _merge(out)


def detect_curves(K: dict, r_min: float = 40.0) -> list[tuple[int, int]]:
    """⭐ The CONTROL population for PRE_REGISTRATION Sec 6.2: road curves with the SAME heading
    change as a junction turn but a LARGE radius. If `detect_turns` is really a junction detector
    and not a curve detector, perpendicular cross traffic must be far more common on turns than
    here — and that is a measurement, not an argument."""
    psi, kap = K["psi"], K["kappa"]
    out = []
    for a, b, _s in curvature_runs(K, kappa_min=1.0 / 400.0):     # R <= 400 m, so curves exist
        dpsi = abs(np.degrees(psi[b] - psi[a]))
        if not (IX_DPSI_LO <= dpsi <= IX_DPSI_HI):
            continue
        seg = np.abs(kap[a:b + 1])
        seg = seg[seg > 0]
        if not len(seg) or (1.0 / np.median(seg)) <= r_min:
            continue
        out.append((a, b))
    return _merge(out)


def detect_intersection(K: dict, cross: np.ndarray | None = None,
                        min_cross_s: float = 1.0) -> tuple[list[tuple[int, int]], list, list]:
    """-> (events, turn_events, cross_events).

    events = turn events NOT inside a roundabout, UNION blocks of >= `min_cross_s` of cross traffic.
    `cross` is the per-frame boolean from `sc_cross.py` (None -> turn half only)."""
    rb = detect_roundabout(K, bracket=True)
    turns = [(a, b) for a, b in detect_turns(K)
             if not any(a <= rb_b and b >= rb_a for rb_a, rb_b in rb)]
    xe = []
    if cross is not None:
        n = int(round(min_cross_s * HZ))
        xe = [(a, b) for a, b in _runs(cross.astype(bool)) if b - a + 1 >= n]
    return _merge(turns + xe), turns, xe


# ------------------------------------------------------------- the ANTICIPATION target and masking
def anticipation_target(T: int, events: list[tuple[int, int]],
                        lead_s: float = LEAD_S) -> tuple[np.ndarray, np.ndarray]:
    """-> (y, valid).

    ``y[t] = 1`` iff an event ONSET falls in ``(t, t + lead_s]``.
    ``valid[t] = 0`` for every frame INSIDE an ongoing event, and for the last `lead` frames
    (no in-episode future).  Masking the ongoing frames is what makes this ANTICIPATION rather than
    recognition: the head is never scored on a frame in which the manoeuvre is already happening.
    """
    L = int(round(lead_s * HZ))
    y = np.zeros(T, bool)
    valid = np.ones(T, bool)
    for a, b in events:
        lo = max(0, a - L)
        y[lo:a] = True                 # (t, t+L] contains the onset  <=>  a-L <= t < a
        valid[a:b + 1] = False         # ongoing -> not scored
    valid[max(0, T - L):] = False
    return y, valid


def event_onsets(events: list[tuple[int, int]]) -> np.ndarray:
    return np.array([a for a, _ in events], dtype=np.int64)
