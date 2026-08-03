#!/usr/bin/env python3
"""Reference<->rig frame alignment: estimate it, and let the estimator REFUSE.

WHY THIS FILE EXISTS
--------------------
`R-2026-08-03-k`: every ABSOLUTE render-fidelity number on NuRec scene `00040136` was
scored against a reference video **6 frames too early**.  The negative control in
`render_quality.py` could not see it *by construction* (`MIN_WRONG_GAP = 40`), so a
silent index error degraded every absolute for as long as the work had been running.
Correcting it was worth **+0.1797 grad-NCC, free** -- 8.6x the entire rolling-shutter
effect and ~40x the pose sweep.

The programme then had TWO observations (+6 on `00040136`, +5 on `7c72937c`) and no rule.
Two observations are not a rule, and

    ⛔ A SEARCH THAT ALWAYS RETURNS AN ARGMAX WILL ALWAYS RETURN AN OFFSET.

That is not a hypothetical.  It has already happened twice in this exact work:

  * an earlier +-3 scan **stopped at its boundary still rising** and reported ">= +3";
  * the GPU-free cross-correlation on `7c72937c` rose **monotonically from -15 to +15**
    (the ego is stationary for the first 9 frames, so image motion carries no signal)
    and its argmax was +15 -- correctly reported by its author as an instrument failure,
    but only because a human looked at the curve.

So every estimator here returns an :class:`AlignEstimate` that can carry ``refused=True``,
and the refusal rules are the four failure modes above:

  ``boundary``       the argmax sits at an end of the scanned window -> the window is
                     the answer, not the data.  This is the ">= +3" failure.
  ``no_turnover``    the curve does not fall on BOTH sides of the peak -> monotone, so
                     the peak is a scan artefact.  This is the `7c72937c` failure.
  ``weak``           the peak score is below the floor a real match must clear.
  ``not_separated``  the peak does not beat the best competitor outside +-``exclusion``
                     by ``min_prominence`` -> the curve is flat and any tie-break is noise.

THREE INDEPENDENT ESTIMATORS, deliberately touching different things
--------------------------------------------------------------------
``count_delta``     mp4 decodable frames minus rig frames.  No pixels, no renderer.
``motion_lag``      cross-correlate the mp4's frame-to-frame image motion against the
                    rig's per-frame ego translation.  Pixels, but NO renderer and NO
                    grad-NCC -- so it cannot inherit a renderer bug.
``render_scan``     render rig frame f at PRODUCTION settings and score it against video
                    frames f-k .. f+k.  The renderer's own answer.  (Driver lives in
                    `rs_frame_offset.py`; the adjudication lives here.)

CONTROLS WITH A KNOWN ANSWER (`--self-test`, and `stack/tests/test_frame_align.py`)
----------------------------------------------------------------------------------
An estimator is only trustworthy if it is shown to return the RIGHT answer when the
right answer is known, including when the right answer is **zero**:

  * ``inject`` -- re-index a real series by a known d; the recovered offset must move by
    exactly d.  Includes d chosen so the truth leaves the window: the estimator must
    then REFUSE (`boundary`), not clamp.
  * ``zero``   -- strip the leader from a real reference series so the true offset IS 0;
    the estimator must return 0, not "6 again".
  * ``null``   -- two independent random series; the estimator must refuse.

Usage
-----
    python frame_align.py --scene-dir <scene> --out /tmp/align.json [--self-test]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

CAM = "camera_front_wide_120fov"

# Defaults are stated once, here, so a caller cannot quietly weaken a refusal rule by
# passing a different literal at each call site.
MIN_PROMINENCE = 0.02   # peak minus best competitor outside +-EXCLUSION
EXCLUSION = 1           # neighbours of the peak are not competitors; they are the peak
MIN_PEAK = 0.05         # a peak below this is not a match on any of our metrics


# --------------------------------------------------------------------------------- #
# the adjudicator                                                                    #
# --------------------------------------------------------------------------------- #
@dataclass
class AlignEstimate:
    """An offset, or an explicit refusal. Never a bare argmax."""
    method: str
    offset: int | None
    refused: bool
    reason: str
    score_by_offset: dict[int, float] = field(default_factory=dict)
    peak: float | None = None
    runner_up: float | None = None
    prominence: float | None = None
    subframe_offset: float | None = None
    n: int | None = None
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d["score_by_offset"] = {int(k): (round(float(v), 6) if v is not None else None)
                                for k, v in self.score_by_offset.items()}
        return d

    def __str__(self) -> str:
        if self.refused:
            return f"{self.method}: REFUSED ({self.reason})"
        bits = []
        if self.peak is not None:
            bits.append(f"peak {self.peak:.4f}")
        if self.prominence is not None:
            bits.append(f"prominence {self.prominence:+.4f}")
        if self.subframe_offset is not None:
            bits.append(f"sub-frame {self.subframe_offset:+.2f}")
        tail = f" ({', '.join(bits)})" if bits else ""
        return f"{self.method}: offset {self.offset:+d}{tail}"


def _parabolic_peak(x_lo: float, x_c: float, x_hi: float) -> float:
    """Sub-sample peak position of a 3-point parabola, in units of the sample spacing.

    Returns the shift of the vertex from the centre sample, clamped to (-1, 1).  A
    fractional answer is informative: a true integer index error lands near 0.0, while a
    sub-frame TIMING error (pose phase, shutter) does not.
    """
    denom = (x_lo - 2.0 * x_c + x_hi)
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (x_lo - x_hi) / denom, -0.999, 0.999))


def adjudicate(score_by_offset: dict[int, float], method: str,
               min_peak: float = MIN_PEAK,
               min_prominence: float = MIN_PROMINENCE,
               exclusion: int = EXCLUSION,
               require_turnover: bool = True,
               extra: dict | None = None) -> AlignEstimate:
    """Turn a scanned curve into an offset **or a refusal**.

    This is the whole point of the module: `max(d, key=score)` cannot fail, so it is
    never called directly anywhere in the render stack.
    """
    if not score_by_offset:
        return AlignEstimate(method, None, True, "empty", {}, extra=extra or {})
    offs = sorted(int(d) for d in score_by_offset)
    sc = {int(d): float(score_by_offset[d]) for d in offs}
    if len(offs) < 3:
        return AlignEstimate(method, None, True, "window_too_small", sc,
                             n=len(offs), extra=extra or {})
    best = max(offs, key=lambda d: sc[d])
    peak = sc[best]
    competitors = [sc[d] for d in offs if abs(d - best) > exclusion]
    runner = max(competitors) if competitors else None
    prom = (peak - runner) if runner is not None else None
    i = offs.index(best)
    sub = (_parabolic_peak(sc[offs[i - 1]], peak, sc[offs[i + 1]])
           if 0 < i < len(offs) - 1 else None)

    def _mk(refused, reason):
        return AlignEstimate(method, (None if refused else best), refused, reason, sc,
                             round(peak, 6),
                             (round(runner, 6) if runner is not None else None),
                             (round(prom, 6) if prom is not None else None),
                             (round(best + sub, 4) if sub is not None else None),
                             len(offs), extra or {})

    # ⚠️ ORDER MATTERS, and getting it wrong caused a MEASURED false alarm. A FLAT curve
    # whose argmax happens to land on the window edge is "no signal", not "the residual
    # is off-window"; the two demand opposite responses (ignore the frame vs. halt the
    # run). `7c72937c` frame 60 spans 0.4003-0.4023 over the whole scan and its max sits
    # at the edge — reported as `boundary`, it blocked a CORRECT offset. So the
    # signal-strength rules are adjudicated FIRST, and only a PROMINENT peak at the edge
    # is a boundary.
    # 1. nothing matches anything
    if peak < min_peak:
        return _mk(True, "weak")
    # 2. flat curve; the argmax is a coin flip
    if prom is not None and prom < min_prominence:
        return _mk(True, "not_separated")
    # 3. the window is the answer, not the data  (the ">= +3" failure)
    if best == offs[0] or best == offs[-1]:
        return _mk(True, "boundary")
    # 4. monotone curve  (the `7c72937c` cross-correlation failure)
    if require_turnover and not (sc[offs[i - 1]] < peak and sc[offs[i + 1]] < peak):
        return _mk(True, "no_turnover")
    return _mk(False, "ok")


def bootstrap_offset(per_unit_curves: list[dict[int, float]], b: int = 2000,
                     seed: int = 0, method: str = "bootstrap",
                     **adj) -> dict:
    """Uncertainty on an integer offset, by resampling the UNITS the mean is over.

    `per_unit_curves` is one score-vs-offset curve per probed frame.  The point estimate
    is the argmax of the mean curve; the interval is the distribution of that argmax over
    B resamples of the frames.  Reported as a mass function, because an integer estimator
    does not have a meaningful standard error and quoting one would be the
    "never quote an interval without its estimator" failure in a new costume.
    """
    if not per_unit_curves:
        return {"point": None, "refused": True, "reason": "empty", "mass": {}}
    offs = sorted(set(int(d) for c in per_unit_curves for d in c))
    M = np.array([[float(c.get(d, np.nan)) for d in offs] for c in per_unit_curves])
    ok = ~np.isnan(M).any(axis=1)
    M = M[ok]
    if M.shape[0] == 0:
        return {"point": None, "refused": True, "reason": "no_complete_curves", "mass": {}}
    point = adjudicate({d: v for d, v in zip(offs, M.mean(axis=0))}, method, **adj)
    rng = np.random.default_rng(seed)
    counts: dict = {}
    n_ref = 0
    for _ in range(b):
        idx = rng.integers(0, M.shape[0], M.shape[0])
        est = adjudicate({d: v for d, v in zip(offs, M[idx].mean(axis=0))}, method, **adj)
        if est.refused:
            n_ref += 1
        else:
            counts[est.offset] = counts.get(est.offset, 0) + 1
    tot = max(b, 1)
    mass = {int(k): round(v / tot, 4) for k, v in sorted(counts.items())}
    return {"point": point.offset, "refused": point.refused, "reason": point.reason,
            "n_units": int(M.shape[0]), "B": b,
            "mass": mass, "frac_refused": round(n_ref / tot, 4),
            "modal_mass": (max(mass.values()) if mass else 0.0),
            "estimator": "bootstrap over probed frames of argmax(mean curve); "
                         "an integer estimator has no meaningful SE, so the mass "
                         "function IS the interval"}


# --------------------------------------------------------------------------------- #
# estimator 1: pure counts (no pixels)                                               #
# --------------------------------------------------------------------------------- #
def scene_ref_offset(scene_dir, n_rig: int, cam: str = CAM) -> int:
    """The per-scene reference offset, derived from the scene, never hard-coded.

    ⛔ Do NOT write `+6`.  It is **+5** on `7c72937c` (MEASURED, renderer neighbour scan,
    `argmax_histogram = {5: 12}`), and hard-coding 6 puts that scene off by one.
    """
    n_dec, _meta, _M, _means, _head = mp4_motion_series(Path(scene_dir) / f"{cam}.mp4")
    e = count_delta(n_dec, int(n_rig))
    if e.refused:
        raise SystemExit(f"cannot derive reference offset for {scene_dir}: {e}")
    return int(e.offset)


def count_delta(n_mp4: int, n_rig: int) -> AlignEstimate:
    """`video_index = rig_index + (n_mp4 - n_rig)` -- the metadata predictor.

    Refuses on a NEGATIVE delta: a video shorter than the rig cannot be explained by a
    leader, and silently returning a negative index would read frames that are not the
    ones the rig describes.
    """
    d = int(n_mp4) - int(n_rig)
    extra = {"n_mp4": int(n_mp4), "n_rig": int(n_rig)}
    if d < 0:
        return AlignEstimate("count_delta", None, True, "negative_delta", extra=extra)
    return AlignEstimate("count_delta", d, False, "ok", extra=extra)


# --------------------------------------------------------------------------------- #
# estimator 2: image motion vs ego motion (pixels, NO renderer)                       #
# --------------------------------------------------------------------------------- #
def motion_lag(image_motion: np.ndarray, ego_motion: np.ndarray, max_lag: int = 15,
               min_overlap: int = 50, min_peak: float = 0.15,
               min_prominence: float = 0.01) -> AlignEstimate:
    """Cross-correlate two 1-D series under `mp4_index = rig_index + L`.

    `image_motion[i]` is mean |frame_i - frame_{i-1}| on the decoded mp4;
    `ego_motion[f]` is |p(f) - p(f-1)| from the rig trajectory.  NaN at index 0 of each.

    ⚠️ This instrument is BLIND when the ego is stationary -- no ego signal, nothing to
    correlate against -- which is exactly what happened on `7c72937c`.  The refusal rules
    catch that as `no_turnover`; the caller must not read a returned argmax without
    checking `refused`.
    """
    M = np.asarray(image_motion, dtype=np.float64)
    S = np.asarray(ego_motion, dtype=np.float64)
    curve: dict[int, float] = {}
    n_used: dict[int, int] = {}
    for L in range(-max_lag, max_lag + 1):
        f = np.arange(1, len(S))
        i = f + L
        ok = (i >= 1) & (i < len(M))
        a, bb = S[f[ok]], M[i[ok]]
        good = np.isfinite(a) & np.isfinite(bb)
        a, bb = a[good], bb[good]
        if len(a) < min_overlap or a.std() < 1e-12 or bb.std() < 1e-12:
            continue
        curve[L] = float(np.corrcoef(a, bb)[0, 1])
        n_used[L] = int(len(a))
    return adjudicate(curve, "motion_lag", min_peak=min_peak,
                      min_prominence=min_prominence,
                      extra={"n_pairs_by_lag": n_used, "max_lag": max_lag,
                             "ego_motion_std": round(float(np.nanstd(S)), 6)})


def leader_pad(head_absdiff: list[float] | np.ndarray, ego_speed_head: list[float] | np.ndarray,
               static_thresh: float = 0.5, moving_mps: float = 1.0) -> AlignEstimate:
    """How many FROZEN frames open the mp4 -- and a refusal when that is unanswerable.

    A frozen head block is only evidence of a synthetic leader **if the ego was moving**.
    On a stationary ego a real camera also produces near-identical frames, so the block
    length is not identifiable from the video alone.  The banked
    `ALIGNMENT_DIRECTION_GPUFREE.json` reports `static_head_block_frames = 5` for
    `7c72937c` whose rig speed is `0.0 m/s for the first 9 frames`; that number is
    therefore not independent evidence, and this function refuses rather than repeat it.
    """
    hd = np.asarray(head_absdiff, dtype=np.float64)
    spd = np.asarray(ego_speed_head, dtype=np.float64)
    extra = {"static_thresh": static_thresh,
             "ego_speed_head": [round(float(x), 3) for x in spd[:9]]}
    if spd.size == 0 or float(np.nanmax(spd[:min(9, spd.size)])) < moving_mps:
        return AlignEstimate("leader_pad", None, True, "ego_stationary_unidentifiable",
                             extra=extra)
    first_moving = next((i for i, v in enumerate(hd, start=1) if v > static_thresh), None)
    if first_moving is None:
        return AlignEstimate("leader_pad", None, True, "no_motion_in_head", extra=extra)
    extra["static_head_block_frames"] = int(first_moving)
    return AlignEstimate("leader_pad", int(first_moving) - 1, False, "ok", extra=extra)


# --------------------------------------------------------------------------------- #
# consensus                                                                          #
# --------------------------------------------------------------------------------- #
def consensus(estimates: list[AlignEstimate]) -> dict:
    """Combine independent estimators. Disagreement is reported, never averaged."""
    used = [e for e in estimates if not e.refused and e.offset is not None]
    vals = sorted({e.offset for e in used})
    return {"per_estimator": {e.method: (e.offset if not e.refused else None) for e in estimates},
            "refusals": {e.method: e.reason for e in estimates if e.refused},
            "n_admissible": len(used),
            "agree": len(vals) == 1 and len(used) >= 1,
            "offset": (vals[0] if len(vals) == 1 and used else None),
            "conflict": (vals if len(vals) > 1 else None)}


# --------------------------------------------------------------------------------- #
# series builders (need cv2 / the loader; imported lazily)                            #
# --------------------------------------------------------------------------------- #
def mp4_motion_series(mp4: str | Path, small=(192, 108)):
    """(n_decoded, meta_count, |dI| series, per-frame mean, full-res head |dI|)."""
    import cv2
    cap = cv2.VideoCapture(str(mp4))
    meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev = None
    M, means, head_full, prev_full = [], [], [], None
    n = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if n < 17:
            f = img.astype(np.int16)
            if prev_full is not None:
                head_full.append(float(np.abs(f - prev_full).mean()))
            prev_full = f
        g = cv2.cvtColor(cv2.resize(img, small, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        means.append(float(g.mean()))
        M.append(float(np.abs(g - prev).mean()) if prev is not None else np.nan)
        prev = g
        n += 1
    cap.release()
    return n, meta, np.array(M), np.array(means), head_full


def rig_ego_series(rig, cam: str = CAM):
    nf = rig.n_frames(cam)
    P = np.array([rig.T_rig_world(cam, f, 1)[:3, 3] for f in range(nf)])
    S = np.r_[np.nan, np.linalg.norm(np.diff(P, axis=0), axis=1)]
    return nf, S


# --------------------------------------------------------------------------------- #
# CLI                                                                                #
# --------------------------------------------------------------------------------- #
def _self_test(M: np.ndarray, S: np.ndarray, truth: int, max_lag: int = 15) -> dict:
    """Controls with a KNOWN answer, on the real series. See module docstring."""
    out = {}

    # (a) INJECTED SHIFT: re-index the image series by d; recovered must move by exactly d.
    for d in (-3, -1, 0, +2, +4):
        Md = np.r_[np.full(d, np.nan), M[:-d]] if d > 0 else (
            np.r_[M[-d:], np.full(-d, np.nan)] if d < 0 else M.copy())
        e = motion_lag(Md, S, max_lag=max_lag)
        want = truth + d
        out[f"inject_{d:+d}"] = {"want": want, "got": e.offset, "refused": e.refused,
                                 "reason": e.reason,
                                 "pass": (not e.refused and e.offset == want)}

    # (b) OUT-OF-WINDOW: push the truth outside the scan; the estimator must REFUSE.
    d = max_lag + truth + 2
    Md = np.r_[np.full(d, np.nan), M[:-d]]
    e = motion_lag(Md, S, max_lag=max_lag)
    out["out_of_window"] = {"want": "REFUSE", "got": e.offset, "refused": e.refused,
                            "reason": e.reason, "pass": bool(e.refused)}

    # (c) ZERO CONTROL: strip the leader so the true answer IS 0. An estimator that
    #     "always finds 6" fails here; one that reads the data returns 0.
    if truth > 0:
        e = motion_lag(M[truth:], S, max_lag=max_lag)
        out["zero_after_strip"] = {"want": 0, "got": e.offset, "refused": e.refused,
                                   "reason": e.reason,
                                   "pass": (not e.refused and e.offset == 0)}

    # (d) NULL CONTROL: independent noise -> must refuse.
    rng = np.random.default_rng(7)
    e = motion_lag(rng.normal(size=len(M)), rng.normal(size=len(S)), max_lag=max_lag)
    out["null_noise"] = {"want": "REFUSE", "got": e.offset, "refused": e.refused,
                         "reason": e.reason, "pass": bool(e.refused)}
    out["all_pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--loader-dir", default=None)
    ap.add_argument("--max-lag", type=int, default=15)
    ap.add_argument("--self-test", action="store_true",
                    help="run the known-answer controls on every scene")
    a = ap.parse_args()

    if a.loader_dir:
        import sys
        sys.path.insert(0, a.loader_dir)
    from nurec_loader import RigTrajectories

    report = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "cam": CAM, "max_lag": a.max_lag,
              "refusal_rules": ["boundary", "no_turnover", "weak", "not_separated"],
              "scenes": {}}
    for sd in a.scene_dir:
        scene = Path(sd).expanduser()
        t0 = time.time()
        n_dec, meta, M, means, head_full = mp4_motion_series(scene / f"{CAM}.mp4")
        rig = RigTrajectories(scene / "rig_trajectories.json")
        nf, S = rig_ego_series(rig)
        period_s = 1.0 / 30.0
        try:
            ts = rig.frame_timestamps_us(CAM, 0), rig.frame_timestamps_us(CAM, 1)
            period_s = (float(ts[1][1]) - float(ts[0][1])) / 1e6
        except Exception:
            pass
        spd = S[1:10] / max(period_s, 1e-9)

        e_cnt = count_delta(n_dec, nf)
        e_mot = motion_lag(M, S, max_lag=a.max_lag)
        e_led = leader_pad(head_full, spd)
        cons = consensus([e_cnt, e_mot, e_led])
        row = {"mp4_decodable_frames": n_dec, "mp4_CAP_PROP_FRAME_COUNT": meta,
               "rig_n_frames": nf, "decode_s": round(time.time() - t0, 1),
               "rig_frame_period_ms": round(period_s * 1000, 4),
               "ego_speed_mps_first_9": [round(float(x), 2) for x in spd],
               "fullres_head_absdiff": [round(float(x), 4) for x in head_full],
               "estimates": {e.method: e.to_json() for e in (e_cnt, e_mot, e_led)},
               "consensus": cons}
        if a.self_test and e_cnt.offset is not None:
            row["self_test"] = _self_test(M, S, e_cnt.offset, a.max_lag)
        report["scenes"][scene.name] = row
        print(f"== {scene.name[:8]}  mp4={n_dec} rig={nf}", flush=True)
        for e in (e_cnt, e_mot, e_led):
            print(f"   {e}", flush=True)
        print(f"   consensus: {cons}", flush=True)
        if "self_test" in row:
            print(f"   self-test all_pass={row['self_test']['all_pass']}: "
                  + ", ".join(f"{k}={'PASS' if v['pass'] else 'FAIL'}"
                              for k, v in row["self_test"].items()
                              if isinstance(v, dict)), flush=True)

    Path(a.out).expanduser().write_text(json.dumps(report, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
