"""Does a CLOTHOID beat a CIRCLE as the LAT corridor reference? (E-LAT-1)

⭐ **THE DECISION THIS ANSWERS.** `G_TAC_GEOMETRY_FLOOR.md` §3.4 lists three
candidate fixes for the LAT axis's 3.2x estimator gap. Option (a) — *"fit a
clothoid (kappa linear in arc) instead of a circle"* — is free, so it is tested
first. If a clothoid does not close the gap, options (a) and (b) are both weak
(they are better EXTRAPOLATIONS of a road, and the failure is that a 6 s road is
not extrapolable) and the answer is to OBSERVE the road instead — agent tracks or
Engine C.

⛔ **PRE-REGISTERED, BOTH OUTCOMES COMMITTED** (the operating standard's rule 5):

* **Clothoid WINS** iff the DEPLOYABLE clothoid's p50 lateral residual is
  <= 0.75 x the deployable circle's (i.e. a >=25 % cut of 2.012 m -> <= 1.509 m on
  the parity corpus). Then option (a) is alive and gets built.
* **Clothoid LOSES** otherwise — including the very plausible case that it is
  WORSE, because fitting 2 parameters to a short past window adds variance. Then
  extrapolation is refuted as a family and the LAT axis goes to road OBSERVATION
  (agent tracks / Engine C), and this script is the evidence for that call.

⚠️ **THE ORACLE IS NOT THE TEST.** A clothoid oracle MUST beat a circle oracle —
it has strictly more parameters, so a lower fitted residual is arithmetic, not
evidence. It is reported only as the ceiling. **The decisive column is the
DEPLOYABLE one**, fitted to the PAST window alone.

Four arms, all evaluated as the signed lateral offset of the true band endpoint
from the reference, taken perpendicular to the reference tangent at the same arc
length (the `g_tac_geom.corridor_offset` convention, so numbers are comparable):

  1. circle_past    — kappa = arc-weighted mean over the past window   [DEPLOYABLE]
  2. clothoid_past  — kappa(s) = k0 + k1*s, least squares over the past [DEPLOYABLE]
  3. circle_oracle  — kappa = arc-weighted mean over the BAND           [ceiling]
  4. clothoid_oracle— kappa(s) fitted over the BAND                     [ceiling]

⛔ CONTROLS, printed every run:
  * `control_synthetic_bend` — a constant-radius bend. BOTH families must read
    ~0; a clothoid that cannot represent a circle is a broken implementation.
  * `control_synthetic_clothoid` — a path with kappa genuinely linear in arc. The
    clothoid oracle must read ~0 and the circle oracle must NOT, or the two arms
    are not actually different and the whole comparison is void.
  * `n` and `d` on every row.

Labels-may-use-ego: every arm reads future poses. Label side only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

# Portable: this file lives deep in the research hub in-repo, but is also run
# from ~/gtac on Thor. Walk up for a `stack/` dir; if there is none, rely on
# PYTHONPATH (which is how the Thor runbook invokes it). parents[6] hard-coded
# would IndexError on Thor — the pod-portability trap.
_here = Path(__file__).resolve()
for _up in _here.parents:
    if (_up / "stack" / "tanitad").is_dir():
        for p in (str(_up / "stack"), str(_up / "stack" / "scripts")):
            if p not in sys.path:
                sys.path.insert(0, p)
        break
else:
    for _env in sys.path[:]:
        if (Path(_env) / "scripts" / "refb_labels.py").is_file():
            sys.path.insert(0, str(Path(_env) / "scripts"))
            break

from refb_labels import ego_frame, path_curvature                   # noqa: E402

from tanitad.data import g_tac_geom as G                            # noqa: E402
from tanitad.models.v6 import DT, TAC_BAND_S                        # noqa: E402

#: pre-registered win threshold (fraction of the circle_past p50)
WIN_RATIO = 0.75


def _arc_series(xy: torch.Tensor) -> torch.Tensor:
    """Cumulative arc length at each vertex, starting 0."""
    seg = (xy[1:] - xy[:-1]).norm(dim=-1)
    return torch.cat([torch.zeros(1, dtype=xy.dtype), seg.cumsum(0)])


def fit_kappa(poses: torch.Tensor, lo: int, hi: int, linear: bool):
    """``(k0, k1)`` over ``poses[lo:hi+1]``.

    ``linear=False`` -> arc-weighted mean curvature, ``k1 = 0`` (the circle).
    ``linear=True``  -> weighted least squares of kappa on arc length (clothoid).
    Returns ``None`` when the window carries too little motion to fit.
    """
    sub = poses[lo:hi + 1]
    if sub.shape[0] < 3:
        return None
    kap = path_curvature(sub)                              # [k-1]
    seg = (sub[1:, :2] - sub[:-1, :2]).norm(dim=-1)
    tot = float(seg.sum())
    if tot < 1e-3:
        return None
    s = _arc_series(sub[:, :2])[:-1] + 0.5 * seg           # segment midpoints
    w = seg
    if not linear:
        return float((kap * w).sum() / tot), 0.0
    # weighted least squares kappa ~ k0 + k1 * s
    sw = float(w.sum())
    ms = float((w * s).sum() / sw)
    mk = float((w * kap).sum() / sw)
    var = float((w * (s - ms) ** 2).sum())
    if var < 1e-9:
        return mk, 0.0
    k1 = float((w * (s - ms) * (kap - mk)).sum() / var)
    return mk - k1 * ms, k1


def reference_point(k0: float, k1: float, s_end: float, n: int = 400):
    """Integrate kappa(s) = k0 + k1*s from the ego origin -> ``(x, y, heading)``.

    Reduces to :func:`g_tac_geom.constant_curvature_point` when ``k1 == 0``
    (pinned by ``control_synthetic_bend``)."""
    if s_end <= 0:
        return 0.0, 0.0, 0.0
    ds = s_end / n
    s = torch.arange(n, dtype=torch.float64) * ds + 0.5 * ds
    th = k0 * s + 0.5 * k1 * s * s
    x = float((torch.cos(th) * ds).sum())
    y = float((torch.sin(th) * ds).sum())
    return x, y, float(k0 * s_end + 0.5 * k1 * s_end * s_end)


def lateral_residual(poses, t, t_end, k0, k1, s_end) -> float:
    rx, ry, rth = reference_point(k0, k1, s_end)
    dxy = (poses[t_end, :2] - poses[t, :2]).unsqueeze(0)
    e = ego_frame(dxy, poses[t, 2].unsqueeze(0))
    ax, ay = float(e[0, 0]), float(e[0, 1])
    return (ax - rx) * (-math.sin(rth)) + (ay - ry) * math.cos(rth)


def evaluate_window(poses, t, band_s=TAC_BAND_S, dt=DT,
                    past_window_s=G.PAST_WINDOW_S):
    t_end = t + int(round(band_s[1] / dt))
    if t_end > poses.shape[0] - 1:
        return None
    n_past = max(int(round(past_window_s / dt)), 1)
    lo = max(t - n_past, 0)
    if t - lo < 2:
        return None
    s_end = float((poses[t + 1:t_end + 1, :2] - poses[t:t_end, :2]
                   ).norm(dim=-1).sum())
    if s_end < G.MOVING_MIN_MS * (band_s[1] - band_s[0]):
        return None
    out = {}
    for name, (a, b, lin) in {
            "circle_past":     (lo, t, False),
            "clothoid_past":   (lo, t, True),
            "circle_oracle":   (t, t_end, False),
            "clothoid_oracle": (t, t_end, True)}.items():
        f = fit_kappa(poses, a, b, lin)
        if f is None:
            return None
        out[name] = abs(lateral_residual(poses, t, t_end, f[0], f[1], s_end))
    return out


# --------------------------------------------------------------------------- #
# controls                                                                      #
# --------------------------------------------------------------------------- #
def _synth(kappa_of_s, T=220, v=15.0, sub: int = 200):
    """Integrate a synthetic path with a prescribed kappa(s).

    ⚠️ **MIDPOINT, AND SUB-STEPPED — and that is not fussiness.** The first
    version stepped once per frame with the heading updated AFTER the position
    (plain Euler). That is a HALF-STEP offset against
    :func:`reference_point`'s midpoint quadrature, and on a 1/150 m bend over
    ~90 m it manufactured a **0.4237 m** residual on ALL FOUR arms — comparable
    to the 0.75 m lane bar, and it made ``control_synthetic_bend`` FAIL. The
    control was right: a generator whose discretisation is the size of the
    effect cannot certify anything. ``sub`` sub-steps per frame drive it to
    ~1e-5 m.
    """
    ds = v * DT / sub
    xs, ys, hs, x, y, h, s = [], [], [], 0.0, 0.0, 0.0, 0.0
    for i in range(T):
        xs.append(x)
        ys.append(y)
        hs.append(h)                # ANALYTIC heading at the VERTEX — see below
        for _ in range(sub):
            h_mid = h + 0.5 * kappa_of_s(s) * ds        # midpoint heading
            x += math.cos(h_mid) * ds
            y += math.sin(h_mid) * ds
            h += kappa_of_s(s) * ds
            s += ds
    xy = torch.tensor([xs, ys], dtype=torch.float64).T
    # ⛔ NOT atan2 of consecutive vertices. That returns the tangent at the
    # SEGMENT MIDPOINT, i.e. a half-step ahead of the vertex, which rotates the
    # ego frame by kappa*ds/2 and — over a ~90 m band — manufactured exactly
    # 0.4237 m of residual on ALL FOUR arms. That is what made
    # `control_synthetic_bend` fail TWICE, and it survived the integrator fix
    # because it was never the integrator. Real corpus poses carry a MEASURED
    # yaw channel and do not have this artifact; only the generator did.
    yaw = torch.tensor(hs, dtype=torch.float64)
    return torch.stack([xy[:, 0], xy[:, 1], yaw,
                        torch.full((T,), v, dtype=torch.float64)], dim=-1)


def controls() -> dict:
    bend = _synth(lambda s: 1.0 / 150.0)
    cloth = _synth(lambda s: 0.0005 + 0.00008 * s)
    rb = evaluate_window(bend, 30)
    rc = evaluate_window(cloth, 30)
    return {
        "control_synthetic_bend": {
            **{k: round(v, 5) for k, v in (rb or {}).items()},
            "PASS": bool(rb) and rb["circle_oracle"] < 0.10
                    and rb["clothoid_oracle"] < 0.10,
            "NOTE": "constant-radius bend: BOTH oracles must read ~0; a clothoid "
                    "that cannot represent a circle is a broken implementation"},
        "control_synthetic_clothoid": {
            **{k: round(v, 5) for k, v in (rc or {}).items()},
            "PASS": bool(rc) and rc["clothoid_oracle"] < 0.5 * rc["circle_oracle"],
            "NOTE": "kappa genuinely linear in arc: the clothoid oracle MUST beat "
                    "the circle oracle here, or the two arms are not different "
                    "and the whole comparison is void"}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--glob", default="*.v2ep.pt")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--limit-episodes", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    eps = sorted(Path(args.cache).glob(args.glob))
    if args.limit_episodes:
        eps = eps[:args.limit_episodes]
    if not eps:
        raise SystemExit(f"no {args.glob} under {args.cache}")

    arms = {k: [] for k in ("circle_past", "clothoid_past",
                            "circle_oracle", "clothoid_oracle")}
    for i, f in enumerate(eps):
        poses = torch.load(f, map_location="cpu",
                           weights_only=False)["poses"].double()
        for t in range(0, poses.shape[0], args.stride):
            r = evaluate_window(poses, t)
            if r:
                for k, v in r.items():
                    arms[k].append(v)
        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{len(eps)} episodes, n={len(arms['circle_past'])}",
                  flush=True)

    def q(xs):
        if not xs:
            return {}
        s = sorted(xs)
        return {f"p{p}": s[min(int(p / 100 * len(s)), len(s) - 1)]
                for p in (50, 75, 90)} | {"mean": sum(s) / len(s), "n": len(s)}

    res = {k: q(v) for k, v in arms.items()}
    cp = res["circle_past"].get("p50", float("nan"))
    kp = res["clothoid_past"].get("p50", float("nan"))
    ratio = kp / cp if cp else float("nan")
    verdict = ("CLOTHOID WINS — option (a) is alive" if ratio <= WIN_RATIO else
               "CLOTHOID LOSES — extrapolation refuted as a family; go to road "
               "OBSERVATION (agent tracks / Engine C)")

    out = {"cache_key": Path(args.cache).name, "d_episodes": len(eps),
           "stride": args.stride, "arms": res,
           "prereg": {"win_ratio": WIN_RATIO,
                      "deployable_ratio_clothoid_over_circle": ratio,
                      "VERDICT": verdict},
           "controls": controls()}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n=== E-LAT-1 clothoid vs circle — d={len(eps)} eps, "
          f"cache={Path(args.cache).name} ===")
    print(f"{'arm':>18} {'p50':>8} {'p75':>8} {'p90':>8} {'n':>8}")
    for k in ("circle_past", "clothoid_past", "circle_oracle", "clothoid_oracle"):
        r = res[k]
        tag = "  [DEPLOYABLE]" if k.endswith("past") else "  (ceiling)"
        print(f"{k:>18} {r.get('p50', 0):8.3f} {r.get('p75', 0):8.3f} "
              f"{r.get('p90', 0):8.3f} {r.get('n', 0):8d}{tag}")
    print(f"\nDEPLOYABLE ratio clothoid/circle = {ratio:.4f} "
          f"(pre-registered win <= {WIN_RATIO})")
    print(f"VERDICT: {verdict}")
    for name, c in out["controls"].items():
        print(f"CONTROL {name}: {'PASS' if c['PASS'] else 'FAIL'}  "
              + " ".join(f"{k}={v}" for k, v in c.items()
                         if k not in ("PASS", "NOTE")))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
