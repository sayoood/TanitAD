"""R4b — IS THE SHAPE DEFECT A DECODER *PARAMETERISATION* DEFECT?

⛔ THE CLAIM UNDER TEST, quoted from `2026-09-06-goal-point/PREREG.md` §7 and
`RESULT.md` §5: on refcv4b's banked T1 dump the PICKED ANCHOR's curvature MAE is
**0.004019** while the EMITTED plan's is **0.008150** — the decoder's free-waypoint
refinement roughly DOUBLES curvature error (2.03x) while HALVING ADE (0.4281 ->
0.2965). If that holds, `os - ha0_ext`'s curvature gap is generated in the DECODER
OUTPUT PARAMETERISATION, not in routing, and the pre-registered next lever is a
curvature-preserving output basis.

⛔ WHAT THIS PROBE DOES AND DOES NOT ANSWER.
It re-parameterises the ALREADY-EMITTED offset `off = emitted - anchor`. So each arm
is a LOWER BOUND on what a decoder TRAINED in that basis could reach: a trained
decoder would spend its basis better than a projection of a free offset does. The
ORACLE-in-basis arms bracket it from ABOVE (they fit the basis to the GROUND TRUTH,
which is inadmissible as a capability claim and stamped as such). A parameterisation
whose ORACLE ceiling still cannot beat the Pareto front is refuted for a trained arm
too; one whose projection floor already beats it is confirmed. Anything between is
reported as "needs the trained arm", not resolved by assertion.

ZERO GPU. Geometry via `taniteval.four_families` — IMPORTED, never re-derived.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

from taniteval import four_families as ff
from tanitad.refs import feasible_decode as fd
from tanitad.refs.refc_tactical import LAT_CLASSES as LATC
from tanitad.refs.refc_tactical import factor_from_kinematics

#: the model-free arms banked in the dump (the controls that must read KNOWN values)
FLOORS = ("ha0", "ha0_ext", "ha")

#: ⛔ K-CONTROLS. Every one of these is a value published from THIS dump by a
#: DIFFERENT probe. If any misses, the surface is not the banked surface and the
#: whole panel is VOID — the discipline that caught three manufactured results.
KNOWN = {
    "ade": {"ha0": 0.6723, "ha0_ext": 0.2874, "ha": 0.2996,
            "emitted": 0.2965, "anchor": 0.4281},
    "curv": {"emitted": 0.008150, "anchor": 0.004019, "ha0": 0.006802},
    "head_deg": {"emitted": 1.2964, "anchor": 1.7665},
    "cross": {"emitted": 0.0978, "anchor": 0.1754},
}
TOL = 5e-4


def load(dump: str) -> dict:
    out = {k: [] for k in ("g", "eid", "anchor", "emitted", "v0")}
    for a in FLOORS:
        out[a] = []
    for f in sorted(glob.glob(os.path.join(dump, "ep*.npz"))):
        a = np.load(f)
        b = np.load(os.path.join(dump, "decisions", os.path.basename(f)))
        n = len(a["ws"])
        out["g"].append(a["g"])
        out["v0"].append(a["v0"])
        out["eid"].append(np.full(n, int(os.path.basename(f)[2:-4]), np.int64))
        # ⛔ the 2 s SCORED grid, 4 waypoints at dt = 0.5 s. `g` is [n, 4, 2],
        # so both model arms are sliced to the same 4 slots — comparing an
        # 8-slot plan against a 4-slot target is the derived-constant trap.
        out["anchor"].append(b["sel_bank_nav_true"][:, :4])
        out["emitted"].append(b["plan_full_nav_true"][:, :4])
        for fl in FLOORS:
            out[fl].append(a[fl])
    D = {k: np.concatenate(v, 0) for k, v in out.items()}
    # the identity that pins the slice: plan_full[:, :4] IS the banked `os` arm
    os_arm = np.concatenate(
        [np.load(f)["os"] for f in sorted(glob.glob(os.path.join(dump,
                                                                 "ep*.npz")))],
        0)
    D["_slice_identity_max_abs"] = float(np.abs(D["emitted"] - os_arm).max())
    return D


# ---------------------------------------------------------------------------
# the parameterisations — each is a map on the OFFSET, never on the anchor
# ---------------------------------------------------------------------------

def _basis(h: int, k: int) -> np.ndarray:
    """[h, k+1] polynomial basis in TIME, orthonormalised. Column 0 is the
    constant, so `poly_0` is a PURE TRANSLATION of the refinement.

    ⚠️ CORRECTED, MEASURED. A pure translation is NOT curvature-inert here, and
    a first draft of this comment said it was. `_seq_geometry` PREPENDS THE EGO
    ORIGIN before differencing, so translating the path changes the FIRST step
    (origin -> waypoint 0) and therefore the first heading and the first
    curvature pair. `poly_0` measures 0.012097 1/m — WORSE than the free offset's
    0.008150 — which is the arithmetic saying exactly that. Kept as the
    structural end of the family and as a reminder that "smoother offset" and
    "smoother PATH" are different objects."""
    t = np.linspace(1.0, float(h), h) / float(h)
    B = np.stack([t ** j for j in range(k + 1)], 1)
    Q, _ = np.linalg.qr(B)
    return Q


def poly_project(off: np.ndarray, k: int) -> np.ndarray:
    """Least-squares projection of the offset onto a degree-k polynomial in
    time, per axis. ⛔ The ANCHOR is untouched: this constrains only the
    REFINEMENT, which is the object PREREG §7 named."""
    Q = _basis(off.shape[1], k)                     # [H, k+1]
    return np.einsum("hj,njc->nhc", Q, np.einsum("hj,nhc->njc", Q, off))


def ridge_smooth(off: np.ndarray, lam: float) -> np.ndarray:
    """argmin_u ||u - off||^2 + lam * ||D2 u||^2, closed form, per axis.

    ⭐ This IS "penalise curvature in the refinement", solved exactly instead of
    trained: D2 is the discrete second difference of the offset, and a second
    difference in position is what a curvature error is made of. lam -> 0 gives
    the free offset; lam -> inf gives the affine offset (D2 u = 0)."""
    h = off.shape[1]
    D2 = np.zeros((max(h - 2, 0), h))
    for i in range(h - 2):
        D2[i, i], D2[i, i + 1], D2[i, i + 2] = 1.0, -2.0, 1.0
    A = np.eye(h) + lam * (D2.T @ D2)
    Ainv = np.linalg.inv(A)
    return np.einsum("gh,nhc->ngc", Ainv, off)


def feasible(path: np.ndarray, v0: np.ndarray, dt: float) -> np.ndarray:
    """The SHIPPED Stage-0 kinematic projection, applied to a 4-waypoint path.

    ⭐ THIS IS AN INTEGRATION QUESTION, NOT A NEW LEVER: `project_feasible`
    already exists behind `--feasible-decode`, and before proposing a smoothing
    seam we must know whether the seam we ALREADY HAVE does the job. It does
    NOT, and the reason is a scope distinction worth stating: it caps curvature
    MAGNITUDE (|kappa| <= kappa_max, plus the Kamm disc), while R4b is about
    curvature ERROR against the GT. A plan can sit far inside the friction
    circle and still be wrong.

    ⛔ `project_feasible` requires index 0 to BE the ego origin and returns it
    unchanged, so the origin is prepended here and stripped after — the same
    convention `_seq_geometry` uses when it differences from the car.
    """
    p = np.concatenate([np.zeros_like(path[:, :1]), path], 1)
    out = fd.project_feasible(torch.from_numpy(p.astype(np.float32)),
                              v0=torch.from_numpy(v0.astype(np.float32)),
                              dt=dt, clamp_entry=False)
    return out.numpy()[:, 1:]


def build_arms(anchor: np.ndarray, emitted: np.ndarray,
               g: np.ndarray) -> dict:
    off = emitted - anchor
    arms = {"anchor": anchor, "emitted": emitted}
    for a in (0.25, 0.5, 0.75):
        arms["shrink_%0.2f" % a] = anchor + a * off
    for k in (0, 1, 2):
        arms["poly_%d" % k] = anchor + poly_project(off, k)
    for lam in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        arms["ridge_%g" % lam] = anchor + ridge_smooth(off, lam)
    # ⛔⛔ ORACLE CEILINGS — INADMISSIBLE AS A CAPABILITY CLAIM. These fit the
    # basis to the GROUND TRUTH, i.e. they read the answer at inference. Their
    # only job is to price what a decoder TRAINED in that basis could reach, so
    # that a refutation of the parameterisation is a refutation of the BASIS and
    # not of one projection of one free offset.
    ogoff = g - anchor
    for k in (0, 1, 2, 3):
        arms["ORACLE_poly_%d" % k] = anchor + poly_project(ogoff, k)
    # ⛔⛔ THE DELIBERATE-REGRESSION CONTROL. `ROUGHEN` puts BACK, doubled, the
    # exact high-frequency component the ridge removes: off + (off - ridge(off)).
    # If the curvature instrument does not read this arm as separated WORSE, it
    # is not measuring what the smoothing arms claim to improve, and the whole
    # panel is uninterpretable. A gate a regression cannot fail is not a gate.
    hf = off - ridge_smooth(off, 1.0)
    arms["ROUGHEN_hf_x2"] = anchor + off + hf
    return arms, off


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def ade(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    return np.linalg.norm(p - g, axis=-1).mean(1)          # [n] per window


def step_errors(pred: np.ndarray, gt: np.ndarray, dt: float) -> dict:
    P = ff._seq_geometry(torch.from_numpy(pred.astype(np.float32)), dt)
    G = ff._seq_geometry(torch.from_numpy(gt.astype(np.float32)), dt)
    dh = (P["heading"] - G["heading"]).numpy()
    dh = (dh + math.pi) % (2 * math.pi) - math.pi
    return dict(
        head=np.abs(dh), head_m=(P["valid"] & G["valid"]).numpy(),
        curv=np.abs((P["curvature"] - G["curvature"]).numpy()),
        curv_m=(P["pair_valid"] & G["pair_valid"]).numpy(),
        cross=np.abs((P["cross"] - G["cross"]).numpy()),
        speed=np.abs((P["speed"] - G["speed"]).numpy()),
        speed_m=np.ones_like(P["speed"].numpy(), dtype=bool),
    )


def pooled(err, m, rows) -> float:
    e, mm = err[rows], m[rows]
    d = mm.sum()
    return float((e * mm).sum() / d) if d else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    D = load(args.dump)
    g, eid = D["g"], D["eid"]
    n = g.shape[0]
    arms, off = build_arms(D["anchor"], D["emitted"], g)
    for fl in FLOORS:
        arms[fl] = D[fl]

    # the SHIPPED Stage-0 projection, alone and composed with the R4b smoother
    arms["FEASIBLE_emitted"] = feasible(arms["emitted"], D["v0"], args.dt)
    for lam in (1.0, 3.0):
        k = "ridge_%g" % lam
        arms["FEASIBLE_" + k] = feasible(arms[k], D["v0"], args.dt)

    E = {a: step_errors(v, g, args.dt) for a, v in arms.items()}
    A = {a: ade(v, g) for a, v in arms.items()}
    allrows = np.arange(n)

    R = {
        "_is": ("R4b: is the `os - ha0_ext` SHAPE gap a DECODER OUTPUT "
                "PARAMETERISATION defect? Re-parameterises the emitted "
                "refinement `off = emitted - anchor` on the banked refcv4b T1 "
                "dump. ZERO GPU."),
        "evidence_class": "MEASURED (ours)",
        "tier": "T1 (self-action open loop; refcv4b's decoder is deterministic "
                "at inference, D-REFCV4B-SEED-SCOPE)",
        "dump": os.path.abspath(args.dump),
        "n_windows": int(n), "n_episodes": int(len(np.unique(eid))),
        "dt_s": args.dt,
        "grid": "the 2 s SCORED grid, 4 waypoints at 0.5 s",
        "estimator": ("paired episode-cluster bootstrap over the %d episodes, "
                      "steps RE-POOLED inside each draw (n_boot=%d, seed=0)"
                      % (len(np.unique(eid)), args.n_boot)),
        "variance_answered": (
            "would another draw of EPISODES say this? — NOT another training "
            "run (H-ESTIM-SEED-1) and NOT another inference sample. Every arm "
            "here is ONE checkpoint's OWN output re-parameterised, so the two "
            "operands of every margin come from the SAME forward pass: there "
            "is no training or sampling variance to answer for."),
        "geometry_source": ("taniteval.four_families._seq_geometry — imported, "
                            "not re-derived"),
    }

    # ---- CONTROLS -------------------------------------------------------- #
    ctl = {}
    ctl["slice_identity_plan_full_4slots_vs_os_maxabs"] = D[
        "_slice_identity_max_abs"]
    ctl["K0_slice_identity"] = ("PASS" if D["_slice_identity_max_abs"] < 1e-6
                                else "FAIL")
    # the two structural identities of the shrink family
    s0 = D["anchor"] + 0.0 * off
    s1 = D["anchor"] + 1.0 * off
    ctl["K1_shrink0_IS_anchor"] = ("PASS" if np.abs(s0 - D["anchor"]).max() == 0
                                   else "FAIL")
    ctl["K1_shrink1_IS_emitted"] = ("PASS"
                                    if np.abs(s1 - D["emitted"]).max() < 1e-6
                                    else "FAIL")
    # a NO-INFORMATION control: poly_0 of a ZERO offset must equal the anchor
    ctl["K2_zero_offset_IS_anchor"] = (
        "PASS" if np.abs(poly_project(np.zeros_like(off), 2)).max() == 0
        else "FAIL")
    known = {}
    for arm, want in KNOWN["ade"].items():
        got = float(A[arm].mean())
        known["ade_%s" % arm] = {"known": want, "got": round(got, 6),
                                 "d": round(got - want, 6),
                                 "verdict": "PASS" if abs(got - want) < TOL
                                 else "FAIL"}
    for arm, want in KNOWN["curv"].items():
        got = pooled(E[arm]["curv"], E[arm]["curv_m"], allrows)
        known["curv_%s" % arm] = {"known": want, "got": round(got, 6),
                                  "d": round(got - want, 6),
                                  "verdict": "PASS" if abs(got - want) < TOL
                                  else "FAIL"}
    for arm, want in KNOWN["head_deg"].items():
        got = math.degrees(pooled(E[arm]["head"], E[arm]["head_m"], allrows))
        known["head_deg_%s" % arm] = {"known": want, "got": round(got, 4),
                                      "d": round(got - want, 4),
                                      "verdict": "PASS" if abs(got - want) < 5e-3
                                      else "FAIL"}
    for arm, want in KNOWN["cross"].items():
        got = float(E[arm]["cross"].mean())
        known["cross_%s" % arm] = {"known": want, "got": round(got, 6),
                                   "d": round(got - want, 6),
                                   "verdict": "PASS" if abs(got - want) < TOL
                                   else "FAIL"}
    ctl["K3_known_values"] = known
    ctl["K3_verdict"] = ("PASS" if all(v["verdict"] == "PASS"
                                       for v in known.values()) else "FAIL")
    R["controls"] = ctl

    # ---- per-arm, FOUR FAMILIES ------------------------------------------ #
    R["per_arm"] = {}
    for a, v in arms.items():
        pt = torch.from_numpy(v.astype(np.float32))
        gt = torch.from_numpy(g.astype(np.float32))
        lat = ff.lateral(pt, gt, dt=args.dt, eid=None, n_boot=0)
        lon = ff.longitudinal(pt, gt, dt=args.dt, eid=None, n_boot=0)
        tac = ff.tactical_from_trajectory(pt, gt, dt=args.dt, eid=None,
                                          n_boot=0)
        R["per_arm"][a] = {
            "ADE_2s_m": round(float(A[a].mean()), 6),
            "LATERAL": {
                "curvature_mae_1pm": lat["curvature_mae_1pm"],
                "heading_mae_deg": lat["heading_mae_deg"],
                "yaw_rate_mae_dps": lat.get("yaw_rate_mae_dps"),
                "cross_mae_m": lat["cross_mae_m"],
                "n_steps_curvature": lat["n_steps_curvature"],
            },
            "LONGITUDINAL": {
                # ⚠️ `speed_mae_ms` / `accel_mae_ms2` / the distance-keeping block
                # come back None on this dump: `ff.longitudinal` needs a LEAD
                # AGENT and the banked dump carries none. Reported per family
                # WITH ITS REASON and its n, never silently dropped (the
                # four-families rule, point 5). The speed axis IS covered, by
                # the step-wise `speed_mae_ms` computed from `_seq_geometry` in
                # the paired blocks below — same geometry, no lead needed.
                "speed_mae_ms": lon.get("speed_mae_ms"),
                "speed_mae_ms_UNAVAILABLE_REASON": (
                    None if lon.get("speed_mae_ms") is not None else
                    "ff.longitudinal returns None without a lead-agent block; "
                    "see paired__vs_* -> speed_mae_ms for the step-wise form"),
                "along_mae_m": lon.get("along_mae_m"),
                "accel_mae_ms2": lon.get("accel_mae_ms2"),
                "speed_mae_ms_stepwise": round(
                    float(np.abs(E[a]["speed"]).mean()), 6),
            },
            "TACTICAL": {
                "lat_accuracy": tac.get("lat", {}).get("accuracy"),
                "lon_accuracy": tac.get("lon", {}).get("accuracy"),
            },
            "STRATEGIC": ("NOT COMPUTABLE on this dump: no route/goal label "
                          "travels with it (ff.strategic needs the option set); "
                          "reported per family with its reason, never dropped."),
        }

    # ---- PAIRED margins -------------------------------------------------- #
    eps = np.unique(eid)
    rows_by_ep = {e: np.where(eid == e)[0] for e in eps}
    rng = np.random.default_rng(0)
    draws = [np.concatenate([rows_by_ep[eps[j]] for j in
                             rng.integers(0, len(eps), len(eps))])
             for _ in range(args.n_boot)]

    def m_pooled(a, b, field, mask):
        m = E[a][mask] & E[b][mask]          # INTERSECTION mask: strictly paired
        pt = pooled(E[a][field], m, allrows) - pooled(E[b][field], m, allrows)
        bs = np.array([pooled(E[a][field], m, r) - pooled(E[b][field], m, r)
                       for r in draws])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(delta=round(float(pt), 6),
                    ci95=[round(float(lo), 6), round(float(hi), 6)],
                    separated=bool(lo > 0 or hi < 0), n_steps=int(m.sum()))

    def m_window(a, b, vals):
        d = vals[a] - vals[b]
        bs = np.array([d[r].mean() for r in draws])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(delta=round(float(d.mean()), 6),
                    ci95=[round(float(lo), 6), round(float(hi), 6)],
                    separated=bool(lo > 0 or hi < 0))

    cross_w = {a: E[a]["cross"].mean(1) for a in arms}
    speed_w = {a: E[a]["speed"].mean(1) for a in arms}

    def block(a, b):
        return {
            "ADE_2s_m": m_window(a, b, A),
            "curvature_mae_1pm": m_pooled(a, b, "curv", "curv_m"),
            "heading_mae_rad": m_pooled(a, b, "head", "head_m"),
            "cross_mae_m": m_window(a, b, cross_w),
            "speed_mae_ms": m_window(a, b, speed_w),
        }

    R["paired__vs_EMITTED"] = {a: block(a, "emitted") for a in arms
                               if a != "emitted"}
    R["paired__vs_ANCHOR"] = {a: block(a, "anchor") for a in arms
                              if a != "anchor"}
    R["paired__the_MECHANISM"] = {
        "emitted__minus__anchor": block("emitted", "anchor"),
        "emitted__minus__ha0": block("emitted", "ha0"),
        "anchor__minus__ha0": block("anchor", "ha0"),
        "emitted__minus__ha0_ext": block("emitted", "ha0_ext"),
    }
    for k, v in list(R["paired__vs_EMITTED"].items()) + \
            list(R["paired__vs_ANCHOR"].items()) + \
            list(R["paired__the_MECHANISM"].items()):
        h = v["heading_mae_rad"]
        v["heading_mae_deg"] = dict(
            delta=round(math.degrees(h["delta"]), 4),
            ci95=[round(math.degrees(h["ci95"][0]), 4),
                  round(math.degrees(h["ci95"][1]), 4)],
            separated=h["separated"])

    # ---- TACTICAL, paired, per lateral class ----------------------------- #
    def lat_of(arm):
        dy, dv, v0_, v1_, _ = ff.maneuver_kinematics(
            torch.from_numpy(arms[arm].astype(np.float32)), args.dt)
        return factor_from_kinematics(dy, dv, v0_, v1_)[0].numpy()

    dyg, dvg, v0g, v1g, _ = ff.maneuver_kinematics(
        torch.from_numpy(g.astype(np.float32)), args.dt)
    lat_gt = factor_from_kinematics(dyg, dvg, v0g, v1g)[0].numpy()
    lat_pred = {a: lat_of(a) for a in arms}
    R["tactical_label_source"] = (
        "trajectory-derived via refc_tactical.factor_from_kinematics — the "
        "same gate ff.tactical_from_trajectory uses. ⛔ NOT the |dyaw| > 0.15 "
        "gate: the HUMAN fails that one on 3 of 9.")
    turn = np.where(lat_gt != 0)[0]
    R["paired_TACTICAL"] = {"n_turn_windows": int(len(turn)),
                            "per_class": {}}
    for ci, cname in enumerate(LATC):
        sub = np.where(lat_gt == ci)[0]
        blk = {"n": int(len(sub)), "per_arm": {}, "vs_emitted": {}}
        for a in arms:
            blk["per_arm"][a] = round(float((lat_pred[a][sub] == ci).mean()), 4) \
                if len(sub) else None
        dc = [r[np.isin(r, sub)] for r in draws]
        dc = [r for r in dc if len(r)]
        for a in arms:
            if a == "emitted" or not len(sub):
                continue
            ha_ = (lat_pred[a] == ci).astype(float)
            hb_ = (lat_pred["emitted"] == ci).astype(float)
            pt = ha_[sub].mean() - hb_[sub].mean()
            bs = np.array([ha_[r].mean() - hb_[r].mean() for r in dc])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            blk["vs_emitted"][a] = dict(
                delta=round(float(pt), 4),
                ci95=[round(float(lo), 4), round(float(hi), 4)],
                separated=bool(lo > 0 or hi < 0))
        R["paired_TACTICAL"]["per_class"][cname] = blk
    # turn-window lateral ACCURACY (the PREREG P1 endpoint's instrument)
    # ⛔ FULL-LENGTH, indexed by GLOBAL row. The bootstrap draws are global row
    # indices; an array built over `turn` positions only would be silently
    # mis-indexed (it raised here rather than lying, which is the good case).
    acc = {a: (lat_pred[a] == lat_gt).astype(float) for a in arms}
    dc = [r[np.isin(r, turn)] for r in draws]
    dc = [r for r in dc if len(r)]
    R["paired_TACTICAL"]["turn_accuracy_per_arm"] = {
        a: round(float(v[turn].mean()), 4) for a, v in acc.items()}
    R["paired_TACTICAL"]["turn_accuracy_vs_emitted"] = {}
    for a in arms:
        if a == "emitted":
            continue
        d = acc[a] - acc["emitted"]
        bs = np.array([d[r].mean() for r in dc])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        R["paired_TACTICAL"]["turn_accuracy_vs_emitted"][a] = dict(
            delta=round(float(d[turn].mean()), 4),
            ci95=[round(float(lo), 4), round(float(hi), 4)],
            separated=bool(lo > 0 or hi < 0))

    # ---- ⛔⛔ THE HELD-OUT LAMBDA. The sweep above is SCORED ON THE DATA IT
    # WAS SWEPT OVER, which is failure mode (3) of the probe rules verbatim:
    # "lambda selected on the test set -> picked 1e6, which shrinks the ridge to
    # the constant predictor and scores EXACTLY +0.0000". So: split the EPISODES,
    # select lambda on the FIT half by the pre-stated rule, and report the SCORE
    # half only. ⭐ The sweep's own answer is that this barely matters here —
    # EVERY lambda in [0.3, 100] is separated-better on curvature and none is
    # separated-worse on ADE — but "it did not matter" is a RESULT, not a reason
    # to skip the split.
    lam_grid = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    fit_eps = eps[::2]
    sco_eps = eps[1::2]
    fit_rows = np.where(np.isin(eid, fit_eps))[0]
    sco_rows = np.where(np.isin(eid, sco_eps))[0]
    sel_rule = ("argmin curvature_mae over the FIT episodes, subject to "
                "ADE(FIT) <= ADE_emitted(FIT) + 0.005 m; stated BEFORE the "
                "split was scored")
    cand = {}
    for lam in lam_grid:
        a = "ridge_%g" % lam
        cand[a] = {
            "curv_fit": pooled(E[a]["curv"], E[a]["curv_m"], fit_rows),
            "ade_fit": float(A[a][fit_rows].mean()),
        }
    ade_e_fit = float(A["emitted"][fit_rows].mean())
    ok = {k: v for k, v in cand.items()
          if v["ade_fit"] <= ade_e_fit + 0.005}
    best = min(ok, key=lambda k: ok[k]["curv_fit"]) if ok else None
    hold = {
        "selection_rule": sel_rule,
        "fit_episodes": int(len(fit_eps)), "score_episodes": int(len(sco_eps)),
        "fit_windows": int(len(fit_rows)), "score_windows": int(len(sco_rows)),
        "candidates_fit": {k: {kk: round(vv, 6) for kk, vv in v.items()}
                           for k, v in cand.items()},
        "ade_emitted_fit": round(ade_e_fit, 6),
        "selected": best,
    }
    if best is not None:
        draws_s = [r[np.isin(r, sco_rows)] for r in draws]
        draws_s = [r for r in draws_s if len(r)]

        def m_pooled_s(a, b, field, mask):
            m = E[a][mask] & E[b][mask]
            pt = (pooled(E[a][field], m, sco_rows)
                  - pooled(E[b][field], m, sco_rows))
            bs = np.array([pooled(E[a][field], m, r) - pooled(E[b][field], m, r)
                           for r in draws_s])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            return dict(delta=round(float(pt), 6),
                        ci95=[round(float(lo), 6), round(float(hi), 6)],
                        separated=bool(lo > 0 or hi < 0))

        def m_window_s(a, b, vals):
            d = vals[a] - vals[b]
            bs = np.array([d[r].mean() for r in draws_s])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            return dict(delta=round(float(d[sco_rows].mean()), 6),
                        ci95=[round(float(lo), 6), round(float(hi), 6)],
                        separated=bool(lo > 0 or hi < 0))

        hold["SCORE_HALF__selected_minus_emitted"] = {
            "ADE_2s_m": m_window_s(best, "emitted", A),
            "curvature_mae_1pm": m_pooled_s(best, "emitted", "curv", "curv_m"),
            "heading_mae_rad": m_pooled_s(best, "emitted", "head", "head_m"),
            "cross_mae_m": m_window_s(best, "emitted", cross_w),
            "speed_mae_ms": m_window_s(best, "emitted", speed_w),
        }
        hold["SCORE_HALF__points"] = {
            a: {"ADE_2s_m": round(float(A[a][sco_rows].mean()), 6),
                "curvature_mae_1pm": round(
                    pooled(E[a]["curv"], E[a]["curv_m"], sco_rows), 6)}
            for a in ("emitted", "anchor", "ha0", "ha0_ext", best)
        }
    R["HELD_OUT_LAMBDA"] = hold

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print(json.dumps(R["controls"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
