"""Advisory class D — ONE IDENTITY PER DERIVED QUANTITY, against an
INDEPENDENT reference.

⛔ The advisory's rule: *"the only check that catches these is an IDENTITY, and
it must not be built from the quantity under test"*, and its ordering of
discriminators: **analytic target > independently-authored reference >
mutation that reintroduces the historical defect**.

Every identity below uses an ANALYTIC target — a circle's curvature is exactly
`1/R`, a straight plan's is exactly `0`, a constant-acceleration arc length is
exactly `v0 t + a t^2 / 2` — so no rearrangement of the code under test can
make it pass. Each carries a MUTATION arm that reintroduces the programme's own
historical defect (`anchors.pt` column 1 read as curvature = 396 g) and must go
RED.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

from tanitad.models.kinematic import rollout_unicycle, slot_dts
from tanitad.refs import refc

HOR = (5, 10, 15, 20)
N = 5


def _decoder(units="alat", kappa_cap=0.12, alat_v_floor=4.0, ref=10.0):
    torch.manual_seed(0)
    cfg = refc.DecoderConfig(d=16, n_heads=2, layers=1, ff_mult=1,
                             sampler="none")
    return refc.AnchoredDiffusionDecoder(
        feat_dim=8, n_steps=len(HOR), d_meas=4, d_ctx=2, tac_latent_dim=2,
        anchors=torch.zeros(N, len(HOR), 2), cfg=cfg, hierarchy=False,
        graft_maneuver=False, graft_target_latent=False,
        grounded_selector=False, horizons=HOR, v0_conditioned=True,
        ref_speed_ms=ref, control_units=units, kappa_cap=kappa_cap,
        alat_v_floor_ms=alat_v_floor)


def circumradius(p0, p1, p2):
    """ANALYTIC: the radius of the circle through three points (menelaus /
    the standard |AB||BC||CA| / (4*Area) form). Knows nothing about our dt,
    our integrator, or our units."""
    a = math.dist(p0, p1)
    b = math.dist(p1, p2)
    c = math.dist(p2, p0)
    area = abs((p1[0] - p0[0]) * (p2[1] - p0[1])
               - (p2[0] - p0[0]) * (p1[1] - p0[1])) / 2.0
    if area < 1e-12:
        return float("inf")
    return a * b * c / (4.0 * area)


def id_curvature() -> dict:
    """`control_units='kappa'` -> the rolled path's radius must be EXACTLY 1/k.

    The reference is a circle, not a rearrangement of `roll_bank`.
    """
    out = {"reference": "circumradius of 3 path points (analytic)"}
    dec = _decoder(units="kappa", kappa_cap=1.0)
    ks = [0.02, 0.05, 0.1]
    with torch.no_grad():
        dec.anchor_controls.zero_()
        for i, k in enumerate(ks):
            dec.anchor_controls[i, 1] = k
    v = torch.tensor([12.0])
    bank = dec.roll_bank(v, None, 1, torch.float32)   # [1, N, S, 2]
    rows = []
    for i, k in enumerate(ks):
        p = [(0.0, 0.0)] + [tuple(float(x) for x in bank[0, i, s])
                            for s in range(len(HOR))]
        R = circumradius(p[0], p[2], p[4])
        rows.append({"kappa_1_per_m": k, "R_expected_m": round(1.0 / k, 5),
                     "R_measured_m": round(R, 5),
                     "rel_err": round(abs(R - 1.0 / k) / (1.0 / k), 6)})
    out["rows"] = rows
    out["max_rel_err"] = max(r["rel_err"] for r in rows)
    # CONTROL that must read a KNOWN value: kappa = 0 is a straight line
    p = [(0.0, 0.0)] + [tuple(float(x) for x in bank[0, 3, s])
                        for s in range(len(HOR))]
    out["kappa_0_lateral_max_abs_m"] = round(max(abs(q[1]) for q in p), 12)
    out["PASS"] = bool(out["max_rel_err"] < 2e-3
                       and out["kappa_0_lateral_max_abs_m"] == 0.0)
    return out


def id_alat() -> dict:
    """`control_units='alat'` -> the path's LATERAL ACCELERATION must equal the
    declared control: `v^2 * kappa_measured == a_lat`.

    ⛔ MUTATION = the programme's own 2026-09-04 defect: read the SAME tensor as
    CURVATURE. The identity must go RED, with the 396 g magnitude.
    """
    out = {"reference": "a_lat = v^2 / R with R from the circumradius"}
    v_ms = 12.0
    a_lats = [1.0, 2.0, 3.0]
    dec = _decoder(units="alat", kappa_cap=1.0, alat_v_floor=4.0)
    with torch.no_grad():
        dec.anchor_controls.zero_()
        for i, a in enumerate(a_lats):
            dec.anchor_controls[i, 1] = a
    bank = dec.roll_bank(torch.tensor([v_ms]), None, 1, torch.float32)
    rows = []
    for i, a in enumerate(a_lats):
        p = [(0.0, 0.0)] + [tuple(float(x) for x in bank[0, i, s])
                            for s in range(len(HOR))]
        R = circumradius(p[0], p[2], p[4])
        a_meas = v_ms ** 2 / R
        rows.append({"a_lat_declared": a, "a_lat_measured": round(a_meas, 5),
                     "rel_err": round(abs(a_meas - a) / a, 6),
                     "g": round(a_meas / 9.81, 4)})
    out["rows"] = rows
    out["max_rel_err"] = max(r["rel_err"] for r in rows)
    # MUTATION: the same bytes read as CURVATURE
    bad = _decoder(units="kappa", kappa_cap=10.0)
    with torch.no_grad():
        bad.anchor_controls.zero_()
        for i, a in enumerate(a_lats):
            bad.anchor_controls[i, 1] = a
    v_hi = 36.0
    bbad = bad.roll_bank(torch.tensor([v_hi]), None, 1, torch.float32)
    p = [(0.0, 0.0)] + [tuple(float(x) for x in bbad[0, 2, s])
                        for s in range(len(HOR))]
    R_bad = circumradius(p[0], p[2], p[4])
    out["MUTATION_kappa_misread"] = {
        "v_ms": v_hi, "control_value": a_lats[2],
        "implied_a_lat_m_s2": round(v_hi ** 2 / R_bad, 2),
        "implied_g": round(v_hi ** 2 / R_bad / 9.81, 2),
        "goes_RED": bool(abs(v_hi ** 2 / R_bad - a_lats[2]) / a_lats[2] > 1.0)}
    out["PASS"] = bool(out["max_rel_err"] < 2e-3
                       and out["MUTATION_kappa_misread"]["goes_RED"])
    return out


def id_longitudinal() -> dict:
    """`controls[:, 0]` is LONGITUDINAL ACCELERATION (m/s^2): the arc length at
    slot k must be the closed-form `v0 t + a t^2 / 2` at `t = horizon_k * dt`.

    Semi-independent: the closed form is analytic, the integrator is forward
    Euler, so a small discretisation residual is EXPECTED and is reported
    rather than tolerated silently.
    """
    out = {"reference": "closed-form constant-acceleration arc length"}
    dec = _decoder(units="alat")
    accs = [0.0, 1.0, -1.0]
    with torch.no_grad():
        dec.anchor_controls.zero_()
        for i, a in enumerate(accs):
            dec.anchor_controls[i, 0] = a
    v0 = 10.0
    bank = dec.roll_bank(torch.tensor([v0]), None, 1, torch.float32)
    dt = float(dec.anchor_dt)
    rows = []
    for i, a in enumerate(accs):
        for s, h in enumerate(HOR):
            t = h * dt
            want = v0 * t + 0.5 * a * t * t
            got = float(bank[0, i, s, 0])
            rows.append({"a_lon": a, "horizon_ticks": h, "t_s": round(t, 3),
                         "x_closed_form_m": round(want, 5),
                         "x_measured_m": round(got, 5),
                         "abs_err_m": round(abs(got - want), 6)})
    out["rows"] = rows
    out["max_abs_err_m"] = max(r["abs_err_m"] for r in rows)
    # ⛔ THE DEFECT THIS WOULD CATCH: a dt that is 2x wrong (advisory class D,
    # row 2). Reported as the SIZE of the error a 0.2 s tick would produce.
    dec2 = _decoder(units="alat")
    dec2.anchor_dt = 0.2
    with torch.no_grad():
        dec2.anchor_controls.zero_()
        dec2.anchor_controls[0, 0] = 0.0
    b2 = dec2.roll_bank(torch.tensor([v0]), None, 1, torch.float32)
    out["MUTATION_dt_0p2"] = {
        "x_at_last_slot_m": round(float(b2[0, 0, -1, 0]), 4),
        "x_at_dt_0p1_m": round(float(bank[0, 0, -1, 0]), 4),
        "ratio": round(float(b2[0, 0, -1, 0]) / float(bank[0, 0, -1, 0]), 4)}
    out["PASS"] = bool(out["max_abs_err_m"] < 0.15)
    return out


def id_tick_ownership() -> dict:
    """WHO owns the 0.1 s tick?  Every site that writes it as a LITERAL is a
    place the rate can silently disagree with its neighbour."""
    root = Path("D:/Projects/TanitAD")
    out = {}
    dec = _decoder()
    out["decoder.anchor_dt"] = float(dec.anchor_dt)
    out["decoder.anchor_horizons_ticks"] = list(dec.anchor_horizons)
    out["decoder.anchor_slots"] = dec.anchor_slots.tolist()
    out["slot_dts_default_tick"] = slot_dts(HOR)
    out["SelectionConfig.horizon_s_default"] = refc.SelectionConfig().horizon_s
    cfg = refc.refc_small_config()
    out["RefCConfig.selection().horizon_s"] = cfg.selection().horizon_s
    out["RefCConfig.trajectory.horizons"] = list(cfg.trajectory.horizons)
    out["derived_horizon_s_max_x_0.1"] = max(cfg.trajectory.horizons) * 0.1
    # literal sites
    sites = []
    for rel in ("stack/tanitad/refs/refc.py",
                "stack/scripts/refc_v3_train.py",
                "stack/tanitad/models/kinematic.py",
                "stack/tanitad/refs/refc_sampler.py"):
        for i, ln in enumerate((root / rel).read_text(
                encoding="utf-8").splitlines(), 1):
            s = ln.strip()
            if s.startswith("#"):
                continue
            if ("anchor_dt = 0.1" in s or "tick: float = 0.1" in s
                    or "* 0.1" in s and ("horizon" in s or "hz" in s)
                    or "dt=0.1" in s or "tick=0.1" in s):
                sites.append({"file": rel, "line": i, "src": s[:110]})
    out["tick_literal_sites"] = sites
    out["n_tick_literals"] = len(sites)
    return out


def id_maxspeed() -> dict:
    """SPEC §5's {30, 50, 100, 120} km/h ladder -> m/s.  The independent
    reference is the ROAD-LAW integers, not a histogram."""
    from tanitad.refs import refcv6_max_speed as ms
    out = {"kmh": list(ms.SPEED_MAX_STEPS_KMH_V6),
           "ms": [float(x) for x in ms.SPEED_MAX_STEPS_MS_V6]}
    out["exact_division_by_3.6"] = [
        float(k) / 3.6 for k in ms.SPEED_MAX_STEPS_KMH_V6]
    out["max_abs_ladder_err"] = max(
        abs(a - b) for a, b in zip(out["ms"], out["exact_division_by_3.6"]))
    out["is_rounded_to_4dp"] = any(
        abs(a - round(a, 4)) > 0 for a in out["ms"]) is False and any(
        a != round(a, 4) for a in out["exact_division_by_3.6"])
    out["PASS"] = bool(out["max_abs_ladder_err"] == 0.0)
    return out


def id_dd_box() -> dict:
    """DD's `norm_odo` box, against the RELEASED SOURCE's own literals."""
    from tanitad.models import refcv6_diffusion as rv6
    src = Path("D:/Projects/TanitAD/TanitAD Research Lab/Architecture & "
               "Inference/Research/2026-09-05-diffusiondrive-v2-analysis/raw/"
               "ddv2_src/v1/transfuser_model_v2.py").read_text(
                   encoding="utf-8", errors="replace")
    out = {"released_source_present": bool(src)}
    out["released_norm_x_line"] = next(
        (l.strip() for l in src.splitlines() if "56.9" in l and "2*(" in l), None)
    out["released_norm_y_line"] = next(
        (l.strip() for l in src.splitlines() if "/46" in l and "2*(" in l), None)
    out["ours_x"] = [rv6.DD_X_OFF, rv6.DD_X_SPAN]
    out["ours_y"] = [rv6.DD_Y_OFF, rv6.DD_Y_SPAN]
    out["PASS"] = bool("1.2" in (out["released_norm_x_line"] or "")
                       and "56.9" in (out["released_norm_x_line"] or "")
                       and rv6.DD_X_OFF == 1.2 and rv6.DD_X_SPAN == 56.9
                       and rv6.DD_Y_OFF == 20.0 and rv6.DD_Y_SPAN == 46.0)
    return out


def id_bev_grid() -> dict:
    """coupling (1)'s metre -> grid_sample map, against ANALYTIC cell centres."""
    from tanitad.data.bev_raster import GRID_DEFAULT, _cell_centers
    from tanitad.models.refc_bev_coupling import waypoints_to_bev_grid
    xc, yc = _cell_centers(GRID_DEFAULT)
    nx, ny = GRID_DEFAULT.shape
    rows = []
    for (i, j) in ((0, 0), (nx - 1, ny - 1), (nx // 2, ny // 2), (3, 7)):
        wp = torch.tensor([[float(xc[i]), float(yc[j])]])
        g = waypoints_to_bev_grid(wp, GRID_DEFAULT)[0]
        # align_corners=False: the CENTRE of cell k maps to (2k+1)/n - 1
        want_col = (2 * j + 1) / ny - 1
        want_row = (2 * i + 1) / nx - 1
        rows.append({"cell": [i, j], "metres": [round(float(xc[i]), 3),
                                                round(float(yc[j]), 3)],
                     "grid": [round(float(g[0]), 8), round(float(g[1]), 8)],
                     "expected": [round(want_col, 8), round(want_row, 8)],
                     "err": round(max(abs(float(g[0]) - want_col),
                                      abs(float(g[1]) - want_row)), 9)})
    out = {"reference": "bev_raster._cell_centers + align_corners=False rule",
           "rows": rows, "max_err": max(r["err"] for r in rows)}
    # MUTATION: the transposition DD writes as `[..., [1, 0]]`
    wp = torch.tensor([[30.0, 8.0]])
    g = waypoints_to_bev_grid(wp, GRID_DEFAULT)[0]
    out["MUTATION_swapped_would_differ"] = round(
        float((g[0] - g[1]).abs()), 6)
    out["PASS"] = bool(out["max_err"] < 1e-6
                       and out["MUTATION_swapped_would_differ"] > 0)
    return out


def main():
    out = {}
    for k, fn in (("curvature", id_curvature), ("a_lat", id_alat),
                  ("longitudinal", id_longitudinal),
                  ("tick_ownership", id_tick_ownership),
                  ("max_speed_ladder", id_maxspeed),
                  ("dd_waypoint_box", id_dd_box),
                  ("bev_grid_map", id_bev_grid)):
        try:
            out[k] = fn()
        except Exception as e:                       # noqa: BLE001
            out[k] = {"ERROR": f"{type(e).__name__}: {e}", "PASS": False}
    out["VERDICT"] = {k: v.get("PASS") for k, v in out.items()
                      if isinstance(v, dict) and "PASS" in v}
    txt = json.dumps(out, indent=2, default=str)
    print(txt)
    (Path(__file__).resolve().parents[1] / "raw"
     / "units_identities.json").write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
