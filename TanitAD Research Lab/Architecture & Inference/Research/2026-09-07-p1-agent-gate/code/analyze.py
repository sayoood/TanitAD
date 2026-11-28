"""P1 agent-conditioning gate -- the four-family panel + the pre-registered bar.

⛔ Every estimator is the programme's own (`taniteval.four_families`,
`taniteval.ci.paired_episode_cluster_bootstrap`, `taniteval.lead_metrics`).
⛔ `overlapping_holdout_se` is NOT used anywhere -- it biases the point estimate
bidirectionally, up to a sign flip.

Controls, each of which MUST read its known value or the panel is void:
  * `const`    -- a plan that never moves (pred = 0): the no-information value.
  * `straight` -- a plan that never steers (pred = v0 * t along +x): the
                  STRAIGHT-LINE FLOOR that must be printed beside curvature.
  * `n` and `d` printed on every interval.
"""
from __future__ import annotations
import argparse, json, lzma, math, os, sys
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, r"C:/Users/Admin/tanitad-p1gate/stack")
sys.path.insert(0, r"C:/Users/Admin/tanitad-p1gate/taniteval")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from taniteval import four_families as ff                       # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as pb  # noqa: E402
from tanitad.data.v2_dataset import stable_episode_id           # noqa: E402

JOIN = r"C:/Users/Admin/tanitad-caches/b1-agent-join-20260906/b1eval_agents.jsonl.xz"
LEAD_LAT_M = 2.0


# ------------------------------------------------------------------ join ---- #
def load_join_rows(path: str, want_eids: set[int]) -> dict:
    """-> {(eid, frame): [ {cx,cy,l,track_id}, ... ] } for the wanted episodes."""
    out: dict = defaultdict(list)
    with lzma.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            eid = stable_episode_id(d["clip_id"])
            if eid not in want_eids:
                continue
            out[(eid, int(d["frame"]))] = d["agents"]
    return out


def lead_block(z, rows: dict, slots: list[int], dt: float) -> dict:
    """The GT lead track in each window's NOW-ego frame, at the arm's horizons.

    ⛔ ARM-INDEPENDENT BY CONSTRUCTION: it is built from the JOIN and the GT ego
    poses only, so all three arms are scored against the SAME lead track and the
    paired bootstrap is genuinely paired.
    """
    hz = [int(h) for h in z["horizons"]]
    hz = [hz[i] for i in slots]        # UNIFORM prefix only -- see uniform_prefix
    pl, fe = z["pose_last"], z["fut_ext"]
    eid, nf = z["eid"], z["now_frame"]
    W, K = pl.shape[0], len(hz)
    leads = np.full((W, K, 2), np.nan)
    lens = np.full(W, np.nan)
    n_lead = 0
    for i in range(W):
        e, f0 = int(eid[i]), int(nf[i])
        ag0 = rows.get((e, f0))
        if not ag0:
            continue
        # the lead AT t0: nearest agent ahead inside the corridor
        cand = [a for a in ag0
                if float(a["cx"]) > 0 and abs(float(a["cy"])) < LEAD_LAT_M]
        if not cand:
            continue
        lead = min(cand, key=lambda a: float(a["cx"]))
        tid = str(lead.get("track_id"))
        lens[i] = float(lead.get("l", np.nan))
        n_lead += 1
        x0, y0, th0 = float(pl[i, 0]), float(pl[i, 1]), float(pl[i, 2])
        c0, s0 = math.cos(th0), math.sin(th0)
        for j, h in enumerate(hz):
            ag = rows.get((e, f0 + h))
            if not ag:
                continue
            m = [a for a in ag if str(a.get("track_id")) == tid]
            if not m:
                continue
            # pose of the ego at frame f0+h (fut_ext[h-1] == frame NOW+h)
            if h - 1 >= fe.shape[1]:
                continue
            xf, yf, thf = (float(fe[i, h - 1, 0]), float(fe[i, h - 1, 1]),
                           float(fe[i, h - 1, 2]))
            cf, sf = math.cos(thf), math.sin(thf)
            ax, ay = float(m[0]["cx"]), float(m[0]["cy"])
            wx = xf + cf * ax - sf * ay                  # agent -> world
            wy = yf + sf * ax + cf * ay
            dx, dy = wx - x0, wy - y0                    # world -> NOW ego
            leads[i, j, 0] = c0 * dx + s0 * dy
            leads[i, j, 1] = -s0 * dx + c0 * dy
    return {"leads": leads, "lead_lens": lens, "speeds": z["v0"].astype(float),
            "eid": [int(x) for x in eid], "path_steps": list(slots),
            "dt_s": float(dt),
            "_n_windows_with_lead": int(n_lead)}


# --------------------------------------------------------------- controls -- #
def control_pred(kind: str, z) -> np.ndarray:
    hz = np.asarray(z["horizons"], dtype=float) * 0.1     # seconds
    W = z["pred"].shape[0]
    p = np.zeros((W, len(hz), 2), dtype=np.float64)
    if kind == "const":
        return p                                          # never moves
    if kind == "straight":
        p[:, :, 0] = z["v0"][:, None] * hz[None, :]       # never steers
        return p
    raise ValueError(kind)


# ------------------------------------------------------------ per-window --- #
def uniform_prefix(hz: list[int]) -> tuple[list[int], float]:
    """The leading slots that lie on a UNIFORM time grid, and their dt.

    ⛔ THIS IS NOT COSMETIC. The v3 horizon set is [5,10,15,20,30,40,50,60] --
    0.5 s spacing for four slots, then 1.0 s. `_seq_geometry` divides by ONE dt,
    so scoring all eight with dt=0.5 inflates every RATE (speed, yaw-rate,
    accel) by 2x on the tail, and with dt=1.0 halves them on the head. Curvature
    and heading are dt-invariant and survive either way; speed does not.
    ⇒ every RATE metric here is read on the uniform prefix only, and the prefix
    is printed with the number. Positional metrics (ADE, cross-track, along) are
    dt-invariant and use ALL slots.
    """
    d0 = hz[1] - hz[0]
    k = 1
    while k + 1 < len(hz) and hz[k + 1] - hz[k] == d0:
        k += 1
    return list(range(k + 1)), float(d0) * 0.1


def per_window(pred: np.ndarray, gt: np.ndarray, dt: float,
               rate_slots: list[int] | None = None) -> tuple[dict, dict]:
    """-> (per-window scalars, per-metric window mask).

    ⛔ THE MASK IS DERIVED FROM THE GT ONLY, so it is ARM-INDEPENDENT: every arm
    is scored on the same windows and the paired bootstrap stays paired. A
    curvature statistic computed over a subset that shifts with the arm is not
    comparable across arms — `_seq_geometry`'s own docstring says the count must
    travel with the number, so the count travels here.
    """
    rs = rate_slots if rate_slots is not None else list(range(pred.shape[1]))
    P = ff._seq_geometry(torch.tensor(pred[:, rs], dtype=torch.float64), dt)
    G = ff._seq_geometry(torch.tensor(gt[:, rs], dtype=torch.float64), dt)
    ok1 = G["valid"].numpy()                       # [W, H]  step-level
    ok2 = G["pair_valid"].numpy()                  # [W, H-1] pair-level
    d, m = {}, {}
    full = np.ones(pred.shape[0], dtype=bool)
    d["ade_m"] = np.linalg.norm(pred - gt, axis=-1).mean(axis=1); m["ade_m"] = full
    d["cross_mae_m"] = np.abs(pred[:, :, 1] - gt[:, :, 1]).mean(axis=1)
    m["cross_mae_m"] = full
    d["along_mae_m"] = np.abs(pred[:, :, 0] - gt[:, :, 0]).mean(axis=1)
    m["along_mae_m"] = full

    def masked(err, ok, name):
        cnt = ok.sum(axis=1)
        val = np.where(ok, np.abs(err), 0.0).sum(axis=1) / np.maximum(cnt, 1)
        d[name] = val
        m[name] = cnt > 0

    masked((P["speed"] - G["speed"]).numpy(), ok1, "speed_mae_mps")
    masked((P["heading"] - G["heading"]).numpy(), ok1, "heading_mae")
    masked((P["curvature"] - G["curvature"]).numpy(), ok2, "curvature_mae")
    masked((P["yaw_rate"] - G["yaw_rate"]).numpy(), ok2, "yaw_rate_mae")
    return d, m


def fam_block(z, pred, lead, dt, tag):
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    lon = ff.longitudinal(t(pred), t(z["gt"]), dt=dt, lead=lead,
                          eid=[int(x) for x in z["eid"]])
    lat = ff.lateral(t(pred), t(z["gt"]), dt=dt,
                     eid=[int(x) for x in z["eid"]])
    return {"arm": tag, "longitudinal": lon, "lateral": lat}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True)   # name=path
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    Z = {}
    for spec in a.dumps:
        name, path = spec.split("=", 1)
        Z[name] = dict(np.load(path, allow_pickle=False))
    ref = Z[list(Z)[0]]
    hz = [int(h) for h in ref["horizons"]]
    rate_slots, dt = uniform_prefix(hz)
    eid = [int(x) for x in ref["eid"]]
    n_win, n_ep = len(eid), len(set(eid))
    print(f"[panel] n_windows={n_win} n_episodes={n_ep} horizons={hz}")
    print(f"[panel] RATE metrics on the uniform prefix slots {rate_slots} "
          f"(horizons {[hz[i] for i in rate_slots]}, dt={dt}s); positional "
          f"metrics on all {len(hz)} slots")

    # every arm must be on the SAME windows or the pairing is a lie
    for k, z in Z.items():
        assert (z["eid"] == ref["eid"]).all(), f"{k}: window order differs"
        assert (z["gt"] == ref["gt"]).all(), f"{k}: GT differs"

    # ⭐ FRAME-CHAIN SELF-TEST. The lead track is placed into the window's NOW
    # ego frame by MY OWN transform. If that transform is wrong the whole
    # distance-keeping family is wrong and every number still looks plausible.
    # So the SAME transform is applied to the GT EGO POSES and compared against
    # `refb_labels.waypoint_targets`, which the trainer itself used to build the
    # `gt` in the dump. Agreement is a positive control on the chain; the
    # deliberate NEGATIVE control (heading forced to 0) must NOT agree.
    def ego_from_world(pl, fe, hz_sel, theta_scale=1.0):
        x0, y0 = pl[:, 0], pl[:, 1]
        th0 = pl[:, 2] * theta_scale
        c0, s0 = np.cos(th0), np.sin(th0)
        o = np.zeros((pl.shape[0], len(hz_sel), 2))
        for j, h in enumerate(hz_sel):
            dx = fe[:, h - 1, 0] - x0
            dy = fe[:, h - 1, 1] - y0
            o[:, j, 0] = c0 * dx + s0 * dy
            o[:, j, 1] = -s0 * dx + c0 * dy
        return o
    _hzall = [int(h) for h in ref["horizons"]]
    _mine = ego_from_world(ref["pose_last"].astype(float),
                           ref["fut_ext"].astype(float), _hzall)
    _err = float(np.abs(_mine - ref["gt"].astype(float)).max())
    _neg = ego_from_world(ref["pose_last"].astype(float),
                          ref["fut_ext"].astype(float), _hzall, theta_scale=0.0)
    _negerr = float(np.abs(_neg - ref["gt"].astype(float)).max())
    print(f"[control] frame chain: max|mine - trainer_gt| = {_err:.6f} m "
          f"(must be ~0); heading-zeroed NEGATIVE control = {_negerr:.4f} m "
          f"(must be LARGE)")
    out_frame_ctl = {"max_abs_err_m": _err, "negative_control_err_m": _negerr,
                     "passes": bool(_err < 1e-3 and _negerr > 1.0)}

    rows = load_join_rows(JOIN, set(eid))
    lead = lead_block(ref, rows, rate_slots, dt)
    print(f"[panel] lead track: {lead['_n_windows_with_lead']} of {n_win} "
          f"windows have a lead inside |lat|<{LEAD_LAT_M} m ahead at t0")

    PW, FAM, MASK = {}, {}, {}
    for k, z in Z.items():
        PW[k], MASK[k] = per_window(z["pred"].astype(float),
                                    z["gt"].astype(float), dt, rate_slots)
        FAM[k] = fam_block(z, z["pred"].astype(float), lead, dt, k)
    for c in ("const", "straight"):
        p = control_pred(c, ref)
        PW[c], MASK[c] = per_window(p, ref["gt"].astype(float), dt, rate_slots)
        FAM[c] = fam_block(ref, p, lead, dt, c)
    # the masks are GT-derived, so they MUST be identical across arms --
    # a same-breath control on the pairing itself
    _m0 = MASK[list(Z)[0]]
    for k in MASK:
        for mk in _m0:
            assert (MASK[k][mk] == _m0[mk]).all(), (
                f"mask for {mk} differs on arm {k} -- the pairing is broken")
    out_masks = {mk: int(_m0[mk].sum()) for mk in _m0}
    print("[panel] windows entering each metric:", out_masks)

    # ---- distance-keeping per-window, per arm (the HEADLINE family) -------- #
    from taniteval.lead_metrics import distance_keeping
    DK = {}
    for k in list(Z) + ["const", "straight"]:
        p = (Z[k]["pred"].astype(float) if k in Z else control_pred(k, ref))
        DK[k] = distance_keeping(p[:, rate_slots], lead["leads"],
                                 lead["lead_lens"], lead["speeds"],
                                 lead["dt_s"])
    have = np.isfinite(DK[list(Z)[0]]["headway_min_m"])
    for k in DK:
        have &= np.isfinite(DK[k]["headway_min_m"])
    print(f"[panel] distance-keeping n (all arms have a lead) = {int(have.sum())}")

    # ---- the contrasts ---------------------------------------------------- #
    eid_a = np.asarray(eid)
    out = {"n_windows": n_win, "n_episodes": n_ep, "horizons": hz,
           "rate_slots": rate_slots, "dt_s": dt,
           "d_params_tiny": 16989725,
           "n_vs_d_note": ("n = windows scored, d = trainable params. n << d is "
                           "UNDERPOWERED BY CONSTRUCTION, not a negative result."),
           "estimator": "paired_episode_cluster_bootstrap (taniteval.ci)",
           "not_used": "overlapping_holdout_se -- biases the point estimate",
           "tier": "T1 self-action open loop (planner rolls its own plan from "
                   "measured state at t0). NOT closed loop.",
           "lead_windows": lead["_n_windows_with_lead"],
           "families": FAM, "contrasts": {}, "dk": {}}

    def contrast(name, av, bv, e, lower_is_better=True, sel=None):
        av, bv, e = np.asarray(av, float), np.asarray(bv, float), np.asarray(e)
        if sel is not None:
            av, bv, e = av[sel], bv[sel], e[sel]
        if av.size < 5:
            out["contrasts"][name] = {"status": "UNAVAILABLE", "n": int(av.size),
                                      "reason": "fewer than 5 admissible windows"}
            print(f"  {name:38s} UNAVAILABLE n={av.size}")
            return None
        r = pb(av, bv, [int(x) for x in e], n_boot=10000, seed=7)
        r["favourable"] = ((r["delta"] < 0) if lower_is_better
                           else (r["delta"] > 0))
        r["verdict"] = ("SEPARATED " + ("BETTER" if r["favourable"] else "WORSE")
                        if r["separated"] else "not separated")
        out["contrasts"][name] = r
        print(f"  {name:38s} {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
              f"n={r['n_windows']}/{r['n_episodes']}  {r['verdict']}")
        return r

    pairs = [("head", "off"), ("head", "shuf"), ("shuf", "off")]
    print("\n[LONGITUDINAL / LATERAL / ADE contrasts]  (delta = A - B, lower better)")
    for A, B in pairs:
        if A not in PW or B not in PW:
            continue
        for m in ("ade_m", "speed_mae_mps", "along_mae_m", "cross_mae_m",
                  "heading_mae", "curvature_mae", "yaw_rate_mae"):
            if m in PW[A] and m in PW[B]:
                contrast(f"{A}-{B}:{m}", PW[A][m], PW[B][m], eid_a,
                         sel=_m0[m])

    print("\n[DISTANCE-KEEPING contrasts]  (headway/time-gap: higher = safer; "
          "TTC: higher = safer)")
    for A, B in pairs:
        if A not in DK or B not in DK:
            continue
        for m, lower in (("headway_min_m", False), ("time_gap_min_s", False),
                         ("min_ttc_s", False)):
            x, y = DK[A][m], DK[B][m]
            sel = have & np.isfinite(x) & np.isfinite(y)
            if sel.sum() < 5:
                out["dk"][f"{A}-{B}:{m}"] = {
                    "status": "UNAVAILABLE", "n": int(sel.sum()),
                    "reason": "fewer than 5 windows have a lead in both arms"}
                continue
            r = pb(x[sel], y[sel], eid_a[sel].tolist(), n_boot=10000, seed=7)
            r["favourable"] = (r["delta"] > 0) if not lower else (r["delta"] < 0)
            r["verdict"] = ("SEPARATED " +
                            ("BETTER" if r["favourable"] else "WORSE")
                            if r["separated"] else "not separated")
            out["dk"][f"{A}-{B}:{m}"] = r
            print(f"  {A}-{B}:{m:20s} {r['delta']:+.4f} "
                  f"[{r['lo']:+.4f}, {r['hi']:+.4f}] "
                  f"n={r['n_windows']}/{r['n_episodes']}  {r['verdict']}")

    # ---- controls at their known values ----------------------------------- #
    print("\n[CONTROLS -- each must read its known value]")
    ctl = {}
    for k in list(Z) + ["const", "straight"]:
        ctl[k] = {m: float(np.nanmean(PW[k][m][_m0[m]])) for m in PW[k]}
        ctl[k]["dk_n"] = int(np.isfinite(DK[k]["headway_min_m"]).sum())
        ctl[k]["dk_mean_headway_m"] = (
            float(np.nanmean(DK[k]["headway_min_m"]))
            if np.isfinite(DK[k]["headway_min_m"]).any() else float("nan"))
    out["controls"] = ctl
    out["windows_per_metric"] = out_masks
    for k, v in ctl.items():
        print(f"  {k:9s} ade={v['ade_m']:.4f} speed_mae={v['speed_mae_mps']:.4f} "
              f"curv_mae={v.get('curvature_mae', float('nan')):.5f} "
              f"cross={v['cross_mae_m']:.4f} dk_n={v['dk_n']}")
    out["control_checks"] = {
        "frame_chain": out_frame_ctl,
        "const_is_no_information": {
            "expect": "pred == 0 exactly; ADE == mean |gt|",
            "ade_m": ctl["const"]["ade_m"],
            "gt_mean_norm": float(np.linalg.norm(ref["gt"], axis=-1).mean()),
            "tol": 1e-4,
            "tol_note": ("the dump is float32 and the check recomputes in "
                         "float64, so the two agree to ~1e-6 relative, not to "
                         "float64 equality"),
            "passes": bool(abs(ctl["const"]["ade_m"] -
                               float(np.linalg.norm(ref["gt"],
                                                    axis=-1).mean())) < 1e-4)},
        "straight_line_curvature_floor": {
            "note": "⛔ QUOTE THIS BESIDE EVERY CURVATURE NUMBER",
            "curvature_mae": ctl["straight"].get("curvature_mae"),
            "arms": {k: ctl[k].get("curvature_mae") for k in Z}},
    }
    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items() if not k.startswith("_")}
        if isinstance(o, (list, tuple)):
            return [_san(v) for v in o]
        if isinstance(o, np.ndarray):
            return (float(o) if o.ndim == 0 else
                    {"_array_summary": {"n": int(o.size),
                                        "mean": float(np.nanmean(o)) if o.size else None,
                                        "n_finite": int(np.isfinite(o).sum())}})
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (bool, int, float, str)) or o is None:
            return o
        return str(o)
    json.dump(_san(out), open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"\n[panel] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
