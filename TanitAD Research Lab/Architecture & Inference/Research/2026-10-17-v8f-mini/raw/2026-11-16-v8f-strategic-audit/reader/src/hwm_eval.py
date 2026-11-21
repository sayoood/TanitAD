"""HWM-PLAN T1-kin evaluation: per-window metrics per band and family, plan units, and the paired episode-cluster report (taniteval/ci.py).
T1-kin: the plan is the model's OWN action sequence, integrated by the kinematic model (friction + speed bounds) and scored against the logged actions integrated by the same model. No logged action after a0 reaches
a planner. HWM-ORACLE / FLAT-ORACLE read a true-future goal: tier T0, never a driving number. Families are reported separately, never pooled; TACTICAL is PLAN-DERIVED (the planned and the logged 6 s trajectories classified by one rule, the programme's own factored manoeuvre labeller, geometry only);
LONGITUDINAL headway / time gap / TTC come from taniteval.lead_metrics over the b1 eval lead block (2 s horizon: the 0-2 s band only); STRATEGIC (PI 2026-11-16, the pause ended) reads the plan against the NVIDIA navigation text
(strategic_cells, strategic_text): S_ADH, its signed-degrees twin and S_FALSE on the full 6 s band only, a T0 HWM-ORACLE diagnostic on the oracle arms."""
import json, os, sys, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import hwm_plan as H
import strategic_cells as S
import strategic_text as ST

BANDS = {"0-2s": (0, 10), "2-6s": (10, 30), "full6s": (0, 30)}           # step ranges [lo, hi); states lo+1 .. hi
FAMILIES = {"LONGITUDINAL": ("acc_mae", "speed_err", "hw_err", "tg_err", "ttc_err"), "LATERAL": ("yaw_mae", "head_err", "curv_err", "xtrack"),
            "TACTICAL": ("tac_man_dis", "tac_lat_dis", "tac_lon_dis"), "STRATEGIC": ("s_adh", "s_adh_deg", "s_false")}
BESIDE = ("ade",)                                  # LATERAL only
SIGNED = ("hw_d", "tg_d", "ttc_d")                 # LONGITUDINAL, plan minus logged, arm rows only (direction, not a lower-is-better score)
LEAD_METRICS = ("hw_err", "tg_err", "ttc_err") + SIGNED
CLS_KEYS = ("_lat_p", "_lon_p", "_man_p", "_lat_l", "_lon_l", "_man_l")
LOWER_IS_BETTER = True
STRAT_BAND = "full6s"                              # the only band a STRATEGIC cell is filled in; the other two are empty by construction
DIRECTION = {"s_adh": "higher", "s_adh_deg": "higher", "s_false": "neutral"}   # every other metric: lower is better. S_FALSE is read as plan minus logged, never as better or worse
STRAT_LOGGED = {m: "_" + m + "_l" for m in FAMILIES["STRATEGIC"]}             # the logged-action path read by the same cell: S_ADH ceiling, S_FALSE logged-path rate
DEFAULT_TABLE = "F:/TanitAD-scratch/datascale/navtok/navtok_table.npz"          # navtok_table.npz, sha256 prefix 2836369e (strategic_text.TABLE_SHA_PREFIX)
V_CURV = 1.0                                       # curvature compared only where the logged speed >= 1 m/s
NOT_COMPUTED = {"LATERAL.cross_track_vs_lane": "n = 0: no lane geometry; xtrack here is the lateral offset of the plan end from the logged-action trajectory end",
                "TACTICAL.goal_setting": "n = 0: the planner exposes no goal or anchor selection; TACTICAL here is the manoeuvre implied by the planned trajectory, not a vocabulary output"}
NOT_COMPUTED.update(S.STRATEGIC_NOT_COMPUTED)
HEADWAY_SCOPE = ("headway / time gap / TTC use taniteval.lead_metrics.distance_keeping over the b1 eval lead blocks (taniteval/tools/build_lead_block_b1.py): the 0-2 s band reads the banked K = 10 block (2 s lead track); the 2-6 s and "
                 "full 6 s bands read the K = 30 block hwm_plan/data/b1_eval_lead_block_k30.npz (same sources, sha256 equal, 6 s lead track). A window needs a lead track and labelled span over the whole of the block's horizon, so the K = 30 block has fewer LEAD rows than the K = 10 block. "
                 "In a band that starts after NOW the time-gap denominator is each path's own speed at the band start")
TACTICAL_SCOPE = ("plan-derived decision, NOT a vocabulary output: the planned and the logged trajectory (both the kinematic integration of the action sequence from the same v0) are classified by ONE rule, "
                  "tanitad.refs.refc_tactical.factor_from_kinematics through taniteval.four_families.maneuver_kinematics, on the path of each band re-expressed in the frame of the pose at the band start. The rule reads path geometry only "
                  "(net path tangent, chord speeds); the goal that conditioned the plan is not an input of the rule. Thresholds are the programme's 2 s thresholds applied unchanged to each band (not rescaled to 4 s or 6 s)")
_WRAP = lambda a: (a + np.pi) % (2 * np.pi) - np.pi


def lead_band(xp, xl, vp0, vl0, lead, lo, hi):
    """Headway family on one band: distance_keeping (taniteval.lead_metrics, reused) over the plan path and over the logged-action path against the SAME lead track, both in the window-origin frame, knots 1..hi.
    Defined for a band inside the lead track (hi <= K); otherwise every metric is NaN (n = 0). vp0 and vl0 are the speeds of the plan and the logged path at the band start, the time-gap denominators. A window enters only in state LEAD; a window where the plan or the logged path never has the
    lead inside the corridor ahead is NaN for the error (counted by the report as plan-lost / logged-lost, never dropped silently). Errors are |plan - logged|; the signed value is plan minus logged."""
    import taniteval.lead_metrics as LM
    n = len(vp0)
    nan = np.full(n, np.nan)
    out = {k: nan.copy() for k in LEAD_METRICS}
    out.update(_hw_p=nan.copy(), _hw_l=nan.copy(), _ttc_p=nan.copy(), _ttc_l=nan.copy(), _tg_p=nan.copy(), _tg_l=nan.copy())
    if lead is None or hi > lead["leads"].shape[1]:
        return out
    lens = np.where(np.asarray(lead["state"]) == "LEAD", lead["lens"], np.nan)
    lk = lead["leads"][:, lo:hi]
    dp = LM.distance_keeping(xp[:, lo + 1:hi + 1], lk, lens, vp0, lead["dt"])
    dl = LM.distance_keeping(xl[:, lo + 1:hi + 1], lk, lens, vl0, lead["dt"])
    for k, (p, l, t) in dict(hw=("headway_min_m", "headway_min_m", "hw"), tg=("time_gap_min_s", "time_gap_min_s", "tg"), ttc=("min_ttc_s", "min_ttc_s", "ttc")).items():
        a, b = np.asarray(dp[p], float), np.asarray(dl[l], float)
        out[f"_{k}_p"], out[f"_{k}_l"] = a, b
        out[f"{k}_err"], out[f"{k}_d"] = np.abs(a - b), a - b
    return out


def tac_band(xp, pp, xl, pl, lo, hi):
    """Plan-derived manoeuvre classes of the plan and of the logged-action trajectory by ONE rule: the band's path (knots lo+1..hi) re-expressed in the frame of the pose at knot lo, through
    four_families.maneuver_kinematics and refc_tactical.factor_from_kinematics (the programme's factored labeller: lateral lane_keep / turn_left / turn_right, longitudinal brake_stop / steady / accelerate,
    collapsed to the 5-way by its COLLAPSE_TABLE). Geometry only. Returns the class ids ('_lat_p', ... ) and per-window disagreement indicators (0.0 agree, 1.0 differ)."""
    import torch
    import taniteval.four_families as FF
    from tanitad.refs.refc_tactical import COLLAPSE_TABLE, factor_from_kinematics

    def classes(x, psi):
        d = x[:, lo + 1:hi + 1] - x[:, lo:lo + 1]
        c, s = np.cos(psi[:, lo])[:, None], np.sin(psi[:, lo])[:, None]
        rel = np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)
        dyaw, dv, v0_, v1_, _ = FF.maneuver_kinematics(torch.from_numpy(rel).float(), H.DT)
        lat, lon = factor_from_kinematics(dyaw, dv, v0_, v1_)
        man = torch.as_tensor(COLLAPSE_TABLE, dtype=torch.long)[lat, lon]
        return lat.numpy(), lon.numpy(), man.numpy()

    lp, np_, mp = classes(xp, pp)
    ll, nl, ml = classes(xl, pl)
    return {"_lat_p": lp, "_lon_p": np_, "_man_p": mp, "_lat_l": ll, "_lon_l": nl, "_man_l": ml,
            "tac_lat_dis": (lp != ll).astype(float), "tac_lon_dis": (np_ != nl).astype(float), "tac_man_dis": (mp != ml).astype(float)}


def strategic_cells_of(psi, xy, text):
    """The STRATEGIC cells of ONE path (plan or logged-action path from NOW): s_adh, s_adh_deg, s_false [n]. Without a text row every cell is NaN (n = 0), never 0."""
    n = psi.shape[0]
    if text is None:
        return {m: np.full(n, np.nan) for m in FAMILIES["STRATEGIC"]}
    c = S.cells(psi, xy, text["side"], text["theta_deg"], text["d_text"], text["text_ok"])
    return {"s_adh": c["s_adh"], "s_adh_deg": c["s_adh_signed_deg"], "s_false": c["s_false"]}


def attach_text(ws, tab=None):
    """Attach the navigation-text row of every window's NOW (strategic_text.attach, offset ST.NOW_OFFSET) as ws.text. The table is ws.nav_table, else tab, else the default file (sha256-checked)."""
    tab = getattr(ws, "nav_table", None) or tab or ST.load_table(DEFAULT_TABLE)
    ws.text = ST.attach(tab, ws.eid, ws.now)
    return ws.text


def strategic_band(sp, sl, band):
    """STRATEGIC metrics of one band: the plan's cells and, beside them, the logged path's cells (keys '_<m>_l'). Only the full 6 s band is filled; the other bands are NaN (n = 0) by construction."""
    n = len(sp["s_adh"])
    out = {}
    for m in FAMILIES["STRATEGIC"]:
        out[m] = sp[m] if band == STRAT_BAND else np.full(n, np.nan)
        out[STRAT_LOGGED[m]] = sl[m] if band == STRAT_BAND else np.full(n, np.nan)
    return out


def band_metrics(plan, A_raw, v0, lead=None, lead6=None, text=None):
    """plan (n, 30, 2), A_raw (n, >=30, 2) raw (a, yaw rate), v0 (n,), lead (optional) the dict of hwm_data.lead_for on the K = 10 block, lead6 (optional) the same on the K = 30 block, used by a band past 2 s. Returns {band: {metric: (n,)}} and kinematic-bound rates for plan and logged.
    Without a lead block the headway metrics are NaN (n = 0), never zero. text (optional) is strategic_text.attach's dict for the windows; without it every STRATEGIC cell is NaN (n = 0), never zero."""
    L = np.asarray(A_raw[:, :plan.shape[1]], np.float64)
    vp, pp, xp, ap = H.integrate(v0, plan)
    vl, pl, xl, al = H.integrate(v0, L)
    out = {}
    sp, sl = strategic_cells_of(pp, xp, text), strategic_cells_of(pl, xl, text)
    for b, (lo, hi) in BANDS.items():
        s = slice(lo, hi); e = hi                                         # steps lo..hi-1, band end state hi
        km = (vl[:, lo:hi] >= V_CURV)
        cp = plan[:, s, 1] / np.maximum(vp[:, lo:hi], V_CURV); cl = L[:, s, 1] / np.maximum(vl[:, lo:hi], V_CURV)
        ce = np.where(km, np.abs(cp - cl), np.nan)
        curv = np.full(len(v0), np.nan)
        has = km.any(1)
        curv[has] = np.nanmean(ce[has], 1)
        nrm = np.stack([-np.sin(pl[:, e]), np.cos(pl[:, e])], 1)
        out[b] = dict(acc_mae=np.abs(plan[:, s, 0] - L[:, s, 0]).mean(1), yaw_mae=np.abs(plan[:, s, 1] - L[:, s, 1]).mean(1), speed_err=np.abs(vp[:, e] - vl[:, e]),
                      head_err=np.abs(_WRAP(pp[:, e] - pl[:, e])), curv_err=curv, xtrack=np.abs(((xp[:, e] - xl[:, e]) * nrm).sum(1)),
                      ade=np.linalg.norm(xp[:, lo + 1:hi + 1] - xl[:, lo + 1:hi + 1], axis=-1).mean(1),
                      **lead_band(xp, xl, (np.asarray(v0, np.float64) if lo == 0 else vp[:, lo]), (np.asarray(v0, np.float64) if lo == 0 else vl[:, lo]), (lead6 if (hi > 10 and lead6 is not None) else lead), lo, hi), **tac_band(xp, pp, xl, pl, lo, hi),
                      **strategic_band(sp, sl, b))
    return out, dict(plan_bound_frac=float(ap), logged_bound_frac=float(al))


ARMS = {  # name: (mode, goal kind, tier)
    "hwm_oracle": ("hier", "oracle", "T0"), "flat_oracle": ("flat", "oracle", "T0"),
    "hwm_plan": ("hier", "plan", "T1-kin"), "flat_plan": ("flat", "plan", "T1-kin")}
REFS = ("hold", "prior")


def run_arm(planner, ws, arm, goal_fn, seed=0, idx=None):
    mode = ARMS[arm][0]
    t = time.time()
    r = planner.plan_windows(mode, goal_fn, ws.z[:, 0], ws.A[:, 0], idx=idx, seed=seed)
    r["wall_s"] = np.array(time.time() - t)
    return r


def run_refs(planner, ws, n_traj=H.N_TRAJ, idx=None):
    ii = np.arange(ws.n) if idx is None else np.asarray(idx)
    return {"hold": H.hold_plan(ws.A[ii, 0], n_traj), "prior": H.prior_plan(planner.wm, planner.st, len(ii), n_traj)}


def save_unit(outdir, tag, arm, seed, r):
    os.makedirs(outdir, exist_ok=True)
    np.savez(f"{outdir}/{tag}_{arm}_s{seed}.npz", **r)


# ---------------------------------------------------------------------------------------------- the report
def _ci():
    import taniteval.ci as CI
    return CI


def contrast(a, b, eid, n_boot, direction="lower"):
    """Paired episode-cluster bootstrap of mean(a) - mean(b) over the windows where both are finite. direction 'lower' (default): lower is better, 'better' = interval wholly below 0; 'higher': higher is better
    (S_ADH); 'neutral': no better or worse, the label reads 'higher' / 'lower' for the side of the interval (S_FALSE, plan minus logged)."""
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return dict(delta=float("nan"), lo=float("nan"), hi=float("nan"), label="n/a", n_windows=0, n_episodes=0, estimator="paired_episode_cluster_bootstrap", reason="n = 0: no window finite in both rows")
    r = _ci().paired_episode_cluster_bootstrap(a[ok], b[ok], np.asarray(eid)[ok], n_boot=n_boot, seed=0)
    if direction == "neutral":
        lab = "lower" if r["hi"] < 0 else ("higher" if r["lo"] > 0 else "none")
    elif direction == "higher":
        lab = "better" if r["lo"] > 0 else ("worse" if r["hi"] < 0 else "none")
    else:
        lab = "better" if r["hi"] < 0 else ("worse" if r["lo"] > 0 else "none")
    return dict(delta=r["delta"], lo=r["lo"], hi=r["hi"], label=lab, n_windows=r["n_windows"], n_episodes=r["n_episodes"], estimator=r["estimator"])


def single(a, eid, n_boot):
    ok = np.isfinite(a)
    if not ok.any():
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n_windows=0, estimator="episode_cluster_bootstrap", reason="n = 0: no finite window")
    r = _ci().episode_cluster_bootstrap(a[ok], np.asarray(eid)[ok], n_boot=n_boot, seed=0)
    return dict(mean=r.get("mean", r.get("point")), lo=r.get("lo"), hi=r.get("hi"), n_windows=int(ok.sum()), estimator=r.get("estimator", "episode_cluster_bootstrap"))


def metrics_of(plans, ws):
    """plans {name: plan_raw (n,30,2)} -> {name: {band: {metric: array}}}."""
    lead = getattr(ws, "lead", None)
    ws.text = getattr(ws, "text", None) if (getattr(ws, "text", None) is not None or not hasattr(ws, "now")) else attach_text(ws)        # TEXT-ATTACH: the one line that attaches the nav-text row
    text = getattr(ws, "text", None)
    return {k: band_metrics(v, ws.A, ws.v0, lead, getattr(ws, "lead6", None), text)[0] for k, v in plans.items()}


CONTRASTS = (("hwm_oracle", "hold"), ("hwm_oracle", "flat_oracle"), ("flat_oracle", "hold"), ("hwm_plan", "hold"), ("hwm_plan", "flat_plan"), ("flat_plan", "hold"), ("hwm_plan", "prior"))


def coverage_of(M, ws):
    """Per family, per band, per row: n windows that carry the metric and the reason when that is fewer than the windows. Headway: the lead-state counts of the join (NO_LABEL and NOT_STRAIGHT are never free flow), the windows
    in state LEAD, and per row the windows where the plan / the logged path kept the lead inside the corridor (lead lost = n_lead - n finite)."""
    lead = getattr(ws, "lead", None)
    lead6 = getattr(ws, "lead6", None)
    out = {"headway_scope": HEADWAY_SCOPE}
    if lead is None:
        out["lead_states"] = "n = 0: no lead block attached to the window set"
    else:
        st = np.asarray(lead["state"])
        out["lead_states"] = {k: int((st == k).sum()) for k in ("LEAD", "NO_LEAD", "NOT_STRAIGHT", "NO_LABEL", "NO_ROW")}
    if lead6 is not None:
        st6 = np.asarray(lead6["state"])
        out["lead_states_6s_block"] = {k: int((st6 == k).sum()) for k in ("LEAD", "NO_LEAD", "NOT_STRAIGHT", "NO_LABEL", "NO_ROW")}
    rows = {}
    for a, per in M.items():
        for b in BANDS:
            hw = np.asarray(per[b]["hw_err"], float)
            blk = lead6 if (BANDS[b][1] > 10 and lead6 is not None) else lead
            nl = int((np.asarray(blk["state"]) == "LEAD").sum()) if blk is not None else 0
            rows.setdefault(b, {})[a] = dict(n_windows=int(ws.n), n_lead_state=nl, n_headway_finite=int(np.isfinite(hw).sum()), n_ttc_finite=int(np.isfinite(per[b]["ttc_err"]).sum()),
                                             n_plan_lost_lead=int(np.isfinite(np.asarray(per[b]["_hw_l"], float)).sum() - np.isfinite(hw).sum()),
                                             reason=None if np.isfinite(hw).any() else ("n = 0: " + ("band outside the 2 s lead track" if (b != "0-2s" and lead6 is None) else "no window with a lead in both paths")))
    out["bands"] = rows
    return out


def tactical_blocks(M):
    """Selected-against-executed form for a plan-derived decision: per row and band, confusion (logged rows, planned columns), per-class recall, precision and n, and Cohen's kappa, over the lateral 3-way, the longitudinal
    3-way and the collapsed 5-way. taniteval.four_families._class_report / _kappa_k, reused."""
    import taniteval.four_families as FF
    from tanitad.refs.refc_tactical import LAT_CLASSES, LON_CLASSES, MAN5_NAMES
    out = {}
    for a, per in M.items():
        for b in BANDS:
            d = per[b]
            blk = {"n": int(len(d["_man_p"])), "decision_form": "plan-derived (planned trajectory classified) against executed (logged-action trajectory classified); one rule"}
            for nm, (kp, kl, names) in dict(lateral=("_lat_p", "_lat_l", LAT_CLASSES), longitudinal=("_lon_p", "_lon_l", LON_CLASSES), maneuver_5way_collapsed=("_man_p", "_man_l", MAN5_NAMES)).items():
                r = FF._class_report(d[kl], d[kp], list(names))
                r["kappa"] = FF._kappa_k(d[kl], d[kp], len(names))
                r["agreement"] = float((np.asarray(d[kl]) == np.asarray(d[kp])).mean())
                blk[nm] = r
            out.setdefault(a, {})[b] = blk
    return out


def _arr(M, a, b, m, n):
    """M[a][b][m], or NaN [n] when the arm's metrics do not carry m (n = 0 rows, never silently dropped)."""
    v = M[a][b].get(m)
    return np.full(n, np.nan) if v is None else v


def _empty_row(fam, b, m, a, tier, reason):
    return dict(kind="arm", family=fam, band=b, metric=m, arm=a, tier=tier, evidence="MEASURED", mean=float("nan"), lo=float("nan"), hi=float("nan"), n_windows=0, estimator="episode_cluster_bootstrap", reason=reason)


def _rate(x, eid):
    """n windows, n episodes and mean of the finite entries (a rate for the 0 / 1 cells)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    return dict(n_windows=int(ok.sum()), n_episodes=int(len(set(np.asarray(eid)[ok]))), mean=float(x[ok].mean()) if ok.any() else None)


def strategic_block(M, ws):
    """The computed STRATEGIC block: state, tier stamp, hypotheses, text coverage, per arm the S_ADH and S_FALSE rate with n, the logged-path ceiling (S_ADH) and logged-path rate (S_FALSE), and the n/a entries."""
    first = next(iter(M)) if M else None
    text = getattr(ws, "text", None)
    blk = {"state": S.STRATEGIC_STATE, "tier_stamp": S.STRATEGIC_TIER_STAMP, "scope": S.STRATEGIC_SCOPE, "bands": {b: (S.STRATEGIC_BAND_EMPTY if b != STRAT_BAND else "filled") for b in BANDS},
           "hypotheses": {"evidence": "HYPOTHESIS", "COMPLETE_M": S.COMPLETE_M, "theta_deg": dict(S.THETA_DEG), "false_margin_m": S.FALSE_MARGIN_M},
           "nav_text": {"table": "navtok_table.npz sha256 prefix " + ST.TABLE_SHA_PREFIX, "now_offset": ST.NOW_OFFSET, "evidence": "MEASURED (STRATEGIC_TEXT_OFFSET_CHECK.json)"},
           "not_computed": {k: v for k, v in NOT_COMPUTED.items() if k.startswith("STRATEGIC.")}}
    if text is None:
        blk["text_rows"] = "n = 0: no navigation-text row attached to the window set, so every STRATEGIC cell is NaN (n = 0), never 0"
    else:
        blk["text_rows"] = {"n_windows": int(len(text["text_ok"])), "n_text_ok": int(text["text_ok"].sum()), "flag_counts": {str(int(k)): int((text["flag"] == k).sum()) for k in np.unique(text["flag"])}}
    arms = {}
    for a in M:
        tier = ARMS[a][2] if a in ARMS else "T1-kin"
        arms[a] = dict(tier=tier, **{m: _rate(_arr(M, a, STRAT_BAND, m, ws.n), ws.eid) for m in ("s_adh", "s_false")})
    blk["arms"] = arms
    if first is not None:
        blk["logged_path"] = {"s_adh_ceiling": _rate(_arr(M, first, STRAT_BAND, STRAT_LOGGED["s_adh"], ws.n), ws.eid), "s_false_logged_rate": _rate(_arr(M, first, STRAT_BAND, STRAT_LOGGED["s_false"], ws.n), ws.eid),
                              "note": "the logged-action path read by the same cells; S_FALSE is read only as plan minus this rate"}
    return blk


def build_report(M, ws, n_boot=2000, seeds_M=None):
    """M {arm: metrics} at the primary inference seed (0); seeds_M {arm: {seed: metrics}} for the inference-seed replicate. Rows carry tier, estimator, evidence class.
    STRATEGIC: filled in the full 6 s band only (the other bands emit n = 0 rows with the reason); beside each arm row a 'logged' row (the logged path through the same cell) and a contrast of every arm against it."""
    rows = []
    n = ws.n
    first = next(iter(M)) if M else None
    for fam, mets in FAMILIES.items():
        for b in BANDS:
            for m in mets + (BESIDE if fam == "LATERAL" else ()) + (SIGNED if fam == "LONGITUDINAL" else ()):
                if fam == "STRATEGIC" and b != STRAT_BAND:
                    for a in [x for x in ARMS if x in M] + [x for x in REFS if x in M]:
                        rows.append(_empty_row(fam, b, m, a, ARMS[a][2] if a in ARMS else "T1-kin", S.STRATEGIC_BAND_EMPTY))
                    continue
                dr = DIRECTION.get(m, "lower") if fam == "STRATEGIC" else "lower"
                for a, (mode, gk, tier) in ARMS.items():
                    if a in M:
                        rows.append(dict(kind="arm", family=fam, band=b, metric=m, arm=a, tier=tier, evidence="MEASURED", **single(_arr(M, a, b, m, n), ws.eid, n_boot)))
                for ref in REFS:
                    if ref in M:
                        rows.append(dict(kind="arm", family=fam, band=b, metric=m, arm=ref, tier="T1-kin", evidence="MEASURED", **single(_arr(M, ref, b, m, n), ws.eid, n_boot)))
                for a, bb in CONTRASTS:
                    if a in M and bb in M and m not in SIGNED:
                        t = "T0" if "oracle" in a else "T1-kin"
                        rows.append(dict(kind="contrast", family=fam, band=b, metric=m, a=a, b=bb, tier=t, evidence="MEASURED", **contrast(_arr(M, a, b, m, n), _arr(M, bb, b, m, n), ws.eid, n_boot, dr)))
                if fam == "STRATEGIC" and first is not None:
                    rows.append(dict(kind="arm", family=fam, band=b, metric=m, arm="logged", tier="logged path", evidence="MEASURED", **single(_arr(M, first, b, STRAT_LOGGED[m], n), ws.eid, n_boot)))
                    for a in [x for x in ARMS if x in M] + [x for x in REFS if x in M]:
                        t = "T0" if "oracle" in a else "T1-kin"
                        rows.append(dict(kind="contrast", family=fam, band=b, metric=m, a=a, b="logged", tier=t, evidence="MEASURED", **contrast(_arr(M, a, b, m, n), _arr(M, first, b, STRAT_LOGGED[m], n), ws.eid, n_boot, dr)))
    rep = []
    for a, per in (seeds_M or {}).items():
        if 0 in per and 1 in per:
            for fam, mets in FAMILIES.items():
                for b in BANDS:
                    if fam == "STRATEGIC" and b != STRAT_BAND:
                        continue
                    for m in mets:
                        rep.append(dict(kind="inference_seed_replicate", arm=a, family=fam, band=b, metric=m, tier=ARMS[a][2], evidence="MEASURED",
                                        **contrast(per[1][b].get(m, np.full(n, np.nan)), per[0][b].get(m, np.full(n, np.nan)), ws.eid, n_boot, DIRECTION.get(m, "lower") if fam == "STRATEGIC" else "lower")))
    return dict(rows=rows, replicate=rep, coverage=coverage_of(M, ws), tactical=tactical_blocks(M), scopes=dict(headway=HEADWAY_SCOPE, tactical=TACTICAL_SCOPE, strategic=S.STRATEGIC_SCOPE), not_computed=NOT_COMPUTED, strategic=strategic_block(M, ws), n_windows=int(ws.n),
                n_episodes=int(len(set(ws.eid))), bands=BANDS,
                note="a claim that a lever moved a metric needs the replicate row to be smaller than the contrast (H-ESTIM-SEED-1); the bootstrap answers the episode question only")


def to_markdown(rep):
    st = rep["strategic"]
    L = [f"# HWM-PLAN read, every row stamped with its tier: T0 oracle-goal rows are diagnostics, T1-kin rows score the plan integrated by the kinematic model ({rep['n_windows']} windows, {rep['n_episodes']} episodes; estimator paired episode-cluster bootstrap, taniteval/ci.py)",
         f"STRATEGIC: {st['state']}", f"* STRATEGIC tier: {st['tier_stamp']}", f"* STRATEGIC text rows: {st['text_rows']}"]
    lp = st.get("logged_path")
    if lp:
        L.append(f"* STRATEGIC logged path: S_ADH ceiling {lp['s_adh_ceiling']}; S_FALSE logged-path rate {lp['s_false_logged_rate']} (S_FALSE is read only as plan minus this rate)")
    L.append("")
    for k, v in rep["not_computed"].items():
        L.append(f"* {k}: {v}")
    L += ["", f"* headway scope: {rep['scopes']['headway']}", f"* tactical scope: {rep['scopes']['tactical']}", f"* strategic scope: {rep['scopes']['strategic']}", f"* lead-state counts of the join: {rep['coverage']['lead_states']}"]
    L += ["", "| kind | family | band | metric | row | tier | n | delta or mean | 95 % interval | label |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["rows"] + rep["replicate"]:
        nm = r.get("arm") or f"{r['a']} - {r['b']}"
        v = r.get("delta", r.get("mean"))
        L.append(f"| {r['kind']} | {r['family']} | {r['band']} | {r['metric']} | {nm} | {r['tier']} | {r.get('n_windows')} | {v:.4f} | [{r['lo']:.4f}, {r['hi']:.4f}] | {r.get('label', '') if r.get('n_windows') else r.get('reason', 'n = 0')} |")
    return "\n".join(L) + "\n"
