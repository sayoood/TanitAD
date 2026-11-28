#!/usr/bin/env python3
"""r0e_diagnostics: the D1 / D2 / D3 diagnostics of PREREG_R0E_PICK.draft.md section 8, read from the dual's own log (and, for D3, the arm's own score dumps).  Launch blocker B6.
Stdlib for the log side (D1, D2, the per-check table); numpy only for D3's validation side.  No pod, no GPU.

    python eval/r0e_diagnostics.py --arm-dir <arm dir holding sft.jsonl> [--reference-log <R0d sft.jsonl>] [--outcome-json <safeobj_outcome.json> --slot R0d | --through-update N]
                                   [--registered-temp 0.01 --scores-dir <dir of scores_update{250,500,750,1000}_{A,B}.npz> --val-pdm <dir> --val-terms <dir>] [--out <json>]

It prints `ZZR0E_DIAG <D1|D2|D3|ROW> <status> <why>` lines and writes ONE json (`--out`, default `<arm dir>/r0e_diagnostics.json`).  Exit code: 0 = every diagnostic that was asked for was read, 2 = one is NOT_READ, 3 = usage.

WHAT THE TRAINER LOGS (every 25 updates, scorer_finetune.main): `con_g_<c>` = c^k_A - c^k_B (E_pi[label] of arm A minus arm B under the constraint's softmax policy; POSITIVE = B worse: the dual's own violation, a LOSS),
`con_p1_gap_<c>` = the label of B's argmax pick minus A's (NEGATIVE = B picks the worse one: the pick's loss with its sign flipped), `con_lam_<c>`, `con_cA_<c>` / `con_cB_<c>`, `con_hinge_<c>`, the cumulative
`con_cap_hits` / `con_lam_pos_updates`, and `epo_J`.  All label quantities are in label units; "points" below are x100.  In loss form the dual's quantity is L_T = 100 g and the pick's is L_P1 = -100 p1gap,
so the WEDGE 100 (g + p1gap) = L_T - L_P1: NEGATIVE = the softmax at the constraint's T UNDERSTATES the pick's loss (A2R / A2R2's `diff`), the quantity the draft's D1 holds to 0.10 points.

  D1  the in-run wedge: the mean over rows of 100 (con_g_ttc + con_p1_gap_ttc), +- its standard error (sample sd / sqrt(n), ddof 1).  Rows: update >= ramp + 26 (the draft's "post-ramp rows u226-u1000": the
      first is u250, 31 rows for R0d), through the first breached check (`--outcome-json`) or `--through-update`, else through the last logged row.  ALIGNED iff |D1| <= 0.10; BORDERLINE iff |D1| is within one
      standard error of 0.10.  The rows with update > ramp (the u225 row too) are read beside it as a sensitivity: the draft does not say which, and R0d reads -0.317 (31 rows) or -0.294 (32 rows).  NC is read the same way (descriptive).
  D2  J-crowding: rise = mean(epo_J over rows 751..1000) - epo_J at update 25, against the reference arm's rise (default: the banked R0d log, +0.0595).  CROWDED iff rise < half of it.  A log that ends before u751 is NOT_READ.
  D3  the constrained policy's transfer gap, per check u in {250, 500, 750, 1000}: the in-sample window (the 100 logged updates ending at u: addendum R3's window) mean of L_T = 100 g and L_P1 = -100 p1gap, against the VALIDATION
      L_T at the arm's constraint T and the validation L_P1, both from the arm's own scores_update{u}_{A,B}.npz on the RL-val sets (the A2 harness's quantities).  gap = val - in-sample (points; + = worse on val).  The in-sample
      side is one number per window (no resampling); the CI is the log-cluster bootstrap of the validation side only (10,000, seed 20260927: another draw of LOGS).
  ROW (a reading aid, not the registered table): the draft's section 6 splits O4 on |D1| (P-O4-PICK / P-O4-WEDGE) and O3 on lambda > 0 on >= 50 % of the post-ramp updates and D2 (O3-a / O3-b / O3 SLACK).

FAIL CLOSED: a missing, non-finite or non-contiguous log column, a missing ramp, a window or reference that cannot be built reads NOT_READ for that diagnostic; nothing is imputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.dirname(HERE)

TOOL = "r0e_diagnostics/1"
OUT_NAME = "r0e_diagnostics.json"
LOG_EVERY = 25                                                                  # the trainer writes one `train` row per 25 updates
CHECKS = (250, 500, 750, 1000)                                                  # the RL-val checks
WINDOW_UPDATES = 100                                                            # addendum R3: the 100 logged updates ending at the check
REGISTERED_RAMP = 200
D1_BAR = 0.10                                                                   # draft section 8 D1: aligned iff |D1| <= 0.10 points
D1_FIRST_ROW_OFFSET = LOG_EVERY + 1                                             # the draft's "post-ramp rows u226-u1000": update >= ramp + 26
D2_LATE_FROM, D2_FROM_ROW = 751, 25                                             # draft section 6 O3-a: the epo_J mean over u751-1000 against the u25 row
D2_FRACTION = 0.5                                                               # "less than half of R0d's rise"
N_BOOT, SEED = 10000, 20260927                                                  # PREREG_SAFEOBJ section 4: the estimator
LAM_POS_SHARE_MIN = 0.5                                                         # PREREG_SAFEOBJ O3 / O4: lambda > 0 on >= 50 % of the post-ramp updates
REQ = ("update", "con_lam_ttc", "con_lam_nc", "con_g_ttc", "con_g_nc", "con_p1_gap_ttc", "con_p1_gap_nc", "con_cA_ttc", "con_cB_ttc", "con_cA_nc", "con_cB_nc", "con_hinge_ttc", "con_hinge_nc",
       "con_cap_hits", "con_lam_pos_updates", "con_lam_max_ttc", "con_lam_max_nc", "epo_J", "epo_lam_pg")
DEFAULT_REFERENCE = os.path.join(PK, "raw", "2026-10-18-r0d-outcome", "raw", "sft.jsonl")        # the banked R0d trainer log


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _jsonl(path):
    """-> (rows, bad_lines, last_line_bad).  A half-written FINAL line (a SIGKILLed trainer) is tolerated and reported."""
    rows, bad, last_bad = [], 0, False
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [ln for ln in f.read().split("\n") if ln.strip()]
    for i, ln in enumerate(lines):
        try:
            r = json.loads(ln)
        except ValueError:
            bad += 1
            last_bad = i == len(lines) - 1
            continue
        if isinstance(r, dict):
            rows.append(r)
        else:
            bad += 1
    return rows, bad, last_bad


def _mean_se(xs):
    """-> (mean, standard error of the mean with ddof 1, n).  se is None for n < 2."""
    n = len(xs)
    m = math.fsum(xs) / n
    if n < 2:
        return m, None, n
    var = math.fsum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var) / math.sqrt(n), n


# ============================================================ the log
def read_log(path):
    """-> {"readable", "problems", "notes", "ramp", "trains", "data", "done", "sha256"}.  Never raises."""
    out = {"readable": False, "problems": [], "notes": [], "ramp": None, "trains": [], "data": None, "done": None, "path": os.path.abspath(path), "sha256": None}
    pb = out["problems"]
    if not os.path.exists(path):
        pb.append(f"{path} is missing")
        return out
    try:
        out["sha256"] = _sha(path)
        rows, bad, last_bad = _jsonl(path)
        if bad:
            (out["notes"] if (bad == 1 and last_bad) else pb).append(f"{bad} unparsable line(s)" + (" (only the final, half-written one: tolerated)" if (bad == 1 and last_bad) else ""))
        data = next((r for r in rows if r.get("event") == "data"), None)
        trains = [r for r in rows if r.get("event") == "train"]
        out["data"], out["trains"] = data, trains
        out["done"] = next((r for r in reversed(rows) if r.get("event") == "done"), None)
        ramp = ((data or {}).get("epo") or {}).get("ramp")
        if not isinstance(ramp, int) or isinstance(ramp, bool) or ramp < 0:
            pb.append(f"the data event does not carry the BUILT --epo-ramp as an integer (got {ramp!r})")
        else:
            out["ramp"] = ramp
        if not trains:
            pb.append("no `train` row")
        else:
            ups = [r.get("update") for r in trains]
            if ups != list(range(LOG_EVERY, LOG_EVERY * len(trains) + 1, LOG_EVERY)):
                pb.append(f"the `train` rows are not the contiguous {LOG_EVERY}-update series from {LOG_EVERY} (updates {ups[:4]}...{ups[-2:]})")
            for r in trains:
                miss = [k for k in REQ if not _num(r.get(k))]
                if miss:
                    pb.append(f"train row at update {r.get('update')!r} lacks finite {miss}")
                    break
        out["readable"] = not pb
    except (OSError, ValueError, TypeError) as e:
        pb.append(f"the log could not be interpreted: {type(e).__name__}: {str(e)[:120]}")
        out["readable"] = False
    return out


def _rows_between(trains, lo, hi):
    return [r for r in trains if lo <= r["update"] <= hi]


# ============================================================ D1
def _wedge_block(rows, comp):
    g = [100.0 * r[f"con_g_{comp}"] for r in rows]
    p = [100.0 * r[f"con_p1_gap_{comp}"] for r in rows]
    w = [a + b for a, b in zip(g, p)]
    (mg, sg, n), (mp, sp, _), (mw, sw, _) = _mean_se(g), _mean_se(p), _mean_se(w)
    return {"n": n, "g_x100": {"mean": mg, "se": sg}, "p1gap_x100": {"mean": mp, "se": sp}, "wedge_x100": {"mean": mw, "se": sw}}


def d1_wedge(lg, through=None, bar=D1_BAR):
    """D1 from a read_log() result.  `through` = the last update included (None = the last logged row)."""
    out = {"status": "NOT_READ", "bar": bar, "problems": []}
    if not lg["readable"]:
        out["problems"] += [f"log: {p}" for p in lg["problems"]]
        return out
    ramp, trains = lg["ramp"], lg["trains"]
    last = trains[-1]["update"]
    hi = last if through is None else through
    if not _num(hi) or hi > last:
        out["problems"].append(f"through-update {through!r} is not a logged update (the log ends at {last})")
        return out
    rows = _rows_between(trains, ramp + D1_FIRST_ROW_OFFSET, hi)
    alt = _rows_between(trains, ramp + 1, hi)
    if len(rows) < 2:
        out["problems"].append(f"{len(rows)} post-ramp row(s) (updates {ramp + D1_FIRST_ROW_OFFSET}..{hi}): a standard error needs two")
        return out
    ttc = _wedge_block(rows, "ttc")
    d1, se = ttc["wedge_x100"]["mean"], ttc["wedge_x100"]["se"]
    out.update({"status": "READ", "ramp": ramp, "through_update": hi, "first_row_rule": f"update >= ramp + {D1_FIRST_ROW_OFFSET}", "rows": {"first": rows[0]["update"], "last": rows[-1]["update"], "n": len(rows)},
                "ttc": ttc, "D1_x100": d1, "se": se, "aligned": bool(abs(d1) <= bar), "borderline": bool(abs(abs(d1) - bar) <= se),
                "nc_descriptive": _wedge_block(rows, "nc"),
                "sensitivity_rows_gt_ramp": {"rows": {"first": alt[0]["update"], "last": alt[-1]["update"], "n": len(alt)}, "ttc": _wedge_block(alt, "ttc"), "nc_descriptive": _wedge_block(alt, "nc")},
                "sign": "negative = the softmax at the constraint's T understates the pick's loss (L_T - L_P1)"})
    alt_d1 = out["sensitivity_rows_gt_ramp"]["ttc"]["wedge_x100"]["mean"]
    out["sensitivity_rows_gt_ramp"]["D1_x100"], out["sensitivity_rows_gt_ramp"]["aligned"] = alt_d1, bool(abs(alt_d1) <= bar)
    out["sensitivity_changes_the_verdict"] = bool(out["aligned"] != out["sensitivity_rows_gt_ramp"]["aligned"])
    return out


# ============================================================ D2
def d2_crowding(lg, ref_rise=None, fraction=D2_FRACTION):
    out = {"status": "NOT_READ", "fraction": fraction, "problems": []}
    if not lg["readable"]:
        out["problems"] += [f"log: {p}" for p in lg["problems"]]
        return out
    by_u = {r["update"]: r for r in lg["trains"]}
    late = _rows_between(lg["trains"], D2_LATE_FROM, 1000)
    last = lg["trains"][-1]["update"]
    if D2_FROM_ROW not in by_u or last < 1000 or not late:
        out["problems"].append(f"D2 needs the u{D2_FROM_ROW} row and the rows u{D2_LATE_FROM}..1000; the log ends at u{last} (a killed arm has no D2)")
        return out
    j0, jl = by_u[D2_FROM_ROW]["epo_J"], math.fsum(r["epo_J"] for r in late) / len(late)
    rise = jl - j0
    out.update({"status": "READ", "J_u25": j0, "J_mean_u751_1000": jl, "n_late_rows": len(late), "rise": rise, "reference_rise": ref_rise})
    if ref_rise is None or not _num(ref_rise):
        out["status"] = "NOT_READ"
        out["problems"].append("no reference rise (the banked R0d log was not given or not readable): crowding cannot be judged")
        return out
    if ref_rise <= 0:
        out["status"] = "NOT_READ"
        out["problems"].append(f"the reference rise {ref_rise!r} is not positive: 'less than half of it' is undefined")
        return out
    out["ratio_to_reference"] = rise / ref_rise
    out["crowded"] = bool(rise < fraction * ref_rise)
    return out


def reference_rise(path):
    """-> (rise, problems): the reference arm's own D2 rise (the banked R0d log by default)."""
    lg = read_log(path)
    r = d2_crowding(lg, ref_rise=1.0)                  # the placeholder reference only lets the rise itself be read
    if r.get("rise") is None:
        return None, [f"reference log {path}: {p}" for p in r["problems"]] or ["reference log: no rise"]
    return r["rise"], []


# ============================================================ the per-check table and D3's in-sample side
def per_check(lg, checks=CHECKS, window=WINDOW_UPDATES):
    out = {}
    if not lg["readable"]:
        return out
    ramp = lg["ramp"]
    by_u = {r["update"]: r for r in lg["trains"]}
    for c in checks:
        if c not in by_u:
            continue
        need = [c - window + LOG_EVERY * i for i in range(1, window // LOG_EVERY + 1)]
        if not all(u in by_u for u in need):
            out[str(c)] = {"update": c, "status": "NOT_READ", "problems": [f"the rows {need} that make the {window}-update window ending at u{c} are not all present"]}
            continue
        rows = [by_u[u] for u in need]
        row = by_u[c]
        m = lambda k: math.fsum(r[k] for r in rows) / len(rows)                       # noqa: E731
        ins = {"L_T_ttc": 100.0 * m("con_g_ttc"), "L_T_nc": 100.0 * m("con_g_nc"), "L_P1_ttc": -100.0 * m("con_p1_gap_ttc"), "L_P1_nc": -100.0 * m("con_p1_gap_nc"),
               "wedge_ttc": 100.0 * (m("con_g_ttc") + m("con_p1_gap_ttc")), "wedge_nc": 100.0 * (m("con_g_nc") + m("con_p1_gap_nc")),
               "cA_ttc": 100.0 * m("con_cA_ttc"), "cB_ttc": 100.0 * m("con_cB_ttc"), "cA_nc": 100.0 * m("con_cA_nc"), "cB_nc": 100.0 * m("con_cB_nc"),
               "hinge_ttc": m("con_hinge_ttc"), "hinge_nc": m("con_hinge_nc")}
        out[str(c)] = {"update": c, "status": "READ", "window": [need[0] - LOG_EVERY + 1, c], "window_overlaps_ramp": bool(need[0] - LOG_EVERY + 1 <= ramp), "lam_ttc": row["con_lam_ttc"], "lam_nc": row["con_lam_nc"],
                       "lam_pos_updates": row["con_lam_pos_updates"], "cap_hits": row["con_cap_hits"], "lam_seen_max_ttc": row["con_lam_max_ttc"], "lam_seen_max_nc": row["con_lam_max_nc"],
                       "epo_J": row["epo_J"], "in_sample_x100": ins}
    return out


def post_ramp_lambda_share(lg, through=None):
    """-> (share, k, m): lambda > 0 on k of the m post-ramp updates (ramp+1 .. through), from the cumulative counter.  None when the ramp row or the end row is missing."""
    if not lg["readable"]:
        return None
    by_u = {r["update"]: r for r in lg["trains"]}
    end = lg["trains"][-1]["update"] if through is None else through
    ramp = lg["ramp"]
    if ramp not in by_u or end not in by_u or end <= ramp:
        return None
    k, m = by_u[end]["con_lam_pos_updates"] - by_u[ramp]["con_lam_pos_updates"], end - ramp
    return k / m, k, m


# ============================================================ D3's validation side (numpy; pure functions on log-scores, plus a loader that reuses the A2 harness's own loaders)
def _np():
    import numpy as np
    return np


def softmax_T(l, T):
    np = _np()
    z = l / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def tie_pick_mean(l, y, tau):
    """mean label over Q(tau) = {j : l_j >= l_max - tau}; tau = 0 is the ordinary pick when the top is not an exact tie (A2R2's qmean)."""
    np = _np()
    w = (l >= l.max(-1, keepdims=True) - tau).astype(np.float64)
    return (w * y).sum(-1) / w.sum(-1)


def val_side(lA, lB, y, logs, T, tau0, n_boot=N_BOOT, seed=SEED):
    """-> the validation quantities of D3 for one check and one label: L_T at T, L_P1 (argmax) and L_P1^tau0 (tie-aware), in points, loss POSITIVE (arm B worse than arm A); the log-cluster bootstrap CI of each.
    lA / lB [N, M] = log(sel_v1_lane(sigmoid(logits)) + eps); y [N, M] the label; logs [N] the log name of each set."""
    np = _np()
    ul, inv = np.unique(np.asarray(logs).astype(str), return_inverse=True)
    nlog = len(ul)
    ar = np.arange(len(y))
    dT = 100.0 * ((softmax_T(lB, T) * y).sum(-1) - (softmax_T(lA, T) * y).sum(-1))
    d1 = 100.0 * (y[ar, lB.argmax(-1)] - y[ar, lA.argmax(-1)])
    dt = 100.0 * (tie_pick_mean(lB, y, tau0) - tie_pick_mean(lA, y, tau0))
    VI = np.random.default_rng(seed).integers(0, nlog, size=(n_boot, nlog))

    def L(d):
        s, c = np.bincount(inv, weights=d, minlength=nlog), np.bincount(inv, minlength=nlog).astype(np.float64)
        pt = float(-(s.sum()) / c.sum())
        boot = -(s[VI].sum(-1)) / c[VI].sum(-1)
        return {"value": pt, "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    return {"T": T, "tau0": tau0, "n_sets": int(len(y)), "n_logs": int(nlog), "L_T": L(dT), "L_P1_argmax": L(d1), "L_P1_tie_tau0": L(dt)}


def d3_gap(ins, val, comp="ttc"):
    """the transfer gap in points (validation - in-sample; positive = worse on validation) for the expectation and for the pick."""
    return {"gap_expectation": val["L_T"]["value"] - ins[f"L_T_{comp}"], "gap_pick_argmax": val["L_P1_argmax"]["value"] - ins[f"L_P1_{comp}"],
            "wedge_val_tie": val["L_T"]["value"] - val["L_P1_tie_tau0"]["value"], "wedge_in_sample": ins[f"wedge_{comp}"]}


def eps_eq():
    """A2R's eps_eq = T_c ln((M - 1) / rho) with T_c 1e-4, M 64 proposals, rho 5e-9: the tie-aware pick's tau (so_a2r_neartie.EPS_EQ, computed with the same numpy call so the float is the same)."""
    np = _np()
    return float(1e-4 * np.log((64 - 1) / 5e-9))


def load_val(pk, pdm_dir, terms_dir):
    """-> (AT, keys, comps, ck): the A2 harness's own loader (attrib_rls1b.load_side) and label split."""
    sys.path.insert(0, os.path.join(pk, "raw", "2026-10-14-rls1b-attribution"))
    import attrib_rls1b as AT  # noqa: E402
    keys, tg, d, ck, _ = AT.load_side(pdm_dir, terms_dir)
    return AT, keys, AT.comps(tg), ck


def d3_val(lg, scores_dir, pdm_dir, terms_dir, T, pk=PK, comps=("ttc",)):
    """D3: the table of the in-sample windows beside the validation reads, per check present in both.  -> dict."""
    out = {"status": "NOT_READ", "registered_T": T, "problems": [], "checks": {}}
    if not lg["readable"]:
        out["problems"] += [f"log: {p}" for p in lg["problems"]]
        return out
    ec = (lg["data"] or {}).get("epo_con") or {}
    built = ec["temp"] if "temp" in ec else 0.1                                  # a log with no `temp` key is a trainer that predates --epo-con-temp: the constraint ran at 0.1
    if built != T:
        out["problems"].append(f"the T of the validation side ({T!r}) is not the T the log says the constraint RAN at ({built!r}): the two sides would not measure one quantity")
        return out
    np = _np()
    AT, keys, cmp_, ck = load_val(pk, pdm_dir, terms_dir)
    tau0 = eps_eq()
    logs = [k[0] for k in keys]
    table = per_check(lg)
    for c, rec in table.items():
        if rec["status"] != "READ":
            continue
        pa, pb_ = os.path.join(scores_dir, f"scores_update{c}_A.npz"), os.path.join(scores_dir, f"scores_update{c}_B.npz")
        if not (os.path.exists(pa) and os.path.exists(pb_)):
            out["checks"][c] = {"status": "NOT_READ", "problems": [f"no scores dump for u{c} in {scores_dir}"]}
            continue
        lA = np.log(AT.sel_v1_lane(AT.RV.sigmoid(AT.load_scores_aligned(pa, keys, ck))) + AT.EPS)
        lB = np.log(AT.sel_v1_lane(AT.RV.sigmoid(AT.load_scores_aligned(pb_, keys, ck))) + AT.EPS)
        row = {"status": "READ", "in_sample_x100": rec["in_sample_x100"], "window_overlaps_ramp": rec["window_overlaps_ramp"], "by_comp": {}}
        for comp in comps:
            v = val_side(lA, lB, cmp_[comp], logs, T, tau0)
            row["by_comp"][comp] = {"val": v, "gap": d3_gap(rec["in_sample_x100"], v, comp)}
        out["checks"][c] = row
    out["status"] = "READ" if any(v.get("status") == "READ" for v in out["checks"].values()) else "NOT_READ"
    if out["status"] == "NOT_READ":
        out["problems"].append("no check could be read on the validation side")
    return out


# ============================================================ the reading aid for section 6's rows
def refine_row(outcome, flags, d1, d2, share):
    """-> (row, why).  outcome / flags from safeobj_outcome.json; d1 / d2 from above; share = (lambda>0 share, k, m) through the end of the read.  Only the two splits of the draft's section 6:
    O4 (not SLACK) on D1 -> P-O4-PICK / P-O4-WEDGE;  O3 on lambda share and D2 -> O3-a / O3-b / O3 SLACK.  Anything else is returned as it came; an unread input reads UNREAD, never a guess."""
    if outcome == "O4" and "SLACK" not in (flags or []):
        if d1.get("status") != "READ":
            return "UNREAD", "O4 splits on D1, which is not read"
        return ("P-O4-PICK" if d1["aligned"] else "P-O4-WEDGE") + (" (BORDERLINE)" if d1["borderline"] else ""), f"D1 = {d1['D1_x100']:+.3f} +- {d1['se']:.3f} points"
    if outcome == "O3":
        if share is None:
            return "UNREAD", "O3 splits on the post-ramp lambda share, which is not readable"
        if share[0] < LAM_POS_SHARE_MIN:
            return "O3 SLACK", f"lambda > 0 on {share[1]} of {share[2]} post-ramp updates"
        if d2.get("status") != "READ" or "crowded" not in d2:
            return "UNREAD", "O3 splits on D2, which is not read"
        return ("O3-a" if d2["crowded"] else "O3-b"), f"D2 rise {d2['rise']:+.4f} = {d2['ratio_to_reference']:.2f} x the reference"
    return outcome, "no split in the draft's section 6 for this outcome"


# ============================================================ the whole read
def run(arm_dir, *, reference_log=DEFAULT_REFERENCE, through=None, outcome_arm=None, registered_temp=None, scores_dir=None, val_pdm=None, val_terms=None, pk=PK):
    sft = os.path.join(arm_dir, "sft.jsonl")
    lg = read_log(sft)
    rec = {"tool": TOOL, "arm_dir": os.path.abspath(arm_dir), "log": {k: lg[k] for k in ("path", "sha256", "readable", "problems", "notes", "ramp")}, "problems": []}
    rec["ramp_is_registered_200"] = bool(lg["ramp"] == REGISTERED_RAMP)
    ref, ref_pb = reference_rise(reference_log) if reference_log else (None, ["no reference log"])
    rec["reference"] = {"log": os.path.abspath(reference_log) if reference_log else None, "rise": ref, "problems": ref_pb, "sha256": _sha(reference_log) if reference_log and os.path.exists(reference_log) else None}
    if outcome_arm is not None and through is None:
        c = outcome_arm.get("first_breached_check")
        if isinstance(c, str) and c.startswith("update") and c[6:].isdigit():
            through = int(c[6:])
    rec["through_update_requested"] = through
    rec["D1"] = d1_wedge(lg, through)
    rec["D2"] = d2_crowding(lg, ref)
    rec["per_check"] = per_check(lg)
    rec["lambda_post_ramp_share"] = None
    sh = post_ramp_lambda_share(lg, through)
    if sh is not None:
        rec["lambda_post_ramp_share"] = {"share": sh[0], "k": sh[1], "m": sh[2], "through": through if through is not None else lg["trains"][-1]["update"]}
    if outcome_arm is not None:
        row, why = refine_row(outcome_arm.get("outcome"), outcome_arm.get("outcome_flags"), rec["D1"], rec["D2"], sh)
        rec["ROW"] = {"outcome": outcome_arm.get("outcome"), "flags": outcome_arm.get("outcome_flags"), "refined": row, "why": why}
    if scores_dir:
        if registered_temp is None or not _num(registered_temp) or registered_temp <= 0 or not (val_pdm and val_terms):
            rec["D3"] = {"status": "NOT_READ", "problems": ["D3's validation side needs --registered-temp (the arm's registration), --val-pdm and --val-terms"]}
        else:
            try:
                rec["D3"] = d3_val(lg, scores_dir, val_pdm, val_terms, registered_temp, pk)
            except (ImportError, OSError, SystemExit, ValueError, KeyError) as e:
                rec["D3"] = {"status": "NOT_READ", "problems": [f"the validation side could not be built: {type(e).__name__}: {str(e)[:160]}"]}
    body = json.dumps({k: v for k, v in rec.items() if k != "record_sha256"}, sort_keys=True, separators=(",", ":"), default=str)
    rec["record_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm-dir", required=True)
    ap.add_argument("--reference-log", default=DEFAULT_REFERENCE)
    ap.add_argument("--through-update", type=int, default=None)
    ap.add_argument("--outcome-json", default=None)
    ap.add_argument("--slot", default=None, help="R0d / R1d: the arm's slot in --outcome-json")
    ap.add_argument("--registered-temp", type=float, default=None)
    ap.add_argument("--scores-dir", default=None)
    ap.add_argument("--val-pdm", default=None)
    ap.add_argument("--val-terms", default=None)
    ap.add_argument("--pk", default=PK)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if not os.path.isdir(a.arm_dir) or bool(a.outcome_json) != bool(a.slot) or (a.through_update is not None and a.outcome_json):
        print("usage: --arm-dir must exist; --outcome-json and --slot go together and exclude --through-update", flush=True)
        return 3
    arm = None
    if a.outcome_json:
        try:
            with open(a.outcome_json, encoding="utf-8") as f:
                arm = (json.load(f).get("arms") or {}).get(a.slot)
        except (OSError, ValueError):
            arm = None
        if not isinstance(arm, dict):
            print(f"usage: {a.outcome_json} has no arms.{a.slot}", flush=True)
            return 3
    rec = run(a.arm_dir, reference_log=a.reference_log, through=a.through_update, outcome_arm=arm, registered_temp=a.registered_temp, scores_dir=a.scores_dir, val_pdm=a.val_pdm, val_terms=a.val_terms, pk=a.pk)
    out = a.out or os.path.join(a.arm_dir, OUT_NAME)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, out)
    d1, d2 = rec["D1"], rec["D2"]
    if d1["status"] == "READ":
        print(f"ZZR0E_DIAG D1 READ {d1['D1_x100']:+.4f} +- {d1['se']:.4f} points over {d1['rows']['n']} rows u{d1['rows']['first']}..u{d1['rows']['last']}: "
              f"{'ALIGNED' if d1['aligned'] else 'NOT ALIGNED'}{' BORDERLINE' if d1['borderline'] else ''} (bar {d1['bar']}); rows > ramp: {d1['sensitivity_rows_gt_ramp']['D1_x100']:+.4f}", flush=True)
    else:
        print(f"ZZR0E_DIAG D1 NOT_READ {'; '.join(d1['problems'])[:300]}", flush=True)
    if d2["status"] == "READ":
        print(f"ZZR0E_DIAG D2 READ rise {d2['rise']:+.4f} (J u25 {d2['J_u25']:.4f} -> mean u751-1000 {d2['J_mean_u751_1000']:.4f}) = {d2['ratio_to_reference']:.3f} x the reference {d2['reference_rise']:+.4f}: "
              f"{'CROWDED' if d2['crowded'] else 'NOT crowded'}", flush=True)
    else:
        print(f"ZZR0E_DIAG D2 NOT_READ {'; '.join(d2['problems'])[:300]}", flush=True)
    if "D3" in rec:
        print(f"ZZR0E_DIAG D3 {rec['D3']['status']} {'; '.join(rec['D3'].get('problems', []))[:200]}", flush=True)
    if "ROW" in rec:
        print(f"ZZR0E_DIAG ROW {rec['ROW']['refined']} ({rec['ROW']['outcome']}) {rec['ROW']['why']}", flush=True)
    print(f"ZZR0E_DIAG FILE {out} record_sha256={rec['record_sha256']}", flush=True)
    asked = [d1, d2] + ([rec["D3"]] if "D3" in rec else [])
    return 0 if all(x["status"] == "READ" for x in asked) else 2


if __name__ == "__main__":
    sys.exit(main())
