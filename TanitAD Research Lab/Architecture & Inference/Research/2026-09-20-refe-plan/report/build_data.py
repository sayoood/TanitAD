"""Collect the REFe training-progress data into one JSON for the report page (every number from an artifact)."""
import json
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# the learning-curve points (and the proptable/ beside them); overridable so a fixture can be rendered without
# touching the live data directory
PTS = os.environ.get("REFE_REPORT_POINTS", "D:/Projects/TanitAD/data/refe_navtest/points")
BERLIN = timezone(timedelta(hours=2))


def loc(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(BERLIN).strftime("%Y-%m-%d %H:%M")


steps, banks, epochs, ckpts, declared, ops = [], [], [], [], [], []
start_at = None
for line in open(os.path.join(HERE, "metrics.jsonl"), encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line.replace("NaN", "null"))
    ev = r.get("event")
    if ev == "start":
        start_at = start_at or r["at"]
    elif ev == "bank":
        banks.append(r)
    elif ev == "epoch":
        epochs.append(r)
    elif ev == "ckpt":
        ckpts.append(r)
    elif ev == "declared_change":
        declared.append(r)
    elif ev == "onpolicy":
        ops.append(r)
    elif "step" in r and ev is None:
        steps.append(r)

# one record per step: a resume can re-log a step -- keep the LAST record of each step
by = {}
for r in steps:
    by[r["step"]] = r
S = [by[k] for k in sorted(by)]
step = np.array([r["step"] for r in S])
el = np.array([r["elapsed_s"] for r in S], dtype=float)
l1 = np.array([r["traj_L1"] for r in S], dtype=float)
sc = np.array([r["score"] if r["score"] is not None else np.nan for r in S], dtype=float)
win = np.array([r["winners"] for r in S], dtype=float)
ad = np.array([r["assign_d"] if r["assign_d"] is not None else np.nan for r in S], dtype=float)
cov = np.array([r["scorer_cov"] for r in S], dtype=float)
lr = np.array([r["lr"] for r in S], dtype=float)
ep = np.array([r["epoch"] for r in S])
smp = np.array([r.get("samples", np.nan) for r in S], dtype=float)
mem = np.array([r.get("cuda_max_mem_gb", np.nan) for r in S], dtype=float)
dt = np.diff(el, prepend=np.nan)
dt[0] = el[0]


def rolling_median(x, w=51):
    h = w // 2
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - h): i + h + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = float(np.median(seg))
    return out


def rolling_mean(x, w=51):
    h = w // 2
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - h): i + h + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = float(np.mean(seg))
    return out


# windowed scorer coverage from the CUMULATIVE ratio: n_cov(t) = cov(t) * samples(t)
ncov = cov * smp
win_cov = np.full(len(S), np.nan)
W = 50
for i in range(len(S)):
    j = max(0, i - W)
    ds = smp[i] - smp[j]
    if ds > 0:
        win_cov[i] = (ncov[i] - ncov[j]) / ds

# ⭐ THE SCORER'S SUPERVISION CHANGED AT A DECLARED STEP (PI decision B; the trainer's own `declared_change` line).
# Before it, `score` is the fixed-candidate loss and `assign_d` the banked candidate's distance to the nearest
# proposal. After it, `score` is the LAST micro-batch's on-policy loss -- 0 unless that micro-batch held a labelled
# set, so it is not a loss curve -- and `assign_d` is the labelled trajectories' drift to the proposals of now.
# One series across the switch would draw a scorer loss "collapsing" to 0 and a distance "falling" from 4 m to
# 0.3 m, both artefacts of the change in meaning; every chart and caption reads the two regimes separately.
switch_step = min((d["step"] for d in declared), default=None)
post = step > switch_step if switch_step is not None else np.zeros(len(S), dtype=bool)
sc_fix = np.where(post, np.nan, sc)
ad_fix = np.where(post, np.nan, ad)
drift = np.where(post, ad, np.nan)

# the trainer's on-policy windows (one per logged step, sets > 0), pooled into bins weighted by their set counts:
# the training-side selection skill on the labelled sets (SPEC_NAVTEST Amendment 4, secondary readout)
opw = {}
for o in ops:
    if (o.get("sets") or 0) > 0:
        opw[o["step"]] = o                     # a replayed step keeps its last record


def op_pool(keys):
    rows = [opw[k] for k in keys]
    n = sum(x["sets"] for x in rows)
    if not n:
        return None
    pk, rd, bs = (sum(x[f] * x["sets"] for x in rows) / n for f in ("pick", "random", "best"))
    lags = [(x["lag_steps"], x["sets"]) for x in rows if x.get("lag_steps") is not None]
    return {"sets": n, "steps": len(rows), "pick": round(pk, 4), "random": round(rd, 4), "best": round(bs, 4),
            "skill": round((pk - rd) / (bs - rd), 4) if bs - rd > 1e-9 else None,
            "lag_steps": round(sum(l * s for l, s in lags) / sum(s for _, s in lags), 1) if lags else None}


OP_BIN = 25
op_bins = []
if opw:
    ks = sorted(opw)
    for lo in range(ks[0] - ks[0] % OP_BIN, ks[-1] + 1, OP_BIN):
        pool = op_pool([k for k in ks if lo <= k < lo + OP_BIN])
        if pool:
            # a bin still filling holds fewer sets and is the noisiest point: charts draw complete bins only
            op_bins.append({"step_lo": lo, "step_hi": lo + OP_BIN - 1,
                            "complete": bool(lo + OP_BIN - 1 <= int(step[-1])), **pool})

r = lambda a, n=4: [None if (a_ is None or not np.isfinite(a_)) else round(float(a_), n) for a_ in a]
series = {
    "step": step.tolist(), "elapsed_h": r(el / 3600.0, 4), "traj_L1": r(l1, 4),
    "traj_L1_med": r(rolling_median(l1), 4), "score": r(sc, 5), "score_med": r(rolling_median(sc), 5),
    "winners": win.astype(int).tolist(), "winners_mean": r(rolling_mean(win), 3),
    "assign_d": r(ad, 3), "assign_d_med": r(rolling_median(ad), 3),
    # the regime-split twins the charts draw (medians never reach across the switch)
    "score_fixed": r(sc_fix, 5), "score_fixed_med": r(np.where(post, np.nan, rolling_median(sc_fix)), 5),
    "assign_d_fixed": r(ad_fix, 3), "assign_d_fixed_med": r(np.where(post, np.nan, rolling_median(ad_fix)), 3),
    "op_drift": r(drift, 3), "op_drift_med": r(np.where(post, rolling_median(drift), np.nan), 3),
    "scorer_cov_cum": r(cov, 4), "scorer_cov_win": r(win_cov, 4), "lr": r(lr, 8),
    "epoch": ep.tolist(), "s_per_step": r(dt, 2), "mem_gb": r(mem, 2),
}

# per-epoch aggregates (the table-view twin of the step charts)
per_epoch = []
for e in sorted(set(ep.tolist())):
    m = ep == e
    idx = np.where(m)[0]
    dts = dt[m][1:] if m.sum() > 1 else dt[m]
    bank = next((b for b in banks if b["epoch"] == e), None)
    fixm, opm = m & ~post, m & post
    per_epoch.append({
        "epoch": int(e), "steps": int(m.sum()), "first_step": int(step[idx[0]]), "last_step": int(step[idx[-1]]),
        "traj_L1_median": round(float(np.nanmedian(l1[m])), 4),
        # fixed-candidate steps only: after the switch these fields mean something else (see above)
        "score_median": (round(float(np.nanmedian(sc[fixm][sc[fixm] > 0])), 5) if np.any(sc[fixm] > 0) else None),
        "winners_mean": round(float(np.nanmean(win[m])), 3),
        "assign_d_median": (round(float(np.nanmedian(ad[fixm])), 3) if np.any(np.isfinite(ad[fixm])) else None),
        "scorer_mode": "onpolicy" if post[m].all() else ("switched" if post[m].any() else "fixed"),
        "op_drift_median": (round(float(np.nanmedian(ad[opm])), 3) if np.any(np.isfinite(ad[opm])) else None),
        "op": op_pool([k for k in sorted(opw) if k in set(step[m].tolist())]),
        "scorer_cov_window": (round(float((ncov[idx[-1]] - (ncov[idx[0] - 1] if idx[0] > 0 else 0))
                                          / max(smp[idx[-1]] - (smp[idx[0] - 1] if idx[0] > 0 else 0), 1)), 4)),
        "s_per_step_median": round(float(np.nanmedian(dts)), 2),
        "bank_tuples": bank["n_tuples"] if bank else None, "bank_scenes": bank["n_scenes"] if bank else None,
        "bank_rank1": (bank["per_rank"].get("1") if bank else None),
        "bank_scorer_frames": bank["scorer_frames"] if bank else None,
    })

# ONE ROW PER EPOCH. A restart inside an epoch re-logs its bank (the on-policy switch at step 3,708 did,
# in epoch 11): keep the epoch's FIRST read -- what most of the epoch trained on -- and record the mode the
# epoch ended in, so a chart never draws two bars for one epoch and never shows epoch 11 as unsupervised.
be = {}
for b in banks:
    row = {"epoch": b["epoch"], "tuples": b["n_tuples"], "scenes": b["n_scenes"],
           "rank0": b["per_rank"].get("0"), "rank1": b["per_rank"].get("1", 0),
           "scorer_frames": b["scorer_frames"], "cov_norm": round(b["cov_norm"], 4),
           "scorer_mode": b.get("scorer_mode", "fixed"), "onpolicy_sets": b.get("onpolicy_sets"),
           "onpolicy_navsim_dac_sets": b.get("onpolicy_navsim_dac_sets"), "at_local": loc(b["at"])}
    if b["epoch"] not in be:
        be[b["epoch"]] = row
    else:
        be[b["epoch"]].update({"restarted_at_local": row["at_local"], "mode_end": row["scorer_mode"]})
bank_events = [be[e] for e in sorted(be)]
epoch_events = [{"epoch": e["epoch"], "step": e["step"], "at_local": loc(e["at"])} for e in epochs]

# steady speed: median of per-step times excluding epoch-boundary steps
bstep = {e["step"] for e in epochs}
steady = np.array([dt[i] for i in range(1, len(S)) if step[i] not in bstep and step[i] - 1 not in bstep])
cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
total_steps = 10075
last_i = len(S) - 1
remaining = total_steps - 1 - int(step[last_i])
avg_all = float(el[last_i] / (step[last_i] + 1))
eta_h = remaining * avg_all / 3600.0

# ⭐ THE FINISH IS PROJECTED FROM THE PACE NOW, AGAINST THE WALL-CLOCK 7-DAY LINE. The lifetime average
# hides a pace change: from 09:20Z on 2026-09-26 the pod-local on-policy dump shares the GPU and costs
# ~4.8 s/step -- MEASURED by pausing it (raw/2026-09-26-onpolicy-switch/dump_cost_pause_ab.txt), which is
# why the steps of that A/B window are left out of the current pace. `el` also stops across a restart,
# so the finish is counted from the wall clock now, never from elapsed training time.
AB_PAUSE = set(range(3736, 3742))            # the dump was SIGSTOPped for these steps
PRE_PIPE = (3300, 3640)                     # the last 340 steps before the pipeline started (09:17Z)
stepl = step.tolist()
reg = [i for i in range(1, len(S)) if step[i] not in bstep and step[i] - 1 not in bstep]
pace_now = float(np.mean([dt[i] for i in reg if step[i] not in AB_PAUSE][-50:]))
pace_pre = float(np.median([dt[i] for i in reg if PRE_PIPE[0] <= step[i] < PRE_PIPE[1]] or [np.nan]))
pace_paused = float(np.mean([dt[i] for i in reg if step[i] in AB_PAUSE] or [np.nan]))
bsteps = sorted(b for b in bstep if b in stepl)
over = []                                   # what a boundary adds over the regular pace around it
for b in bsteps[-5:]:
    j = stepl.index(b)
    local = [dt[i] for i in reg if 0 < j - i <= 12]
    if local:
        over.append(sum(dt[i] - float(np.median(local)) for i in (j, j + 1) if i < len(S)))
b_over = float(np.median(over)) if over else 0.0
spe = int(np.median(np.diff(bsteps[-4:]))) if len(bsteps) >= 4 else 403
b_left = max(0, (total_steps - 1 - bsteps[-1]) // spe) if bsteps else 0
now_utc = datetime.now(timezone.utc)
launch_utc = datetime.fromisoformat(start_at.replace("Z", "+00:00"))
budget_utc = launch_utc + timedelta(days=7)
# THE PI'S BUDGET DECISION (Project Steering/PI_DECISION_QUEUE.md, 2026-09-26 ~13:35 Berlin): the ~1 h overrun
# the dump causes is ACCEPTED -- "I accept the finish time extension". The page flags the budget again only if
# the projection moves more than REFLAG_H past the finish that was accepted (a monitoring rule, not the PI's).
ACCEPTED_FINISH = datetime(2026, 10, 1, 2, 19, tzinfo=BERLIN)
REFLAG_H = 1.0
# THE PI'S RESPONSE TO THE AMENDMENT 4 FAILURE (PI_DECISION_QUEUE.md, 2026-09-26 ~20:28 Berlin): "do 2 and 3" -- NAVSIM's own
# comfort label in the on-policy labels (label version 3) and selection with NAVSIM v1's own formula (SPEC Amendment 5,
# adopted after its harm guard). The test's outcome is not changed by it: it FAILED, and the page keeps saying so.
A4_RESPONSE = {"decided_local": "2026-09-26 20:28", "quote": "do 2 and 3",
               "levers": ["NAVSIM's own comfort label (label version 3)", "selection with NAVSIM v1's own formula (Amendment 5)"]}
eta_utc = now_utc + timedelta(seconds=remaining * pace_now + b_left * b_over)
# the trainer own summary.json (done=true) replaces the projection once the run has finished
_sj = os.path.join(os.environ.get("REFE_FINAL_DIR", "D:/Projects/TanitAD-artifacts/refe-final-2026-10-01"), "summary.json")
FINISHED = None
if os.path.exists(_sj):
    _s = json.load(open(_sj, encoding="utf-8"))
    if _s.get("done"):
        FINISHED = datetime.fromisoformat(_s["finished_at"].replace("Z", "+00:00"))
        eta_utc = FINISHED
eta_free_utc = now_utc + timedelta(seconds=remaining * pace_paused + b_left * b_over) \
    if np.isfinite(pace_paused) else None
pace = {"pace_now": round(pace_now, 2), "pace_pre_pipeline": round(pace_pre, 2),
        "pace_dump_paused": round(pace_paused, 2) if np.isfinite(pace_paused) else None,
        "boundary_overhead_s": round(b_over, 1), "boundaries_left": int(b_left), "steps_per_epoch": spe,
        "eta_utc": eta_utc.isoformat(), "eta_local": eta_utc.astimezone(BERLIN).strftime("%Y-%m-%d %H:%M"),
        "eta_h_now": round((eta_utc - now_utc).total_seconds() / 3600.0, 1),
        "budget_utc": budget_utc.isoformat(), "budget_local": budget_utc.astimezone(BERLIN).strftime("%Y-%m-%d %H:%M"),
        "inside_budget": bool(eta_utc <= budget_utc),
        "margin_h": round((budget_utc - eta_utc).total_seconds() / 3600.0, 2),
        "days_total": round((eta_utc - launch_utc).total_seconds() / 86400.0, 2),
        "accepted_finish_utc": ACCEPTED_FINISH.astimezone(timezone.utc).isoformat(),
        "reflag_h": REFLAG_H,
        "finished": bool(FINISHED), "finished_local": FINISHED.astimezone(BERLIN).strftime("%a %d %b %H:%M") if FINISHED else None,
        "final_steps": int(_s["steps"]) if FINISHED else None,
        "within_accepted": bool(eta_utc <= ACCEPTED_FINISH + timedelta(hours=REFLAG_H)),
        "past_accepted_h": round((eta_utc - ACCEPTED_FINISH).total_seconds() / 3600.0, 2),
        "eta_dump_free_local": eta_free_utc.astimezone(BERLIN).strftime("%Y-%m-%d %H:%M") if eta_free_utc else None,
        "margin_dump_free_h": round((budget_utc - eta_free_utc).total_seconds() / 3600.0, 2) if eta_free_utc else None}

# evaluation points (the NAVSIM v1 learning curve on W3's 200 tokens). A point is an epoch snapshot `epNNN` -- written
# at the boundary INTO epoch NNN -- or `final`, the model after the last step; the final model is placed one past the
# last epoch the trainer read a bank for, and every chart and sentence names it through `label` / `long`.
FINAL_EPOCH = (max(b["epoch"] for b in banks) + 1) if banks else None


def point_id(tag):
    if tag == "final" and FINAL_EPOCH is not None:
        return {"epoch": FINAL_EPOCH, "label": "final", "long": "the final model"}
    if tag.startswith("ep") and tag[2:].isdigit():
        e = int(tag[2:])
        return {"epoch": e, "label": f"e{e}", "long": f"after epoch {e}"}
    return None


evals = []
for name in sorted(os.listdir(PTS)):
    if not (name.startswith("sub200_") and name.endswith(".json")):
        continue
    d = json.load(open(os.path.join(PTS, name), encoding="utf-8"))
    if d.get("verdict") != "OK":
        continue
    fl = d.get("floors") or {}
    arms = fl.get("arms", {})
    pairs = fl.get("pairs", {})

    def iv(a):
        x = arms.get(a, {}).get("interval") or {}
        lo, hi = x.get("lo", x.get("ci_lo")), x.get("hi", x.get("ci_hi"))
        return [None if lo is None else round(100 * lo, 2), None if hi is None else round(100 * hi, 2)]

    def pr(k):
        p = pairs.get(k)
        if not p:
            return None
        x = p.get("interval") or {}
        return {"delta": p["delta_x100"], "wins": p["wins"], "ties": p["ties"], "losses": p["losses"],
                "lo": None if x.get("lo") is None else round(100 * x["lo"], 2),
                "hi": None if x.get("hi") is None else round(100 * x["hi"], 2),
                "separated": x.get("separated")}

    fam = ((d.get("families") or {}).get("families") or {}).get("REFe") or {}
    lon, lat, tac = fam.get("longitudinal", {}), fam.get("lateral", {}), fam.get("tactical", {})
    prop = (d.get("seam") or {}).get("proposals") or {}
    sm = d["score"]["summary_x100_4dp"]
    tag = name[len("sub200_"):-len(".json")]
    pid = point_id(tag)
    if pid is None:
        continue                      # not a learning-curve point
    evals.append({
        "point": tag, **pid, "pdms": sm["PDMS"],
        "rule": (d.get("seam") or {}).get("rule") or "v2_shape",     # the selection rule of this point's pick
        # SPEC Amendment 7: the executed plan's last-pose heading repaired (absent = before the amendment = False)
        "repair": bool((d.get("seam") or {}).get("repair_last_heading", False)),
        "subscores": {k: sm[k] for k in ("NC", "DAC", "EP", "TTC", "C", "DDC")},
        "pdms_ci": iv("REFe"), "n_tokens": fl.get("n_tokens"), "n_logs": fl.get("n_logs"),
        "estimator": (fl.get("estimator") or {}).get("name"),
        "floors": {a: {"pdms": arms[a].get("PDMS"), "ci": iv(a)} for a in ("STOP", "CV", "HUMAN", "refcv4b_A1") if a in arms},
        "zeroed": arms.get("REFe", {}).get("zeroed_by_NC_or_DAC"),
        "vs_stop": pr("REFe__minus__STOP"), "vs_cv": pr("REFe__minus__CV"),
        "proposals": {k: prop.get(k) for k in ("ade_selected_m", "ade_random_m", "ade_oracle_m",
                                              "endpoint_spread_m", "oracle_index_distinct", "M")},
        "families": {
            "tier": (d.get("families") or {}).get("tier"),
            "speed_mae_mps": lon.get("speed_mae_mps"), "speed_bias_mps": lon.get("speed_bias_mps"),
            "along_bias_m": lon.get("along_bias_m"), "along_mae_m": lon.get("along_mae_m"),
            "progress_ratio_median": (lon.get("ego_progress") or {}).get("progress_ratio_median"),
            "distance_keeping": (lon.get("distance_keeping") or {}).get("status"),
            "heading_mae_deg": lat.get("heading_mae_deg"), "curvature_mae_1pm": lat.get("curvature_mae_1pm"),
            "cross_mae_m": lat.get("cross_mae_m"),
            "goal_point_error_m": (tac.get("goal_setting") or {}).get("goal_point_error_m"),
            "tactical_status": tac.get("status"),
            "strategic": ((fam.get("strategic") or {}).get("status")),
        },
    })

# SPEC E-6 (AMENDMENT 3): the selection diagnosis -- every one of the 64 proposals scored by the same
# harness -- one readout per snapshot that has one (eval/proposal_table.py + eval/selection_readout.py)
selection = []
PT = os.path.join(os.path.dirname(PTS), "proptable")
# the earlier picks re-selected with NAVSIM v1's rule from their own E-6 tables (eval/rule_mismatch_diag.py, EXPLORATORY)
# and SPEC Amendment 5's readout (the harm guard that made v1 the rule)
_rd = os.path.join(PT, "rule_mismatch_diag.json")
RULE_DIAG = json.load(open(_rd, encoding="utf-8")) if os.path.exists(_rd) else {}
# SPEC Amendment 7's confirmation readout (the last-pose heading repair), for the page's chip and finding
_a7 = os.path.join(os.environ.get("REFE_PKG", "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & "
                                  "Inference/Research/2026-09-20-refe-plan"), "eval", "raw", "a7_confirm", "a7_confirm_ep015.json")
A7 = None
if os.path.exists(_a7):
    _j = json.load(open(_a7, encoding="utf-8"))
    A7 = {"decision": _j["decision"], "N": _j["tokens"]["n"], "n_logs": _j["tokens"]["n_logs"],
          "shipped": _j["statistic"]["shipped_pdms"], "repaired": _j["statistic"]["repaired_pdms"],
          "delta": _j["statistic"]["delta"], "ci95": _j["statistic"]["ci95"],
          "helped": _j["statistic"]["helped"], "hurt": _j["statistic"]["hurt"],
          "sub_delta": _j["reported_not_gating"]["subscore_deltas_x100_repaired_minus_shipped"]}
# the like-for-like readout of the first repaired step (after epoch 15 -> 16, BOTH repaired; eval/like_for_like_016.py)
_lfl = os.path.join(os.environ.get("REFE_PKG", "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & "
                                   "Inference/Research/2026-09-20-refe-plan"), "eval", "raw", "like_for_like_016")
LFL = None
if os.path.exists(os.path.join(_lfl, "like_for_like_016.json")):
    _c = json.load(open(os.path.join(_lfl, "like_for_like_016.json"), encoding="utf-8"))["statistic"]["comparisons"]
    _t, _r = _c["016_repaired_minus_015_repaired"], _c["015_repaired_minus_015_shipped"]
    _k = json.load(open(os.path.join(_lfl, "like_for_like_016_64.json"), encoding="utf-8"))["trend"]["result"]["015_repaired:016"]
    LFL = {"train": {k: _t[k] for k in ("delta", "ci95", "better", "worse", "tied", "separated")},
           "sub": _t.get("subscore_deltas_x100") or _t.get("sub_deltas_x100") or {},
           "repair": {k: _r[k] for k in ("delta", "ci95")},
           "best64": {k: _k["best_of_64"][k] for k in ("a", "b", "b_minus_a", "ci95")}}
# consecutive snapshots compared under ONE rule, paired on the same tokens (eval/snapshot_pair_under_rule.py, EXPLORATORY):
# from Amendment 5 the pick's rule changed, so the curve's step from the last v2-picked point mixes model and rule
import glob as _glob
PAIRS = [json.load(open(f, encoding="utf-8"))
         for f in sorted(_glob.glob(os.path.join(os.environ.get("REFE_PKG", "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"), "eval", "raw", "e6_sub200_*", "pair_*.json")))]
_a5 = os.path.join(PT, "a5confirm_ep013", "a5_readout.json")
A5 = json.load(open(_a5, encoding="utf-8")) if os.path.exists(_a5) else None
if os.path.isdir(PT):
    for name in sorted(os.listdir(PT)):
        rp, gp = os.path.join(PT, name, "readout.json"), os.path.join(PT, name, "gates.json")
        pid = point_id(name[len("sub200_"):]) if name.startswith("sub200_") else None
        if pid is not None and os.path.exists(rp) and os.path.exists(gp):
            r = json.load(open(rp, encoding="utf-8"))
            r.update(pid)
            # SPEC Amendment 4 (the on-policy scorer's pre-registered test), where it was read
            a4p = os.path.join(PT, name, "amendment4.json")
            r["a4"] = json.load(open(a4p, encoding="utf-8")) if os.path.exists(a4p) else None
            # the selection RULE the pick was made with (tables before SPEC Amendment 5 carry none: the v2 shape)
            r["rule"] = r.get("rule") or "v2_shape"
            r["v1"] = (RULE_DIAG.get(name) or {}).get("NAVSIM v1 formula") if r["rule"] == "v2_shape" else None
            r["gates"] = json.load(open(gp, encoding="utf-8")).get("gates")
            # descriptive, from the table itself (not pre-registered): how many proposals would do well
            T = np.load(os.path.join(PT, name, "table.npz"))
            tp, ts, tk = T["pdms"], T["sub"], T["pick"]
            ar_, good = np.arange(tp.shape[0]), (T["pdms"] >= 0.8).sum(1)
            r["desc"] = {"good_mean": round(float(good.mean()), 1), "good_median": float(np.median(good)),
                         "tok_ge1_good": round(float((good >= 1).mean()), 4),
                         "pick_zero": round(float((tp[ar_, tk] == 0).mean()), 4),
                         "all_zero": round(float((tp.max(1) == 0).mean()), 4),
                         "dac_any": round(float((ts[:, :, 1].max(1) == 1).mean()), 4),
                         "pick_dac_fail": round(float((ts[ar_, tk, 1] == 0).mean()), 4)}
            selection.append(r)
selection.sort(key=lambda r: r["epoch"])

# the bank ASSEMBLER's own log: every pass reports the scorer frames it appended (sc0 = rank 0,
# sc1 = augmented). Their sum is what the bank holds, whether or not training has read it yet.
asm = {"passes": 0, "sc0": 0, "sc1": 0, "last_local": None, "last_sc_new": 0}
ap = os.path.join(HERE, "assembler.log")
if os.path.exists(ap):
    for line in open(ap, encoding="utf-8"):
        if line.startswith("ZZASSEMBLE_OK"):
            parts = line.split(" ", 2)
            j = json.loads(parts[2].rsplit("}", 1)[0] + "}")
            asm["passes"] += 1
            asm["sc0"] += j.get("sc0_frames_new", 0)
            asm["sc1"] += j.get("sc1_frames_new", 0)
            asm["last_local"] = loc(parts[1])
            asm["last_sc_new"] = j.get("sc0_frames_new", 0) + j.get("sc1_frames_new", 0)
asm["scorer_frames"] = asm["sc0"] + asm["sc1"]

out = {
    "generated_local": datetime.now(BERLIN).strftime("%Y-%m-%d %H:%M"),
    "assembler": asm,
    "run": {"name": "vitl16_navtrain10_grow_tau0.3", "start_local": loc(start_at),
            "argv": cfg.get("args", {}), "params": cfg.get("params"), "torch": cfg.get("torch"),
            "device": cfg.get("device")},
    "progress": {"last_step": int(step[last_i]), "total_steps": total_steps,
                 "elapsed_h": round(float(el[last_i] / 3600.0), 2),
                 "avg_s_per_step_all": round(avg_all, 2),
                 "steady_s_per_step_median": round(float(np.median(steady)), 2),
                 "steady_p10_p90": [round(float(np.percentile(steady, 10)), 2), round(float(np.percentile(steady, 90)), 2)],
                 "eta_h": round(eta_h, 1), "epoch_now": int(ep[last_i]),
                 # the run's LAST epoch index: grow mode made epochs 0-3 short, so 10,075 steps hold more than the
                 # 25 full-scene epochs the plan counted; the last one ends part-way at step 10,075
                 "epoch_last": int(ep[last_i]) + int(b_left),
                 "traj_L1_first": round(float(l1[0]), 3), "traj_L1_med_now": round(float(rolling_median(l1)[last_i]), 4),
                 "score_med_now": round(float(rolling_median(sc)[last_i]), 5),
                 "scorer_cov_cum_now": round(float(cov[last_i]), 4),
                 "scorer_cov_win_now": round(float(win_cov[last_i]), 4) if np.isfinite(win_cov[last_i]) else None,
                 "winners_mean_now": round(float(rolling_mean(win)[last_i]), 3),
                 "mem_gb_max": round(float(np.nanmax(mem)), 2), "n_ckpts": len(ckpts), **pace},
    "series": series, "per_epoch": per_epoch, "bank_events": bank_events, "epoch_events": epoch_events,
    "evals": evals, "selection": selection, "a5": A5, "a4_response": A4_RESPONSE, "pairs": PAIRS, "a7": A7, "lfl": LFL,
    # the trainer's own on-policy windows since the switch, pooled (training side; the navtest test is E-6/A4)
    "onpolicy_train": {"switch_step": switch_step, "bin": OP_BIN, "bins": op_bins, "total": op_pool(sorted(opw))},
    # the on-policy scorer pipeline's status (PI decision B), fetched from the pod beside the logs
    "onpolicy": (json.load(open(os.path.join(HERE, "onpolicy.json"), encoding="utf-8"))
                 if os.path.exists(os.path.join(HERE, "onpolicy.json")) else None),
}
if out["onpolicy"] and out["onpolicy"].get("switch_at_utc"):
    out["onpolicy"]["switch_at"] = loc(out["onpolicy"]["switch_at_utc"])     # Berlin, like every time here
# a full epoch is 402.5 optimiser steps on average (103,037 scenes / 4 per micro-batch = 25,760 micro-batches, / 64 per
# step), so boundaries alternate 403 / 402 steps apart: project them at the MEAN of the last full epochs, not at `spe`
# (the integer median, 402), which falls one step further behind every second epoch
spe_f = float(np.mean(np.diff(bsteps[-9:]))) if len(bsteps) >= 3 else float(spe)
# ⭐ LABEL VERSION 3 (PI option 2, 2026-09-26): from then on the comfort target is NAVSIM's own ego_is_comfortable.
# The labeller changed, not the trainer, so no `declared_change` line marks it: training meets the new labels at the
# first epoch boundary whose bank was read AFTER the first live version-3 label was written (op_status.py).
OPS_ = out["onpolicy"]
if OPS_ and OPS_.get("v3_first_live_at"):
    OPS_["v3_first_live"] = loc(OPS_["v3_first_live_at"])
    v3_after = sorted(b["epoch"] for b in banks
                      if b.get("scorer_mode") == "onpolicy" and b.get("at", "") >= OPS_["v3_first_live_at"])
    OPS_["v3_read"] = bool(v3_after)
    OPS_["v3_epoch"] = v3_after[0] if v3_after else int(ep[last_i]) + 1
    OPS_["v3_step"] = next((e["step"] for e in epochs if e["epoch"] == OPS_["v3_epoch"]),
                           int(bsteps[-1]) + int(math.floor(spe_f + 0.5)))

# ⭐ WHAT COMES NEXT, at the pace now (the page's "Coming up" table): each epoch boundary writes a snapshot that the
# dev box evaluates as soon as the file appears (eval/wait_and_eval.sh); the final model is written at the last step.
now_step = int(step[last_i])
v1_done = any(e.get("rule") == "navsim_v1" for e in out["evals"])
milestones = []
for k in range(1, int(b_left) + 1):
    e_k, s_k = int(ep[last_i]) + k, int(bsteps[-1]) + int(math.floor(k * spe_f + 0.5))
    milestones.append({
        "kind": "snapshot", "epoch": e_k, "step": s_k,
        "at_utc": (now_utc + timedelta(seconds=(s_k - now_step) * pace_now + (k - 1) * b_over)).isoformat(),
        "first_v1": (not v1_done) and k == 1,
        "v3_starts": bool(OPS_ and OPS_.get("v3_epoch") == e_k and not OPS_.get("v3_read")),
        "first_v3_trained": bool(OPS_ and OPS_.get("v3_epoch") is not None and OPS_["v3_epoch"] + 1 == e_k)})
milestones.append({"kind": "final", "epoch": None, "step": total_steps, "at_utc": eta_utc.isoformat()})
out["milestones"] = milestones
out["progress"]["steps_per_epoch_mean"] = round(spe_f, 2)
# ⭐ THE FULL NAVTEST of the final model (12,146 tokens), once its point and readout are banked (2026-10-04)
_pkg = os.environ.get("REFE_PKG", "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
_ntp, _ntr = os.path.join(PTS, "navtest_final.json"), os.path.join(_pkg, "raw", "2026-10-04-navtest-final", "navtest_final_readout.json")
out["navtest_full"] = None
if os.path.exists(_ntp) and os.path.exists(_ntr):
    _p, _r = json.load(open(_ntp, encoding="utf-8")), json.load(open(_ntr, encoding="utf-8"))
    _pairs = (_p.get("floors") or {}).get("pairs", {})

    def _pair(k):           # parse_navtest6: delta_x100 + interval {lo, hi} as fractions + wins / ties / losses
        x = next((v for kk, v in _pairs.items() if kk.endswith("__minus__" + k)), None) or {}
        iv = x.get("interval") or {}
        return {"d": x.get("delta_x100"), "lo": None if iv.get("lo") is None else round(100 * iv["lo"], 2),
                "hi": None if iv.get("hi") is None else round(100 * iv["hi"], 2),
                "wtl": [x.get("wins"), x.get("ties"), x.get("losses")]}
    out["navtest_full"] = {
        "all": _r["all_12146"], "clean": _r["frame_control_clean_12110"], "on_route": _r["on_route"],
        "off_route": _r["off_route_gt20m"], "by_command": _r["by_command"], "zero": _r["pdms_zero"],
        "scorer": _r.get("scorer_on_pick"), "off_cost": _r["off_route_cost_to_full_mean"],
        "floors": {k: (_p["floors"]["arms"].get(k) or {}).get("PDMS") for k in ("STOP", "CV", "HUMAN", "refcv4b_A1")},
        "pairs": {k: _pair(k) for k in ("STOP", "CV", "HUMAN", "refcv4b_A1")}, "scored": _p.get("rescored")}
    # the EXACT full-navtest read of the adopted route fix (raw/2026-10-04-navtest-final/routefix_full/3_parse.json)
    _rf = os.path.join(_pkg, "raw", "2026-10-04-navtest-final", "routefix_full", "3_parse.json")
    if os.path.exists(_rf):
        _q = json.load(open(_rf, encoding="utf-8"))
        _a, _iv = _q["arms"]["R6_A1"], _q["arms"]["R6_A1"]["interval"]
        _pp = _q["pairs"]["R6_A1__minus__R6_final_off"]
        out["navtest_full"]["routefix"] = {
            "PDMS": _a["PDMS"], "lo": round(100 * _iv["lo"], 2), "hi": round(100 * _iv["hi"], 2),
            "d": _pp["delta_x100"], "d_lo": round(100 * _pp["interval"]["lo"], 2), "d_hi": round(100 * _pp["interval"]["hi"], 2),
            "wtl": [_pp["wins"], _pp["ties"], _pp["losses"]], "zero": _a["zeroed_by_NC_or_DAC"],
            "sub": {k: _a[k] for k in ("NC", "DAC", "EP", "TTC", "C", "DDC")}}
    # SPEC_NAVTEST Amendment 9 (the off-route nav goal), when its registered analysis exists
    _a9p = os.path.join(_pkg, "raw", "2026-10-01-goal-trigger", "a9", "result_a9.json")
    if os.path.exists(_a9p):
        _a9 = json.load(open(_a9p, encoding="utf-8"))["arms"]
        out["navtest_full"]["a9"] = {a: {"verdict": v.get("verdict"), "n": v["reads"]["PRIMARY"]["n_tokens"],
                                         "logs": v["reads"]["PRIMARY"]["n_logs"], "d": v["reads"]["PRIMARY"]["D_mean"],
                                         "ci": v["reads"]["PRIMARY"]["ci95"], "full": v["full_navtest"]["delta_full_navtest_pdms"]}
                                     for a, v in _a9.items() if "reads" in v}
# ⭐ SFT-1, the scorer-only fine-tune (eval/PREREG_SFT1.md): the pod's sft.jsonl, pulled next to this script
_sft = os.environ.get("REFE_SFT_JSONL", os.path.join(HERE, "sft.jsonl"))
out["sft1"] = None
if os.path.exists(_sft):
    _tr, _ev, _meta = [], [], {}
    for _ln in open(_sft, encoding="utf-8"):
        try:
            _r = json.loads(_ln)
        except json.JSONDecodeError:
            continue
        _e = _r.get("event")
        if _e == "train":
            _tr.append({k: _r.get(k) for k in ("update", "of", "bceA", "lossB", "listnet", "s_per_update", "at")})
        elif _e == "eval":
            _ev.append(_r)
        elif _e in ("data", "start", "done"):
            _meta[_e] = _r
    out["sft1"] = {"train": _tr, "evals": _ev, "meta": _meta}
    _sv = os.path.join(_pkg, "raw", "2026-10-04-sft1", "verdict.json")   # the registered verdict, once there is one
    if os.path.exists(_sv):
        out["sft1"]["verdict"] = json.load(open(_sv, encoding="utf-8"))
# SFT-3 / SFT-4 (eval/PREREG_SFT3.md, PREREG_SFT4.md): the pod's sft.log of each run, pulled next to this script
def _sft_events(path):
    if not os.path.exists(path):
        return None
    tr, ev, meta = [], [], {}
    for ln in open(path, encoding="utf-8", errors="replace"):
        if not ln.startswith("{"):
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        e = r.get("event")
        if e == "train":
            tr.append({k: r.get(k) for k in ("update", "of", "bceA", "lossB", "listnet", "s_per_update", "at")})
        elif e == "eval":
            ev.append(r)
        elif e in ("data", "start", "done"):
            meta[e] = r
    return {"train": tr, "evals": ev, "meta": meta}


out["sft3"] = _sft_events(os.environ.get("REFE_SFT3_LOG", os.path.join(HERE, "sft3.log")))
out["sft4"] = _sft_events(os.environ.get("REFE_SFT4_LOG", os.path.join(HERE, "sft4.log")))
for _k, _d in (("sft3", "2026-10-04-sft3"), ("sft4", "2026-10-05-sft4")):   # registered verdicts, once they exist
    _vp = os.path.join(_pkg, "raw", _d, "verdict.json")
    if out[_k] is not None and os.path.exists(_vp):
        out[_k]["verdict"] = json.load(open(_vp, encoding="utf-8"))
# ⭐ LANE-1, lane discipline (raw/2026-10-04-lane-discipline/PREREG_LANE1.md): the registered analysis, when it exists
_ld = os.path.join(_pkg, "raw", "2026-10-04-lane-discipline")
out["lane1"] = None
for _nm in ("lane_result.json", "lane_result_partial.json"):
    if os.path.exists(os.path.join(_ld, _nm)):
        out["lane1"] = json.load(open(os.path.join(_ld, _nm), encoding="utf-8"))
        break
if out["lane1"] is not None and os.path.exists(os.path.join(_ld, "lane2_result.json")):
    out["lane1"]["lane2"] = json.load(open(os.path.join(_ld, "lane2_result.json"), encoding="utf-8"))
for _k, _nm in (("teacher_check", "teacher_check_heldout_summary.json"), ("teacher_ddc", "teacher_vs_navsim_ddc_heldout.json"),
                ("ddc_valid", "validate_navsim_ddc.json"), ("label_census", "teacher_label_census.json")):
    if out["lane1"] is not None and os.path.exists(os.path.join(_ld, _nm)):
        out["lane1"][_k] = json.load(open(os.path.join(_ld, _nm), encoding="utf-8"))
_scene = os.path.join(_ld, "scene_01949_02501_plans20-25_dir.png")
out["lane_scene"] = None
if out["lane1"] is not None and os.path.exists(_scene):
    import base64
    out["lane_scene"] = "data:image/png;base64," + base64.b64encode(open(_scene, "rb").read()).decode("ascii")
json.dump(out, open(os.path.join(HERE, "report_data.json"), "w", encoding="utf-8"))
p = out["progress"]
print(json.dumps(p, indent=1))
print("start", out["run"]["start_local"], "epochs", len(epoch_events), "banks", len(bank_events), "evals", [e["point"] for e in evals])
for e in per_epoch:
    print(e)
print(os.path.getsize(os.path.join(HERE, "report_data.json")), "bytes")
