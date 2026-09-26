"""Collect the REFe training-progress data into one JSON for the report page (every number from an artifact)."""
import json
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PTS = "D:/Projects/TanitAD/data/refe_navtest/points"
BERLIN = timezone(timedelta(hours=2))


def loc(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(BERLIN).strftime("%Y-%m-%d %H:%M")


steps, banks, epochs, ckpts = [], [], [], []
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

r = lambda a, n=4: [None if (a_ is None or not np.isfinite(a_)) else round(float(a_), n) for a_ in a]
series = {
    "step": step.tolist(), "elapsed_h": r(el / 3600.0, 4), "traj_L1": r(l1, 4),
    "traj_L1_med": r(rolling_median(l1), 4), "score": r(sc, 5), "score_med": r(rolling_median(sc), 5),
    "winners": win.astype(int).tolist(), "winners_mean": r(rolling_mean(win), 3),
    "assign_d": r(ad, 3), "assign_d_med": r(rolling_median(ad), 3),
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
    per_epoch.append({
        "epoch": int(e), "steps": int(m.sum()), "first_step": int(step[idx[0]]), "last_step": int(step[idx[-1]]),
        "traj_L1_median": round(float(np.nanmedian(l1[m])), 4),
        "score_median": (round(float(np.nanmedian(sc[m][sc[m] > 0])), 5) if np.any(sc[m] > 0) else None),
        "winners_mean": round(float(np.nanmean(win[m])), 3),
        "assign_d_median": (round(float(np.nanmedian(ad[m])), 3) if np.any(np.isfinite(ad[m])) else None),
        "scorer_cov_window": (round(float((ncov[idx[-1]] - (ncov[idx[0] - 1] if idx[0] > 0 else 0))
                                          / max(smp[idx[-1]] - (smp[idx[0] - 1] if idx[0] > 0 else 0), 1)), 4)),
        "s_per_step_median": round(float(np.nanmedian(dts)), 2),
        "bank_tuples": bank["n_tuples"] if bank else None, "bank_scenes": bank["n_scenes"] if bank else None,
        "bank_rank1": (bank["per_rank"].get("1") if bank else None),
        "bank_scorer_frames": bank["scorer_frames"] if bank else None,
    })

bank_events = [{"epoch": b["epoch"], "tuples": b["n_tuples"], "scenes": b["n_scenes"],
                "rank0": b["per_rank"].get("0"), "rank1": b["per_rank"].get("1", 0),
                "scorer_frames": b["scorer_frames"], "cov_norm": round(b["cov_norm"], 4),
                "at_local": loc(b["at"])} for b in banks]
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

# evaluation points (the NAVSIM v1 learning curve on W3's 200 tokens)
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
    evals.append({
        "point": tag, "epoch": int(tag.replace("ep", "")), "pdms": sm["PDMS"],
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
if os.path.isdir(PT):
    for name in sorted(os.listdir(PT)):
        rp, gp = os.path.join(PT, name, "readout.json"), os.path.join(PT, name, "gates.json")
        if name.startswith("sub200_ep") and os.path.exists(rp) and os.path.exists(gp):
            r = json.load(open(rp, encoding="utf-8"))
            r["epoch"] = int(name[len("sub200_ep"):])
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
                 "traj_L1_first": round(float(l1[0]), 3), "traj_L1_med_now": round(float(rolling_median(l1)[last_i]), 4),
                 "score_med_now": round(float(rolling_median(sc)[last_i]), 5),
                 "scorer_cov_cum_now": round(float(cov[last_i]), 4),
                 "scorer_cov_win_now": round(float(win_cov[last_i]), 4) if np.isfinite(win_cov[last_i]) else None,
                 "winners_mean_now": round(float(rolling_mean(win)[last_i]), 3),
                 "mem_gb_max": round(float(np.nanmax(mem)), 2), "n_ckpts": len(ckpts)},
    "series": series, "per_epoch": per_epoch, "bank_events": bank_events, "epoch_events": epoch_events,
    "evals": evals, "selection": selection,
}
json.dump(out, open(os.path.join(HERE, "report_data.json"), "w", encoding="utf-8"))
p = out["progress"]
print(json.dumps(p, indent=1))
print("start", out["run"]["start_local"], "epochs", len(epoch_events), "banks", len(bank_events), "evals", [e["point"] for e in evals])
for e in per_epoch:
    print(e)
print(os.path.getsize(os.path.join(HERE, "report_data.json")), "bytes")
