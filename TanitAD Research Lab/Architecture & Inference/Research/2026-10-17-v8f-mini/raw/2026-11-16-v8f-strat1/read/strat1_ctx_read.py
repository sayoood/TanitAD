"""strat1_ctx_read.py -- STRATEGIC detected information (PREREG_V8F_STRAT1 section 4.5, READ_PLAN 2.5).  Tier T0 (a per-frame perception output at the window's NOW frame).
Evidence class: MEASURED (ours; T0 evals, zero training).  Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2,000, seed 0 over the 141 eval episodes ("another draw of EPISODES");
the second training seed is read beside, a variable counts only when it is DETECTS at both seeds.
Inputs: S1's eval_STRAT1-S1_s{0,1}_real.npz `ctx_logits` [N, 41] split by CTX_N in CTX_VARS order, the K4 pass eval_STRAT1-S1_s{0,1}_k4.npz (ctx_head re-initialised, seed 0), labels from
strat_labels_table.npz at the table row eval_off[ci] + t, platform per clip from strat_platform_by_clip.npz.  Labels are masked to -1 where flag != 0, <var>_valid is False, and for
illumination class 'twilight' (CTX_TWILIGHT_ID 3; 240 steps / 1 clip, reported n/a with n).  C0 and N1 have no head: n/a, reason "no output".
Outcomes per variable: DETECTS(v) = accuracy - constant > 0 separated at both seeds AND K4 not detecting; country also needs DETECTS-BEYOND-RIG (accuracy - rig baseline > 0 separated);
NOT-DETECTED(v) otherwise (next: the frozen-adapter probe, then the representation lever).  If K4 reads DETECTS on ANY variable the reader prints INSTRUMENT-FAIL and quotes no detected-info number.
Controls: K2 constant predictor reads exactly the constant baseline; K3 the labels themselves read 1.0 and DETECT; K4 above; K5 clip-shuffled labels (seed 0) read NOT DETECTS; K6 the rig-only
country predictor reads exactly the rig baseline and NOT DETECTS-BEYOND-RIG.  Constant baselines are recomputed from the table and compared with LABELS_REPORT (country 0.092, weather 0.846,
surface 0.785, illumination 0.595, period of day 0.241; I): a mismatch beyond 0.0015 is printed as a control problem.
Usage:  python strat1_ctx_read.py --t0 DIR --out OUT.json [--grid GRID.json] [--smoke]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strat1_lib as L  # noqa: E402

CTX_VARS = ("country", "weather", "surface", "illumination", "period_of_day")
CTX_N = (25, 4, 3, 4, 5)
TWILIGHT_ID = 3
LABELS_REPORT = {"country": 0.092, "weather": 0.846, "surface": 0.785, "illumination": 0.595, "period_of_day": 0.241}
TOL_REPORT = 0.0015
ILL = CTX_VARS.index("illumination")


# ------------------------------------------------------------------------------------------------------------------------------ labels
def label_rows(tab, r, mask_flag=True):
    """y [N, 5] int64 at table rows r, -1 where flag != 0, the variable's valid bit is False, or illumination is twilight (the card's mask).  mask_flag=False is the MUTATION target."""
    y = np.full((len(r), 5), -1, np.int64)
    fl = np.asarray(tab["eval_flag"])[r]
    for k, v in enumerate(CTX_VARS):
        ids = np.asarray(tab[f"eval_{v}_id"], np.int64)[r]
        ok = np.asarray(tab[f"eval_{v}_valid"], bool)[r]
        if mask_flag:
            ok = ok & (fl == 0)
        if k == ILL:
            ok = ok & (ids != TWILIGHT_ID)
        y[ok, k] = ids[ok]
    return y


def table_baselines(tab):
    """per variable: the constant (majority) class and its accuracy over ALL masked eval steps of the table (what LABELS_REPORT counts), plus the twilight step / clip count"""
    r = np.arange(len(tab["eval_flag"]))
    y = label_rows(tab, r)
    out = {}
    for k, v in enumerate(CTX_VARS):
        yy = y[y[:, k] >= 0, k]
        cnt = np.bincount(yy, minlength=CTX_N[k])
        out[v] = {"constant_class": int(cnt.argmax()), "constant_acc_steps": float(cnt.max() / cnt.sum()), "n_steps": int(cnt.sum())}
    ids = np.asarray(tab["eval_illumination_id"], np.int64)
    tw_n = int(((ids == TWILIGHT_ID) & np.asarray(tab["eval_illumination_valid"], bool) & (np.asarray(tab["eval_flag"]) == 0)).sum())
    for v in CTX_VARS:                         # LABELS_REPORT's illumination 0.595 counts the twilight steps in the denominator (MEASURED: 0.5947); the masked baseline is the deciding one
        out[v]["report_basis_acc"] = out[v]["constant_acc_steps"] * out[v]["n_steps"] / (out[v]["n_steps"] + (tw_n if v == "illumination" else 0))
    tw = (ids == TWILIGHT_ID) & np.asarray(tab["eval_illumination_valid"], bool) & (np.asarray(tab["eval_flag"]) == 0)
    clip_of_row = np.repeat(np.arange(len(tab["eval_off"])), np.asarray(tab["eval_n"]))
    out["twilight"] = {"n_steps": int(tw.sum()), "n_clips": int(len(np.unique(clip_of_row[tw])))}
    return out


def rig_baseline(tab, plat, ci):
    """country predicted by the rig alone: the most common TRAIN country per platform class (train clip country = its modal masked step label), applied to the eval windows' clips.  -> (pred [N], map)"""
    yt = np.full((len(tab["train_flag"]), 1), -1, np.int64)
    ok = np.asarray(tab["train_country_valid"], bool) & (np.asarray(tab["train_flag"]) == 0)
    ids = np.asarray(tab["train_country_id"], np.int64)
    clip_of_row = np.repeat(np.arange(len(tab["train_off"])), np.asarray(tab["train_n"]))
    clip_country = np.full(len(tab["train_off"]), -1, np.int64)
    for c in range(len(tab["train_off"])):
        m = (clip_of_row == c) & ok
        if m.any():
            clip_country[c] = np.bincount(ids[m], minlength=CTX_N[0]).argmax()
    tp = np.asarray(plat["train_platform_id"], np.int64)
    rig = {}
    for p in np.unique(tp):
        cc = clip_country[(tp == p) & (clip_country >= 0)]
        rig[int(p)] = int(np.bincount(cc, minlength=CTX_N[0]).argmax())
    ep = np.asarray(plat["eval_platform_id"], np.int64)
    return np.asarray([rig[int(ep[c])] for c in ci], np.int64), rig


# ------------------------------------------------------------------------------------------------------------------------------ per-variable rows
def split_logits(lg):
    lg = np.asarray(lg)
    assert lg.shape[1] == sum(CTX_N), lg.shape
    cuts = np.cumsum((0,) + CTX_N)
    return [lg[:, cuts[k]:cuts[k + 1]] for k in range(5)]


def var_rows(CI, pred, y, eid, const_cls, n_class, nb=L.NB, rig_pred=None, plat_w=None):
    """accuracy, balanced accuracy, per-class recall, confusion, constant, paired (model correct - constant correct) over the scorable windows; for country also the rig contrast and strata"""
    m = y >= 0
    yy, pp, ee = y[m], pred[m], eid[m]
    correct = (pp == yy).astype(np.float64)
    const_correct = (yy == const_cls).astype(np.float64)
    cm = np.zeros((n_class, n_class), np.int64)
    np.add.at(cm, (yy, pp), 1)
    rec = [float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None for i in range(n_class)]
    got = [r for r in rec if r is not None]
    out = {"n_windows": int(m.sum()), "n_episodes": int(len(np.unique(ee))), "accuracy": float(correct.mean()), "balanced_accuracy": float(np.mean(got)) if got else None,
           "per_class_recall": rec, "confusion": cm.tolist(), "constant_class": int(const_cls), "constant_accuracy": float(const_correct.mean()),
           "model_minus_constant": L.paired(CI, correct, const_correct, ee, nb)}
    if rig_pred is not None:
        rc = (rig_pred[m] == yy).astype(np.float64)
        out["rig_accuracy"] = float(rc.mean())
        out["model_minus_rig"] = L.paired(CI, correct, rc, ee, nb)
        strata = {}
        for p in np.unique(plat_w[m]):
            s = plat_w[m] == p
            maj = int(np.bincount(yy[s], minlength=n_class).argmax())
            strata[int(p)] = {"n_windows": int(s.sum()), "n_episodes": int(len(np.unique(ee[s]))), "accuracy": float(correct[s].mean()), "constant_accuracy": float((yy[s] == maj).mean())}
        out["by_platform"] = strata
    return out


def detects(row):
    r = row["model_minus_constant"]
    return bool(r.get("status") == "OK" and r["separated"] and r["delta"] > 0)


def detects_rig(row):
    r = row.get("model_minus_rig")
    return bool(r and r.get("status") == "OK" and r["separated"] and r["delta"] > 0)


def read_arm(CI, logits, y, eid, consts, rig_pred=None, plat_w=None, nb=L.NB):
    parts = split_logits(logits)
    rows = {}
    for k, v in enumerate(CTX_VARS):
        rows[v] = var_rows(CI, parts[k].argmax(1), y[:, k], eid, consts[v]["constant_class"], CTX_N[k], nb, rig_pred if k == 0 else None, plat_w if k == 0 else None)
        rows[v]["detects"] = detects(rows[v])
        if k == 0:
            rows[v]["detects_beyond_rig"] = detects_rig(rows[v])
    return rows


def onehot_logits(y):
    """logits that predict y exactly (K3); -1 labels get class 0"""
    out = []
    for k in range(5):
        z = np.full((len(y), CTX_N[k]), -9.0)
        z[np.arange(len(y)), np.maximum(y[:, k], 0)] = 9.0
        out.append(z)
    return np.concatenate(out, 1)


def const_logits(consts, n):
    out = []
    for k, v in enumerate(CTX_VARS):
        z = np.full((n, CTX_N[k]), -9.0)
        z[:, consts[v]["constant_class"]] = 9.0
        out.append(z)
    return np.concatenate(out, 1)


def clip_shuffle(y, eid, seed=0):
    """K5: windows of episode e take the labels of window (k mod n) of episode perm[e], a seed-0 derangement-like permutation of the episodes"""
    u = np.unique(eid)
    perm = np.random.RandomState(seed).permutation(len(u))
    if len(u) > 1 and (perm == np.arange(len(u))).any():
        perm = np.roll(np.arange(len(u)), 1)
    out = np.full_like(y, -1)
    for i, e in enumerate(u):
        dst = np.where(eid == e)[0]
        src = np.where(eid == u[perm[i]])[0]
        out[dst] = y[src[np.arange(len(dst)) % len(src)]]
    return out


def controls(CI, y, eid, consts, rig_pred, plat_w, nb=300):
    """K2, K3, K5, K6 on the real labels / windows.  Each returns the value it must read next to the value it read."""
    n = len(y)
    r2 = read_arm(CI, const_logits(consts, n), y, eid, consts, rig_pred, plat_w, nb)
    r3 = read_arm(CI, onehot_logits(y), y, eid, consts, rig_pred, plat_w, nb)
    ys = clip_shuffle(y, eid, 0)
    # K5: S1-like predictions are scored against the shuffled labels; the planted model here is the labels themselves, which must now NOT detect
    r5 = read_arm(CI, onehot_logits(y), ys, eid, consts, rig_pred, plat_w, nb)
    rig_logits = np.zeros((n, sum(CTX_N)))
    rig_logits[np.arange(n), rig_pred] = 9.0
    parts = [rig_logits[:, :25]]
    r6 = var_rows(CI, rig_pred, y[:, 0], eid, consts["country"]["constant_class"], 25, nb, rig_pred, plat_w)
    return {"K2": {v: {"read": r2[v]["accuracy"], "must_read": r2[v]["constant_accuracy"], "pass": r2[v]["accuracy"] == r2[v]["constant_accuracy"]} for v in CTX_VARS},
            "K3": {v: {"read": r3[v]["accuracy"], "detects": r3[v]["detects"], "pass": bool(r3[v]["accuracy"] == 1.0 and (r3[v]["detects"] or r3[v]["constant_accuracy"] == 1.0))} for v in CTX_VARS},
            "K5": {v: {"detects": r5[v]["detects"], "pass": not r5[v]["detects"]} for v in CTX_VARS},
            "K6": {"read": r6["accuracy"], "must_read": r6["rig_accuracy"], "detects_beyond_rig": detects_rig(r6), "pass": bool(r6["accuracy"] == r6["rig_accuracy"] and not detects_rig(r6))}}


def decide(per_seed, k4_detects):
    """per_seed {seed: {var: row}}; k4_detects {var: bool (any seed)} -> {var: label}.  INSTRUMENT-FAIL when K4 detects on any variable."""
    if any(k4_detects.values()):
        return {"__all__": "INSTRUMENT-FAIL", "reason": "K4 (C-RAND) reads DETECTS on " + ", ".join(v for v, d in k4_detects.items() if d) + ": no detected-info number is quoted"}
    out = {}
    for v in CTX_VARS:
        rows = [per_seed[s][v] for s in per_seed]
        if len(rows) < 2:
            out[v] = "NOT-YET"
        elif all(r["detects"] for r in rows):
            if v == "country":
                out[v] = "DETECTS(country) + DETECTS-BEYOND-RIG" if all(r["detects_beyond_rig"] for r in rows) else "DETECTS(country), rig-level (not beyond the rig)"
            else:
                out[v] = f"DETECTS({v})"
        else:
            out[v] = f"NOT-DETECTED({v})"
    return out


NEXT_NOT_DETECTED = "zero-training probe on S1's frozen adapter (attention pooling over all 40 tokens and the 4 history frames, fit on train once, scored on eval once); if it detects, the next arm is that head, if not the representation lever (the 8x20 grid or the 416x1024 encode)"


def check_alignment(tab, plat, join):
    """a label table or platform sidecar whose clip order / offsets / counts differ from the join's is REFUSED"""
    for nm in ("clip", "off", "n"):
        if not np.array_equal(tab[f"eval_{nm}"], join[nm]):
            raise RuntimeError(f"strat_labels_table eval_{nm} differs from the join's: a misaligned table is refused")
    if not np.array_equal(plat["eval_clip"], join["clip"]):
        raise RuntimeError("platform sidecar clip order differs from the join's")


# ------------------------------------------------------------------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default="")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--n-boot", type=int, default=L.NB)
    ap.add_argument("--smoke", action="store_true", help="a reduced grid / one seed: a code check, never a result")
    a = ap.parse_args()
    CI = L.load_ci()
    join = dict(np.load(L.pinned("join_eval141.npz")))
    tab = dict(np.load(L.pinned("strat_labels_table.npz")))
    plat = dict(np.load(L.INP + "/strat_platform_by_clip.npz"))
    ci, t = L.grid_windows(a.grid or L.pinned("eval_grid_windows.json"), join["clip"])
    check_alignment(tab, plat, join)
    r = L.rows_of(tab["eval_off"], ci, t)
    y = label_rows(tab, r)
    eid = ci
    base = table_baselines(tab)
    probs = [f"{v}: constant baseline (report basis) {base[v]['report_basis_acc']:.4f} differs from LABELS_REPORT {LABELS_REPORT[v]}" for v in CTX_VARS
             if abs(base[v]["report_basis_acc"] - LABELS_REPORT[v]) > TOL_REPORT]
    rig_pred, rig_map = rig_baseline(tab, plat, ci)
    plat_w = np.asarray(plat["eval_platform_id"], np.int64)[ci]
    consts = base
    rep = {"tool": "strat1_ctx_read.py", "tier": "T0 (per-frame perception output at the window's NOW frame)", "evidence_class": "MEASURED (ours; T0 evals, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0 ({CI.SOURCE}); 'another draw of EPISODES' only",
           "smoke": bool(a.smoke), "inputs_sha256": {k: L.PINS[k] for k in ("strat_labels_table.npz", "join_eval141.npz")} | {"strat_platform_by_clip.npz": L.sha256_file(L.INP + "/strat_platform_by_clip.npz")},
           "baselines_from_table": base, "labels_report": LABELS_REPORT, "rig_map": rig_map, "illumination_twilight": {"status": "n/a (excluded)", **base["twilight"], "note": "the card text says 1 clip; the table has this many clips with at least one twilight step"},
           "not_in_arm": {"C0": "no output", "N1": "no output"}, "problems": probs, "arms": {}}
    rep["controls"] = controls(CI, y, eid, consts, rig_pred, plat_w)
    for kk in ("K2", "K3", "K5"):
        for v, d in rep["controls"][kk].items():
            if not d["pass"]:
                probs.append(f"{kk} {v} failed: {d}")
    if not rep["controls"]["K6"]["pass"]:
        probs.append(f"K6 failed: {rep['controls']['K6']}")
    per, k4d = {}, {v: False for v in CTX_VARS}
    for s in a.seeds:
        p, p4 = f"{a.t0}/eval_STRAT1-S1_s{s}_real.npz", f"{a.t0}/eval_STRAT1-S1_s{s}_k4.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        if not (np.array_equal(z["win_ci"], ci) and np.array_equal(z["win_t"], t)):
            raise RuntimeError(f"{p}: the evaluator's windows differ from the grid")
        per[s] = read_arm(CI, z["ctx_logits"], y, eid, consts, rig_pred, plat_w, a.n_boot)
        # time-base sensitivity (beside): labels 0.2 s later (one cache step), clipped inside the clip
        r1 = np.minimum(r + 1, np.asarray(tab["eval_off"])[ci] + np.asarray(tab["eval_n"])[ci] - 1)
        y1 = label_rows(tab, r1)
        ps = split_logits(z["ctx_logits"])
        sens = {}
        for v in ("weather", "illumination"):
            k = CTX_VARS.index(v)
            m = y1[:, k] >= 0
            sens[v] = {"accuracy_shifted_0.2s": float((ps[k].argmax(1)[m] == y1[m, k]).mean()), "accuracy_unshifted": per[s][v]["accuracy"]}
        rep["arms"][f"S1_s{s}"] = {"rows": per[s], "time_base_sensitivity_beside": sens}
        if os.path.exists(p4):
            z4 = np.load(p4, allow_pickle=True)
            r4 = read_arm(CI, z4["ctx_logits"], y, eid, consts, rig_pred, plat_w, a.n_boot)
            rep["arms"][f"S1_s{s}"]["K4_C-RAND"] = {v: {"accuracy": r4[v]["accuracy"], "detects": r4[v]["detects"]} for v in CTX_VARS}
            for v in CTX_VARS:
                k4d[v] = k4d[v] or r4[v]["detects"]
        else:
            probs.append(f"K4 pass missing for S1_s{s}: nothing is quoted without it")
            k4d["__missing__"] = True
    k4d.pop("__missing__", None) if not any("K4 pass missing" in x for x in probs) else None
    rep["outcome"] = decide(per, k4d) if not any("K4 pass missing" in x for x in probs) else {"__all__": "NOT-YET", "reason": "K4 pass missing"}
    if probs and "__all__" not in rep["outcome"]:
        rep["outcome"] = {"__all__": "INVALID", "reason": "control problems: " + "; ".join(probs)}
    rep["next_step"] = NEXT_NOT_DETECTED if any(str(o).startswith("NOT-DETECTED") for o in rep["outcome"].values()) else None
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print("WROTE", a.out, "|", json.dumps(rep["outcome"])[:300])
    for p in probs:
        print("  PROBLEM:", p)


if __name__ == "__main__":
    main()
