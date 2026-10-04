"""R8-4 BASELINE ROW for refcv7-r101-s0 @ 50,400 -- every R8-4 measure the banked capture supports (DESIGN §4, X8).

Zero GPU, zero training.  Re-scores the route package's banked capture (`eval_s0g.npz`, replicate `eval_s1.npz`) with the
independent metric module `taniteval.tactical_conditioning` (code/fix/taniteval/taniteval/).  Writes
`raw/r84_baseline_refcv7.json`.

TIER / ESTIMATOR STAMP (CLAUDE.md): OPEN-LOOP, single-shot planning on logged frames of the held-out eval139 clips (not T1).
ONE TRAINING SEED.  Every interval is the episode-cluster bootstrap (`taniteval.ci`, B = 2000, seed 0, ratio-of-sums over
ALL 139 episodes) and answers "would another draw of EPISODES say this?" ONLY.  The sampler-seed-1 replicate (`eval_s1`)
is carried for the pick-dependent rows and is the INFERENCE-variance floor; the training-variance floor is NOT measured.

CONTROLS (the route package's published numbers, which this module did not produce).  The script computes them FIRST and
refuses to write rows if a HARD control does not reproduce:
  H1  GT classes: my `gt_route_class` == route_metrics.gt_class window by window, and the counts are RESULT.md §1.1's
      (107 turns = 40 L + 67 R, 588 straight, 105 gentle, 312 unclassified);
  H2  the pick: turn direction-correct 90/107 = 0.841, heading-within-15 55/107 = 0.514 (seed 0; RESULT.md §1.1,
      raw/route_analysis.json), and seed 1: 91/107, 53/107;
  H3  `terminal_heading` / `dir_class` equal route_metrics' on every fan / emitted / GT path (exact).
SOFT cross-checks (earlier agent's D0, `raw/d0_dose_response.json::baseline_R8_4`, different code): the side-collapsed
tactical accuracy / F1 and the consistency numbers must reproduce; a miss is REPORTED, not adjusted.

Run:  PYTHONIOENCODING=utf-8 PYTHONPATH="<PKG>/code/fix;C:/Users/Admin/r8_wpb/stack;C:/Users/Admin/r8_wpb/taniteval" \
        C:/Users/Admin/venvs/tanitad/Scripts/python.exe code/r84_baseline_refcv7.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import pathlib
import sys
import time
import warnings

import numpy as np

PKG = pathlib.Path(os.path.abspath(__file__)).parent.parent        # abspath, not resolve(): keep the D: spelling
INNER = PKG / "code" / "fix" / "taniteval" / "taniteval"
ROUTE_FILE = pathlib.Path(r"D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/"
                          r"2026-10-04-refcv7-route-following/code/route_metrics.py")
BIN = pathlib.Path(r"D:/refcv7_route_bin/2026-10-04")
MD5 = {"eval_s0g": "1c48a53e84e3a6296004efc0a81a0a64", "eval_s1": "5f783c842497b9dff0d878c98b23f883",
       "train_s0": "53e06ec57663474905cef487f2ebc4a9"}
B, SEED = 2000, 0

import taniteval  # noqa: E402  (the regular package from the tip extract on PYTHONPATH)

taniteval.__path__.insert(0, str(INNER))
sys.modules.pop("taniteval.tactical_conditioning", None)
import taniteval.tactical_conditioning as tc  # noqa: E402
from taniteval import ci as eci  # noqa: E402

assert pathlib.Path(tc.__file__).resolve() == (INNER / "tactical_conditioning.py").resolve(), tc.__file__
print(f"[r84] taniteval.tactical_conditioning imported from {tc.__file__}")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# the D0 side-collapse of the v7 lat head and its lon collapse (typed here; D0 code is not imported)
LAT_SIDE = np.array([0, 1, -1, 0, 1, -1, 1, -1])
V7LON_TO_5 = {"FOLLOW": 4, "CRUISE": 4, "YIELD_MERGE": 2, "BRAKE_TO": 2, "CREEP": 1, "HOLD": 0,
              "ADAPT_SPEED_FOR_CURVE": 2, "ACCELERATE": 3}             # LON5 = HOLD, CREEP, BRAKE, ACCEL, CRUISE
LON5 = ("HOLD", "CREEP", "BRAKE", "ACCEL", "CRUISE")
PLAN_LON_TO_5 = {"HOLD": 0, "CREEP": 1, "STOP": 2, "DECELERATE": 2, "ACCELERATE": 3, "KEEP": 4, "FOLLOW": 4}


# ------------------------------------------------------------------------------------------------- #
def md5(p: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(tag: str) -> dict:
    p = BIN / f"{tag}.npz"
    m = md5(p)
    if m != MD5[tag]:
        raise SystemExit(f"[r84] {p} md5 {m} != {MD5[tag]} -- refusing")
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def d0_module():
    """the earlier agent's D0 script, loaded by path for the SOFT cross-checks only (never by the metric module)."""
    spec = importlib.util.spec_from_file_location("d0_dose_response_controls", PKG / "code" / "d0_dose_response.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def route_module():
    spec = importlib.util.spec_from_file_location("route_metrics_controls", ROUTE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def ci_row(per_window, eid, reducer="mean"):
    """episode-cluster bootstrap of a NaN-masked per-window vector (ratio-of-sums over ALL episodes)."""
    v = np.asarray(per_window, dtype=np.float64)
    r = eci.episode_cluster_bootstrap(v, eid, reduce=reducer, n_boot=B, seed=SEED)
    sc = np.isfinite(v)
    return {"value": r["mean"], "lo": r["lo"], "hi": r["hi"], "n_scored": int(sc.sum()),
            "n_episodes_scored": int(len(np.unique(np.asarray(eid)[sc]))), "n_episodes_bootstrap": r["n_episodes"],
            "estimator": f"episode_cluster_bootstrap(B={B}, seed={SEED}, ratio-of-sums over all episodes)"}


def paired_row(a, b, eid):
    r = eci.paired_episode_cluster_bootstrap(np.asarray(a, float), np.asarray(b, float), eid, n_boot=B, seed=SEED)
    return {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"],
            "n_scored": int(np.isfinite(np.asarray(a, float)).sum()),
            "estimator": f"paired_episode_cluster_bootstrap(B={B}, seed={SEED})"}


def acc_block(pred, target, names, eid, allowed=None, train_marginal=None, with_f1_ci=True):
    """class_report + intervals on accuracy / the constant controls / head - control (paired) / macro-F1."""
    C = len(names)
    pred, target = np.asarray(pred), np.asarray(target)
    rep = tc.class_report(pred, target, C, names, allowed=allowed, train_marginal=train_marginal)
    ok = tc.per_window_correct(pred, target, allowed)
    out = {"report": rep, "accuracy_ci": ci_row(ok, eid)}
    cm = names.index(rep["majority_control"]["class"])
    ok_m = tc.per_window_correct(np.full(len(pred), cm), target, allowed)
    out["majority_control_ci"] = ci_row(ok_m, eid)
    out["accuracy_minus_majority"] = paired_row(ok, ok_m, eid)
    if train_marginal is not None:
        ct = names.index(rep["train_marginal_control"]["class"])
        ok_t = tc.per_window_correct(np.full(len(pred), ct), target, allowed)
        out["train_marginal_control_ci"] = ci_row(ok_t, eid)
        out["accuracy_minus_train_marginal"] = paired_row(ok, ok_t, eid)
    if with_f1_ci:
        idx = np.where(np.isfinite(ok), np.arange(len(pred), dtype=np.float64), np.nan)

        def macro_f1(v):
            i = v[np.isfinite(v)].astype(np.int64)
            r = tc.class_report(pred[i], target[i], C, names, allowed=None if allowed is None else allowed[i])
            return float("nan") if r["macro_f1"] is None else r["macro_f1"]
        out["macro_f1_ci"] = ci_row(idx, eid, reducer=macro_f1)
    return out


# ------------------------------------------------------------------------------------------------- #
def derive(z):
    W = len(z["sel_idx"])
    ar = np.arange(W)
    gt, gv = z["gt"], z["gt_valid"].astype(bool)
    d = {"W": W, "ar": ar, "eid": np.asarray(z["win_sha12"]).astype(str), "gt": gt, "gv": gv,
         "v0": z["v0"].astype(np.float64), "fan": z["fan"], "reach": z["reach"].astype(bool), "sel": z["sel_idx"]}
    d["pick"] = z["fan"][ar, z["sel_idx"]]
    d["cls_route"], d["th_gt"] = tc.gt_route_class(gt, gv)
    d["classified"] = d["cls_route"] != "unclassified"
    d["valid6"] = gv[:, -1]
    d["plan_lat3"] = tc.lat3_class(gt)
    d["plan_lon"] = tc.lon_class(gt, d["v0"])
    d["p_lat"], d["p_lon"] = z["p_lat"].astype(np.float64), z["p_lon"].astype(np.float64)
    d["tac_lat3"] = np.array(tc.V7_LAT_TO_LAT3)[d["p_lat"].argmax(1)]
    return d


def train_marginals(zt):
    gt, gv = zt["gt"], zt["gt_valid"].astype(bool)
    v6 = gv[:, -1]
    lat = tc.lat3_class(gt)[v6]
    lon = np.array(tc.PLAN_LON_TO_V7)[tc.lon_class(gt, zt["v0"].astype(np.float64))[v6]]
    return {"lat3": np.bincount(lat, minlength=3) / len(lat), "lon_v7": np.bincount(lon, minlength=8) / len(lon),
            "n_windows": int(v6.sum())}


def row_tactical(z, d, tm):
    eid = d["eid"]
    out = {"note": ("tactical head argmax vs the GT PLAN's own class (lat3 = 30-deg heading excursion, lon = v9 §3.5 literals on the "
                    "6-s plan).  A [0, 6] s PROXY for the v9 label, NOT v9 (v9's band is [NOW+2, NOW+8] s).  Head predictions are "
                    "identical on eval_s0g and eval_s1 (the sampler seed does not touch the tactical decoder).")}
    # --- lateral, 3-way ---------------------------------------------------------------------------- #
    pred = d["tac_lat3"]
    for name, mask in (("valid6_n%d" % d["valid6"].sum(), d["valid6"]), ("route_classified_n%d" % d["classified"].sum(), d["classified"])):
        tgt = np.where(mask, d["plan_lat3"], tc.IGNORE_INDEX)
        out["lat3__" + name] = acc_block(pred, tgt, tc.LAT3_NAMES, eid, train_marginal=tm["lat3"])
    # --- longitudinal, v7 8-way head vs plan class mapped through v9 §3.7 ----------------------------- #
    plan_v7 = np.array(tc.PLAN_LON_TO_V7)[np.where(d["plan_lon"] >= 0, d["plan_lon"], 0)]
    m6 = d["valid6"] & (d["plan_lon"] >= 0)
    tgt = np.where(m6, plan_v7, tc.IGNORE_INDEX)
    pred_lon = d["p_lon"].argmax(1)
    allowed = np.where(m6[:, None], tc.plan_lon_to_v7_allowed(d["plan_lon"]), False)
    out["lon_v7__partial_FOLLOW_undetermined"] = acc_block(pred_lon, tgt, tc.V7_LON_NAMES, eid, allowed=allowed,
                                                           train_marginal=tm["lon_v7"])
    out["lon_v7__strict_no_allowed"] = acc_block(pred_lon, tgt, tc.V7_LON_NAMES, eid, train_marginal=tm["lon_v7"])
    out["lon_v7__gameability_warning"] = ("under the partial reading a CONSTANT FOLLOW predictor is 'correct' on every DECELERATE / ACCELERATE / KEEP window "
                                          "(its majority_control, ~0.91 here): the leniency makes the score gameable, so the partial accuracy is "
                                          "never quotable without that control.  The STRICT row (CRUISE-majority control) is the headline.")
    out["lon_v7__note"] = ("partial = v9 §3.6: a 6-s plan has no agent data, so FOLLOW is undetermined and a FOLLOW prediction is "
                           "allowed for DECELERATE / ACCELERATE / KEEP; strict scores FOLLOW / YIELD_MERGE / ADAPT_SPEED_FOR_CURVE "
                           "as wrong whenever the plan class is another")
    out["head_prediction_counts"] = {
        "lat3": {n: int((pred == i).sum()) for i, n in enumerate(tc.LAT3_NAMES)},
        "lon_v7": {n: int((pred_lon == i).sum()) for i, n in enumerate(tc.V7_LON_NAMES)}}
    out["plan_class_counts"] = {
        "lat3": {n: int(((d["plan_lat3"] == i) & d["valid6"]).sum()) for i, n in enumerate(tc.LAT3_NAMES)},
        "lon": {n: int(((d["plan_lon"] == i) & d["valid6"]).sum()) for i, n in enumerate(tc.LON_NAMES)}}
    # --- D0-compatible lon5 collapse (a CONTROL on D0, not a new reading) -------------------------- #
    pl5 = np.array([PLAN_LON_TO_5[tc.LON_NAMES[c]] if c >= 0 else tc.IGNORE_INDEX for c in d["plan_lon"]])
    tg5 = np.where(d["valid6"], pl5, tc.IGNORE_INDEX)
    pr5 = np.array([V7LON_TO_5[tc.V7_LON_NAMES[i]] for i in pred_lon])
    out["lon5_D0_collapse"] = acc_block(pr5, tg5, LON5, eid, with_f1_ci=False)
    return out


def row_old_band(z, d):
    eid = d["eid"]
    o = {}
    for key, pred, names in (("lat_v7", d["p_lat"].argmax(1), tc.V7_LAT_NAMES), ("lon_v7", d["p_lon"].argmax(1), tc.V7_LON_NAMES)):
        tgt = z["lat_gt" if key == "lat_v7" else "lon_gt"].astype(np.int64)
        o[key] = acc_block(pred, tgt, names, eid, with_f1_ci=True)
    o["note"] = ("the shipped v7 labels on the old anchor band (lat_gt / lon_gt in the capture; -100 = unlabelled): "
                 "272 / 272 labelled windows.  A different label set from the plan-level classes above.")
    return o


def row_consistency(d):
    eid, fan, sel = d["eid"], d["fan"], d["sel"]
    W = d["W"]
    reach = d["reach"]
    masks = {"all_classified": d["classified"], "turn": np.isin(d["cls_route"], ["turnL", "turnR"]),
             "all_windows": np.ones(W, bool)}
    out = {"tags": "tag := the tactical head's lat decision, the SAME tag on every reach-kept candidate of the window; non-reach "
                   "candidates untagged (-1).  Reach-restricted as in D0.  cand_share = window-mean of the per-window share."}
    # reading A: lat3 (the task's reading): argmax of p_lat, TURN_L->1 TURN_R->2 else 0; candidate class = lat3_class (30 deg)
    tagsA = np.where(reach, d["tac_lat3"][:, None], -1)
    A = tc.consistency(fan, tagsA, sel, d["tac_lat3"], masks=masks, reading="lat3")
    # reading B: DESIGN §4 / D0 side-collapsed: summed side posterior argmax; candidate class = dir_class(terminal heading)
    p3 = np.stack([d["p_lat"][:, LAT_SIDE == s].sum(1) for s in (-1, 0, 1)], axis=1)
    side = p3.argmax(1) - 1
    tac_side_id = np.where(side == 1, 1, np.where(side == -1, 2, 0))
    tagsB = np.where(reach, tac_side_id[:, None], -1)
    Bk = tc.consistency(fan, tagsB, sel, tac_side_id, masks=masks, reading="dir")
    out["reading_A_lat3"] = {"summary": A["summary"], "definition": "tag = argmax(p_lat) -> lat3 (TURN_L 1, TURN_R 2, everything else 0); cand class = lat3_class (|heading excursion| >= 30 deg)"}
    out["reading_B_side_collapsed"] = {"summary": Bk["summary"], "definition": "tag = argmax of the side-summed p_lat (L = LC_L+NUDGE_L+TURN_L, R likewise, K the rest); cand class = dir_class(terminal_heading), tau 10.35 deg (DESIGN §4 / D0)"}
    # intervals on the headline rows (mean over scored windows, episode-cluster)
    cis = {}
    for rd, R in (("A_lat3", A), ("B_side", Bk)):
        for mk, m in masks.items():
            for fld in ("cand_share", "pick_own_tag"):
                v = np.where(m, R[fld], np.nan)
                cis[f"{rd}__{mk}__{fld}"] = ci_row(v, eid)
    out["intervals"] = cis
    # NULL: the same readings with the tactical decision taken from a DIFFERENT window (random permutation, 20 draws, seed 0) --
    # the reach-restriction stays the window's own.  A consistency number that does not clear this is the class base rate.
    rng = np.random.default_rng(0)
    nulls = {"A_lat3": [], "B_side": []}
    for _ in range(20):
        perm = rng.permutation(W)
        nA = tc.consistency(fan, np.where(reach, d["tac_lat3"][perm][:, None], -1), sel, d["tac_lat3"][perm], masks=masks, reading="lat3")["summary"]
        nB = tc.consistency(fan, np.where(reach, tac_side_id[perm][:, None], -1), sel, tac_side_id[perm], masks=masks, reading="dir")["summary"]
        nulls["A_lat3"].append(nA)
        nulls["B_side"].append(nB)
    out["null_tags_from_a_permuted_window"] = {}
    for rd, lst in nulls.items():
        out["null_tags_from_a_permuted_window"][rd] = {
            mk: {f: {"mean": float(np.mean([s[mk][f] for s in lst])), "min": float(np.min([s[mk][f] for s in lst])), "max": float(np.max([s[mk][f] for s in lst]))}
                 for f in ("cand_share", "pick_vs_tac")} for mk in masks}
    # chance reference for the pick: the fan's own share (a random reach-kept candidate), already in cand_share
    return out, A, Bk


def main():
    t0 = time.time()
    rm = route_module()
    caps = {"eval_s0g": load("eval_s0g"), "eval_s1": load("eval_s1")}
    zt = load("train_s0")
    tm = train_marginals(zt)
    out = {
        "meta": {
            "what": "R8-4 baseline row for refcv7-r101-s0 @ 50,400 (DESIGN.md §4, X8)",
            "module": {"path": (INNER / "tactical_conditioning.py").as_posix(), "md5": md5(INNER / "tactical_conditioning.py")},
            "inputs_md5": {k: MD5[k] for k in MD5}, "inputs_dir": BIN.as_posix(),
            "evidence_class": "MEASURED (this script on the banked route-package capture; artifact = this file)",
            "tier": "OPEN-LOOP (logged frames of eval139; planner + tactical head one-shot); NOT T1",
            "seeds": "ONE training seed (refcv7-r101-s0); inference replicate = sampler seed 1 (eval_s1)",
            "interval": f"episode-cluster bootstrap (taniteval.ci), B={B}, seed={SEED}; answers 'another draw of EPISODES' only",
            "train_marginal_source": {"file": "train_s0.npz (TRAIN-DIAG grid, 1,112 windows)", "n_valid6": tm["n_windows"],
                                      "lat3": tm["lat3"].round(6).tolist(), "lon_v7": tm["lon_v7"].round(6).tolist()},
        },
        "controls": {}, "rows": {}, "absent": {}}

    # ---------------- HARD CONTROLS ---------------- #
    hard_ok = True
    ctrl = out["controls"]
    lit_counts = {"turnL": 40, "turnR": 67, "straight": 588, "gentle": 105, "unclassified": 312}
    lit_pick = {"eval_s0g": (90, 55), "eval_s1": (91, 53)}
    D = {}
    for tag, z in caps.items():
        d = derive(z)
        D[tag] = d
        c = {}
        cls_r, th_r = rm.gt_class(z["gt"], z["gt_valid"])
        c["H1_gt_classes_equal_route_windowwise"] = bool(np.array_equal(d["cls_route"].astype(object), cls_r) and np.array_equal(d["th_gt"], th_r))
        u, n = np.unique(d["cls_route"], return_counts=True)
        c["H1_counts"] = dict(zip(u.tolist(), n.tolist()))
        c["H1_counts_match_RESULT_1_1"] = c["H1_counts"] == lit_counts
        rf = tc.route_following(d["pick"], d["gt"], d["gv"])
        t = rf["summary"]["turn"]
        c["H2_turn_n"] = t["n"]
        c["H2_dir_correct"] = f'{t["n_dir_correct"]}/{t["n"]} = {t["dir_correct"]:.4f}'
        c["H2_head15"] = f'{t["n_head15"]}/{t["n"]} = {t["head15"]:.4f}'
        c["H2_matches_RESULT"] = bool((t["n"], t["n_dir_correct"], t["n_head15"]) == (107,) + lit_pick[tag])
        exact = True
        for nm, P in (("fan", z["fan"]), ("traj", z["traj"]), ("gt", z["gt"])):
            a, b = tc.terminal_heading(P), rm.terminal_heading(P)
            exact &= bool(np.array_equal(a, b) and np.array_equal(tc.dir_class(a), rm.dir_class(b)))
        c["H3_terminal_heading_and_dir_class_exact_vs_route"] = bool(exact)
        c["emitted_traj_equals_fan_at_sel_idx"] = bool(np.array_equal(z["traj"], d["pick"]))
        ctrl[tag] = c
        hard_ok &= (c["H1_gt_classes_equal_route_windowwise"] and c["H1_counts_match_RESULT_1_1"] and c["H2_matches_RESULT"]
                    and c["H3_terminal_heading_and_dir_class_exact_vs_route"] and c["emitted_traj_equals_fan_at_sel_idx"])
    out["controls"]["HARD_ALL_PASS"] = bool(hard_ok)
    if not hard_ok:
        out["STOPPED"] = "a HARD control did not reproduce -- no rows computed (CLAUDE.md: report, do not adjust)"
        (PKG / "raw" / "r84_baseline_refcv7.json").write_text(json.dumps(jsonable(out), indent=1), encoding="utf-8")
        print("[r84] HARD CONTROL FAILED:", json.dumps(jsonable(ctrl), indent=1))
        raise SystemExit(2)

    # ---------------- ROWS ---------------- #
    d0 = json.loads((PKG / "raw" / "d0_dose_response.json").read_text(encoding="utf-8"))["baseline_R8_4"]
    for tag, z in caps.items():
        d = D[tag]
        rows = {}
        rows["i_tactical"] = row_tactical(z, d, tm)
        ob = row_old_band(z, d)
        rows["i_tactical"]["old_band_lat_v7"], rows["i_tactical"]["old_band_lon_v7"] = ob["lat_v7"], ob["lon_v7"]
        rows["i_tactical"]["old_band_note"] = ob["note"]
        cons, A, Bk = row_consistency(d)
        rows["iii_consistency"] = cons
        rows["iv_route_following"] = tc.route_following(d["pick"], d["gt"], d["gv"], eid=d["eid"], n_boot=B, seed=SEED)
        out["rows"][tag] = rows
        # ---------- SOFT cross-checks vs D0 (different code, earlier agent) ---------- #
        soft = {}
        # side-collapsed lat accuracy on the 800 classified windows, target = dir_class(GT terminal heading) + 1
        p3 = np.stack([d["p_lat"][:, LAT_SIDE == s].sum(1) for s in (-1, 0, 1)], axis=1)
        yhat = p3.argmax(1)
        tgt3 = np.where(d["classified"], tc.dir_class(d["th_gt"]).astype(np.int64) + 1, tc.IGNORE_INDEX)
        r3 = tc.class_report(yhat, tgt3, 3, ("R", "K", "L"))
        ref = d0["i_lat3_all_classified"]
        soft["D0_lat3_side_collapsed_accuracy"] = {"mine": r3["accuracy"], "d0": ref["accuracy"], "ok": abs(r3["accuracy"] - ref["accuracy"]) <= 5.0001e-5}
        soft["D0_lat3_majority_control"] = {"mine": r3["majority_control"]["accuracy"], "d0": ref["majority_control_accuracy"], "ok": abs(r3["majority_control"]["accuracy"] - ref["majority_control_accuracy"]) <= 5.0001e-5}
        soft["D0_lat3_macro_f1"] = {"mine": r3["macro_f1"], "d0": ref["macro_f1"], "ok": abs(r3["macro_f1"] - ref["macro_f1"]) <= 5.0001e-5}
        soft["D0_lat3_per_class_n"] = {"mine": {k: r3["per_class"][k]["n"] for k in ("R", "K", "L")}, "d0": {k: ref[k]["n"] for k in ("R", "K", "L")},
                                       "ok": all(r3["per_class"][k]["n"] == ref[k]["n"] for k in ("R", "K", "L"))}
        l5 = rows["i_tactical"]["lon5_D0_collapse"]["report"]
        rl5 = d0["i_lon5_valid_6s"]
        soft["D0_lon5_accuracy"] = {"mine": l5["accuracy"], "d0": rl5["accuracy"], "ok": abs(l5["accuracy"] - rl5["accuracy"]) <= 5.0001e-5}
        soft["D0_lon5_majority"] = {"mine": l5["majority_control"]["accuracy"], "d0": rl5["majority_control_accuracy"], "ok": abs(l5["majority_control"]["accuracy"] - rl5["majority_control_accuracy"]) <= 5.0001e-5}
        soft["D0_lon5_macro_f1"] = {"mine": l5["macro_f1"], "d0": rl5["macro_f1"], "ok": abs(l5["macro_f1"] - rl5["macro_f1"]) <= 5.0001e-5}
        soft["D0_lon5_per_class_n"] = {"mine": {k: l5["per_class"][k]["n"] for k in LON5}, "d0": {k: rl5[k]["n"] for k in LON5},
                                       "ok": all(l5["per_class"][k]["n"] == rl5[k]["n"] for k in LON5)}
        # EXPLAINED difference: D0 (D4's literals) has no STOP class.  v9 §3.5 / the task add it: a plan with a segment <= 0.5 m/s
        # while v0 > 0.5 is STOP unless an ACCEL event comes first -- for 0.5 < v0 < 2.0 that is NOT a +-1.5 m/s event, so D0
        # reads CRUISE / ACCEL there.  The claim to verify is that EVERY window where the two disagree is such a window.
        dmod = d0_module()
        d5 = dmod.lon_class_from_path(z["gt"], d["v0"])
        mine5 = np.array([PLAN_LON_TO_5[tc.LON_NAMES[c]] if c >= 0 else -1 for c in d["plan_lon"]])
        dis = np.nonzero(d["valid6"] & (mine5 != d5))[0]
        expl = [int(i) for i in dis if tc.LON_NAMES[d["plan_lon"][i]] == "STOP" and 0.5 < d["v0"][i] < 2.0]
        soft["D0_lon5_difference_explained_by_the_STOP_class"] = {
            "n_windows_differ": int(len(dis)), "n_explained": len(expl), "windows": [int(i) for i in dis],
            "v0_of_those": [round(float(d["v0"][i]), 3) for i in dis],
            "ok": bool(len(dis) == len(expl)),
            "note": "D0 has no STOP class; the lon5 accuracy/F1/n above differ from D0 ONLY on these windows (all 0.5 < v0 < 2.0 with a segment <= 0.5 m/s)"}
        # the four lon5 rows are NOT required to equal D0 to 4 dp once the difference is explained; flag them separately
        for k in ("D0_lon5_accuracy", "D0_lon5_majority", "D0_lon5_macro_f1", "D0_lon5_per_class_n"):
            soft[k]["ok_or_explained"] = bool(soft[k]["ok"] or soft["D0_lon5_difference_explained_by_the_STOP_class"]["ok"])
        for mk, key in (("all_classified", "iii_all_classified"), ("turn", "iii_turn")):
            if tag != "eval_s0g":
                break                                            # D0's consistency baseline was computed on eval_s0g only
            s = Bk["summary"][mk]
            r = d0[key]
            soft[f"D0_consistency_{mk}"] = {
                "mine": {"fan": s["cand_share"], "pick": s["pick_vs_tac"], "n": s["n_windows"]},
                "d0": {"fan": r["fan_share_consistent_with_tac_argmax"], "pick": r["pick_consistent_with_tac_argmax"], "n": r["n"]},
                "ok": bool(abs(s["cand_share"] - r["fan_share_consistent_with_tac_argmax"]) <= 5.0001e-5
                           and abs(s["pick_vs_tac"] - r["pick_consistent_with_tac_argmax"]) <= 5.0001e-5 and s["n_windows"] == r["n"])}
        ctrl[tag]["SOFT_D0_cross_checks"] = soft
        ctrl[tag]["SOFT_all_reproduce_or_explained"] = bool(all(v.get("ok_or_explained", v["ok"]) for v in soft.values()))
        ctrl[tag]["SOFT_strictly_identical_to_D0"] = bool(all(v["ok"] for k, v in soft.items() if k != "D0_lon5_difference_explained_by_the_STOP_class"))

    # route-package CI reference for the headline row (its own bootstrap, same capture): carried, not recomputed here
    ra = json.loads((ROUTE_FILE.parent.parent / "raw" / "route_analysis.json").read_text(encoding="utf-8"))
    out["controls"]["route_package_reference_intervals"] = {
        "source": (ROUTE_FILE.parent.parent / "raw" / "route_analysis.json").as_posix(),
        "eval_s0_turn_E9_pick_dir_correct": ra["chain_eval_s0"]["turn"]["E_e9_pick"],
        "eval_s0_turn_E9_pick_head15": ra["chain_eval_s0"]["turn"]["E_e9_pick_headagree"],
        "eval_s1_turn_E9_pick_dir_correct": ra["chain_eval_s1"]["turn"]["E_e9_pick"],
        "eval_s1_turn_E9_pick_head15": ra["chain_eval_s1"]["turn"]["E_e9_pick_headagree"]}

    # inference-seed floor on the pick-dependent rows
    s0, s1 = D["eval_s0g"], D["eval_s1"]
    out["inference_replicate"] = {
        "tactical_head_identical_across_seeds": bool(np.array_equal(s0["p_lat"], s1["p_lat"]) and np.array_equal(s0["p_lon"], s1["p_lon"])),
        "pick_index_same_on_share": float((s0["sel"] == s1["sel"]).mean()),
        "note": "the sampler seed changes the fan and the pick, not the tactical decoder; compare iv_* and iii_*pick* between eval_s0g and eval_s1"}

    out["absent"] = {
        "ii_controllability": {"computable": False, "reason": "refcv7 has NO conditioning input: its generator takes one window-level vector and no per-candidate hypothesis or constraint, so there is no way to force TURN_L / TURN_R / LANE_KEEP / STOP-at-d on the same scene. The instrument (`controllability`, `controllability_vs_shuffled`) is built and tested; its first real input is a refcv8 arm. The 'condition ignored' control (forced fan == unforced fan reads the base rate) is what the refcv7 fan would read by construction."},
        "i_goal_ap": {"computable": False, "reason": "the capture carries refcv7's 22-token validity `p_goal` [W, 22] but NOT the GT goal bits (y / w): those are the v9 goal release (WP-A, eval139 split) or the v7.2 jsonl, neither joined into this capture. `goal_ap` is built and tested (constant scorer == prevalence exactly)."},
        "i_constraint_mae": {"computable": False, "reason": "refcv7 has no constraint heads (t_start, dpsi, d_stop, v_target, t_reach are v9 fields and refcv8 heads); `g_tac` [W, 3, 4] in the capture is the E13 goal geometry, a different object. `constraint_mae` is built and tested."}}
    out["meta"]["wall_s"] = round(time.time() - t0, 1)
    dst = PKG / "raw" / "r84_baseline_refcv7.json"
    dst.write_text(json.dumps(jsonable(out), indent=1), encoding="utf-8")
    print(f"[r84] wrote {dst.as_posix()} in {out['meta']['wall_s']} s")
    for tag in caps:
        print(tag, "HARD controls pass; SOFT D0 cross-checks reproduce (or differ only by the explained STOP class):",
              out["controls"][tag]["SOFT_all_reproduce_or_explained"], "| strictly identical:", out["controls"][tag]["SOFT_strictly_identical_to_D0"])


if __name__ == "__main__":
    main()
