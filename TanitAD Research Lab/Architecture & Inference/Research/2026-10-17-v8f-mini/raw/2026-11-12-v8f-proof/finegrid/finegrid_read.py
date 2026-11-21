r"""finegrid_read.py -- the reader of card PREREG_V8F_PROOF_ADD41 (FINEGRID). Tier T0 (a diagnostic on logged frames, never driving performance). CPU only, reads files, touches no pod, no box 2, no git.

The metric logic is IMPORTED, not re-implemented: `tokprobe/tokprobe_read.py` (read_arm, paired_read_controls, arm_files) and through it `spatial/leadsup/read/leadsup_read_n1.py` (M1, M2, M4 of card ADD24 with
note N1, the episode-cluster bootstrap, pair_m1, the record checks and the controls k1 to k4). This file adds: the gap MAE in metres, the paired differences of card section 3 (G16, G16w, G64 on M2, on the
gap MAE and on M1; episode-cluster bootstrap, B 2,000, rng 0, same windows, same draw), the floor, the gap error by gap range, the controls of card section 4 and the names of card section 5.

  python finegrid_read.py --in-f DIR --g2 G2_GRIDS.json --out-dir DIR [--in-tk F:/TanitAD-scratch/v8proof/tokprobe/in] [--f-states-dir DIR] [--tk-states-dir DIR]
                          [--controls-json FILE] [--train-log-dir DIR[,DIR]] [--B 2000]
  python finegrid_read.py --tk-only --out-dir DIR                     (the reproduction of the banked TK reads, control g7; no F arm needed)
Exit 0 = every control passed, 2 = a control failed (no name and no outcome difference is written), 3 = an input record failed.
Controls g1, g3, g4 and the forward part of g6 are dev-box results handed in by the lead (`--controls-json`; a missing input counts as not passed). g2 is read from G2_GRIDS.json, g5 from the state records
of the F arms, p3 and g7 are computed here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402

TOKPROBE_DIR = "F:/TanitAD-scratch/v8proof/tokprobe"
if TOKPROBE_DIR not in sys.path:
    sys.path.insert(0, TOKPROBE_DIR)
import tokprobe_read as TKR  # noqa: E402

N, RP, O = TKR.N, TKR.RP, TKR.O
S0F = N.S0F
HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/")
DEFAULT_IN_TK = "F:/TanitAD-scratch/v8proof/tokprobe/in"
DEFAULT_TK_STATES = "F:/TanitAD-scratch/v8proof/moved_from_c/tokstates"        # the states of the banked TK arms (the in-dir holds only their records and head outputs)
BANKED_RESULT = "F:/TanitAD-scratch/v8proof/tokprobe/out/RESULT_TOKPROBE.json"
OUT_JSON = "RESULT_FINEGRID.json"
OUT_MD = "VERDICT_FINEGRID.md"
SEEDS = ("s0", "s1")
F_TYPES = ("F16", "F64")
TK_TYPES = ("TK16", "TK64")
F_ARMS = tuple("%s_%s" % (t, s) for t in F_TYPES for s in SEEDS)
TK_ARMS = tuple("%s_%s" % (t, s) for t in TK_TYPES for s in SEEDS)
P3_ARM = "F16SH_s0"
P3_BAR = 0.02
G2_BAR = 0.99
G2_N_CLIPS, G2_N_FRAMES = 20, 200
N_TOK_FINE = 160
MAE_FRACTION = 0.10                                   # the gap MAE of F16 at least 10 % below the lower of TK16 and TK64 of the same seed index
GAP_BINS = (("below 20 m", -np.inf, 20.0), ("20 to 40 m", 20.0, 40.0), ("above 40 m", 40.0, np.inf))
TIER_TEXT = "T0 (a diagnostic on logged frames, never driving performance)"
FAMILY_TEXT = ("LONGITUDINAL (distance keeping) is the family this lever serves and it is not measured by this read; LATERAL, TACTICAL: not measured by this read; "
               "STRATEGIC: NOT COMPUTED (a token probe reads no strategic quantity; n = 0)")
NAMES_GAP = ("FINER-GRID-READS-GAP", "FINER-GRID-WORSE", "FINER-GRID-FLAT")
NAME_CLOSING = "FINER-GRID-READS-CLOSING"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================================ the metrics added here (pure; ctx as built by N.window_ctx)
def _gap_mask(ctx, p):
    """Scored windows with a lead and a finite label (card section 3): the mask of M2 (`valid` of the context) and the head's finite output, inside an episode."""
    y = ctx["G15"][:, S0F]
    return ctx["valid"] & np.isfinite(y) & np.isfinite(p) & (ctx["ep10"] >= 0)


def gap_mae(ctx, g_h):
    """Mean absolute error in metres of the head's gap output at S0 on the scored windows with a lead and a finite label, with the episode-cluster interval of the same draw as M2."""
    y = ctx["G15"][:, S0F]
    m = _gap_mask(ctx, g_h)
    ep, n_ep, C = ctx["ep10"], ctx["n10"], ctx["C10"]
    s, n = np.zeros(n_ep), np.zeros(n_ep)
    np.add.at(s, ep[m], np.abs(y - g_h)[m])
    np.add.at(n, ep[m], 1.0)
    pt = float(s.sum() / max(n.sum(), 1.0))
    b = (C @ s) / np.maximum(C @ n, 1.0)
    return {"mae_m": pt, "ci95": RP.ci(b), "n": int(m.sum())}


def pair_r2(ctx, y, pa, pb):
    """Paired difference of the pooled R2 of two predictors of the same target y (a minus b) on the same windows and the same episode draw: pair_m1 of the imported reader with the target given."""
    m = ctx["valid"] & np.isfinite(pa) & np.isfinite(pb) & np.isfinite(y) & (ctx["ep10"] >= 0)
    ep, n, C = ctx["ep10"], ctx["n10"], ctx["C10"]
    Sa, Sb = RP.ep_stats_r2(ep[m], y[m], pa[m], n), RP.ep_stats_r2(ep[m], y[m], pb[m], n)
    d = RP.diff_row(RP.boot_r2(Sa, C), RP.boot_r2(Sb, C), RP.r2_pooled(y[m], pa[m]), RP.r2_pooled(y[m], pb[m]))
    d["n"] = int(m.sum())
    return d


def pair_mae(ctx, ga, gb):
    """Paired difference of the gap MAE (a minus b, metres; negative = a is better) on the windows where both outputs are finite, same draw. `rel` = (MAE_a - MAE_b) / MAE_b."""
    y = ctx["G15"][:, S0F]
    m = ctx["valid"] & np.isfinite(ga) & np.isfinite(gb) & np.isfinite(y) & (ctx["ep10"] >= 0)
    ep, n_ep, C = ctx["ep10"], ctx["n10"], ctx["C10"]
    sa, sb, n = np.zeros(n_ep), np.zeros(n_ep), np.zeros(n_ep)
    np.add.at(sa, ep[m], np.abs(y - ga)[m])
    np.add.at(sb, ep[m], np.abs(y - gb)[m])
    np.add.at(n, ep[m], 1.0)
    d = RP.diff_row((C @ sa) / np.maximum(C @ n, 1.0), (C @ sb) / np.maximum(C @ n, 1.0), float(sa.sum() / max(n.sum(), 1.0)), float(sb.sum() / max(n.sum(), 1.0)))
    d["n"] = int(m.sum())
    d["rel"] = float(d["diff"] / (sb.sum() / max(n.sum(), 1.0))) if sb.sum() > 0 else float("nan")
    return d


def summarize(ctx, out):
    """M1 (closing speed R2), M2 (gap R2), the gap MAE of one arm from its window outputs (dict gap_m, closing_ms) on the scored windows. M1 and M2 are the imported reader's `fid_one` on the same arguments as `eval_read`."""
    ep, n, C = ctx["ep10"], ctx["n10"], ctx["C10"]
    return {"M1": N.fid_one(ctx["c_lab"], out["closing_ms"], ctx["valid"], ep, n, C), "M2": N.fid_one(ctx["G15"][:, S0F], out["gap_m"], ctx["valid"], ep, n, C), "MAE": gap_mae(ctx, out["gap_m"])}


def pair_all(ctx, oa, ob):
    """One paired difference of card section 3 (a minus b): M2, the gap MAE and M1."""
    return {"M2": pair_r2(ctx, ctx["G15"][:, S0F], oa["gap_m"], ob["gap_m"]), "MAE": pair_mae(ctx, oa["gap_m"], ob["gap_m"]), "M1": N.pair_m1(ctx, oa["closing_ms"], ob["closing_ms"])}


def pair_names(seed):
    return {"G16_%s" % seed: ("F16_%s" % seed, "TK16_%s" % seed), "G16w_%s" % seed: ("F16_%s" % seed, "TK64_%s" % seed), "G64_%s" % seed: ("F64_%s" % seed, "TK64_%s" % seed)}


def floors(summ):
    """Card section 3: the largest absolute difference of M2 between the two seeds of one arm type over the four arm types (TK16, TK64, F16, F64); the same for M1."""
    out = {"m2_by_type": {}, "m1_by_type": {}}
    for t in TK_TYPES + F_TYPES:
        a, b = summ.get("%s_s0" % t), summ.get("%s_s1" % t)
        out["m2_by_type"][t] = abs(a["M2"]["r2"] - b["M2"]["r2"]) if a and b else None
        out["m1_by_type"][t] = abs(a["M1"]["r2"] - b["M1"]["r2"]) if a and b else None
    for k in ("m2", "m1"):
        v = list(out[k + "_by_type"].values())
        out["floor_" + k] = float(max(v)) if v and all(x is not None for x in v) else None
    return out


def gap_range_table(ctx, outs, arms=("F16_s0", "F16_s1", "TK64_s0", "TK64_s1")):
    """Beside, names nothing: the gap MAE by label gap range (below 20 m, 20 to 40 m, above 40 m) of the scored windows, for F16 and TK64 at each seed."""
    y = ctx["G15"][:, S0F]
    tab = {}
    for arm in arms:
        if outs.get(arm) is None:
            continue
        m = _gap_mask(ctx, outs[arm]["gap_m"])
        row = {}
        for nm, lo, hi in GAP_BINS:
            mm = m & (y >= lo) & (y < hi)
            row[nm] = {"mae_m": float(np.abs(y - outs[arm]["gap_m"])[mm].mean()) if mm.any() else None, "n": int(mm.sum())}
        tab[arm] = row
    return tab


def _sep_pos(d):
    return bool(d["diff"] > 0 and d["ci95"][0] > 0)


def _sep_neg(d):
    return bool(d["diff"] < 0 and d["ci95"][1] < 0)


def decide(summ, pairs, fl):
    """Card section 5. summ: arm -> summarize(); pairs: 'G16_s0' ... -> pair_all(); fl: floors(). Returns the name of the gap read, the closing name (or none) and every condition."""
    need = [a for a in F_ARMS + TK_ARMS]
    if any(summ.get(a) is None for a in need) or any(pairs.get("%s_%s" % (g, s)) is None for g in ("G16", "G16w", "G64") for s in SEEDS):
        return {"gap": "not named", "closing": "not named", "reason": "an arm was not read: both seeds of every arm type are needed"}
    if fl["floor_m2"] is None or fl["floor_m1"] is None:
        return {"gap": "not named", "closing": "not named", "reason": "the floor needs both seeds of the four arm types"}
    four = ["%s_%s" % (g, s) for g in ("G16", "G16w") for s in SEEDS]
    m2_pos = all(_sep_pos(pairs[k]["M2"]) for k in four)
    m2_neg = all(_sep_neg(pairs[k]["M2"]) for k in four)
    mean16 = float(np.mean([pairs["G16_%s" % s]["M2"]["diff"] for s in SEEDS]))
    mean16w = float(np.mean([pairs["G16w_%s" % s]["M2"]["diff"] for s in SEEDS]))
    above_floor = bool(mean16 > fl["floor_m2"] and mean16w > fl["floor_m2"])
    mae = {}
    for s in SEEDS:
        lower = min(summ["TK16_%s" % s]["MAE"]["mae_m"], summ["TK64_%s" % s]["MAE"]["mae_m"])
        f16 = summ["F16_%s" % s]["MAE"]["mae_m"]
        mae[s] = {"mae_f16": f16, "mae_tk16": summ["TK16_%s" % s]["MAE"]["mae_m"], "mae_tk64": summ["TK64_%s" % s]["MAE"]["mae_m"], "lower_of_tk": lower, "f16_below_lower": float(1.0 - f16 / lower),
                  "clause_met": bool(f16 <= (1.0 - MAE_FRACTION) * lower)}
    mae_ok = all(v["clause_met"] for v in mae.values())
    info = {"m2_g16_g16w_above_0_with_interval_excluding_0_at_both_seeds": m2_pos, "m2_g16_g16w_below_0_with_interval_excluding_0_at_both_seeds": m2_neg, "mean_G16_m2": mean16, "mean_G16w_m2": mean16w,
            "floor_m2": fl["floor_m2"], "each_mean_above_floor": above_floor, "mae_clause_by_seed": mae, "mae_clause_met_at_both_seeds": mae_ok}
    if m2_pos and above_floor and mae_ok:
        gap = NAMES_GAP[0]
    elif m2_neg:
        gap = NAMES_GAP[1]
    else:
        gap = NAMES_GAP[2]
    m1_pos = all(_sep_pos(pairs[k]["M1"]) for k in four)
    mean16c = float(np.mean([pairs["G16_%s" % s]["M1"]["diff"] for s in SEEDS]))
    mean16wc = float(np.mean([pairs["G16w_%s" % s]["M1"]["diff"] for s in SEEDS]))
    cl_floor = bool(mean16c > fl["floor_m1"] and mean16wc > fl["floor_m1"])
    info.update({"m1_g16_g16w_above_0_with_interval_excluding_0_at_both_seeds": m1_pos, "mean_G16_m1": mean16c, "mean_G16w_m1": mean16wc, "floor_m1": fl["floor_m1"], "each_m1_mean_above_floor": cl_floor})
    closing = NAME_CLOSING if (m1_pos and cl_floor) else "none (a flat closing read names nothing)"
    return {"gap": gap, "closing": closing, "reason": "card section 5", **info}


def compute_outcomes(ctx, outs):
    """Everything that depends on the arms' window outputs: per-arm summaries, the nine paired differences, the floor, the names, the gap range table. outs: arm -> dict(gap_m, closing_ms) or None."""
    arms = list(F_ARMS + TK_ARMS)
    summ = {a: (summarize(ctx, outs[a]) if outs.get(a) is not None else None) for a in arms}
    pairs = {}
    for s in SEEDS:
        for k, (a, b) in pair_names(s).items():
            pairs[k] = pair_all(ctx, outs[a], outs[b]) if outs.get(a) is not None and outs.get(b) is not None else None
    fl = floors(summ) if all(summ.values()) else {"floor_m2": None, "floor_m1": None, "m2_by_type": {}, "m1_by_type": {}}
    return {"summary": summ, "paired": pairs, "floors": fl, "names": decide(summ, pairs, fl), "gap_range": gap_range_table(ctx, outs)}


# ============================================================================================ controls
def max_abs_diff(a, b):
    """Largest absolute difference over the numeric leaves of two nested structures (dict / list / number); inf when the structures differ; NaN equals NaN."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return float("inf")
        return max([max_abs_diff(a[k], b[k]) for k in a] + [0.0])
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max([max_abs_diff(x, y) for x, y in zip(a, b)] + [0.0])
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return 0.0 if bool(a) == bool(b) else float("inf")
    if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
        a, b = float(a), float(b)
        if np.isnan(a) and np.isnan(b):
            return 0.0
        return abs(a - b)
    return 0.0 if a == b else float("inf")


def repro_row(R, banked):
    """Control g7, TK side: the M1, M2, M4 of the reader on the banked outputs against the banked RESULT_TOKPROBE.json (eval windows and train probe clips), max absolute difference 0.0."""
    d = {}
    for k in ("M1", "M2", "M4"):
        d["eval_" + k] = max_abs_diff(R[k], banked[k])
        if "train" in banked and "train" in R:
            d["train_" + k] = max_abs_diff(R["train"][k], banked["train"][k])
    mx = max(d.values())
    return {"pass": bool(mx == 0.0), "max_abs_diff": mx, "by_part": d}


def check_g2(path):
    """Control g2: the 2 by 2 mean of the 8 by 20 tokens against the stored 4 by 10 tokens, correlation at least 0.99 on 200 frames of 20 train clips (the file is written by `tok_probe_g.py g2`)."""
    if not path or not os.path.exists(path):
        return {"pass": False, "reason": "no G2_GRIDS.json"}
    g = json.load(open(path, encoding="utf-8"))
    ok = bool(g.get("n_clips_read") == G2_N_CLIPS and g.get("n_frames") == G2_N_FRAMES and "corr" in g and g["corr"] >= G2_BAR and g.get("pass") is True)
    return {"pass": ok, "corr": g.get("corr"), "median_abs_diff": g.get("median_abs_diff"), "max_abs_diff": g.get("max_abs_diff"), "n_clips": g.get("n_clips_read"), "n_frames": g.get("n_frames"),
            "bar_corr": G2_BAR, "reason": None if ok else "correlation under the bar, or not 20 clips and 200 frames: the 8 by 20 grid is then 'another cached grid' and no name is given"}


def check_g5(files, W):
    """Control g5 on the state records an F arm was extracted with: grid pool8x20, 160 tokens, 160 tokens seen in training, state width 160 x W."""
    rows = {}
    for sp in ("eval", "train"):
        p = files[sp + "_states"] + ".json"
        if not os.path.exists(p):
            rows[sp] = {"pass": False, "reason": "no record"}
            continue
        r = json.load(open(p, encoding="utf-8"))
        rows[sp] = {"pass": bool(r.get("grid") == "pool8x20" and r.get("n_tok") == N_TOK_FINE and r.get("n_tok_seen_in_training") == N_TOK_FINE and r.get("D") == N_TOK_FINE * W),
                    "grid": r.get("grid"), "n_tok": r.get("n_tok"), "n_tok_seen_in_training": r.get("n_tok_seen_in_training"), "D": r.get("D")}
    return {"pass": all(v["pass"] for v in rows.values()), **rows}


def p5_card_form(path, window=1000):
    """Control p5 as card ADD27 words it: the mean of the interval means of the lead term over the last 1,000 steps below the same over the first 1,000, and gap-labelled frames above 0 at every logged step
    (the instantaneous count; the stricter interval-minimum form of the tool is printed beside and gates nothing: the banked ADD27 run showed it fails on real data)."""
    recs = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
    hdr, recs = recs[0], recs[1:]
    total = int(hdr["total"])
    body = [r for r in recs if not r["first"]]
    first = [r["lead_sup_mean"] for r in body if r["step"] <= window]
    last = [r["lead_sup_mean"] for r in body if r["step"] > total - window]
    falls = bool(first and last and float(np.mean(last)) < float(np.mean(first)))
    gap_ok = bool(min(r["n_gap"] for r in recs) > 0)
    done = bool(recs and recs[-1]["step"] == total)
    return {"pass": bool(falls and gap_ok and done), "first_mean": float(np.mean(first)) if first else None, "last_mean": float(np.mean(last)) if last else None, "min_n_gap_instantaneous": min(r["n_gap"] for r in recs),
            "min_n_gap_interval_info_only": min(r["n_gap_min_interval"] for r in recs), "last_step": recs[-1]["step"] if recs else None, "total": total}


def _truthy(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, dict):
        return bool(v.get("pass") is True) if "pass" in v else (bool(v) and all(_truthy(x) for x in v.values()))
    return False


def lead_controls(path):
    return json.load(open(path, encoding="utf-8")) if path else {}


def given_controls(given, p5, arms):
    """The dev-box controls handed in (g1, g3, g4, g6_forward) and p5 per arm; a missing one counts as not passed."""
    c = {}
    for k in ("g1", "g3", "g4", "g6_forward"):
        c["%s_given" % k] = {"pass": _truthy(given.get(k)), "given": k in given}
    for a in arms:
        v = p5.get(a) if p5 else None
        if v is None and isinstance(given.get("p5"), dict):
            v = given["p5"].get(a)
        c["p5_train_log|%s" % a] = {"pass": _truthy(v), "given": v is not None}
    return c


def flatten(ctrl):
    out = {}
    for k, v in ctrl.items():
        if isinstance(v, dict) and "pass" in v:
            out[k] = bool(v["pass"])
        elif isinstance(v, dict):
            for k2, x in N._flatten_controls({k: v}).items():
                out[k2] = bool(x)
        elif isinstance(v, bool):
            out[k] = v
    return out


def gated(flat):
    """Every control gates, except k3_card_form_slot (printed, gates nothing; note ADD24_N1) and the control entries that only describe."""
    return {k: v for k, v in flat.items() if not k.startswith("k3_card_form_slot")}


# ============================================================================================ run
def _files(in_dir, states_dir, arm):
    return TKR.arm_files(in_dir, states_dir or in_dir, arm)


def read_tk(a, ctx, PT, cache):
    R, outs, ctrl, repro = {}, {}, {}, {}
    banked = json.load(open(a.banked_result, encoding="utf-8"))["arms"]
    for arm in TK_ARMS:
        R[arm], outs[arm], c = TKR.read_arm(arm, _files(a.in_tk, a.tk_states_dir, arm), ctx, PT, a.B, True, False, cache)
        ctrl.update(c)
        if R[arm].get("read"):
            repro[arm] = repro_row(R[arm], banked[arm])
        else:
            repro[arm] = {"pass": False, "reason": "arm not read (input records failed)"}
    return R, outs, ctrl, repro


def conclude(ctx, outs, failed_controls):
    """Controls FIRST: the paired differences, the floor, the names and the gap range table are computed and returned ONLY when no gated control failed; otherwise the outcome differences are withheld."""
    if failed_controls:
        return {"names": None, "outcome_differences": "withheld: a control did not pass (card section 4): " + ", ".join(failed_controls)}
    oc = compute_outcomes(ctx, outs)
    return {"summary": oc["summary"], "paired": oc["paired"], "floors": oc["floors"], "names": oc["names"], "gap_range_mae": oc["gap_range"]}


def run(a):
    t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    P = N.load_pack(a.pack_dir + "/eval_pack.npz")
    PT = N.load_pack(a.pack_dir + "/train_pack.npz")
    ctx = N.window_ctx(P, a.B)
    cache = {}
    R_tk, outs, ctrl, repro = read_tk(a, ctx, PT, cache)
    if a.tk_only:
        res = {"tier": TIER_TEXT, "card": "PREREG_V8F_PROOF_ADD41 control g7 (TK side)", "B": a.B, "reproduction": repro, "pass": bool(all(r["pass"] for r in repro.values())),
               "records": {k: v for k, v in ctrl.items() if k.startswith("a1_records")}, "elapsed_s": round(time.time() - t0, 1)}
        N.jdump(res, os.path.join(a.out_dir, "TK_REPRO_G7.json"))
        log("g7 TK reproduction", "PASS" if res["pass"] else "FAIL", {k: v["max_abs_diff"] if "max_abs_diff" in v else None for k, v in repro.items()})
        return 0 if res["pass"] else 2
    res = {"tier": TIER_TEXT, "families": FAMILY_TEXT, "card": "PREREG_V8F_PROOF_ADD41", "B": a.B, "evidence_class": "MEASURED once run", "arms": {}, "controls": {}}
    ctrl.update({"k1_label_path": N.control_k1(ctx, P["meta"]["k1_reference_r2_skill"]),
                 "a2_pack_counts": {"pass": bool(ctx["cnt"]["windows"] == N.EXPECT["windows"] and ctx["cnt"]["scored"] == N.EXPECT["scored"] and ctx["n10"] == N.EXPECT["episodes_scored"])}})
    ctrl.update(TKR.paired_read_controls(ctx))
    R_f, g5 = {}, {}
    for arm in F_ARMS:
        W = int(arm[1:3])
        files = _files(a.in_f, a.f_states_dir, arm)
        R_f[arm], outs[arm], c = TKR.read_arm(arm, files, ctx, PT, a.B, True, True, cache)
        ctrl.update(c)
        g5[arm] = check_g5(files, W)
    R3, o3, c3 = TKR.read_arm(P3_ARM, _files(a.in_f, a.f_states_dir, P3_ARM), ctx, PT, a.B, False, False, cache)
    p3_r2 = R3["M1"]["r2"] if R3.get("read") else None
    ctrl["p3_shuffled_labels"] = {"pass": bool(p3_r2 is not None and p3_r2 <= P3_BAR), "closing_r2_scored_windows": p3_r2, "bar": P3_BAR, "arm": P3_ARM}
    ctrl.update({k: v for k, v in c3.items() if k.startswith("a1_records")})
    ctrl["g5_declared_against_built|" + P3_ARM] = check_g5(_files(a.in_f, a.f_states_dir, P3_ARM), 16)
    for arm, v in g5.items():
        ctrl["g5_declared_against_built|%s" % arm] = v
    ctrl["g2_grids_belong_together"] = check_g2(a.g2)
    for arm, v in repro.items():
        ctrl["g7_tk_reproduces_banked|%s" % arm] = v
    p5 = {}
    if a.train_log_dir:
        dirs = [d for d in a.train_log_dir.split(",") if d]
        for arm in F_ARMS + (P3_ARM,):
            for d in dirs:
                p = "%s/%s/train_log.jsonl" % (d, arm)
                if os.path.exists(p):
                    p5[arm] = p5_card_form(p)
    ctrl.update(given_controls(lead_controls(a.controls_json), p5, F_ARMS + (P3_ARM,)))
    for arm in F_ARMS + (P3_ARM,):
        if arm in p5:
            ctrl["p5_train_log|%s" % arm]["detail"] = p5[arm]
    flat = flatten(ctrl)
    bad = sorted(k for k, v in gated(flat).items() if not v)
    all_pass = not bad
    res["controls"], res["controls_flat"], res["controls_failed"], res["controls_pass"] = ctrl, flat, bad, all_pass
    res["tk_reproduction_g7"] = repro
    for arm in F_ARMS + TK_ARMS:
        R = {**R_f, **R_tk}[arm]
        res["arms"][arm] = ({"M1": R["M1"], "M2": R["M2"]["head"], "M4": R["M4"], "train_probe_clips_in_sample": ({k: R["train"][k] if k != "M2" else R["train"]["M2"]["head"] for k in ("M1", "M2", "M4")} if "train" in R else None)}
                            if R.get("read") else {"read": False})
    res["arms"][P3_ARM] = {"M1": R3["M1"], "note": "control p3 arm: closing-speed R2 on the scored windows"} if R3.get("read") else {"read": False}
    res.update(conclude(ctx, outs, bad))
    res["elapsed_s"] = round(time.time() - t0, 1)
    N.jdump(res, os.path.join(a.out_dir, OUT_JSON))
    write_verdict(os.path.join(a.out_dir, OUT_MD), res)
    log("controls", "PASS" if all_pass else "FAILED", (res["names"] or {}).get("gap") if res["names"] else "no name", "elapsed %.0f s" % res["elapsed_s"])
    if any(isinstance(v, dict) and v.get("pass") is False and k.startswith("a1_records") for k, v in ctrl.items()):
        return 3
    return 0 if all_pass else 2


# ============================================================================================ writing
def _ci(c):
    return "[%+.3f, %+.3f]" % (c[0], c[1])


def write_verdict(path, res):
    L = ["# VERDICT_FINEGRID: card PREREG_V8F_PROOF_ADD41 (FINEGRID)", "", "Tier %s. %s." % (res["tier"], res["families"]),
         "Evidence class of every number: %s. B %d, episode-cluster bootstrap, rng 0, the 429 scored windows of 48 evaluation episodes." % (res["evidence_class"], res["B"]), ""]
    L += ["## Controls (card section 4; all must pass before any outcome is named)", ""]
    for k in sorted(res["controls_flat"]):
        L.append("- %s: %s%s" % (k, "PASS" if res["controls_flat"][k] else "FAIL", " (printed, gates nothing)" if k.startswith("k3_card_form_slot") else ""))
    g2 = res["controls"].get("g2_grids_belong_together", {})
    L.append("- g2 detail: correlation %s, median absolute difference %s, max absolute difference %s, %s clips, %s frames" % (g2.get("corr"), g2.get("median_abs_diff"), g2.get("max_abs_diff"), g2.get("n_clips"), g2.get("n_frames")))
    p3 = res["controls"].get("p3_shuffled_labels", {})
    L.append("- p3 (%s): closing-speed R2 on the scored windows %s, bar at most %+.2f" % (p3.get("arm"), "n/a" if p3.get("closing_r2_scored_windows") is None else "%+.3f" % p3["closing_r2_scored_windows"], P3_BAR))
    L.append("")
    if not res["controls_pass"]:
        L += ["## Names", "", "No name is given and no outcome difference is printed: %s" % res["outcome_differences"], ""]
    else:
        n = res["names"]
        L += ["## Names (card section 5; both seeds needed)", "", "- gap read: **%s**" % n["gap"], "- closing read: **%s**" % n["closing"], ""]
        for k in ("m2_g16_g16w_above_0_with_interval_excluding_0_at_both_seeds", "m2_g16_g16w_below_0_with_interval_excluding_0_at_both_seeds", "mean_G16_m2", "mean_G16w_m2", "floor_m2", "each_mean_above_floor",
                  "mae_clause_met_at_both_seeds", "m1_g16_g16w_above_0_with_interval_excluding_0_at_both_seeds", "mean_G16_m1", "mean_G16w_m1", "floor_m1", "each_m1_mean_above_floor"):
            if k in n:
                L.append("  - %s: %s" % (k, n[k]))
        for s, v in n.get("mae_clause_by_seed", {}).items():
            L.append("  - MAE clause %s: F16 %.2f m, TK16 %.2f m, TK64 %.2f m; F16 is %.1f %% below the lower of the two: %s" % (s, v["mae_f16"], v["mae_tk16"], v["mae_tk64"], 100 * v["f16_below_lower"], "met" if v["clause_met"] else "not met"))
        L += ["", "## Paired differences (a minus b, same windows, same seed index; M2 and M1 in R2, the MAE in metres, negative = a is better)", ""]
        for k in sorted(res["paired"]):
            p = res["paired"][k]
            L.append("- %s: M2 %+.4f %s; MAE %+.3f m %s (%+.1f %%); M1 %+.4f %s; n=%d" % (k, p["M2"]["diff"], _ci(p["M2"]["ci95"]), p["MAE"]["diff"], _ci(p["MAE"]["ci95"]), 100 * p["MAE"]["rel"], p["M1"]["diff"],
                                                                                    _ci(p["M1"]["ci95"]), p["M2"]["n"]))
        fl = res["floors"]
        L += ["", "- floor M2 (largest seed difference over the four arm types): %.4f; by type %s" % (fl["floor_m2"], json.dumps(fl["m2_by_type"])), "- floor M1: %.4f; by type %s" % (fl["floor_m1"], json.dumps(fl["m1_by_type"])), ""]
        L += ["## Gap error by gap range (beside, names nothing; MAE in metres on the scored windows)", ""]
        for arm, row in res["gap_range_mae"].items():
            L.append("- %s: " % arm + "; ".join("%s %s (n=%d)" % (nm, "n/a" if v["mae_m"] is None else "%.2f m" % v["mae_m"], v["n"]) for nm, v in row.items()))
        L.append("")
    L += ["## Per arm (reported; M4, the in-sample reads and G64 name nothing)", ""]
    for arm, R in res["arms"].items():
        if "M2" not in R:
            L.append("- %s: %s" % (arm, "control arm: closing-speed R2 %+.3f %s" % (R["M1"]["r2"], _ci(R["M1"]["r2_ci95"])) if "M1" in R else "not read"))
            continue
        s = (res.get("summary") or {}).get(arm)
        L.append("- %s: M2 gap R2 %+.3f %s; M1 closing R2 %+.3f %s; M4 AUROC %.3f%s" % (arm, R["M2"]["r2"], _ci(R["M2"]["r2_ci95"]), R["M1"]["r2"], _ci(R["M1"]["r2_ci95"]), R["M4"]["auroc"],
                                                                                    "; gap MAE %.2f m %s" % (s["MAE"]["mae_m"], _ci(s["MAE"]["ci95"])) if s else ""))
        t = R.get("train_probe_clips_in_sample")
        if t:
            L.append("  - train probe clips (in-sample, names nothing): M2 R2 %+.3f; M1 R2 %+.3f; M4 AUROC %.3f" % (t["M2"]["r2"], t["M1"]["r2"], t["M4"]["auroc"]))
    L += ["", "## Limits", "", "A probe on the tokens says what the tokens allow, not what the model does; 429 scored windows of 48 episodes; one corpus; two initialisation seeds per arm type; the reference arms come from an earlier run of "
          "the same tool (control g1 covers the code path, not the run); tokens stored as float8; the grid before pooling is not read. The paired bootstrap resamples episodes with the arms held fixed.", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-f", default=None, help="folder with the pulled head outputs of the F arms (F16_s0 F16_s1 F64_s0 F64_s1 F16SH_s0)")
    ap.add_argument("--f-states-dir", default=None, help="folder with the state files of the F arms (default: --in-f)")
    ap.add_argument("--in-tk", default=DEFAULT_IN_TK)
    ap.add_argument("--tk-states-dir", default=DEFAULT_TK_STATES)
    ap.add_argument("--g2", default=None, help="G2_GRIDS.json of the pod run")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pack-dir", default=TKR.READ)
    ap.add_argument("--banked-result", default=BANKED_RESULT)
    ap.add_argument("--controls-json", default=None, help="dev-box results g1, g3, g4, g6_forward (and p5 per arm if no --train-log-dir)")
    ap.add_argument("--train-log-dir", default=None, help="comma list of run folders holding <ARM>/train_log.jsonl (p5 is read from them)")
    ap.add_argument("--tk-only", action="store_true", help="only the reproduction of the banked TK reads (control g7)")
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args(argv)
    if not a.tk_only and not a.in_f:
        raise SystemExit("REFUSED: give --in-f (or --tk-only)")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
