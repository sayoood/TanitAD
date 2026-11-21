r"""dscale_lead_read.py -- the reader of card PREREG_V8F_PROOF_ADD45 (DSCALE-LEAD) with note N1. Tier T0. CPU only, numpy only.

TIER: T0 (a probe on cached tokens of logged frames; it says what the tokens allow, not what a model does with them); never driving performance.

Reuses the pilot reader chain (finegrid_read_n1 -> tokprobe_read / leadsup_read_n1): the window context, the per-arm read (M2 gap R2, M4 presence AUROC), the gap error, the episode draw C10, the record checks and the
paired-read controls are imported, not re-derived. This file adds: the corpus arm (runC) against the union arm (runU) per seed K = 0 to 3, the two-level bootstrap, the controls of card section 5 with note N1
(d1 to d7), and the names of card section 4 with note N1 (no closing read: LEAD-READ-SCALES-CLOSING cannot be given). It must run where the pilot reader's fixed paths resolve (the pod mirror folder, as for finegrid_r).

  python dscale_lead_read.py --in-dir OUT/states --run-dir OUT --out-dir DIR [--precheck-json F] [--planted-json F] [--pack-dir DIR] [--B 2000] [--expect-corpus-clips 4512] [--done-marker F]
Folders under --in-dir: runC runU runU2 runCSH runUSH (states of the evaluation split, <ARM>_eval.npz + .json + _eval_lead.npz(.json,.head.npz)); under --run-dir: the same folders with <ARM>/train_log.jsonl and r1_batches.json,
and precheck/PRECHECK.json, precheck/PRECHECK_PLANTED.json.
Exit 0 = every control passed (a name is written), 2 = a control failed (no name, no outcome difference anywhere), 3 = an input record or folder is missing.
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

HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/")


def _base_dir():
    for d in (os.environ.get("FINEGRID_BASE_DIR"), HERE + "/../finegrid", "F:/TanitAD-scratch/v8proof/finegrid"):
        if d and os.path.exists(d + "/finegrid_read_n1.py"):
            return os.path.abspath(d)
    raise SystemExit("REFUSED: finegrid_read_n1.py (the pilot reader) not found; set FINEGRID_BASE_DIR")


sys.path.insert(0, _base_dir())
import finegrid_read_n1 as FR  # noqa: E402

TKR, N, RP, O, S0F = FR.TKR, FR.N, FR.RP, FR.O, FR.S0F
SEEDS = (0, 1, 2, 3)
STEPS_MAIN, STEPS_U2 = 19164, 38328
MIN_POS_SEEDS = 3
MAE_FRACTION = 0.10
ROW97_RANGE = (0.513, 0.791)
D3_AUROC_BAR, D5_AUROC_BAR = 0.85, 0.80
OUT_JSON, OUT_MD = "RESULT_DSCALE_LEAD.json", "VERDICT_DSCALE_LEAD.md"
TIER_LINE = "TIER: T0 (a probe on cached tokens of logged frames; it says what the tokens allow, not what a model does with them); never driving performance"
FAMILY_LINES = ("LONGITUDINAL (distance keeping): not measured by this read (it is the family this lever serves)", "LATERAL: not measured by this read", "TACTICAL: not measured by this read",
                "STRATEGIC: NOT COMPUTED (n = 0; a token probe reads no strategic quantity)")
NAMES = ("LEAD-READ-SCALES-GAP", "LEAD-READ-WORSE-AT-SCALE", "LEAD-READ-FLAT-AT-SCALE")
# card arm label -> (folder, arm name inside the folder, steps expected, permuted gap labels, corpus clip list used)
ARM_MAP = {}
for _k in SEEDS:
    ARM_MAP["P16C_s%d" % _k] = ("runC", "TK16_s%d" % _k, STEPS_MAIN, False, True)
    ARM_MAP["P16U_s%d" % _k] = ("runU", "TK16_s%d" % _k, STEPS_MAIN, False, False)
ARM_MAP["P16U2_s0"] = ("runU2", "TK16_s0", STEPS_U2, False, False)
ARM_MAP["P16CSH_s0"] = ("runCSH", "TK16_s0", STEPS_MAIN, True, True)
ARM_MAP["P16USH_s0"] = ("runUSH", "TK16_s0", STEPS_MAIN, True, False)
FOLDERS = ("runC", "runU", "runU2", "runCSH", "runUSH")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def lab(kind, k):
    return "P16%s_s%d" % (kind, k)


# ============================================================================================ the two-level bootstrap (copy of finegrid_r_read.py, seeds 0 to 3)
def _yp(ctx, measure, out):
    return ctx["G15"][:, S0F], out["gap_m"]


def boot_diff(ctx, measure, oa, ob):
    """(point difference a - b, bootstrap vector over the episode draw C10) of one paired measure on the same windows: M2 = pooled R2 of the gap, MAE = gap error in metres.
    RP.ci of the vector and the point equal the pilot reader's paired row (a test asserts it)."""
    y, pa = _yp(ctx, measure, oa)
    _, pb = _yp(ctx, measure, ob)
    ep, n_ep, C = ctx["ep10"], ctx["n10"], ctx["C10"]
    m = ctx["valid"] & np.isfinite(pa) & np.isfinite(pb) & np.isfinite(y) & (ep >= 0)
    if measure == "MAE":
        sa, sb, n = np.zeros(n_ep), np.zeros(n_ep), np.zeros(n_ep)
        np.add.at(sa, ep[m], np.abs(y - pa)[m])
        np.add.at(sb, ep[m], np.abs(y - pb)[m])
        np.add.at(n, ep[m], 1.0)
        return float(sa.sum() / max(n.sum(), 1.0) - sb.sum() / max(n.sum(), 1.0)), (C @ sa) / np.maximum(C @ n, 1.0) - (C @ sb) / np.maximum(C @ n, 1.0)
    Sa, Sb = RP.ep_stats_r2(ep[m], y[m], pa[m], n_ep), RP.ep_stats_r2(ep[m], y[m], pb[m], n_ep)
    return float(RP.r2_pooled(y[m], pa[m]) - RP.r2_pooled(y[m], pb[m])), RP.boot_r2(Sa, C) - RP.boot_r2(Sb, C)


def two_level(pts, D, rng_seed=0, inner="shared"):
    """Card section 3: the mean over the seed indices of the paired difference with a two-level bootstrap: the seed indices with replacement (outer), then the episodes with replacement (inner), B draws, rng 0.
    pts [n_seeds]; D [n_seeds, B] = their bootstrap vectors over the SAME episode draws (crossed design, the decision form). inner = 'per_seed': an independent episode draw per chosen seed (beside only)."""
    pts, D = np.asarray(pts, float), np.asarray(D, float)
    n, B = D.shape
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(B, n))
    if inner == "shared":
        vals = D[idx, np.arange(B)[:, None]].mean(1)
    else:
        vals = D[idx, rng.integers(0, B, size=(B, n))].mean(1)
    return {"mean": float(pts.mean()), "ci95": RP.ci(vals), "per_seed": [float(x) for x in pts], "n_seeds": int(n), "n_pos": int((pts > 0).sum()), "n_neg": int((pts < 0).sum()), "inner": inner}


def two_level_pair(ctx, outs, measure, inner="shared", a_kind="U", b_kind="C"):
    """S = runU minus runC per seed index K on the same windows (a minus b), then the two-level read."""
    pts, Ds = [], []
    for k in SEEDS:
        p, d = boot_diff(ctx, measure, outs[lab(a_kind, k)], outs[lab(b_kind, k)])
        pts.append(p)
        Ds.append(d)
    return two_level(pts, np.array(Ds), 0, inner)


# ============================================================================================ names (card section 4 with note N1)
def _above(t):
    return bool(t["mean"] > 0 and t["ci95"][0] > 0 and t["n_pos"] >= MIN_POS_SEEDS)


def _below(t):
    return bool(t["mean"] < 0 and t["ci95"][1] < 0)


def decide_u(tl_m2, mae_c, mae_u):
    """SCALES-GAP: the interval of S on M2 above 0, at least 3 of 4 seeds above 0, mean gap error of U at most 0.90 x that of C. WORSE-AT-SCALE: the interval below 0. Else FLAT-AT-SCALE.
    No closing name (note N1)."""
    ratio = float(mae_u / mae_c)
    mae_ok = bool(mae_u <= (1.0 - MAE_FRACTION) * mae_c)
    above, below = _above(tl_m2), _below(tl_m2)
    if above and mae_ok:
        name = NAMES[0]
    elif below:
        name = NAMES[1]
    else:
        name = NAMES[2]
    return {"name": name, "m2_above_0_with_n_pos_at_least_3": above, "m2_below_0": below, "mae_clause_met": mae_ok, "mean_gap_mae_C_m": float(mae_c), "mean_gap_mae_U_m": float(mae_u),
            "mae_ratio_U_to_C": ratio, "closing_name": "cannot be given (note N1: the union lead file carries no closing speed)",
            "margins_to_the_bars_beside": {"m2_ci_lower": tl_m2["ci95"][0], "m2_ci_upper": tl_m2["ci95"][1], "mae_ratio_minus_0.90": ratio - (1.0 - MAE_FRACTION)}}


def summ_one(ctx, out):
    """M2 (gap R2 with its interval) and the gap error of one arm from its window outputs on the scored windows. M1 is not read (note N1)."""
    ep, n, C = ctx["ep10"], ctx["n10"], ctx["C10"]
    return {"M2": N.fid_one(ctx["G15"][:, S0F], out["gap_m"], ctx["valid"], ep, n, C), "MAE": FR.gap_mae(ctx, out["gap_m"])}


def compute_outcomes_u(ctx, outs, extra_u2=True):
    """Everything that compares arms. outs: label -> dict(gap_m[, closing_ms]). Needs the eight main arms; P16U2_s0 and the control arms are beside / separate."""
    main = [lab(t, k) for t in ("C", "U") for k in SEEDS]
    summ = {a: summ_one(ctx, outs[a]) for a in main}
    tl = {"M2": two_level_pair(ctx, outs, "M2"), "MAE": two_level_pair(ctx, outs, "MAE")}
    mae = {t: float(np.mean([summ[lab(t, k)]["MAE"]["mae_m"] for k in SEEDS])) for t in ("C", "U")}
    names = decide_u(tl["M2"], mae["C"], mae["U"])
    beside = {"m2_two_level_independent_episode_draw_per_seed": two_level_pair(ctx, outs, "M2", "per_seed"),
              "std_over_seeds_of_M2": {t: float(np.std([summ[lab(t, k)]["M2"]["r2"] for k in SEEDS], ddof=1)) for t in ("C", "U")}}
    if extra_u2 and "P16U2_s0" in outs:
        u2 = outs["P16U2_s0"]
        beside["P16U2_s0"] = {"summary": summ_one(ctx, u2), "minus_P16U_s0": {"M2": boot_diff(ctx, "M2", u2, outs["P16U_s0"])[0], "MAE": boot_diff(ctx, "MAE", u2, outs["P16U_s0"])[0]},
                              "minus_P16C_s0": {"M2": boot_diff(ctx, "M2", u2, outs["P16C_s0"])[0], "MAE": boot_diff(ctx, "MAE", u2, outs["P16C_s0"])[0]}, "note": "names nothing; says whether the union arm is under-trained at matched steps"}
    return {"summary": summ, "two_level": tl, "names": names, "beside": beside}


# ============================================================================================ controls
def precheck_controls(pre, planted):
    """d1, d2, d2j and the counts from the precheck JSON (dscale_precheck.py), plus the two checks that show the join file (the probe's label source) carries labels on the NEW clips (not in the corpus list):
    d2n (join against the lead file on a sample of new clips, at least 200, no mismatching row) and d2g (per group, corpus and new: presence frames and gap-label frames above 0). A precheck JSON that lacks them
    fails these controls (they are not skipped). The planted-id precheck must show d1 FAILED."""
    c = {}
    checks = (pre or {}).get("checks", {})
    whole = bool(pre) and pre.get("pass") is True
    for nm, key in (("d1_no_leak", "d1_no_leak"), ("d2_labels_one_clock", "d2_labels_one_clock"), ("d2j_join_labels", "d2j_join_matches_lead_file"), ("counts", "counts")):
        c["%s_precheck" % nm] = {"pass": bool(whole and checks.get(key, {}).get("pass") is True), "present": key in checks}
    n_ = checks.get("d2n_join_matches_lead_file_new_clips")
    g = checks.get("d2g_join_labels_per_group") or {}
    c["d2n_new_clips_join_matches_lead_file"] = {"pass": bool(whole and isinstance(n_, dict) and n_.get("pass") is True and n_.get("sampled", 0) >= 200 and n_.get("rows_mismatching") == 0), "present": isinstance(n_, dict)}
    gn, gc = g.get("new"), g.get("corpus")
    ok_g = bool(whole and g.get("pass") is True and isinstance(gn, dict) and isinstance(gc, dict) and gn.get("gap_label_frames", 0) > 0 and gn.get("presence_frames", 0) > 0 and gc.get("gap_label_frames", 0) > 0)
    c["d2g_new_clips_carry_gap_labels"] = {"pass": ok_g, "present": bool(g), "new": gn, "corpus": gc}
    d1p = ((planted or {}).get("checks", {}).get("d1_no_leak") or {})
    c["d1_planted_shared_id_fails"] = {"pass": bool(planted and d1p.get("pass") is False and d1p.get("planted") is True and (planted.get("pass") is False)), "present": bool(planted)}
    return c


def clock_controls(clock, planted):
    """d2c (note N2): the rows of the union lead file sit on the clock of the cached tokens, in both groups (corpus clips, new clips): dscale_clockcheck.py compares the ego speed stored in the lead file at raw
    frame k with the ego speed of the token cache at token frame k / 2 + s; the smallest mean absolute difference must sit at shift 0 and be at most 0.5 times the value at the nearer of the shifts -1, +1.
    The planted record (lead rows moved by P token frames) must show the check FAILED with the smallest difference at shift -P in both groups. An absent record fails the control (it is not skipped)."""
    c = {}
    g = (clock or {}).get("groups") or {}
    ok = bool(clock and clock.get("pass") is True and clock.get("plant_shift") == 0 and clock.get("n_clips", 0) >= 200
              and all(isinstance(g.get(n), dict) and g[n].get("pass") is True and g[n].get("argmin_shift") == 0 and g[n].get("clips_read", 0) >= 200 for n in ("corpus", "new")))
    c["d2c_lead_rows_on_the_token_clock"] = {"pass": ok, "present": bool(clock),
                                              "shift0_over_nearest": {n: (g.get(n) or {}).get("ratio_shift0_to_nearest") for n in ("corpus", "new")},
                                              "mean_abs_diff_mps": {n: (g.get(n) or {}).get("mean_abs_diff_mps") for n in ("corpus", "new")}}
    gp = (planted or {}).get("groups") or {}
    ps = (planted or {}).get("plant_shift")
    okp = bool(planted and planted.get("pass") is False and isinstance(ps, int) and ps != 0
               and all(isinstance(gp.get(n), dict) and gp[n].get("pass") is False and gp[n].get("argmin_shift") == -ps for n in ("corpus", "new")))
    c["d2c_planted_shift_fails"] = {"pass": okp, "present": bool(planted), "plant_shift": ps}
    return c


def evalset_controls(es, planted):
    """d2e (note N2): the evaluation cache of the run holds the study's evaluation set, the same clip ids in the same order (dscale_tokens_same.py); the reader pairs states and pack by position.
    The planted record (two ids swapped) must show the check FAILED. An absent record fails the control."""
    d = (es or {}).get("d2e_eval_set_is_the_study_set") or {}
    ok = bool(es and es.get("pass") is True and d.get("pass") is True and d.get("same_ids_same_order") is True and d.get("planted_swap") is False and d.get("n_here") == 141 and d.get("n_study") == 141)
    dp = (planted or {}).get("d2e_eval_set_is_the_study_set") or {}
    okp = bool(planted and planted.get("pass") is False and dp.get("planted_swap") is True and dp.get("same_ids_same_order") is False)
    return {"d2e_eval_set_is_the_study_set": {"pass": ok, "present": bool(es)}, "d2e_planted_swap_fails": {"pass": okp, "present": bool(planted)}}


def d3_corpus_live(R_c, name):
    """d3 (note N1): the corpus arm is a live gap probe: the interval of its gap R2 lies above 0 and its presence AUROC is at least 0.85. The range of the TK16 arms of row 97 is printed beside and gates nothing."""
    r2, ci, auc = R_c["M2"]["head"]["r2"], R_c["M2"]["head"].get("r2_ci95"), R_c["M4"]["auroc"]
    return {"pass": bool(ci is not None and ci[0] > 0 and auc >= D3_AUROC_BAR), "gap_r2": r2, "gap_r2_ci95": ci, "auroc": auc, "auroc_bar": D3_AUROC_BAR,
            "beside_row97_range": list(ROW97_RANGE), "beside_inside_row97_range": bool(ROW97_RANGE[0] <= r2 <= ROW97_RANGE[1]), "arm": name}


def d5_permuted_dead(R_sh, name):
    """d5 (note N1): an arm with permuted gap labels reads a gap R2 whose interval includes 0 or lies below 0, while its presence AUROC stays at least 0.80 (presence labels are intact: the arm is live)."""
    r2, ci, auc = R_sh["M2"]["head"]["r2"], R_sh["M2"]["head"].get("r2_ci95"), R_sh["M4"]["auroc"]
    return {"pass": bool(ci is not None and ci[0] <= 0 and auc >= D5_AUROC_BAR), "gap_r2": r2, "gap_r2_ci95": ci, "auroc": auc, "auroc_bar": D5_AUROC_BAR, "arm": name}


def _read_log(path):
    recs = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
    return recs[0], recs[1:]


def p5_u(path, sh, expect_total, expect_clip_kept=None, window=1000):
    """Control d4 (the train-log check) on one arm. Header checks of the levers: w_clos 0.0 (note N1), shuffle_gap present exactly for the permuted-gap arms, the evaluation guard recorded with 0 shared ids,
    the step total equal to the expected one (matched compute), the clip list recorded with the expected kept count for the corpus arms and absent for the union arms.
    Body: the lead term of the last 1,000 steps below that of the first 1,000 (permuted-gap arms: on the presence part only), gap-labelled frames above 0 at every logged step, run complete."""
    hdr, body_all = _read_log(path)
    total = int(hdr["total"])
    head = {"total_equals_expected": total == expect_total, "w_clos_is_0": hdr.get("w_clos") == 0.0, "shuffle_gap_recorded_iff_expected": bool(hdr.get("shuffle_gap")) == bool(sh),
            "eval_guard_recorded_no_shared": bool((hdr.get("eval_guard") or {}).get("n_shared") == 0 and (hdr.get("eval_guard") or {}).get("n_eval_ids", 0) > 0)}
    cl = hdr.get("clip_list")
    head["clip_list_as_expected"] = bool((cl is not None and expect_clip_kept is not None and cl.get("n_kept") == expect_clip_kept) if expect_clip_kept is not None else cl is None)
    body = [x for x in body_all if not x["first"]]

    def part(keys, sel):
        v = [sum(x[k] for k in keys) for x in body if sel(x["step"])]
        return float(np.mean(v)) if v else None
    keys = ("lead_sup_bce_mean",) if sh else ("lead_sup_bce_mean", "lead_sup_gap_mean")
    fi, la = part(keys, lambda s: s <= window), part(keys, lambda s: s > total - window)
    gap_ok = bool(body_all and min(x["n_gap"] for x in body_all) > 0)
    done = bool(body_all and body_all[-1]["step"] == total)
    live = bool(fi is not None and la is not None and la < fi and gap_ok and done)
    return {"pass": bool(live and all(head.values())), "form": "presence part only (permuted gap labels)" if sh else "presence + gap parts (the closing part is exactly 0)", "first_mean": fi, "last_mean": la,
            "min_n_gap_instantaneous": min(x["n_gap"] for x in body_all) if body_all else None, "last_step": body_all[-1]["step"] if body_all else None, "total": total, "header_checks": head}


def check_records_u(files, label, steps_expected, arm):
    """Probe form of the eval state record of an arm (grid, token count, tokens seen in training, state width 16, input normalisation, step count, arm name)."""
    p = files["eval_states"] + ".json"
    if not os.path.exists(p):
        return {"pass": False, "reason": "no record " + p}
    r = json.load(open(p, encoding="utf-8"))
    g = bool(r.get("grid") == "pool4x10" and r.get("n_tok") == 40 and r.get("n_tok_seen_in_training") == 40 and r.get("D") == 40 * 16 and r.get("W") == 16)
    f = bool(r.get("input_norm") == "ln" and r.get("step") == steps_expected and r.get("arm") == arm)
    return {"pass": bool(g and f), "grid_tokens_width": g, "probe_form": f, "step": r.get("step"), "arm": r.get("arm"), "label": label}


def d6_limit(run_dir):
    """Control d6 is stated as a LIMIT, not a pass: the corpus arm and the union arm train on different window sets, so no window-for-window order can be equal; the sampler chunk of both is compared
    (the union join is one file, so the same lru/chunk is expected) and printed. Gates nothing."""
    out = {"gates": False, "statement": "LIMIT: the windows of runC and runU differ (corpus clips only against the whole union); the draw order is not equal window for window and this is not claimed as a pass"}
    chunks = {}
    for f in ("runC", "runU"):
        p = "%s/%s/r1_batches.json" % (run_dir, f)
        chunks[f] = json.load(open(p, encoding="utf-8")).get("sampler_chunk") if os.path.exists(p) else "missing"
    out["sampler_chunk"] = chunks
    out["same_sampler_chunk"] = bool(chunks["runC"] == chunks["runU"] and chunks["runC"] != "missing")
    return out


def control_d7(ctx, gain=0.10, base_frac=0.5):
    """Planted difference for the two-level reader on the labels of ``ctx`` (four seed indices): (1) a gain at every seed reads separated above (lower end above 0, all four above 0); (2) a gain at one of four
    does not; (3) four identical pairs read exactly 0 with the interval [0, 0]; (4) the swapped order reads the mirrored interval and the negative mean."""
    y = ctx["G15"][:, S0F]
    m = ctx["valid"] & np.isfinite(y)
    var = float(np.var(y[m]))
    r = np.random.default_rng(5)
    zs = [r.normal(size=len(y)) for _ in SEEDS]
    b_out = [{"gap_m": y + np.sqrt(base_frac * var) * z} for z in zs]
    a_out = [{"gap_m": y + np.sqrt((base_frac - gain) * var) * z} for z in zs]

    def read(gains, swap=False):
        pts, Ds = [], []
        for i in range(len(SEEDS)):
            x, z = (a_out[i] if gains[i] else b_out[i]), b_out[i]
            p, d = boot_diff(ctx, "M2", z, x) if swap else boot_diff(ctx, "M2", x, z)
            pts.append(p)
            Ds.append(d)
        return two_level(pts, np.array(Ds), 0)
    all4, one, none, swapped = read([1] * 4), read([1, 0, 0, 0]), read([0] * 4), read([1] * 4, swap=True)
    ok = bool(all4["ci95"][0] > 0 and all4["n_pos"] == 4 and abs(all4["mean"] - gain) < 0.03 and not _above(one) and one["n_pos"] == 1 and none["mean"] == 0.0 and none["ci95"] == [0.0, 0.0]
              and swapped["mean"] < 0 and _below(swapped) and abs(swapped["mean"] + all4["mean"]) < 1e-9)
    return {"pass": ok, "gain_at_every_seed": all4, "gain_at_one_of_four": one, "identical_pairs": none, "swapped_order": swapped, "planted_gain": gain}


def gated(flat):
    """Every control gates except the closing controls k3_* (note N1: no closing read) and the entries that only describe."""
    return {k: v for k, v in FR.gated(flat).items() if not k.startswith("k3_")}


# ============================================================================================ run
def _files(in_dir, folder, arm):
    d = "%s/%s" % (in_dir, folder)
    return TKR.arm_files(d, d, arm)


def read_all(a, ctx, cache):
    R, outs, ctrl, p5 = {}, {}, {}, {}
    for label, (folder, arm, steps, sh, corpus) in ARM_MAP.items():
        files = _files(a.in_dir, folder, arm)
        R[label], out, c = TKR.read_arm(label, files, ctx, None, a.B, False, True, cache)
        outs[label] = out
        ctrl.update(c)
        ctrl["records_probe_form|" + label] = check_records_u(files, label, steps, arm)
        lp = "%s/%s/%s/train_log.jsonl" % (a.run_dir, folder, arm)
        p5[label] = p5_u(lp, sh, steps, a.expect_corpus_clips if corpus else None) if os.path.exists(lp) else None
        ctrl["d4_train_log|" + label] = {"pass": bool(p5[label] and p5[label]["pass"]), "given": p5[label] is not None, "detail": p5[label]}
    return R, outs, ctrl


def run(a):
    t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    P = N.load_pack(a.pack_dir + "/eval_pack.npz")
    ctx = N.window_ctx(P, a.B)
    cache = {}
    pre = json.load(open(a.precheck_json, encoding="utf-8")) if os.path.exists(a.precheck_json) else None
    planted = json.load(open(a.planted_json, encoding="utf-8")) if os.path.exists(a.planted_json) else None
    ctrl = {"a2_pack_counts": {"pass": bool(ctx["cnt"]["windows"] == N.EXPECT["windows"] and ctx["cnt"]["scored"] == N.EXPECT["scored"] and ctx["n10"] == N.EXPECT["episodes_scored"])}}
    ctrl.update(precheck_controls(pre, planted))
    clock = json.load(open(a.clock_json, encoding="utf-8")) if os.path.exists(a.clock_json) else None
    clock_planted = json.load(open(a.clock_planted_json, encoding="utf-8")) if os.path.exists(a.clock_planted_json) else None
    ctrl.update(clock_controls(clock, clock_planted))
    es = json.load(open(a.evalset_json, encoding="utf-8")) if os.path.exists(a.evalset_json) else None
    es_planted = json.load(open(a.evalset_planted_json, encoding="utf-8")) if os.path.exists(a.evalset_planted_json) else None
    ctrl.update(evalset_controls(es, es_planted))
    ctrl.update(TKR.paired_read_controls(ctx))
    ctrl["d7_planted_difference_two_level"] = control_d7(ctx)
    R, outs, c = read_all(a, ctx, cache)
    ctrl.update(c)
    for k in (0, 1):
        ctrl["d3_corpus_arm_live|" + lab("C", k)] = d3_corpus_live(R[lab("C", k)], lab("C", k)) if R[lab("C", k)].get("read") else {"pass": False, "reason": "arm not read"}
    for nm in ("P16CSH_s0", "P16USH_s0"):
        ctrl["d5_permuted_gap_dead|" + nm] = d5_permuted_dead(R[nm], nm) if R[nm].get("read") else {"pass": False, "reason": "arm not read"}
    flat = FR.flatten(ctrl)
    bad = sorted(k for k, v in gated(flat).items() if not v)
    res = {"tier": TIER_LINE, "families": list(FAMILY_LINES), "card": "PREREG_V8F_PROOF_ADD45 + N1", "B": a.B, "evidence_class": "MEASURED once run", "controls": ctrl, "controls_flat": flat, "controls_failed": bad,
           "controls_pass": not bad, "limits": {"d6": d6_limit(a.run_dir), "d3t_tokens_same": (es or {}).get("d3t_tokens_same", "no record")}, "arms": {}}
    for label in ARM_MAP:
        r = R[label]
        res["arms"][label] = {"M2": r["M2"]["head"], "M4": r["M4"]} if r.get("read") else {"read": False}
    if bad:
        res.update({"names": None, "outcome_differences": "withheld: a control did not pass (card section 5): " + ", ".join(bad)})
    else:
        oc = compute_outcomes_u(ctx, outs)
        for label in oc["summary"]:
            res["arms"][label]["gap_mae"] = oc["summary"][label]["MAE"]
        res.update({"two_level": oc["two_level"], "names": oc["names"], "beside": oc["beside"]})
    if a.done_marker and os.path.exists(a.done_marker):
        res["done_marker_lines"] = [ln.rstrip() for ln in open(a.done_marker, encoding="utf-8")][:6]
    res["elapsed_s"] = round(time.time() - t0, 1)
    N.jdump(res, os.path.join(a.out_dir, OUT_JSON))
    write_verdict(os.path.join(a.out_dir, OUT_MD), res)
    log("controls", "PASS" if not bad else "FAILED", (res["names"] or {}).get("name") if res["names"] else "no name", "elapsed %.0f s" % res["elapsed_s"])
    if any(isinstance(v, dict) and v.get("pass") is False and k.startswith("a1_records") for k, v in ctrl.items()):
        return 3
    return 0 if not bad else 2


# ============================================================================================ writing
def _ci(c):
    return "[%+.4f, %+.4f]" % (c[0], c[1])


def write_verdict(path, res):
    L = ["# VERDICT_DSCALE_LEAD: card PREREG_V8F_PROOF_ADD45 with note N1 (DSCALE-LEAD)", "", res["tier"], ""] + ["- " + x for x in res["families"]]
    L += ["", "Evidence class of every number: %s. B %d; two-level bootstrap (seed indices, then episodes; rng 0); the 429 scored windows of 48 evaluation episodes. M1 is not read (note N1)." % (res["evidence_class"], res["B"]), "",
          "## Controls (card section 5 with note N1; all must pass before any name is given)", ""]
    for k in sorted(res["controls_flat"]):
        L.append("- %s: %s%s" % (k, "PASS" if res["controls_flat"][k] else "FAIL", " (printed, gates nothing: closing control, no closing read)" if k.startswith("k3_") else ""))
    d6 = res["limits"]["d6"]
    L += ["- d6 (a limit, not a pass): %s; sampler chunk %s; same chunk: %s" % (d6["statement"], json.dumps(d6["sampler_chunk"]), d6["same_sampler_chunk"]), ""]
    d3 = {k: v for k, v in res["controls"].items() if k.startswith("d3_corpus_arm_live")}
    for k, v in d3.items():
        if "gap_r2" in v:
            L.append("- beside %s: gap R2 %+.3f; row 97 range of the eight TK16 arms %.3f to %.3f; inside: %s (gates nothing)" % (k, v["gap_r2"], ROW97_RANGE[0], ROW97_RANGE[1], v["beside_inside_row97_range"]))
    L.append("")
    if res["names"] is None:
        L += ["## Names", "", "No name is given and no outcome difference is printed: %s" % res["outcome_differences"], ""]
    else:
        n = res["names"]
        L += ["## Name (card section 4 with note N1)", "", "- gap read: **%s**" % n["name"], "- closing read: %s" % n["closing_name"], "- gap error clause (mean over seeds, U against C): ratio %.3f, bar 0.90: %s"
              % (n["mae_ratio_U_to_C"], "met" if n["mae_clause_met"] else "not met"), "", "## Two-level reads (mean over the four seed indices of P16U minus P16C; per-seed values listed)", ""]
        for m, unit in (("M2", "R2"), ("MAE", "metres, negative = U better")):
            t = res["two_level"][m]
            L.append("- %s (%s): mean %+.4f %s; seeds above 0: %d of %d; per seed %s" % (m, unit, t["mean"], _ci(t["ci95"]), t["n_pos"], t["n_seeds"], ", ".join("%+.4f" % x for x in t["per_seed"])))
        b = res["beside"]
        L += ["", "## Beside (names nothing)", "", "- standard deviation over the four seeds of M2: %s" % json.dumps(b["std_over_seeds_of_M2"]),
              "- the M2 interval with an independent episode draw per chosen seed: %s" % _ci(b["m2_two_level_independent_episode_draw_per_seed"]["ci95"]), "- distance of each deciding comparison from its bar: %s" % json.dumps(n["margins_to_the_bars_beside"])]
        if "P16U2_s0" in b:
            u = b["P16U2_s0"]
            L.append("- P16U2_s0 (twice the steps): gap R2 %+.3f, gap error %.2f m; minus P16U_s0 on M2 %+.4f, on the gap error %+.3f m; minus P16C_s0 on M2 %+.4f. %s" % (
                u["summary"]["M2"]["r2"], u["summary"]["MAE"]["mae_m"], u["minus_P16U_s0"]["M2"], u["minus_P16U_s0"]["MAE"], u["minus_P16C_s0"]["M2"], u["note"]))
        L.append("")
    L += ["## Per arm (reported; M4 and the control arms name nothing)", ""]
    for nm, Rr in res["arms"].items():
        if "M2" not in Rr:
            L.append("- %s: not read" % nm)
            continue
        L.append("- %s: M2 gap R2 %+.3f; M4 AUROC %.3f%s" % (nm, Rr["M2"]["r2"], Rr["M4"]["auroc"], "; gap error %.2f m" % Rr["gap_mae"]["mae_m"] if "gap_mae" in Rr else ""))
    L += ["", "## Limits", "", "One frozen encoder; pooled tokens; the evaluation windows come from the corpus only, so a gain on 10k-like scenes is not seen; four seeds bound the run-to-run variance of the probe and not that of the encoder; "
          "matched steps mean the union arm sees each clip less often (P16U2_s0 is reported beside); the windows of the two arm types differ (d6); a probe says what the tokens allow, not what a model does with them.", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="OUT/states: one sub-folder per run folder (runC runU runU2 runCSH runUSH) with the evaluation states of its arms")
    ap.add_argument("--run-dir", required=True, help="OUT of pod_dscale_lead_run.sh: run folders with train logs and r1_batches.json; precheck/")
    ap.add_argument("--precheck-json", default=None)
    ap.add_argument("--planted-json", default=None)
    ap.add_argument("--clock-json", default=None, help="result of dscale_clockcheck.py (default: <run-dir>/precheck/CLOCK.json)")
    ap.add_argument("--clock-planted-json", default=None, help="its planted-shift run (default: <run-dir>/precheck/CLOCK_PLANTED.json)")
    ap.add_argument("--evalset-json", default=None, help="result of dscale_tokens_same.py (default: <run-dir>/precheck/EVALSET.json)")
    ap.add_argument("--evalset-planted-json", default=None, help="its planted-swap run (default: <run-dir>/precheck/EVALSET_PLANTED.json)")
    ap.add_argument("--done-marker", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pack-dir", default=TKR.READ)
    ap.add_argument("--expect-corpus-clips", type=int, default=4512)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args(argv)
    a.precheck_json = a.precheck_json or a.run_dir + "/precheck/PRECHECK.json"
    a.planted_json = a.planted_json or a.run_dir + "/precheck/PRECHECK_PLANTED.json"
    a.clock_json = a.clock_json or a.run_dir + "/precheck/CLOCK.json"
    a.clock_planted_json = a.clock_planted_json or a.run_dir + "/precheck/CLOCK_PLANTED.json"
    a.evalset_json = a.evalset_json or a.run_dir + "/precheck/EVALSET.json"
    a.evalset_planted_json = a.evalset_planted_json or a.run_dir + "/precheck/EVALSET_PLANTED.json"
    for f in FOLDERS:
        if not os.path.isdir("%s/%s" % (a.in_dir, f)):
            print("MISSING: states folder %s/%s" % (a.in_dir, f))
            return 3
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
