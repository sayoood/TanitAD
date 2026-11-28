r"""finegrid_r_read.py -- the reader of card PREREG_V8F_PROOF_ADD43 (FINEGRID-R), sections 3 to 5. Tier T0 (a diagnostic on logged frames, never driving performance). CPU only, numpy only.

Reuses the pilot's reader `finegrid_read_n1.py` (which reuses tokprobe_read / leadsup_read_n1): per-arm M1, M2, M4, the gap MAE, the episode draw `C10`, the record checks and the controls k1 to k4
are imported, not re-derived. This file adds: eight seed indices (K = 2 to 9), the two-level bootstrap, the controls of card section 4 (g1 to g7 with the p5 form for permuted-label arms, g6 for both
control arms, r1, r2, r3) and the outcome rule of card section 5. It must run with the working directory that makes the pilot reader's fixed paths (`F:/TanitAD-scratch/v8proof/...`) resolve
(on the pod: the mirror folder `rk`, see run_pod_read_r.sh).

  python finegrid_r_read.py --in-dir STATES --run-dir RUN --g2 G2_GRIDS.json --controls-json FILE --done-marker FILE --out-dir DIR [--pilot-f-dir DIR] [--g7-tol 1e-5] [--B 2000]
  python finegrid_r_read.py --tk-only --out-dir DIR      (the reproduction of the banked TK reads, control g7)
Exit 0 = every control passed (a name of section 5 is written), 2 = a control failed (no name, no outcome difference anywhere), 3 = an input record is missing.
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
SEEDS = tuple(range(2, 10))                                           # seed indices K = 2 to 9
TYPES = ("TK16", "TK64", "F16")
SH_ARMS = ("TK16SH_s2", "F16SH_s2")
STEPS_MAIN, STEPS_P3 = 19164, 2000
G6_BAR = 0.02
MAE_FRACTION = 0.10
MIN_POS_SEEDS = 6
OUT_JSON, OUT_MD = "RESULT_FINEGRID_R.json", "VERDICT_FINEGRID_R.md"
TIER_TEXT = FR.TIER_TEXT
FAMILY_TEXT = FR.FAMILY_TEXT
NAMES_GAP = ("FINER-GRID-READS-GAP-R", "FINER-GRID-WORSE-R", "FINER-GRID-FLAT-R")
NAME_CLOSING = "FINER-GRID-READS-CLOSING-R"
RUN_DIRS = {"TK": ("train_A", "p3_A"), "F": ("train_B", "p3_B")}      # where the train logs of an arm family live under the run folder
GRID_OF = {"TK": ("pool4x10", 40), "F": ("pool8x20", 160)}
PAIRS = {"G16": ("F16", "TK16"), "G16w": ("F16", "TK64")}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def arm(t, k):
    return "%s_s%d" % (t, k)


def main_arms():
    return [arm(t, k) for t in TYPES for k in SEEDS]


# ============================================================================================ the two-level bootstrap
def _yp(ctx, measure, out):
    if measure == "M1":
        return ctx["c_lab"], out["closing_ms"]
    return ctx["G15"][:, S0F], out["gap_m"]


def boot_diff(ctx, measure, oa, ob):
    """(point difference a - b, bootstrap vector over the episode draw C10) of one paired measure on the same windows: the arithmetic of FR.pair_r2 / FR.pair_mae (M2, M1: pooled R2; MAE: metres),
    with the vector kept so that the seed level can be resampled on top. `RP.ci` of the vector and the point equal the pilot reader's paired row (a test asserts it)."""
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
    pts [n_seeds] = the point differences; D [n_seeds, B] = their bootstrap vectors over the SAME episode draws. inner = 'shared': one episode draw per replicate for every chosen seed (the
    evaluation set is the same for all seeds: a crossed design; the decision form); 'per_seed': an independent episode draw per chosen seed (beside only)."""
    pts, D = np.asarray(pts, float), np.asarray(D, float)
    n, B = D.shape
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(B, n))                              # SEED-LEVEL: the seed indices with replacement
    if inner == "shared":
        vals = D[idx, np.arange(B)[:, None]].mean(1)
    else:
        vals = D[idx, rng.integers(0, B, size=(B, n))].mean(1)
    return {"mean": float(pts.mean()), "ci95": RP.ci(vals), "per_seed": [float(x) for x in pts], "n_seeds": int(n), "n_pos": int((pts > 0).sum()), "n_neg": int((pts < 0).sum()), "inner": inner}


def two_level_pair(ctx, outs, pair, measure, inner="shared"):
    a_t, b_t = PAIRS[pair]
    pts, Ds = [], []
    for k in SEEDS:
        p, d = boot_diff(ctx, measure, outs[arm(a_t, k)], outs[arm(b_t, k)])
        pts.append(p)
        Ds.append(d)
    return two_level(pts, np.array(Ds), 0, inner)


# ============================================================================================ outcomes (card section 5)
def _above(t):
    return bool(t["mean"] > 0 and t["ci95"][0] > 0 and t["n_pos"] >= MIN_POS_SEEDS)


def _below(t):
    return bool(t["mean"] < 0 and t["ci95"][1] < 0)


def decide_r(tl, mae_means):
    pos2 = {p: _above(tl[p]["M2"]) for p in PAIRS}
    neg2 = {p: _below(tl[p]["M2"]) for p in PAIRS}
    lower = min(mae_means["TK16"], mae_means["TK64"])
    ratio = float(mae_means["F16"] / lower)
    mae_ok = bool(mae_means["F16"] <= (1.0 - MAE_FRACTION) * lower)
    if all(pos2.values()) and mae_ok:
        gap = NAMES_GAP[0]
    elif all(neg2.values()):
        gap = NAMES_GAP[1]
    else:
        gap = NAMES_GAP[2]
    pos1 = {p: _above(tl[p]["M1"]) for p in PAIRS}
    closing = NAME_CLOSING if all(pos1.values()) else "none (a flat closing read names nothing)"
    margins = {"m2_ci_lower_G16": tl["G16"]["M2"]["ci95"][0], "m2_ci_lower_G16w": tl["G16w"]["M2"]["ci95"][0], "m2_ci_upper_G16": tl["G16"]["M2"]["ci95"][1], "m2_ci_upper_G16w": tl["G16w"]["M2"]["ci95"][1],
               "mae_ratio_minus_0.90": ratio - (1.0 - MAE_FRACTION), "m1_ci_lower_G16": tl["G16"]["M1"]["ci95"][0], "m1_ci_lower_G16w": tl["G16w"]["M1"]["ci95"][0]}
    return {"gap": gap, "closing": closing, "m2_above_0_with_n_pos_at_least_6": pos2, "m2_below_0": neg2, "m1_above_0_with_n_pos_at_least_6": pos1, "mean_gap_mae_by_type_m": mae_means,
            "mae_ratio_F16_to_lower_of_TK": ratio, "mae_clause_met": mae_ok, "margins_to_the_bars_beside": margins}


def compute_outcomes_r(ctx, outs):
    """Everything that compares arms: per-arm summaries, the two-level reads of G16 and G16w on M2, the gap MAE and M1, the names, the beside reads. outs: arm -> dict(gap_m, closing_ms)."""
    summ = {a: FR.summarize(ctx, outs[a]) for a in main_arms()}
    tl = {p: {m: two_level_pair(ctx, outs, p, m) for m in ("M2", "MAE", "M1")} for p in PAIRS}
    mae_means = {t: float(np.mean([summ[arm(t, k)]["MAE"]["mae_m"] for k in SEEDS])) for t in TYPES}
    std_m2 = {t: float(np.std([summ[arm(t, k)]["M2"]["r2"] for k in SEEDS], ddof=1)) for t in TYPES}
    tl_ps = {p: two_level_pair(ctx, outs, p, "M2", "per_seed") for p in PAIRS}
    gr = FR.gap_range_table(ctx, outs, tuple(main_arms()))
    gr_mean = {t: {b: float(np.mean([gr[arm(t, k)][b]["mae_m"] for k in SEEDS if gr[arm(t, k)][b]["mae_m"] is not None])) if any(gr[arm(t, k)][b]["mae_m"] is not None for k in SEEDS) else None
                   for b in ("below 20 m", "20 to 40 m", "above 40 m")} for t in TYPES}
    return {"summary": summ, "two_level": tl, "names": decide_r(tl, mae_means),
            "beside": {"std_over_seeds_of_M2_by_type": std_m2, "m2_two_level_independent_episode_draw_per_seed_beside": tl_ps, "gap_mae_by_gap_range_mean_over_seeds": gr_mean,
                       "M4_by_arm": "see arms"}}


# ============================================================================================ control r3 (planted difference for the two-level reader)
def control_r3(ctx, gain=0.10, base_frac=0.5):
    """Synthetic head outputs from the labels of ``ctx``: per seed index the b arm has an error of variance base_frac * Var(y), the a arm the same noise draw scaled to (base_frac - gain) * Var(y), so the paired
    R2 gain is about ``gain``. (1) a gain at every seed is read as separated above (interval lower end above 0, all eight seeds above 0); (2) a gain at two of eight seeds and none at six is not;
    (3) eight identical pairs read exactly 0 with the interval [0, 0]."""
    y = ctx["G15"][:, S0F]
    m = ctx["valid"] & np.isfinite(y)
    var = float(np.var(y[m]))
    r = np.random.default_rng(5)
    zs = [r.normal(size=len(y)) for _ in SEEDS]
    b_out = [{"gap_m": y + np.sqrt(base_frac * var) * z} for z in zs]
    a_out = [{"gap_m": y + np.sqrt((base_frac - gain) * var) * z} for z in zs]

    def read(gains):
        pts, Ds = [], []
        for i in range(len(SEEDS)):
            p, d = boot_diff(ctx, "M2", a_out[i] if gains[i] else b_out[i], b_out[i])
            pts.append(p)
            Ds.append(d)
        return two_level(pts, np.array(Ds), 0)
    all8, two, none = read([1] * 8), read([1, 1, 0, 0, 0, 0, 0, 0]), read([0] * 8)
    ok = bool(all8["ci95"][0] > 0 and all8["n_pos"] == 8 and abs(all8["mean"] - gain) < 0.03 and not (two["ci95"][0] > 0 and two["n_pos"] >= MIN_POS_SEEDS) and two["n_pos"] == 2
              and none["mean"] == 0.0 and none["ci95"] == [0.0, 0.0])
    return {"pass": ok, "gain_at_every_seed": all8, "gain_at_two_of_eight": two, "identical_pairs": none, "planted_gain": gain}


# ============================================================================================ controls: r1, g5/r2, p5, g6
def r1_compare(path_a, path_b):
    """Control r1 (same logic as tok_probe_gr.r1_compare): the two processes logged the same window indices in the same order for the first 200 batches."""
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        return {"pass": False, "reason": "no r1_batches.json in one of the two train folders", "files": [path_a, path_b]}
    a, b = json.load(open(path_a, encoding="utf-8")), json.load(open(path_b, encoding="utf-8"))
    same = bool(a["idx_hashes"] == b["idx_hashes"] and a["full_hashes"] == b["full_hashes"] and len(a["idx_hashes"]) > 0 and a["n_batches"] == b["n_batches"])
    return {"pass": same, "n_a": a["n_batches"], "n_b": b["n_batches"], "sampler_chunk_a": a.get("sampler_chunk"), "sampler_chunk_b": b.get("sampler_chunk"), "files": [path_a, path_b]}


def family(name):
    return "TK" if name.startswith("TK") else "F"


def width(name):
    return int(name[len(family(name)):].split("_")[0].replace("SH", ""))


def check_records_r(files, name, steps_expected):
    """Control g5 (grid, token count, tokens seen in training, state width) and r2 (the probe of card ADD27: width, input normalisation, step count) on the eval state record of an arm."""
    p = files["eval_states"] + ".json"
    if not os.path.exists(p):
        return {"pass": False, "reason": "no record " + p}
    r = json.load(open(p, encoding="utf-8"))
    grid, ntok = GRID_OF[family(name)]
    W = width(name)
    g5 = bool(r.get("grid") == grid and r.get("n_tok") == ntok and r.get("n_tok_seen_in_training") == ntok and r.get("D") == ntok * W and r.get("W") == W)
    r2 = bool(r.get("input_norm") == "ln" and r.get("step") == steps_expected and r.get("arm") == name)
    return {"pass": bool(g5 and r2), "g5_grid_tokens_width": g5, "r2_probe_form": r2, "grid": r.get("grid"), "n_tok": r.get("n_tok"), "n_tok_seen_in_training": r.get("n_tok_seen_in_training"), "D": r.get("D"),
            "W": r.get("W"), "step": r.get("step"), "input_norm": r.get("input_norm")}


def p5_form(path, window=1000):
    """Control p5. For an arm with permuted closing labels (SH in the name): the lead term read on its INTACT parts (presence + gap) of the last 1,000 steps below those of the first 1,000, gap-labelled frames above
    0 at every logged step, run complete; the permuted closing part is reported beside and gates nothing. Every other arm: the check of card ADD27 as written (FR.p5_card_form)."""
    recs = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
    hdr = recs[0]
    if "SH_" not in str(hdr.get("arm")):
        d = FR.p5_card_form(path, window)
        d["form"] = "card ADD27 as written"
        return d
    body = [x for x in recs[1:] if not x["first"]]
    total = int(hdr["total"])

    def part(keys, sel):
        v = [sum(x[k] for k in keys) for x in body if sel(x["step"])]
        return float(np.mean(v)) if v else None
    keys = ("lead_sup_bce_mean", "lead_sup_gap_mean")
    fi, la = part(keys, lambda s: s <= window), part(keys, lambda s: s > total - window)
    gap_ok = bool(min(x["n_gap"] for x in recs[1:]) > 0)
    done = bool(recs[1:] and recs[-1]["step"] == total)
    return {"pass": bool(fi is not None and la is not None and la < fi and gap_ok and done), "form": "intact parts (presence + gap); closing part permuted", "intact_first_mean": fi, "intact_last_mean": la,
            "permuted_closing_part_first_mean": part(("lead_sup_clos_mean",), lambda s: s <= window), "permuted_closing_part_last_mean": part(("lead_sup_clos_mean",), lambda s: s > total - window),
            "min_n_gap_instantaneous": min(x["n_gap"] for x in recs[1:]), "last_step": recs[-1]["step"], "total": total}


def tool_sha_control(given, done_marker, tool="tok_probe_gr.py"):
    """The dev-box controls g1, g3, g4, g6 were run on a tool with a recorded sha256; the pod's finished marker records the sha256 of the tool that trained. They must be the same file."""
    want = ((given.get("notes") or {}).get("code_sha256") or {}).get(tool)
    got = None
    if done_marker and os.path.exists(done_marker):
        for ln in open(done_marker, encoding="utf-8"):
            if ln.startswith(tool + " sha256:"):
                got = ln.split(":", 1)[1].strip()
    return {"pass": bool(want and got and want == got), "controls_json_sha256": want, "pod_marker_sha256": got}


def train_log_path(run_dir, name, sh):
    return "%s/%s/%s/train_log.jsonl" % (run_dir, RUN_DIRS[family(name)][1 if sh else 0], name)


# ============================================================================================ run
def _files(in_dir, name):
    return TKR.arm_files(in_dir, in_dir, name)


def read_new_arms(a, ctx, cache):
    R, outs, ctrl, g5, p5 = {}, {}, {}, {}, {}
    for name in main_arms() + list(SH_ARMS):
        sh = name in SH_ARMS
        files = _files(a.in_dir, name)
        R[name], out, c = TKR.read_arm(name, files, ctx, None, a.B, False, not sh, cache)
        outs[name] = out
        ctrl.update(c)
        g5["records|" + name] = check_records_r(files, name, STEPS_P3 if sh else STEPS_MAIN)
        lp = train_log_path(a.run_dir, name, sh)
        p5[name] = p5_form(lp) if os.path.exists(lp) else None
    return R, outs, ctrl, g5, p5


def pilot_beside(a, ctx, outs_tk, cache):
    """Beside, naming nothing: the pilot's seeds 0 and 1 (F arms of the pilot, TK arms of card ADD27), not pooled with the new seeds."""
    if not a.pilot_f_dir:
        return {"read": False, "reason": "no --pilot-f-dir"}
    o, summ = dict(outs_tk), {}
    for nm in ("F16_s0", "F16_s1", "F64_s0", "F64_s1"):
        try:
            R, out, _ = TKR.read_arm(nm, _files(a.pilot_f_dir, nm), ctx, None, a.B, False, False, cache)
            o[nm] = out if R.get("read") else None
        except Exception as e:  # a missing pilot file is reported, not fatal
            o[nm] = None
            log("pilot arm", nm, "not read:", repr(e)[:120])
    for nm, out in o.items():
        if out is not None:
            summ[nm] = FR.summarize(ctx, out)
    pairs = {}
    for s in ("s0", "s1"):
        for k, (x, y) in FR.pair_names(s).items():
            if k.startswith("G64"):
                continue
            if o.get(x) is not None and o.get(y) is not None:
                pairs[k] = FR.pair_all(ctx, o[x], o[y])
    return {"read": True, "summary": summ, "paired": pairs, "note": "the pilot's seeds 0 and 1: reported beside, not pooled with the new seeds, name nothing"}


def run(a):
    t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    P = N.load_pack(a.pack_dir + "/eval_pack.npz")
    PT = N.load_pack(a.pack_dir + "/train_pack.npz")
    ctx = N.window_ctx(P, a.B)
    cache = {}
    fa = argparse.Namespace(banked_result=a.banked_result, in_tk=a.in_tk, tk_states_dir=a.tk_states_dir, B=a.B, g7_tol=a.g7_tol)
    R_tk, outs_tk, ctrl_tk, repro = FR.read_tk(fa, ctx, PT, cache)
    if a.tk_only:
        res = {"tier": TIER_TEXT, "card": "PREREG_V8F_PROOF_ADD43 control g7 (TK side)", "reproduction": repro, "pass": bool(all(r["pass"] for r in repro.values()))}
        N.jdump(res, os.path.join(a.out_dir, "TK_REPRO_G7.json"))
        log("g7 TK reproduction", "PASS" if res["pass"] else "FAIL")
        return 0 if res["pass"] else 2
    ctrl = dict(ctrl_tk)
    ctrl.update({"k1_label_path": N.control_k1(ctx, P["meta"]["k1_reference_r2_skill"]),
                 "a2_pack_counts": {"pass": bool(ctx["cnt"]["windows"] == N.EXPECT["windows"] and ctx["cnt"]["scored"] == N.EXPECT["scored"] and ctx["n10"] == N.EXPECT["episodes_scored"])}})
    ctrl.update(TKR.paired_read_controls(ctx))
    R_new, outs, c_new, g5, p5 = read_new_arms(a, ctx, cache)
    ctrl.update(c_new)
    ctrl.update({"g5_r2_records|" + k.split("|")[1]: v for k, v in g5.items()})
    for nm in SH_ARMS:                                                                    # g6 for both control arms: closing R2 at most +0.02
        r2c = R_new[nm]["M1"]["r2"] if R_new[nm].get("read") else None
        ctrl["g6_shuffled_closing|" + nm] = {"pass": bool(r2c is not None and r2c <= G6_BAR), "closing_r2_scored_windows": r2c, "bar": G6_BAR}
    ctrl["g2_grids_belong_together"] = FR.check_g2(a.g2)
    for nm, v in repro.items():
        ctrl["g7_tk_reproduces_banked|" + nm] = v
    ctrl["r1_same_windows_same_order"] = r1_compare(a.run_dir + "/train_A/r1_batches.json", a.run_dir + "/train_B/r1_batches.json")
    ctrl["r3_planted_difference_two_level"] = control_r3(ctx)
    given = FR.lead_controls(a.controls_json)
    ctrl.update(FR.given_controls(given, {k: v for k, v in p5.items() if v is not None}, main_arms() + list(SH_ARMS)))
    for nm, v in p5.items():
        if v is not None:
            ctrl["p5_train_log|" + nm]["detail"] = v
    ctrl["tool_sha_matches_the_devbox_controls"] = tool_sha_control(given, a.done_marker)
    flat = FR.flatten(ctrl)
    bad = sorted(k for k, v in FR.gated(flat).items() if not v)
    res = {"tier": TIER_TEXT, "families": FAMILY_TEXT, "card": "PREREG_V8F_PROOF_ADD43", "B": a.B, "evidence_class": "MEASURED once run", "controls": ctrl, "controls_flat": flat, "controls_failed": bad,
           "controls_pass": not bad, "arms": {}}
    for nm in main_arms() + list(SH_ARMS):
        R = {**R_new}[nm]
        res["arms"][nm] = ({"M1": R["M1"], "M2": R["M2"]["head"], "M4": R["M4"]} if R.get("read") else {"read": False})
    if bad:
        res.update({"names": None, "outcome_differences": "withheld: a control did not pass (card section 4): " + ", ".join(bad)})
    else:
        oc = compute_outcomes_r(ctx, outs)
        for nm in main_arms():
            res["arms"][nm]["gap_mae"] = oc["summary"][nm]["MAE"]
        res.update({"two_level": oc["two_level"], "names": oc["names"], "beside": oc["beside"], "pilot_seeds_beside": pilot_beside(a, ctx, outs_tk, cache)})
    res["elapsed_s"] = round(time.time() - t0, 1)
    N.jdump(res, os.path.join(a.out_dir, OUT_JSON))
    write_verdict(os.path.join(a.out_dir, OUT_MD), res)
    log("controls", "PASS" if not bad else "FAILED", (res["names"] or {}).get("gap") if res["names"] else "no name", "elapsed %.0f s" % res["elapsed_s"])
    if any(isinstance(v, dict) and v.get("pass") is False and k.startswith("a1_records") for k, v in ctrl.items()):
        return 3
    return 0 if not bad else 2


# ============================================================================================ writing
def _ci(c):
    return "[%+.4f, %+.4f]" % (c[0], c[1])


def write_verdict(path, res):
    L = ["# VERDICT_FINEGRID_R: card PREREG_V8F_PROOF_ADD43 (FINEGRID-R)", "", "Tier %s. %s." % (res["tier"], res["families"]),
         "Evidence class of every number: %s. B %d; two-level bootstrap (seed indices, then episodes; rng 0); the 429 scored windows of 48 evaluation episodes." % (res["evidence_class"], res["B"]), "",
         "## Controls (card section 4; all must pass before any outcome is named)", ""]
    for k in sorted(res["controls_flat"]):
        L.append("- %s: %s%s" % (k, "PASS" if res["controls_flat"][k] else "FAIL", " (printed, gates nothing)" if k.startswith("k3_card_form_slot") else ""))
    L.append("")
    if res["names"] is None:
        L += ["## Names", "", "No name is given and no outcome difference is printed: %s" % res["outcome_differences"], ""]
    else:
        n = res["names"]
        L += ["## Names (card section 5)", "", "- gap read: **%s**" % n["gap"], "- closing read: **%s**" % n["closing"], "- gap MAE clause (mean over seeds, F16 against the lower of TK16 and TK64): ratio %.3f, bar 0.90: %s"
              % (n["mae_ratio_F16_to_lower_of_TK"], "met" if n["mae_clause_met"] else "not met"), "", "## Two-level reads (mean over the eight seed indices of F minus TK; the eight per-seed values listed)", ""]
        for p in PAIRS:
            for m, unit in (("M2", "R2"), ("MAE", "metres, negative = F better"), ("M1", "R2")):
                t = res["two_level"][p][m]
                L.append("- %s on %s (%s): mean %+.4f %s; seeds above 0: %d of %d; per seed %s" % (p, m, unit, t["mean"], _ci(t["ci95"]), t["n_pos"], t["n_seeds"], ", ".join("%+.4f" % x for x in t["per_seed"])))
        b = res["beside"]
        L += ["", "## Beside (names nothing)", "", "- standard deviation over the eight seeds of M2 per arm type: %s" % json.dumps(b["std_over_seeds_of_M2_by_type"]),
              "- gap MAE (metres) by gap range, mean over seeds: %s" % json.dumps(b["gap_mae_by_gap_range_mean_over_seeds"]),
              "- the M2 interval with an independent episode draw per chosen seed: %s" % json.dumps({p: v["ci95"] for p, v in b["m2_two_level_independent_episode_draw_per_seed_beside"].items()}),
              "- distance of each deciding comparison from its bar: %s" % json.dumps(n["margins_to_the_bars_beside"]),
              "- the pilot's seeds 0 and 1 (not pooled): %s" % ("see RESULT_FINEGRID_R.json, pilot_seeds_beside" if res.get("pilot_seeds_beside", {}).get("read") else "not read")]
        L.append("")
    L += ["## Per arm (reported; M4 and the control arms name nothing)", ""]
    for nm, R in res["arms"].items():
        if "M2" not in R:
            L.append("- %s: not read" % nm)
            continue
        L.append("- %s: M2 gap R2 %+.3f; M1 closing R2 %+.3f; M4 AUROC %.3f%s" % (nm, R["M2"]["r2"], R["M1"]["r2"], R["M4"]["auroc"], "; gap MAE %.2f m" % R["gap_mae"]["mae_m"] if "gap_mae" in R else ""))
    L += ["", "## Limits", "", "A probe on the tokens says what the tokens allow, not what a model does with them; 429 scored windows of 48 episodes; one corpus; tokens stored as float8; the grid before pooling is not read; "
          "eight seeds bound the probe's run-to-run variance and not the variance of the encoder (one frozen encoder).", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=None, help="folder with <ARM>_eval.npz(+.json) and <ARM>_eval_lead.npz(+.json,+.head.npz) of the 26 new arms")
    ap.add_argument("--run-dir", default=None, help="OUT of pod_finegrid_r_run.sh: train_A, train_B, p3_A, p3_B (train logs and r1_batches.json)")
    ap.add_argument("--g2", default=None)
    ap.add_argument("--controls-json", default=None, help="dev-box results g1, g3, g4, g6_forward with notes.code_sha256")
    ap.add_argument("--done-marker", default=None, help="FINEGRID_R_DONE.txt (carries the sha256 of the tool that trained)")
    ap.add_argument("--pilot-f-dir", default=None, help="pilot states of the F arms (seeds 0 and 1), beside only")
    ap.add_argument("--in-tk", default=FR.DEFAULT_IN_TK)
    ap.add_argument("--tk-states-dir", default=FR.DEFAULT_TK_STATES)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pack-dir", default=TKR.READ)
    ap.add_argument("--banked-result", default=FR.BANKED_RESULT)
    ap.add_argument("--tk-only", action="store_true")
    ap.add_argument("--g7-tol", type=float, default=0.0, help="tolerance of the TK reproduction of control g7 (0.0 as frozen; 1e-5 on the pod)")
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args(argv)
    if not a.tk_only and not (a.in_dir and a.run_dir):
        raise SystemExit("REFUSED: give --in-dir and --run-dir (or --tk-only)")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
