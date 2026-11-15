r"""tokprobe_read.py -- the reader of card PREREG_V8F_PROOF_ADD27 (TOKPROBE) with its note PREREG_V8F_PROOF_ADD27_N1 and control p2b.
Tier T0 (a diagnostic on logged frames, never driving performance). CPU only, reads files, touches no pod, no box 2, no git. STRATEGIC: PAUSED (PI item 25).

The metric logic is IMPORTED from spatial/leadsup/read/leadsup_read_n1.py (M1, M2, M4, DELTA of note ADD24_N1, the episode-cluster bootstrap, the paired difference `pair_m1`, the
controls k1, k2, k3-closing, k3-slot (N1), k4, the record checks). This file adds: the arm layout of the probe arms, the paired read against the LSD arm of the same seed index,
the floor, the paired-read controls, p3, the inputs p1 / p2a / p2b / p5 given by the lead, and the names of card section 5 with the rule of note N1 section 2.

  python tokprobe_read.py --in-dir DIR --states-dir DIR --out-dir DIR [--ref-in-dir DIR --ref-states-dir DIR] [--sens-in-dir DIR --sens-states-dir DIR]
                          [--controls-json FILE | --p-flags p1=true,p2a=true,p2b=true,p5=true] [--alias TK16_s0=LSD_s0,...] [--B 2000]
Exit 0 = every control passed, 2 = a control failed (no name is written for the arm types it touches), 3 = an input record failed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402

READ = "F:/TanitAD-scratch/v8proof/spatial/leadsup/read"
if READ not in sys.path:
    sys.path.insert(0, READ)
import leadsup_read_n1 as N  # noqa: E402

RP, O = N.RP, N.O
HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/")
OUT_JSON = "RESULT_TOKPROBE.json"
OUT_MD = "VERDICT_TOKPROBE.md"
TYPES = ("TK16", "TK64")
SEEDS = ("s0", "s1")
PROBE_ARMS = tuple("%s_%s" % (t, s) for t in TYPES for s in SEEDS)
REF_ARMS = ("LSD_s0", "LSD_s1")
P3_ARM = "TK16SH_s0"
P3_BAR = 0.02
PAIRED_TOL = 1e-9
TIER_TEXT = "T0 (a diagnostic on logged frames, never driving performance)"
NO_READ_TEXT = "no read above the bar; sensitivity of the instrument not shown"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================================ files
def parse_alias(text):
    return dict(x.split("=") for x in (text or "").split(",") if x)


def arm_files(in_dir, states_dir, arm, alias=None):
    """Files of an arm; an alias loads the files of ANOTHER arm under this arm's name (lead outputs and states alike)."""
    nm = (alias or {}).get(arm, arm)
    sd = states_dir or in_dir
    f = {"eval_states": "%s/%s_eval.npz" % (sd, nm), "train_states": "%s/%s_train.npz" % (sd, nm)}
    for sp in ("eval", "train"):
        f[sp + "_lead"] = "%s/%s_%s_lead.npz" % (in_dir, nm, sp)
    return f


# ============================================================================================ one arm (the per-arm steps of leadsup_read_n1.run, same functions)
def read_arm(arm, files, ctx, PT, B, do_train=True, controls=True, cache=None):
    """Returns (R, out, ctrl). R has M1, M2, M3 (incl. delta_n1), M4 on the scored windows and, with do_train, the same on the train probe clips. ctrl is {control|arm: ...}."""
    key = (tuple(sorted(files.items())), bool(do_train), bool(controls))
    if cache is not None and key in cache:
        R, out, ctrl = cache[key]
        return R, out, {k.split("|")[0] + "|" + arm: v for k, v in ctrl.items()}
    ctrl = {}
    recs, ck = N.check_records(files, False, do_train)
    ctrl["a1_records|%s" % arm] = recs
    if not recs["pass"]:
        log(arm, "input records failed; arm not read")
        return {"files": files, "ckpt_sha256": ck, "read": False}, None, ctrl
    Th_e = N.load_head(files["eval_lead"] + ".head.npz")
    Th_t = None
    if do_train:
        Th_t = N.load_head(files["train_lead"] + ".head.npz")
        same = all(np.array_equal(Th_e[k], Th_t[k]) for k in N.HEAD_KEYS) and Th_e["hist"] == Th_t["hist"]
        ctrl["a3_same_head_eval_train|%s" % arm] = {"pass": bool(same)}
    st = N.load_states(files["eval_states"])
    skey = N.state_key(st)
    Z15, rows = N.assemble_windows(st["Z"], skey, ctx["eid"], ctx["g0"])
    missing_hist = int((rows[:, :N.S0F + 1] < 0).sum())
    finite = bool(np.isfinite(Z15[:, :N.S0F + 1].astype(np.float32)).all())
    ctrl["a4_window_states|%s" % arm] = {"pass": bool(missing_hist == 0 and finite), "window_frames_0to4_without_state": missing_hist, "window_states_0to4_finite": finite}
    idx_e, _ = N._history_idx(st, Th_e["hist"])
    ze = np.load(files["eval_lead"], allow_pickle=False)
    pod_e = {k: ze[k] for k in ("presence_logit", "gap_m", "closing_ms", "has_history")}
    ctrl["k2_head_forward_eval|%s" % arm] = N.control_k2(Th_e, st["Z"], idx_e, pod_e, rows[:, N.S0F])
    out = N.window_outputs(Th_e, Z15)
    R, _PV = N.eval_read(ctx, out, None, None)
    R["ckpt_sha256"], R["files"], R["read"] = ck, files, True
    if controls:
        k3c, k3f = N.control_k3_split(Th_e, Z15, ctx)
        ctrl["k3_closing|%s" % arm] = k3c
        ctrl["k3_card_form_slot|%s" % arm] = k3f
        ctrl["k3_slot_n1|%s" % arm] = N.control_k3_slot_n1(out["gap_m"], out["closing_ms"], ctx)
        ctrl["k4_no_future_input|%s" % arm] = N.control_k4(Th_e, Z15, ctx)
    if do_train:
        stt = N.load_states(files["train_states"])
        lay = {"clips_equal": bool(list(map(str, stt["clips"])) == list(map(str, PT["clips"]))), "clip_idx_equal": bool(np.array_equal(stt["clip_idx"].astype(np.int64), PT["clip_idx"])),
               "t_equal": bool(np.array_equal(stt["t"].astype(np.int64), PT["t"])), "n_rows": int(len(stt["Z"]))}
        lay["pass"] = bool(lay["clips_equal"] and lay["clip_idx_equal"] and lay["t_equal"])
        ctrl["a5_train_layout|%s" % arm] = lay
        if lay["pass"]:
            idx_t, _ = N._history_idx(stt, Th_t["hist"])
            zt = np.load(files["train_lead"], allow_pickle=False)
            pod_t = {k: zt[k] for k in ("presence_logit", "gap_m", "closing_ms", "has_history")}
            rows_ok = np.flatnonzero((idx_t >= 0).all(1))
            ctrl["k2_head_forward_train|%s" % arm] = N.control_k2(Th_t, stt["Z"], idx_t, pod_t, rows_ok)
            Rt, _, _ = N.train_read(Th_t, stt["Z"], PT, B)
            R["train"] = Rt
        del stt
    del st, Z15
    if cache is not None:
        cache[key] = (R, out, ctrl)
    return R, out, ctrl


# ============================================================================================ paired read and its controls
def paired_m1(ctx, ca, cb):
    return N.pair_m1(ctx, ca, cb)


def _r2_expected(ctx, ca):
    m = ctx["valid"] & np.isfinite(ca) & np.isfinite(ctx["c_lab"]) & (ctx["ep10"] >= 0)
    y = ctx["c_lab"][m]
    return float(1.0 - ((y - ca[m]) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def paired_read_controls(ctx, seed=11):
    """Controls of the paired read (note N1 section 3): an arm against itself reads exactly 0; the swapped order reads the negative; a planted shift is read back
    (the difference of the pooled R2 of two planted predictors, computed here without the reader, within 1e-9; the interval excludes 0)."""
    rng = np.random.default_rng(seed)
    lab = np.where(np.isfinite(ctx["c_lab"]), ctx["c_lab"], 0.0)
    a = lab + rng.normal(size=lab.shape) * 1.5
    b = lab + rng.normal(size=lab.shape) * 2.5
    ss = paired_m1(ctx, a, a)
    self_ok = bool(ss["diff"] == 0.0 and ss["ci95"][0] == 0.0 and ss["ci95"][1] == 0.0)
    ab, ba = paired_m1(ctx, a, b), paired_m1(ctx, b, a)
    swap_ok = bool(abs(ab["diff"] + ba["diff"]) <= PAIRED_TOL and abs(ab["ci95"][0] + ba["ci95"][1]) <= PAIRED_TOL and abs(ab["ci95"][1] + ba["ci95"][0]) <= PAIRED_TOL and ab["diff"] != 0.0)
    pa, pb = lab.copy(), lab + 1.0                                          # a perfect predictor against the same predictor shifted by 1 m/s
    pl = paired_m1(ctx, pa, pb)
    exp = _r2_expected(ctx, pa) - _r2_expected(ctx, pb)
    planted_ok = bool(abs(pl["diff"] - exp) <= PAIRED_TOL and pl["separated_pos"] and pl["ci95"][0] <= exp <= pl["ci95"][1] + 1e-9)
    return {"paired_self_reads_exactly_0": {"pass": self_ok, "diff": ss["diff"], "ci95": ss["ci95"]},
            "paired_swap_reads_the_negative": {"pass": swap_ok, "diff_ab": ab["diff"], "diff_ba": ba["diff"], "ci95_ab": ab["ci95"], "ci95_ba": ba["ci95"]},
            "paired_planted_shift_read_back": {"pass": planted_ok, "diff": pl["diff"], "expected": exp, "ci95": pl["ci95"]}}


# ============================================================================================ inputs of the lead and the names
def parse_flags(text):
    d = {}
    for kv in (text or "").split(","):
        if not kv:
            continue
        k, v = kv.split("=")
        d[k.strip()] = (v.strip().lower() in ("1", "true", "yes", "pass"))
    return d


def _truthy(v, arms=None):
    if isinstance(v, bool):
        return v
    if isinstance(v, dict):
        if arms is None:
            return bool(v) and all(bool(x) for x in v.values())
        return all(bool(v.get(a, False)) for a in arms)
    return False


def lead_inputs(controls_json, flags):
    d = {}
    if controls_json:
        d.update(json.load(open(controls_json, encoding="utf-8")))
    d.update(parse_flags(flags))
    return d


def type_arms(typ):
    return ["%s_%s" % (typ, s) for s in SEEDS]


def failed_controls_for_type(typ, ctrl, given, p3):
    """Every control that touches the arm type: global ones, the controls of its two arms and of the two LSD arms (their outputs enter the paired read), p3 and the inputs p1, p2a, p5.
    p2b is NOT in this list: it blocks the flat name only. k3_card_form_slot is printed and gates nothing (note ADD24_N1)."""
    arms = set(type_arms(typ)) | set(REF_ARMS)
    flat = N._flatten_controls(ctrl)
    bad = []
    for k, ok in flat.items():
        fam, _, a = k.partition("|")
        if fam == "k3_card_form_slot":
            continue
        if (a == "" or a in arms) and not ok:
            bad.append(k)
    if not p3["pass"]:
        bad.append("p3")
    for p in ("p1", "p2a"):
        if not _truthy(given.get(p)):
            bad.append("%s (%s)" % (p, "false" if p in given else "not given"))
    if not _truthy(given.get("p5"), sorted(arms) if isinstance(given.get("p5"), dict) else None):
        bad.append("p5 (%s)" % ("false" if "p5" in given else "not given"))
    return bad


def name_type(typ, arm_m1, pairs, floor, failed, p2b):
    """Card section 5 with note N1 section 2. arm_m1: arm -> {'r2', 'r2_ci95'} or None; pairs: arm -> {'diff', 'ci95'} or None; floor: float or None; failed: list; p2b: bool or None."""
    arms = type_arms(typ)
    if any(arm_m1.get(a) is None or pairs.get(a) is None for a in arms):
        return {"name": "not named", "reason": "one seed names nothing (an arm or its LSD partner was not read)"}
    if floor is None:
        return {"name": "not named", "reason": "the floor needs both LSD arms"}
    if failed:
        return {"name": "not named", "reason": "a control failed: " + ", ".join(failed)}
    m1_ok = all(arm_m1[a]["r2"] > 0 and arm_m1[a]["r2_ci95"][0] > 0 for a in arms)
    pr_ok = all(pairs[a]["diff"] > 0 and pairs[a]["ci95"][0] > 0 for a in arms)
    mean_d = float(np.mean([pairs[a]["diff"] for a in arms]))
    gt_floor = bool(mean_d > floor)
    info = {"m1_above_0_both_seeds": m1_ok, "paired_above_0_both_seeds": pr_ok, "mean_paired_diff": mean_d, "floor": floor, "mean_above_floor": gt_floor}
    if m1_ok and pr_ok and gt_floor:
        return dict(name="TOKENS-HOLD-CLOSING", reason="M1, the paired difference to LSD and its mean above the floor all clear at both seeds", **info)
    if p2b is True:
        return dict(name="TOKENS-FLAT", reason="at least one of the three conditions of the card is not met; control p2b passed", **info)
    return dict(name="not named", reason=NO_READ_TEXT + (" (p2b %s)" % ("false" if p2b is False else "not given")), **info)


# ============================================================================================ run
def _m1(R):
    return {"r2": R["M1"]["r2"], "r2_ci95": R["M1"]["r2_ci95"]} if R and R.get("read") else None


def run(a):
    t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    alias = parse_alias(a.alias)
    ref_in = a.ref_in_dir or a.in_dir
    ref_st = a.ref_states_dir or a.states_dir or ref_in
    P = N.load_pack(a.pack_dir + "/eval_pack.npz")
    PT = N.load_pack(a.pack_dir + "/train_pack.npz")
    ctx = N.window_ctx(P, a.B)
    given = lead_inputs(a.controls_json, a.p_flags)
    res = {"tier": TIER_TEXT, "families": "LONGITUDINAL (distance keeping) is the family this lever serves and is not measured by this read; LATERAL, TACTICAL: not measured; STRATEGIC: PAUSED (PI item 25)",
           "card": "PREREG_V8F_PROOF_ADD27 with PREREG_V8F_PROOF_ADD27_N1", "B": a.B, "alias": alias, "lead_inputs": given, "arms": {}, "paired": {}, "controls": {}, "evidence_class": "MEASURED once run"}
    ctrl = {"k1_label_path": N.control_k1(ctx, P["meta"]["k1_reference_r2_skill"]),
            "a2_pack_counts": {"pass": bool(ctx["cnt"]["windows"] == N.EXPECT["windows"] and ctx["cnt"]["scored"] == N.EXPECT["scored"] and ctx["n10"] == N.EXPECT["episodes_scored"])}}
    ctrl.update(paired_read_controls(ctx))
    cache, outs = {}, {}
    for arm in list(PROBE_ARMS):
        R, out, c = read_arm(arm, arm_files(a.in_dir, a.states_dir, arm, alias), ctx, PT, a.B, True, True, cache)
        res["arms"][arm], outs[arm] = R, out
        ctrl.update(c)
    for arm in REF_ARMS:
        R, out, c = read_arm(arm, arm_files(ref_in, ref_st, arm, alias), ctx, PT, a.B, True, True, cache)
        res["arms"][arm], outs[arm] = R, out
        ctrl.update(c)
    R3, o3, c3 = read_arm(P3_ARM, arm_files(a.in_dir, a.states_dir, P3_ARM, alias), ctx, PT, a.B, False, False, cache)
    res["arms"][P3_ARM] = R3
    p3_r2 = R3["M1"]["r2"] if R3.get("read") else None
    p3 = {"pass": bool(p3_r2 is not None and p3_r2 <= P3_BAR), "closing_r2_scored_windows": p3_r2, "bar": P3_BAR, "arm": P3_ARM,
          "note": "closing-speed R2 of the arm trained on permuted closing labels; the record checks of this arm: %s" % ("pass" if R3.get("read") else "FAILED or missing")}
    if alias.get(P3_ARM):
        p3["alias_of"] = alias[P3_ARM]
    ctrl["p3_shuffled_labels"] = p3
    ctrl.update({k: v for k, v in c3.items() if k.startswith("a1_records")})
    for t in TYPES:
        for s in SEEDS:
            pa, rb = "%s_%s" % (t, s), "LSD_%s" % s
            if outs.get(pa) is not None and outs.get(rb) is not None:
                res["paired"][pa + " minus " + rb] = paired_m1(ctx, outs[pa]["closing_ms"], outs[rb]["closing_ms"])
    floor = abs(res["arms"]["LSD_s0"]["M1"]["r2"] - res["arms"]["LSD_s1"]["M1"]["r2"]) if all(res["arms"][r].get("read") for r in REF_ARMS) else None
    res["floor_lsd_seeds_abs_diff_m1_r2"] = floor
    res["controls"] = ctrl
    res["controls_failed_by_type"], res["names"] = {}, {}
    for t in TYPES:
        arm_m1 = {x: _m1(res["arms"].get(x)) for x in type_arms(t)}
        pairs = {x: res["paired"].get("%s minus LSD_%s" % (x, x[-2:])) for x in type_arms(t)}
        failed = failed_controls_for_type(t, ctrl, given, p3)
        res["controls_failed_by_type"][t] = failed
        res["names"][t] = name_type(t, arm_m1, pairs, floor, failed, given.get("p2b") if isinstance(given.get("p2b"), bool) else None)
    flat = N._flatten_controls(ctrl)
    res["controls_pass"] = bool(all(v for k, v in flat.items() if not k.startswith("k3_card_form_slot")) and all(not f for f in res["controls_failed_by_type"].values()))
    # sensitivity pass: tokens not normalised, same four arm names; reported apart, names nothing
    if a.sens_in_dir:
        sens = {"note": "sensitivity pass (tokens not normalised, `--input-norm none`); names nothing", "arms": {}, "paired": {}}
        for arm in PROBE_ARMS:
            R, out, _c = read_arm(arm, arm_files(a.sens_in_dir, a.sens_states_dir or a.sens_in_dir, arm, alias), ctx, PT, a.B, False, False, cache)
            sens["arms"][arm] = R
            rb = "LSD_%s" % arm[-2:]
            if out is not None and outs.get(rb) is not None:
                sens["paired"][arm + " minus " + rb] = paired_m1(ctx, out["closing_ms"], outs[rb]["closing_ms"])
        res["sensitivity"] = sens
    res["elapsed_s"] = round(time.time() - t0, 1)
    N.jdump(res, os.path.join(a.out_dir, OUT_JSON))
    write_verdict(os.path.join(a.out_dir, OUT_MD), res)
    log("controls", "PASS" if res["controls_pass"] else "FAILED", {t: res["names"][t]["name"] for t in TYPES}, "elapsed %.0f s" % res["elapsed_s"])
    if any(isinstance(v, dict) and v.get("pass") is False and k.startswith("a1_records") for k, v in ctrl.items()):
        return 3
    return 0 if res["controls_pass"] else 2


# ============================================================================================ writing
def _ci(c):
    return "[%+.3f, %+.3f]" % (c[0], c[1])


def _pair_line(k, d):
    return "- %s: %+.4f %s; n=%d%s" % (k, d["diff"], _ci(d["ci95"]), d["n"], " (separated above 0)" if d["separated_pos"] else (" (separated below 0)" if d["separated_neg"] else ""))


def write_verdict(path, res):
    L = ["# VERDICT_TOKPROBE: card PREREG_V8F_PROOF_ADD27 (TOKPROBE) with note N1", "",
         "Tier %s. %s." % (res["tier"], res["families"]), "Evidence class of every number: %s. B %d, episode-cluster bootstrap, rng 0." % (res["evidence_class"], res["B"]), ""]
    if res.get("alias"):
        L += ["ALIAS RUN: arm names are loaded from other files (%s); no number of this run is a result of the probe." % json.dumps(res["alias"]), ""]
    L += ["## Names (card section 5; both seeds needed)", ""]
    for t in TYPES:
        n = res["names"][t]
        L.append("- %s: **%s** (%s)" % (t, n["name"], n["reason"]))
        if "floor" in n:
            L.append("  - M1 above 0 at both seeds: %s; paired difference above 0 at both seeds: %s; mean paired difference %+.4f against floor %.4f: %s"
                     % (n["m1_above_0_both_seeds"], n["paired_above_0_both_seeds"], n["mean_paired_diff"], n["floor"], "above" if n["mean_above_floor"] else "not above"))
    L += ["", "## Controls", "", "Inputs of the lead: " + json.dumps(res["lead_inputs"]) + " (p2b blocks only the flat name; a missing input counts as not passed).", ""]
    flat = N._flatten_controls(res["controls"])
    for k in sorted(flat):
        L.append("- %s: %s%s" % (k, "PASS" if flat[k] else "FAIL", " (printed, gates nothing)" if k.startswith("k3_card_form_slot") else ""))
    p3 = res["controls"]["p3_shuffled_labels"]
    L.append("- p3 (%s): closing-speed R2 on the scored windows %s, bar at most %+.2f: %s" % (p3["arm"], "n/a" if p3["closing_r2_scored_windows"] is None else "%+.3f" % p3["closing_r2_scored_windows"], p3["bar"], "PASS" if p3["pass"] else "FAIL"))
    for t, f in res["controls_failed_by_type"].items():
        L.append("- failed controls touching %s: %s" % (t, ", ".join(f) if f else "none"))
    L += ["", "## Paired difference of M1 (probe arm minus the LSD arm of the same seed index)", ""]
    for k, d in res["paired"].items():
        L.append(_pair_line(k, d))
    L.append("- floor (|M1 LSD_s0 - M1 LSD_s1|): %s" % ("not available" if res["floor_lsd_seeds_abs_diff_m1_r2"] is None else "%.4f" % res["floor_lsd_seeds_abs_diff_m1_r2"]))
    L += ["", "## Per arm (reported; M2, M4, DELTA and the train probe clips name nothing)", ""]
    for arm, R in res["arms"].items():
        L.append("### %s" % arm)
        if not R.get("read"):
            L += ["- not read (input records failed or missing)", ""]
            continue
        L += N._block(R, "scored windows")
        if "train" in R:
            L.append("- train probe clips (in-sample ceiling):")
            L += ["  " + x for x in N._block(R["train"], "train probe clips") if x]
        L.append("")
    if "sensitivity" in res:
        L += ["## Sensitivity pass (tokens not normalised; names nothing)", ""]
        for arm, R in res["sensitivity"]["arms"].items():
            if R.get("read"):
                L.append("- %s: M1 R2 %+.3f %s; M2 gap R2 %+.3f; M4 AUROC %.3f; DELTA %+.3f" % (arm, R["M1"]["r2"], _ci(R["M1"]["r2_ci95"]), R["M2"]["head"]["r2"], R["M4"]["auroc"], R["M3"]["delta_n1"]["skill"]))
            else:
                L.append("- %s: not read" % arm)
        for k, d in res["sensitivity"]["paired"].items():
            L.append(_pair_line(k + " (sensitivity)", d))
        L.append("")
    L += ["## Limits", "", "A probe on the tokens says what the pooled tokens allow, not what the model does; 429 scored windows of 48 evaluation episodes; one corpus. "
          "The paired bootstrap resamples episodes with the arms held fixed; the two seeds differ in initialisation only, and the floor comes from the two LSD runs.", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--states-dir", default=None)
    ap.add_argument("--ref-in-dir", default=None)
    ap.add_argument("--ref-states-dir", default=None)
    ap.add_argument("--sens-in-dir", default=None)
    ap.add_argument("--sens-states-dir", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pack-dir", default=READ)
    ap.add_argument("--controls-json", default=None)
    ap.add_argument("--p-flags", default="", help="p1=true,p2a=true,p2b=true,p5=true")
    ap.add_argument("--alias", default="", help="TK16_s0=LSD_s0,... (tests: load the files of the second name under the first)")
    ap.add_argument("--B", type=int, default=2000)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
