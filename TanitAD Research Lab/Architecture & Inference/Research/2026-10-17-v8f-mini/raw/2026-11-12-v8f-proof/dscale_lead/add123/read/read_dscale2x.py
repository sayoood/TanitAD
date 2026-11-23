r"""read_dscale2x.py -- the reader of card PREREG_V8F_PROOF_ADD123 (DSCALE-LEAD-2X). Tier T0. CPU only, numpy only.

TIER: T0 (a probe on cached tokens of logged frames; it says what the tokens allow, not what a model does with them); never driving performance.

The banked reader of card ADD45 (dscale_lead_read.py) and its wrapper of note N5 (dscale_lead_read_n5.py) are IMPORTED and never edited: the window context, the per-probe read, the paired
difference (boot_diff), the two-level bootstrap (two_level, two_level_pair), the controls d3 / d5 / d7 and the precheck controls are the banked functions. This file adds the new run's probes at three
steps, the measures S, L, F of card section 3, the controls q0 to q6 of card section 5 and the names of card section 4.

  python read_dscale2x.py controls --banked-run-dir D --banked-in-dir D --new-run-dir D --new-in-dir D --banked-done F --new-done F --pack-dir D --out-dir D --test-out F --test-before EPOCH
  python read_dscale2x.py names    (the same arguments)

Mode controls: writes DSCALE2X_CONTROLS.json and prints one line per control. It computes no S, L, F, no name, and writes no gap R2 or gap error of a new probe.
Mode names: refuses (exit 3) unless the controls file exists with all_pass true and the same input hashes; then writes DSCALE2X_RESULT.json and DSCALE2X_VERDICT.md and prints the name line.
Exit codes: 0 ok, 2 a control failed, 3 refused (an input is missing, or the controls file does not allow the names).
The kit folder of the banked reader is found through the environment variable DSCALE_LEAD_KIT_DIR (default: two folders above this file).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/")
KIT = os.environ.get("DSCALE_LEAD_KIT_DIR") or os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, KIT)
import dscale_lead_read as D  # noqa: E402
import dscale_lead_read_n5 as N5  # noqa: E402

N, RP, S0F = D.N, D.RP, D.S0F
SEEDS = D.SEEDS
STEPS_MAIN, STEPS_Q = 19164, 38328
STEPS_MID, STEPS_LATE = 28746, 38328
NEW_STEPS = (19164, 28746, 38328)
DOSE_BAR = 0.020
MAE_FRACTION = 0.10
LOSS_TOL_Q1 = 1e-4
CONTROL_NAMES = ("q0", "q1", "q2", "q3", "q4", "q5", "q6")
NAMES = ("LEAD-READ-2X-SCALES-GAP", "LEAD-READ-2X-WORSE", "LEAD-READ-2X-FLAT", "LEAD-READ-2X-SHORT-OF-TRAINING")
CONTROLS_FILE, RESULT_FILE, VERDICT_FILE = "DSCALE2X_CONTROLS.json", "DSCALE2X_RESULT.json", "DSCALE2X_VERDICT.md"
TIER_LINE = D.TIER_LINE
FAMILY_LINES = D.FAMILY_LINES
LOSS_TERMS = ("lead_sup_bce", "lead_sup_gap", "lead_sup_bce_mean", "lead_sup_gap_mean", "lead_sup_clos_mean")
LOSS_TERMS_REQUIRED = ("lead_sup_bce", "lead_sup_gap")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def lab_q(k, step=None):
    return "P16Q_s%d" % k if step is None else "P16Q_s%d@%d" % (k, step)


# ============================================================================================ the probe table
def probe_specs():
    """label -> dict(side banked|new, folder, arm, steps, sh (permuted gap labels), corpus). The new probes are listed per step: P16Q_sK@STEP; P16Q_sK is the last step."""
    sp = {}
    for label, (folder, arm, steps, sh, corpus) in D.ARM_MAP.items():
        sp[label] = {"side": "banked", "folder": folder, "arm": arm, "steps": steps, "sh": sh, "corpus": corpus}
    folders = {19164: "runQ_s19164", 28746: "runQ_s28746", 38328: "runQ"}
    for k in SEEDS:
        for st in NEW_STEPS:
            sp[lab_q(k, st)] = {"side": "new", "folder": folders[st], "arm": "TK16_s%d" % k, "steps": st, "sh": False, "corpus": False}
    return sp


def read_probe(label, spec, files, ctx, a, cache):
    """The per-probe read of the banked chain (tokprobe_read.read_arm). Replaceable in tests. Returns (R, out, ctrl)."""
    return D.TKR.read_arm(label, files, ctx, None, a.B, False, False, cache)


def build_ctx(a):
    P = N.load_pack(a.pack_dir + "/eval_pack.npz")
    return N.window_ctx(P, a.B)


def eval_ids(files):
    z = np.load(files["eval_states"], allow_pickle=False)
    return [str(x) for x in z["clips"]]


def state_files(a, spec):
    in_dir = a.banked_in_dir if spec["side"] == "banked" else a.new_in_dir
    return D._files(in_dir, spec["folder"], spec["arm"])


def log_path(a, spec):
    run_dir = a.banked_run_dir if spec["side"] == "banked" else a.new_run_dir
    return "%s/%s/%s/train_log.jsonl" % (run_dir, "runQ" if spec["side"] == "new" else spec["folder"], spec["arm"])


def load_all(a, ctx, specs):
    P, cache = {}, {}
    for label, sp in specs.items():
        files = state_files(a, sp)
        try:
            R, out, c = read_probe(label, sp, files, ctx, a, cache)
            P[label] = {"R": R, "out": out, "ctrl": c, "files": files, "read": bool(R.get("read") and out is not None)}
        except Exception as e:                                   # an unreadable probe fails the controls that need it; it is never skipped
            P[label] = {"R": None, "out": None, "ctrl": {}, "files": files, "read": False, "error": repr(e)}
    return P


# ============================================================================================ small file helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def _json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def parse_done(path):
    """Lines `ckpt <folder>/<arm> <sha256>` and `ckpt_step <folder>/<arm> <step> <sha256>`; first line = the marker line."""
    out = {"first": None, "ckpt": {}, "ckpt_step": {}}
    if not path or not os.path.exists(path):
        return out
    for i, ln in enumerate(open(path, encoding="utf-8")):
        t = ln.split()
        if i == 0:
            out["first"] = ln.strip()
        if len(t) == 3 and t[0] == "ckpt":
            parts = t[1].split("/")                                    # the run script of card ADD123 writes ckpt <folder>/<arm>/ckpt_last and .../ckpt_step<N>
            if len(parts) == 3 and parts[2] == "ckpt_last":
                out["ckpt"]["/".join(parts[:2])] = t[2]
            elif len(parts) == 3 and parts[2].startswith("ckpt_step") and parts[2][len("ckpt_step"):].isdigit():
                out["ckpt_step"][("/".join(parts[:2]), int(parts[2][len("ckpt_step"):]))] = t[2]
            else:
                out["ckpt"][t[1]] = t[2]
        elif len(t) == 4 and t[0] == "ckpt_step":
            try:
                out["ckpt_step"][(t[1], int(t[2]))] = t[3]
            except ValueError:
                pass
    return out


def record_of(files):
    return _json(files["eval_states"] + ".json") or {}


def _jd(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def unique_path(d, name):
    """Never overwrite: name, then name_2, name_3, ..."""
    base, ext = os.path.splitext(name)
    p, i = os.path.join(d, name), 2
    while os.path.exists(p):
        p = os.path.join(d, "%s_%d%s" % (base, i, ext))
        i += 1
    return p


def latest_path(d, name):
    base, ext = os.path.splitext(name)
    best, i = None, 1
    while True:
        p = os.path.join(d, name if i == 1 else "%s_%d%s" % (base, i, ext))
        if os.path.exists(p):
            best = p
        elif i > 1:
            break
        i += 1
        if i > 1000:
            break
    return best


def input_hashes(a, specs):
    """sha256 of every small input the controls read: done files, the new run's record files, the batch logs, the precheck records and the record file of every probe state (it binds the state file's hash)."""
    paths = [a.banked_done, a.new_done, a.new_run_dir + "/runQ/r1_batches.json", a.banked_run_dir + "/runU2/r1_batches.json", a.new_run_dir + "/runQ/CK_RECORD.json",
             a.new_run_dir + "/runQ/TRAIN_DONE.json"]
    paths += ["%s/precheck/%s.json" % (a.new_run_dir, n) for n in ("PRECHECK", "PRECHECK_PLANTED", "CLOCK", "CLOCK_PLANTED", "EVALSET", "EVALSET_PLANTED")]
    for label, sp in specs.items():
        paths.append(state_files(a, sp)["eval_states"] + ".json")
    for k in SEEDS:
        paths.append("%s/runQ/TK16_s%d/train_log.jsonl" % (a.new_run_dir, k))
    return {p: (sha256_file(p) if os.path.exists(p) else "MISSING") for p in paths}


# ============================================================================================ the controls q0 to q6
def q0_inputs(a, specs):
    done = parse_done(a.banked_done)
    rows, ok = {}, True
    for label, (folder, arm, _s, _sh, _c) in D.ARM_MAP.items():
        key = "%s/%s" % (folder, arm)
        rec = record_of(D._files(a.banked_in_dir, folder, arm))
        good = bool(key in done["ckpt"] and rec.get("ckpt_sha256") and rec.get("ckpt_sha256") == done["ckpt"][key])
        rows[label] = good
        ok = ok and good
    pre_dir = a.new_run_dir + "/precheck"
    pre, pre_p = _json(pre_dir + "/PRECHECK.json"), _json(pre_dir + "/PRECHECK_PLANTED.json")
    ck, ck_p = _json(pre_dir + "/CLOCK.json"), _json(pre_dir + "/CLOCK_PLANTED.json")
    es, es_p = _json(pre_dir + "/EVALSET.json"), _json(pre_dir + "/EVALSET_PLANTED.json")
    pc = {}
    pc.update(D.precheck_controls(pre, pre_p))
    pc.update(D.clock_controls(ck, ck_p))
    pc.update(D.evalset_controls(es, es_p))
    pc_pass = {k: bool(v.get("pass")) for k, v in pc.items()}
    ok = bool(ok and pc_pass and all(pc_pass.values()))
    return {"pass": ok, "banked_done_hash_equals_record": rows, "precheck_controls": pc_pass}


def _first_record(path):
    if not os.path.exists(path):
        return None
    for i, ln in enumerate(open(path, encoding="utf-8")):
        if i == 1 and ln.strip():
            return json.loads(ln)
    return None


def _all_lines(path):
    return [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()] if os.path.exists(path) else None


def q1_saving_tool(a):
    rq, ru = _json(a.new_run_dir + "/runQ/r1_batches.json"), _json(a.banked_run_dir + "/runU2/r1_batches.json")
    batches_equal = bool(rq is not None and ru is not None and rq == ru)
    diff_keys = None
    if isinstance(rq, dict) and isinstance(ru, dict):
        diff_keys = sorted(k for k in set(rq) | set(ru) if rq.get(k) != ru.get(k))[:20]
    pn, pb = a.new_run_dir + "/runQ/TK16_s0/train_log.jsonl", a.banked_run_dir + "/runU2/TK16_s0/train_log.jsonl"
    fn, fb = _first_record(pn), _first_record(pb)
    terms, ok_terms = {}, bool(fn is not None and fb is not None and fn.get("step") == 1 and fb.get("step") == 1)
    if fn is not None and fb is not None:
        for t in LOSS_TERMS:
            if t in fb or t in fn:
                good = bool(t in fb and t in fn and abs(float(fn[t]) - float(fb[t])) <= LOSS_TOL_Q1)
                terms[t] = good
                ok_terms = ok_terms and good
        ok_terms = bool(ok_terms and all(t in terms for t in LOSS_TERMS_REQUIRED))
    ln, lb = _all_lines(pn), _all_lines(pb)
    return {"pass": bool(batches_equal and ok_terms), "r1_batches_equal_entry_for_entry": batches_equal, "r1_differing_top_level_keys": diff_keys, "first_record_terms_within_1e-4": terms,
            "first_record_step_1_in_both": bool(fn is not None and fb is not None and fn.get("step") == 1 and fb.get("step") == 1),
            "printed_not_judged_full_logs_equal": bool(ln is not None and lb is not None and ln == lb)}


def q2_corpus_live(P):
    flags = {}
    for k in SEEDS:
        for label in (lab_q(k, STEPS_Q), D.lab("C", k)):
            p = P[label]
            flags[label] = bool(p["read"] and D.d3_corpus_live(p["R"], label)["pass"])
    return {"pass": bool(all(flags.values())), "per_probe_pass": flags}


def q3_permuted_dead(P):
    flags = {}
    for label in ("P16CSH_s0", "P16USH_s0"):
        p = P[label]
        flags[label] = bool(p["read"] and D.d5_permuted_dead(p["R"], label)["pass"])
    return {"pass": bool(all(flags.values())), "per_probe_pass": flags}


def q4_runs_complete(a, P, specs):
    done = parse_done(a.new_done)
    run_ok = bool(done["first"] and done["first"].split()[0] == "DSCALE_2X_DONE")
    rows, ok, beside = {}, run_ok, {}
    for k in SEEDS:
        arm = "TK16_s%d" % k
        lp = "%s/runQ/%s/train_log.jsonl" % (a.new_run_dir, arm)
        row = {}
        try:
            r = D.p5_u(lp, False, STEPS_Q, None)
            row["header_fits_the_arm"] = bool(r["header_checks"] and all(r["header_checks"].values()))
            row["last_step_is_38328"] = bool(r.get("last_step") == STEPS_Q)
            _, body = D._read_log(lp)
            g = N5.gap_clause_n5(body)
            mg = g.get("min_n_gap_full_batches")
            row["every_full_batch_record_has_gap_frames"] = bool(mg is not None and mg > 0)
            beside[arm] = {"first_mean_lead_term": r.get("first_mean"), "last_mean_lead_term": r.get("last_mean"), "fall_clause_as_frozen_pass": bool(r["pass"]),
                           "short_records_share_printed_not_judged": g.get("short_share")}
        except Exception as e:
            row["log_error"] = repr(e)
            row["header_fits_the_arm"] = row["last_step_is_38328"] = row["every_full_batch_record_has_gap_frames"] = False
        for st in NEW_STEPS:
            label = lab_q(k, st)
            files = P[label]["files"]
            rec = D.check_records_u(files, label, st, arm)
            rec_sha = record_of(files).get("ckpt_sha256")
            want = done["ckpt_step"].get(("runQ/%s" % arm, st))
            if want is None and st == STEPS_Q:
                want = done["ckpt"].get("runQ/%s" % arm)
            row["state_%d_record_step_and_arm" % st] = bool(rec["pass"])
            row["state_%d_ckpt_hash_equals_done_file" % st] = bool(want and rec_sha and want == rec_sha)
        rows[arm] = row
        ok = bool(ok and all(v for kk, v in row.items() if kk != "log_error"))
    return {"pass": bool(ok), "done_marker_line_ok": run_ok, "per_probe": rows, "beside": beside}


def tests_clause(path, before):
    ok_file = bool(path and os.path.exists(path))
    txt = open(path, encoding="utf-8", errors="replace").read() if ok_file else ""
    passed = [int(x) for x in re.findall(r"(\d+) passed", txt)]
    bad = re.search(r"\b\d+ (failed|errors?|skipped|xfailed|xpassed|deselected)\b|\bFAILED\b|\bERROR\b|Traceback", txt)
    older = bool(ok_file and before is not None and os.path.getmtime(path) < before)
    return {"pass": bool(ok_file and passed and passed[-1] > 0 and not bad and older), "file_found": ok_file, "n_passed_last_summary": passed[-1] if passed else None, "fail_text_found": bool(bad), "older_than_read_start": older}


def q5_reader(ctx, a):
    d7 = D.control_d7(ctx)
    t = tests_clause(a.test_out, a.test_before)
    return {"pass": bool(d7["pass"] and t["pass"]), "d7_planted_difference_pass": bool(d7["pass"]), "tests": t}


def q6_one_evalset(ctx, P):
    ids0, rows, n0, ok = None, {}, None, True
    for label, p in P.items():
        try:
            ids = eval_ids(p["files"])
        except Exception:
            ids = None
        n = (((p.get("R") or {}).get("M2") or {}).get("head") or {}).get("n")
        if ids0 is None and ids is not None:
            ids0 = ids
        if n0 is None and n is not None:
            n0 = n
        good = bool(ids is not None and ids == ids0 and n is not None and n == n0)
        rows[label] = good
        ok = ok and good
    scored_ok = bool(ctx["cnt"]["scored"] == N.EXPECT["scored"])
    return {"pass": bool(ok and scored_ok and ids0 is not None), "per_probe_same_ids_same_order_same_count": rows, "ctx_scored_windows_equal_the_card_count": scored_ok}


def run_controls(a, ctx, P, specs):
    c = {}
    c["q0"] = q0_inputs(a, specs)
    c["q1"] = q1_saving_tool(a)
    c["q2"] = q2_corpus_live(P)
    c["q3"] = q3_permuted_dead(P)
    c["q4"] = q4_runs_complete(a, P, specs)
    c["q5"] = q5_reader(ctx, a)
    c["q6"] = q6_one_evalset(ctx, P)
    return c


# ============================================================================================ S, L, F and the names
def outs_of(P):
    outs = {l: p["out"] for l, p in P.items() if p["read"]}
    for k in SEEDS:
        outs[lab_q(k)] = outs[lab_q(k, STEPS_Q)]
    return outs


def two_level_steps(ctx, outs, measure, k_labels_a, k_labels_b, inner="shared"):
    pts, Ds = [], []
    for la, lb in zip(k_labels_a, k_labels_b):
        p, d = D.boot_diff(ctx, measure, outs[la], outs[lb])
        pts.append(p)
        Ds.append(d)
    return D.two_level(pts, np.array(Ds), 0, inner)


def measures(ctx, P):
    outs = outs_of(P)
    S = {"M2": D.two_level_pair(ctx, outs, "M2", a_kind="Q", b_kind="C"), "MAE": D.two_level_pair(ctx, outs, "MAE", a_kind="Q", b_kind="C")}
    L = two_level_steps(ctx, outs, "M2", [lab_q(k, STEPS_LATE) for k in SEEDS], [lab_q(k, STEPS_MID) for k in SEEDS])
    pt, vec = D.boot_diff(ctx, "M2", outs[lab_q(0, STEPS_Q)], outs["P16U2_s0"])
    r2q, r2u2 = D.summ_one(ctx, outs[lab_q(0, STEPS_Q)])["M2"]["r2"], D.summ_one(ctx, outs["P16U2_s0"])["M2"]["r2"]
    F = {"abs_diff_gap_r2": float(abs(r2q - r2u2)), "signed_diff_Q_s0_minus_U2_s0": float(pt), "paired_episode_interval_beside": RP.ci(vec), "note": "q1 does not enter F"}
    beside = {
        "gain_19164_to_28746": two_level_steps(ctx, outs, "M2", [lab_q(k, STEPS_MID) for k in SEEDS], [lab_q(k, STEPS_MAIN) for k in SEEDS]),
        "Q_minus_banked_U_M2": D.two_level_pair(ctx, outs, "M2", a_kind="Q", b_kind="U"),
        "Q_minus_banked_U_MAE": D.two_level_pair(ctx, outs, "MAE", a_kind="Q", b_kind="U"),
    }
    summ = {l: D.summ_one(ctx, o) for l, o in outs.items()}
    mae = {t: float(np.mean([summ[D.lab(t, k)]["MAE"]["mae_m"] for k in SEEDS])) for t in ("C",)}
    mae["Q"] = float(np.mean([summ[lab_q(k)]["MAE"]["mae_m"] for k in SEEDS]))
    return {"S": S, "L": L, "F": F, "beside": beside, "mean_gap_mae": mae, "summ": summ}


def decide(S, L, F, mae):
    """Card section 4. SCALES-GAP first; then SHORT-OF-TRAINING when the dose clause fails; then WORSE; else FLAT."""
    above = D._above(S["M2"])
    mae_ok = bool(mae["Q"] <= (1.0 - MAE_FRACTION) * mae["C"])
    mean_gt_f = bool(S["M2"]["mean"] > F["abs_diff_gap_r2"])
    dose = bool(L["ci95"][1] <= DOSE_BAR)
    scales = bool(above and mae_ok and mean_gt_f)
    worse = bool(D._below(S["M2"]) and S["M2"]["mean"] < -F["abs_diff_gap_r2"] and dose)
    if scales:
        name = NAMES[0]
    elif not dose:
        name = NAMES[3]
    elif worse:
        name = NAMES[1]
    else:
        name = NAMES[2]
    return {"name": name, "s_above_0_with_n_pos_at_least_3": above, "gap_error_clause_met": mae_ok, "mean_s_larger_than_f": mean_gt_f, "dose_clause_holds": dose, "dose_bar": DOSE_BAR,
            "l_upper_end": L["ci95"][1], "worse_clause_met": worse, "mae_ratio_Q_to_C": float(mae["Q"] / mae["C"]), "mae_bar_ratio": 1.0 - MAE_FRACTION}


# ============================================================================================ writing
def _ci(c):
    return "[%+.4f, %+.4f]" % (c[0], c[1])


def write_verdict(path, res):
    n, M = res["names"], res["measures"]
    L = ["# DSCALE2X_VERDICT: card PREREG_V8F_PROOF_ADD123 (DSCALE-LEAD-2X)", "", TIER_LINE, ""] + ["- " + x for x in FAMILY_LINES]
    L += ["", "Controls q0 to q6: all passed before this file was written (DSCALE2X_CONTROLS.json). B %d; two-level bootstrap (seed indices, then episodes; rng 0)." % res["B"], "",
          "## Name: **%s**" % n["name"], ""]
    for k in ("s_above_0_with_n_pos_at_least_3", "gap_error_clause_met", "mean_s_larger_than_f", "dose_clause_holds", "worse_clause_met"):
        L.append("- %s: %s" % (k, n[k]))
    L.append("- gap error ratio Q to C %.3f (bar %.2f); upper end of L %+.4f (dose bar %+.3f)" % (n["mae_ratio_Q_to_C"], n["mae_bar_ratio"], n["l_upper_end"], DOSE_BAR))
    S, Lg, F = M["S"], M["L"], M["F"]
    L += ["", "## S, L, F", "",
          "- S gap R2 (P16Q last step minus P16C, per seed): mean %+.4f %s; seeds above 0: %d of 4; per seed %s" % (S["M2"]["mean"], _ci(S["M2"]["ci95"]), S["M2"]["n_pos"], ", ".join("%+.4f" % x for x in S["M2"]["per_seed"])),
          "- S gap error (metres; negative = Q better): mean %+.4f %s" % (S["MAE"]["mean"], _ci(S["MAE"]["ci95"])),
          "- L gap R2 (38,328 minus 28,746): mean %+.4f %s; per seed %s" % (Lg["mean"], _ci(Lg["ci95"]), ", ".join("%+.4f" % x for x in Lg["per_seed"])),
          "- F |P16Q_s0 minus P16U2_s0| gap R2: %.4f; paired episode interval of the signed difference %s" % (F["abs_diff_gap_r2"], _ci(F["paired_episode_interval_beside"])), "",
          "## Beside (names nothing)", ""]
    for k, t in M["beside"].items():
        L.append("- %s: mean %+.4f %s; per seed %s" % (k, t["mean"], _ci(t["ci95"]), ", ".join("%+.4f" % x for x in t["per_seed"])))
    L += ["", "## Per probe (M2 gap R2 / gap error m / M4 AUROC)", ""]
    for lbl, r in res["probes"].items():
        L.append("- %s: R2 %+.3f; gap error %.2f m; AUROC %s" % (lbl, r["M2"]["r2"], r["MAE"]["mae_m"], "%.3f" % r["M4"]["auroc"] if r.get("M4") and "auroc" in r["M4"] else "n/a"))
    L += ["", "## Limits", "", "One frozen encoder; pooled tokens; evaluation windows from the corpus only; four seeds, and the two-level interval treats them as the draw of seeds; the learning rate falls on a cosine, "
          "so a small late gain is partly the schedule; a probe is not a model.", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


# ============================================================================================ main
def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("controls", "names"))
    for nm in ("banked-run-dir", "banked-in-dir", "new-run-dir", "new-in-dir", "banked-done", "new-done", "out-dir", "pack-dir"):
        ap.add_argument("--" + nm, required=True if nm != "pack-dir" else False, default=D.TKR.READ if nm == "pack-dir" else None)
    ap.add_argument("--test-out", default=None)
    ap.add_argument("--test-before", type=float, default=None)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--k2-tol", type=float, default=1e-3)
    ap.add_argument("--controls-json", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    t0 = time.time()
    specs = probe_specs()
    for d in (a.banked_run_dir, a.banked_in_dir, a.new_run_dir, a.new_in_dir):
        if not os.path.isdir(d):
            print("ZZ DSCALE2X REFUSED missing folder %s" % d)
            return 3
    for f in (a.banked_done, a.new_done):
        if not os.path.exists(f):
            print("ZZ DSCALE2X REFUSED missing done file %s" % f)
            return 3
    os.makedirs(a.out_dir, exist_ok=True)
    hashes = input_hashes(a, specs)
    if a.mode == "names":
        cp = a.controls_json or latest_path(a.out_dir, CONTROLS_FILE)
        cj = _json(cp) if cp else None
        if not (cj and cj.get("all_pass") is True and cj.get("input_hashes") == hashes and cj.get("mode") == "controls"):
            print("ZZ DSCALE2X REFUSED no passing controls file for these inputs")
            return 3
    try:
        mod = D.N.control_k2.__globals__
        if "TOL_K2" in mod:
            mod["TOL_K2"] = a.k2_tol
    except Exception:
        pass
    ctx = build_ctx(a)
    P = load_all(a, ctx, specs)
    if a.mode == "controls":
        c = run_controls(a, ctx, P, specs)
        failed = [k for k in CONTROL_NAMES if not c[k]["pass"]]
        res = {"mode": "controls", "card": "PREREG_V8F_PROOF_ADD123", "tier": TIER_LINE, "controls": c, "all_pass": not failed, "failed": failed, "input_hashes": hashes, "B": a.B,
               "unread_probes": sorted(l for l, p in P.items() if not p["read"]), "elapsed_s": round(time.time() - t0, 1)}
        with open(unique_path(a.out_dir, CONTROLS_FILE), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=_jd)
        for k in CONTROL_NAMES:
            print("ctl %s %s" % (k, "PASS" if c[k]["pass"] else "FAIL"))
        print("ZZ DSCALE2X CONTROLS ALL-PASS" if not failed else "ZZ DSCALE2X CONTROLS FAILED " + " ".join(failed))
        return 0 if not failed else 2
    # names
    if any(not p["read"] for p in P.values()):
        print("ZZ DSCALE2X REFUSED a probe could not be read: " + ", ".join(sorted(l for l, p in P.items() if not p["read"])))
        return 3
    M = measures(ctx, P)
    nm = decide(M["S"], M["L"], M["F"], M["mean_gap_mae"])
    probes = {l: {"M2": M["summ"][l]["M2"], "MAE": M["summ"][l]["MAE"], "M4": P[l]["R"].get("M4")} for l in P}
    fall = {}
    for l, sp in specs.items():
        if sp["side"] == "banked":
            try:
                r = D.p5_u(log_path(a, sp), sp["sh"], sp["steps"], 4512 if sp["corpus"] else None)
                fall[l] = {"first_mean": r.get("first_mean"), "last_mean": r.get("last_mean")}
            except Exception as e:
                fall[l] = {"error": repr(e)}
    res = {"mode": "names", "card": "PREREG_V8F_PROOF_ADD123", "tier": TIER_LINE, "families": list(FAMILY_LINES), "evidence_class": "MEASURED once run", "B": a.B, "names": nm, "measures": {k: M[k] for k in ("S", "L", "F", "beside", "mean_gap_mae")},
           "probes": probes, "fall_values_banked_beside": fall, "controls_file": cp, "input_hashes": hashes, "elapsed_s": round(time.time() - t0, 1)}
    with open(unique_path(a.out_dir, RESULT_FILE), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=_jd)
    write_verdict(unique_path(a.out_dir, VERDICT_FILE), res)
    print("ZZ DSCALE2X NAME %s" % nm["name"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
