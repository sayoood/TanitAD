r"""read_dscale_eqp.py -- the reader of card PREREG_V8F_PROOF_ADD125 (DSCALE-LEAD-EQP). Tier T0. CPU only, numpy only.

TIER: T0 (a probe on cached tokens of logged frames; it says what the tokens allow, not what a model does with them); never driving performance.

The reader of card ADD123 (read_dscale2x.py) is IMPORTED UNCHANGED: its sha256 prefix is asserted at import, and a SECOND INSTANCE of the module is loaded under another name, with its step constants set for a run of
61,568 steps (the file is never edited and never copied). Through it this file reuses the probe loading, the two-level bootstrap, parse_done, the record readers, the test clause and the control forms.

Three runs are read:
  banked run (card ADD45)   --banked-run-dir --banked-in-dir --banked-done : P16C_s0..3 (runC), P16U_s0..3 (runU), P16CSH_s0 (runCSH), P16USH_s0 (runUSH)
  run of card ADD123        --q-run-dir --q-in-dir --q-done                 : P16Q_s0..3 = states/runQ (38,328 steps)
  new run of this card      --new-run-dir --new-in-dir --new-done           : P16E_s0..3 = states/runQ of the new output folder (61,568 steps)

  python read_dscale_eqp.py controls  <the arguments above> --out-dir D --test-out F --test-before EPOCH
  python read_dscale_eqp.py names     (the same arguments)

Mode controls: writes DSCALE_EQP_CONTROLS.json and prints one line per control (e0 to e6). It computes and prints no measure of E or T and no name.
Mode names: refuses (exit 3) unless the controls file of the same out-dir says all_pass with the same input hashes; then writes DSCALE_EQP_RESULT.json and DSCALE_EQP_VERDICT.md and prints the name line.
Exit codes: 0 all pass, 2 a control failed, 3 refused. A result file is never overwritten (a second call writes ..._2).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
READER_SHA_PREFIX = "0deb4744579f651c"
READER_PATH = os.environ.get("D2X_READER") or os.path.abspath(os.path.join(HERE, "..", "..", "add123", "read", "read_dscale2x.py")).replace(os.sep, "/")
RUN_SCRIPT_SHA_PREFIX = "a01c5ca8cb43a2c5"          # dscale_lead/add123/run/pod_dscale_lead_2x.sh, landed
TOOL_SHA_PREFIX = "5888527b5c37003b"                # dscale_lead/add123/run/tok_probe_u_ck.py, landed
RUN_SCRIPT_DEFAULT = os.environ.get("EQP_RUN_SCRIPT") or os.path.abspath(os.path.join(HERE, "..", "..", "add123", "run", "pod_dscale_lead_2x.sh")).replace(os.sep, "/")


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def _load_reader():
    if not os.path.exists(READER_PATH):
        raise AssertionError("REFUSED: the reader of card ADD123 is not found at %s (set D2X_READER)" % READER_PATH)
    got = _sha_file(READER_PATH)
    if not got.startswith(READER_SHA_PREFIX):
        raise AssertionError("REFUSED: sha256 of %s begins %s, the card's reader begins %s" % (READER_PATH, got[:16], READER_SHA_PREFIX))
    spec = importlib.util.spec_from_file_location("read_dscale2x_eqp", READER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["read_dscale2x_eqp"] = mod
    spec.loader.exec_module(mod)
    return mod


X = _load_reader()
D, N, RP = X.D, X.N, X.RP
SEEDS = X.SEEDS
STEPS_E, STEPS_Q = 61568, 38328
X.STEPS_Q = STEPS_E                  # the second instance of the imported reader reads a run of 61,568 steps (the file itself is untouched)
X.STEPS_LATE = STEPS_E
X.NEW_STEPS = (STEPS_E,)
EXPECT_UNION, EXPECT_CORPUS = 14512, 4512
MAE_FRACTION = 0.10
CONTROL_NAMES = tuple("e%d" % i for i in range(7))
NAMES = ("LEAD-READ-EQP-SCALES-GAP", "LEAD-READ-EQP-R2-ONLY", "LEAD-READ-EQP-WORSE", "LEAD-READ-EQP-FLAT")
CONTROLS_FILE, RESULT_FILE, VERDICT_FILE = "DSCALE_EQP_CONTROLS.json", "DSCALE_EQP_RESULT.json", "DSCALE_EQP_VERDICT.md"
TIER_LINE = D.TIER_LINE
FAMILY_LINES = D.FAMILY_LINES
BANKED_LABELS = tuple(["P16C_s%d" % k for k in SEEDS] + ["P16U_s%d" % k for k in SEEDS] + ["P16CSH_s0", "P16USH_s0"])
KNOWN_Q_MINUS_U = "row 192 of card ADD123: P16Q minus P16U (2.5 against 1.25 passes, end states): +0.0466 [-0.1181, +0.2197]"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def lab(kind, k):
    return "P16%s_s%d" % (kind, k)


# ============================================================================================ the probe table
def probe_specs():
    """label -> dict(side banked|q|new, folder, arm, steps, sh (permuted gap labels), corpus)."""
    sp = {}
    for label in BANKED_LABELS:
        folder, arm, steps, sh, corpus = D.ARM_MAP[label]
        sp[label] = {"side": "banked", "folder": folder, "arm": arm, "steps": steps, "sh": sh, "corpus": corpus}
    for k in SEEDS:
        sp[lab("Q", k)] = {"side": "q", "folder": "runQ", "arm": "TK16_s%d" % k, "steps": STEPS_Q, "sh": False, "corpus": False}
    for k in SEEDS:
        sp[lab("E", k)] = {"side": "new", "folder": "runQ", "arm": "TK16_s%d" % k, "steps": STEPS_E, "sh": False, "corpus": False}
    return sp


def read_probe(label, spec, files, ctx, a, cache):
    """The per-probe read of the banked chain (tokprobe_read.read_arm). Replaceable in tests. Returns (R, out, ctrl)."""
    return D.TKR.read_arm(label, files, ctx, None, a.B, False, False, cache)


def build_ctx(a):
    P = N.load_pack(a.pack_dir + "/eval_pack.npz")
    return N.window_ctx(P, a.B)


def _side_dirs(a, side):
    return {"banked": (a.banked_run_dir, a.banked_in_dir), "q": (a.q_run_dir, a.q_in_dir), "new": (a.new_run_dir, a.new_in_dir)}[side]


def state_files(a, spec):
    return D._files(_side_dirs(a, spec["side"])[1], spec["folder"], spec["arm"])


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


def input_hashes(a, specs):
    """sha256 of every small input the controls read: the three done files, the record files, the batch logs, the precheck records, the train logs of the first-record clause, the test output and every probe's record."""
    nr, qr = a.new_run_dir, a.q_run_dir
    paths = [a.banked_done, a.q_done, a.new_done, nr + "/runQ/r1_batches.json", qr + "/runQ/r1_batches.json", nr + "/runQ/CK_RECORD.json", nr + "/runQ/TRAIN_DONE.json"]
    paths += ["%s/precheck/%s.json" % (nr, n) for n in ("PRECHECK", "PRECHECK_PLANTED", "CLOCK", "CLOCK_PLANTED", "EVALSET", "EVALSET_PLANTED")]
    for label, sp in specs.items():
        paths.append(state_files(a, sp)["eval_states"] + ".json")
    for k in SEEDS:
        paths += ["%s/runQ/TK16_s%d/train_log.jsonl" % (nr, k), "%s/runQ/TK16_s%d/train_log.jsonl" % (qr, k)]
    if a.test_out:
        paths.append(a.test_out)
    return {p: (X.sha256_file(p) if os.path.exists(p) else "MISSING") for p in paths}


# ============================================================================================ the controls e0 to e6
def _ckpt_equals_record(done, key, files):
    rec = X.record_of(files)
    return bool(key in done["ckpt"] and rec.get("ckpt_sha256") and rec.get("ckpt_sha256") == done["ckpt"][key])


def e0_inputs(a, specs, run_script_path=None):
    rows = {}
    done = {"banked": X.parse_done(a.banked_done), "q": X.parse_done(a.q_done), "new": X.parse_done(a.new_done)}
    for label, sp in specs.items():
        key = "%s/%s" % (sp["folder"], sp["arm"])
        rows[label] = _ckpt_equals_record(done[sp["side"]], key, state_files(a, sp))
    first_ok = {k: bool(v["first"]) for k, v in done.items()}
    pre_dir = a.new_run_dir + "/precheck"
    pre, pre_p = X._json(pre_dir + "/PRECHECK.json"), X._json(pre_dir + "/PRECHECK_PLANTED.json")
    ck, ck_p = X._json(pre_dir + "/CLOCK.json"), X._json(pre_dir + "/CLOCK_PLANTED.json")
    es, es_p = X._json(pre_dir + "/EVALSET.json"), X._json(pre_dir + "/EVALSET_PLANTED.json")
    pc = {}
    pc.update(D.precheck_controls(pre, pre_p))
    pc.update(D.clock_controls(ck, ck_p))
    pc.update(D.evalset_controls(es, es_p))
    pc_pass = {k: bool(v.get("pass")) for k, v in pc.items()}
    cnt = ((pre or {}).get("checks") or {}).get("counts") or {}
    counts_ok = bool(cnt.get("pass") is True and cnt.get("union_manifest_clips") == EXPECT_UNION and cnt.get("corpus_ids") == EXPECT_CORPUS)
    # the hashes of the saving tool and of the run script
    new_lines = open(a.new_done, encoding="utf-8").read().splitlines() if os.path.exists(a.new_done) else []
    tool_done = [m.group(1) for ln in new_lines for m in [re.match(r"^tok_probe_u_ck\.py sha256:\s*([0-9a-f]{64})\s*$", ln)] if m]
    script_done = [m.group(1) for ln in new_lines for m in [re.match(r"^pod_dscale_lead_2x\.sh sha256:\s*([0-9a-f]{64})\s*$", ln)] if m]
    ckrec = X._json(a.new_run_dir + "/runQ/CK_RECORD.json") or {}
    sp_path = run_script_path or RUN_SCRIPT_DEFAULT
    script_sha = script_done[0] if script_done else (_sha_file(sp_path) if sp_path and os.path.exists(sp_path) else "")
    hashes = {"saving_tool_in_done_file_is_the_landed_one": bool(tool_done and tool_done[0].startswith(TOOL_SHA_PREFIX)),
              "saving_tool_in_ck_record_is_the_landed_one": bool(str(ckrec.get("tool_sha256", "")).startswith(TOOL_SHA_PREFIX)),
              "run_script_is_the_landed_one": bool(script_sha.startswith(RUN_SCRIPT_SHA_PREFIX))}
    td = X._json(a.new_run_dir + "/runQ/TRAIN_DONE.json") or {}
    arms = td.get("arms")
    arm_names = sorted(arms.keys() if isinstance(arms, dict) else (arms or []))
    p5 = td.get("p5") or {}
    steps_ok = {"train_done_steps_is_61568": bool(td.get("steps") == STEPS_E), "train_done_arms_are_the_four": bool(arm_names == ["TK16_s%d" % k for k in SEEDS])}
    for k in SEEDS:
        steps_ok["arm_TK16_s%d_last_step_is_61568" % k] = bool((p5.get("TK16_s%d" % k) or {}).get("last_step") == STEPS_E)
    ok = bool(all(rows.values()) and all(first_ok.values()) and pc_pass and all(pc_pass.values()) and counts_ok and all(hashes.values()) and all(steps_ok.values()))
    return {"pass": ok, "done_hash_equals_record": rows, "done_files_have_a_marker_line": first_ok, "precheck_controls": pc_pass, "precheck_counts_14512_and_4512": counts_ok, "hashes": hashes, "new_run_steps": steps_ok}


def batch_compare(path_new, path_q):
    """Printed beside e1, never judged: the batch hashes of the first 38,328 steps of the new run against the run of card ADD123, entry for entry, where the two files allow it."""
    rn, rq = X._json(path_new), X._json(path_q)
    if not isinstance(rn, dict) or not isinstance(rq, dict):
        return {"comparable": False, "why": "a batch file is missing or not a record"}
    hn, hq = rn.get("hashes"), rq.get("hashes")
    if isinstance(hn, dict) and isinstance(hq, dict):
        common = sorted(set(hn) & set(hq), key=lambda s: (len(s), s))
        diff = [k for k in common if hn[k] != hq[k]]
        return {"comparable": True, "common_entries": len(common), "equal_entries": len(common) - len(diff), "first_differing_entry": diff[0] if diff else None}
    return {"comparable": False, "why": "no per-batch hash entries in both files"}


def e1_same_start(a):
    rows, ok = {}, True
    for k in SEEDS:
        arm = "TK16_s%d" % k
        fn = X._first_record("%s/runQ/%s/train_log.jsonl" % (a.new_run_dir, arm))
        fq = X._first_record("%s/runQ/%s/train_log.jsonl" % (a.q_run_dir, arm))
        row = {"step_1_in_both": bool(fn is not None and fq is not None and fn.get("step") == 1 and fq.get("step") == 1)}
        if fn is not None and fq is not None:
            for t in X.LOSS_TERMS:
                if t in fn or t in fq:
                    row["term_" + t] = bool(t in fn and t in fq and abs(float(fn[t]) - float(fq[t])) <= X.LOSS_TOL_Q1)
            row["required_terms_present"] = bool(all(("term_" + t) in row for t in X.LOSS_TERMS_REQUIRED))
        else:
            row["required_terms_present"] = False
        rows[arm] = row
        ok = bool(ok and all(row.values()))
    beside = batch_compare(a.new_run_dir + "/runQ/r1_batches.json", a.q_run_dir + "/runQ/r1_batches.json")
    return {"pass": bool(ok), "per_arm_first_record_within_1e-4": rows, "beside": {"batch_sequence_first_38328_steps_printed_not_judged": beside}}


def e2_live(P):
    flags = {}
    for k in SEEDS:
        for label in (lab("E", k), lab("C", k)):
            p = P[label]
            flags[label] = bool(p["read"] and D.d3_corpus_live(p["R"], label)["pass"])
    return {"pass": bool(all(flags.values())), "per_probe_pass": flags}


def e3_dead(P):
    return X.q3_permuted_dead(P)


def e4_complete(a, P):
    view = {X.lab_q(k, STEPS_E): P[lab("E", k)] for k in SEEDS}
    r = X.q4_runs_complete(a, view, None)
    txt = json.dumps(r).replace("last_step_is_38328", "last_step_is_61568")
    return json.loads(txt)


def e5_reader(ctx, a):
    return X.q5_reader(ctx, a)


def e6_one_evalset(ctx, P):
    return X.q6_one_evalset(ctx, P)


def run_controls(a, ctx, P, specs):
    c = {}
    c["e0"] = e0_inputs(a, specs)
    c["e1"] = e1_same_start(a)
    c["e2"] = e2_live(P)
    c["e3"] = e3_dead(P)
    c["e4"] = e4_complete(a, P)
    c["e5"] = e5_reader(ctx, a)
    c["e6"] = e6_one_evalset(ctx, P)
    return c


def why(c, path=""):
    """The names of the clauses that read False inside one control record (never a measure)."""
    out = []
    if isinstance(c, dict):
        for k, v in c.items():
            if k in ("pass", "beside") or str(k).startswith("printed"):
                continue
            if v is False:
                out.append(path + str(k))
            elif isinstance(v, dict):
                out += why(v, path + str(k) + ".")
    return out


def short_reason(c):
    w = why(c)
    return (", ".join(w[:3]) + (" (+%d)" % (len(w) - 3) if len(w) > 3 else "")) if w else "see the controls file"


# ============================================================================================ E, T and the names
def outs_of(P):
    return {l: p["out"] for l, p in P.items() if p["read"]}


def measures(ctx, P):
    outs = outs_of(P)
    pair = lambda ak, bk: {"M2": D.two_level_pair(ctx, outs, "M2", a_kind=ak, b_kind=bk), "MAE": D.two_level_pair(ctx, outs, "MAE", a_kind=ak, b_kind=bk)}
    E, T = pair("E", "C"), pair("E", "Q")
    beside = {"Q_minus_U": pair("Q", "U"), "S_Q_minus_C_card_ADD123": pair("Q", "C"), "known_value_beside": KNOWN_Q_MINUS_U}
    summ = {l: D.summ_one(ctx, o) for l, o in outs.items()}
    mae = {t: float(np.mean([summ[lab(t, k)]["MAE"]["mae_m"] for k in SEEDS])) for t in ("C", "E", "Q")}
    return {"E": E, "T": T, "beside": beside, "mean_gap_mae": mae, "summ": summ}


def decide(E, mae):
    """Card section 5. SCALES-GAP: the interval of E on gap R2 above 0, at least 3 of 4 seeds above 0, and the mean gap error of E at most 0.90 times that of C. R2-ONLY: the same without the error bar.
    WORSE: the interval below 0 and at least 3 of 4 seeds below 0. Else FLAT."""
    r2 = E["M2"]
    above = bool(r2["mean"] > 0 and r2["ci95"][0] > 0 and r2["n_pos"] >= 3)
    below = bool(r2["mean"] < 0 and r2["ci95"][1] < 0 and r2["n_neg"] >= 3)
    mae_ok = bool(mae["E"] <= (1.0 - MAE_FRACTION) * mae["C"])
    if above and mae_ok:
        name = NAMES[0]
    elif above:
        name = NAMES[1]
    elif below:
        name = NAMES[2]
    else:
        name = NAMES[3]
    return {"name": name, "e_above_0_with_n_pos_at_least_3": above, "e_below_0_with_n_neg_at_least_3": below, "gap_error_bar_met": mae_ok, "mae_ratio_E_to_C": float(mae["E"] / mae["C"]),
            "mae_bar_ratio": 1.0 - MAE_FRACTION, "e_upper_end_gap_r2": float(r2["ci95"][1]), "e_upper_end_note": "the gain in gap R2 the read leaves open"}


# ============================================================================================ writing
def _ci(c):
    return "[%+.4f, %+.4f]" % (c[0], c[1])


def _row(t):
    return "mean %+.4f %s; seeds above 0: %d of 4; per seed %s" % (t["mean"], _ci(t["ci95"]), t["n_pos"], ", ".join("%+.4f" % x for x in t["per_seed"]))


def write_verdict(path, res):
    n, M = res["names"], res["measures"]
    L = ["# DSCALE_EQP_VERDICT: card PREREG_V8F_PROOF_ADD125 (DSCALE-LEAD-EQP)", "", TIER_LINE, ""] + ["- " + x for x in FAMILY_LINES]
    L += ["", "Controls e0 to e6: all passed before this file was written (DSCALE_EQP_CONTROLS.json). B %d; two-level bootstrap (seed indices, then episodes; rng 0)." % res["B"], "",
          "## Name: **%s**" % n["name"], ""]
    for k in ("e_above_0_with_n_pos_at_least_3", "e_below_0_with_n_neg_at_least_3", "gap_error_bar_met"):
        L.append("- %s: %s" % (k, n[k]))
    L.append("- gap error ratio E to C %.3f (bar %.2f)" % (n["mae_ratio_E_to_C"], n["mae_bar_ratio"]))
    L.append("- upper end of the interval of E on gap R2 (the gain the read leaves open): %+.4f" % n["e_upper_end_gap_r2"])
    E, T, b = M["E"], M["T"], M["beside"]
    L += ["", "## E, T", "",
          "- E gap R2 (P16E minus P16C, per seed, last steps): " + _row(E["M2"]),
          "- E gap error (metres; negative = E better): mean %+.4f %s" % (E["MAE"]["mean"], _ci(E["MAE"]["ci95"])),
          "- T gap R2 (P16E minus P16Q: 4.0 against 2.5 passes): " + _row(T["M2"]),
          "- T gap error (metres): mean %+.4f %s" % (T["MAE"]["mean"], _ci(T["MAE"]["ci95"])), "",
          "## Beside (names nothing)", "",
          "- P16Q minus P16U gap R2 (2.5 against 1.25 passes, recomputed): " + _row(b["Q_minus_U"]["M2"]),
          "- known value: " + b["known_value_beside"],
          "- S of card ADD123 recomputed (P16Q minus P16C) gap R2: " + _row(b["S_Q_minus_C_card_ADD123"]["M2"]),
          "- S of card ADD123 recomputed gap error (metres): mean %+.4f %s" % (b["S_Q_minus_C_card_ADD123"]["MAE"]["mean"], _ci(b["S_Q_minus_C_card_ADD123"]["MAE"]["ci95"])), "",
          "## Per probe (M2 gap R2 / gap error m / M4 AUROC)", ""]
    for lbl, r in res["probes"].items():
        L.append("- %s: R2 %+.3f; gap error %.2f m; AUROC %s" % (lbl, r["M2"]["r2"], r["MAE"]["mae_m"], "%.3f" % r["M4"]["auroc"] if r.get("M4") and "auroc" in r["M4"] else "n/a"))
    L += ["", "## Limits", "", "One frozen encoder; pooled tokens; the evaluation windows come from the corpus only; four seeds, and the two-level interval treats them as the draw of seeds; the corpus probes are banked runs of "
          "2026-11-20 on the same pod and tool. An interval above or below 0 on four seeds is necessary for a name and not sufficient for a claim about training runs at large. A probe is not a model.", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


# ============================================================================================ main
def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("controls", "names"))
    for nm in ("banked-run-dir", "banked-in-dir", "banked-done", "q-run-dir", "q-in-dir", "q-done", "new-run-dir", "new-in-dir", "new-done", "out-dir"):
        ap.add_argument("--" + nm, required=True)
    ap.add_argument("--pack-dir", default=D.TKR.READ)
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
    for d in (a.banked_run_dir, a.banked_in_dir, a.q_run_dir, a.q_in_dir, a.new_run_dir, a.new_in_dir):
        if not os.path.isdir(d):
            print("ZZ DSCALE-EQP REFUSED missing folder %s" % d)
            return 3
    for f in (a.banked_done, a.q_done, a.new_done):
        if not os.path.exists(f):
            print("ZZ DSCALE-EQP REFUSED missing done file %s" % f)
            return 3
    os.makedirs(a.out_dir, exist_ok=True)
    hashes = input_hashes(a, specs)
    cp = None
    if a.mode == "names":
        cp = a.controls_json or X.latest_path(a.out_dir, CONTROLS_FILE)
        cj = X._json(cp) if cp else None
        if not (cj and cj.get("all_pass") is True and cj.get("input_hashes") == hashes and cj.get("mode") == "controls"):
            print("ZZ DSCALE-EQP REFUSED no passing controls file for these inputs")
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
        res = {"mode": "controls", "card": "PREREG_V8F_PROOF_ADD125", "tier": TIER_LINE, "controls": c, "all_pass": not failed, "failed": failed, "input_hashes": hashes, "B": a.B,
               "unread_probes": sorted(l for l, p in P.items() if not p["read"]), "elapsed_s": round(time.time() - t0, 1)}
        with open(X.unique_path(a.out_dir, CONTROLS_FILE), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=X._jd)
        for k in CONTROL_NAMES:
            print("ctl %s PASS" % k if c[k]["pass"] else "ctl %s FAIL %s" % (k, short_reason(c[k])[:100]))
        b = c["e1"]["beside"]["batch_sequence_first_38328_steps_printed_not_judged"]
        print("beside e1 batch sequence %s" % ("common %d equal %d" % (b["common_entries"], b["equal_entries"]) if b.get("comparable") else "not comparable"))
        print("ZZ DSCALE-EQP CONTROLS ALL-PASS" if not failed else "ZZ DSCALE-EQP CONTROLS FAILED")
        return 0 if not failed else 2
    if any(not p["read"] for p in P.values()):
        print("ZZ DSCALE-EQP REFUSED a probe could not be read: " + ", ".join(sorted(l for l, p in P.items() if not p["read"])))
        return 3
    M = measures(ctx, P)
    nm = decide(M["E"], M["mean_gap_mae"])
    probes = {l: {"M2": M["summ"][l]["M2"], "MAE": M["summ"][l]["MAE"], "M4": P[l]["R"].get("M4")} for l in P}
    res = {"mode": "names", "card": "PREREG_V8F_PROOF_ADD125", "tier": TIER_LINE, "families": list(FAMILY_LINES), "evidence_class": "MEASURED once run", "B": a.B, "names": nm,
           "measures": {k: M[k] for k in ("E", "T", "beside", "mean_gap_mae")}, "probes": probes, "controls_file": cp, "input_hashes": hashes, "elapsed_s": round(time.time() - t0, 1)}
    with open(X.unique_path(a.out_dir, RESULT_FILE), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=X._jd)
    write_verdict(X.unique_path(a.out_dir, VERDICT_FILE), res)
    print("ZZ DSCALE-EQP NAME %s" % nm["name"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
