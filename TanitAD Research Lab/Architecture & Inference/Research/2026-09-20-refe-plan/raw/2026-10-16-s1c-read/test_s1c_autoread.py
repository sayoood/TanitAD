#!/usr/bin/env python3
"""Tests for s1c_rules.py + s1c_autoread.py.  Prints `ZZTEST_PASS <n>` (exit 0) or `ZZTEST_FAIL ...` (exit 1).  Run in the tanitad venv:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I test_s1c_autoread.py

A  the decision rules (PREREG_S1C section 4, PREREG_AUXVETO section 6): the real prereg tables are parsed (the O3 row contains '|Delta|' pipes), the rows are applied over a GRID against an independent oracle
   written from the prereg's literal predicates, the O3 / O4 overlap is flagged, V-S1C's list equals auxveto_read.decide on a grid, the prereg guards accept the real files and refuse edited ones.
B  the driver's pure parts: arm_status (every missing file blocks), check_evidence on S1's REAL H1-s0 evidence under S1's budget and on an S1c-shaped transform of it (each registered fact edited -> a problem),
   deep_compare, the command lines, the scrub.
C  the REAL job end to end with a scripted HF stand-in and STUB children (no GPU, no network, no model): poll (nothing -> partial -> complete), fetch refusal on a corrupt file, every outcome row,
   INVALID -> FAILED marker, GPU busy waits (never touches the lock), auto falls back to CPU, an identity miss on CUDA falls back, resume (stages cached), token hygiene.
D  deliberate regressions: the SOURCE is edited one line at a time and the scenario that guards it must go RED.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import itertools
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s1c_autoread as A  # noqa: E402
import s1c_rules as R  # noqa: E402

PK = A.PK
S1R = A.S1R
SRC_A = (HERE / "s1c_autoread.py").read_text(encoding="utf-8")
SRC_R = (HERE / "s1c_rules.py").read_text(encoding="utf-8")
S1C_TEXT = R.read_lf(PK / "eval/PREREG_S1C.md")
AV_TEXT = R.read_lf(PK / "eval/PREREG_AUXVETO.md")
N_OK, FAILS = 0, []
FAKE_TOKEN = "hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"


def check(name, cond, detail=""):
    global N_OK
    if cond:
        N_OK += 1
    else:
        FAILS.append(f"{name} {detail}".strip())
        print("  FAIL", name, detail, flush=True)


def mutated(src, fname, modname, old, new):
    assert src.count(old) == 1, f"mutation anchor not unique: {old!r} x{src.count(old)}"
    m = types.ModuleType(modname)
    m.__file__ = str(HERE / fname)
    sys.modules[modname] = m                                  # dataclass (from __future__ annotations) looks its module up by name
    exec(compile(src.replace(old, new), str(HERE / fname), "exec"), m.__dict__)
    return m


def mut_a(old, new):
    return mutated(SRC_A, "s1c_autoread.py", "s1c_autoread_mut", old, new)


def mut_r(old, new):
    return mutated(SRC_R, "s1c_rules.py", "s1c_rules_mut", old, new)


# ================================================================================================================ A. rules
def oracle_row(valid, c3, nat, dc3, dnat):
    """PREREG_S1C section 4 from its LITERAL predicates (independent of s1c_rules.separated): not separated := CI includes 0 or |Delta| <= F."""
    def not_sep(d, F):
        lo, hi = d["ci95"]
        return (lo <= 0 <= hi) or abs(d["est"]) <= F

    def pos(d, F):
        return (not not_sep(d, F)) and d["est"] > 0

    def neg(d, F):
        return (not not_sep(d, F)) and d["est"] < 0
    if not valid:
        return "INVALID"
    if c3 >= 0.8 and nat >= 0.8:
        return "O1"
    if pos(dc3, 0.035) and pos(dnat, 0.023) and (c3 < 0.8 or nat < 0.8):
        return "O2"
    if not_sep(dc3, 0.035) or not_sep(dnat, 0.023):
        return "O3"
    if neg(dc3, 0.035) or neg(dnat, 0.023):
        return "O4"
    return "??"


DELTAS = {"big_pos": {"est": 0.10, "ci95": [0.05, 0.15]}, "small_pos_sep_ci": {"est": 0.02, "ci95": [0.01, 0.03]}, "pos_ci_has_0": {"est": 0.06, "ci95": [-0.01, 0.12]},
          "zero": {"est": 0.0, "ci95": [-0.02, 0.02]}, "big_neg": {"est": -0.10, "ci95": [-0.15, -0.05]}, "neg_ci_has_0": {"est": -0.06, "ci95": [-0.12, 0.01]},
          "neg_small": {"est": -0.01, "ci95": [-0.02, -0.003]}}


def rules_tests():
    for k, row in (("O1", "O1 K2 CLEARED"), ("O2", "O2 PARTIAL"), ("O3", "O3 NO GAIN"), ("O4", "O4 WORSE"), ("INVALID", "INVALID")):
        c = R.s1c_row_cells(S1C_TEXT, row)
        check(f"s1c_row_parsed_{k}", c["label"] == row and c["next"] and c["blocks"] and c["meaning"] and c["rule"], str(c)[:100])
    o3 = R.s1c_row_cells(S1C_TEXT, "O3 NO GAIN")
    check("o3_row_survives_its_|Delta|_pipes", "|Delta| <= F" in o3["rule"] and o3["next"].startswith("**close the budget lever for the visual tap**") and "PI: trunk lever" in o3["blocks"], o3["next"][:80])
    check("o1_next_names_the_replicate", "replicate seed 1" in R.s1c_row_cells(S1C_TEXT, "O1 K2 CLEARED")["next"])
    check("invalid_next_names_the_cause", R.s1c_row_cells(S1C_TEXT, "INVALID")["next"].startswith("find the cause"))
    check("o4_next_names_the_u2k_arm", "H1u2k-s0" in R.s1c_row_cells(S1C_TEXT, "O4 WORSE")["next"])
    for r in ("AV-GO", "AV-PROMISING", "AV-HARM", "AV-CEILING-ONLY", "AV-NULL"):
        c = R.auxveto_row_cells(AV_TEXT, r)
        check(f"auxveto_row_parsed_{r}", c["label"] == r and c["next"] and c["condition"])
    # separated(): the boundaries
    S = R.separated
    check("sep_pos", S({"est": 0.04, "ci95": [0.01, 0.07]}, 0.035) == "pos")
    check("sep_est_equal_to_F_is_NOT_separated", S({"est": 0.035, "ci95": [0.01, 0.07]}, 0.035) == "none")
    check("sep_lo_equal_to_0_is_NOT_separated", S({"est": 0.05, "ci95": [0.0, 0.1]}, 0.035) == "none")
    check("sep_neg", S({"est": -0.05, "ci95": [-0.1, -0.01]}, 0.035) == "neg")
    check("sep_neg_within_F_is_none", S({"est": -0.03, "ci95": [-0.1, -0.01]}, 0.035) == "none")
    check("sep_hi_equal_to_0_is_NOT_separated", S({"est": -0.05, "ci95": [-0.1, 0.0]}, 0.035) == "none")
    check("sep_nan_is_none", S({"est": float("nan"), "ci95": [0.1, 0.2]}, 0.035) == "none")
    # the grid against the oracle
    P1S = (0.40, 0.7999999, 0.80, 0.86)
    n_grid, bad = 0, []
    for valid, c3, nat, kc, kn in itertools.product((True, False), P1S, P1S, DELTAS, DELTAS):
        m = {"c1_ok": valid, "runcheck_ok": True, "evidence_ok": True, "p1_c3": c3, "p1_nat": nat, "d_c3": DELTAS[kc], "d_nat": DELTAS[kn]}
        got = R.decide_s1c(m, S1C_TEXT)["row"]
        want = oracle_row(valid, c3, nat, DELTAS[kc], DELTAS[kn])
        n_grid += 1
        if got != want:
            bad.append((valid, c3, nat, kc, kn, got, want))
    check(f"decide_s1c_equals_the_oracle_on_{n_grid}_grid_cases", not bad, str(bad[:3]))
    for flag in ("runcheck_ok", "evidence_ok", "c1_ok"):
        m = {"c1_ok": True, "runcheck_ok": True, "evidence_ok": True, "p1_c3": 0.9, "p1_nat": 0.9, "d_c3": DELTAS["big_pos"], "d_nat": DELTAS["big_pos"]}
        m[flag] = False
        check(f"invalid_precedes_O1_when_{flag}_is_false", R.decide_s1c(m, S1C_TEXT)["row"] == "INVALID")
    base = {"c1_ok": True, "runcheck_ok": True, "evidence_ok": True, "p1_c3": 0.6, "p1_nat": 0.65}
    d = R.decide_s1c(dict(base, d_c3=DELTAS["big_neg"], d_nat=DELTAS["zero"]), S1C_TEXT)
    check("overlap_O3_and_O4_is_flagged_and_O4_quoted", d["row"] == "O3" and d["overlap"] == ["O4"] and "H1u2k-s0" in d["overlap_rows_quoted"]["O4"]["next"])
    d = R.decide_s1c(dict(base, d_c3=DELTAS["big_pos"], d_nat=DELTAS["big_neg"]), S1C_TEXT)
    check("pos_and_neg_is_O4_without_overlap", d["row"] == "O4" and d["overlap"] == [])
    d = R.decide_s1c(dict(base, d_c3=DELTAS["big_pos"], d_nat=DELTAS["big_pos"]), S1C_TEXT)
    check("both_separated_positive_below_bar_is_O2", d["row"] == "O2" and abs(d["remaining_gap_to_0.80"]["C3"] - 0.2) < 1e-12 and d["remaining_gap_to_0.80"]["worse_population"] == "C3")
    d = R.decide_s1c(dict(base, p1_c3=0.82, p1_nat=0.79, d_c3=DELTAS["zero"], d_nat=DELTAS["zero"]), S1C_TEXT)
    check("k2_disagreement_flagged_c3_governs", d["k2"]["populations_disagree"] and d["k2"]["governing_verdict_C3"] == "K2 CLEARED" and d["row"] == "O3")
    d = R.decide_s1c(dict(base, p1_c3=0.80, p1_nat=0.80, d_c3=DELTAS["zero"], d_nat=DELTAS["zero"]), S1C_TEXT)
    check("p1_exactly_0.80_clears_the_bar", d["row"] == "O1")
    check("quoted_next_equals_the_prereg_cell", R.decide_s1c(dict(base, d_c3=DELTAS["zero"], d_nat=DELTAS["zero"]), S1C_TEXT)["quoted_row"]["next"] == o3["next"])
    # the AUXVETO list == auxveto_read.decide on a grid, and the quoted next arm == auxveto_read.prereg_next_arm
    sys.path.insert(0, str(PK / "raw/2026-10-16-auxveto-read"))
    sys.path.insert(0, str(PK / "raw/2026-10-16-auxveto-feasibility"))
    import auxveto_read as AV
    vals = [-1.5, -0.2, 0.0, 1e-9, 0.29999, 0.30, 0.31, 0.9]
    bad = []
    for lo, hi, pt, glo in itertools.product(vals, vals, vals, (-0.1, 0.0, 1e-9, 0.2)):
        if not (lo <= pt <= hi):
            continue
        a = R.decide_veto({"lo": lo, "hi": hi, "point": pt}, {"lo": glo})
        b = AV.decide({"lo": lo, "hi": hi, "point": pt}, {"lo": glo})["row"]
        if a != b:
            bad.append((lo, hi, pt, glo, a, b))
    check("decide_veto_equals_auxveto_read_decide_on_the_grid", not bad, str(bad[:3]))
    for r in ("AV-GO", "AV-PROMISING", "AV-HARM", "AV-CEILING-ONLY", "AV-NULL"):
        check(f"veto_next_arm_equals_auxveto_read_{r}", R.auxveto_row_cells(AV_TEXT, r)["next"] == AV.prereg_next_arm(r))
    vr = R.veto_row({"point": -0.8, "lo": -1.5, "hi": -0.2}, {"point": 0.37, "lo": 0.015, "hi": 0.64}, AV_TEXT)
    check("veto_harm_with_positive_ceiling_branch", vr["row"] == "AV-HARM" and vr["gt_branch"]["V_GT_lower_bound_gt_0"] is True and "raster quality (S1c)" in vr["gt_branch"]["reading"])
    vr = R.veto_row({"point": -0.8, "lo": -1.5, "hi": -0.2}, {"point": 0.1, "lo": -0.1, "hi": 0.4}, AV_TEXT)
    check("veto_harm_with_flat_ceiling_quotes_AVNULL_next_arm", vr["gt_branch"]["AV-NULL_next_arm_quoted"] == R.auxveto_row_cells(AV_TEXT, "AV-NULL")["next"])
    check("c1_boundaries", R.c1_check(0.1586791569036157 + 0.0009)["ok"] and not R.c1_check(0.1586791569036157 + 0.0011)["ok"] and R.c1_check(0.1586791569036157)["abs_diff"] == 0.0)

    # ---- the prereg guards
    g1 = R.guard_prereg_s1c(PK / "eval/PREREG_S1C.md")
    check("guard_s1c_accepts_the_registered_text", g1["amendments"] == [])
    g2 = R.guard_prereg_auxveto(PK / "eval/PREREG_AUXVETO.md")
    check("guard_auxveto_accepts_body_and_A1", len(g2["amendments"]) == 1 and g2["amendments"][0].startswith("## Amendment A1"))
    tmp = Path(tempfile.mkdtemp(prefix="s1c_guard_"))

    def raises(fn, path):
        try:
            fn(path)
            return False
        except ValueError:
            return True
    p = tmp / "s1c.md"
    p.write_text(S1C_TEXT.replace("0.80", "0.79", 1), encoding="utf-8", newline="\n")
    check("guard_s1c_REFUSES_an_edited_threshold", raises(R.guard_prereg_s1c, p))
    p.write_text(S1C_TEXT.replace("\n", "\r\n"), encoding="utf-8", newline="")
    check("guard_s1c_accepts_the_CRLF_form", not raises(R.guard_prereg_s1c, p))
    p.write_text(S1C_TEXT + "\n## Amendment A1 (2026-10-17): a dated amendment\nx\n", encoding="utf-8", newline="\n")
    check("guard_s1c_accepts_an_appended_amendment_and_lists_it", R.guard_prereg_s1c(p)["amendments"] == ["## Amendment A1 (2026-10-17): a dated amendment"])
    p.write_text(AV_TEXT.replace("more than 0.25 m outside", "more than 0.30 m outside", 1), encoding="utf-8", newline="\n")
    check("guard_auxveto_REFUSES_an_edited_body", raises(R.guard_prereg_auxveto, p))
    i = AV_TEXT.index("## Amendment A1")
    p.write_text(AV_TEXT[:i], encoding="utf-8", newline="\n")
    check("guard_auxveto_REFUSES_a_missing_A1", raises(R.guard_prereg_auxveto, p))
    p.write_text(AV_TEXT.replace("V-S1C", "V-S1X", 1), encoding="utf-8", newline="\n")
    check("guard_auxveto_REFUSES_an_edited_A1", raises(R.guard_prereg_auxveto, p))
    p.write_text(AV_TEXT.replace("\n", "\r\n"), encoding="utf-8", newline="")
    check("guard_auxveto_accepts_the_CRLF_form", not raises(R.guard_prereg_auxveto, p))
    shutil.rmtree(tmp, ignore_errors=True)


# ================================================================================================================ B. driver, pure parts
def real_h1s0_evidence():
    return S1R.load_evidence("H1-s0", A.S1_CACHE)


def s1c_shaped_evidence():
    """S1's real H1-s0 evidence re-shaped to what the registered S1c arm would push: 10,000 x batch 8 x accum 1, evals at 5000 / 10000, profile refe_aux_s1c, the budget flags NAMED."""
    ev = copy.deepcopy(real_h1s0_evidence())
    arm = "H1u10k-s0"
    tok = ev["token"]
    argv = list(tok["binding"]["argv"])

    def setv(flag, val):
        argv[argv.index(flag) + 1] = val
    setv("--max-updates", "10000")
    setv("--eval-every", "5000")
    setv("--dump-scores", "/workspace/data/s1aux/runs/H1u10k-s0/dump")
    setv("--out", "/workspace/data/s1aux/runs/H1u10k-s0")
    argv += ["--batch", "8", "--accum", "1"]
    tok["binding"]["argv"] = argv
    tok["binding"]["argv_sha256"] = S1R.argv_sha256(argv)
    tok["binding"]["profile"] = "refe_aux_s1c"
    ev["token_sha256"], ev["token_file"] = "f" * 64, "PASS_aaaaaaaaaaaa.json"
    ev["manifest"] = [{"arm": arm, "event": "token_verified", "argv_sha256": tok["binding"]["argv_sha256"], "token_sha256": "f" * 64}]
    ev["run_check"] = {"arm": arm, "ok": True, "problems": []}
    sft = []
    for r in ev["sft"]:
        r = dict(r)
        if r["event"] == "start":
            r.update(updates=10000, micro_per_update=1, batch=8)
        if r["event"] == "done":
            r.update(updates=10000, hours=4.3)
        if r["event"] in ("eval", "aux_eval", "score_dump") and r.get("tag") == "update400":
            for t in ("update5000", "update10000"):
                sft.append(dict(r, tag=t))
            continue
        if r["event"] == "train":
            r.update(update=min(r["update"] * 25, 10000), of=10000, s_per_update=1.5)
        sft.append(r)
    ev["sft"] = sft
    return ev


def evidence_tests():
    ev = real_h1s0_evidence()
    P, W, F = A.check_evidence(A.DRY, ev, None)
    check("evidence_real_H1s0_is_clean_under_S1_budget", P == [], str(P))
    check("evidence_facts_step0_iou", abs(F["step0_iou"] - 0.1586791569036157) < 1e-15)
    muts = {
        "run_check_wrong_arm": lambda e: e["run_check"].update(arm="H1-s1"),
        "run_check_not_ok": lambda e: e["run_check"].update(ok=False),
        "run_check_has_problems": lambda e: e["run_check"].update(problems=["x"]),
        "token_not_pass": lambda e: e["token"].update(verdict="FAIL"),
        "token_profile": lambda e: e["token"]["binding"].update(profile="refe_aux_s1c"),
        "token_argv_sha": lambda e: e["token"]["binding"].update(argv_sha256="0" * 64),
        "manifest_row_missing": lambda e: e.update(manifest=[]),
        "manifest_last_row_other_arm": lambda e: e["manifest"].append({"arm": "H2-s0"}),
        "start_updates": lambda e: [r.update(updates=399) for r in e["sft"] if r["event"] == "start"],
        "start_batch": lambda e: [r.update(batch=16) for r in e["sft"] if r["event"] == "start"],
        "done_missing": lambda e: e.update(sft=[r for r in e["sft"] if r["event"] != "done"]),
        "done_updates": lambda e: [r.update(updates=399) for r in e["sft"] if r["event"] == "done"],
        "eval_tag_missing": lambda e: e.update(sft=[r for r in e["sft"] if not (r["event"] == "eval" and r.get("tag") == "update400")]),
        "aux_eval_tag_missing": lambda e: e.update(sft=[r for r in e["sft"] if not (r["event"] == "aux_eval" and r.get("tag") == "update400")]),
        "heldout_leak": lambda e: [r.update(train_rows_in_heldout_logs=1) for r in e["sft"] if r["event"] == "data"],
        "train_sets": lambda e: [r.update(train_sets=174911) for r in e["sft"] if r["event"] == "data"],
        "head_params": lambda e: [r["aux_drv"].update(n_params=1) for r in e["sft"] if r["event"] == "data"],
        "feed_built": lambda e: [r["aux_drv"].update(feed_built=True) for r in e["sft"] if r["event"] == "data"],
        "head_store": lambda e: [r["aux_drv"].update(store_sha256="0" * 64) for r in e["sft"] if r["event"] == "data"],
        "aux_bce_nan": lambda e: [r.update(aux_bce=float("nan")) for r in e["sft"] if r["event"] == "train"][:1],
    }
    for name, fn in muts.items():
        e = copy.deepcopy(ev)
        fn(e)
        P, _, _ = A.check_evidence(A.DRY, e, None)
        check(f"evidence_RED_{name}", bool(P))

    def argv_edit(fn):
        e = copy.deepcopy(ev)
        argv = list(e["token"]["binding"]["argv"])
        fn(argv)
        e["token"]["binding"]["argv"] = argv
        e["token"]["binding"]["argv_sha256"] = S1R.argv_sha256(argv)
        e["manifest"] = [dict(r, argv_sha256=e["token"]["binding"]["argv_sha256"]) if r.get("arm") == "H1-s0" else r for r in e["manifest"]]
        return A.check_evidence(A.DRY, e, None)[0]
    check("evidence_RED_argv_seed", bool(argv_edit(lambda a: a.__setitem__(a.index("--seed") + 1, "1"))))
    check("evidence_RED_argv_max_updates", bool(argv_edit(lambda a: a.__setitem__(a.index("--max-updates") + 1, "10000"))))
    check("evidence_RED_argv_names_a_budget_flag_S1_leaves_default", bool(argv_edit(lambda a: a.extend(["--batch", "8"]))))
    check("evidence_RED_argv_feed_on", bool(argv_edit(lambda a: a.extend(["--aux-drv-feed", "on"]))))
    check("evidence_RED_argv_head_lr", bool(argv_edit(lambda a: a.__setitem__(a.index("--aux-drv-lr") + 1, "1e-3"))))
    check("evidence_RED_argv_out_names_another_arm", bool(argv_edit(lambda a: a.__setitem__(a.index("--out") + 1, "/workspace/data/s1aux/runs/H1-s1"))))
    # S1c-shaped
    e2 = s1c_shaped_evidence()
    P, W, F = A.check_evidence(A.REAL, e2, None)
    check("evidence_S1c_shaped_is_clean_under_the_S1c_budget", P == [], str(P))
    check("evidence_S1c_operational_check_in_range", F["operational_check"]["s_per_update_in_range"] and F["operational_check"]["train_hours_in_range"])
    P, _, _ = A.check_evidence(A.DRY, e2, None)
    check("evidence_S1c_shaped_is_REFUSED_under_S1_budget", bool(P))
    P, _, _ = A.check_evidence(A.REAL, ev, None)
    check("evidence_S1_H1s0_is_REFUSED_as_the_S1c_arm", bool(P))
    smuts = {
        "profile_is_the_S1_profile": lambda e: e["token"]["binding"].update(profile="refe_aux_s1"),
        "start_micro_per_update_is_8": lambda e: [r.update(micro_per_update=8) for r in e["sft"] if r["event"] == "start"],
        "last_train_row_not_at_lr0": lambda e: [r.update(lr=1e-5) for r in e["sft"] if r["event"] == "train" and r["update"] == 10000],
        "last_train_row_not_update_N": lambda e: e.update(sft=[r for r in e["sft"] if not (r["event"] == "train" and r["update"] == 10000)]),
        "eval_5000_missing": lambda e: e.update(sft=[r for r in e["sft"] if not (r.get("tag") == "update5000" and r["event"] == "eval")]),
        "done_is_400": lambda e: [r.update(updates=400) for r in e["sft"] if r["event"] == "done"],
    }
    for name, fn in smuts.items():
        e = copy.deepcopy(e2)
        fn(e)
        P, _, _ = A.check_evidence(A.REAL, e, None)
        check(f"evidence_S1c_RED_{name}", bool(P))
    # budget flags dropped from the S1c argv
    e = copy.deepcopy(e2)
    argv = [x for i, x in enumerate(e["token"]["binding"]["argv"]) if x not in ("--batch", "--accum") and not (i > 0 and e["token"]["binding"]["argv"][i - 1] in ("--batch", "--accum"))]
    e["token"]["binding"]["argv"], e["token"]["binding"]["argv_sha256"] = argv, S1R.argv_sha256(argv)
    e["manifest"][0]["argv_sha256"] = e["token"]["binding"]["argv_sha256"]
    check("evidence_S1c_RED_budget_flags_not_named", bool(A.check_evidence(A.REAL, e, None)[0]))
    # expected-argv drift is a WARNING (path drift must not fake an INVALID)
    P, W, _ = A.check_evidence(A.DRY, ev, ["--x"])
    check("expected_argv_drift_is_only_a_warning", P == [] and any("ARGV_EXPECTED_DIFF" in w for w in W))
    # arm_status
    listing = {}
    for rel in A.required_paths(A.REAL) + [f"arms/{A.REAL.name}/model_sft_B.pt", f"arms/{A.REAL.name}/PASS_abc123.json"]:
        listing[rel] = {"size": 1, "sha256": None, "blob_id": "x"}
    st = A.arm_status(listing, A.REAL)
    check("arm_status_complete_is_ready", st["ready"] and st["token"].endswith("PASS_abc123.json"))
    for rel in list(listing):
        l2 = {k: v for k, v in listing.items() if k != rel}
        s2 = A.arm_status(l2, A.REAL)
        check(f"arm_status_NOT_ready_without_{rel.split('/', 2)[2]}", not s2["ready"] and bool(s2["missing"]))
    l3 = dict(listing, **{f"arms/{A.REAL.name}/PASS_def456.json": {"size": 1}})
    check("arm_status_two_tokens_not_ready", not A.arm_status(l3, A.REAL)["ready"])
    check("arm_status_ignores_other_arms_files", not A.arm_status({k.replace(A.REAL.name, "H1-s0"): v for k, v in listing.items()}, A.REAL)["ready"])
    check("arm_status_dry_spec_does_not_need_model_B", A.arm_status({k.replace(A.REAL.name, "H1-s0").replace("update5000", "update400").replace("update10000", "update400"): v for k, v in listing.items()
                                                                    if "model_sft_B" not in k}, A.DRY)["ready"])
    # deep_compare
    a = {"p1": {"est": 0.5, "ci": [0.4, 0.6]}, "decision": "K2_FIRES", "input": {"x": 1}}
    check("deep_compare_equal", A.deep_compare(a, copy.deepcopy(a), 0.0)["ok"])
    b = copy.deepcopy(a)
    b["p1"]["est"] = 0.5 + 1e-7
    dc = A.deep_compare(a, b, 1e-9)
    check("deep_compare_RED_on_numeric_diff_and_locates_it", (not dc["ok"]) and dc["at"] == "/p1/est" and abs(dc["max_abs_diff"] - 1e-7) < 1e-12)
    check("deep_compare_within_tolerance", A.deep_compare(a, b, 1e-6)["ok"])
    b = copy.deepcopy(a)
    b["decision"] = "K2_CLEARED"
    check("deep_compare_RED_on_a_string_leaf", not A.deep_compare(a, b, 1.0)["ok"])
    b = copy.deepcopy(a)
    b["p1"]["ci"] = [0.4]
    check("deep_compare_RED_on_structure", not A.deep_compare(a, b, 1.0)["ok"])
    b = copy.deepcopy(a)
    b["input"] = {"x": 2}
    check("deep_compare_skips_input", A.deep_compare(a, b, 0.0)["ok"])
    # command lines
    c = A.forward_cmd("py", {"B": "b.pt", "A": "a.pt"}, "o", "ctx", "cpu")
    check("forward_cmd_cpu_has_no_gpu_lock_and_sorted_heads", "--no-gpu-lock" in c and c[-2:] == ["A=a.pt", "B=b.pt"] and c[c.index("--device") + 1] == "cpu")
    c = A.forward_cmd("py", {"A": "a.pt"}, "o", "ctx", "cuda")
    check("forward_cmd_cuda_keeps_the_lock", "--no-gpu-lock" not in c and c[c.index("--device") + 1] == "cuda")
    check("nat_cmd_contrast", "--contrast" in A.nat_cmd("py", {"A": "a"}, [("A", "B")], "o") and "A:B" in A.nat_cmd("py", {"A": "a"}, [("A", "B")], "o"))
    check("arm_seed", A.arm_seed("H1u10k-s0") == 0 and A.arm_seed("H1-s2") == 2)
    j = A.Job({"repo": "r", "out": tempfile.mkdtemp(prefix="s1c_t_"), "work": tempfile.mkdtemp(prefix="s1c_w_")}, None, A.REAL, secrets=[FAKE_TOKEN])
    check("scrub_masks_secret_and_pattern", FAKE_TOKEN not in j.scrub(f"x {FAKE_TOKEN} y") and "hf_***" in j.scrub("hf_ABCDEFGH12345678"))
    os.environ["HF_TOKEN"] = FAKE_TOKEN
    env = j.child_env()
    del os.environ["HF_TOKEN"]
    check("child_env_has_no_token", "HF_TOKEN" not in env and env["PYTHONIOENCODING"] == "utf-8")


# ================================================================================================================ C. the job, end to end, stub children
BANKED_NAV = A.jload(A.BANKED_NAV)
BANKED_BLOCK = BANKED_NAV["arms"]["H1-s0"]


def fake_blk(p1_c3, perturb=0.0):
    b = copy.deepcopy(BANKED_BLOCK)
    for sp in b["p1"].values():
        if isinstance(sp, dict) and "0-40" in sp:
            sp["0-40"]["est"] += 0.0
    b["p1"]["all"]["0-40"]["est"] = p1_c3 + perturb
    return b


def fake_veto_block(point, lo, hi):
    row = {"point": point, "lo": lo, "hi": hi, "z": point, "z_lo": lo, "z_hi": hi, "n": 0.0, "n_lo": -0.1, "n_hi": 0.1}
    return {"d_pdms": row, "d_nc": row, "d_dac": row, "d_ep": row, "d_ttc": row, "d_ddc": row, "d_comfort": row, "lvl_pdms": row, "z_recovery": {"ncdac_count": 30, "of": 529, "dac_only_count": 100, "pdms_pos_count": 40},
            "false_veto_N": {"b_pick_vetoed_count": 5, "of": 529}, "all_vetoed_tokens": {"Z": 1, "N": 0}}


class Fixture:
    """A temp world: a fake 'HF repo' dir holding arms/H1u10k-s0/*, the stub children's behaviour knobs, a call log."""

    def __init__(self, p1_c3=0.60, p1_nat=0.65, d_c3=None, d_nat=None, step0_iou=None, veto=(-0.4, -1.0, 0.1), vgt=(0.37, 0.015, 0.64), veto_valid=True, mutate_ev=None, nat_c3_skew=0.0, pair_skew=0.0, heldout_pair_refused=False):
        self.root = Path(tempfile.mkdtemp(prefix="s1c_fx_"))
        self.repo = self.root / "repo"
        self.out = self.root / "out"
        self.work = self.root / "work"
        self.p1_c3, self.p1_nat = p1_c3, p1_nat
        self.d_c3 = d_c3 or {"est": 0.05, "ci95": [0.01, 0.09]}
        self.d_nat = d_nat or {"est": 0.04, "ci95": [0.01, 0.07]}
        self.veto, self.vgt, self.veto_valid = veto, vgt, veto_valid
        self.nat_c3_skew, self.pair_skew, self.heldout_pair_refused = nat_c3_skew, pair_skew, heldout_pair_refused
        self.calls = []
        self.busy = {"n": 0}                        # cuda forward returns rc 5 this many times first (a big number = always busy)
        self.cuda_perturb = 0.0                     # added to the comparator's C3 P1 on cuda (an identity miss when > 1e-4)
        self.ev = s1c_shaped_evidence()
        if mutate_ev is not None:
            mutate_ev(self.ev)
        if step0_iou is not None:
            for r in self.ev["sft"]:
                if r["event"] == "aux_eval" and r["tag"] == "step0":
                    r["iou"] = step0_iou
        self.write_repo()

    def write_repo(self):
        a = self.repo / "arms" / A.REAL.name
        (a / "dump").mkdir(parents=True, exist_ok=True)
        (a / "arms_manifest.jsonl").write_text("\n".join(json.dumps(r) for r in self.ev["manifest"]) + "\n", encoding="utf-8")
        (a / "run_check.json").write_text(json.dumps(self.ev["run_check"]), encoding="utf-8")
        (a / "PREFLIGHT_PASS.json").write_text(json.dumps(self.ev["preflight"]), encoding="utf-8")
        (a / "sft.jsonl").write_text("\n".join(json.dumps(r) for r in self.ev["sft"]) + "\n", encoding="utf-8")
        (a / "token_verify.json").write_text(json.dumps(self.ev["token_verify"]), encoding="utf-8")
        (a / "launch_gate.log").write_text("ZZEXIT 0\n", encoding="utf-8")
        (a / "PASS_aaaaaaaaaaaa.json").write_text(json.dumps(self.ev["token"]), encoding="utf-8")
        self.ev["token_sha256"] = A.sha256_file(a / "PASS_aaaaaaaaaaaa.json")
        self.ev["manifest"][0]["token_sha256"] = self.ev["token_sha256"]
        (a / "arms_manifest.jsonl").write_text("\n".join(json.dumps(r) for r in self.ev["manifest"]) + "\n", encoding="utf-8")
        for rel, n in (("aux_drv.pt", 4096 + 17), ("model_sft_B.pt", 5000), ("dump/head_pred_step0.npz", 5000), ("dump/head_pred_update5000.npz", 5001), ("dump/head_pred_update10000.npz", 5002),
                       ("dump/scores_update10000_B.npz", 5003)):
            (a / rel).write_bytes(os.urandom(n))

    # ---- the stub children
    def runner(self, label, argv, outfile, timeout_h):
        self.calls.append(label)
        ok = {"rc": 0, "seconds": 0.0, "tail": ["stub"]}
        Path(outfile).write_text("stub\n")
        if label.startswith("forward_"):
            dev = label.split("_", 1)[1]
            if dev == "cuda" and self.busy["n"] > 0:
                self.busy["n"] -= 1
                return {"rc": 5, "seconds": 0.0, "tail": ["GPU BUSY (nothing loaded, nothing written): stub"]}
            d = Path(argv[argv.index("--out") + 1])
            heads = [x.split("=")[0] for x in argv[argv.index("--heads") + 1:]]
            (d / "identity.json").write_text(json.dumps({"PASS": True, "n_tokens": 1058, "n_tokens_bit_identical": 1058, "max_abs_delta": 0.0}))
            runj = {"heads": {h: {"sha256": "ab" * 32, "update": 10000 if h == A.REAL.name else 400, "n_params": A.HEAD_PARAMS, "feed_present": False,
                                  "config": {"feed": "off", "tap": "visual", "seed": 0, "depth": 2}} for h in heads}, "wall_s": 1.0}
            (d / "run.json").write_text(json.dumps(runj))
            for h in heads:
                (d / f"head_pred_{h}.npz").write_bytes(b"x")
            return ok
        if label.startswith("perception_navtest_"):
            dev = label.rsplit("_", 1)[1]
            out = Path(argv[argv.index("--out") + 1])
            arms = {A.REAL.name: fake_blk(self.p1_c3), "H1-s0": fake_blk(BANKED_BLOCK["p1"]["all"]["0-40"]["est"], self.cuda_perturb if dev == "cuda" else 0.0)}
            arms["H1-s0"]["p1"]["all"]["0-40"]["est"] = BANKED_BLOCK["p1"]["all"]["0-40"]["est"] + (self.cuda_perturb if dev == "cuda" else 0.0)
            pair = f"{A.REAL.name}-minus-H1-s0"
            out.write_text(json.dumps({"tool": "aux_s1_readout/1", "estimator": "paired log-cluster bootstrap, 10000 resamples, 95 %, seed 20260927; x", "arms": arms,
                                       "head_fit_pairs": {pair: {"iou_0-40": dict(self.d_c3, est=self.d_c3["est"] + self.pair_skew)}}}))
            return ok
        if label.startswith("perception_heldout_"):
            out = Path(argv[argv.index("--out") + 1])
            n_pred = argv.index("--pred-init") - argv.index("--pred") - 1
            if self.heldout_pair_refused and n_pred > 1:
                return {"rc": 1, "seconds": 0.0, "tail": ["ZZAUXS1_REFUSED stub: the comparator's frames differ"]}
            out.write_text(json.dumps({"tool": "aux_s1_readout/1", "estimator": "paired log-cluster bootstrap, 10000 resamples, 95 %, seed 20260927; x", "arms": {A.REAL.name: fake_blk(0.65)}}))
            return ok
        if label.startswith("nat_p1_"):
            dev = label.rsplit("_", 1)[1]
            out = Path(argv[argv.index("--out") + 1])
            pair = f"{A.REAL.name}-minus-H1-s0"
            comp_c3 = BANKED_BLOCK["p1"]["all"]["0-40"]["est"] + (self.cuda_perturb if dev == "cuda" else 0.0)
            nat_arm = {"c3": {"est": self.p1_c3 + self.nat_c3_skew, "ci95": [self.p1_c3 - 0.04, self.p1_c3 + 0.04]}, "natural": {"est": self.p1_nat, "ci95": [self.p1_nat - 0.04, self.p1_nat + 0.04]},
                       "per_city": {m: {"n_frames": 10, "n_logs": 3, "p1_natural_weighted": 0.5, "p1_unweighted": 0.5} for m in ("LV", "BOS", "PIT", "SG")}}
            comp_nat = N_BANKED["natural"] + (self.cuda_perturb if dev == "cuda" else 0.0)
            js = {"population": {"definition": "d", "spec": {"share_map": {"LV": 0.33, "BOS": 0.31, "PIT": 0.2, "SG": 0.16}, "share_stratum": {"zero": 0.0435, "random": 0.9565}}, "ess": 575.3679059904473},
                  "arms": {A.REAL.name: nat_arm, "H1-s0": {"c3": {"est": comp_c3, "ci95": [0, 1]}, "natural": {"est": comp_nat, "ci95": [0, 1]}, "per_city": nat_arm["per_city"]}},
                  "contrasts": {pair: {"c3": dict(self.d_c3), "natural": dict(self.d_nat, sensitivity_rerake={"ci95": [self.d_nat["ci95"][0] + 0.001, self.d_nat["ci95"][1] - 0.001]})}},
                  "identity": {"H1-s0": {"arm": "H1-s0", "natural": comp_nat, "banked_natural": N_BANKED["natural"], "c3": comp_c3, "banked_c3": BANKED_BLOCK["p1"]["all"]["0-40"]["est"],
                                         "ess": 575.3679059904473, "banked_ess": 575.3679059904473, "ok": True}}, "identity_ok": True}
            out.write_text(json.dumps(js))
            return ok
        if label == "veto":
            out = Path(argv[argv.index("--out") + 1])
            vs, vg = fake_veto_block(*self.veto), fake_veto_block(*self.vgt)
            dec = R.veto_row({"point": self.veto[0], "lo": self.veto[1], "hi": self.veto[2]}, {"point": self.vgt[0], "lo": self.vgt[1], "hi": self.vgt[2]}, AV_TEXT)
            if not self.veto_valid:
                dec = dict(dec, row_if_valid=dec["row"], row="INVALID", invalid_checks=["3_zero_control"])
            val = {"all_pass": self.veto_valid}
            for k in ("1_B_reproduces_pick", "0_population_identity", "2_A1_mask_iou_equals_harness_P1", "3_zero_control", "4_odac_bounds_rasters", "5_yflip_mutation_below_vgt"):
                val[k] = {"ok": self.veto_valid or k != "3_zero_control", "detail": "stub"}
            js = {"arm": "V-S1C", "arms": {"V-S1C": vs, "V-GT": vg, "O-DAC": fake_veto_block(2.5, 2.2, 2.9), "V-GT-YFLIP": fake_veto_block(-0.9, -1.6, -0.1)}, "validity": val,
                  "instrument_identity": {"ok": True, "max_abs_diff": 0.0}, "decision": dec, "tactical_levels": {"random": {"point": 73.0}, "B": {"point": 88.0}, "best_of_64": {"point": 98.0}},
                  "margin_curve": {"V-S1C": {"0.25": {"d_pdms": vs["d_pdms"]}}}}
            out.write_text(json.dumps(js))
            return ok
        if label == "desc_selection":
            out = Path(argv[argv.index("--out") + 1])
            sub = {"DAC": 97.5, "EP": 84.6, "TTC": 96.7, "NC": 99.0, "DDC": 96.0, "C": 99.9}
            out.write_text(json.dumps({"n_sets": 3137, "stamp": "BUDGET-CONFOUNDED", "arms": {a: {"PDMS_x100": 90.7, "subscores_x100": sub, "zero_picks": 105} for a in ("H1u10k-s0", "N-s0", "N-s1", "N-s2", "H1-s0")}}))
            return ok
        raise AssertionError(f"unexpected child {label}")

    def cfg(self, mode="cpu", gpu_wait_min=0.001):
        return {"repo": "Sayood/stub", "out": str(self.out), "work": str(self.work), "py_eval": "py_eval", "py_t": "py_t", "ctx_cache": "ctx", "floor_dir": A.FLOOR_DIR, "heldout_logs": A.HELDOUT_LOGS,
                "s1_cache": A.S1_CACHE, "poll_s": 0.01, "max_wait_h": 0.01, "max_errors": 3, "step_timeout_h": 1, "min_free_gib": 0.001, "force": False, "forward_device": mode,
                "gpu_wait_min": gpu_wait_min, "extra_heads": ()}

    def job(self, mode="cpu", remote=None, mod=A, gpu_wait_min=0.001, sleeps=None):
        remote = remote or S1R.LocalRemote(str(self.repo))
        sl = sleeps if sleeps is not None else []

        def capped_sleep(sec):                                   # a runaway loop (a mutated deadline) must fail the scenario, never hang the suite
            sl.append(sec)
            if len(sl) > 400:
                raise RuntimeError("runaway wait loop")
        j = mod.Job(self.cfg(mode, gpu_wait_min), remote, mod.REAL, dry=False, sleep=capped_sleep, secrets=[FAKE_TOKEN], runner=self.runner)
        j.preflight = lambda: None
        j.s1_file = lambda rel, pin: str(self.root / "s1" / rel)             # S1's big cached files are not hashed in the tests
        j.pdm_dir = lambda: str(self.root / "pdm")
        dummy = self.root / "dummy_head.pt"
        dummy.write_bytes(b"head")
        j.comparator_files = lambda: {mod.REAL.name: str(self.work / "cache" / "arms" / mod.REAL.name / "aux_drv.pt"), "H1-s0": str(dummy)}
        return j

    def markers(self):
        return sorted(p.name for p in self.out.glob("S1CREAD_*"))

    def marker(self, name):
        return json.loads((self.out / name).read_text(encoding="utf-8"))


N_BANKED = {"natural": 0.5608520369346311}


class SeqRemote(S1R.LocalRemote):
    """listings come from a script: the pod's push in several states; downloads from the finished repo dir"""

    def __init__(self, root, states):
        super().__init__(root)
        self.states, self.i = states, -1
        self.full = None

    def listing(self):
        self.i += 1
        if self.full is None:
            self.full = S1R.LocalRemote.listing(self)
        return self.states[min(self.i, len(self.states) - 1)](self.full)


def all_text(root: Path) -> str:
    chunks = []
    for p in Path(root).rglob("*"):
        if p.is_file() and p.stat().st_size < 5_000_000:
            chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def status_battery(mod=A) -> dict:
    """arm_status over the REAL spec: every file the read needs blocks completeness, and exactly one PASS token"""
    out = {}
    listing = {rel: {"size": 1, "sha256": None, "blob_id": "x"} for rel in mod.required_paths(mod.REAL) + [f"arms/{mod.REAL.name}/model_sft_B.pt", f"arms/{mod.REAL.name}/PASS_abc123.json"]}
    out["status_complete"] = mod.arm_status(listing, mod.REAL)["ready"]
    out["status_each_file_blocks"] = all(not mod.arm_status({k: v for k, v in listing.items() if k != rel}, mod.REAL)["ready"] for rel in listing)
    out["status_two_tokens_block"] = not mod.arm_status(dict(listing, **{f"arms/{mod.REAL.name}/PASS_def456.json": {"size": 1}}), mod.REAL)["ready"]
    return out


def pure_battery(mod=A) -> dict:
    out = status_battery(mod)
    os.environ["HF_TOKEN"] = FAKE_TOKEN
    try:
        j = mod.Job({"repo": "r", "out": tempfile.mkdtemp(prefix="s1c_t_"), "work": tempfile.mkdtemp(prefix="s1c_w_")}, None, mod.REAL, secrets=[FAKE_TOKEN])
        env = j.child_env()
    finally:
        del os.environ["HF_TOKEN"]
    out["child_env_has_no_token"] = "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env
    out["scrub_masks_the_secret"] = FAKE_TOKEN not in j.scrub(f"x {FAKE_TOKEN} y") and "hf_***" in j.scrub("hf_ABCDEFGH12345678")
    out["ident_tol_cuda_is_1e-4"] = mod.IDENT_TOL == {"cpu": 1e-9, "cuda": 1e-4}
    return out


def scenarios(mod=A):
    with contextlib.redirect_stdout(io.StringIO()):                # the driver logs to stdout; the suite stays readable
        return _scenarios(mod)


def _scenarios(mod=A):
    """{name: bool}.  `mod` may be a mutated copy of the driver (section D)."""
    out = {}
    # ---- O3: the expected outcome (no gain), through the whole job
    fx = Fixture(p1_c3=0.60, p1_nat=0.65, d_c3={"est": 0.01, "ci95": [-0.02, 0.04]}, d_nat={"est": 0.02, "ci95": [-0.01, 0.05]})
    rc = fx.job(mod=mod).run()
    res_md = (fx.out / "RESULT.md").read_text(encoding="utf-8") if (fx.out / "RESULT.md").exists() else ""
    out["O3_job_exit0_DONE_marker"] = rc == 0 and fx.markers() == ["S1CREAD_DONE"] and fx.marker("S1CREAD_DONE")["s1c_row"] == "O3"
    out["O3_result_quotes_the_next_lever_verbatim"] = "close the budget lever for the visual tap" in res_md
    out["O3_result_quotes_the_veto_next_arm_and_pauses_strategic"] = "STRATEGIC: PAUSED" in res_md and R.auxveto_row_cells(AV_TEXT, "AV-CEILING-ONLY")["next"][:40] in res_md
    out["O3_result_names_the_estimator_question"] = "another draw of LOGS" in res_md and "single seed" in res_md
    out["O3_marker_payload_has_both_rows"] = fx.marker("S1CREAD_DONE")["veto_row"] == "AV-CEILING-ONLY" and fx.marker("S1CREAD_DONE")["veto_valid"] is True
    out["O3_calls_every_instrument_once"] = fx.calls.count("forward_cpu") == 1 and fx.calls.count("veto") == 1 and fx.calls.count("desc_selection") == 1 and fx.calls.count("nat_p1_cpu") == 1
    out["O3_result_has_four_families_and_provenance"] = all(x in res_md for x in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC", "## 7. Provenance", "T0-like"))
    out["no_token_in_any_output"] = FAKE_TOKEN not in all_text(fx.out) and FAKE_TOKEN not in all_text(fx.work)
    # ---- O1 / O2 / O4
    fx = Fixture(p1_c3=0.83, p1_nat=0.85, d_c3={"est": 0.3, "ci95": [0.2, 0.4]}, d_nat={"est": 0.28, "ci95": [0.2, 0.36]})
    fx.job(mod=mod).run()
    out["O1_row_and_replicate_lever"] = fx.markers() == ["S1CREAD_DONE"] and fx.marker("S1CREAD_DONE")["s1c_row"] == "O1" and "replicate seed 1" in (fx.out / "RESULT.md").read_text(encoding="utf-8")
    fx = Fixture(p1_c3=0.62, p1_nat=0.66, d_c3={"est": 0.12, "ci95": [0.07, 0.17]}, d_nat={"est": 0.1, "ci95": [0.05, 0.15]})
    fx.job(mod=mod).run()
    out["O2_row"] = fx.markers() == ["S1CREAD_DONE"] and fx.marker("S1CREAD_DONE")["s1c_row"] == "O2"
    fx = Fixture(p1_c3=0.40, p1_nat=0.45, d_c3={"est": -0.1, "ci95": [-0.15, -0.05]}, d_nat={"est": -0.1, "ci95": [-0.15, -0.05]})
    fx.job(mod=mod).run()
    out["O4_row"] = fx.markers() == ["S1CREAD_DONE"] and fx.marker("S1CREAD_DONE")["s1c_row"] == "O4"
    # ---- INVALID (C1 fails) -> FAILED marker; RESULT.md still carries the INVALID row and its lever
    fx = Fixture(p1_c3=0.83, p1_nat=0.85, d_c3={"est": 0.3, "ci95": [0.2, 0.4]}, d_nat={"est": 0.28, "ci95": [0.2, 0.36]}, step0_iou=0.30)
    rc = fx.job(mod=mod).run()
    md = (fx.out / "RESULT.md").read_text(encoding="utf-8") if (fx.out / "RESULT.md").exists() else ""
    out["INVALID_C1_is_FAILED_not_DONE"] = rc == 1 and fx.markers() == ["S1CREAD_FAILED"] and "INVALID" in fx.marker("S1CREAD_FAILED")["reason"] and "C1 fails" in fx.marker("S1CREAD_FAILED")["reason"]
    out["INVALID_result_md_quotes_find_the_cause"] = "find the cause" in md and "INFORMATIONAL ONLY" in md
    # ---- the arm's evidence contradicts its label -> INVALID
    fx = Fixture(p1_c3=0.83, p1_nat=0.85, d_c3={"est": 0.3, "ci95": [0.2, 0.4]}, d_nat={"est": 0.28, "ci95": [0.2, 0.36]},
                 mutate_ev=lambda ev: [r.update(updates=399) for r in ev["sft"] if r["event"] == "done"])
    rc = fx.job(mod=mod).run()
    out["INVALID_evidence_contradiction_is_FAILED"] = rc == 1 and fx.markers() == ["S1CREAD_FAILED"] and "ID_SFT_DONE" in fx.marker("S1CREAD_FAILED")["reason"]
    # ---- the veto's own INVALID is recorded and does not fail the S1c marker
    fx = Fixture(veto_valid=False)
    fx.job(mod=mod).run()
    md = (fx.out / "RESULT.md").read_text(encoding="utf-8") if (fx.out / "RESULT.md").exists() else ""
    out["veto_INVALID_recorded_S1c_row_still_DONE"] = fx.markers() == ["S1CREAD_DONE"] and fx.marker("S1CREAD_DONE")["veto_row"] == "INVALID" and fx.marker("S1CREAD_DONE")["veto_valid"] is False
    out["veto_INVALID_commits_no_next_arm"] = "commits NO next arm" in md
    # ---- poll: nothing -> partial (no run_check) -> complete
    fx = Fixture()
    states = [lambda full: {}, lambda full: {k: v for k, v in full.items() if "run_check" not in k and "model_sft_B" not in k}, lambda full: dict(full)]
    sl = []
    rc = fx.job(mod=mod, remote=SeqRemote(str(fx.repo), states), sleeps=sl).run()
    log = (fx.out / "autoread.log").read_text(encoding="utf-8")
    out["poll_waits_through_empty_and_partial_then_reads"] = rc == 0 and len(sl) == 2 and log.count("not ready") == 2 and "COMPLETE" in log
    # ---- a push that lacks only model_sft_B.pt is NOT complete
    fx = Fixture()
    sl = []
    rc = fx.job(mod=mod, remote=SeqRemote(str(fx.repo), [lambda full: {k: v for k, v in full.items() if "model_sft_B" not in k}, lambda full: dict(full)]), sleeps=sl).run()
    out["poll_model_B_missing_is_not_ready"] = rc == 0 and len(sl) == 1
    # ---- poll deadline
    fx = Fixture()
    j = fx.job(mod=mod, remote=SeqRemote(str(fx.repo), [lambda full: {}]))
    j.c["max_wait_h"] = -1.0
    rc = j.run()
    out["poll_deadline_is_FAILED"] = rc == 1 and "did not become complete" in fx.marker("S1CREAD_FAILED")["reason"]
    # ---- listing errors are counted, then fatal; a token in the exception text never reaches a log or marker
    fx = Fixture()

    class Boom(S1R.LocalRemote):
        def listing(self):
            raise ConnectionError(f"401 unauthorised for {FAKE_TOKEN}")
    rc = fx.job(mod=mod, remote=Boom(str(fx.repo))).run()
    out["listing_errors_become_FAILED_after_max_errors"] = rc == 1 and "consecutive listing errors" in fx.marker("S1CREAD_FAILED")["reason"]
    out["token_in_an_exception_text_is_scrubbed_everywhere"] = FAKE_TOKEN not in all_text(fx.out) and "hf_***" in all_text(fx.out)
    # ---- a corrupt download is refused (the listing says one hash, the bytes are another), as a NAMED refusal
    fx = Fixture()
    lies = S1R.LocalRemote(str(fx.repo), lies={f"arms/{mod.REAL.name}/aux_drv.pt": {"sha256": "0" * 64}})
    rc = fx.job(mod=mod, remote=lies).run()
    out["corrupt_download_REFUSED_named_and_deleted"] = (rc == 1 and "sha256" in fx.marker("S1CREAD_FAILED")["reason"] and "unexpected" not in fx.marker("S1CREAD_FAILED")["reason"]
                                                        and not (fx.work / "cache" / "arms" / mod.REAL.name / "aux_drv.pt").exists())
    # ---- GPU busy: wait with backoff, never break the lock, then proceed (cuda mode)
    fx = Fixture()
    fx.busy["n"] = 3
    sl = []
    rc = fx.job(mode="cuda", mod=mod, gpu_wait_min=60, sleeps=sl).run()
    out["cuda_busy_waits_with_backoff_then_reads"] = rc == 0 and sl == [60, 120, 240] and fx.calls.count("forward_cuda") == 4 and fx.markers() == ["S1CREAD_DONE"] and fx.marker("S1CREAD_DONE")["device"] == "cuda"
    fx = Fixture()
    fx.busy["n"] = 10 ** 6
    rc = fx.job(mode="cuda", mod=mod, gpu_wait_min=0.0).run()
    out["cuda_always_busy_is_FAILED_not_cpu"] = rc == 1 and "not available within" in fx.marker("S1CREAD_FAILED")["reason"] and "forward_cpu" not in fx.calls
    out["busy_result_not_cached"] = not (fx.out / "stages" / "forward_cuda.json").exists()
    fx = Fixture()
    fx.busy["n"] = 10 ** 6
    rc = fx.job(mode="auto", mod=mod, gpu_wait_min=0.0).run()
    out["auto_busy_falls_back_to_cpu"] = rc == 0 and fx.marker("S1CREAD_DONE")["device"] == "cpu" and "forward_cpu" in fx.calls and "forward on cuda" in (fx.out / "autoread.log").read_text(encoding="utf-8")
    # ---- an identity miss on CUDA falls back to CPU (auto) / fails (cuda); a miss under 1e-4 is accepted
    fx = Fixture()
    fx.cuda_perturb = 5e-4
    rc = fx.job(mode="auto", mod=mod).run()
    out["auto_identity_miss_on_cuda_falls_back_to_cpu"] = rc == 0 and fx.marker("S1CREAD_DONE")["device"] == "cpu" and "forward_cuda" in fx.calls and "forward_cpu" in fx.calls
    fx = Fixture()
    fx.cuda_perturb = 5e-4
    rc = fx.job(mode="cuda", mod=mod).run()
    out["cuda_identity_miss_is_FAILED"] = rc == 1 and "instrument identity FAILED" in fx.marker("S1CREAD_FAILED")["reason"]
    fx = Fixture()
    fx.cuda_perturb = 5e-5
    rc = fx.job(mode="cuda", mod=mod).run()
    out["cuda_within_1e-4_is_accepted"] = rc == 0 and fx.marker("S1CREAD_DONE")["device"] == "cuda"
    # ---- resume: a DONE marker makes a relaunch a no-op; without it every completed stage is reused
    fx = Fixture()
    fx.job(mod=mod).run()
    n1 = len(fx.calls)
    rc2 = fx.job(mod=mod).run()
    out["done_marker_makes_a_relaunch_a_noop"] = rc2 == 0 and len(fx.calls) == n1
    (fx.out / "S1CREAD_DONE").unlink()
    fx.job(mod=mod).run()
    out["resume_reuses_cached_stages"] = len(fx.calls) == n1 and fx.markers() == ["S1CREAD_DONE"]
    # ---- ... and a CHANGED head invalidates them (the stage cache is keyed by input fingerprint)
    (fx.repo / "arms" / mod.REAL.name / "aux_drv.pt").write_bytes(os.urandom(5000))
    (fx.out / "S1CREAD_DONE").unlink()
    fx.job(mod=mod).run()
    out["a_changed_head_invalidates_the_stage_cache"] = fx.calls.count("forward_cpu") == 2
    # ---- the two instruments must agree on the same frames (nat_p1's C3 read vs the harness's; the C3 contrast vs head_fit_pairs)
    fx = Fixture(nat_c3_skew=1e-6)
    rc = fx.job(mod=mod).run()
    out["nat_vs_harness_C3_disagreement_is_FAILED"] = rc == 1 and "disagree on the same frames" in fx.marker("S1CREAD_FAILED")["reason"]
    fx = Fixture(pair_skew=1e-6)
    rc = fx.job(mod=mod).run()
    out["nat_vs_harness_C3_contrast_disagreement_is_FAILED"] = rc == 1 and "head_fit_pairs" in fx.marker("S1CREAD_FAILED")["reason"]
    # ---- the comparator beside the arm in the held-out read is informational: a refusal re-reads the arm alone, loudly
    fx = Fixture(heldout_pair_refused=True)
    rc = fx.job(mod=mod).run()
    out["heldout_pair_refusal_rereads_the_arm_alone"] = (rc == 0 and "perception_heldout_update10000_alone" in fx.calls and "re-reading the arm alone" in (fx.out / "autoread.log").read_text(encoding="utf-8")
                                                         and fx.markers() == ["S1CREAD_DONE"])
    return out


def job_tests():
    sc = scenarios()
    for k, v in sc.items():
        check(f"job_{k}", bool(v))
    for k, v in pure_battery().items():
        check(f"pure_{k}", bool(v))
    return sc


# ================================================================================================================ D. mutations
def rules_scen(mr):
    """a battery that exercises s1c_rules directly (for the rules mutations)"""
    out = {}
    base = {"c1_ok": True, "runcheck_ok": True, "evidence_ok": True, "p1_c3": 0.6, "p1_nat": 0.65}
    n_bad = 0
    for valid, c3, nat, kc, kn in itertools.product((True, False), (0.40, 0.80, 0.86), (0.40, 0.80, 0.86), DELTAS, DELTAS):
        m = {"c1_ok": valid, "runcheck_ok": True, "evidence_ok": True, "p1_c3": c3, "p1_nat": nat, "d_c3": DELTAS[kc], "d_nat": DELTAS[kn]}
        if mr.decide_s1c(m, S1C_TEXT)["row"] != oracle_row(valid, c3, nat, DELTAS[kc], DELTAS[kn]):
            n_bad += 1
    out["grid"] = n_bad == 0
    d = mr.decide_s1c(dict(base, d_c3=DELTAS["big_neg"], d_nat=DELTAS["zero"]), S1C_TEXT)
    out["overlap"] = d["overlap"] == ["O4"]
    out["veto_go_bar"] = mr.decide_veto({"lo": 0.1, "hi": 0.5, "point": 0.2}, {"lo": 0.0}) == "AV-PROMISING" and mr.decide_veto({"lo": 0.1, "hi": 0.5, "point": 0.3}, {"lo": 0.0}) == "AV-GO"
    tmp = Path(tempfile.mkdtemp(prefix="s1c_g2_")) / "x.md"
    tmp.write_text(S1C_TEXT.replace("0.80", "0.79", 1), encoding="utf-8", newline="\n")
    try:
        mr.guard_prereg_s1c(tmp)
        out["guard"] = False
    except ValueError:
        out["guard"] = True
    return out


DRIVER_MUTATIONS = [
    ("M9 the GPU-busy retry removed (rc 5 treated as a failure)", 'if r["rc"] == 5 and device == "cuda":', "if False:"),
    ("M10 the GPU-busy backoff never grows", "backoff = min(backoff * 2, 300)", "backoff = 60"),
    ("M11 the comparator identity is always ok", 'ok = bool(dc["ok"] and nat_ok)', "ok = True"),
    ("M12 a busy result is cached", 'if not (isinstance(res, dict) and res.get("status") in ("busy", "NOT_READ")):', "if True:"),
    ("M13 an INVALID read is recorded as DONE", 'if dec["row"] == "INVALID":', "if False:"),
    ("M14 children inherit HF_TOKEN", 'if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN")}', "if True}"),
    ("M15 the token is not scrubbed", 's = TOKEN_RE.sub("hf_***", str(s))\n        for sec in self.secrets:\n            s = s.replace(sec, "hf_***")\n', "s = str(s)\n"),
    ("M16 model_sft_B.pt not required for completeness", "miss = [p for p in required_paths(spec) + [a + f for f in spec.extra_listed] if p not in listing]", "miss = [p for p in required_paths(spec) if p not in listing]"),
    ("M17 a push without run_check.json counts as complete", 'ARM_SMALL = ("arms_manifest.jsonl", "run_check.json", ', 'ARM_SMALL = ("arms_manifest.jsonl", '),
    ("M18 the poll deadline ignored", 'if time.time() > deadline:\n                raise Refuse(f"arms/{spec.name}/ did not become complete', 'if False:\n                raise Refuse(f"arms/{spec.name}/ did not become complete'),
    ("M19 the CUDA tolerance loosened to 1e-2", 'IDENT_TOL = {"cpu": 1e-9, "cuda": 1e-4}', 'IDENT_TOL = {"cpu": 1e-9, "cuda": 1e-2}'),
    ("M20 evidence problems no longer gate the read", '"evidence_ok": not P, "token_sha256"', '"evidence_ok": True, "token_sha256"'),
    ("M21 the stage cache ignores the fingerprint", 'if d.get("fp") == fp:', "if True:"),
    ("M22 the C1 verdict is not passed to the rule", '"c1_ok": ev["c1"]["ok"],', '"c1_ok": True,'),
    ("M23 a failed download verification is not a named refusal", "except (Refuse, S1R.FetchRefused, S1R.EvidenceRefused) as e:", "except (Refuse, S1R.EvidenceRefused) as e:"),
    ("M24 the natural read is not cross-checked against the harness's C3", 'if abs(a_nat["c3"]["est"] - p1_c3) > 1e-12:', "if False:"),
    ("M26 a refused held-out pair is fatal instead of re-read alone", "if len(preds) < 2:", "if True:"),
    ("M25 the C3 contrast is not cross-checked against the harness's head_fit_pairs", 'if cross["est_abs_diff"] > 1e-12 or cross["ci_abs_diff"] > 1e-12:', "if False:"),
]


def mutation_tests():
    rule_muts = [
        ("M1 INVALID does not gate the row (C1 ignored)", 'valid = bool(m["c1_ok"] and m["runcheck_ok"] and m["evidence_ok"])', 'valid = bool(m["runcheck_ok"] and m["evidence_ok"])'),
        ("M2 O3 and O4 swapped in the table order", '    elif conds["O3: Delta_C3 or Delta_NAT not separated"]:\n        row = "O3"\n    elif conds["O4: Delta_C3 or Delta_NAT separated and < 0"]:\n        row = "O4"',
         '    elif conds["O4: Delta_C3 or Delta_NAT separated and < 0"]:\n        row = "O4"\n    elif conds["O3: Delta_C3 or Delta_NAT not separated"]:\n        row = "O3"'),
        ("M3 F dropped from separated()", "    if lo > 0 and est > F:", "    if lo > 0:"),
        ("M4 the K2 bar >= becomes >", 'k2_c3 = m["p1_c3"] >= K2_BAR', 'k2_c3 = m["p1_c3"] > K2_BAR'),
        ("M5 O1 needs only C3", '"O1: P1_C3 >= 0.80 and P1_NAT >= 0.80": bool(k2_c3 and k2_nat),', '"O1: P1_C3 >= 0.80 and P1_NAT >= 0.80": bool(k2_c3),'),
        ("M6 the overlap flag dropped", '    if row == "O3" and conds["O4: Delta_C3 or Delta_NAT separated and < 0"]:\n        overlap.append("O4")', '    if False:\n        overlap.append("O4")'),
        ("M7 the AV-GO point bar 0.30 -> 0.0", "AV_GO_POINT = 0.30 ", "AV_GO_POINT = 0.0 "),
        ("M8 the S1c prereg body guard removed", "    if got != PREREG_S1C_BODY_SHA256:", "    if False:"),
    ]
    for label, old, new in rule_muts:
        try:
            red = not all(rules_scen(mut_r(old, new)).values())
        except AssertionError:
            raise
        except Exception:
            red = True
        check(f"mutation_RED: {label}", red)
    base = scenarios()
    check("mutation_baseline_scenarios_green", all(base.values()), str([k for k, v in base.items() if not v]))
    for label, old, new in DRIVER_MUTATIONS:
        try:
            ma = mut_a(old, new)
            sc = scenarios(ma)
            red = (not all(sc.values())) or (not all(pure_battery(ma).values()))
        except AssertionError:
            raise
        except Exception:
            red = True
        check(f"mutation_RED: {label}", red)


def main():
    rules_tests()
    evidence_tests()
    job_tests()
    mutation_tests()
    if FAILS:
        print(f"ZZTEST_FAIL {len(FAILS)} failed, {N_OK} passed: " + " | ".join(FAILS))
        return 1
    print(f"ZZTEST_PASS {N_OK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
