#!/usr/bin/env python3
"""R0e-pick launch blockers B9 + B10 (raw/2026-10-18-r0e-pick/PREREG_R0E_PICK.md sections 4 and 5.1): PLANTED inputs, driven through the real code, and every guard mutated RED.  Run:  python refe/test_r0e_b9b10.py -> ZZTEST_PASS <n>.
CPU only, stdlib only (no numpy, no torch, no pod, no GPU, no token).  A work tree is COPIED to a temp dir for the unmutated run and for every mutation, so no source file is ever touched.

B9  R0e-id runs K3 OBSERVE-ONLY ("R0e-id and R1o run K3 observe-only; K2 and K6 stay live"); R0e-pick and R1e-pick keep K3 LIVE.  The kill monitor is the real `rls1_checks.main(["monitor", ...])` with the RL-val checker, the liveness probe and the group
    kill replaced by stubs that PLANT two consecutive g3 breaches (a K3): R0e-id with --observe-only logs ZZRLS1_K3_OBSERVED and is NOT killed; R0e-pick and R1e-pick (flag refused, and without it) are killed K3; R1o is unchanged; K4 and K2 still kill R0e-id.
    The chain's own OBS block is RUN under bash for every arm.
B10 R0e-pick refuses to start unless R0e-id's directory carries a genuine eval/r0e_id_read.py verdict reading ID-PASS.  The reader is driven with planted directories for EVERY clause of 5.1 (each broken alone, and at its boundary), the gate with
    missing / malformed / ID-FAIL / ID-DIVERGED / NOT_READ / stale / copied-from-another-run / forged / ID-PASS verdicts, and the chain's own gate block RUN under bash against the real gate.
Then every MUTATION must turn a named check RED: removing R0e-id from the allowed set, making R0e-pick observe-only, deleting the gate call from the chain, a gate that cannot refuse, an ID-FAIL clause deleted, ... ("built, tested, unreachable from its caller")."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
BASH = shutil.which("bash")
TREE_FILES = ("refe/rls1_checks.py", "refe/rls1_pod.sh", "eval/r0e_id_read.py", "eval/r0d_id_reference.json", "eval/rls1e_arms.json", "eval/rls1d_arms.json")
N = 0
_UNIQ = [0]


def posix(p):
    return str(p).replace("\\", "/")


def load(path, tag):
    _UNIQ[0] += 1
    spec = importlib.util.spec_from_file_location(f"{tag}_{_UNIQ[0]}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def copy_tree(dst, mutate=None):
    """copy the six files under test to dst/<same layout>; `mutate` = (relpath, old, new) applied once (exactly one occurrence) -> the mutated tree"""
    for rel in TREE_FILES:
        (Path(dst) / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PKG / rel, Path(dst) / rel)
    if mutate:
        rel, old, new = mutate
        p = Path(dst) / rel
        t = p.read_text(encoding="utf-8")
        assert t.count(old) == 1, (rel, t.count(old), old[:70])
        p.write_text(t.replace(old, new), encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------------------------------------------- planted R0e-id directories
def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def build_dir(root, ref, name="R0e-id", **m):
    """-> the planted arm directory <root>/arms/<name>.  Defaults = a genuine ID-PASS run; each keyword breaks exactly one thing:
    temp, drop_con_key, epo_edit, train_sets, d_base / d_A / d_B (step0 pick offsets, points), g13_ok, g13_mutate, dttc / dpdms (u250 contrast offsets from R0d, points), no_u250, no_preflight, two_data, foreign, salt, wrong_scores_sha"""
    d = Path(root) / "arms" / name
    (d / "scores").mkdir(parents=True)
    salt = str(m.get("salt", "")).encode()
    shas = {}
    for side in ("A", "B"):
        b = f"u250{side}".encode() + salt
        (d / "scores" / f"scores_update250_{side}.npz").write_bytes(b)
        shas[side] = sha_b(b)
    if m.get("wrong_scores_sha"):
        shas["B"] = sha_b(b"another run's dump")
    econ = dict(ref["epo_con"])
    econ["temp"] = m.get("temp", 0.1)
    if m.get("drop_con_key"):
        econ.pop(m["drop_con_key"])
    epo = json.loads(json.dumps(ref["epo"]))
    if m.get("epo_edit"):
        epo["lam"] = 2.0
    data = {"event": "data", "train_sets": m.get("train_sets", 16000), "epo": epo, "epo_con": econ, "at": "2026-10-20T08:00:00" + str(m.get("salt", ""))}
    p0 = ref["step0_pick"]
    ev0 = {"event": "eval", "tag": "step0", "n": 3000, **{s: {"pick": p0[s] + m.get(f"d_{s}", 0.0)} for s in ("base", "A", "B")}}
    base = f"/workspace/data/rls1/arms/{m.get('foreign', name)}/scores/"
    dump = lambda t: {"event": "score_dump", "tag": t, "n_sets": 3000, "files": {s: f"{base}scores_{t}_{s}.npz" for s in ("base", "A", "B")}}          # noqa: E731
    rows = [data] + ([dict(data)] if m.get("two_data") else []) + [{"event": "start", "updates": 1000}, ev0, dump("step0"), {"event": "eval", "tag": "update250"}, dump("update250"), {"event": "done", "updates": 1000}]
    (d / "sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    if not m.get("no_preflight"):
        g13 = {"ok": m.get("g13_ok", True), "temp_reach": {"ok": True}}
        if m.get("g13_mutate"):
            g13["mutate"] = ["REFE_RL_MUTATE_CON_DETACH"]
        (d / "PREFLIGHT_PASS.json").write_text(json.dumps({"event": "preflight", "verdict": "PASS", "gates": {"G1_trainable_is_exactly_the_scorer": {"ok": True}, "G13_epo_con_live_and_identity": g13}}), encoding="utf-8", newline="\n")
    if not m.get("no_u250"):
        rc = ref["u250_contrast_B_minus_ownA_x100"]
        q = {"base": "A", "arms": {"A": {}, "B": {}}, "eligibility": {"sets_used": 3000}, "inputs": {"scores_sha256": shas},
             "paired_vs_base": {"B": {"quantities": {"TTC": {"mean": rc["TTC"] + m.get("dttc", 0.0)}, "PDMS": {"mean": rc["PDMS"] + m.get("dpdms", 0.0)}}}}}
        (d / "rlval_id_u250.json").write_text(json.dumps(q), encoding="utf-8", newline="\n")
    return d


def read_and_write(ID, d, eval_dir=None):
    rec = ID.read_arm(str(d), eval_dir)
    rec["record_sha256"] = ID.record_digest(rec)
    (Path(d) / ID.OUT_NAME).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return rec


def forge(ID, d, **over):
    """a forger who knows the format: a hand-written ID-PASS record with the right tool, right input hashes and a correctly recomputed record_sha256"""
    rec = {"tool": ID.TOOL, "arm": ID.ARM, "outcome": "ID-PASS", "fail_reasons": [], "problems": [], "checks": {k: True for k in ID.EXPECTED_CHECKS}, "values": {},
           "inputs_sha256": {n: ID.sha256_file(os.path.join(d, n)) for n in ID.REQUIRED_INPUTS}, "package_sha256": {}, "thresholds": {}}
    rec.update(over)
    rec["record_sha256"] = ID.record_digest(rec)
    (Path(d) / ID.OUT_NAME).write_text(json.dumps(rec), encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------------------------------------------- group 1: python (the reader, the gate, the kill monitor)
def checks_py(tree):
    """-> [names of the checks that FAILED] for the six files under `tree`"""
    global N
    bad = []

    def c(name, cond):
        global N
        N += 1
        try:
            ok = bool(cond() if callable(cond) else cond)
        except Exception as e:                                                                                       # noqa: BLE001
            ok = False
            name = f"{name} [raised {type(e).__name__}: {str(e)[:80]}]"
        if not ok:
            bad.append(name)
    tree = Path(tree)
    RC = load(tree / "refe" / "rls1_checks.py", "rc")
    ID = load(tree / "eval" / "r0e_id_read.py", "idr")
    ref = json.loads((tree / "eval" / "r0d_id_reference.json").read_text(encoding="utf-8"))
    tmp = Path(tempfile.mkdtemp(prefix="b9b10_"))
    ev = tree / "eval"

    # ======================================================================= B9: the allowed set and the real monitor on planted K3s
    c("B9_allowed_set_is_exactly_R1o_and_R0e_id", lambda: tuple(RC.OBSERVE_ONLY_ARMS) == ("R1o", "R0e-id"))

    def run_cli(arm, observe, events, recs, alive_seq=(True, False)):
        od = tmp / f"mon_{arm}_{int(observe)}_{len(list(tmp.glob('mon_*')))}"
        od.mkdir()
        (od / "sft.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
        killed, seq = [], list(alive_seq)

        class Stub:
            def __init__(self, *a, **k):
                pass

            def __call__(self, tag, files):
                return recs[tag]
        saved = (RC.RlValChecker, RC._alive, RC._kill_group)
        RC.RlValChecker, RC._alive, RC._kill_group = Stub, (lambda pid: seq.pop(0) if len(seq) > 1 else seq[0]), killed.append
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = RC.main(["monitor", "--arm", arm, "--out", str(od), "--refdir", "r", "--pdm", "p", "--terms", "t", "--sets", "s", "--tsv", "v", "--pid", "99", "--poll", "0", *(["--observe-only"] if observe else [])])
        finally:
            RC.RlValChecker, RC._alive, RC._kill_group = saved
        kp, cp = od / "KILLED.json", od / "kill_checks.jsonl"
        code = json.loads(kp.read_text(encoding="utf-8"))["code"] if kp.exists() else None
        rows = [json.loads(x) for x in cp.read_text(encoding="utf-8").splitlines()] if cp.exists() else []
        return rc, killed, code, buf.getvalue(), rows
    rec = lambda t, g3=True, g5=True, k4=False: {"tag": t, "g3_ok": g3, "g3": None, "g5_ok": g5, "k4_breach": k4, "k5_breach": False}          # noqa: E731
    dump = lambda t: {"event": "score_dump", "tag": t, "files": {"B": "b", "A": "a", "base": "z"}}          # noqa: E731
    evs = [dump("step0"), dump("update250"), dump("update500"), dump("update750")]
    k3 = {"step0": rec("step0"), "update250": rec("update250", g3=False), "update500": rec("update500", g3=False), "update750": rec("update750", g3=False)}
    k4 = {"step0": rec("step0"), "update250": rec("update250", k4=True), "update500": rec("update500", k4=True), "update750": rec("update750")}
    r = run_cli("R0e-id", True, evs, k3)
    c("B9_PLANTED_K3_R0e_id_observe_only_logs_and_is_NOT_killed", lambda: r[0] == 0 and r[1] == [] and r[2] is None and r[3].count("ZZRLS1_K3_OBSERVED R0e-id update500") == 1 and r[3].count("ZZRLS1_K3_OBSERVED R0e-id update750") == 1
      and len(r[4]) == 3 and all(x["kill"] is None for x in r[4]))
    r = run_cli("R0e-id", False, evs, k3)
    c("B9_R0e_id_without_the_flag_would_be_killed_K3_the_control_of_the_test", lambda: r[0] == 2 and r[2] == "K3" and r[1] == [99])
    for arm in ("R0e-pick", "R1e-pick", "R0d"):
        r = run_cli(arm, False, evs, k3)
        c(f"B9_PLANTED_K3_{arm}_is_live_and_KILLS", lambda r=r: r[0] == 2 and r[2] == "K3" and r[1] == [99] and "ZZRLS1_K3_OBSERVED" not in r[3])
        r = run_cli(arm, True, evs, k3)
        c(f"B9_{arm}_cannot_be_made_observe_only_the_flag_is_refused_and_nothing_runs", lambda r=r: r[0] == 1 and r[3].startswith("ZZRLS1_OBSERVE_ONLY_REFUSED") and r[1] == [] and r[2] is None)
    r = run_cli("R1o", True, evs, k3)
    c("B9_R1o_is_unchanged_observe_only_logs_and_is_not_killed", lambda: r[0] == 0 and r[2] is None and r[3].count("ZZRLS1_K3_OBSERVED R1o update500") == 1)
    r = run_cli("R1o", False, evs, k3)
    c("B9_R1o_without_the_flag_is_killed_as_ever", lambda: r[0] == 2 and r[2] == "K3")
    r = run_cli("R0e-id", True, evs, k4)
    c("B9_R0e_id_observe_only_still_dies_on_K4", lambda: r[0] == 2 and r[2] == "K4")
    r = run_cli("R0e-id", True, [{"event": "train", "update": 25, "bceA": 0.3, "lossB": float("nan")}], {})
    c("B9_R0e_id_observe_only_still_dies_on_K2_a_non_finite_loss", lambda: r[0] == 2 and r[2] == "K2")
    # replay (the retroactive re-judge) takes the same flag
    od = tmp / "replay_od"
    od.mkdir()
    (od / "sft.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    c("B9_replay_R0e_id_observe_only_does_not_retro_kill", lambda: (lambda x: x[0] is None and all(y.get("observe_only") is True for y in x[2]))(RC.replay("R0e-id", str(od), lambda t, f: k3[t], observe_only=True)))
    c("B9_replay_R0e_pick_finds_the_K3", lambda: RC.replay("R0e-pick", str(od), lambda t, f: k3[t])[0] == "K3")

    # ======================================================================= B10: the reader, clause by clause (each clause broken ALONE, and at its boundary)
    def verdict(**m):
        d = build_dir(tmp / f"d{len(list(tmp.glob('d*')))}", ref, **m)
        return d, read_and_write(ID, d, str(ev))
    d_ok, rec_ok = verdict()
    c("B10_reader_planted_genuine_run_reads_ID_PASS_with_all_nine_clauses_evaluated", lambda: rec_ok["outcome"] == "ID-PASS" and sorted(rec_ok["checks"]) == sorted(ID.EXPECTED_CHECKS) and all(rec_ok["checks"].values()))
    c("B10_gate_accepts_the_genuine_ID_PASS", lambda: RC.id_gate(str(d_ok))[0] is True)
    fail_cases = {
        "epo_con_temp_0.01": dict(temp=0.01), "epo_con_temp_missing_key": dict(drop_con_key="lr"), "epo_block_edited": dict(epo_edit=True), "train_sets_15999": dict(train_sets=15999),
        "G13_not_ok": dict(g13_ok=False), "G13_under_a_regression_switch": dict(g13_mutate=True),
        "step0_base_pick_0.11_off": dict(d_base=0.11), "step0_A_pick_minus_0.11_off": dict(d_A=-0.11), "step0_B_pick_0.2_off": dict(d_B=0.2),
    }
    for nm, kw in fail_cases.items():
        d, rr = verdict(**kw)
        c(f"B10_reader_ID_FAIL_alone_{nm}", lambda rr=rr: rr["outcome"] == "ID-FAIL" and len(rr["fail_reasons"]) == 1)
        c(f"B10_gate_REFUSES_ID_FAIL_{nm}", lambda d=d: (lambda g: g[0] is False and "ID-FAIL" in g[1])(RC.id_gate(str(d))))
    # the arms-file self-check: a planted arms file whose R0e-id argv is NOT R0d's plus the lever
    ev2 = tmp / "eval_broken_arms"
    shutil.copytree(ev, ev2)
    af = json.loads((ev2 / "rls1e_arms.json").read_text(encoding="utf-8"))
    argv = af["arms"]["R0e-id"]["argv"]
    argv[argv.index("--lr") + 1] = "2e-4"
    (ev2 / "rls1e_arms.json").write_text(json.dumps(af), encoding="utf-8", newline="\n")
    d = build_dir(tmp / "dself", ref)
    rr = ID.read_arm(str(d), str(ev2))
    c("B10_reader_ID_FAIL_alone_arms_file_self_check", lambda: rr["outcome"] == "ID-FAIL" and len(rr["fail_reasons"]) == 1 and "self-check" in rr["fail_reasons"][0])
    d = build_dir(tmp / "dself2", ref)
    af = json.loads((ev / "rls1e_arms.json").read_text(encoding="utf-8"))
    af["self_check"]["per_arm"]["R0e-id"]["equals_R0d_after_swap_and_lever_removal"] = False
    ev3 = tmp / "eval_flag_false"
    shutil.copytree(ev, ev3)
    (ev3 / "rls1e_arms.json").write_text(json.dumps(af), encoding="utf-8", newline="\n")
    c("B10_reader_ID_FAIL_when_the_arms_file_s_stored_self_check_says_false", lambda: ID.read_arm(str(d), str(ev3))["outcome"] == "ID-FAIL")
    # boundaries: exactly 0.10 / 0.50 is not "greater"; just beyond is
    d, rr = verdict(d_base=0.10, d_A=-0.10, d_B=0.10)
    c("B10_reader_boundary_step0_exactly_0.10_points_is_still_ID_PASS", lambda: rr["outcome"] == "ID-PASS")
    d, rr = verdict(d_B=0.1001)
    c("B10_reader_boundary_step0_0.1001_points_is_ID_FAIL", lambda: rr["outcome"] == "ID-FAIL")
    d, rr = verdict(dttc=0.50, dpdms=-0.50)
    c("B10_reader_boundary_u250_exactly_0.50_points_is_still_ID_PASS", lambda: rr["outcome"] == "ID-PASS")
    for nm, kw in {"TTC_plus_0.51": dict(dttc=0.51), "PDMS_minus_0.51": dict(dpdms=-0.51), "TTC_minus_0.6_PDMS_plus_0.6": dict(dttc=-0.6, dpdms=0.6)}.items():
        d, rr = verdict(**kw)
        c(f"B10_reader_ID_DIVERGED_{nm}", lambda rr=rr: rr["outcome"] == "ID-DIVERGED")
        c(f"B10_gate_REFUSES_ID_DIVERGED_{nm}", lambda d=d: (lambda g: g[0] is False and "ID-DIVERGED" in g[1])(RC.id_gate(str(d))))
    c("B10_ID_FAIL_wins_over_ID_DIVERGED_the_order_of_5_1", lambda: verdict(temp=0.01, dttc=0.9)[1]["outcome"] == "ID-FAIL")
    # NOT_READ: an unevaluated clause is never a pass
    for nm, kw in {"u250_read_missing": dict(no_u250=True), "preflight_missing": dict(no_preflight=True), "two_data_events": dict(two_data=True), "another_runs_log_in_the_dir": dict(foreign="R0d"),
                   "u250_read_from_other_dumps": dict(wrong_scores_sha=True)}.items():
        d, rr = verdict(**kw)
        c(f"B10_reader_NOT_READ_{nm}", lambda rr=rr: rr["outcome"] == "NOT_READ" and rr["problems"])
        c(f"B10_gate_REFUSES_NOT_READ_{nm}", lambda d=d: RC.id_gate(str(d))[0] is False)
    d = build_dir(tmp / "dnoscores", ref)
    shutil.rmtree(d / "scores")
    c("B10_reader_NOT_READ_when_the_u250_dumps_are_not_on_disk_the_read_cannot_be_tied_to_the_run", lambda: ID.read_arm(str(d), str(ev))["outcome"] == "NOT_READ")
    # missing / malformed verdict files
    c("B10_gate_REFUSES_a_missing_output_no_directory_at_all", lambda: RC.id_gate(str(tmp / "nowhere" / "R0e-id"))[0] is False)
    d = build_dir(tmp / "dmissing", ref)
    c("B10_gate_REFUSES_a_directory_without_a_verdict_file", lambda: (lambda g: g[0] is False and "r0e_id_read.py" in g[1])(RC.id_gate(str(d))))
    for nm, body in {"garbage": "not json at all", "list_not_object": "[1, 2]", "truncated": '{"tool": "r0e_id_read/1", "outcome": "ID-PA', "empty": "", "null": "null", "outcome_only": '{"outcome": "ID-PASS"}'}.items():
        d = build_dir(tmp / f"dmal_{nm}", ref)
        (d / "id_verdict.json").write_text(body, encoding="utf-8")
        c(f"B10_gate_REFUSES_a_malformed_verdict_{nm}", lambda d=d: RC.id_gate(str(d))[0] is False)
    # stale / another run's / forged
    d, _ = verdict()
    (d / "sft.jsonl").write_text((d / "sft.jsonl").read_text(encoding="utf-8") + json.dumps({"event": "train", "update": 1}) + "\n", encoding="utf-8")
    c("B10_gate_REFUSES_a_stale_verdict_sft_jsonl_changed_after_the_read", lambda: (lambda g: g[0] is False and "changed since the read" in g[1])(RC.id_gate(str(d))))
    d, _ = verdict()
    (d / "rlval_id_u250.json").write_text((d / "rlval_id_u250.json").read_text(encoding="utf-8") + " ", encoding="utf-8")
    c("B10_gate_REFUSES_a_stale_verdict_the_u250_read_changed_after_the_read", lambda: RC.id_gate(str(d))[0] is False)
    d, _ = verdict()
    (d / "PREFLIGHT_PASS.json").unlink()
    c("B10_gate_REFUSES_a_verdict_whose_input_vanished", lambda: RC.id_gate(str(d))[0] is False)
    dx, _ = verdict(salt="X")                                           # run X: its own genuine PASS
    dy, _ = verdict(salt="Y")                                           # run Y: ALSO genuine ID-PASS, a different run (other dumps, other log)
    shutil.copyfile(dx / "id_verdict.json", dy / "id_verdict.json")
    c("B10_gate_REFUSES_another_runs_verdict_copied_into_this_directory", lambda: (lambda g: g[0] is False and "changed since the read" in g[1])(RC.id_gate(str(dy))))
    d, _ = verdict()
    af_live = json.loads((tree / "eval" / "rls1e_arms.json").read_text(encoding="utf-8"))
    tree_arms = tree / "eval" / "rls1e_arms.json"
    keep = tree_arms.read_text(encoding="utf-8")
    af_live["budget"]["note_added_after_the_read"] = 1
    tree_arms.write_text(json.dumps(af_live), encoding="utf-8", newline="\n")
    try:
        c("B10_gate_REFUSES_when_a_package_file_changed_after_the_read_only_the_fresh_re_read_can_see_it", lambda: (lambda g: g[0] is False and "stale" in g[1])(RC.id_gate(str(d))))
    finally:
        tree_arms.write_text(keep, encoding="utf-8", newline="\n")
    d = build_dir(tmp / "dforge", ref, temp=0.01)                       # the TRUE reading is ID-FAIL ...
    forge(ID, d)                                                        # ... a forger hand-writes ID-PASS with the right tool, hashes and digest
    c("B10_gate_REFUSES_a_forged_ID_PASS_with_the_right_tool_hashes_and_digest_only_the_re_read_can_see_it", lambda: (lambda g: g[0] is False and "stale or not the reader's output" in g[1])(RC.id_gate(str(d))))
    d, rr = verdict()
    forge(ID, d, arm="R0e-pick")
    c("B10_gate_REFUSES_a_verdict_stamped_for_another_arm", lambda: (lambda g: g[0] is False and "arm is" in g[1])(RC.id_gate(str(d))))
    d, rr = verdict()
    rec2 = json.loads((d / "id_verdict.json").read_text(encoding="utf-8"))
    rec2["tool"] = "somebody_else/9"
    (d / "id_verdict.json").write_text(json.dumps(rec2), encoding="utf-8")
    c("B10_gate_REFUSES_a_verdict_of_another_tool", lambda: (lambda g: g[0] is False and "tool is" in g[1])(RC.id_gate(str(d))))
    d, rr = verdict()
    rec2 = json.loads((d / "id_verdict.json").read_text(encoding="utf-8"))
    rec2["checks"]["g13_pass"] = False
    (d / "id_verdict.json").write_text(json.dumps(rec2), encoding="utf-8")
    c("B10_gate_REFUSES_an_edited_verdict_record_sha256_no_longer_recomputes", lambda: (lambda g: g[0] is False and "record_sha256" in g[1])(RC.id_gate(str(d))))
    # the CLI the chain calls
    def cli2(d, *extra):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = RC.main(["id-gate", "--dir", str(d), *[str(x) for x in extra]])
        return rc, buf.getvalue()

    def cli(d):
        return cli2(d)
    c("B10_cli_id_gate_PASS_rc0_marker", lambda: cli(d_ok)[0] == 0 and cli(d_ok)[1].startswith("ZZRLS1_ID_PASS"))
    c("B10_cli_id_gate_missing_rc1_marker", lambda: cli(tmp / "nowhere")[0] == 1 and cli(tmp / "nowhere")[1].startswith("ZZRLS1_ID_REFUSED"))

    # ---- MASTER MIND RULING 2026-10-19 on a silence of PREREG_R0E_PICK 5.1 (NOT a prereg change; the registered ID criteria are untouched): ID-PASS fails CLOSED when R0e-id carries the chain's kill marker
    #      stage/arm_R0e-id_killed, whatever the kill rule, however clean the u250 verdict.  The SAME genuine verdict d_ok is accepted with a clean stage dir and refused with the marker: the marker alone is the difference.
    def stage_with(name, **files):
        sd = tmp / f"stage_{name}"
        sd.mkdir()
        for fn, body in files.items():
            (sd / fn).write_text(body, encoding="utf-8")
        return sd
    planted_kill = '{"arm": "R0e-id", "code": "K2", "why": "planted kill after a clean u250 read", "at": "2026-10-19T00:00:00"}'
    st_clean = stage_with("clean")
    st_killed = stage_with("killed", **{"arm_R0e-id_killed": planted_kill})
    st_empty = stage_with("empty_marker", **{"arm_R0e-id_killed": ""})
    st_other = stage_with("other_arms", **{"arm_R0e-pick_killed": planted_kill, "arm_R0d_killed": planted_kill, "arm_R1o_killed": planted_kill, "train_R0e-id": "done"})
    c("MM_control_the_planted_verdict_reads_ID_PASS_and_the_gate_accepts_it_with_no_marker", lambda: rec_ok["outcome"] == "ID-PASS" and RC.id_gate(str(d_ok), str(st_clean))[0] is True)
    c("MM_gate_REFUSES_a_clean_u250_ID_PASS_when_the_R0e_id_kill_marker_exists", lambda: (lambda g: g[0] is False and "kill marker" in g[1] and "arm_R0e-id_killed" in g[1] and "K2" in g[1] and "MM ruling" in g[1])(RC.id_gate(str(d_ok), str(st_killed))))
    c("MM_gate_REFUSES_when_the_marker_file_is_empty_existence_is_the_test", lambda: (lambda g: g[0] is False and "kill marker" in g[1])(RC.id_gate(str(d_ok), str(st_empty))))
    c("MM_gate_a_kill_marker_of_ANOTHER_arm_or_a_train_marker_does_not_block_R0e_pick", lambda: RC.id_gate(str(d_ok), str(st_other))[0] is True)
    c("MM_gate_a_missing_stage_dir_is_not_a_kill", lambda: RC.id_gate(str(d_ok), str(tmp / "no_stage_dir_here"))[0] is True)
    c("MM_gate_the_marker_check_runs_even_when_the_verdict_is_missing_the_kill_is_named", lambda: (lambda g: g[0] is False and "kill marker" in g[1])(RC.id_gate(str(tmp / "nowhere" / "R0e-id"), str(st_killed))))
    Rk = tmp / "Rdefault"
    dk = build_dir(Rk, ref)
    read_and_write(ID, dk, str(ev))
    (Rk / "stage").mkdir()
    c("MM_gate_default_stage_dir_is_R_stage_next_to_R_arms_control_accepts_without_marker", lambda: RC.id_gate(str(dk))[0] is True)
    (Rk / "stage" / "arm_R0e-id_killed").write_text(planted_kill, encoding="utf-8")
    c("MM_gate_default_stage_dir_is_R_stage_next_to_R_arms_refuses_with_marker", lambda: (lambda g: g[0] is False and "kill marker" in g[1] and posix(Rk / "stage") in g[1].replace(chr(92), "/"))(RC.id_gate(str(dk))))
    c("MM_cli_id_gate_stage_dir_flag_refuses_rc1_and_names_the_marker", lambda: (lambda r: r[0] == 1 and r[1].startswith("ZZRLS1_ID_REFUSED") and "kill marker" in r[1])(cli2(d_ok, "--stage-dir", st_killed)))
    c("MM_cli_id_gate_stage_dir_flag_clean_stage_still_passes_rc0", lambda: (lambda r: r[0] == 0 and r[1].startswith("ZZRLS1_ID_PASS"))(cli2(d_ok, "--stage-dir", st_clean)))
    # the reader's own CLI: exit codes and the file it writes
    rcs = {}
    for nm, kw in {"pass": {}, "fail": dict(temp=0.01), "div": dict(dttc=0.9), "notread": dict(no_u250=True)}.items():
        d = build_dir(tmp / f"dcli_{nm}", ref, **kw)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rcs[nm] = ID.main(["--arm-dir", str(d)])
        rcs[nm + "_file"] = (d / "id_verdict.json").exists() and buf.getvalue().startswith("ZZR0EID R0e-id ")
    c("B10_reader_cli_exit_codes_0_pass_1_fail_1_diverged_2_not_read_and_the_file_is_written", lambda: (rcs["pass"], rcs["fail"], rcs["div"], rcs["notread"]) == (0, 1, 1, 2) and all(rcs[k] for k in rcs if k.endswith("_file")))
    # the reference is FROZEN from the banked R0d files: re-derive it whenever they are in this checkout
    banked = PKG / "raw" / "2026-10-18-r0d-outcome" / "raw" / "sft.jsonl"
    if banked.exists() and (banked.parent / "rlval_B_vs_ownA_u250.json").exists():
        rows = [json.loads(x) for x in banked.read_text(encoding="utf-8").splitlines() if x.strip()]
        data0 = [x for x in rows if x.get("event") == "data"][0]
        s00 = [x for x in rows if x.get("event") == "eval" and x.get("tag") == "step0"][0]
        q = json.loads((banked.parent / "rlval_B_vs_ownA_u250.json").read_text(encoding="utf-8"))["paired_vs_base"]["B"]["quantities"]
        c("B10_reference_equals_R0d_s_banked_files", lambda: ref["epo_con"] == data0["epo_con"] and ref["epo"] == data0["epo"] and ref["train_sets"] == data0["train_sets"] == 16000
          and ref["step0_pick"] == {s: s00[s]["pick"] for s in ("base", "A", "B")} and ref["u250_contrast_B_minus_ownA_x100"] == {"TTC": q["TTC"]["mean"], "PDMS": q["PDMS"]["mean"]}
          and abs(ref["step0_pick"]["base"] - 90.8428) < 1e-4 and abs(ref["u250_contrast_B_minus_ownA_x100"]["TTC"] + 1.167) < 1e-3 and abs(ref["u250_contrast_B_minus_ownA_x100"]["PDMS"] - 0.683) < 1e-3)
    else:
        print("ZZTEST_SKIP B10_reference_equals_R0d_s_banked_files (raw/2026-10-18-r0d-outcome is not in this checkout)", flush=True)
    c("B10_reference_has_R0d_s_eight_epo_con_keys_and_no_temp", lambda: sorted(ref["epo_con"]) == sorted(["mode", "comps", "cols", "eps", "lr", "max", "lam0", "ref"]))
    shutil.rmtree(tmp, ignore_errors=True)
    return bad


# ---------------------------------------------------------------------------------------------------------------- group 2: the chain's own text, RUN under bash
def chain_block(text):
    """the arm loop's pre-GPU block, verbatim: from `OBS=""` to the line before `mkdir -p "$ARMDIR" "$LG"` (the R1o S gate, R0e-id's OBS, R0e-pick's ID gate)"""
    m = re.search(r'^  OBS=""\n.*?(?=^  mkdir -p "\$ARMDIR" "\$LG"\n)', text, re.S | re.M)
    return m.group(0) if m else ""


def checks_chain(tree):
    global N
    bad = []

    def c(name, cond):
        global N
        N += 1
        try:
            ok = bool(cond() if callable(cond) else cond)
        except Exception as e:                                                                                       # noqa: BLE001
            ok = False
            name = f"{name} [raised {type(e).__name__}: {str(e)[:80]}]"
        if not ok:
            bad.append(name)
    tree = Path(tree)
    text = (tree / "refe" / "rls1_pod.sh").read_text(encoding="utf-8")
    ref = json.loads((tree / "eval" / "r0d_id_reference.json").read_text(encoding="utf-8"))
    ID = load(tree / "eval" / "r0e_id_read.py", "idr2")
    tmp = Path(tempfile.mkdtemp(prefix="b9b10c_"))
    blk = chain_block(text)
    c("B10_chain_block_found", bool(blk))
    py = posix(sys.executable)
    refe = posix(tree / "refe")

    def run(arm, R, shim=None):
        sc = tmp / f"run_{arm}_{len(list(tmp.glob('run_*')))}.sh"
        sc.write_text(f'set -u\ncd "{refe}"\nR="{posix(R)}"; S="$R/stage"; ARM="{arm}"; PY="{shim or py}"\nrefuse() {{ echo "REFUSED: $*"; exit 1; }}\n{blk}\necho "PROCEED OBS=[${{OBS:-}}]"\n', encoding="utf-8", newline="\n")
        r = subprocess.run([BASH, posix(sc)], capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k not in ("RLS1_R1O_LINEAGE", "HF_TOKEN")}, timeout=180)
        return r.returncode, r.stdout + r.stderr
    # ---- B9 on the chain's own text: OBS per arm
    shim = tmp / "shim.sh"
    shim.write_text('#!/bin/bash\ncase "$*" in *safeobj-gate*) echo "ZZRLS1_SAFEOBJ_S_PASS shim";; esac\n', encoding="utf-8", newline="\n")
    os.chmod(shim, 0o755)
    Rempty = tmp / "Rempty"
    (Rempty / "arms").mkdir(parents=True)
    obs = {}
    for arm in ("R0e-id", "R0e-pick", "R1e-pick", "R0d", "R1d", "R1o", "C0"):
        if arm == "R0e-pick":
            continue                                                  # its block needs a verdict: below
        rc, out = run(arm, Rempty, shim=posix(shim) if arm == "R1o" else None)
        m = re.search(r"PROCEED OBS=\[(.*)\]", out)
        obs[arm] = m.group(1) if (rc == 0 and m) else f"rc{rc}:{out[-120:]}"
    c("B9_chain_R0e_id_gets_OBS_observe_only", lambda: obs["R0e-id"] == "--observe-only")
    c("B9_chain_R1o_keeps_OBS_observe_only", lambda: obs["R1o"] == "--observe-only")
    c("B9_chain_R1e_pick_K3_stays_live_no_OBS", lambda: obs["R1e-pick"] == "")
    c("B9_chain_R0d_R1d_C0_no_OBS", lambda: obs["R0d"] == "" and obs["R1d"] == "" and obs["C0"] == "")
    # ---- B10 on the chain's own text: R0e-pick against the REAL gate, planted R0e-id directories
    scenarios = {}

    def scen(name, **m):
        R = tmp / f"R_{name}"
        d = build_dir(R, ref, **m) if m.get("_build", True) else None
        return R, d
    R, d = scen("missing", _build=False)
    (R / "arms").mkdir(parents=True)
    scenarios["missing_output"] = run("R0e-pick", R)
    R, d = scen("noverdict")
    scenarios["no_verdict_file"] = run("R0e-pick", R)
    R, d = scen("fail", temp=0.01)
    read_and_write(ID, d, str(tree / "eval"))
    scenarios["ID_FAIL"] = run("R0e-pick", R)
    R, d = scen("div", dttc=0.9)
    read_and_write(ID, d, str(tree / "eval"))
    scenarios["ID_DIVERGED"] = run("R0e-pick", R)
    R, d = scen("malformed")
    (d / "id_verdict.json").write_text('{"outcome": "ID-PASS"', encoding="utf-8")
    scenarios["malformed"] = run("R0e-pick", R)
    R, d = scen("stale")
    read_and_write(ID, d, str(tree / "eval"))
    (d / "sft.jsonl").write_text((d / "sft.jsonl").read_text(encoding="utf-8") + json.dumps({"event": "train"}) + "\n", encoding="utf-8")
    scenarios["stale"] = run("R0e-pick", R)
    Rx, dx = scen("anotherX", salt="X")
    read_and_write(ID, dx, str(tree / "eval"))
    Ry, dy = scen("anotherY", salt="Y")
    shutil.copyfile(dx / "id_verdict.json", dy / "id_verdict.json")
    scenarios["another_runs_verdict"] = run("R0e-pick", Ry)
    R, d = scen("pass")
    read_and_write(ID, d, str(tree / "eval"))
    (R / "stage").mkdir(parents=True, exist_ok=True)                  # the chain's own `mkdir -p "$R" "$S"`
    scenarios["ID_PASS"] = run("R0e-pick", R)
    R, d = scen("killed")
    read_and_write(ID, d, str(tree / "eval"))
    (R / "stage").mkdir(parents=True, exist_ok=True)
    (R / "stage" / "arm_R0e-id_killed").write_text('{"arm": "R0e-id", "code": "K2", "why": "planted kill after a clean u250 read"}\n', encoding="utf-8")
    scenarios["killed_after_clean_u250"] = run("R0e-pick", R)
    c("MM_chain_control_the_killed_scenario_s_verdict_is_a_genuine_ID_PASS_the_marker_is_the_only_difference", lambda: json.loads((d / "id_verdict.json").read_text(encoding="utf-8"))["outcome"] == "ID-PASS")
    for nm in ("missing_output", "no_verdict_file", "ID_FAIL", "ID_DIVERGED", "malformed", "stale", "another_runs_verdict", "killed_after_clean_u250"):
        c(f"B10_chain_R0e_pick_REFUSES_{nm}_before_any_directory_preflight_or_GPU_work", lambda nm=nm: scenarios[nm][0] == 1 and "REFUSED: R0e-pick runs only after R0e-id reads ID-PASS" in scenarios[nm][1] and "PROCEED" not in scenarios[nm][1])
    c("MM_chain_R0e_pick_REFUSES_a_clean_u250_ID_PASS_when_stage_arm_R0e_id_killed_exists_and_names_the_marker", lambda: scenarios["killed_after_clean_u250"][0] == 1 and "kill marker" in scenarios["killed_after_clean_u250"][1] and "arm_R0e-id_killed" in scenarios["killed_after_clean_u250"][1])
    c("MM_chain_passes_the_stage_dir_to_the_gate", lambda: 'id-gate --dir "$R/arms/R0e-id" --stage-dir "$S" >' in text)
    c("B10_chain_R0e_pick_PROCEEDS_on_a_genuine_ID_PASS_with_OBS_empty_K3_live", lambda: scenarios["ID_PASS"][0] == 0 and "PROCEED OBS=[]" in scenarios["ID_PASS"][1])
    # ---- the gate is wired where it must be: before any directory / preflight / GPU work, and the reader + its reference are required at the precheck
    gi = text.find("rls1_checks.py id-gate --dir")
    c("B10_chain_gate_precedes_mkdir_preflight_and_the_trainer", lambda: 0 < gi < text.find('mkdir -p "$ARMDIR" "$LG"') and gi < text.find("scorer_finetune.py $ARGS --preflight") and gi < text.find("scorer_finetune.py $ARGS >>"))
    c("B10_chain_gate_reads_R0e_id_s_directory_and_only_for_R0e_pick", lambda: text.count('rls1_checks.py id-gate --dir "$R/arms/R0e-id"') == 1 and text.count("id-gate") >= 1
      and re.search(r'if \[ "\$ARM" = R0e-pick \]; then\n    env -u HF_TOKEN "\$PY" rls1_checks\.py id-gate --dir "\$R/arms/R0e-id"', text) is not None)
    c("B10_chain_the_precheck_requires_the_reader_and_its_reference", lambda: "eval/r0e_id_read.py eval/r0d_id_reference.json" in text)
    c("B10_chain_R0e_id_writes_the_u250_read_the_reader_needs_and_refuses_without_it", lambda: 'rlval_id_u250.json" >> "$ARMDIR/rlval.log"' in text and "wrote no rlval_id_u250.json for R0e-id" in text and "scores_update250_A.npz B=$SC/scores_update250_B.npz --base A" in text)
    rc = subprocess.run([BASH, "-n", posix(tree / "refe" / "rls1_pod.sh")], capture_output=True)
    c("CHAIN_PARSES", rc.returncode == 0)
    shutil.rmtree(tmp, ignore_errors=True)
    return bad


# ---------------------------------------------------------------------------------------------------------------- mutations
def mutations():
    """(name, group, (relpath, old, new), a check-name fragment that MUST be among the RED ones)"""
    RC, POD, IDR = "refe/rls1_checks.py", "refe/rls1_pod.sh", "eval/r0e_id_read.py"
    gate_call = '    grep -q "^ZZRLS1_ID_PASS" "$R/id_gate_R0e-pick.log" || refuse "R0e-pick runs only after R0e-id reads ID-PASS (PREREG_R0E_PICK 5.1): $(tail -n 1 "$R/id_gate_R0e-pick.log" | cut -c1-300)"\n'
    return [
        # ---- B9
        ("B9 R0e-id is removed from the allowed set", "py", (RC, 'OBSERVE_ONLY_ARMS = ("R1o", "R0e-id")', 'OBSERVE_ONLY_ARMS = ("R1o",)'), "B9_"),
        ("B9 R0e-pick is made observe-only (allowed set)", "py", (RC, 'OBSERVE_ONLY_ARMS = ("R1o", "R0e-id")', 'OBSERVE_ONLY_ARMS = ("R1o", "R0e-id", "R0e-pick")'), "B9_"),
        ("B9 R1e-pick is made observe-only (allowed set)", "py", (RC, 'OBSERVE_ONLY_ARMS = ("R1o", "R0e-id")', 'OBSERVE_ONLY_ARMS = ("R1o", "R0e-id", "R1e-pick")'), "B9_"),
        ("B9 R1o is dropped (R1o unchanged must be tested)", "py", (RC, 'OBSERVE_ONLY_ARMS = ("R1o", "R0e-id")', 'OBSERVE_ONLY_ARMS = ("R0e-id",)'), "B9_R1o"),
        ("B9 the judge ignores observe_only (a K3 kills R0e-id)", "py", (RC, "if k3_enabled and not observe_only and prev is not None", "if k3_enabled and prev is not None"), "B9_PLANTED_K3_R0e_id"),
        ("B9 the monitor does not log the observed K3", "py", (RC, '                    say("ZZRLS1_K3_OBSERVED", arm, r["tag"], would_why)\n', "                    pass\n"), "B9_PLANTED_K3_R0e_id"),
        ("B9 the chain does not set OBS for R0e-id", "chain", (POD, '  [ "$ARM" = R0e-id ] && OBS="--observe-only"\n', ""), "B9_chain_R0e_id"),
        ("B9 the chain sets OBS for R0e-pick too", "chain", (POD, '  [ "$ARM" = R0e-id ] && OBS="--observe-only"\n', '  [ "$ARM" = R0e-id ] || [ "$ARM" = R0e-pick ] && OBS="--observe-only"\n'), "B10_chain_R0e_pick_PROCEEDS"),
        ("B9 the chain sets OBS for R1e-pick too", "chain", (POD, '  [ "$ARM" = R0e-id ] && OBS="--observe-only"\n', '  [ "$ARM" = R0e-id ] || [ "$ARM" = R1e-pick ] && OBS="--observe-only"\n'), "B9_chain_R1e_pick"),
        # ---- B10: the chain
        ("B10 the gate call is deleted from the chain", "chain", (POD, gate_call, ""), "B10_chain_R0e_pick_REFUSES"),
        ("B10 the chain never runs the id-gate (a refusal for everyone, a pass for no one)", "chain", (POD, '    env -u HF_TOKEN "$PY" rls1_checks.py id-gate --dir "$R/arms/R0e-id" --stage-dir "$S" > "$R/id_gate_R0e-pick.log" 2>&1 200>&-\n', ""), "B10_chain_R0e_pick_PROCEEDS"),
        ("B10 the chain's gate cannot refuse", "chain", (POD, '|| refuse "R0e-pick runs only after R0e-id reads ID-PASS', '|| echo "R0e-pick runs only after R0e-id reads ID-PASS'), "B10_chain_R0e_pick_REFUSES"),
        ("B10 the chain's gate guards the wrong arm", "chain", (POD, 'if [ "$ARM" = R0e-pick ]; then\n    env -u HF_TOKEN "$PY" rls1_checks.py id-gate', 'if [ "$ARM" = R1e-pick ]; then\n    env -u HF_TOKEN "$PY" rls1_checks.py id-gate'), "B10_chain_R0e_pick_REFUSES"),
        ("B10 the chain's gate reads the wrong directory", "chain", (POD, 'id-gate --dir "$R/arms/R0e-id"', 'id-gate --dir "$R/arms/R0d"'), "B10_chain_R0e_pick_"),
        ("B10 the precheck forgets the reader", "chain", (POD, "eval/r0e_id_read.py eval/r0d_id_reference.json", "eval/r0d_id_reference.json"), "B10_chain_the_precheck"),
        ("B10 the precheck forgets the reference", "chain", (POD, "eval/r0e_id_read.py eval/r0d_id_reference.json", "eval/r0e_id_read.py"), "B10_chain_the_precheck"),
        ("B10 R0e-id does not write the u250 read", "chain", (POD, '--out "$ARMDIR/rlval_id_u250.json" >> "$ARMDIR/rlval.log"', '--out "$ARMDIR/rlval_u250_other.json" >> "$ARMDIR/rlval.log"'), "B10_chain_R0e_id_writes"),
        # ---- B10: the gate
        ("B10 the gate accepts any outcome", "py", (RC, 'if out != "ID-PASS":', "if False:"), "B10_gate_REFUSES_ID_"),
        ("B10 the gate skips the genuineness check", "py", (RC, "    why = id_not_genuine(d, rec)\n    if why:", '    why = ""\n    if why:'), "B10_gate_REFUSES_a_forged"),
        ("B10 the genuineness check skips the fresh re-read", "py", (RC, "        if not why:\n            diff = ID.differing_fields(rec, ID.read_arm(d))", "        if False:\n            diff = ID.differing_fields(rec, ID.read_arm(d))"), "B10_gate_REFUSES_a_forged"),
        ("B10 the genuineness check skips the sha256 of the inputs", "py", (RC, "            elif ID.sha256_file(fp) != want:", "            elif False:"), "B10_gate_REFUSES_"),
        ("B10 the genuineness check skips the record digest", "py", (RC, 'if rec.get("record_sha256") != ID.record_digest(rec):', "if False:"), "B10_gate_REFUSES_an_edited"),
        ("B10 the genuineness check skips the arm", "py", (RC, 'if rec.get("arm") != ID.ARM:', "if False:"), "B10_gate_REFUSES_a_verdict_stamped"),
        ("B10 the genuineness check skips the tool", "py", (RC, 'if rec.get("tool") != ID.TOOL:', "if False:"), "B10_gate_REFUSES_a_verdict_of_another_tool"),
        ("B10 a missing verdict file is not a refusal", "py", (RC, "    except (OSError, ValueError, AttributeError) as e:\n        return False, f\"{p}: {type(e).__name__} (no readable R0e-id verdict", "    except (ValueError, AttributeError) as e:\n        return False, f\"{p}: {type(e).__name__} (no readable R0e-id verdict"), "B10_gate_REFUSES_a_missing"),
        ("B10 the gate's CLI always passes", "py", (RC, 'say("ZZRLS1_ID_PASS" if ok else "ZZRLS1_ID_REFUSED", why)\n        return 0 if ok else 1', 'say("ZZRLS1_ID_PASS" if ok else "ZZRLS1_ID_REFUSED", why)\n        return 0'), "B10_cli_id_gate_missing"),
        # ---- MM ruling 2026-10-19: the kill marker
        ("MM the gate ignores the kill marker (the marker check is deleted)", "py", (RC, '    km = id_kill_marker(d, stage_dir)\n    if km:\n        try:', '    km = ""\n    if km:\n        try:'), "MM_gate_REFUSES_a_clean_u250_ID_PASS_when_the_R0e_id_kill_marker_exists"),
        ("MM the gate ignores the kill marker (the marker check is deleted), seen through the chain run under bash", "chain", (RC, '    km = id_kill_marker(d, stage_dir)\n    if km:\n        try:', '    km = ""\n    if km:\n        try:'), "MM_chain_R0e_pick_REFUSES_a_clean_u250_ID_PASS_when_stage_arm_R0e_id_killed"),
        ("MM the gate looks for another arm's marker", "py", (RC, 'm = os.path.join(sd, "arm_R0e-id_killed")', 'm = os.path.join(sd, "arm_R0e-pick_killed")'), "MM_gate_REFUSES_a_clean_u250_ID_PASS_when_the_R0e_id_kill_marker_exists"),
        ("MM the gate refuses on ANY arm's kill marker (over-refusal)", "py", (RC, '    m = os.path.join(sd, "arm_R0e-id_killed")\n    return m if os.path.lexists(m) else ""', '    import glob\n    g = glob.glob(os.path.join(sd, "arm_*_killed"))\n    return g[0] if g else ""'), "MM_gate_a_kill_marker_of_ANOTHER_arm"),
        ("MM the gate's default stage dir is wrong (the arms dir's own parent instead of R)", "py", (RC, 'sd = stage_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(d))), "stage")', 'sd = stage_dir or os.path.join(os.path.dirname(os.path.abspath(d)), "stage")'), "MM_gate_default_stage_dir"),
        ("MM the gate's CLI ignores --stage-dir", "py", (RC, "        ok, why = id_gate(a.dir, a.stage_dir)\n", "        ok, why = id_gate(a.dir)\n"), "MM_cli_id_gate_stage_dir_flag_refuses"),
        ("MM the chain hands the gate the wrong stage dir", "chain", (POD, '--stage-dir "$S" >', '--stage-dir "$R/stagex" >'), "MM_chain_R0e_pick_REFUSES_a_clean_u250_ID_PASS_when_stage_arm_R0e_id_killed"),
        # ---- B10: the reader
        ("B10 reader: the epo_con clause is deleted", "py", (IDR, 'chk["epo_con_block_is_R0d_plus_temp_0.1"] = _canon(d.get("epo_con")) == _canon({**ref["epo_con"], "temp": REGISTERED_TEMP})', 'chk["epo_con_block_is_R0d_plus_temp_0.1"] = True'), "B10_reader_ID_FAIL_alone_epo_con"),
        ("B10 reader: the epo clause is deleted", "py", (IDR, 'chk["epo_block_is_R0d"] = _canon(d.get("epo")) == _canon(ref["epo"])', 'chk["epo_block_is_R0d"] = True'), "B10_reader_ID_FAIL_alone_epo_block"),
        ("B10 reader: the train_sets clause is deleted", "py", (IDR, 'chk["train_sets_is_16000"] = _num(ts) and ts == TRAIN_SETS', 'chk["train_sets_is_16000"] = True'), "B10_reader_ID_FAIL_alone_train_sets"),
        ("B10 reader: the arms-file self-check is deleted", "py", (IDR, "    ok, why = arms_file_self_check(eval_dir)\n", "    ok, why = True, 'ok'\n"), "B10_reader_ID_FAIL_alone_arms_file"),
        ("B10 reader: G13 is not judged", "py", (IDR, 'g13_ok = bool(pf.get("verdict") == "PASS" and g13[0].get("ok") is True and not g13[0].get("mutate"))', "g13_ok = True"), "B10_reader_ID_FAIL_alone_G13"),
        ("B10 reader: G13 under a regression switch counts", "py", (IDR, ' and not g13[0].get("mutate"))', ")"), "B10_reader_ID_FAIL_alone_G13_under"),
        ("B10 reader: the step0 tolerance is 1.0 point", "py", (IDR, "PICK_TOL = 0.10 ", "PICK_TOL = 1.00 "), "B10_reader_boundary_step0"),
        ("B10 reader: the step0 tolerance is 0.0", "py", (IDR, "PICK_TOL = 0.10 ", "PICK_TOL = 0.00 "), "B10_reader_boundary_step0_exactly"),
        ("B10 reader: the divergence tolerance is 5 points", "py", (IDR, "DIV_TOL = 0.50 ", "DIV_TOL = 5.00 "), "B10_reader_ID_DIVERGED"),
        ("B10 reader: the divergence read is deleted", "py", (IDR, "diverged = abs(ttc - rc[\"TTC\"]) > DIV_TOL + EPS or abs(pdms - rc[\"PDMS\"]) > DIV_TOL + EPS", "diverged = False"), "B10_reader_ID_DIVERGED"),
        ("B10 reader: only TTC is compared for divergence", "py", (IDR, " or abs(pdms - rc[\"PDMS\"]) > DIV_TOL + EPS", ""), "B10_reader_ID_DIVERGED_PDMS"),
        ("B10 reader: run identity (dump paths) is not checked", "py", (IDR, "        if foreign or not dumps:", "        if False:"), "B10_reader_NOT_READ_another_runs_log"),
        ("B10 reader: the u250 read is not tied to this run's dumps", "py", (IDR, "                    elif sha256_file(f) != want:", "                    elif False:"), "B10_reader_NOT_READ_u250_read_from_other_dumps"),
        ("B10 reader: a problem does not stop a PASS (NOT_READ -> pass)", "py", (IDR, "    elif probs:\n        outcome = \"NOT_READ\"", "    elif False:\n        outcome = \"NOT_READ\""), "B10_reader_NOT_READ"),
        ("B10 reader: ID-DIVERGED is not ranked after ID-FAIL", "py", (IDR, "    if fails:\n        outcome = \"ID-FAIL\"", "    if diverged:\n        outcome = \"ID-DIVERGED\"\n    elif fails:\n        outcome = \"ID-FAIL\""), "B10_ID_FAIL_wins_over_ID_DIVERGED"),
        ("B10 reader: the step0 comparison drops arm B", "py", (IDR, 'for side in ("base", "A", "B"):\n            try:\n                x, y', 'for side in ("base", "A"):\n            try:\n                x, y'), "B10_reader_ID_FAIL_alone_step0_B"),
    ]


def main():
    global N
    for rel in TREE_FILES:
        assert (PKG / rel).exists(), rel
    with tempfile.TemporaryDirectory(prefix="b9b10_tree_") as t0:
        copy_tree(t0)
        v = checks_py(t0) + checks_chain(t0)
        assert v == [], v
        n0 = N
        print(f"  unmutated: {n0} checks, all green", flush=True)
        muts = mutations()
        for name, group, mut, frag in muts:
            with tempfile.TemporaryDirectory(prefix="b9b10_mut_") as t1:
                copy_tree(t1, mut)
                got = checks_py(t1) if group == "py" else checks_chain(t1)
            hit = [g for g in got if frag in g]
            assert got and hit, f"mutation {name!r} left {frag!r} green (RED checks: {got[:4]})"
            print(f"  RED  {name}  ->  {len(got)} check(s), e.g. {hit[0][:110]}", flush=True)
    print(f"ZZTEST_PASS {N}  ({n0} checks on the unmutated tree + {len(muts)} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
