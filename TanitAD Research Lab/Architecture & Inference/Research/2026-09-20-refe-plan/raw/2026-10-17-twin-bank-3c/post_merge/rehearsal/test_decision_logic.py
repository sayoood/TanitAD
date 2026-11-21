#!/usr/bin/env python3
"""Offline test of the DECISION LOGIC of post_merge_to_p6gate.sh on SYNTHETIC chain-log text (no pod, no ssh, no GPU).   python test_decision_logic.py  ->  ZZDECISION_PASS <n>

Covers the three pure judges in pod/pod_helpers.py that decide whether anything is launched:
  judge_chain_text  the dry run's chain.log slice: is there EXACTLY ONE `launch gate PASS (token verified)` line, in order, with no REFUSED / FALLBACK / FAIL / trainer line?
  busy_decision     is the pod free for P6 (no GPU compute process, no other chain lock, no rls1_pod.sh = R0d)?
  judge_launch      after the REAL launch: refused | trainer_exited | preflight_ok | training (first step line) | ambiguous | gone
and a MUTATION arm: each guard is removed from a copy of the source, and the suite must go RED for every mutant (a guard no test can see is not a guard).
The synthetic lines copy the real chain.log's format (refe/p6_pod.sh `log()`: `<ISO time without Z> <message>`; egress ticks carry a Z).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_PATH = HERE.parent / "pod" / "pod_helpers.py"

PASS_SLICE = """\
2026-10-18T02:00:01 chain start (p6_pod.sh md5 0123456789ab)
2026-10-18T02:00:20 dd write test 16 GB ok
2026-10-18T02:00:21 cores 11 (cgroup v1 cfs): label workers 7, cache workers 11
2026-10-18T02:00:21 WARNING: 7 label workers (< 12): G-LIVE-LAB's 6.735 sets/s bar is unlikely to hold; the gate decides
2026-10-18T02:00:30 env ok
2026-10-18T02:00:31 weights ok
2026-10-18T02:00:31 recover: ZZRECOVER_OK labels=no ckpt=no
2026-10-18T02:00:32 inputs ok (bank, 1192 DBs, 32 pixel shards, maps: reused in place where present)
2026-10-18T02:00:32 commands ok: ZZCMD_TABLE_OK rank0=103037 matched=103037 missing=0 logs_missing=0
2026-10-18T02:00:33 metric cache ok: ZZMETRIC_CACHE_DONE built=103037 skipped_5s_guard=0 failed=0 logs=1191 60s
2026-10-18T02:03:40 D7 rank 1: ZZD7_DISTINCT_OK k=1 sampled=40 differ=40 frac=1.0 min_frac=0.9
2026-10-18T02:04:10 D7 rank 2: ZZD7_DISTINCT_OK k=2 sampled=40 differ=40 frac=1.0 min_frac=0.9
2026-10-18T02:04:40 D7 rank 3: ZZD7_DISTINCT_OK k=3 sampled=40 differ=40 frac=1.0 min_frac=0.9
2026-10-18T02:04:40 lanes ok
2026-10-18T02:04:40 ARGS sha 0f0f0f0f0f0f0f0f (--yaw-wrap off, --label-workers 7)
2026-10-18T02:08:41 lab: ZZLAB_THROUGHPUT 10.323 sets/s at 7 workers (the gate judges it against 6.735)
2026-10-18T02:08:41 lab ok
2026-10-18T02:08:52 ZZP6_WAIT_OK (no other chain holds the pod, GPU idle): the GPU stages start
2026-10-18T02:13:30 mem ok
2026-10-18T02:13:35Z ZZEGRESS_OK G-EGR 1 files 125 B
2026-10-18T02:13:36Z ZZEGRESS_FAIL G-EGR-REGRESSION exception:HfHubHTTPError:(Request ID: Root=1-abc) 403 Forbidden: You don't have the rights to create a model under the namespace "google".
2026-10-18T02:13:40Z ZZEGRESS_OK G-EGR-CKPT 1 files 228324301 B
2026-10-18T02:13:40Z ZZEGR_PASS main_ok=True regression_refused=True
2026-10-18T02:13:40 egr ok
2026-10-18T02:50:00 launch gate PASS (token verified)
2026-10-18T02:50:00 stopping after launch_gate (P6_STOP_AFTER)
"""
OLD_REFUSED = "2026-10-16T18:50:29 REFUSED: the launch gate did not mint a PASS token that verifies for this argv (launch_gate.log, lg/, token_verify.log)\n"
STEP_LINE = '{"step": 0, "epoch": 0, "pos": 200, "traj_L1": 12.487878799438477, "score": 0.05731813982129097, "winners": 3, "batch": 4}'
PRE = "2026-10-18T03:02:11 train.py preflight OK; launching\n"


def load(src_text: str):
    m = types.ModuleType("pod_helpers_under_test")
    m.__file__ = str(SRC_PATH)
    exec(compile(src_text, str(SRC_PATH), "exec"), m.__dict__)
    return m


def suite(H) -> int:
    n = 0

    def ok(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1

    def chain(text, ranks=(1, 2, 3)):
        return H.judge_chain_text(text, list(ranks))

    # ------------------------------------------------------------------ judge_chain_text: the gate line parsing
    r, why = chain(PASS_SLICE); ok(r, f"T1 a clean PASS slice must be ok: {why}")
    ok(chain(PASS_SLICE, (1,))[0], "T1b fewer ranks than the slice carries is still ok")
    r, why = chain(PASS_SLICE.replace("2026-10-18T02:50:00 launch gate PASS (token verified)\n", "")); ok(not r and "PASS" in why, f"T2 missing PASS line: {why}")
    r, why = chain(PASS_SLICE.replace("2026-10-18T02:50:00 launch gate PASS (token verified)\n",
                                      "2026-10-18T02:50:00 REFUSED: the launch gate did not mint a PASS token that verifies for this argv (launch_gate.log, lg/, token_verify.log)\n"))
    ok(not r and "REFUSED" in why, f"T3 a REFUSED line instead of PASS: {why}")
    r, why = chain(PASS_SLICE + "2026-10-18T02:50:01 REFUSED: something afterwards\n"); ok(not r and "REFUSED" in why, f"T4 PASS and then REFUSED (contradiction) must not be ok: {why}")
    r, why = chain(PASS_SLICE.replace("2026-10-18T02:04:10 D7 rank 2: ZZD7_DISTINCT_OK", "2026-10-18T02:04:10 D7 FALLBACK (P6_D7_FALLBACK=1): PDM-Closed did not honour the route patch for rank 2\n2026-10-18T02:04:11 D7 rank 2: ZZD7_DISTINCT_OK"))
    ok(not r and "FALLBACK" in why, f"T5 the D7 fallback must not be ok: {why}")
    r, why = chain(PASS_SLICE + "2026-10-18T02:50:01 train.py preflight OK; launching\n"); ok(not r and "trainer" in why, f"T6 a trainer-side line in the dry-run slice: {why}")
    r, why = chain(PASS_SLICE + "2026-10-18T02:50:02 attempt 1 exited rc 0\n"); ok(not r, f"T6b 'attempt N exited' in the slice: {why}")
    r, why = chain(PASS_SLICE.replace("2026-10-18T02:50:00 stopping", "2026-10-18T02:50:00 launch gate PASS (token verified)\n2026-10-18T02:50:00 stopping"))
    ok(not r and "exactly_one" in why, f"T7 a duplicated PASS line is ambiguous: {why}")
    r, why = chain(PASS_SLICE.replace("2026-10-18T02:50:00 launch gate PASS (token verified)\n", "2026-10-18T02:50:00 note: the sentence 'launch gate PASS (token verified)' is quoted inside another line\n"))
    ok(not r, f"T8 a marker quoted inside a longer sentence is not a PASS line: {why}")
    r, why = chain(PASS_SLICE.replace("\n", "\r\n")); ok(r, f"T9 CRLF line endings must not break the parse: {why}")
    s = PASS_SLICE.replace("2026-10-18T02:04:40 lanes ok\n", "").replace("2026-10-18T02:50:00 launch gate PASS (token verified)\n", "2026-10-18T02:50:00 launch gate PASS (token verified)\n2026-10-18T02:50:00 lanes ok\n")
    r, why = chain(s); ok(not r and "order" in why, f"T10 'lanes ok' after the PASS line is out of order: {why}")
    r, why = chain(PASS_SLICE.replace("2026-10-18T02:04:10 D7 rank 2: ZZD7_DISTINCT_OK k=2 sampled=40 differ=40 frac=1.0 min_frac=0.9\n", "")); ok(not r and "rank_2" in why, f"T11 a rank without its D7 line: {why}")
    ok(chain(PASS_SLICE.replace("2026-10-18T02:04:10 D7 rank 2: ZZD7_DISTINCT_OK k=2 sampled=40 differ=40 frac=1.0 min_frac=0.9\n", ""), (1, 3))[0], "T11b ranks the bank does not carry are not required")
    r, why = chain(PASS_SLICE.replace("ZZEGRESS_OK G-EGR 1 files 125 B", "ZZEGRESS_FAIL G-EGR exception:PermissionError:403 Forbidden: token hf_*** cannot write Sayood/x"))
    ok(not r and "FAIL" in why, f"T12 a main-arm ZZ...FAIL token blocks the launch: {why}")
    ok(chain(PASS_SLICE)[0], "T12b the EXPECTED ZZEGRESS_FAIL G-EGR-REGRESSION line (the must-fail arm) does NOT block (it is in every good slice)")
    r, why = chain(PASS_SLICE.replace("2026-10-18T02:50:00 stopping after launch_gate (P6_STOP_AFTER)\n", "")); ok(not r and "stopping" in why, f"T13 no stop line: {why}")
    r, why = chain(PASS_SLICE + "ZZP6_REFUSED another p6_pod.sh holds /x/chain.lock\n"); ok(not r, f"T15 ZZP6_REFUSED: {why}")
    r, why = chain(""); ok(not r, "T16 an empty slice is not a PASS")
    # the slice offset: a log that still carries the PREVIOUS day's REFUSED line is judged by what came AFTER the offset
    old = OLD_REFUSED * 2
    full = old + PASS_SLICE
    off = len(old.splitlines())
    ok(chain("\n".join(full.splitlines()[off:]))[0], "T17 the old REFUSED lines before the offset are not part of the slice")
    ok(not chain("\n".join(full.splitlines()[0:]))[0], "T17b WITHOUT the offset the same log is (rightly) refused: the offset is what makes the judgement about THIS run")

    # ------------------------------------------------------------------ busy_decision: the R0d rule
    d = H.busy_decision("idle", 0, [], []); ok(d["state"] == "idle" and d["p6_held"] == 0, "B1 idle")
    d = H.busy_decision("busy", 1, [], []); ok(d["state"] == "busy" and "gpu" in d["why"], "B2 a GPU compute process => busy")
    d = H.busy_decision("unknown", -1, [], []); ok(d["state"] == "busy", "B3 nvidia-smi that cannot answer => busy (p6_pod.sh gpu_busy)")
    d = H.busy_decision("idle", 0, [], [4242]); ok(d["state"] == "busy" and "rls1_pod.sh" in d["why"], "B4 rls1_pod.sh alive (R0d) with an idle GPU => busy")
    d = H.busy_decision("idle", 0, ["refe_p2rep"], []); ok(d["state"] == "busy" and "chain_lock" in d["why"], "B5 another chain's lock => busy")
    d = H.busy_decision("idle", 0, ["refe_p6"], []); ok(d["state"] == "idle" and d["p6_held"] == 1, "B6 only P6's own lock: not 'busy' (the orchestrator FAILs on p6_held)")
    d = H.busy_decision("idle", 0, ["rls1(unopenable:13)"], []); ok(d["state"] == "busy", "B7 a lock that cannot even be opened counts as held")

    # ------------------------------------------------------------------ _first_step_line / judge_launch: did TRAINING start?
    ok(H._first_step_line([STEP_LINE]) == {"step": 0, "key": "traj_L1", "value": 12.487878799438477}, "S1 the real metrics.jsonl step line")
    ok(H._first_step_line(['{"event": "start", "at": "x"}', '{"event": "ckpt", "step": 4, "epoch": 0, "pos": 800, "bytes": 228324301}']) is None, "S2 event lines (start, ckpt) are not steps")
    ok(H._first_step_line(['{"event": "eval", "step": 4, "traj_L1": 9.1}']) is None, "S2b an EVENT line that happens to carry a step and a loss-like field is still not a training step")
    ok(H._first_step_line(['{"step": 1, "traj_L1": NaN}', '{"step": 1, "traj_L1": Infinity}']) is None, "S3 a non-finite loss is not a first step")
    ok(H._first_step_line(['not json', '{"step": true, "traj_L1": 1.0}', '{"step": 3}']) is None, "S4 garbage, a bool step, no loss field")
    L = lambda chain_text, metrics, alive=True: H.judge_launch(chain_text, metrics, alive)       # noqa: E731
    ok(L("", [])["state"] == "starting", "L1 nothing yet, chain alive")
    ok(L("2026-10-18T03:00:05 dd write test 16 GB ok\n", [])["state"] == "starting", "L1b stages running")
    ok(L(PRE, [])["state"] == "preflight_ok", "L2 preflight OK, no step yet")
    r = L(PRE, [STEP_LINE]); ok(r["state"] == "training" and r["step"] == 0 and r["key"] == "traj_L1", f"L3 preflight + first step line => training: {r}")
    ok(L("", [STEP_LINE])["state"] == "ambiguous", "L4 a step line WITHOUT the chain's preflight line is ambiguous (not 'training')")
    r = L("2026-10-18T03:01:00 REFUSED: the launch-gate token does not verify for this argv (attempt 1): train.py is NOT started\n", []); ok(r["state"] == "refused", f"L5 {r}")
    r = L(PRE + "2026-10-18T03:04:00 attempt 1 exited rc 1\n", []); ok(r["state"] == "trainer_exited", f"L6 {r}")
    ok(L(PRE + "2026-10-18T03:04:00 attempt 1 exited rc 1\n", [STEP_LINE])["state"] == "training", "L7 a step line that did arrive wins over a later exit line")
    ok(L("", [], alive=False)["state"] == "gone", "L8 chain not running, no preflight, no refusal => gone (never 'starting' forever)")
    ok(L(PRE, ['{"event": "ckpt", "step": 4}'])["state"] == "preflight_ok", "L9 a ckpt event line is not a step")
    ok(L("2026-10-18T03:30:00 ZZP6_FAILED (no done=true after the attempts)\n", [])["state"] == "failed", "L11 ZZP6_FAILED")
    ok(L("2026-10-18T03:00:00 ZZP6_REFUSED another p6_pod.sh holds /x/chain.lock\n", [])["state"] == "refused", "L12 ZZP6_REFUSED (a second chain)")
    return n


# ---------------------------------------------------------------------------------------------------------------------------------- mutation arm
MUTANTS = [
    ("REFUSED guard removed", '    if starts("REFUSED:"):\n        return False,', '    if False:\n        return False,'),
    ("FALLBACK guard removed", '    if starts("D7 FALLBACK"):\n        return False,', '    if False:\n        return False,'),
    ("trainer-side guard removed", '    if starts("train.py preflight OK") or starts("attempt ") or has("ZZP6_TRAIN_DONE") or starts("ZZP6_FAILED"):\n        return False,',
     '    if False:\n        return False,'),
    ("PASS count != 1 relaxed to >= 1", "if pass_ct != 1:", "if pass_ct < 1:"),
    ("PASS regex loosened to a substring", 'if re.fullmatch(r"(?:\\d{4}-\\d\\d-\\d\\dT[\\d:]+Z?\\s+)?launch gate PASS \\(token verified\\)", ln))',
     'if "launch gate PASS (token verified)" in ln)'),
    ("ordering check removed", "if not (idx[\"lanes ok\"] < idx[\"launch gate PASS (token verified)\"] < idx[\"stopping after launch_gate\"]):", "if False:"),
    ("per-rank D7 line not required", '        if not starts(f"D7 rank {k}:"):', "        if False:"),
    ("FAIL-token guard removed (judge_chain_text)", "    if any(re.search(r\"ZZ[A-Z0-9_]*FAIL\", ln) and \"ZZEGRESS_FAIL G-EGR-REGRESSION\" not in ln for ln in lines):\n        return False,",
     "    if False:\n        return False,"),
    ("gpu unknown no longer busy", 'if gpu_state != "idle":', 'if gpu_state == "busy":'),
    ("rls1 (R0d) ignored", "    if rls1_pids:\n", "    if False:\n"),
    ("other chain lock ignored", "    if others:\n", "    if False:\n"),
    ("step line without preflight accepted as training", "    if step and not pre:\n", "    if False:\n"),
    ("event lines count as steps", '"event" in r or not isinstance(r.get("step"), int)', 'not isinstance(r.get("step"), int)'),
    ("non-finite loss accepted", "v == v and abs(v) != float(\"inf\")", "True"),
    ("a refusal after launch ignored", '    ref = first(r"REFUSED:") or next((ln for ln in lines if "ZZP6_REFUSED" in ln), None)\n    if ref:', "    ref = None\n    if ref:"),
]


def main() -> int:
    src = SRC_PATH.read_text(encoding="utf-8")
    n = suite(load(src))
    killed = 0
    for name, old, new in MUTANTS:
        assert src.count(old) == 1, f"mutant '{name}': the guard text is not found exactly once in pod_helpers.py (the mutation arm is blind): {old[:70]!r}"
        try:
            suite(load(src.replace(old, new)))
        except AssertionError:
            killed += 1
            continue
        raise SystemExit(f"ZZDECISION_FAIL mutant SURVIVED: {name}")
    print(f"ZZDECISION_PASS {n} checks, {killed}/{len(MUTANTS)} mutants killed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
