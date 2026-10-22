#!/usr/bin/env python3
"""P7-WIRE-INTEGRATE: the two landed halves of P7's wiring, connected and proved from the CALLER.

The HEAD half (`launch_gate.py` profile `refe_p7`, `gate_checks.check_reserve`) was tested only against FAKES of `session_reserve.audit_inputs`; the RESERVE half (`session_reserve.py`, the chain's block 5a)
was tested only against ITSELF.  "Built, tested, and unreachable from its caller" (CLAUDE.md) is exactly that shape, and it was real here: the head handed the audit the PARSED JSON DOCUMENT and the real module
takes the SESSION SET, so `G-RESERVE` FAILED ON A CLEAN TREE (MEASURED before the fix, `raw/probes/probe_before.out`: "'closure' is not a session name").  Every section below runs the REAL module on a small
synthetic P7 data tree built with the reserve suite's own fixtures (`test_session_reserve.make_inputs`) and the REAL reserve file (`fixtures/p7_reserve_sessions.json`, sha256 pinned).

  I    G-RESERVE vs the real module: a clean tree PASSes; a planted leaked rank-1 twin, a planted D9 cache key (and a rank-0 row, a label-store shard + stale manifest, an image zip, a calibration row, a logless
       key, an empty input, a changed reserve file, an input the argv does not name) each FAIL for THEIR reason; the OLD adapter (parsed JSON) is replayed and FAILs on the clean tree
  II   the same through the gate machinery (`launch_gate.run_checks(only=("G-RESERVE",))`, evidence written)
  III  the pins that refuse today (`P7_AUX_STORE_PINS`, `P7_BANK_TUPLES`): the gate REFUSES until they are set (nothing / half / malformed), and PASSES on a synthetic tree once they are set to the values
       `p7_pins.py` READS off that tree's chain outputs (the reserve stage is replayed with the module's own functions: filter bank, filter calibration, move cache + images, split + re-finalise the store)
  V    the bank's CONTENT pin (P7_TARGETS_SHA256): a SAME-COUNT, DIFFERENT-BYTES bank is refused by G-CLOCK (the last gate before the PASS token), the trainer's own identity digest is the registered one, and
       `p7_pins.py --check-run` compares the trainer's RECORDED identity with the pins after the first start
  IV   design pins: refe_p7's G-SUITE carries the reserve's own suite and this one; the P6 profile is unchanged; no trainer-side `--reserve-manifest` exists; the shipped pins are unregistered or well-formed
  --mutations   (minutes) one-line mutations of the gate / profile / pin tool in a COPY of this tree: this test must go RED on each

Run:  python refe/test_p7_integration.py [--mutations]       -> ZZP7INT_PASS <n>
Needs numpy (aux_label_store.finalize), as the reserve suite does.  CPU only, no pod, no network.  Never prints ZZTEST_SKIP.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))
import aux_drivable_head as AUX  # noqa: E402
import aux_label_store as AL  # noqa: E402
import gate_checks as C  # noqa: E402
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402
import p7_pins as PINS  # noqa: E402
import session_reserve as SR  # noqa: E402
import test_p7_head_wiring as HW  # noqa: E402
import test_session_reserve as TS  # noqa: E402

N = 0
FAILS: list = []


def ck(name: str, cond, detail: str = "") -> bool:
    global N
    if cond:
        N += 1
        return True
    FAILS.append(f"{name}: {detail}"[:700])
    return False


def red(name: str, reasons, frag: str) -> bool:
    """`reasons` (a list of strings) must be non-empty AND name `frag`: a negative twin that is red for the WRONG reason is not a twin."""
    txt = " | ".join(reasons) if isinstance(reasons, (list, tuple)) else str(reasons)
    return ck(name, bool(reasons) and frag in txt, f"expected a reason containing {frag!r}; got {txt[:400]!r}")


# ------------------------------------------------------------------------------------------------ the REAL reserve + synthetic logs
FIXTURE = HERE / "fixtures" / "p7_reserve_sessions.json"
RES_BYTES = FIXTURE.read_bytes().replace(b"\r\n", b"\n")                     # LF form: a CRLF checkout (core.autocrlf) is the same text, and load_reserve accepts it; the GATE's sha pin is on the LF bytes
DOC = json.loads(RES_BYTES.decode("utf-8"))
RES_LOGS = [lg["log"] for x in DOC["sessions"] for lg in x["logs"]]           # the real draw's 116 logs (16 sessions)
R_A, R_B = RES_LOGS[0], RES_LOGS[-1]                                          # two reserved logs in different sessions
T1, T2, T3 = "2021.05.12.19.36.12_veh-35_00001_00099", "2021.06.09.14.50.36_veh-26_00007_00070", "2021.11.03.10.20.30_veh-17_00100_00200"       # training logs: sessions NOT in the reserve (asserted)
TRAIN = (T1, T2, T3)
ALL5 = (R_A, R_B) + TRAIN
BANK_NAME = "train_grow_cmd_3c_p7"
P7 = L.PROFILES["refe_p7"]


def reserve_file(td: Path, data: bytes = RES_BYTES, name: str = "reserve_sessions.json") -> Path:
    p = Path(td) / name
    p.write_bytes(data)
    return p


def mk_tree(root, logs=TRAIN, bank_name=BANK_NAME) -> Path:
    """TS.make_inputs (bank dir, D9 cache, aux label store, image root, calibration table) over `logs`, the store RE-FINALISED by the real `aux_label_store.finalize`, the bank dir named as the gate keys it."""
    root = Path(root)
    TS.make_inputs(root, logs)
    AL.finalize(root / "store")
    (root / "bank").rename(root / bank_name)
    return root


def gate_argv(root, bank_name=BANK_NAME) -> list:
    return ["--targets", str(Path(root) / bank_name), "--metric-cache", str(Path(root) / "mc"), "--aux-drv-labels", str(Path(root) / "store"), "--images", str(Path(root) / "pix"),
            "--calib", str(Path(root) / "calib.json")]


def g_reserve(argv, rj, prof=None, **kw) -> dict:
    return C.check_reserve(argv, prof or P7, rj, **kw)


# ======================================================================================================== 0: the digest definition (P7-WIRE-INTEGRATE-2)
def s_digest_definition() -> None:
    """Three copies of ONE definition -- the trainer's `_digest_files` (train.py), the reserve module's, the gate's -- must give the same value on the same files, and the gate's must BE the trainer's (AST)."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "bank"
        d.mkdir()
        (d / "targets_rank0.jsonl").write_bytes(b'{"a": 1}\n{"a": 2}\n')
        (d / "targets_rank1.jsonl").write_bytes(b'{"a": 3}\n')
        (d / "reserve_filter_report.json").write_bytes(b"{}")                         # not a rank file: no definition may hash it
        files = sorted(d.glob("targets_rank*.jsonl"))
        h = hashlib.sha256()
        for q in files:
            h.update(q.name.encode("utf-8"))
            h.update(q.read_bytes())
        ck("0 the gate's bank_targets_sha256 == an independent recompute == the reserve module's digest == the gate's _digest_files",
           C.bank_targets_sha256(d) == h.hexdigest() == SR._digest_files(files) == C._digest_files(files), f"{C.bank_targets_sha256(d)} vs {h.hexdigest()}")
        ck("0 a directory with no rank file has no digest (None, never the digest of nothing)", C.bank_targets_sha256(Path(td) / "empty") is None and C.bank_targets_sha256(d.parent) is None)

    def fn(src, name):
        return next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    ck("0 the gate's _digest_files has the SAME AST as the trainer's train._digest_files (train.py:1067-1074)",
       ast.dump(fn((HERE / "gate_checks.py").read_text(encoding="utf-8"), "_digest_files")) == ast.dump(fn((HERE / "train.py").read_text(encoding="utf-8"), "_digest_files")))


# ======================================================================================================== I: G-RESERVE vs the REAL module
def s_real_module() -> None:
    res = SR.load_reserve(FIXTURE)
    ck("I the synthetic training logs are NOT in the real reserve (else 'clean' is a lie)", all(SR.session_of(t) not in res for t in TRAIN), str([SR.session_of(t) in res for t in TRAIN]))
    ck("I the planted logs ARE in the real reserve", SR.session_of(R_A) in res and SR.session_of(R_B) in res and SR.session_of(R_A) != SR.session_of(R_B))
    ck("I the gate's registered reserve sha256 == the module's own pin (two pins, one draw)", L.P7_RESERVE_SHA256 == SR.RESERVE_FILE_SHA256 == P7["reserve_sha256"] == hashlib.sha256(RES_BYTES).hexdigest())
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rj = reserve_file(td)
        root = mk_tree(td / "clean")
        argv = gate_argv(root)
        # ---- the argument TYPES, against the real signature (a drift in either half is red here, not on the pod)
        kw = C.reserve_audit_inputs(argv)
        params = inspect.signature(SR.audit_inputs).parameters
        kwonly = {n for n, p in params.items() if p.kind is inspect.Parameter.KEYWORD_ONLY} - {"max_report"}
        ck("I reserve_audit_inputs keys == the real audit_inputs' keyword-only parameters", set(kw) == kwonly, f"{sorted(kw)} vs {sorted(kwonly)}")
        ck("I ... bank_files is a list of existing rank files, the other four are path strings", isinstance(kw["bank_files"], list) and len(kw["bank_files"]) == 2 and all(Path(p).is_file() for p in kw["bank_files"])
           and all(isinstance(kw[k], str) and Path(kw[k]).exists() for k in ("metric_cache_dir", "label_store", "image_index", "calib_rows")), str(kw))
        # ---- the CLEAN tree PASSes through the real loader + the real audit
        ev = g_reserve(argv, rj)
        ck("I a CLEAN tree PASSes G-RESERVE (real load_reserve + real audit_inputs)", ev["status"] == "PASS", str(ev["reasons"]))
        d = ev["details"]
        ck("I ... the audit examined all FIVE inputs, found no leak, and the reserve is the 16-session draw", d.get("audit_counts", {}).get("inputs_audited") == sorted(C.RESERVE_AUDITED)
           and not any(d["audit_counts"]["leak_totals"].values()) and all(n > 0 for n in d["audit_counts"]["examined"].values()) and d.get("reserve_sessions") == 16, str(d.get("audit_counts")))
        # ---- the OLD adapter (the parsed JSON document) replayed against the real module: this is the defect that made G-RESERVE unpassable
        old = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))                       # noqa: E731
        try:
            SR.audit_inputs(old(rj), **kw)
            ck("I the real audit REFUSES the parsed JSON document it used to be handed", False, "it accepted it")
        except SR.ReserveError as e:
            ck("I the real audit REFUSES the parsed JSON document it used to be handed", "not a session name" in str(e), str(e)[:200])
        red("I RED ARM: the OLD adapter (parsed JSON) FAILS a clean tree", g_reserve(argv, rj, loader=old)["reasons"], "audit_inputs raised ReserveError")

    def planted(name, plant, frag, inp):
        with tempfile.TemporaryDirectory() as td_:
            td_ = Path(td_)
            rj_ = reserve_file(td_)
            r_ = mk_tree(td_ / "t")
            plant(r_)
            e_ = g_reserve(gate_argv(r_), rj_)
            ck(f"I FAIL: {name}", e_["status"] == "FAIL", str(e_["reasons"]))
            red(f"I FAIL: {name} -- names the key", e_["reasons"], frag)
            red(f"I FAIL: {name} -- names the input", e_["reasons"], inp)
            ck(f"I FAIL: {name} -- the evidence carries the true leak total", sum(e_["details"].get("audit_counts", {}).get("leak_totals", {}).values()) >= 1 or "logless" in name.lower() or "empty" in name.lower(),
               str(e_["details"].get("audit_counts")))
    LR1, GH = TS.LR1, TS.GH
    planted("a planted LEAKED RANK-1 TWIN", lambda r: open(r / BANK_NAME / "targets_rank1.jsonl", "ab").write(json.dumps(TS.row(R_A, "tw", 1, LR1)).encode() + b"\n"),
            f"{R_A}/tw|rank1|lane_rank=1", "'bank'")
    planted("a planted D9 CACHE KEY", lambda r: TS.touch(r / "mc" / R_B / "planted.pkl.xz", b"p"), f"{R_B}/planted.pkl.xz", "metric_cache")
    planted("a planted leaked rank-0 row", lambda r: open(r / BANK_NAME / "targets_rank0.jsonl", "ab").write(json.dumps(TS.row(R_B, "z9", 0, command=1)).encode() + b"\n"), f"{R_B}/z9|rank0", "'bank'")
    planted("a twin part (lane_rank_2) in a reserved log's cache directory", lambda r: TS.touch(r / "mc" / R_A / "tok__lane_rank_2.pkl.xz", b"p"), f"{R_A}/tok__lane_rank_2.pkl.xz", "metric_cache")

    def plant_shard(r):
        TS.write_aux_shard(r / "store", R_A, ("a1", "b2"))
        AL.finalize(r / "store")                                                              # the manifest lists it too (a re-finalised store that still holds the reserve)
    planted("a label-store shard of a reserved log (+ its manifest entry)", plant_shard, f"{R_A}/a1", "label_store")
    planted("an image zip of a reserved log", lambda r: TS.touch(r / "pix" / f"{R_B}_CAM_F0.zip", b"z"), f"{R_B}_CAM_F0.zip", "images")

    def plant_calib(r):
        doc = json.loads((r / "calib.json").read_text(encoding="utf-8"))
        doc["logs"][R_A] = {"CAM_F0": [1.0]}
        (r / "calib.json").write_text(json.dumps(doc), encoding="utf-8")
    planted("a calibration row of a reserved log", plant_calib, f"calib:{R_A}", "calibration")
    planted("a LOGLESS bank row (it would read as clean)", lambda r: open(r / BANK_NAME / "targets_rank0.jsonl", "ab").write(json.dumps({"token": "x", "step": 0, "rank": 0}).encode() + b"\n"),
            "no resolvable log", "'bank'")

    def empty_cache(r):
        shutil.rmtree(r / "mc")
        (r / "mc").mkdir()
    planted("an EMPTY metric-cache directory (a mis-wired probe is not a clean input)", empty_cache, "metric_cache", "ZERO keys")
    # ---- an input the launch argv does not name is NOT AUDITED by the real module, which reads as clean: the gate must refuse it
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rj = reserve_file(td)
        root = mk_tree(td / "t")
        base = gate_argv(root)
        for flag in ("--targets", "--metric-cache", "--aux-drv-labels", "--images", "--calib"):
            e = g_reserve(HW.without(base, flag), rj)
            ck(f"I FAIL: the argv omits {flag}", e["status"] == "FAIL", str(e["reasons"]))
            red(f"I FAIL: the argv omits {flag} -- says which input was not audited", e["reasons"], f"no {flag} input")
        # ---- P7-WIRE-INTEGRATE-2, Gap 1: the EARLY RETURN is behaviour of its own.  Mutant M2 (the return removed) SURVIVED the verdict-only arms above, because the verdict is FAIL with or without it (the
        # omission reason is appended BEFORE the return).  MEASURED (raw/probes/probe_m2_*.out): without the return the audit RUNS on the partial input set (1x instead of 0x -- at the real scale it walks the
        # 92,896-scene bank, the metric cache and a 413,184-file image listing), the verdict carries a SECOND reason (the partial audit's by-product, not the cause), and the evidence records an `audit` block
        # for an audit that proved less than the spec asks.  So the arms below pin the three observables, each against a CONTROL that must read the known value (all five named -> the audit runs once).
        seen: list = []

        def recording(reserve, **k):
            seen.append(sorted(n for n, v in k.items() if v))
            return {"ok": True, "leaks": {}, "unresolved": {}, "counts": {"inputs_audited": list(C.RESERVE_AUDITED), "examined": {}, "leak_totals": {}}}
        ctrl = g_reserve(base, rj, audit=recording)
        ck("I control: with all five inputs named the recording audit runs EXACTLY once and the check PASSes", len(seen) == 1 and len(seen[0]) == 5 and ctrl["status"] == "PASS", str((seen, ctrl["reasons"])))
        ck("I control: ... and the evidence carries the audit block", "audit" in ctrl["details"] and "audit_counts" in ctrl["details"], str(sorted(ctrl["details"])))
        for flag in ("--targets", "--metric-cache", "--aux-drv-labels", "--images", "--calib"):
            seen.clear()
            e = g_reserve(HW.without(base, flag), rj, audit=recording)
            ck(f"I the argv omits {flag}: the audit does NOT RUN on the partial input set (a recording audit, called {len(seen)}x)", seen == [], str(seen))
            ck(f"I ... {flag}: the omission is the ONLY reason (the cause, not a partial audit's by-product)", len(e["reasons"]) == 1 and f"no {flag} input" in e["reasons"][0], str(e["reasons"]))
            ck(f"I ... {flag}: the evidence carries NO audit block (a partial audit is not recorded as if it were the audit)", "audit" not in e["details"] and "audit_counts" not in e["details"], str(sorted(e["details"])))
        calls: list = []
        real_audit = SR.audit_inputs

        def counting(reserve, **k):
            calls.append(1)
            return real_audit(reserve, **k)
        with mock.patch.object(SR, "audit_inputs", counting):
            for flag in ("--targets", "--metric-cache", "--aux-drv-labels", "--images", "--calib"):
                calls.clear()
                e = g_reserve(HW.without(base, flag), rj)
                ck(f"I the REAL audit is not invoked either when the argv omits {flag}", calls == [] and len(e["reasons"]) == 1, str((calls, e["reasons"])))
            calls.clear()
            ck("I control: the same wrapper counts 1 on the full argv (the probe works)", g_reserve(base, rj)["status"] == "PASS" and calls == [1], str(calls))
        (root / "empty_bank").mkdir()
        red("I FAIL: --targets names a directory with no targets_rank*.jsonl", g_reserve(C._set(list(base), "--targets", str(root / "empty_bank")), rj)["reasons"], "no targets_rank*.jsonl")
        # a module that is OK but did not examine all five is a FAIL (the result-side check)
        real = SR.audit_inputs

        def partial(reserve, **k):
            r = real(reserve, **k)
            r["counts"]["inputs_audited"] = [x for x in r["counts"]["inputs_audited"] if x != "calibration"]
            return r
        with mock.patch.object(SR, "audit_inputs", partial):
            red("I FAIL: an audit that did not examine the calibration table, whatever its ok", g_reserve(base, rj)["reasons"], "not all of")
        # the reserve FILE: one changed byte -> both pins refuse; a CRLF copy -> the gate's sha pin refuses (fail closed) though load_reserve accepts it
        e = g_reserve(base, reserve_file(td, RES_BYTES + b" ", "changed.json"))
        ck("I FAIL: a reserve file with one byte appended", e["status"] == "FAIL", str(e["reasons"]))
        red("I FAIL: ... not the registered draw", e["reasons"], "not the registered draw")
        red("I FAIL: ... and the module refuses to load it", e["reasons"], "does not load")
        e = g_reserve(base, reserve_file(td, RES_BYTES.replace(b"\n", b"\r\n"), "crlf.json"))
        ck("I FAIL (closed): a CRLF checkout of the same draw fails the gate's sha pin, and ONLY that", e["status"] == "FAIL" and all("not the registered draw" in r for r in e["reasons"]), str(e["reasons"]))
        ck("I ... while load_reserve itself accepts the CRLF form (the documented asymmetry)", SR.load_reserve(td / "crlf.json") == SR.load_reserve(FIXTURE))
    # ---- the gate audits the SAME files the trainer loads
    tsrc = (HERE / "train.py").read_text(encoding="utf-8")
    gsrc = (HERE / "gate_checks.py").read_text(encoding="utf-8")
    ck("I the trainer's TargetBank loads sorted(Path(target_dir).glob('targets_rank*.jsonl'))", 'sorted(Path(target_dir).glob("targets_rank*.jsonl"))' in tsrc)
    ck("I ... and the gate's reserve_audit_inputs globs the same pattern from --targets", 'Path(tg).glob("targets_rank*.jsonl")' in gsrc)
    ck("I ... and a stray targets_*.jsonl is REFUSED by the trainer (so it cannot be a bank row the audit never saw)", "would be SILENTLY DROPPED. Rename them" in tsrc and 'glob("targets_*.jsonl"))' in tsrc)


# ======================================================================================================== II: through the gate machinery
def s_machinery() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rj = reserve_file(td)
        prof = HW.registered(P7)                                                              # TEST-ONLY copy with the unknown pins registered, never written back
        for label, plant, want in (("clean", None, "PASS"), ("a planted leaked twin", lambda r: open(r / BANK_NAME / "targets_rank1.jsonl", "ab").write(json.dumps(TS.row(R_A, "tw", 1, TS.LR1)).encode() + b"\n"), "FAIL"),
                                   ("a planted D9 key", lambda r: TS.touch(r / "mc" / R_B / "planted.pkl.xz", b"p"), "FAIL")):
            root = mk_tree(td / label.replace(" ", "_"))
            if plant:
                plant(root)
            argv = HW.argv_of(HW.P7_ARGS)
            for flag, val in (("--targets", root / BANK_NAME), ("--images", root / "pix"), ("--calib", root / "calib.json"), ("--metric-cache", root / "mc"), ("--aux-drv-labels", root / "store")):
                argv = C._set(list(argv), flag, str(val))
            ck(f"II [{label}] the profile (test pins registered) refuses nothing on this argv", L.profile_refusals(prof, argv) == [], str(L.profile_refusals(prof, argv)))
            out = td / ("out_" + label.replace(" ", "_"))
            st = L.run_checks(prof, argv, PKG, out, only=("G-RESERVE",), reserve_json=str(rj))
            ck(f"II [{label}] run_checks(only G-RESERVE) -> {want}", st == {"G-RESERVE": want}, str(st))
            evd = GT.read_json(out / "evidence_G-RESERVE.json")
            ck(f"II [{label}] the evidence file records the reserve sha256 and the audit's counts", evd["details"].get("reserve_sha256") == L.P7_RESERVE_SHA256 and "audit_counts" in evd["details"], str(evd["details"])[:300])
            if want == "FAIL":
                red(f"II [{label}] the evidence names the leak", evd["reasons"], "reserve keys in")
        # the reserve file is a bound gate input: it reaches the token's data manifest
        ck("II the reserve file is bound into the token (gate:reserve_json)", L.gate_inputs_of(reserve_json=str(rj)) == {"gate:reserve_json": str(rj)})
        # refe_p7 REFUSES today (nothing invented): the live profile with its pins UNREGISTERED fails every check, G-RESERVE included, before the audit runs
        un = unregistered(P7)
        root = mk_tree(td / "refuse")
        argv = HW.argv_of(HW.P7_ARGS)
        for flag, val in (("--targets", root / BANK_NAME), ("--images", root / "pix"), ("--calib", root / "calib.json"), ("--metric-cache", root / "mc"), ("--aux-drv-labels", root / "store")):
            argv = C._set(list(argv), flag, str(val))
        out = td / "out_unreg"
        st = L.run_checks(un, argv, PKG, out, only=("G-RESERVE",), reserve_json=str(rj))
        ck("II an UNREGISTERED-pins profile FAILS G-RESERVE on a CLEAN tree (the refusal comes before the audit)", st == {"G-RESERVE": "FAIL"}, str(st))
        red("II ... naming both pins", GT.read_json(out / "evidence_G-RESERVE.json")["reasons"], "P7_AUX_STORE_PINS")
        red("II ... and the bank's tuple count", GT.read_json(out / "evidence_G-RESERVE.json")["reasons"], "no registered tuple count")


def unregistered(prof: dict) -> dict:
    """A TEST copy of a P7 profile with the three unknown pins UNREGISTERED -- what the shipped profile is until the pin procedure has run."""
    return {**prof, "aux_store_pins": {"sha256": None, "keys": None}, "expect_schedule": {**prof["expect_schedule"], "n_tuples_by_bank": {}, "targets_sha256_by_bank": {}}}


# ======================================================================================================== III: the pins
def chain_replay(td: Path, rj: Path) -> dict:
    """The reserve stage (p6_pod.sh block 5a) replayed with the module's own functions on a synthetic tree that holds BOTH reserved and training logs: filter the bank (rank-0 AND twins), filter the
    calibration table, MOVE the reserve out of the D9 cache and the image root, split the aux label store and RE-FINALISE it.  The inputs for the unfiltered/full variants are kept for the arms."""
    res = SR.load_reserve(rj)
    root = TS.make_inputs(td / "raw", ALL5)                                                  # bank/ (rank-0 over all five logs + 2 twins), mc/, store/, pix/, calib.json
    src = root / "bank"
    # make_inputs wrote rank-1 twins for (R_A, a1) and (T3, b2); these add (T1, a1), (T2, b2) and (R_B, b2): 5 twins, 2 of them reserved
    with open(src / "targets_rank1.jsonl", "ab") as f:
        for r in (TS.row(T1, "a1", 1, TS.GH), TS.row(T2, "b2", 1, TS.LR1), TS.row(R_B, "b2", 1, TS.LR1)):             # one rank-1 row per scene (the trainer refuses a scene holding a rank twice)
            f.write(json.dumps(r).encode() + b"\n")
    full_store = td / "store_full"
    AL.finalize(root / "store")
    shutil.copytree(root / "store", full_store)                                              # the FULL store (reserve shards present), finalised: a wrong-store arm with REAL numbers
    rep = SR.filter_bank_dir(src, td / BANK_NAME, res, reserve_info=SR._reserve_sha_info(rj))
    SR.filter_calib(root / "calib.json", td / "calib_filtered.json", res)
    SR.move_reserved("cache", root / "mc", td / "mc_reserve", res)
    SR.move_reserved("images", root / "pix", td / "pix_reserve", res)
    SR.split_label_store(root / "store", td / "store_reserve", res)
    man = AL.finalize(root / "store")
    return {"root": root, "bank": td / BANK_NAME, "src_bank": src, "calib": td / "calib_filtered.json", "store": root / "store", "mc": root / "mc", "pix": root / "pix", "full_store": full_store,
            "report": rep, "manifest": man, "res": res}


def count_rows(bank: Path) -> int:
    return sum(1 for p in sorted(bank.glob("targets_rank*.jsonl")) for ln in p.read_bytes().splitlines() if ln.strip())


def pinned(prof: dict, R: dict) -> dict:
    return {**prof, "aux_store_pins": dict(R["registration"]["P7_AUX_STORE_PINS"]),
            "expect_schedule": {**prof["expect_schedule"], "n_tuples_by_bank": dict(R["registration"]["P7_BANK_TUPLES"]), "targets_sha256_by_bank": dict(R["registration"]["P7_TARGETS_SHA256"])}}


def s_pins() -> None:
    argv0 = HW.argv_of(HW.P7_ARGS)
    # ---- the shipped profile today: both pins refuse (via p7_refusals, G-CLOCK and G-DVB independently); partial and malformed registrations refuse too
    un = unregistered(P7)
    rs = L.p7_refusals(un, argv0)
    red("III nothing registered REFUSES: the label store's pins", rs, "label store's pins")
    red("III nothing registered REFUSES: the bank's tuple count", rs, "no registered tuple count")
    want, how = C.clock_bank_tuples(argv0, un["expect_schedule"])
    ck("III G-CLOCK independently has no number for the unregistered bank", want is None and "no registered tuple count" in str(how), str((want, how)))
    a_, b_, s_, p_ = HW.dvb_fixture(prof__aux_store_pins={"sha256": None, "keys": None})
    red("III G-DVB independently FAILS the store row of an unregistered store", [f"{r['flag']}: {r['how']}" for r in C.aux_dvb_rows(a_, b_, s_, p_) if not r["ok"]], "registers NO pins")
    sha_ok = "ab" * 32
    dig_ok = "cd" * 32
    half_store = {**un, "expect_schedule": {**un["expect_schedule"], "n_tuples_by_bank": {BANK_NAME: 7}, "targets_sha256_by_bank": {BANK_NAME: dig_ok}}}
    red("III only the BANK registered: the store pins still refuse", L.p7_refusals(half_store, argv0), "label store's pins")
    ck("III ... and nothing else does", len(L.p7_refusals(half_store, argv0)) == 1, str(L.p7_refusals(half_store, argv0)))
    half_bank = {**un, "aux_store_pins": {"sha256": sha_ok, "keys": 5}}
    red("III only the STORE registered: the bank's count still refuses", L.p7_refusals(half_bank, argv0), "no registered tuple count")
    red("III ... and its content digest too", L.p7_refusals(half_bank, argv0), "no registered targets_sha256")
    ck("III ... and nothing else does", len(L.p7_refusals(half_bank, argv0)) == 2, str(L.p7_refusals(half_bank, argv0)))
    only_count = {**half_bank, "expect_schedule": {**un["expect_schedule"], "n_tuples_by_bank": {BANK_NAME: 7}}}
    red("III the COUNT registered without the CONTENT digest: the digest still refuses (a count cannot tell two same-size banks apart)", L.p7_refusals(only_count, argv0), "no registered targets_sha256")
    ck("III ... and nothing else does", len(L.p7_refusals(only_count, argv0)) == 1, str(L.p7_refusals(only_count, argv0)))
    only_dig = {**half_bank, "expect_schedule": {**un["expect_schedule"], "targets_sha256_by_bank": {BANK_NAME: dig_ok}}}
    red("III the DIGEST registered without the count: the count still refuses", L.p7_refusals(only_dig, argv0), "no registered tuple count")
    keyless = {**half_bank, "expect_schedule": {k: v for k, v in un["expect_schedule"].items() if k != "targets_sha256_by_bank"}}
    red("III a profile WITHOUT the targets_sha256_by_bank key is itself a refusal (a key that can be absent is a default that disables the check)", L.p7_refusals(keyless, argv0), "carries no targets_sha256_by_bank")
    other = {**half_bank, "expect_schedule": {**un["expect_schedule"], "n_tuples_by_bank": {BANK_NAME: 7}, "targets_sha256_by_bank": {BANK_NAME: dig_ok, "another_bank": dig_ok}}}
    red("III the two registrations naming DIFFERENT banks are refused", L.p7_refusals(other, argv0), "register DIFFERENT banks")
    for label, h in (("a short digest", "abc"), ("uppercase", dig_ok.upper()), ("not hex", "zz" * 32), ("an int", 12345), ("None", None), ("63 digits", dig_ok[:-1])):
        red(f"III a MALFORMED content digest is refused: {label}", L.p7_refusals({**half_bank, "expect_schedule": {**un["expect_schedule"], "n_tuples_by_bank": {BANK_NAME: 7}, "targets_sha256_by_bank": {BANK_NAME: h}}}, argv0), "not a sha256")
    for label, pins, frag in (("a short sha", {"sha256": "abc", "keys": 5}, "not a registration"), ("an uppercase sha", {"sha256": sha_ok.upper(), "keys": 5}, "not a registration"),
                              ("keys 0", {"sha256": sha_ok, "keys": 0}, "not a registration"), ("keys as a string", {"sha256": sha_ok, "keys": "103139"}, "not a registration"),
                              ("keys as a bool", {"sha256": sha_ok, "keys": True}, "not a registration")):
        red(f"III a MALFORMED store registration is refused: {label}", L.p7_refusals({**half_bank, "aux_store_pins": pins, "expect_schedule": {**un["expect_schedule"], "n_tuples_by_bank": {BANK_NAME: 7}, "targets_sha256_by_bank": {BANK_NAME: dig_ok}}}, argv0), frag)
    for label, n_t in (("0", 0), ("a string", "150834"), ("a float", 150834.0), ("a bool", True), ("negative", -5)):
        red(f"III a MALFORMED bank registration is refused: {label}", L.p7_refusals({**half_bank, "expect_schedule": {**un["expect_schedule"], "n_tuples_by_bank": {BANK_NAME: n_t}, "targets_sha256_by_bank": {BANK_NAME: dig_ok}}}, argv0), "not a positive int")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rj = reserve_file(td)
        C_ = chain_replay(td, rj)
        res = C_["res"]
        # ---- the tool READS the numbers; recount them independently (a tautology proves nothing)
        R = PINS.measure(rj, C_["bank"], C_["store"], calib=C_["calib"], expect_scenes=6)
        n_rows = count_rows(C_["bank"])
        shards = sorted((C_["store"] / "shards").glob("*.aux1"))
        store_sha = hashlib.sha256("\n".join(f"shards/{p.name}:{hashlib.sha256(p.read_bytes()).hexdigest()}" for p in shards).encode()).hexdigest()
        store_keys = sum(len(SR._read_aux_header(p)["entries"]) for p in shards)
        ck("III the replayed filter excluded the reserve from rank-0 AND the twins (6 rank-0 + 3 twin survivors of 10 + 5)", (C_["report"]["n_in"], C_["report"]["n_out"], C_["report"]["n_excluded"]) == (15, 9, 6), str({k: C_["report"][k] for k in ("n_in", "n_out", "n_excluded")}))
        ck("III p7_pins' tuple count == an independent recount of the rank files", R["bank"]["n_tuples"] == n_rows == 9, f"{R['bank']['n_tuples']} vs {n_rows}")
        ck("III p7_pins' store pins == an independent recompute (shard bytes -> files map -> sha256; header entries -> keys)", (R["store"]["sha256"], R["store"]["keys"]) == (store_sha, store_keys) == (C_["manifest"]["store_sha256"], 6),
           f"{R['store']['sha256'][:16]} {R['store']['keys']} vs {store_sha[:16]} {store_keys}")
        ck("III p7_pins' registration is keyed by the bank DIRECTORY name (what the gate keys by)", R["registration"]["P7_BANK_TUPLES"] == {BANK_NAME: 9}, str(R["registration"]))
        # ---- the profile once the pins are set to THIS tree's values: refusals empty, G-CLOCK has the number, G-DVB's store rows pass on an `aux` event read by the trainer's own reader
        P = pinned(P7, R)
        argv = HW.argv_of(HW.P7_ARGS)
        for flag, val in (("--targets", C_["bank"]), ("--images", C_["pix"]), ("--calib", C_["calib"]), ("--metric-cache", C_["mc"]), ("--aux-drv-labels", C_["store"])):
            argv = C._set(list(argv), flag, str(val))
        ck("III pins set to the tree's values: p7_refusals is EMPTY", L.p7_refusals(P, argv) == [], str(L.p7_refusals(P, argv)))
        ck("III ... profile_refusals (the whole profile + the P7 argv) is EMPTY", L.profile_refusals(P, argv) == [], str(L.profile_refusals(P, argv)))
        want, how = C.clock_bank_tuples(argv, P["expect_schedule"])
        ck("III ... G-CLOCK's registered tuple count == the bank's rows", want == n_rows, str((want, how)))
        man = json.loads((C_["store"] / "manifest.json").read_text(encoding="utf-8"))
        src = AUX.AuxLabelSource(C_["store"], pinned_sha256=man["store_sha256"], pinned_keys=man["counts"]["stored_keys"])       # == train._open_aux_store
        keys = set(src.all_keys())
        rows = [json.loads(ln) for p in sorted(C_["bank"].glob("targets_rank*.jsonl")) for ln in p.read_bytes().splitlines() if ln.strip()]
        lab = sum(1 for r in rows if (r["log_name"], r["token"]) in keys)
        ev = {"event": "aux", "store_sha256": src.manifest["store_sha256"], "store_keys": len(keys), "rows": len(rows), "rows_labelled": lab, "shuffle_seed": None, "tap": "visual", "depth": 2, "w_aux": 0.1}
        a_, b_, s_, p_ = HW.dvb_fixture(prof__aux_store_pins=P["aux_store_pins"], s__metrics=[ev, {"traj_L1": 1.0, "aux_w": 0.1}])
        bad_rows = [r for r in C.aux_dvb_rows(a_, b_, s_, p_) if not r["ok"]]
        ck("III ... G-DVB's aux rows are ALL ok on the `aux` event the trainer's own reader reports", not bad_rows, str(bad_rows[:2]))
        ck("III ... with label coverage 1.0 of the bank's rows", lab == len(rows) == 9, f"{lab}/{len(rows)}")
        # wrong-store arm with REAL numbers: the FULL (pre-split) store
        fman = json.loads((C_["full_store"] / "manifest.json").read_text(encoding="utf-8"))
        fsrc = AUX.AuxLabelSource(C_["full_store"], pinned_sha256=fman["store_sha256"], pinned_keys=fman["counts"]["stored_keys"])
        fev = {**ev, "store_sha256": fman["store_sha256"], "store_keys": len(set(fsrc.all_keys()))}
        a_, b_, s_, p_ = HW.dvb_fixture(prof__aux_store_pins=P["aux_store_pins"], s__metrics=[fev, {"traj_L1": 1.0, "aux_w": 0.1}])
        red("III RED ARM: the FULL store (reserve shards present) is not the registered one", [f"{r['flag']}: {r['how']}" for r in C.aux_dvb_rows(a_, b_, s_, p_) if not r["ok"]], "registered store's")
        ck("III ... the full store really differs (sha and key count)", fman["store_sha256"] != man["store_sha256"] and fev["store_keys"] > ev["store_keys"], f"{fev['store_keys']} vs {ev['store_keys']}")
        # ---- G-RESERVE on the chain's OUTPUTS passes with the profile once pinned; on the pre-stage INPUTS it FAILS (the audit really sees the reserve there)
        out = td / "out_chain"
        st = L.run_checks(P, argv, PKG, out, only=("G-RESERVE",), reserve_json=str(rj))
        ck("III G-RESERVE PASSes on the replayed reserve stage's outputs, through run_checks, with the tree's pins", st == {"G-RESERVE": "PASS"}, str(st))
        raw_argv = gate_argv(C_["root"], "bank")
        raw_argv = C._set(list(raw_argv), "--targets", str(C_["src_bank"]))
        raw_argv = C._set(list(raw_argv), "--aux-drv-labels", str(C_["full_store"]))
        e = g_reserve(raw_argv, rj)
        ck("III G-RESERVE FAILS on the UNFILTERED bank + the FULL store (the stage's inputs)", e["status"] == "FAIL", str(e["reasons"]))
        red("III ... naming the bank and the label store", e["reasons"], "'bank'")
        red("III ... naming the label store", e["reasons"], "label_store")
        # ---- p7_pins' own refusals: each artifact defect refuses, for ITS reason, and no registration is printed
        def refuses(name, frag, bank=None, store=None, calib="default", exp=6):
            try:
                PINS.measure(rj, bank or C_["bank"], store or C_["store"], calib=C_["calib"] if calib == "default" else calib, expect_scenes=exp)
            except (PINS.Refused, SR.ReserveError, OSError) as ex:
                ck(f"III p7_pins REFUSES {name}", frag in str(ex), f"expected {frag!r}; got {str(ex)[:300]!r}")
                return
            ck(f"III p7_pins REFUSES {name}", False, "it printed a registration")
        nb = td / "nb"
        shutil.copytree(C_["bank"], nb)
        (nb / "reserve_filter_report.json").unlink()
        refuses("a bank no filter produced (no reserve_filter_report.json)", "no `session_reserve filter` run produced", bank=nb)
        nb2 = td / "nb2"
        shutil.copytree(C_["bank"], nb2)
        rp = json.loads((nb2 / "reserve_filter_report.json").read_text(encoding="utf-8"))
        rp["n_out"] += 1
        (nb2 / "reserve_filter_report.json").write_text(json.dumps(rp), encoding="utf-8")
        refuses("a report whose n_out is not the rows", "n_out", bank=nb2)
        nb3 = td / "nb3"
        shutil.copytree(C_["bank"], nb3)
        r0 = (nb3 / "targets_rank0.jsonl").read_bytes()
        (nb3 / "targets_rank0.jsonl").write_bytes(r0.replace(b'"a1"', b'"a9"', 1))              # same rows, one byte different: n_out still equal, the digest is not
        refuses("a bank edited after the filter (same row count, different bytes -> digest)", "targets_sha256", bank=nb3)
        nb5 = td / "nb5"
        shutil.copytree(C_["bank"], nb5)
        with open(nb5 / "targets_rank1.jsonl", "ab") as f:
            f.write(json.dumps(TS.row(R_A, "tw", 1, TS.LR1)).encode() + b"\n")
        rp = json.loads((nb5 / "reserve_filter_report.json").read_text(encoding="utf-8"))
        rp["n_out"], rp["targets_sha256"] = count_rows(nb5), SR._digest_files(sorted(nb5.glob("targets_rank*.jsonl")))
        (nb5 / "reserve_filter_report.json").write_text(json.dumps(rp), encoding="utf-8")
        refuses("a LEAKED twin whose report was patched to match (the audit still sees it)", "reserve audit", bank=nb5, exp=7, calib=None)
        nb4 = td / "nb4"
        shutil.copytree(C_["bank"], nb4)
        (nb4 / "targets_aug.jsonl").write_text("{}\n", encoding="utf-8")
        refuses("a stray targets_*.jsonl (the trainer refuses the directory)", "do not match targets_rank", bank=nb4)
        refuses("a scene count the schedule is not sized on (default epoch_len 92,896 vs 6)", "epoch_len", exp=L.P7_SCENES)
        cal = json.loads(C_["calib"].read_text(encoding="utf-8"))
        cal["logs"].pop(T2)
        (td / "calib_missing.json").write_text(json.dumps(cal), encoding="utf-8")
        refuses("a bank log with no calibration row (the trainer would DROP its tuples)", "no calibration row", calib=td / "calib_missing.json")
        refuses("a bank holding a reserved log, which the filtered calibration table (rightly) lacks", "no calibration row", bank=nb5, exp=7)
        refuses("the FULL store (reserve shards present)", "reserve audit", store=C_["full_store"])
        unfin = td / "store_unfinalised"
        shutil.copytree(C_["store"], unfin)
        os.remove(next((unfin / "shards").glob("*.aux1")))                                       # a split that was not re-finalised: the manifest lists a shard that is gone
        refuses("a store split but NOT re-finalised", "refuses this store", store=unfin)
        tam = td / "store_tampered"
        shutil.copytree(C_["store"], tam)
        tm = json.loads((tam / "manifest.json").read_text(encoding="utf-8"))
        tm["counts"]["stored_keys"] += 1                                                        # store_sha256 hashes only the files map: the trainer's reader would still open this store
        (tam / "manifest.json").write_text(json.dumps(tm), encoding="utf-8")
        refuses("a manifest whose stored_keys is not the shard headers' key count", "distinct keys", store=tam)
        ck("III ... and each refusal left the good inputs usable (control: the clean run still reads the same numbers)", PINS.measure(rj, C_["bank"], C_["store"], calib=C_["calib"], expect_scenes=6)["registration"] == R["registration"])
        # ---- the CLI prints a paste-ready registration that parses back to the SAME numbers, and exits 1 on a refusal
        cli = [sys.executable, str(HERE / "p7_pins.py"), "--reserve", str(rj), "--bank", str(C_["bank"]), "--store", str(C_["store"]), "--calib", str(C_["calib"]), "--expect-scenes", "6"]
        r = subprocess.run(cli, capture_output=True, text=True, encoding="utf-8", cwd=str(td))
        ck("III CLI: rc 0 and ZZP7PINS_OK", r.returncode == 0 and "ZZP7PINS_OK tuples=9 scenes=6 store_keys=6" in r.stdout, f"rc {r.returncode}: {r.stdout[-300:]} {r.stderr[-300:]}")
        lines = {ln.split(" =", 1)[0].split(":", 1)[0]: ln.split("=", 1)[1].strip() for ln in r.stdout.splitlines() if ln.startswith(("P7_BANK_TUPLES", "P7_TARGETS_SHA256", "P7_AUX_STORE_PINS"))}
        ck("III CLI: the printed lines parse back to the registration", ast.literal_eval(lines.get("P7_BANK_TUPLES", "None")) == R["registration"]["P7_BANK_TUPLES"]
           and ast.literal_eval(lines.get("P7_AUX_STORE_PINS", "None")) == R["registration"]["P7_AUX_STORE_PINS"]
           and ast.literal_eval(lines.get("P7_TARGETS_SHA256", "None")) == R["registration"]["P7_TARGETS_SHA256"], str(lines))
        ck("III CLI: the OK line carries the content digest's prefix", f"targets_sha256={R['registration']['P7_TARGETS_SHA256'][BANK_NAME][:16]}" in r.stdout, r.stdout[-300:])
        r = subprocess.run(cli[:-2] + ["--expect-scenes", "5"], capture_output=True, text=True, encoding="utf-8", cwd=str(td))
        ck("III CLI: a refusal exits 1, prints ZZP7PINS_REFUSED and NO registration", r.returncode == 1 and "ZZP7PINS_REFUSED" in r.stderr and "P7_BANK_TUPLES" not in r.stdout, f"rc {r.returncode}: {r.stdout[-200:]} {r.stderr[-200:]}")
        s_content(td, rj, C_, R, P, argv)
        # ---- the pod's own chain output must reproduce the dev-box numbers: a second replay from the same inputs gives the SAME pins (the filter and the re-finalise are deterministic)
        td2 = td / "again"
        td2.mkdir()
        C2 = chain_replay(td2, rj)
        R2 = PINS.measure(rj, C2["bank"], C2["store"], calib=C2["calib"], expect_scenes=6)
        ck("III a second replay of the reserve stage reproduces the SAME registration (the pins do not depend on the box or the path)", R2["registration"] == R["registration"], f"{R2['registration']} vs {R['registration']}")


# ======================================================================================================== V: the bank's CONTENT pin (P7-WIRE-INTEGRATE-2, Gap 2)
def s_content(td: Path, rj: Path, C_: dict, R: dict, P: dict, argv: list) -> None:
    """A tuple count cannot tell two same-size banks apart.  P6's content pin (`targets_sha256 8775e73e...`, PREREG_P6.md:191) is the trainer's RESUME identity: `train.run_identity` records
    `_digest_files(sorted(Path(a.targets).glob("targets_rank*.jsonl")))` (train.py:1077-1100) and `_identity_diff` refuses a resume whose digest differs (train.py:1102-1125, called at 1525).  A FRESH start has no
    recorded identity to compare with, so the profile registers the digest (`P7_TARGETS_SHA256`) and the gate refuses at G-CLOCK, the last data check before the PASS token."""
    bank = C_["bank"]
    files = sorted(bank.glob("targets_rank*.jsonl"))
    dig = R["registration"]["P7_TARGETS_SHA256"][BANK_NAME]
    h = hashlib.sha256()
    for q in files:
        h.update(q.name.encode("utf-8"))
        h.update(q.read_bytes())
    ck("V p7_pins' digest is keyed by the bank DIRECTORY name and is a sha256", list(R["registration"]["P7_TARGETS_SHA256"]) == [BANK_NAME] and re.fullmatch(r"[0-9a-f]{64}", dig) is not None, str(R["registration"]))
    ck("V ... == an INDEPENDENT recompute (sha256 over basename + bytes of each rank file, sorted)", dig == h.hexdigest(), f"{dig[:16]} vs {h.hexdigest()[:16]}")
    ck("V ... == the reserve module's digest == the filter report's == the gate's definition (two copies of one definition agree)",
       dig == SR._digest_files(files) == C_["report"]["targets_sha256"] == C.bank_targets_sha256(bank) == C._digest_files(files), f"{dig[:16]} {C.bank_targets_sha256(bank)}")
    # ---- the gate's digest IS the trainer's: the function by AST, the call site and the comparison by source (train.py is not importable here: no torch)
    ttxt = (HERE / "train.py").read_text(encoding="utf-8")

    def fn(src, name):
        return next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    ck("V the gate's _digest_files has the SAME AST as the trainer's _digest_files", ast.dump(fn((HERE / "gate_checks.py").read_text(encoding="utf-8"), "_digest_files")) == ast.dump(fn(ttxt, "_digest_files")))
    ck("V the trainer hashes sorted(Path(a.targets).glob('targets_rank*.jsonl')) into identity.targets_sha256 (unless --grow)", 'tfiles = sorted(Path(a.targets).glob("targets_rank*.jsonl"))' in ttxt
       and '"targets_sha256": None if grow else _digest_files(tfiles)' in ttxt)
    ck("V ... and a resume compares it (train._identity_diff) and the first start records it in config.json", 'for k in ("n_tuples", "n_scenes", "per_rank", "targets_sha256", "scorer_sha256",' in ttxt
       and '"identity": ident, "meta": meta, "warm_start": warm,' in ttxt)
    ck("V --grow (which leaves the digest null) is FORBIDDEN in a refe_p7 launch, and G-CLOCK cannot be NA", "--grow" in P7["forbidden_flags"] and "G-CLOCK" in P7["checks"] and "G-CLOCK" not in P7.get("na_ok", ()))
    ck("V the gate's G-CLOCK reads the SAME files the trainer hashes (same glob)", 'sorted(Path(str(target_dir)).glob("targets_rank*.jsonl"))' in (HERE / "gate_checks.py").read_text(encoding="utf-8"))

    # ---- G-CLOCK on a fake BUILD capture that reports the registered count and schedule: only the CONTENT can differ
    def fake_b(n=9):
        return SimpleNamespace(rc="STOPPED_AT_BUILD", args=SimpleNamespace(epochs=25), steps=9075, samplers=[{"n": 92896}], bank_len={"TargetBank": {"n_tuples": n, "n_scenes": 92896}})

    def clock(prof, argv_, b=None):
        return C.check_clock(argv_, b or fake_b(), prof)
    ev = clock(P, argv)
    ck("V CONTROL: the registered bank passes G-CLOCK, and the evidence carries the digest it compared", ev["status"] == "PASS" and ev["details"].get("bank_content", {}).get("targets_sha256") == dig, str(ev["reasons"]))

    def variant(name, edit):
        d = td / ("v_" + name) / BANK_NAME
        shutil.copytree(bank, d)
        edit(d)
        return d, C._set(list(argv), "--targets", str(d))
    r0 = (bank / "targets_rank0.jsonl").read_bytes()
    lines = r0.splitlines(keepends=True)
    swapped = b"".join([lines[1], lines[0]] + lines[2:])
    flipped = r0.replace(b'"a1"', b'"a9"', 1)
    for label, new in (("ONE BYTE changed in place (same size, same row count)", flipped), ("two rows SWAPPED (same size, same row count, same multiset)", swapped)):
        ck(f"V [{label}] the tampered file really has the same size and a different content", len(new) == len(r0) and new != r0, f"{len(new)} vs {len(r0)}")
        d, av = variant(re.sub(r"\W+", "_", label[:12]), lambda d_, new=new: (d_ / "targets_rank0.jsonl").write_bytes(new))
        ck(f"V [{label}] the row count is unchanged (a count-only gate cannot see it)", count_rows(d) == count_rows(bank) == 9)
        evx = clock(P, av)
        ck(f"V [{label}] G-CLOCK REFUSES the bank at the point where training would start", evx["status"] == "FAIL", str(evx["reasons"]))
        red(f"V [{label}] ... for its CONTENT, naming both digests", evx["reasons"], "CONTENT is not the registered one")
        ck(f"V [{label}] ... and for nothing else (the count and the schedule passed: only the content caught it)", len(evx["reasons"]) == 1, str(evx["reasons"]))
        ck(f"V [{label}] ... the evidence records the digest it read", evx["details"]["bank_content"]["targets_sha256"] not in (None, dig) and evx["details"]["bank_content"]["registered"] == dig)
        # the deliberate-regression arm: the COUNT-ONLY gate (the profile as it was before this package) passes the very same bank
        old = {**P, "p7_rules": False, "expect_schedule": {k: v for k, v in P["expect_schedule"].items() if k != "targets_sha256_by_bank"}}
        ck(f"V [{label}] RED ARM: the count-only gate (no content pin) PASSes the tampered bank -- the defect was real", clock(old, av)["status"] == "PASS")
    d, av = variant("unreg", lambda d_: None)
    d2 = d.parent / "some_other_bank"
    d.rename(d2)
    red("V an UNREGISTERED bank name FAILS G-CLOCK for its content", clock(P, C._set(list(argv), "--targets", str(d2)))["reasons"], "no registered targets_sha256")
    d, av = variant("norank", lambda d_: [p.unlink() for p in d_.glob("targets_rank*.jsonl")])
    red("V a --targets directory with no rank file FAILS G-CLOCK for its content", clock(P, av)["reasons"], "holds no targets_rank")
    keyless = {**P, "expect_schedule": {k: v for k, v in P["expect_schedule"].items() if k != "targets_sha256_by_bank"}}
    red("V a refe_p7 profile WITHOUT the key FAILS G-CLOCK (never a skip)", clock(keyless, argv)["reasons"], "registers no targets_sha256_by_bank")
    # refe_p6 / refe_p2 take none of these lines: no key, no p7_rules -> the content is never read, the evidence is exactly the count-only evidence
    p6like = {**P, "p7_rules": False, "expect_schedule": {k: v for k, v in P["expect_schedule"].items() if k != "targets_sha256_by_bank"}}

    def boom(_d):
        raise AssertionError("a profile with no content registration must not read the bank")
    with mock.patch.object(C, "bank_targets_sha256", boom):
        ev6 = clock(p6like, argv)
    ck("V a profile with no targets_sha256_by_bank (refe_p6, refe_p2) NEVER reads the bank's bytes and its G-CLOCK evidence has no bank_content block", ev6["status"] == "PASS" and "bank_content" not in ev6["details"], str(ev6))
    ck("V ... and the P6 key (n_tuples_by_bank only) is unchanged: refe_p6 has no content key", "targets_sha256_by_bank" not in L.PROFILES["refe_p6"]["expect_schedule"] and "targets_sha256_by_bank" in P7["expect_schedule"])
    # ---- through the gate machinery: run_checks(only G-CLOCK) with the BUILD capture replaced by the fake
    with mock.patch.object(C, "instrumented_run", lambda *a_, **k_: fake_b()):
        out = td / "out_clock_ok"
        st = L.run_checks(P, argv, PKG, out, only=("G-CLOCK",), reserve_json=str(rj))
        ck("V run_checks(only G-CLOCK) on the registered bank -> PASS", st == {"G-CLOCK": "PASS"}, str(st))
        d, av = variant("mach", lambda d_: (d_ / "targets_rank0.jsonl").write_bytes(flipped))
        out2 = td / "out_clock_bad"
        st2 = L.run_checks(P, av, PKG, out2, only=("G-CLOCK",), reserve_json=str(rj))
        ck("V run_checks(only G-CLOCK) on the SAME-COUNT, DIFFERENT-BYTES bank -> FAIL (so no PASS token can be minted)", st2 == {"G-CLOCK": "FAIL"}, str(st2))
        red("V ... and the evidence file says why", GT.read_json(out2 / "evidence_G-CLOCK.json")["reasons"], "CONTENT is not the registered one")
    # ---- p7_pins: the two definitions must agree on THIS bank, or it refuses (the gate would FAIL it on the pod)
    with mock.patch.object(C, "bank_targets_sha256", lambda d: "0" * 64):
        try:
            PINS.measure(rj, bank, C_["store"], calib=C_["calib"], expect_scenes=6)
            ck("V p7_pins REFUSES when the gate's digest definition disagrees with the reserve module's", False, "it printed a registration")
        except PINS.Refused as ex:
            ck("V p7_pins REFUSES when the gate's digest definition disagrees with the reserve module's", "drifted" in str(ex), str(ex)[:200])
    # ---- AFTER the first start: the trainer's OWN recorded identity (config.json) against the registered pins
    def mk_run(name, ident, targets=bank, cfg=True):
        d = td / name
        d.mkdir(parents=True, exist_ok=True)
        if cfg:
            (d / "config.json").write_text(json.dumps({"argv": ["train.py"], "args": {"targets": str(targets)}, "identity": ident}), encoding="utf-8")
        return d
    good = {"targets_sha256": dig, "n_tuples": 9, "n_scenes": 6}
    r_ok = PINS.check_run(mk_run("run_ok", good), P)
    ck("V check_run: the trainer's recorded identity == the registration -> ok", r_ok["ok"] and r_ok["reasons"] == [] and r_ok["bank"] == BANK_NAME, str(r_ok))
    red("V check_run REFUSES a recorded digest that is not the registered one (the trainer trained on another bank)", PINS.check_run(mk_run("run_bad", {**good, "targets_sha256": "0" * 64}), P)["reasons"], "trained on another bank")
    red("V check_run REFUSES a recorded tuple count that is not the registered one", PINS.check_run(mk_run("run_n", {**good, "n_tuples": 10}), P)["reasons"], "the trainer built 10 tuples")
    red("V check_run REFUSES a null digest (a --grow run)", PINS.check_run(mk_run("run_null", {**good, "targets_sha256": None}), P)["reasons"], "recorded no targets_sha256")
    red("V check_run REFUSES a bank the profile never vouched for", PINS.check_run(mk_run("run_unreg", good, targets=td / "elsewhere" / "some_other_bank"), P)["reasons"], "is not registered")
    red("V check_run REFUSES a run that has no config.json yet", PINS.check_run(mk_run("run_none", good, cfg=False), P)["reasons"], "is missing")
    red("V check_run on the LIVE profile (pins unregistered) refuses everything", PINS.check_run(mk_run("run_live", good))["reasons"], "is not registered")
    buf_o, buf_e = io.StringIO(), io.StringIO()
    with mock.patch.object(L, "PROFILES", {**L.PROFILES, "refe_p7": P}), contextlib.redirect_stdout(buf_o), contextlib.redirect_stderr(buf_e):
        rc_ok = PINS.main(["--check-run", str(td / "run_ok")])
        rc_bad = PINS.main(["--check-run", str(td / "run_bad")])
    ck("V CLI --check-run: rc 0 + ZZP7RUN_OK on the registered identity, rc 1 + ZZP7RUN_REFUSED on another bank", rc_ok == 0 and "ZZP7RUN_OK" in buf_o.getvalue() and rc_bad == 1 and "ZZP7RUN_REFUSED" in buf_e.getvalue(), f"{rc_ok} {rc_bad} {buf_o.getvalue()[:100]} {buf_e.getvalue()[:100]}")
    r = subprocess.run([sys.executable, str(HERE / "p7_pins.py"), "--check-run", str(td / "run_ok")], capture_output=True, text=True, encoding="utf-8", cwd=str(td))
    ck("V CLI --check-run as a subprocess on the shipped (unregistered) profile: rc 1, ZZP7RUN_REFUSED, nothing printed to stdout", r.returncode == 1 and "ZZP7RUN_REFUSED" in r.stderr and r.stdout.strip() == "", f"{r.returncode} {r.stdout[:100]} {r.stderr[:200]}")
    r = subprocess.run([sys.executable, str(HERE / "p7_pins.py")], capture_output=True, text=True, encoding="utf-8", cwd=str(td))
    ck("V the pin tool with no arguments still exits 2 (argparse), as it did when --reserve / --bank / --store were required", r.returncode == 2 and "--check-run" in r.stderr, f"{r.returncode} {r.stderr[-200:]}")


# ======================================================================================================== IV: design pins
def s_design() -> None:
    P6 = L.PROFILES["refe_p6"]
    ck("IV refe_p6 is byte-for-byte what it was (profile digest)", hashlib.sha256(repr(sorted(P6.items())).encode()).hexdigest() == HW.P6_PROFILE_SHA256)
    ck("IV refe_p6's suite and checks carry nothing of P7", "G-RESERVE" not in P6["checks"] and not any("session_reserve" in str(e) or "p7" in str(e) for e in P6["suite"]), str(P6["suite"]))
    suite = P7["suite"]
    ck("IV refe_p7's suite = P6's + the head's static block + the reserve's own suite + this test", suite[:len(P6["suite"])] == P6["suite"]
       and suite[len(P6["suite"]):] == [{"script": "test_p7_head_wiring.py", "args": ["--static"], "token": "ZZP7HEAD_PASS"}, {"script": "test_session_reserve.py", "token": "ZZSESSRES_PASS"},
                                        {"script": "test_p7_integration.py", "token": "ZZP7INT_PASS"}], str(suite[len(P6["suite"]):]))
    for e in suite[len(P6["suite"]):]:
        p = HERE / e["script"]
        ck(f"IV {e['script']} exists and its source prints its own token {e['token']} (a renamed token would fail G-SUITE on the pod, not here)", p.is_file() and e["token"] in p.read_text(encoding="utf-8"))
    ck("IV the reserve suite's token is the REAL one (ZZSESSRES_PASS, not the head's ZZRESERVE_PASS 17)", all(e["token"] != "ZZRESERVE_PASS" for e in suite))
    pins, tup, dg = P7["aux_store_pins"], P7["expect_schedule"]["n_tuples_by_bank"], P7["expect_schedule"]["targets_sha256_by_bank"]
    ck("IV the shipped refe_p7 pins are UNREGISTERED or WELL-FORMED, never partial or typed-in garbage (this survives the day the real pins are registered)",
       (pins == {"sha256": None, "keys": None} and tup == {} and dg == {})
       or bool(re.fullmatch(r"[0-9a-f]{64}", str(pins["sha256"])) and type(pins["keys"]) is int and pins["keys"] > 0 and tup and all(type(v) is int and v > 0 for v in tup.values())
               and set(dg) == set(tup) and all(re.fullmatch(r"[0-9a-f]{64}", str(v)) for v in dg.values())), str((pins, tup, dg)))
    tsrc = (HERE / "train.py").read_text(encoding="utf-8")
    ck("IV DESIGN (a): the trainer has NO --reserve-manifest flag, and refe_p7 neither requires nor fingerprints one (adding it later is a deliberate change: a G-DVB entry, P6 bit-identity, REQUIRED in argv)",
       "reserve-manifest" not in tsrc and "--reserve-manifest" not in P7["required_flags"] and "--reserve-manifest" not in P7["data_flags"])
    ck("IV the pin procedure's tool ships beside the gate (refe/p7_pins.py) and the gate names it", (HERE / "p7_pins.py").is_file() and "p7_pins.py" in (HERE / "launch_gate.py").read_text(encoding="utf-8"))


def main() -> int:
    if "--mutations" in sys.argv:
        return mutations()
    for section in (s_digest_definition, s_real_module, s_machinery, s_pins, s_design):
        try:
            section()
        except Exception as e:                                                          # noqa: BLE001 -- a section that RAISES is a failure with a name, not a traceback that hides the other sections
            import traceback
            FAILS.append(f"{section.__name__} RAISED {type(e).__name__}: {str(e)[:300]} @ {traceback.extract_tb(e.__traceback__)[-1].lineno}")
    if FAILS:
        for f in FAILS:
            print("ZZP7INT_RED", f)
        print(f"ZZP7INT_FAIL {len(FAILS)} failing, {N} passing")
        return 1
    print(f"ZZP7INT_PASS {N}")
    return 0


# ======================================================================================================== --mutations
MUTATIONS = [
    # (label, file, old, new, the section(s) that must go red)
    ("M1 REVERT THE FIX: hand the audit the parsed JSON again", "gate_checks.py", "        reserve = loader(reserve_json)\n",
     "        reserve = json.loads(Path(str(reserve_json)).read_text(encoding=\"utf-8\"))\n"),
    ("M2 an input the argv does not name is no longer refused", "gate_checks.py", "    if missing:\n        return verdict(\"G-RESERVE\", reasons, det, arm=arm, rehearsal=rehearsal)\n",
     "    if False:\n        return verdict(\"G-RESERVE\", reasons, det, arm=arm, rehearsal=rehearsal)\n"),
    ("M3 an audit with ok False no longer fails", "gate_checks.py", "if not isinstance(res, dict) or res.get(\"ok\") is not True:\n        why =", "if not isinstance(res, dict):\n        why ="),
    ("M4 the result-side 'examined all five' check is gone", "gate_checks.py", "        if not set(RESERVE_AUDITED) <= done:\n", "        if False:\n"),
    ("M5 the gate stops requiring the registered reserve sha256", "gate_checks.py", "    elif sha != pin:\n", "    elif False:\n"),
    ("M6 the gate audits another glob than the trainer loads", "gate_checks.py", "Path(tg).glob(\"targets_rank*.jsonl\")", "Path(tg).glob(\"targets_rank0.jsonl\")"),
    ("M7 unregistered store pins no longer refuse", "launch_gate.py", "    if pins.get(\"sha256\") is None or pins.get(\"keys\") is None:\n", "    if False:\n"),
    ("M8 an unregistered bank no longer refuses", "launch_gate.py", "    if name not in reg:\n", "    if False:\n"),
    ("M9 a malformed store registration is accepted", "launch_gate.py", "    elif not (isinstance(pins[\"sha256\"], str)", "    elif False and not (isinstance(pins[\"sha256\"], str)"),
    ("M10 a malformed bank registration is accepted", "launch_gate.py", "        if type(n_t) is not int or n_t <= 0:\n", "        if False:\n"),
    ("M11 the reserve's own suite is dropped from refe_p7's G-SUITE", "launch_gate.py", "                  {\"script\": \"test_session_reserve.py\", \"token\": \"ZZSESSRES_PASS\"},\n", ""),
    ("M12 p7_pins no longer checks the filter report's digest", "p7_pins.py", "    if rep.get(\"targets_sha256\") != digest:\n", "    if False:\n"),
    ("M13 p7_pins no longer audits the bank + store", "p7_pins.py", "    if not a[\"ok\"]:\n", "    if False:\n"),
    ("M14 p7_pins reads the key count from the manifest instead of the shard headers", "p7_pins.py", "    if len(set(all_keys)) != keys or len(all_keys) != keys:\n", "    if False:\n"),
    ("M15 p7_pins pins the wrong store field (the first shard's sha, not the store's)", "p7_pins.py", "\"sha256\": sha, \"keys\": keys, \"logs\"", "\"sha256\": next(iter(man['files'].values()))['sha256'], \"keys\": keys, \"logs\""),
    # ---- P7-WIRE-INTEGRATE-2, Gap 2: the bank's CONTENT pin
    ("M16 G-CLOCK stops comparing the bank's content digest with the registered one", "gate_checks.py", "            elif got_sha != want_sha:\n", "            elif False:\n"),
    ("M17 an unregistered bank no longer refuses for its content digest", "launch_gate.py", "        if name not in dig:\n", "        if False:\n"),
    ("M18 a malformed content digest is accepted", "launch_gate.py", "            if not (isinstance(h, str) and re.fullmatch(r\"[0-9a-f]{64}\", h)):\n", "            if False:\n"),
    ("M19 a refe_p7 profile without the content key is skipped instead of failing G-CLOCK", "gate_checks.py", ") if required else (None, None)\n", ") if False else (None, None)\n"),
    ("M20 the gate's digest drifts from the trainer's (hashes the full path, not the basename)", "gate_checks.py", "        h.update(os.path.basename(str(q)).encode(\"utf-8\"))\n", "        h.update(str(q).encode(\"utf-8\"))\n"),
    ("M21 p7_pins registers a digest that is not the bank's", "p7_pins.py", "\"P7_TARGETS_SHA256\": {b[\"dir_name\"]: b[\"targets_sha256\"]}", "\"P7_TARGETS_SHA256\": {b[\"dir_name\"]: b[\"targets_sha256\"][::-1]}"),
    ("M22 p7_pins stops cross-checking the gate's digest definition", "p7_pins.py", "    if gate_digest != digest:", "    if False:"),
    ("M23 check_run stops comparing the trainer's recorded digest", "p7_pins.py", "    elif want_sha is not None and ident[\"targets_sha256\"] != want_sha:\n", "    elif False:\n"),
    ("M24 check_run stops comparing the trainer's recorded tuple count", "p7_pins.py", "    if want_n is not None and ident.get(\"n_tuples\") != want_n:\n", "    if False:\n"),
    ("M25 the count and the digest may name different banks", "launch_gate.py", "        if reg and dig and set(reg) != set(dig):\n", "        if False:\n"),
]


def mutations() -> int:
    """Each mutation is applied to a COPY of this tree (its target text must occur EXACTLY once, so 'not applied' can never read as 'stayed green') and the copy's own static run must go RED."""
    root = Path(tempfile.mkdtemp(prefix="p7int_mut_"))
    bad = 0
    try:
        ctrl = root / "ctrl"
        shutil.copytree(HERE, ctrl / "refe", ignore=shutil.ignore_patterns("__pycache__"))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1")
        r = subprocess.run([sys.executable, str(ctrl / "refe" / "test_p7_integration.py")], capture_output=True, text=True, encoding="utf-8", env=env, cwd=str(ctrl / "refe"))
        ok = r.returncode == 0 and "ZZP7INT_PASS" in r.stdout
        print(f"CONTROL (unmutated copy): {'PASS' if ok else 'NOT GREEN'} {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]}")
        bad += 0 if ok else 1
        for label, fname, old, new in MUTATIONS:
            t = root / re.sub(r"\W+", "_", label[:6])
            shutil.copytree(ctrl, t)
            f = t / "refe" / fname
            s = f.read_text(encoding="utf-8")
            if s.count(old) != 1:
                print(f"  ERROR {label}: target text occurs {s.count(old)} times in {fname}")
                bad += 1
                continue
            f.write_text(s.replace(old, new), encoding="utf-8", newline="")
            r = subprocess.run([sys.executable, str(t / "refe" / "test_p7_integration.py")], capture_output=True, text=True, encoding="utf-8", env=env, cwd=str(t / "refe"))
            reds = [ln for ln in r.stdout.splitlines() if ln.startswith("ZZP7INT_RED")]
            is_red = r.returncode != 0 and (bool(reds) or "Traceback" in r.stderr)
            print(f"  {'RED   ' if is_red else 'NOT RED'} {label}: {len(reds)} failing" + (f"; first: {reds[0][12:150]}" if reds else (f"; stderr: {r.stderr[-150:]}" if r.stderr else "")))
            bad += 0 if is_red else 1
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("ZZP7INT_MUTATIONS_OK" if not bad else f"ZZP7INT_MUTATIONS_NOT_OK {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
