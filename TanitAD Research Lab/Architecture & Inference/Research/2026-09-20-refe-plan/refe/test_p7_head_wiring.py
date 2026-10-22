#!/usr/bin/env python3
"""P7-WIRE-HEAD: the training-only drivable-area head (P7.1) in the TRUNK trainer `refe/train.py`, and its launch-gate coverage (SPEC_P7.md sections 1-2;
SPEC_AUX_DRIVABLE_HEAD.md sections 4, 7).  "Built, tested and UNREACHABLE from its caller" is a class (CLAUDE.md, four instances in one day), so every claim below is
proved from the CALLER: the flag -> namespace -> built head -> tap -> loss -> BOTH backward sites -> optimiser -> checkpoint, each hop with a mutation that turns it RED.

  default / --static  (seconds, no trainer run; THIS is what the gate's own G-SUITE runs, so it needs no dev-box data)   prints `ZZP7HEAD_PASS <n>`
      S1  refe_p6 is byte-for-byte what it was (a digest of its profile dict), and its own argv is still clean
      S2  the refe_p7 profile: clone of P6 + the head's levers REQUIRED with their registered values, `--aux-drv-labels` a DATA flag, the shuffle control FORBIDDEN, G-RESERVE a check,
          the 25 x ceil(92,896/256) = 9,075-step schedule; the three UNKNOWN pins (store sha256 + keys, bank tuple count) are unregistered ON PURPOSE and REFUSE
      S3  every default-off lever omitted from the argv is REFUSED (`--w-aux-drv` DEFAULTS to 0.0: the arm would train head-less and read as clean), and each moved value
      S4  G-RESERVE: the registered reserve file's sha256 + `session_reserve.audit_inputs(...)` with every input the launch argv names; `ok` False / absent / raising / non-dict FAILS
      S5  the judges re-run from RAW FIELDS, each with a green fixture and one negative twin per defect (the head's G-DVB rows, G-LIVE facts and reach)
      S6  train.py's hop census (flag -> head -> tap -> loss -> both backward sites -> optimiser -> checkpoints) and a SOURCE-MUTATION harness: every hop's deletion goes RED
  --full  (minutes, CPU; needs the dev box's `D:/Projects/TanitAD/data/refe_consistent`, READ ONLY)   prints `ZZP7HEAD_FULL <n>` after the static block
      F1  default-off BIT-IDENTITY: the new trainer without the flag == the frozen pre-edit trainer (`fixtures/train_pre_p7_head.py.txt`), with its own controls
      F2  a head arm through the REAL gate (G-DVB, G-LIVE, G-EVAL, G-CKPT) on a tiny rehearsal bank + a mini label store: every check PASSes clean
      F3  the gate-instrument arms (weight off / zero / detached term, resume drops the head) each make THEIR check FAIL
  --mutations  (minutes, CPU)   the deliberate-regression arms on the TRAINER SOURCE and the ARGV, each through the real gate in a temp copy of this tree; prints the table
Any skip prints ZZTEST_SKIP and is NOT counted.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))
import gate_checks as C  # noqa: E402
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402

N = 0
FAILS: list = []


def ck(name: str, cond, detail: str = "") -> bool:
    """One counted assertion.  A failure is recorded (and printed at the end), never silently dropped; the run keeps going so ONE report lists every red."""
    global N
    if cond:
        N += 1
        return True
    FAILS.append(f"{name}: {detail}"[:600])
    return False


def red(name: str, reasons, frag: str) -> bool:
    """`reasons` (a list of strings) must be non-empty AND name `frag`: a negative twin that is red for the WRONG reason is not a twin."""
    txt = " | ".join(reasons) if isinstance(reasons, (list, tuple)) else str(reasons)
    return ck(name, bool(reasons) and frag in txt, f"expected a reason containing {frag!r}; got {txt[:300]!r}")


# ======================================================================================================== pins (each is a MEASURED digest or a spec number)
P6_PROFILE_SHA256 = "80fa4d7ec757ad4658df04b1ecfd453ae98653e729e2ee24458769d851cdf9c3"          # sha256(repr(sorted(PROFILES["refe_p6"].items()))) on the tip 1664143b, MEASURED 2026-10-22
PRE_P7_TRAIN_SHA256 = "6b425ac7e72ce2b33c35e7c71431beb526faa9b6b0c5a9b159860784a5efcaaf"        # refe/train.py at the tip BEFORE this wiring (fixtures/train_pre_p7_head.py.txt, LF form)
RESERVE_SHA256 = "53f3bcd4684513f0ee765fbcfbaf583b587f58c97e9c8915f4e4df3ae3e32f29"            # raw/2026-10-22-p7-session-reserve/raw/reserve_sessions.json (R-A), SESSION_RESERVE.md
AUX_DESTS = ("aux_drv", "w_aux_drv", "aux_drv_tap", "aux_drv_depth", "aux_drv_feed", "aux_drv_labels", "aux_drv_shuffle_seed")
SHA_T = "ab" * 32                                                                                 # a TEST registration of a label store's sha256 (never a real pin)

P6_ARGS = ("--backbone vitl16 --targets /w/train_grow --images /w/pix --calib /w/calib.json --metric-cache /w/mc --scorer-sees-trunk --yaw-wrap off --nav-mode command "
           "--labeller online --label-chunks 2 --epochs 25 --batch 4 --accum 64 --lr 2e-4 --weight-decay 0.01 --amp bf16 --score-w 0.1 --tf32 --compile --workers 2 "
           "--log-every 1 --ckpt-every-min 20 --out /w/run --resume")
P7_ARGS = (P6_ARGS.replace("/w/train_grow", "/w/train_grow_cmd_3c_p7")
           + " --aux-drv --w-aux-drv 0.1 --aux-drv-tap visual --aux-drv-depth 2 --aux-drv-feed off --aux-drv-labels /w/aux_store --w-field 0.0")


def argv_of(s: str) -> list:
    return L.parse_args_string(s)


def without(argv: list, flag: str, nvalues: int = 1) -> list:
    return C._drop(list(argv), flag, nvalues)


def registered(prof: dict) -> dict:
    """TEST-ONLY copy of a P7 profile with its three unknown pins REGISTERED (the real ones come from filter_reserve's report, SPEC_P7 2).  Never written back to PROFILES."""
    exp = {**prof["expect_schedule"], "n_tuples_by_bank": {"train_grow_cmd_3c_p7": 150800}}
    return {**prof, "aux_store_pins": {"sha256": SHA_T, "keys": 5}, "expect_schedule": exp}


# ======================================================================================================== S1 + S2: the profiles
def s_profiles() -> None:
    P6, P7 = L.PROFILES["refe_p6"], L.PROFILES["refe_p7"]
    ck("S1 refe_p6 profile digest", hashlib.sha256(repr(sorted(P6.items())).encode()).hexdigest() == P6_PROFILE_SHA256,
       "refe_p6 changed: P6 resumes on the pod from a pinned older tree and its ARGS sha is pinned -- the P7 profile is a CLONE, never an edit")
    ck("S1 refe_p6 argv still clean", L.profile_refusals(P6, argv_of(P6_ARGS)) == [], str(L.profile_refusals(P6, argv_of(P6_ARGS))))
    ck("S1 the P6 profile names no head lever", not any("aux" in f for f in P6["required_flags"]) and "G-RESERVE" not in P6["checks"] and not P6.get("p7_rules"))
    # ---- S2 the refe_p7 profile
    ck("S2 name + p7_rules", P7["name"] == "refe_p7" and P7.get("p7_rules") is True)
    ck("S2 checks = P6's + G-RESERVE", P7["checks"] == P6["checks"] + ("G-RESERVE",) and "G-RESERVE" in L.CHECKS)
    ck("S2 every check the profile names is known", all(c in L.CHECKS for c in tuple(P7["checks"]) + tuple(P7.get("na_ok", ()))))
    new_req = set(P7["required_flags"]) - set(P6["required_flags"])
    ck("S2 the head's levers + --w-field are REQUIRED on top of P6's", new_req == {"--aux-drv", "--w-aux-drv", "--aux-drv-tap", "--aux-drv-depth", "--aux-drv-feed", "--aux-drv-labels", "--w-field"}, str(new_req))
    ck("S2 every P6 requirement is kept verbatim", all(P7["required_flags"][k] == v for k, v in P6["required_flags"].items()))
    rq = P7["required_flags"]
    ck("S2 registered values", (rq["--w-aux-drv"], rq["--aux-drv-tap"], rq["--aux-drv-depth"], rq["--aux-drv-feed"], rq["--w-field"]) == ("0.1", "visual", "2", "off", "0.0")
       and rq["--aux-drv"] is None and rq["--aux-drv-labels"] is None)
    ck("S2 the label store is a DATA flag (the token fingerprints it)", "--aux-drv-labels" in P7["data_flags"] and "--aux-drv-labels" not in P6["data_flags"]
       and P7["data_flags"][:len(P6["data_flags"])] == P6["data_flags"])
    ck("S2 the shuffle control is forbidden in a launch", "--aux-drv-shuffle-seed" in P7["forbidden_flags"] and "--aux-drv-shuffle-seed" not in P6["forbidden_flags"])
    ck("S2 live terms carry the head's", set(P7["live_terms"]) == set(P6["live_terms"]) | {"aux_bce", "aux_huber"})
    ck("S2 the head's terms exist in the trainer's source (a declared term the trainer never emits is a launch blocker)", all(f'"{t}"' in (HERE / "train.py").read_text(encoding="utf-8") for t in ("aux_bce", "aux_huber", "aux_w")))
    exp = P7["expect_schedule"]
    ck("S2 schedule: 25 epochs x ceil(92,896 / 256) = 9,075 steps", (exp["epochs"], exp["epoch_len"], exp["steps"]) == (25, 92896, 9075) == (25, 92896, 25 * math.ceil(92896 / 256)), str(exp))
    ck("S2 the profile's suite carries P6's entries + this test", P7["suite"][:len(P6["suite"])] == P6["suite"]
       and {"script": "test_p7_head_wiring.py", "args": ["--static"], "token": "ZZP7HEAD_PASS"} in P7["suite"], str(P7["suite"][-1]))
    ck("S2 the reserve sha256 is the registered R-A draw", P7["reserve_sha256"] == RESERVE_SHA256)
    ck("S2 the keys P7 adds to P6's", sorted(set(P7) - set(P6)) == ["aux_store_pins", "p7_rules", "reserve_sha256"], str(sorted(set(P7) - set(P6))))
    ck("S2 the unknown pins are UNREGISTERED in the profile (never defaulted)", P7["aux_store_pins"] == {"sha256": None, "keys": None} and P7["expect_schedule"]["n_tuples_by_bank"] == {})
    ck("S2 P7 is not a rehearsal and can mint", P7.get("mintable", True) and not P7.get("rehearsal"))
    # the pins' refusal reasons on the real (unregistered) profile
    rs = L.profile_refusals(P7, argv_of(P7_ARGS))
    red("S2 unregistered store pins REFUSE", rs, "label store's pins")
    red("S2 unregistered bank REFUSES", rs, "has no registered tuple count")
    ck("S2 ... and nothing else is refused on the clean P7 argv", len(rs) == 2, str(rs))
    # G-CLOCK independently FAILS an unregistered bank (the refusal above is not the only defence)
    want, how = C.clock_bank_tuples(argv_of(P7_ARGS), P7["expect_schedule"])
    ck("S2 G-CLOCK has no number for the unregistered bank", want is None and "no registered tuple count" in how, str(how))
    ck("S2 the profile with pins REGISTERED (test copy) is clean", L.profile_refusals(registered(P7), argv_of(P7_ARGS)) == [], str(L.profile_refusals(registered(P7), argv_of(P7_ARGS))))
    # the sibling pin checks of p7_refusals
    bad = registered(P7)
    bad["expect_schedule"] = {**bad["expect_schedule"], "steps": 9076}
    red("S2 a moved schedule is refused", L.p7_refusals(bad, argv_of(P7_ARGS)), "9,075")
    red("S2 no reserve sha registered is refused", L.p7_refusals({**registered(P7), "reserve_sha256": None}, argv_of(P7_ARGS)), "no reserve sha256")
    red("S2 one-sided store pin is refused", L.p7_refusals({**registered(P7), "aux_store_pins": {"sha256": SHA_T, "keys": None}}, argv_of(P7_ARGS)), "label store's pins")


# ======================================================================================================== S3: argv refusals
def s_argv() -> None:
    P = registered(L.PROFILES["refe_p7"])
    base = argv_of(P7_ARGS)
    ck("S3 control: the clean argv is not refused", L.profile_refusals(P, base) == [])

    def swap(argv, flag, val):
        return C._set(list(argv), flag, val)
    cases = [
        ("flag omitted: --aux-drv", without(base, "--aux-drv", 0), "requires --aux-drv"),
        ("DEFAULT-OFF lever omitted: --w-aux-drv (its default 0.0 trains head-less and reads as clean)", without(base, "--w-aux-drv"), "requires --w-aux-drv 0.1; the argv does not carry it"),
        ("weight 0.0 named", swap(base, "--w-aux-drv", "0.0"), "requires --w-aux-drv 0.1; the argv has --w-aux-drv 0.0"),
        ("weight 0.2", swap(base, "--w-aux-drv", "0.2"), "requires --w-aux-drv 0.1"),
        ("tap scene", swap(base, "--aux-drv-tap", "scene"), "requires --aux-drv-tap visual"),
        ("depth 1", swap(base, "--aux-drv-depth", "1"), "requires --aux-drv-depth 2"),
        ("feed on (H2)", swap(base, "--aux-drv-feed", "on"), "requires --aux-drv-feed off"),
        ("tap omitted (default is visual: still must be NAMED)", without(base, "--aux-drv-tap"), "requires --aux-drv-tap visual; the argv does not carry it"),
        ("labels omitted", without(base, "--aux-drv-labels"), "requires --aux-drv-labels"),
        ("--w-field 0.1 (the field loss is its own arm)", swap(base, "--w-field", "0.1"), "requires --w-field 0.0"),
        ("--w-field omitted", without(base, "--w-field"), "requires --w-field 0.0; the argv does not carry it"),
        ("the shuffle control", base + ["--aux-drv-shuffle-seed", "3"], "FORBIDS"),
        ("a P6 argv (no head at all) under the P7 profile", argv_of(P6_ARGS.replace("/w/train_grow", "/w/train_grow_cmd_3c_p7")), "requires --aux-drv"),
        ("a bank with no registered tuple count", swap(base, "--targets", "/w/train_grow_cmd_3c"), "has no registered tuple count"),
        ("P6's moved lever still refused (--yaw-wrap on)", swap(base, "--yaw-wrap", "on"), "requires --yaw-wrap off"),
    ]
    for name, argv, frag in cases:
        red(f"S3 refused: {name}", L.profile_refusals(P, argv), frag)
    # the argv-level arms refuse EVERY check, G-RESERVE included, and the token cannot mint
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        key = GT.load_key(td / "k" / "gate.key", forbid_under=[PKG])
        bad_argv = without(base, "--w-aux-drv")
        out = td / "refused"
        st = L.run_checks(P, bad_argv, PKG, out)
        ck("S3 run_checks FAILS every check of refe_p7 on the omitted weight", set(st) == set(P["checks"]) and set(st.values()) == {"FAIL"} and "G-RESERVE" in st, str(st))
        ev = GT.read_json(out / "evidence_G-RESERVE.json")
        ck("S3 ... with the profile refusal named in G-RESERVE's evidence too", any("requires --w-aux-drv" in r for r in ev["reasons"]), str(ev["reasons"][:2]))
        v, _, t = L.finalize(P, bad_argv, PKG, out, key)
        ck("S3 ... and finalize refuses to mint", v == "FAIL", str(t["reasons"][:2]))
    # the registry: the head's seven dests are classified, and the table is still EQUAL to the parser's dests
    import train as T
    dests = {x.dest for x in T.build_parser()._actions if x.dest != "help"}
    ck("S3 DVB_KINDS == the parser's dests (68)", set(C.DVB_KINDS) == dests and len(dests) == 68, str((sorted(dests - set(C.DVB_KINDS)), sorted(set(C.DVB_KINDS) - dests))))
    ck("S3 the seven head dests are in the parser AND the registry", all(d in dests and C.DVB_KINDS.get(d) == "built" for d in AUX_DESTS))
    ap = T.build_parser()
    d0 = ap.parse_args(["--targets", "/x"])
    ck("S3 EVERY head default builds NOTHING", (d0.aux_drv, d0.w_aux_drv, d0.aux_drv_tap, d0.aux_drv_depth, d0.aux_drv_feed, d0.aux_drv_labels, d0.aux_drv_shuffle_seed)
       == (False, 0.0, "visual", 2, "off", "", None), str(vars(d0)))
    # a NEW flag with no registry entry is still refused by the gate (CLAUDE.md), now with the head's seven in the table
    orig = T.build_parser

    def bp():
        a_ = orig()
        a_.add_argument("--aux-drv-feeed", type=float, default=0.0)                  # the G-HYG typo class
        return a_
    T.build_parser = bp
    try:
        ev = C.check_dvb(argv_of(P7_ARGS), SimpleNamespace(parser_unknown=[], rc="X", error=None, log="-"), None, L.PROFILES["refe_p7"])
        red("S3 a typo'd head flag has no registry entry -> G-DVB FAILS", ev["reasons"], "aux_drv_feeed")
    finally:
        T.build_parser = orig
    # the trainer's own refusals (the defence behind the profile's)
    import aux_drivable_head as AUX
    ns = lambda **k: SimpleNamespace(**{"aux_drv": True, "w_aux_drv": 0.1, "aux_drv_tap": "visual", "aux_drv_depth": 2, "aux_drv_feed": "off", "aux_drv_labels": "/s", "aux_drv_shuffle_seed": None, **k})  # noqa: E731
    ck("S3 trainer: a consistent head argv is accepted by aux_refusals", AUX.aux_refusals(ns()) == [])
    red("S3 trainer: a head weight without --aux-drv is refused", AUX.aux_refusals(ns(aux_drv=False)), "silently do nothing")
    red("S3 trainer: --aux-drv without labels is refused", AUX.aux_refusals(ns(aux_drv_labels="")), "needs --aux-drv-labels")
    src = (HERE / "train.py").read_text(encoding="utf-8")
    ck("S3 trainer: --aux-drv with weight 0 is refused IN train.py (a grep for the refusal must hit the implementation, not only this test)", "--aux-drv needs --w-aux-drv > 0" in src)
    ck("S3 trainer: the H2 feed is refused IN train.py (the trunk trainer builds H1 only)", "--aux-drv-feed on (H2) is not built in the trunk trainer" in src)


# ======================================================================================================== S4: G-RESERVE
def s_reserve() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rfile = td / "reserve_sessions.json"
        rfile.write_text(json.dumps({"seed": 20261022, "sessions": ["s1", "s2"], "logs": ["l1", "l2", "l3"]}), encoding="utf-8", newline="\n")
        sha = GT.sha256_file(rfile)
        prof = {"name": "t7", "reserve_sha256": sha}
        bank = td / "bank"
        bank.mkdir()
        for k in (0, 1):
            (bank / f"targets_rank{k}.jsonl").write_text("{}\n", encoding="utf-8")
        argv = ["--targets", str(bank), "--metric-cache", "/m", "--aux-drv-labels", "/store", "--images", "/img", "--calib", "/c.json", "--aux-drv"]
        seen = {}

        def audit_ok(reserve, **kw):
            seen["reserve"], seen["kw"] = reserve, kw
            return {"ok": True, "overlap": 0, "note": "fake"}
        ev = C.check_reserve(argv, prof, rfile, audit=audit_ok)
        ck("S4 PASS: registered file + ok True", ev["status"] == "PASS", str(ev["reasons"]))
        ck("S4 the audit received the PARSED reserve", seen["reserve"] == {"seed": 20261022, "sessions": ["s1", "s2"], "logs": ["l1", "l2", "l3"]})
        ck("S4 ... with exactly the five inputs of SPEC_P7 2, as keywords", set(seen["kw"]) == {"bank_files", "metric_cache_dir", "label_store", "image_index", "calib_rows"}, str(sorted(seen["kw"])))
        ck("S4 ... mapped from the LAUNCH argv", seen["kw"]["metric_cache_dir"] == "/m" and seen["kw"]["label_store"] == "/store" and seen["kw"]["image_index"] == "/img"
           and seen["kw"]["calib_rows"] == "/c.json" and [Path(p).name for p in seen["kw"]["bank_files"]] == ["targets_rank0.jsonl", "targets_rank1.jsonl"], str(seen["kw"]))
        ck("S4 the evidence records the reserve sha256 and the audit's scalars", ev["details"]["reserve_sha256"] == sha and ev["details"]["audit"]["overlap"] == 0)
        # ---- the negative twins: each must FAIL, for ITS reason
        red("S4 FAIL: ok False", C.check_reserve(argv, prof, rfile, audit=lambda r, **k: {"ok": False, "reasons": ["1 reserve row in rank-0"]})["reasons"], "did not pass")
        red("S4 FAIL: ok False names the audit's reason", C.check_reserve(argv, prof, rfile, audit=lambda r, **k: {"ok": False, "reasons": ["1 reserve row in rank-0"]})["reasons"], "1 reserve row in rank-0")
        red("S4 FAIL: ok missing", C.check_reserve(argv, prof, rfile, audit=lambda r, **k: {"overlap": 0})["reasons"], "did not pass")
        red("S4 FAIL: ok truthy but not True", C.check_reserve(argv, prof, rfile, audit=lambda r, **k: {"ok": "yes"})["reasons"], "did not pass")
        red("S4 FAIL: audit returns a non-dict", C.check_reserve(argv, prof, rfile, audit=lambda r, **k: True)["reasons"], "did not pass")
        red("S4 FAIL: audit returns None", C.check_reserve(argv, prof, rfile, audit=lambda r, **k: None)["reasons"], "did not pass")

        def boom(r, **k):
            raise ValueError("bank file unreadable")
        red("S4 FAIL: audit raises", C.check_reserve(argv, prof, rfile, audit=boom)["reasons"], "audit_inputs raised ValueError")
        red("S4 FAIL: a different reserve file (wrong manifest sha)", C.check_reserve(argv, {**prof, "reserve_sha256": "0" * 64}, rfile, audit=audit_ok)["reasons"], "not the registered draw")
        red("S4 FAIL: no sha registered in the profile", C.check_reserve(argv, {"name": "t7"}, rfile, audit=audit_ok)["reasons"], "registers no reserve sha256")
        red("S4 FAIL: no file", C.check_reserve(argv, prof, None, audit=audit_ok)["reasons"], "no reserve file")
        red("S4 FAIL: a missing file", C.check_reserve(argv, prof, td / "nope.json", audit=audit_ok)["reasons"], "no reserve file")
        bad = td / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        red("S4 FAIL: an unreadable file", C.check_reserve(argv, {**prof, "reserve_sha256": GT.sha256_file(bad)}, bad, audit=audit_ok)["reasons"], "unreadable")
        ck("S4 a regression arm cannot PASS (rehearsal flag is carried)", C.check_reserve(argv, prof, rfile, arm="x", audit=audit_ok).get("arm") == "x")
        # ---- the LAZY import: absent module FAILS, a module with the registered signature is called, a module without the function FAILS
        saved = sys.modules.get("session_reserve", "ABSENT")
        try:
            sys.modules["session_reserve"] = None                                          # `import session_reserve` raises ImportError
            red("S4 FAIL: session_reserve absent", C.check_reserve(argv, prof, rfile)["reasons"], "not importable")
            fake = type(sys)("session_reserve")
            sys.modules["session_reserve"] = fake
            red("S4 FAIL: session_reserve without audit_inputs", C.check_reserve(argv, prof, rfile)["reasons"], "not importable")
            fake.audit_inputs = lambda reserve, *, bank_files, metric_cache_dir, label_store, image_index, calib_rows: {"ok": True}
            ck("S4 PASS through the lazily imported module (keyword-only signature of the brief)", C.check_reserve(argv, prof, rfile)["status"] == "PASS")
            fake.audit_inputs = lambda reserve, *, bank_files, metric_cache_dir, label_store, image_index: {"ok": True}           # a signature drift: TypeError -> FAIL, never a pass
            red("S4 FAIL: a signature drift is a FAIL", C.check_reserve(argv, prof, rfile)["reasons"], "audit_inputs raised TypeError")
        finally:
            if saved == "ABSENT":
                sys.modules.pop("session_reserve", None)
            else:
                sys.modules["session_reserve"] = saved
    ck("S4 the reserve file is a bound gate input (gate:reserve_json)", L.gate_inputs_of(reserve_json="/r.json") == {"gate:reserve_json": "/r.json"} and L.gate_inputs_of() == {})
    ck("S4 refe_p6 has no G-RESERVE (P6 is untouched)", "G-RESERVE" not in L.PROFILES["refe_p6"]["checks"])
    ck("S4 the CLI flag exists", "--gate-reserve-json" in (HERE / "launch_gate.py").read_text(encoding="utf-8"))


# ======================================================================================================== S5: the judges, from raw fields
def dvb_fixture(**mut):
    """A GREEN capture of a head arm (2 steps x accum 2 = 4 micro-batches) as `aux_dvb_rows` reads it; `mut` replaces / edits one thing."""
    w, accum, steps = mut.pop("w", 0.1), 2, 2
    n = steps * accum
    pv, ps = (100, (1, 7680, 256)), (200, (1, 64, 256))
    a = SimpleNamespace(aux_drv=True, w_aux_drv=w, aux_drv_tap="visual", aux_drv_depth=2, aux_drv_feed="off", aux_drv_labels="/s", aux_drv_shuffle_seed=None)
    b = SimpleNamespace(banks={"TargetBank": [{"args": [], "kwargs": {"aux_labels": "/s", "aux_shuffle_seed": None}}]},
                        aux_facts={"depth": 2, "feed_built": False, "scorer_in_width": 60, "w_aux": w, "n_params": 2440450}, scorer_in_width=60)
    ev = {"event": "aux", "store_sha256": SHA_T, "store_keys": 5, "rows": 6, "rows_labelled": 6, "shuffle_seed": None, "tap": "visual", "depth": 2, "w_aux": w}
    s = SimpleNamespace(args=SimpleNamespace(steps=steps, accum=accum), aux_loss_calls=[{"valid": True, "labelled": 1}] * n, aux_loss_grads=[w / accum] * n,
                        aux_tap_obs=[{"ctx": pv, "scene": ps, "visual": pv, "sctx": pv}] * n, metrics=[ev, {"traj_L1": 1.0, "aux_w": w}, {"traj_L1": 1.0, "aux_w": w}])
    prof = {"name": "t7", "data_flags": ("--aux-drv-labels",), "aux_store_pins": {"sha256": SHA_T, "keys": 5}}
    for k, v in mut.items():
        tgt, _, attr = k.partition("__")
        obj = {"a": a, "b": b, "s": s, "ev": ev, "prof": prof}[tgt]
        if isinstance(obj, dict):
            obj[attr] = v
        else:
            setattr(obj, attr, v)
    return a, b, s, prof


def s_judges() -> None:
    a, b, s, prof = dvb_fixture()
    rows = C.aux_dvb_rows(a, b, s, prof)
    ck("S5 G-DVB rows: the green capture is all-ok", all(r["ok"] for r in rows) and len(rows) >= 18, str([r for r in rows if not r["ok"]][:3]))
    ck("S5 ... and covers every one of the seven levers", {r["dest"] for r in rows} == set(AUX_DESTS), str(sorted({r["dest"] for r in rows} ^ set(AUX_DESTS))))
    # default-off: no head declared, none built -> only the three BUILD rows, all ok; a head built without the flag is a FAIL
    a0 = SimpleNamespace(aux_drv=False, w_aux_drv=0.0, aux_drv_tap="visual", aux_drv_depth=2, aux_drv_feed="off", aux_drv_labels="", aux_drv_shuffle_seed=None)
    b0 = SimpleNamespace(banks={"TargetBank": [{"args": [], "kwargs": {"aux_labels": None, "aux_shuffle_seed": None}}]}, aux_facts=None)
    r0 = C.aux_dvb_rows(a0, b0, None, None)
    ck("S5 G-DVB default-off: three BUILD rows, all ok", len(r0) == 3 and all(r["ok"] for r in r0), str(r0))
    b0.aux_facts = {"depth": 2}
    ck("S5 G-DVB: a head BUILT with the flag off is a FAIL", not all(r["ok"] for r in C.aux_dvb_rows(a0, b0, None, None)))
    b0.aux_facts, b0.banks["TargetBank"][0]["kwargs"]["aux_labels"] = None, "/somewhere"
    ck("S5 G-DVB: a loader handed a store the argv did not declare is a FAIL", not all(r["ok"] for r in C.aux_dvb_rows(a0, b0, None, None)))

    def bad(name, frag, **mut):
        a_, b_, s_, p_ = dvb_fixture(**mut)
        rs = [f"{r['flag']}: {r['how']}" for r in C.aux_dvb_rows(a_, b_, s_, p_) if not r["ok"]]
        red(f"S5 G-DVB red: {name}", rs, frag)
    pv, ps = (100, (1, 7680, 256)), (200, (1, 64, 256))
    bad("flag on, head not built", "constructed iff", b__aux_facts=None)
    bad("depth 1 built", "CrossBlocks", b__aux_facts={"depth": 1, "feed_built": False, "scorer_in_width": 60, "w_aux": 0.1})
    bad("the BUILT model's scorer reads 63 features with the flag off (feed wired behind the head's back)", "score_q_mlp[0].in_features", b__scorer_in_width=63)
    bad("the feed wired with the flag off (scorer width 63)", "scorer's input width", b__aux_facts={"depth": 2, "feed_built": True, "scorer_in_width": 63, "w_aux": 0.1})
    bad("tap = scene_ctx", "storage the head's input IS", s__aux_tap_obs=[{"ctx": ps, "scene": ps, "visual": pv, "sctx": pv}] * 4)
    bad("tap = neither tensor", "storage the head's input IS", s__aux_tap_obs=[{"ctx": (7, (1, 2, 3)), "scene": ps, "visual": pv, "sctx": pv}] * 4)
    bad("scorer reads another context than visual_ctx", "context the scoring decoder reads", s__aux_tap_obs=[{"ctx": pv, "scene": ps, "visual": pv, "sctx": (9, (1, 2, 3))}] * 4)
    bad("the term wired off by a factor (x0.5)", "multiplier measured at the graph", s__aux_loss_grads=[0.025] * 4)
    bad("the term never reaches the loss (no hook fires)", "multiplier measured at the graph", s__aux_loss_grads=[])
    bad("the term reaches only ONE of two sites (2 of 4 micro-batches)", "multiplier measured at the graph", s__aux_loss_grads=[0.05, 0.05])
    bad("a frame without the valid mask", "valid mask", s__aux_loss_calls=[{"valid": False, "labelled": 1}] * 4)
    bad("the loss was computed too few times", "computed once per micro-batch", s__aux_loss_calls=[{"valid": True, "labelled": 1}] * 3)
    bad("logged aux_w differs", "logged aux_w", s__metrics=[{"event": "aux", "store_sha256": SHA_T, "store_keys": 5, "rows": 6, "rows_labelled": 6, "shuffle_seed": None, "tap": "visual", "depth": 2, "w_aux": 0.1},
                                                         {"traj_L1": 1.0, "aux_w": 0.2}])
    good_ev = {"event": "aux", "store_sha256": SHA_T, "store_keys": 5, "rows": 6, "rows_labelled": 6, "shuffle_seed": None, "tap": "visual", "depth": 2, "w_aux": 0.1}
    bad("no `aux` event", "`aux` event", s__metrics=[{"traj_L1": 1.0, "aux_w": 0.1}])
    bad("the store is NOT the registered one (sha256)", "registered store's", s__metrics=[{**good_ev, "store_sha256": "0" * 64}, {"aux_w": 0.1}])
    bad("the FULL store (reserve keys present): wrong key count", "key count", s__metrics=[{**good_ev, "store_keys": 103139}, {"aux_w": 0.1}])
    bad("coverage 99.75 % < 99.9 %", "label coverage", s__metrics=[{**good_ev, "rows": 4000, "rows_labelled": 3990}, {"aux_w": 0.1}])
    bad("derangement seed differs", "derangement seed", s__metrics=[{**good_ev, "shuffle_seed": 3}, {"aux_w": 0.1}])
    bad("the trainer built another tap", "the tap the trainer built", s__metrics=[{**good_ev, "tap": "scene"}, {"aux_w": 0.1}])
    bad("no pins registered in the profile", "registers NO pins", prof__aux_store_pins={"sha256": None, "keys": None})
    bad("the profile does not fingerprint the label store", "PASS token must fingerprint it", prof__data_flags=("--targets",))
    bad("labels declared but loader got none", "TargetBank(aux_labels=)", b__banks={"TargetBank": [{"args": [], "kwargs": {"aux_labels": None, "aux_shuffle_seed": None}}]})

    # ---- judge_aux_live
    def gr(abs_sum=1.0, n=2, n_grad=2, n_req=2, finite=True):
        return {"abs_sum": abs_sum, "finite": finite, "n": n, "n_grad": n_grad, "n_zero": 0, "numel": 10, "n_req": n_req}
    rowsl = [{"step": 1, "aux_bce": math.log(2.0) + 0.01, "aux_huber": 0.2, "aux_w": 0.1}, {"step": 2, "aux_bce": 0.69, "aux_huber": 0.2, "aux_w": 0.1}]

    def live(**mut):
        f = {"built": True, "n_head_params": 2, "grads": [gr(), gr()], "in_optimizer": 2, "adam": {"n": 2, "with_state": 2}, "rows": copy.deepcopy(rowsl)}
        f.update(mut)
        return C.judge_aux_live(f, 0.1)
    ck("S5 G-LIVE head: the green facts are proved", live() == [], str(live()))
    red("S5 G-LIVE red: no head built", live(built=False), "never built")
    red("S5 G-LIVE red: head frozen", live(grads=[gr(n_req=1)]), "FROZEN")
    red("S5 G-LIVE red: a non-finite gradient", live(grads=[gr(finite=False)]), "non-finite")
    red("S5 G-LIVE red: only part of the head has a gradient", live(grads=[gr(n_grad=1)]), "received a gradient")
    red("S5 G-LIVE red: gradient EXACTLY zero on every step", live(grads=[gr(abs_sum=0.0), gr(abs_sum=0.0)]), "EXACTLY zero")
    red("S5 G-LIVE red: no optimiser step observed", live(grads=[]), "no optimiser step")
    red("S5 G-LIVE red: the head's param group is NOT in the optimiser", live(in_optimizer=0), "NOT held by the optimiser")
    red("S5 G-LIVE red: no Adam state after the step (probe_zerograd3)", live(adam={"n": 2, "with_state": 0}), "hold Adam state")
    red("S5 G-LIVE red: aux_bce absent from a row", live(rows=[{"step": 1, "aux_huber": 0.2, "aux_w": 0.1}]), "'aux_bce' ABSENT")
    red("S5 G-LIVE red: aux_huber not finite", live(rows=[{"step": 1, "aux_bce": 0.69, "aux_huber": float("nan"), "aux_w": 0.1}]), "'aux_huber' not finite")
    red("S5 G-LIVE red: the initial BCE is not ~ ln 2", live(rows=[{"step": 1, "aux_bce": 1.4, "aux_huber": 0.2, "aux_w": 0.1}]), "ln 2")
    red("S5 G-LIVE red: the logged weight is not the declared one", live(rows=[{"step": 1, "aux_bce": 0.69, "aux_huber": 0.2, "aux_w": 0.0}]), "aux_w")
    red("S5 G-LIVE red: no rows", live(rows=[]), "no logged step rows")

    # ---- judge_aux_reach
    def grp(v, n=3):
        return {"abs_sum": v, "n": n, "n_none": 0 if v else n}
    groups = {g: grp(1.0) for g in C.AUX_REACH["visual"]}
    groups.update({g: grp(0.0) for g in C.AUX_NO_REACH["visual"]})
    zero = {g: grp(0.0) for g in groups}

    def reach(**mut):
        fr = {"ran": True, "labelled": 1, "tap": "visual", "score_to_head_grads_all_none": True, "declared": {"loss": 0.07, "groups": copy.deepcopy(groups)}, "zero": {"loss": 0.0, "groups": copy.deepcopy(zero)}}
        for k, v in mut.items():
            if k in ("declared_groups", "zero_groups"):
                fr["declared" if k.startswith("d") else "zero"]["groups"].update(v)
            else:
                fr[k] = v
        return C.judge_aux_reach(fr, 0.1, "visual")
    ck("S5 G-LIVE reach: the green probe is proved", reach() == [], str(reach()))
    red("S5 reach red: the probe did not run", C.judge_aux_reach({"ran": False}, 0.1), "could not run")
    red("S5 reach red: no labelled sample", C.judge_aux_reach({"ran": True, "labelled": 0}, 0.1), "no labelled sample")
    for g in C.AUX_REACH["visual"]:
        red(f"S5 reach red: L_aux-only gradient EXACTLY zero on {g}", reach(declared_groups={g: grp(0.0)}), f"EXACTLY zero (or is absent) on {g!r}")
    miss = {"ran": True, "labelled": 1, "tap": "visual", "score_to_head_grads_all_none": True, "declared": {"loss": 0.07, "groups": {k: v for k, v in groups.items() if k != "scene_proj"}},
            "zero": {"loss": 0.0, "groups": copy.deepcopy(zero)}}
    red("S5 reach red: a group missing from the model", C.judge_aux_reach(miss, 0.1, "visual"), "scene_proj")
    for g in ("traj_head", "reg_compress", "dec", "ego_enc", "cmd_embed", "score_head", "score_dec"):
        red(f"S5 reach red: L_aux reaches {g} (a goal / proposal path = a back door)", reach(declared_groups={g: grp(0.3)}), f"reaches {g!r}")
    red("S5 reach red: the weight-0 control reads non-zero (the probe cannot be trusted)", reach(zero_groups={"scene_proj": grp(0.5)}), "weight-0 control")
    red("S5 reach red: an autograd path score -> head (feed off)", reach(score_to_head_grads_all_none=False), "score to the head")
    # the scene tap has its OWN ancestors (registers + reg_compress) and its own zeros
    sg = {g: grp(1.0) for g in C.AUX_REACH["scene"]}
    sg.update({g: grp(0.0) for g in C.AUX_NO_REACH["scene"]})
    frs = {"ran": True, "labelled": 1, "tap": "scene", "score_to_head_grads_all_none": True, "declared": {"loss": 0.07, "groups": sg}, "zero": {"loss": 0.0, "groups": {g: grp(0.0) for g in sg}}}
    ck("S5 reach: the scene tap's probe is proved", C.judge_aux_reach(frs, 0.1, "scene") == [], str(C.judge_aux_reach(frs, 0.1, "scene")))
    red("S5 reach red: the scene tap with reg_compress silent", C.judge_aux_reach({**frs, "declared": {"loss": 0.07, "groups": {**sg, "reg_compress": grp(0.0)}}}, 0.1, "scene"), "reg_compress")


# ======================================================================================================== S6: hop census + source mutation of train.py
def census(src: str) -> dict:
    """The hops of the head in the trunk trainer, each a predicate on the SOURCE.  A hop is proved by what it EMITS (a named call at a named place), and the mutation table below
    deletes each one and demands this census go RED -- an AST census that read 0 suspects on BOTH the fixed and the broken trainer (2026-09-11) is the failure this guards."""
    s = src

    def has(rx):
        return re.search(rx, s) is not None
    ident = s[s.index("IDENTITY_ARGS = ("):s.index("def warm_start")]
    return {
        "parser registers the head's seven flags": "AUX.add_aux_args(ap)" in s,
        "head built ONLY under --aux-drv, from the parsed levers": has(r"if a\.aux_drv:\n\s+aux = AUX\.AuxDrivable\(AUX\.AuxDrivableConfig\.from_args\(a\)\)\.to\(dev\)"),
        "tap hook on scene_proj": "model.scene_proj.register_forward_hook(" in s,
        "the tap is live for ONE forward only": 'aux_cap["on"] = aux is not None' in s and 'aux_cap["on"] = False' in s,
        "two scene_proj outputs per forward or RAISE": 'len(aux_cap["taps"]) != 2' in s,
        "predict reads the DECLARED tap (visual = scene_proj's 2nd output)": 'aux.predict(aux_cap["taps"][1 if a.aux_drv_tap == "visual" else 0])' in s,
        "the loss masks frames without a label": "aux.loss(aux_logit, aux_sdf, aux_lab, aux_valid)" in s,
        "loader tail = the LAST two batch elements": "aux_lab, aux_valid = batch[-2].to(dev, non_blocking=True), batch[-1].to(dev, non_blocking=True)" in s,
        "SITE 1 (online, deferred backward) adds L_aux": has(r'tot_ = aux\.combine\(tot_, P\["l_aux"\]\)[^\n]*\n\s+\(tot_ / a\.accum\)\.backward\(\)'),
        "SITE 2 (plain backward) adds L_aux": has(r"loss = aux\.combine\(loss, l_aux\)[^\n]*\n\s+loss = loss / a\.accum\n\s+loss\.backward\(\)"),
        "EXACTLY two combine sites": len(re.findall(r"aux\.combine\(", s)) == 2,
        "the deferred step carries the head's terms": 'P["l_aux"]' in s and 'P["aux_bce"]' in s and 'P["aux_hub"]' in s and "l_aux, aux_bce, aux_hub = P[" in s,
        "head params join the optimiser BEFORE it is built": "params += aux_params" in s and s.index("params += aux_params") < s.index("opt = torch.optim.AdamW(params"),
        "head grads are clipped with the trunk's": "] + aux_params, 1.0)" in s,
        "the head rides in EVERY checkpoint dict (ckpt_last, snap, model_final), beside `model`": len(re.findall(r"\*\*_aux_ckpt\(\)", s)) == 3,
        "resume LOADS the head (and refuses a checkpoint without it)": 'aux.load_state_dict(st["aux_drv"])' in s and "carries no `aux_drv` head state" in s,
        "identity: the seven dests are identity args": "+ AUX.AUX_IDENTITY_ARGS" in ident,
        "identity: ... with pre-P7 defaults (old checkpoints resume)": all(f'"{k}"' in ident[ident.index("IDENTITY_DEFAULTS"):] for k in AUX_DESTS),
        "identity: the store's CONTENT digest": 'ident["aux_store_sha256"]' in s and '"aux_store_sha256"' in s[s.index("def _identity_diff"):s.index("def _identity_diff") + 1800],
        "refusal: --aux-drv needs a weight > 0 (in the trainer, not only a test)": "--aux-drv needs --w-aux-drv > 0" in s,
        "refusal: the trainer's own head-argv rules run": "AUX.aux_refusals(a)" in s,
        "logs the head's terms and multiplier": all(k in s for k in ('"aux_bce": float(aux_bce)', '"aux_huber": float(aux_hub)', '"aux_w": float(aux.w_aux)')),
        "the `aux` event carries the store sha256 + keys + coverage": '"event": "aux"' in s and '"store_sha256": ds.aux.manifest["store_sha256"]' in s and '"store_keys": ds.aux_cov["store_keys"]' in s
        and '"rows": ds.aux_cov["rows"]' in s and '"rows_labelled": ds.aux_cov["labelled"]' in s,
        "the head is a SIBLING (never a submodule of the model)": not re.search(r"model\.(aux|add_module)", s) and "model.aux" not in s,
        "TargetBank gets the label store from the parsed flag (the training bank AND the --grow rebuild)": s.count("aux_labels=a.aux_drv_labels or None, aux_shuffle_seed=a.aux_drv_shuffle_seed)") == 2,
        "no label-store code is imported at module level": "import aux_label_store" not in s.split("def _open_aux_store")[0],
    }


MUTATIONS = [
    # (name, old, new, hops that must go RED)
    ("delete SITE 1 (the online deferred backward)", '                        tot_ = aux.combine(tot_, P["l_aux"])           # P7.1 SITE 1 of 2: L_aux rides the SAME deferred backward (always multiplied by w_aux)\n', "                        pass\n",
     {"SITE 1 (online, deferred backward) adds L_aux", "EXACTLY two combine sites"}),
    ("delete SITE 2 (the plain backward)", "                    loss = aux.combine(loss, l_aux)                 # P7.1 SITE 2 of 2 (editing one site leaves the other arm head-less: the 'V2' failure)\n", "                    pass\n",
     {"SITE 2 (plain backward) adds L_aux", "EXACTLY two combine sites"}),
    ("SITE 2 adds the term DETACHED", "loss = aux.combine(loss, l_aux)  ", "loss = aux.combine(loss, l_aux.detach())  ", {"SITE 2 (plain backward) adds L_aux"}),
    ("drop the head's params from the optimiser", "        params += aux_params            # P7.1:", "        pass  # P7.1:", {"head params join the optimiser BEFORE it is built"}),
    ("drop the head from the clip", "[p for p in model.parameters() if p.requires_grad] + aux_params, 1.0)", "[p for p in model.parameters() if p.requires_grad], 1.0)", {"head grads are clipped with the trunk's"}),
    ("tap the SCENE context (index 0) always", 'aux.predict(aux_cap["taps"][1 if a.aux_drv_tap == "visual" else 0])', 'aux.predict(aux_cap["taps"][0])', {"predict reads the DECLARED tap (visual = scene_proj's 2nd output)"}),
    ("lose the valid mask", "aux.loss(aux_logit, aux_sdf, aux_lab, aux_valid)", "aux.loss(aux_logit, aux_sdf, aux_lab, None)", {"the loss masks frames without a label"}),
    ("lose the tap's two-output check", '                    if len(aux_cap["taps"]) != 2:\n', "                    if False:\n", {"two scene_proj outputs per forward or RAISE"}),
    ("build the head without --aux-drv", "    if a.aux_drv:\n        aux = AUX.AuxDrivable(", "    if True:\n        aux = AUX.AuxDrivable(", {"head built ONLY under --aux-drv, from the parsed levers"}),
    ("forget the head's flags in the parser", "    AUX.add_aux_args(ap)", "    pass", {"parser registers the head's seven flags"}),
    ("forget the head in the snapshot checkpoint", '"meta": dict(meta, epoch=epoch, step=step), "saved_at": _now(), **_aux_ckpt()}', '"meta": dict(meta, epoch=epoch, step=step), "saved_at": _now()}', {"the head rides in EVERY checkpoint dict (ckpt_last, snap, model_final), beside `model`"}),
    ("resume forgets the head", '                aux.load_state_dict(st["aux_drv"])', "                pass", {"resume LOADS the head (and refuses a checkpoint without it)"}),
    ("identity forgets the head's levers", " + AUX.AUX_IDENTITY_ARGS      # P7.1:", "      # P7.1:", {"identity: the seven dests are identity args"}),
    ("drop the weight refusal", '        bad_aux.append(f"--aux-drv needs --w-aux-drv > 0 (got {a.w_aux_drv}):', '        bad_aux.append(f"(dropped) (got {a.w_aux_drv}):', {"refusal: --aux-drv needs a weight > 0 (in the trainer, not only a test)"}),
    ("lose the logged multiplier", ', "aux_w": float(aux.w_aux)}', "}", {"logs the head's terms and multiplier"}),
    ("lose the `aux` event's coverage", '"rows_labelled": ds.aux_cov["labelled"]', '"rows_labelled": 0', {"the `aux` event carries the store sha256 + keys + coverage"}),
    ("the training loader gets no label store", "online=online,\n                    field_labels=a.field_labels, field_shuffle_seed=a.field_shuffle_seed,\n                    aux_labels=a.aux_drv_labels or None,",
     "online=online,\n                    field_labels=a.field_labels, field_shuffle_seed=a.field_shuffle_seed,\n                    aux_labels=None,",
     {"TargetBank gets the label store from the parsed flag (the training bank AND the --grow rebuild)"}),
]


def s_census() -> None:
    src = (HERE / "train.py").read_text(encoding="utf-8")
    base = census(src)
    ck("S6 every hop of the unmutated trainer is present", all(base.values()), str([k for k, v in base.items() if not v]))
    ck("S6 the census has a hop per claim", len(base) >= 24, str(len(base)))
    for name, old, new, must in MUTATIONS:
        cnt = src.count(old)
        if not ck(f"S6 mutation anchor occurs EXACTLY once: {name}", cnt == 1, f"{cnt} occurrences (a vacuous or ambiguous edit)"):
            continue
        msrc = src.replace(old, new)
        try:
            compile(msrc, "train.py(mutated)", "exec")
        except SyntaxError as e:
            ck(f"S6 the mutated trainer still PARSES: {name}", False, f"{e} -- a mutation that does not parse is red for the wrong reason")
            continue
        res = census(msrc)
        went_red = {k for k, v in res.items() if not v}
        ck(f"S6 RED when mutated: {name}", must <= went_red, f"expected {sorted(must)} red; got {sorted(went_red)}")
    # the head's own default-off bit-identity at the SOURCE level: every addition is guarded (no head => aux None => the ops that run are the ops that ran)
    guard = [m.start() for m in re.finditer(r"aux\.(predict|loss|combine)\(", src)]
    ck("S6 every call into the head sits under an `aux is not None` / `--aux-drv` guard", all(re.search(r"aux is not None|a\.aux_drv", src[max(0, g - 700):g]) for g in guard), str(len(guard)))
    # the frozen pre-edit trainer is the oracle of the default-off identity: its digest is pinned (CRLF-normalised: an export may have rewritten line endings)
    fx = HERE / "fixtures" / "train_pre_p7_head.py.txt"
    raw = fx.read_bytes().replace(b"\r\n", b"\n") if fx.is_file() else b""
    ck("S6 the frozen pre-P7 trainer fixture is the pinned blob", hashlib.sha256(raw).hexdigest() == PRE_P7_TRAIN_SHA256, hashlib.sha256(raw).hexdigest())
    # the new trainer differs from the frozen one by the head's hunks, WHOLE and ONLY: every changed hunk (added or removed lines) mentions the head.  A hunk that does not is an unrelated edit
    # to a file P6 shares, which this deliverable is not allowed to make.
    import difflib
    old_l, new_l = raw.decode("utf-8").splitlines(), src.splitlines()
    sm = difflib.SequenceMatcher(None, old_l, new_l, autojunk=False)
    hunks = [(tag, i1, i2, j1, j2) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
    stray = [(i1 + 1, "\n".join(old_l[i1:i2] + new_l[j1:j2])[:160]) for tag, i1, i2, j1, j2 in hunks if not re.search(r"aux|AUX|P7\.1", "\n".join(old_l[i1:i2] + new_l[j1:j2]))]
    ck("S6 every changed hunk of train.py relative to the frozen tip mentions the head (nothing else moved)", not stray and len(hunks) >= 20, f"{len(hunks)} hunks; unrelated: {stray[:3]}")


# ======================================================================================================== --full / --mutations: real trainer runs (CPU, tiny rehearsal bank)
def make_mini_store(out: Path, bank_dir: str) -> dict:
    """A REAL (tiny) label store, written with aux_label_store's own writer and finalised by its own `finalize`, keyed by the rehearsal bank's (log_name, token) rows, so the trainer opens it through
    the SAME `AuxLabelSource` path as the pinned store (manifest self-consistency, shard hashes, raster spec)."""
    import numpy as np
    import aux_label_store as ALS
    by_log: dict = {}
    for f in sorted(Path(bank_dir).glob("targets_rank*.jsonl")):
        for ln in f.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                by_log.setdefault(r["log_name"], set()).add(r["token"])
    yy = (-32.0 + 0.5 * (np.arange(ALS.NY) + 0.5))[None, :]
    for lg, toks in sorted(by_log.items()):
        items = []
        for k, tk in enumerate(sorted(toks)):
            # a corridor along the ego's x axis, + inside (0.1 m units), +-11 m wide so ~35 % of cells are positive: the REAL store's mean positive share is 0.38 (MEASURED, 40 shards, 2,144 labels:
            # raw/scratch/store_pos_fraction.py).  The initial BCE of the seed-0 head is ln 2 only for balanced labels (its common-mode logit is about -0.43: raw/scratch/logit_probe.py), so a
            # fixture with ~10 % positives would sit at 0.54 and trip the SPEC's ln 2 +- 0.15 band for a reason that is the fixture's, not the wiring's.
            s = np.clip(np.round((11.0 + 0.5 * (k % 3) - np.abs(yy)) / 0.1), -127, 127).astype(np.int8)
            sdf = np.broadcast_to(s, (ALS.NX, ALS.NY)).copy()
            items.append(dict(t=tk, blob=ALS.encode_sdf(sdf), s="bank", p=[0.0, 0.0, 0.0], q=[int((sdf > 0).sum()), int(sdf.size)]))
        ALS.write_shard(Path(out) / "shards" / f"{lg}.aux1", lg, "test-map", items)
    man = ALS.finalize(out)
    return {"path": str(out), "sha256": man["store_sha256"], "keys": man["counts"]["stored_keys"]}


HEAD_FLAGS = ["--aux-drv", "--w-aux-drv", "0.1", "--aux-drv-tap", "visual", "--aux-drv-depth", "2", "--aux-drv-feed", "off"]


def rehearsal_profile(store: dict, *, require_weight: bool = True) -> dict:
    """The rehearsal profile (P2-shaped, CPU, synthetic images, 1 camera) + the head: what a P7 launch asserts, at rehearsal scale.  Pins are the mini store's (REGISTERED before the run)."""
    base = L.PROFILES["refe_rehearsal"]
    req = {**base["required_flags"], "--aux-drv": None, "--aux-drv-labels": None}
    if require_weight:
        req.update({"--w-aux-drv": "0.1", "--aux-drv-tap": "visual", "--aux-drv-depth": "2", "--aux-drv-feed": "off"})
    return {**base, "name": "refe_rehearsal_p7", "data_flags": base["data_flags"] + ("--aux-drv-labels",), "required_flags": req,
            "live_terms": ("traj_L1", "score", "aux_bce", "aux_huber"), "aux_store_pins": {"sha256": store["sha256"], "keys": store["keys"]},
            "forbidden_flags": base["forbidden_flags"] + ("--aux-drv-shuffle-seed",)}


def run_trainer_module(mod, argv: list, label: str) -> dict:
    """Run `mod.run` in-process; -> metrics rows (time-like keys dropped), model_final tensors digest, ckpt_last parts."""
    import torch
    ap = mod.build_parser()
    a = ap.parse_args(argv)
    rc = mod.run(a)
    out = Path(a.out)
    rows = []
    for ln in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(ln)
        if "traj_L1" in r:
            rows.append({k: v for k, v in r.items() if not re.search(r"elapsed|_s$|time|saved_at|^at$|cuda_max_mem", k)})
    fin = torch.load(out / "model_final.pt", map_location="cpu", weights_only=False)
    last = torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False)
    return {"rc": rc, "rows": rows, "model": C._digest_sd(fin["model"]), "opt": C._opt_digest(last["opt"]), "state": {k: v for k, v in last["state"].items() if k != "elapsed_s"}, "identity": last["identity"],
            "keys_final": sorted(fin), "keys_last": sorted(last), "label": label}


def load_frozen_trainer(tmp: Path):
    import importlib.util
    fx = HERE / "fixtures" / "train_pre_p7_head.py.txt"
    p = tmp / "train_pre_p7.py"
    p.write_bytes(fx.read_bytes().replace(b"\r\n", b"\n"))
    spec = importlib.util.spec_from_file_location("train_pre_p7", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_pre_p7"] = mod
    spec.loader.exec_module(mod)
    return mod


def full() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import gate_fixture as F
    import train as T
    root = Path(tempfile.mkdtemp(prefix="p7head_"))
    try:
        fx = F.tiny_p2_fixture(str(root))
        store = make_mini_store(root / "aux_store", fx["bank"])
        ck("F0 the mini store holds every bank row's key", store["keys"] == len({(k[0], k[1]) for k in fx["keys"]}), str(store))
        base_argv = F.rehearsal_argv(fx, str(root / "out_x"), steps=4) + ["--ckpt-every-steps", "2"]

        def argv_for(out):
            return C._set(list(base_argv), "--out", str(out))
        # ---------------------------------------------------------------- F1 default-off bit-identity
        frozen = load_frozen_trainer(root)
        r_old1 = run_trainer_module(frozen, argv_for(root / "old1"), "old#1")
        r_old2 = run_trainer_module(frozen, argv_for(root / "old2"), "old#2")
        r_new = run_trainer_module(T, argv_for(root / "new"), "new, flag absent")
        ck("F1 control: the instrument is deterministic (frozen trainer run twice is bit-identical)", (r_old1["rows"], r_old1["model"], r_old1["opt"]) == (r_old2["rows"], r_old2["model"], r_old2["opt"]),
           "the comparison below could not be judged")
        ck("F1 the frozen run logged steps", len(r_old1["rows"]) == 4 and r_old1["rc"] == 0)
        ck("F1 default-off: step rows are bit-identical to the frozen trainer's (every loss, lr, winner count)", r_new["rows"] == r_old1["rows"], str((r_new["rows"][:1], r_old1["rows"][:1])))
        ck("F1 default-off: model_final tensors are bit-identical", r_new["model"] == r_old1["model"] and len(r_new["model"]) > 100)
        ck("F1 default-off: AdamW state and the data position are bit-identical", r_new["opt"] == r_old1["opt"] and r_new["state"] == r_old1["state"])
        ck("F1 default-off: the checkpoints carry EXACTLY the frozen trainer's top-level keys (no `aux_drv`)", r_new["keys_final"] == r_old1["keys_final"] and r_new["keys_last"] == r_old1["keys_last"]
           and "aux_drv" not in r_new["keys_last"], str((r_new["keys_last"], r_old1["keys_last"])))
        ck("F1 a pre-P7 checkpoint's identity resumes under the new trainer (IDENTITY_DEFAULTS): no difference", T._identity_diff(r_old1["identity"], r_new["identity"]) == [], str(T._identity_diff(r_old1["identity"], r_new["identity"])))
        # the control that the comparison CAN go red: the same run WITH the head differs
        argv_h = argv_for(root / "head") + ["--aux-drv-labels", store["path"]] + HEAD_FLAGS
        r_head = run_trainer_module(T, argv_h, "new, head on")
        ck("F1 control: with the head ON the weights differ from the frozen trainer's (the comparison can go red)", r_head["model"] != r_old1["model"])
        ck("F1 ... and the head arm's checkpoint carries the head beside `model`", "aux_drv" in r_head["keys_last"] and "aux_drv" in r_head["keys_final"])
        ck("F1 ... and its step rows carry the head's terms", all(k in r_head["rows"][0] for k in ("aux_bce", "aux_huber", "aux_L", "aux_w")) and abs(r_head["rows"][0]["aux_w"] - 0.1) < 1e-12)
        ck("F1 ... and a head run's identity refuses a head-less resume (and vice versa)", T._identity_diff(r_old1["identity"], r_head["identity"]) != [] and T._identity_diff(r_head["identity"], r_old1["identity"]) != [])
        # ---------------------------------------------------------------- F2 the head arm through the REAL gate
        prof = rehearsal_profile(store)
        argv = F.rehearsal_argv(fx, str(root / "launch_out")) + ["--aux-drv-labels", store["path"]] + HEAD_FLAGS
        ck("F2 the rehearsal argv is clean under the head profile", L.profile_refusals(prof, argv) == [], str(L.profile_refusals(prof, argv)))
        out = root / "clean"
        st = L.run_checks(prof, argv, PKG, out, only={"G-DVB", "G-LIVE", "G-EVAL", "G-CKPT"})
        ck("F2 G-DVB, G-LIVE, G-EVAL, G-CKPT all PASS on the real trainer with the head", all(st.get(c) == "PASS" for c in ("G-DVB", "G-LIVE", "G-EVAL", "G-CKPT")), str(st))
        dvb = GT.read_json(out / "evidence_G-DVB.json")["details"]
        aux_rows = [r for r in dvb["rows"] if r["dest"] in AUX_DESTS]
        ck("F2 G-DVB observed the head's rows (>= 18) and every one is ok", len(aux_rows) >= 18 and all(r["ok"] for r in aux_rows), str([r for r in aux_rows if not r["ok"]][:2]))
        mult = [r for r in aux_rows if "multiplier measured at the graph" in r["how"]]
        ck("F2 the multiplier in the loss, measured at the graph, is the declared 0.1 on every micro-batch", len(mult) == 1 and mult[0]["ok"], str(mult))
        live = GT.read_json(out / "evidence_G-LIVE.json")["details"]
        ck("F2 G-LIVE reached the head's probes (details carry aux_reach and aux_live)", "aux_reach" in live and "aux_live" in live, str(GT.read_json(out / "evidence_G-LIVE.json")["reasons"][:2]))
        gr = live.get("aux_reach") or {"declared": {"groups": {}}, "zero": {"groups": {}, "loss": None}, "score_to_head_grads_all_none": None}
        ck("F2 G-LIVE: the L_aux-only gradient reaches the head, LoRA, scene_proj, pos3d_mlp ...", all(gr["declared"]["groups"].get(g, {"abs_sum": 0.0})["abs_sum"] > 0 for g in C.AUX_REACH["visual"]), str({g: gr["declared"]["groups"].get(g) for g in C.AUX_REACH["visual"]}))
        ck("F2 ... EXACTLY 0.0 on traj_head / reg_compress / dec / the scorer", all(gr["declared"]["groups"][g]["abs_sum"] == 0.0 for g in C.AUX_NO_REACH["visual"] if g in gr["declared"]["groups"]))
        ck("F2 ... and the weight-0 control reads EXACTLY 0.0 on every group", bool(gr["zero"]["groups"]) and all(v["abs_sum"] == 0.0 for v in gr["zero"]["groups"].values()) and gr["zero"]["loss"] == 0.0)
        ck("F2 ... with no autograd path from the score to the head", gr["score_to_head_grads_all_none"] is True)
        al = live.get("aux_live") or {"in_optimizer": -1, "n_head_params": -2, "adam": {"with_state": -3}, "grad_abs_sum_max": None}
        ck("F2 G-LIVE: the head is held by the optimiser, holds Adam state, and its gradient is non-zero", al["in_optimizer"] == al["n_head_params"] == al["adam"]["with_state"] and (al["grad_abs_sum_max"] or 0) > 0, str(al))
        ev_e = GT.read_json(out / "evidence_G-EVAL.json")["details"]
        ck("F2 G-EVAL: the head is beside `model` in model_final.pt and the planner's load is bit-identical", ev_e["aux_in_model_final"] == {"aux_drv_beside_model": True, "head_keys_inside_model": []} and ev_e["tensors_not_bit_identical"] == 0, str(ev_e.get("aux_in_model_final")))
        ev_c = GT.read_json(out / "evidence_G-CKPT.json")["details"]
        ck("F2 G-CKPT: the head's weights (and Adam moments) round-trip bit for bit through a halt + resume", ev_c.get("aux_state_equal") is True and ev_c["optimizer_state_equal"] is True and ev_c["model_tensors_differing"] == 0, str(ev_c))
        # ---------------------------------------------------------------- F3 the gate-instrument arms
        arms = [("G-LIVE", "aux_weight_zero", "EXACTLY zero"), ("G-DVB", "aux_weight_off", "multiplier measured at the graph"), ("G-DVB", "aux_term_detached", "multiplier measured at the graph"),
                ("G-CKPT", "resume_drops_aux", "head's weights")]
        for chk, arm, frag in arms:
            o2 = root / f"arm_{arm}"
            s2 = L.run_checks(prof, argv, PKG, o2, only={chk}, arm=arm)
            ev = GT.read_json(o2 / f"evidence_{chk}.json")
            ck(f"F3 arm `{arm}` makes {chk} FAIL for its reason", s2[chk] == "FAIL" and frag in " ".join(ev["reasons"]) and ev["arm"] == arm, str(ev["reasons"][:2]))
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---- the mutation arms through the real gate, each on a temp COPY of this tree (nothing in the real tree is touched)
def driver(spec_path: str) -> int:
    """Runs INSIDE a temp copy of the tree (`--driver spec.json`): builds the rehearsal profile and runs the named checks; prints one JSON line {check: [status, reasons]}."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    prof = rehearsal_profile(spec["store"], require_weight=spec.get("require_weight", True))
    out = Path(spec["out"])
    argv = spec["argv"]
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):
        st = L.run_checks(prof, argv, PKG, out, only=set(spec["checks"]))
    res = {}
    for c in spec["checks"]:
        p = out / f"evidence_{c}.json"
        ev = GT.read_json(p) if p.is_file() else {"status": st.get(c), "reasons": []}
        res[c] = [ev["status"], ev["reasons"][:3]]
    print("ZZDRIVER " + json.dumps(res))
    return 0


def run_arm(tree: Path, spec: dict) -> dict:
    sp = tree / "spec.json"
    sp.write_text(json.dumps(spec), encoding="utf-8")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    env.pop(C.ARM_ENV, None)
    r = subprocess.run([sys.executable, str(tree / "refe" / "test_p7_head_wiring.py"), "--driver", str(sp)], cwd=str(tree / "refe"), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("ZZDRIVER ")), None)
    if line is None:
        return {"_error": (r.stderr or r.stdout)[-400:]}
    return json.loads(line[len("ZZDRIVER "):])


def mutations() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import gate_fixture as F
    root = Path(tempfile.mkdtemp(prefix="p7mut_"))
    try:
        fx = F.tiny_p2_fixture(str(root))
        store = make_mini_store(root / "aux_store", fx["bank"])
        argv_ok = F.rehearsal_argv(fx, str(root / "launch_out")) + ["--aux-drv-labels", store["path"]] + HEAD_FLAGS
        src = (HERE / "train.py").read_text(encoding="utf-8")

        def tree_with(name, old=None, new=None):
            t = root / f"tree_{name}"
            shutil.copytree(HERE, t / "refe", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            if old is not None:
                s = (t / "refe" / "train.py").read_text(encoding="utf-8")
                assert s.count(old) == 1, f"{name}: mutation anchor occurs {s.count(old)} times"
                (t / "refe" / "train.py").write_text(s.replace(old, new), encoding="utf-8", newline="\n")
            return t
        site2 = "                    loss = aux.combine(loss, l_aux)                 # P7.1 SITE 2 of 2 (editing one site leaves the other arm head-less: the 'V2' failure)\n"
        assert src.count(site2) == 1
        arms = [
            # (label, tree mutation, argv, required profile weight, checks, {check: expected reason fragment})
            ("CONTROL: the unmutated trainer", None, argv_ok, True, ["G-DVB", "G-LIVE"], {}),
            ("M1 delete the line that adds L_aux to the total loss (plain backward site)", (site2, "                    pass\n"), argv_ok, True, ["G-DVB", "G-LIVE"],
             {"G-DVB": "multiplier measured at the graph", "G-LIVE": "EXACTLY zero"}),
            ("M2 default weight 0.0, --w-aux-drv omitted from the argv (profile requires it)", None, C._drop(list(argv_ok), "--w-aux-drv", 1), True, ["G-DVB", "G-LIVE"],
             {"G-DVB": "requires --w-aux-drv", "G-LIVE": "requires --w-aux-drv"}),
            ("M2b ... and on a profile WITHOUT that requirement the trainer's own refusal stops it", None, C._drop(list(argv_ok), "--w-aux-drv", 1), False, ["G-DVB", "G-LIVE"],
             {"G-DVB": "did not reach the optimiser", "G-LIVE": "did not finish"}),
            ("M3 drop the head's param group from the optimiser", ("        params += aux_params            # P7.1:", "        pass  # P7.1:"), argv_ok, True, ["G-LIVE"],
             {"G-LIVE": "NOT held by the optimiser"}),
            ("M4 tap scene_ctx instead of visual_ctx", ('aux.predict(aux_cap["taps"][1 if a.aux_drv_tap == "visual" else 0])', 'aux.predict(aux_cap["taps"][0])'), argv_ok, True, ["G-DVB"],
             {"G-DVB": "storage the head's input IS"}),
            ("M5 flag on, head not built", ("    if a.aux_drv:\n        aux = AUX.AuxDrivable(", "    if False:\n        aux = AUX.AuxDrivable("), argv_ok, True, ["G-DVB"],
             {"G-DVB": "constructed iff"}),
            ("M6 build 1 block instead of 2", ("aux = AUX.AuxDrivable(AUX.AuxDrivableConfig.from_args(a)).to(dev)",
                                               "aux = AUX.AuxDrivable(__import__('dataclasses').replace(AUX.AuxDrivableConfig.from_args(a), depth=1)).to(dev)"), argv_ok, True, ["G-DVB"],
             {"G-DVB": "CrossBlocks"}),
        ]
        print("ZZP7HEAD mutation table (each arm: a real gate run on a temp copy of the tree; RED = the check FAILS with the named reason)")
        for label, mut, argv, req_w, checks, want in arms:
            t = tree_with(re.sub(r"\W+", "_", label[:12]), *(mut or (None, None)))
            res = run_arm(t, {"store": store, "out": str(t / "gate_out"), "argv": argv, "checks": checks, "require_weight": req_w})
            if "_error" in res:
                ck(f"MUT {label}", False, res["_error"])
                print(f"  ERROR  {label}: {res['_error'][-200:]}")
                continue
            if not want:
                good = all(v[0] == "PASS" for v in res.values())
                ck(f"MUT {label}", good, str(res))
                print(f"  {'PASS ' if good else 'FAIL?'}  {label}: " + ", ".join(f"{c} {v[0]}" for c, v in res.items()))
                continue
            good = all(res[c][0] == "FAIL" and frag in " ".join(res[c][1]) for c, frag in want.items())
            ck(f"MUT {label}", good, str(res))
            print(f"  {'RED  ' if good else 'NOT RED'}  {label}: " + "; ".join(f"{c} {res[c][0]} ({(res[c][1] or [''])[0][:110]})" for c in want))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    if "--driver" in sys.argv:
        return driver(sys.argv[sys.argv.index("--driver") + 1])
    s_profiles()
    s_argv()
    s_reserve()
    s_judges()
    s_census()
    if "--mutations" in sys.argv:
        mutations()
    if "--full" in sys.argv:
        full()
    if FAILS:
        for f in FAILS:
            print("ZZP7HEAD_RED", f)
        print(f"ZZP7HEAD_FAIL {len(FAILS)} failing, {N} passing")
        return 1
    print(f"ZZP7HEAD_PASS {N}" if "--full" not in sys.argv else f"ZZP7HEAD_PASS {N}\nZZP7HEAD_FULL {N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
