#!/usr/bin/env python3
"""Tests of the Step B CHAIN's gating logic, on PLANTED inputs only (a planted universe in tmp_path, a fake runner that records every child it would have started).  NEVER the real carve B labels, never a
real Step B output.  Run:  CUDA_VISIBLE_DEVICES=-1 python -m pytest test_stepB_chain.py -q

Covered (each with its deliberate-regression arm):  the chain REFUSES without the seal / on an edited seal / on a pin that differs / without L2-FRESH's seal;  REFUSES on a token mismatch (and starts no child);
STOPS on a failing L2-FRESH gate, on a failing G-REACH-STEPB before ANY search, on a label identity failure, with no analysis;  WAITS (never matches a line that merely mentions a marker);  RESUMES;  and its happy
path runs the children in the registered order and prints the verdict line.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stepB  # noqa: E402
import stepB_chain as C  # noqa: E402

CARVE_FILE = "C:/Users/Admin/par18/l2fresh/raw/fresh_sample_token_disjoint.json"
pytestmark = pytest.mark.skipif(not os.path.exists(CARVE_FILE), reason="the sealed carve's token list (l2fresh/raw) is not on this box")
RAW = os.path.normpath(os.path.join(HERE, "..", "raw"))


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def time(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


class FakeRunner:
    """records every child; emulates the artifacts the real children write, so the chain's own validation is what is under test"""

    def __init__(self, uni, outcome="O1_PASS", gate_fail=(), search_rc=0, label_rc=0, stamp_without_pass=False, no_verdict=False):
        self.u, self.outcome, self.gate_fail, self.search_rc, self.label_rc = uni, outcome, set(gate_fail), search_rc, label_rc
        self.stamp_without_pass, self.no_verdict = stamp_without_pass, no_verdict
        self.calls = []

    @staticmethod
    def arg(argv, flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default

    def kind(self, argv):
        script = os.path.basename(argv[2])
        return script if script != "stepB.py" else f"stepB.py {argv[3]}"

    def __call__(self, argv, env, log_path, cwd=None, on_pid=None):
        k = self.kind(argv)
        self.calls.append((k, argv, env))
        u = self.u
        if k == "g_reach_stepb.py":
            head_sha = self.arg(argv, "--head-sha256")
            s = "s0" if head_sha == u["pins"]["head_s0_sha256"] else "s1"
            if s in self.gate_fail:
                return 1
            if self.stamp_without_pass:
                return 0
            out = self.arg(argv, "--out-dir")
            os.makedirs(out, exist_ok=True)
            st = {"schema": "tanitad.g_reach_stepb/1", "verdict": "PASS", "head_sha256": head_sha, "ckpt_sha256": u["pins"]["ckpt_sha256"], "config_sha256": u["pins"]["config_sha256"],
                  "bundle_sha256": C.sha256_file(u["cfg"].bundle), "code_sha256": stepB.code_shas(), "l1_pins": dict(stepB.L1_PINS)}
            json.dump(st, open(f"{out}/G-REACH-STEPB_PASS_{head_sha[:12]}.json", "w"))
            return 0
        if k == "stepB.py search":
            return self.search_rc
        if k == "stepB.py merge":
            arm, out, n = self.arg(argv, "--arm"), self.arg(argv, "--out"), int(self.arg(argv, "--n"))
            d = self.arg(argv, "--out-dir")
            head = u["pins"]["head_s0_sha256"] if d.endswith("L2s0") else (u["pins"]["head_s1_sha256"] if d.endswith("L2s1") else None)
            meta = {"arm": arm, "head_sha256": head, "ckpt_sha256": u["pins"]["ckpt_sha256"], "config_sha256": u["pins"]["config_sha256"], "bundle_sha256": C.sha256_file(u["cfg"].bundle),
                    "data_clause": {"carve_sha256": u["pins"]["carve_sha256"]}, "n_tokens": n}
            np.savez(out, meta=np.array(json.dumps(meta)), token=np.array(u["tokens"]), bundle_index=np.arange(n))
            return 0
        if k == "label_stepB.py":
            if self.label_rc:
                return self.label_rc
            sp, out = self.arg(argv, "--search"), self.arg(argv, "--out")
            np.savez(out, search_sha256=np.array(C.sha256_file(sp)), identity=np.array(json.dumps({"passed": True, "control_ok": True})))
            return 0
        if k == "analyse_stepB.py":
            res = {"verdict": None if self.no_verdict else {"combined": self.outcome, "rule": "worst outcome over head seeds", "next_arm": "NAMED NEXT ARM", "licensing": "licenses nothing alone",
                                                              "stamps": "SEEN-SCENE, LOG-SHARED, SELECTION", "STRATEGIC": "PAUSED (PI ruling)",
                                                              "per_head": {h: {"outcome": self.outcome, "dPDMS_points": 0.5, "ci": [0.1, 0.9], "seed_range_points": 0.2, "moved_share": 0.4} for h in ("L2s0", "L2s1")}}}
            json.dump(res, open(self.arg(argv, "--out"), "w"))
            return 0
        raise AssertionError(f"unexpected child {argv}")

    def kinds(self):
        return [k for k, _, _ in self.calls]


B_LOG_DONE = ("CHAIN_START 19.10.2026 16:03:40,57 TAG=B N=400 SAMPLE=x W=y\nSTEP3_RC=0   (step 3 log: z; markers ZZGPUWAIT ZZGPULOCKED ZZBUNDLE_OK ZZGPURELEASED)\nSTEP3_DONE 16:40:00,00\n"
              "STEP6_DONE 17:20:00,00\nCHAIN_DONE_NO_READ 17:40:00,00   (step 7 = analyse_fresh.py is SEALED-ONLY)\n")


def write_gates(bdir, **over):
    g = {"step3.log": "ZZGPUWAIT x\nZZGPULOCKED\nZZBUNDLE_OK 400 612 s\nZZGPURELEASED\n",
         "labels.npz.g1.json": {"passed": True, "control_ok": True, "max_abs_diff": 0.0},
         "scores.npz.gates.json": {"G2a": [{"passed": True}] * 3, "G2c": [{"premise_true": True}] * 3, "G3": [{"passed": True}] * 3, "G2b_cross_device": [{"pick_agreement": 0.998}]},
         "fresh_ep_scores.json": {"pass": True}}
    g.update(over)
    for name, v in g.items():
        with open(f"{bdir}/{name}", "w", encoding="utf-8", newline="\n") as f:
            f.write(v if isinstance(v, str) else json.dumps(v))


@pytest.fixture
def uni(tmp_path):
    """a planted universe: planted heads / ckpt / configs (the real C8 configs), a bundle holding the REAL carve's token list (a list of tokens, no label), planted gates, planted L2-FRESH seal"""
    toks = json.load(open(CARVE_FILE, encoding="utf-8"))["tokens"]
    pkg, fresh, bdir, heads = (tmp_path / n for n in ("pkg", "fresh", "B", "heads"))
    for d in (pkg / "raw", fresh, bdir / "bundle", heads):
        d.mkdir(parents=True)
    shutil.copy(f"{RAW}/stepB_config.json", pkg / "raw" / "stepB_config.json")
    shutil.copy(f"{RAW}/stepB_config_O4.json", pkg / "raw" / "stepB_config_O4.json")
    for s, seed in (("s0", 1), ("s1", 2)):
        np.savez(heads / f"final_head_{s}.npz", weight=np.full((6, 4), seed, np.float32), bias=np.zeros(6, np.float32))
    json.dump({"heads": {s: {"sha256": C.sha256_file(str(heads / f"final_head_{s}.npz"))} for s in ("s0", "s1")}}, open(heads / "final_heads.json", "w"))
    ck = tmp_path / "model_final.pt"
    ck.write_bytes(b"planted checkpoint")
    np.savez(bdir / "bundle" / "bundle.npz", token=np.array(toks), log=np.array(["l"] * 400), pick=np.zeros(400, np.int64))
    (bdir / "bundle" / "ctx.npy").write_bytes(b"planted ctx bytes" * 10)
    json.dump({"n": 400, "shape": [400, 3, 2], "ctx_sha256": C.sha256_file(str(bdir / "bundle" / "ctx.npy")), "tokens_sha256": stepB.tokens_sha256(toks)}, open(bdir / "bundle" / "ctx.json", "w"))
    write_gates(str(bdir))
    (fresh / "chain_B.log").write_text(B_LOG_DONE)
    cfg = C.Cfg(pkg=str(pkg), fresh=str(fresh), b_dir=str(bdir), heads_dir=str(heads), ckpt=str(ck), refe_dir="unused", plan="unused", py_cpu="PY", py_label="PYL", poll_s=60.0, max_wait_s=600.0, heartbeat_s=120.0)
    pins = C.collect_pins(cfg, ckpt_sha256=C.sha256_file(str(ck)))
    fresh_text = f"# planted L2-FRESH amendment\nStatus: SEALED\ncarve {pins['carve_sha256']}\nheads {pins['head_s0_sha256']} {pins['head_s1_sha256']}\n"
    (fresh / "PREREG_L2_FRESH.md").write_text(fresh_text)
    import hashlib
    (fresh / "PREREG_L2_FRESH.sha256").write_text(hashlib.sha256(fresh_text.encode()).hexdigest() + "  PREREG_L2_FRESH.md\n")
    u = {"cfg": cfg, "pins": pins, "tokens": toks, "tmp": tmp_path, "fresh": fresh, "bdir": bdir}
    seal(u)
    return u


def seal(u, text=None, draft=False, record=True):
    """write the planted sealed text + its .sha256 (planted pins = the REAL current shas of the real code files)"""
    import hashlib
    pins = u["pins"]
    body = text if text is not None else f"# planted PREREG L2-STEPB\n{'Status: DRAFT' if draft else 'Status: SEALED'}\n\n{C.render_pins(pins)}\n"
    raw = body.encode("utf-8")
    open(u["cfg"].prereg, "wb").write(raw)
    if record:
        open(u["cfg"].prereg_sha, "w").write(hashlib.sha256(raw).hexdigest() + "  PREREG_L2_STEPB.md\n")


def mk(u, runner=None, **kw):
    out = []
    kw.setdefault("ram_fn", lambda: 99.0)                                              # never the real box's RAM in a test (the real chain waits for >= 8 GiB before each label step)
    kw.setdefault("alive_fn", lambda pid: True)
    ch = C.Chain(u["cfg"], runner=runner or FakeRunner(u), clock=kw.pop("clock", FakeClock()), out=out.append, **kw)
    return ch, out


def stop_reason(out):
    s = [x for x in out if x.startswith("ZZSTEPB_STOP")]
    return s[-1] if s else None


# ------------------------------------------------------------------------------------------------------------------------------ the seal guard
def test_green_the_planted_consistent_seal_passes_and_the_chain_runs_to_the_verdict(uni):
    r = FakeRunner(uni)
    ch, out = mk(uni, r)
    assert ch.run() == 0
    assert out[-1].startswith("ZZSTEPB_STRATEGIC") and any(x.startswith("ZZSTEPB_DONE O1_PASS") for x in out)
    assert all(x.startswith("ZZSTEPB_") for x in out)                      # every line is a marker AT THE START: a grep anchored on ^ZZSTEPB_ cannot hit a mention


def test_refuses_without_the_seal_and_starts_no_child(uni):
    for victim in (uni["cfg"].prereg_sha, uni["cfg"].prereg):
        os.rename(victim, victim + ".gone")
        r = FakeRunner(uni)
        ch, out = mk(uni, r)
        assert ch.run() == 5 and "SEAL_MISSING" in stop_reason(out) and r.calls == []
        os.rename(victim + ".gone", victim)


def test_refuses_a_text_edited_after_the_seal(uni):
    open(uni["cfg"].prereg, "ab").write(b"\nan edit after the seal\n")
    r = FakeRunner(uni)
    ch, out = mk(uni, r)
    assert ch.run() == 5 and "SEAL_MISMATCH" in stop_reason(out) and r.calls == []


def test_refuses_a_draft_marker_and_a_missing_pins_block(uni):
    seal(uni, draft=True)
    ch, out = mk(uni)
    assert ch.run() == 5 and "SEAL_NOT_SEALED" in stop_reason(out)
    seal(uni, text="# planted\nStatus: SEALED\nno pins here\n")
    ch, out = mk(uni)
    assert ch.run() == 5 and "SEAL_NO_PINS" in stop_reason(out)


@pytest.mark.parametrize("key,why", [("code:stepB.py", "SEAL_CODE_PIN_MISMATCH"), ("code:stepB_chain.py", "SEAL_CODE_PIN_MISMATCH"), ("config_sha256", "SEAL_CONFIG_PIN_MISMATCH"),
                                     ("config_o4_sha256", "SEAL_CONFIG_PIN_MISMATCH"), ("head_s1_sha256", "SEAL_HEAD_PIN_MISMATCH"), ("carve_sha256", "SEAL_CARVE_PIN_VS_CODE"),
                                     ("l1:tts_cem.py", "SEAL_L1_PINS")])
def test_refuses_each_pin_that_differs_from_what_the_chain_would_run(uni, key, why):
    uni["pins"] = {**uni["pins"], key: "0" * 64}
    seal(uni)
    r = FakeRunner(uni)
    ch, out = mk(uni, r)
    assert ch.run() == 5 and why in stop_reason(out) and r.calls == []


def test_refuses_when_an_operative_file_is_not_pinned(uni):
    uni["pins"] = {k: v for k, v in uni["pins"].items() if k != "code:label_stepB.py"}
    seal(uni)
    ch, out = mk(uni)
    assert ch.run() == 5 and "SEAL_PINS_MISSING" in stop_reason(out)


def test_refuses_a_head_manifest_that_disagrees_with_the_pin(uni):
    json.dump({"heads": {"s0": {"sha256": "1" * 64}, "s1": {"sha256": uni["pins"]["head_s1_sha256"]}}}, open(f"{uni['cfg'].heads_dir}/final_heads.json", "w"))
    ch, out = mk(uni)
    assert ch.run() == 5 and "SEAL_HEAD_MANIFEST" in stop_reason(out)


def test_waits_for_l2_fresh_seal_and_refuses_a_mismatched_or_disagreeing_one(uni):
    os.rename(uni["cfg"].fresh_sha, uni["cfg"].fresh_sha + ".gone")                      # not sealed yet: the chain WAITS (it does not refuse), then times out
    ch, out = mk(uni)
    assert ch.run() == 4 and "WAIT_TIMEOUT" in stop_reason(out)
    os.rename(uni["cfg"].fresh_sha + ".gone", uni["cfg"].fresh_sha)
    open(uni["cfg"].fresh_prereg, "a").write("an edit after L2-FRESH's seal\n")           # sealed text edited: its sha no longer matches
    ch, out = mk(uni)
    assert ch.run() == 5 and "FRESH_SEAL_MISMATCH" in stop_reason(out)


def test_refuses_a_fresh_seal_that_names_another_carve(uni):
    import hashlib
    t = f"Status: SEALED\ncarve {'f' * 64}\nheads {uni['pins']['head_s0_sha256']} {uni['pins']['head_s1_sha256']}\n"
    open(uni["cfg"].fresh_prereg, "w").write(t)
    open(uni["cfg"].fresh_sha, "w").write(hashlib.sha256(t.encode()).hexdigest() + "  x\n")
    ch, out = mk(uni)
    assert ch.run() == 5 and "FRESH_SEAL_PIN_DISAGREES" in stop_reason(out)


# ------------------------------------------------------------------------------------------------------------------------------ the wait for carve B
def test_marker_patterns_are_anchored_and_do_not_match_a_line_that_only_mentions_them():
    assert C.DONE_RE.search(B_LOG_DONE) and not C.FAIL_RE.search(B_LOG_DONE)                       # the real log has the mention inside the STEP3_RC line, and no failure
    mention = "STEP3_RC=0   (markers CHAIN_DONE_NO_READ ZZBUNDLE_OK STEP5_FAIL ZZCFEAT_FAIL)\n  CHAIN_DONE_NO_READ (indented)\necho CHAIN_DONE_NO_READ\n"
    assert not C.DONE_RE.search(mention) and not C.FAIL_RE.search(mention)
    for line in ("STEP3_FAIL rc=1", "STEP5_RAM_GATE_FAIL", "G2B_GATE_NOT_PASSED: no bundle.npz", "EXPORT_FAIL w1", "MERGE_FAIL", "SCORES_FAIL_or_CFEAT_FAIL", "ZZCFEAT_FAIL {}", "ZZLABEL_FAIL G1 (identity or control)",
                 "EMPTY_SAMPLE_REFUSED", "NO_SAMPLE x"):
        assert C.FAIL_RE.search("noise\n" + line + "\n"), line


def test_waits_while_the_b_chain_runs_prints_a_heartbeat_and_times_out(uni):
    (uni["fresh"] / "chain_B.log").write_text("CHAIN_START 19.10.2026 16:03:40,57 TAG=B N=400\n")
    clock = FakeClock()
    ch, out = mk(uni, clock=clock)
    assert ch.run() == 4 and "WAIT_TIMEOUT" in stop_reason(out)
    assert any(x.startswith("ZZSTEPB_WAITING") for x in out) and sum(clock.sleeps) >= 600 and ch.runner.calls == []


def test_a_fail_marker_of_the_b_chain_stops_before_any_gate_or_child(uni):
    (uni["fresh"] / "chain_B.log").write_text("CHAIN_START x\nSTEP5_FAIL\n")
    r = FakeRunner(uni)
    ch, out = mk(uni, r)
    assert ch.run() == 3 and "B_CHAIN_FAILED" in stop_reason(out) and r.calls == []


def test_a_dead_b_chain_without_a_marker_stops_and_a_dead_one_that_finished_does_not(uni):
    (uni["fresh"] / "chain_B.log").write_text("CHAIN_START x\n")
    uni["cfg"].b_pid = 424242
    ch, out = mk(uni, alive_fn=lambda pid: False)
    assert ch.run() == 3 and "B_CHAIN_GONE" in stop_reason(out)
    (uni["fresh"] / "chain_B.log").write_text(B_LOG_DONE)
    ch, out = mk(uni, alive_fn=lambda pid: False)
    assert ch.run() == 0


# ------------------------------------------------------------------------------------------------------------------------------ the L2-FRESH gates
def _gate_cases(u):
    return [("bundle forward not ok", {"step3.log": "ZZGPUWAIT\nZZBUNDLE_OK 399 600 s\n"}), ("bundle forward absent", {"step3.log": "ZZBUNDLE_FAIL missing 1\n"}),
            ("bundle ok only mentioned", {"step3.log": "note: waiting for ZZBUNDLE_OK 400\n"}), ("g1 identity failed", {"labels.npz.g1.json": {"passed": False, "control_ok": True}}),
            ("g1 control failed", {"labels.npz.g1.json": {"passed": True, "control_ok": False}}),
            ("G2a", {"scores.npz.gates.json": {"G2a": [{"passed": True}, {"passed": False}], "G2c": [{"premise_true": True}], "G3": [{"passed": True}], "G2b_cross_device": [{"pick_agreement": 1.0}]}}),
            ("G3", {"scores.npz.gates.json": {"G2a": [{"passed": True}], "G2c": [{"premise_true": True}], "G3": [{"passed": False}], "G2b_cross_device": [{"pick_agreement": 1.0}]}}),
            ("G2c", {"scores.npz.gates.json": {"G2a": [{"passed": True}], "G2c": [{"premise_true": False}], "G3": [{"passed": True}], "G2b_cross_device": [{"pick_agreement": 1.0}]}}),
            ("pick agreement", {"scores.npz.gates.json": {"G2a": [{"passed": True}], "G2c": [{"premise_true": True}], "G3": [{"passed": True}], "G2b_cross_device": [{"pick_agreement": 0.98}]}}),
            ("C-FEAT", {"fresh_ep_scores.json": {"pass": False}}), ("C-FEAT missing key", {"fresh_ep_scores.json": {}})]


@pytest.mark.parametrize("idx", range(11))
def test_each_failing_l2_fresh_gate_stops_the_chain_before_any_child(uni, idx):
    name, over = _gate_cases(uni)[idx]
    write_gates(str(uni["bdir"]), **over)
    r = FakeRunner(uni)
    ch, out = mk(uni, r)
    assert ch.run() == 3 and "B_GATES_FAILED" in stop_reason(out), name
    assert r.calls == []


def test_a_missing_gate_file_stops_the_chain(uni):
    os.remove(f"{uni['bdir']}/scores.npz.gates.json")
    ch, out = mk(uni)
    assert ch.run() == 3 and "B_GATES_FAILED" in stop_reason(out) and "unreadable" in stop_reason(out)


# ------------------------------------------------------------------------------------------------------------------------------ the DATA CLAUSE at the chain
def _write_tokens(u, toks):
    z = np.load(u["cfg"].bundle, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["token"] = np.array(toks)
    np.savez(u["cfg"].bundle, **d)


def test_refuses_a_token_mismatch_and_starts_no_child(uni):
    toks = list(uni["tokens"])
    for bad in ([*toks[:5], "0" * 16, *toks[6:]], toks[:-1], [*toks[:3], toks[4], toks[3], *toks[5:]]):      # one replaced, one dropped, two swapped (same SET, other order)
        _write_tokens(uni, bad)
        r = FakeRunner(uni)
        ch, out = mk(uni, r)
        assert ch.run() == 5 and "DATA_CLAUSE" in stop_reason(out), bad[:2]
        assert r.calls == []


def test_the_data_clause_green_control_and_the_ctx_checks(uni):
    ch, out = mk(uni)
    assert ch.run() == 0
    # ctx.json that names another token list, a torn ctx.npy, a wrong n
    j = json.load(open(uni["cfg"].ctx_json))
    for over in ({"tokens_sha256": "1" * 64}, {"ctx_sha256": "2" * 64}, {"n": 399}):
        json.dump({**j, **over}, open(uni["cfg"].ctx_json, "w"))
        r = FakeRunner(uni)
        ch, out = mk(uni, r)
        assert ch.run() == 5 and r.calls == [], over
    json.dump(j, open(uni["cfg"].ctx_json, "w"))


def test_the_data_clause_red_arm_the_chain_check_is_what_refuses_a_swapped_token(uni, monkeypatch):
    """MUTATE: ONE token replaced (n stays 400, ctx.json untouched).  The real chain refuses (exit 5, no child).  With `check_data_clause` replaced by a no-op the SAME chain runs on through gates, searches, labels
    and the analysis on a bundle that is not the carve: the refusal above IS that call (the launch and the gate carry their own @DATACLAUSE lines as the second and third defence)."""
    toks = list(uni["tokens"])
    toks[5] = "0" * 16
    _write_tokens(uni, toks)
    r = FakeRunner(uni)
    ch, out = mk(uni, r)
    assert ch.run() == 5 and "DATA_CLAUSE" in stop_reason(out) and r.calls == []
    monkeypatch.setattr(stepB, "check_data_clause", lambda *a, **k: {})
    shutil.rmtree(uni["cfg"].run, ignore_errors=True)
    ch2, out2 = mk(uni, FakeRunner(uni))
    assert ch2.run() == 0


# ------------------------------------------------------------------------------------------------------------------------------ G-REACH-STEPB: any failure stops BEFORE any search
def test_a_failing_gate_on_either_head_stops_before_any_search(uni):
    for failing in ({"s0"}, {"s1"}, {"s0", "s1"}):
        shutil.rmtree(uni["cfg"].run, ignore_errors=True)
        r = FakeRunner(uni, gate_fail=failing)
        ch, out = mk(uni, r)
        assert ch.run() == 3 and "REACH_GATE_FAILED" in stop_reason(out)
        assert not any(k.startswith("stepB.py") or k in ("label_stepB.py", "analyse_stepB.py") for k in r.kinds()), failing          # NO SEARCH, no label, no analysis


def test_a_gate_that_exits_zero_without_a_pass_stamp_binding_this_head_still_stops(uni):
    r = FakeRunner(uni, stamp_without_pass=True)
    ch, out = mk(uni, r)
    assert ch.run() == 3 and "REACH_GATE_FAILED" in stop_reason(out) and "stepB.py search" not in r.kinds()


def test_a_stamp_for_another_bundle_is_not_reused(uni):
    r = FakeRunner(uni)
    ch, out = mk(uni, r)
    assert ch.run() == 0
    st = json.load(open(ch.stamp_path("s0")))
    st["bundle_sha256"] = "9" * 64
    json.dump(st, open(ch.stamp_path("s0"), "w"))
    r2 = FakeRunner(uni)
    ch2, out2 = mk(uni, r2)
    assert ch2.run() == 0 and r2.kinds().count("g_reach_stepb.py") == 1                                       # s0's stale stamp is re-minted, s1's is kept


# ------------------------------------------------------------------------------------------------------------------------------ the registered order, the env, the resume, the later stops
def test_happy_path_runs_the_children_in_the_registered_order_cpu_only_and_capped(uni):
    r = FakeRunner(uni)
    ch, out = mk(uni, r)
    assert ch.run() == 0
    assert r.kinds() == ["g_reach_stepb.py", "g_reach_stepb.py", "stepB.py search", "stepB.py merge", "stepB.py search", "stepB.py merge", "stepB.py search", "stepB.py merge",
                         "label_stepB.py", "label_stepB.py", "label_stepB.py", "analyse_stepB.py"]
    for k, argv, env in r.calls:
        assert env["CUDA_VISIBLE_DEVICES"] == "-1" and env["OMP_NUM_THREADS"] == ("1" if k == "label_stepB.py" else "4")
        assert "--lock" not in argv and (k != "stepB.py search" or ("--threads" in argv and argv[argv.index("--threads") + 1] == "4"))
    searches = [a for k, a, _ in r.calls if k == "stepB.py search"]
    assert [x[x.index("--arm") + 1] for x in searches] == ["L2", "L2", "P2HEAD"] and "--head" not in searches[2] and "--head-sha256" not in searches[2]
    assert searches[0][searches[0].index("--head-sha256") + 1] == uni["pins"]["head_s0_sha256"] and searches[1][searches[1].index("--head-sha256") + 1] == uni["pins"]["head_s1_sha256"]
    assert all(os.path.basename(a[a.index("--gate") + 1]).startswith("G-REACH-STEPB_PASS_") for a in searches)
    ana = r.calls[-1][1]
    assert [x for x in ana if x.startswith("L2s") or x.startswith("P2HEAD")] == [f"{n}={ch.search_file(n)}" for n in ("L2s0", "L2s1", "P2HEAD")] + [f"{n}={ch.labels_file(n)}" for n in ("L2s0", "L2s1", "P2HEAD")]
    iv = [i for i, x in enumerate(out) if x.startswith("ZZSTEPB_DONE")]
    assert len(iv) == 1 and out[iv[0]] == "ZZSTEPB_DONE O1_PASS" and out[-1].startswith("ZZSTEPB_STRATEGIC PAUSED")
    assert any(x.startswith("ZZSTEPB_NEXT NAMED NEXT ARM") for x in out) and any(x.startswith("ZZSTEPB_PER_HEAD L2s0=O1_PASS") for x in out)
    # both stamps were minted before the first search
    assert max(i for i, k in enumerate(r.kinds()) if k == "g_reach_stepb.py") < min(i for i, k in enumerate(r.kinds()) if k == "stepB.py search")


def test_label_child_gets_the_navsim_env_and_the_search_children_do_not(uni):
    r = FakeRunner(uni)
    ch, out = mk(uni, r)
    assert ch.run() == 0
    for k, argv, env in r.calls:
        if k == "label_stepB.py":
            assert env["NUPLAN_MAP_VERSION"] == "nuplan-maps-v1.0" and env["REFE_SIM_HZ"] == "10" and "navsim" in env["PYTHONPATH"] and argv[0] == "PYL"
        else:
            assert "PYTHONPATH" not in env and argv[0] == "PY"


def test_resume_skips_every_stage_whose_artifact_is_still_valid(uni):
    r = FakeRunner(uni)
    ch, out = mk(uni, r)
    assert ch.run() == 0
    r2 = FakeRunner(uni)
    ch2, out2 = mk(uni, r2)
    assert ch2.run() == 0 and r2.kinds() == ["analyse_stepB.py"]
    assert sum(1 for x in out2 if x.startswith("ZZSTEPB_REACH_PASS")) == 2
    os.remove(ch.labels_file("L2s1"))                                                                         # one label file lost: only that label step re-runs
    r3 = FakeRunner(uni)
    ch3, out3 = mk(uni, r3)
    assert ch3.run() == 0 and r3.kinds() == ["label_stepB.py", "analyse_stepB.py"]


def test_resume_does_not_trust_a_search_file_for_another_head(uni):
    r = FakeRunner(uni)
    ch, out = mk(uni, r)
    assert ch.run() == 0
    z = np.load(ch.search_file("L2s1"), allow_pickle=False)
    m = json.loads(str(z["meta"]))
    m["head_sha256"] = uni["pins"]["head_s0_sha256"]                                                          # s1's file claims s0's head
    np.savez(ch.search_file("L2s1"), meta=np.array(json.dumps(m)), token=z["token"], bundle_index=z["bundle_index"])
    r2 = FakeRunner(uni)
    ch2, out2 = mk(uni, r2)
    assert ch2.run() == 0 and r2.kinds().count("stepB.py search") == 1 and r2.kinds().count("stepB.py merge") == 1


def test_a_failed_search_stops_and_runs_no_label_or_analysis(uni):
    r = FakeRunner(uni, search_rc=1)
    ch, out = mk(uni, r)
    assert ch.run() == 6 and "SEARCH_FAILED" in stop_reason(out) and "label_stepB.py" not in r.kinds() and "analyse_stepB.py" not in r.kinds()


def test_a_label_identity_failure_exit_3_stops_with_no_analysis(uni):
    r = FakeRunner(uni, label_rc=3)
    ch, out = mk(uni, r)
    assert ch.run() == 7 and "LABEL_IDENTITY_FAILED" in stop_reason(out) and "analyse_stepB.py" not in r.kinds()


def test_ram_gate_waits_then_proceeds_and_times_out(uni):
    seq = iter([2.0, 3.0, 9.0] + [9.0] * 50)
    clock = FakeClock()
    ch, out = mk(uni, FakeRunner(uni), clock=clock, ram_fn=lambda: next(seq))
    assert ch.run() == 0 and clock.sleeps == [20, 20]
    ch, out = mk(uni, FakeRunner(uni), clock=FakeClock(), ram_fn=lambda: 1.0)
    shutil.rmtree(uni["cfg"].run)
    assert ch.run() == 7 and "RAM_TIMEOUT" in stop_reason(out)


def test_an_analysis_result_without_a_verdict_and_next_arm_stops(uni):
    r = FakeRunner(uni, no_verdict=True)
    ch, out = mk(uni, r)
    assert ch.run() == 8 and "NO_VERDICT" in stop_reason(out) and not any(x.startswith("ZZSTEPB_DONE") for x in out)


@pytest.mark.parametrize("outcome", ["O1_PASS", "TRADE", "NOT_CLEARED", "O4_INERT"])
def test_the_verdict_line_carries_each_outcome(uni, outcome):
    ch, out = mk(uni, FakeRunner(uni, outcome=outcome))
    assert ch.run() == 0 and f"ZZSTEPB_DONE {outcome}" in out


def test_the_bundle_changing_during_the_run_stops_the_analysis(uni):
    class Tamper(FakeRunner):
        def __call__(self, argv, env, log_path, cwd=None, on_pid=None):
            if self.kind(argv) == "label_stepB.py":
                open(uni["cfg"].bundle, "ab").write(b"x")
            return super().__call__(argv, env, log_path, cwd, on_pid)
    ch, out = mk(uni, Tamper(uni))
    assert ch.run() == 5 and "BUNDLE_CHANGED" in stop_reason(out) and "analyse_stepB.py" not in ch.runner.kinds()


def test_pins_round_trip_and_the_real_seal_inputs_exist():
    cfg = C.Cfg()
    pins = C.collect_pins(cfg) if os.path.exists(f"{cfg.heads_dir}/final_head_s0.npz") and os.path.exists(cfg.config_o4) else None
    if pins:
        assert C.parse_pins("x\n" + C.render_pins(pins)) == {k: v for k, v in pins.items() if v is not None}
        assert pins["head_s0_sha256"].startswith("bded124e") and pins["head_s1_sha256"].startswith("86f0b20f") and pins["config_sha256"].startswith("299c6e63") and pins["carve_sha256"].startswith("0629811f")


def test_a_bug_inside_a_stage_ends_in_a_marker_and_exit_9_not_a_silent_traceback(uni, monkeypatch):
    def boom(*a, **k):
        raise KeyError("planted")
    ch, out = mk(uni, FakeRunner(uni))
    monkeypatch.setattr(ch, "check_b_gates", boom)
    assert ch.run() == 9 and "ZZSTEPB_STOP CRASH KeyError" in out[-1]
    ch, out = mk(uni, FakeRunner(uni))
    monkeypatch.setattr(ch, "check_b_gates", lambda: (_ for _ in ()).throw(SystemExit("a child library refused")))
    assert ch.run() == 9 and "SystemExit" in out[-1]


def test_a_multi_line_detail_stays_one_marker_line(uni):
    ch, out = mk(uni)
    ch.emit("ZZSTEPB_NOTE", "first line" + chr(10) + "second line" + chr(13) + chr(10) + "third")
    assert out == ["ZZSTEPB_NOTE first line | second line  | third"] and all(x.startswith("ZZSTEPB_") for x in out)
