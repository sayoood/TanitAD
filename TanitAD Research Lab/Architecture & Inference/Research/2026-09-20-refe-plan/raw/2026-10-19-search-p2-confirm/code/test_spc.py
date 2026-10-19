"""Tests of the SEARCH-P2-CONFIRM chain.  CPU only: run with CUDA_VISIBLE_DEVICES=-1.
    cd code && CUDA_VISIBLE_DEVICES=-1 python -m pytest test_spc.py -q -p no:cacheprovider --basetemp=<scratch>

Every gate of the prereg (section 5) has a GREEN control and at least one RED arm that must make the chain REFUSE (a StopChain with the stated exit code, and NO child process started).  Tests on the REAL banked
Step B / carve-B files are skipped when those files are absent; everything else runs on planted data.  Nothing here launches a GPU job or touches the dev-box lock.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import b_rest as BR  # noqa: E402
import seal_spc as S  # noqa: E402
import spc_analyse as AC  # noqa: E402
import spc_chain as CH  # noqa: E402
import spc_cid as CID  # noqa: E402
import spc_gpu_stage as GS  # noqa: E402
import spc_launch as SL  # noqa: E402

PKG0 = os.path.dirname(HERE)
RAW = os.path.join(PKG0, "raw")
TEMPLATE = os.path.join(PKG0, "src", "PREREG_DRAFT_TEMPLATE.md")
BANKED_SEARCH = CH.BANKED_SEARCH
BANKED_BUNDLE = os.path.join(CH.BANKED_BUNDLE_DIR, "bundle.npz")
BANKED_LABELS = "C:/Users/Admin/par18/l2stepb/run/labels_P2HEAD.npz"
real = pytest.mark.skipif(not (os.path.exists(BANKED_SEARCH) and os.path.exists(BANKED_BUNDLE) and os.path.exists(BANKED_LABELS)), reason="banked Step B / carve-B files absent")


# ------------------------------------------------------------------------------------------------------------------------------ fixtures
class NoChild:
    """a runner that FAILS the test if any child process is started (a refusal must come before any child)"""
    def __init__(self):
        self.calls = []

    def __call__(self, argv, env, log_path, cwd=None):
        self.calls.append((list(argv), dict(env), log_path))
        raise AssertionError(f"a child process was started: {argv[:4]}")


class Recorder:
    def __init__(self, rc=0, writer=None):
        self.calls, self.rc, self.writer = [], rc, writer

    def __call__(self, argv, env, log_path, cwd=None):
        self.calls.append((list(argv), dict(env), log_path))
        if self.writer:
            self.writer(argv, env, log_path)
        return self.rc


def make_text(pins_mut=None, status_draft=False):
    tmpl = open(TEMPLATE, encoding="utf-8").read().replace("@@POWER_TABLE@@", "T").replace("@@DQS@@", "D").replace("@@PINS@@", "x")
    pkg_for_pins = PKG0
    pins = S.compute_pins(pkg_for_pins, code_dir=HERE) if os.path.exists(os.path.join(HERE, "test_spc.py")) else None
    if pins_mut:
        pins_mut(pins)
    text = S.build_sealed(tmpl, pins)
    if status_draft:
        text = text.replace("Status: **SEALED**", "Status: **DRAFT**", 1)
    return text


def write_pkg(tmp, text, sha=None, raw_mut=None):
    pkg = os.path.join(str(tmp), "pkg")
    os.makedirs(os.path.join(pkg, "raw"), exist_ok=True)
    for f in S.RAW_PINS:
        shutil.copy(os.path.join(RAW, f), os.path.join(pkg, "raw", f))
    if raw_mut:
        raw_mut(pkg)
    import hashlib
    open(os.path.join(pkg, CH.SEALED_NAME), "w", encoding="utf-8", newline="\n").write(text)
    open(os.path.join(pkg, "PREREG_SEARCH_P2_CONFIRM.sha256"), "w").write((sha or hashlib.sha256(text.encode("utf-8")).hexdigest()) + "\n")
    return pkg


def chain_for(tmp, text, runner=None, **kw):
    pkg = write_pkg(tmp, text, **kw)
    cfg = CH.Cfg(pkg, work=os.path.join(str(tmp), "work"), hash_ckpt=False)
    return CH.Chain(cfg, runner=runner or NoChild(), out=io.StringIO(), ram_fn=lambda: 64.0, free_fn=lambda p: 500.0)


def stops(fn, *a, **k):
    with pytest.raises(CH.StopChain) as ei:
        fn(*a, **k)
    return ei.value


@pytest.fixture(scope="module")
def good_text():
    return make_text()


# ------------------------------------------------------------------------------------------------------------------------------ G-SEAL
def test_seal_green(tmp_path, good_text):
    ch = chain_for(tmp_path, good_text)
    pins = ch.check_seal()
    assert pins["b_rest_tokens_sha256"] == SL.B_REST_SHA256 and "ZZSPC_SEAL_OK" in ch.out.getvalue()


def test_seal_red_missing(tmp_path, good_text):
    ch = chain_for(tmp_path, good_text)
    os.remove(ch.cfg.sha_file)
    assert stops(ch.check_seal).code == 5


def test_seal_red_one_byte_edit(tmp_path, good_text):
    ch = chain_for(tmp_path, good_text + " ")
    import hashlib
    open(ch.cfg.sha_file, "w").write(hashlib.sha256(good_text.encode()).hexdigest() + "\n")       # the sha of the ORIGINAL text, the text edited afterwards
    e = stops(ch.check_seal)
    assert e.code == 5 and e.reason == "SEAL_SHA_MISMATCH"


def test_seal_red_draft_status(tmp_path, good_text):
    e = stops(chain_for(tmp_path, good_text.replace("Status: **SEALED**", "Status: **DRAFT**", 1)).check_seal)
    assert e.code == 5 and e.reason in ("SEAL_NOT_SEALED", "SEAL_STILL_DRAFT")


def test_seal_red_no_sealed_marker(tmp_path, good_text):
    e = stops(chain_for(tmp_path, good_text.replace("Status: **SEALED**", "Status: **", 1)).check_seal)
    assert e.code == 5


def test_seal_red_no_pins_block(tmp_path, good_text):
    e = stops(chain_for(tmp_path, good_text.replace("<!-- PINS-BEGIN -->", "")).check_seal)
    assert e.reason == "SEAL_NO_PINS"


@pytest.mark.parametrize("key", ["b_rest_tokens_sha256", "carve_b_tokens_sha256", "ckpt_sha256", "config_sha256", "head_s0_sha256", "bridge_bundle_script_sha256_lf"])
def test_seal_red_missing_required_pin(tmp_path, key):
    t = make_text(lambda p: p.pop(key))
    e = stops(chain_for(tmp_path, t).check_seal)
    assert e.reason == "SEAL_PIN_MISSING" and key in e.detail


@pytest.mark.parametrize("key", ["b_rest_tokens_sha256", "config_sha256", "head_s0_sha256", "bridge_bundle_script_sha256_lf", "code:spc_launch.py", "code:spc_cid.py", "code:stepB.py", "code:label_stepB.py",
                                 "code:analyse_stepB.py", "l1:stepA.py", "l1:tts_cem.py", "raw:b_rest.json", "raw:cid_sample.json"])
def test_seal_red_changed_pin(tmp_path, key):
    t = make_text(lambda p: p.__setitem__(key, "0" * 64))
    e = stops(chain_for(tmp_path, t).check_seal)
    assert e.code == 5 and e.reason == "SEAL_PIN_MISMATCH" and key.split(":")[-1].split("_sha")[0] in e.detail


def test_seal_red_operative_file_not_pinned(tmp_path):
    t = make_text(lambda p: p.pop("code:spc_analyse.py"))
    e = stops(chain_for(tmp_path, t).check_seal)
    assert e.reason == "SEAL_PIN_MISMATCH" and "not pinned" in e.detail


def test_seal_red_raw_file_edited_after_seal(tmp_path, good_text):
    def edit(pkg):
        with open(os.path.join(pkg, "raw", "design_calc.json"), "a") as f:
            f.write(" ")
    e = stops(chain_for(tmp_path, good_text, raw_mut=edit).check_seal)
    assert e.reason == "SEAL_PIN_MISMATCH" and "design_calc" in e.detail


def test_seal_refuses_unfilled_placeholder_and_missing_draft_status():
    pins = S.compute_pins(PKG0, code_dir=HERE)
    with pytest.raises(S.SealRefused):
        S.build_sealed("Status: **DRAFT**\n@@PINS@@\n", pins)
    with pytest.raises(S.SealRefused):
        S.build_sealed("no status line\n<!-- PINS-BEGIN -->\nx\n<!-- PINS-END -->\n", pins)


def test_seal_refuses_when_a_step_b_pin_no_longer_matches(tmp_path):
    fake = os.path.join(str(tmp_path), "PREREG_L2_STEPB.md")
    t = open("C:/Users/Admin/par18/l2stepb/PREREG_L2_STEPB.md", encoding="utf-8").read()
    t = t.replace("code:stepB.py 74e60d3de6f16aa1bcb3a9503cba54d81bc6e79fa55e4fe6ecb9efc60e390ea8", "code:stepB.py " + "1" * 64)
    open(fake, "w", encoding="utf-8").write(t)
    with pytest.raises(S.SealRefused):
        S.compute_pins(PKG0, code_dir=HERE, stepb_prereg=fake)


def test_chain_run_refuses_without_seal_and_starts_no_child(tmp_path):
    cfg = CH.Cfg(os.path.join(str(tmp_path), "empty"), work=os.path.join(str(tmp_path), "w"), hash_ckpt=False)
    nc = NoChild()
    ch = CH.Chain(cfg, runner=nc, out=io.StringIO(), ram_fn=lambda: 64.0, free_fn=lambda p: 500.0)
    assert ch.run() == 5 and not nc.calls and "ZZSPC_STOP SEAL_MISSING" in ch.out.getvalue()


# ------------------------------------------------------------------------------------------------------------------------------ G-DATA
def _paths():
    plan = json.load(open(CH.PLAN))
    sA = json.load(open(f"{BR.W}/sample.json"))
    cb = json.load(open(BR.CARVE_B, encoding="utf-8"))
    return plan, sA, cb


def test_b_rest_pure_green_and_numbers():
    plan, sA, cb = _paths()
    rest, logs = BR.b_rest(plan["tokens"], plan["token_log"], sA["tokens"], cb["tokens"])
    assert len(rest) == 905 and len(set(logs)) == 24 and BR.sha_tokens(rest) == SL.B_REST_SHA256
    assert not (set(rest) & set(sA["tokens"])) and not (set(rest) & set(cb["tokens"])) and len(rest) + 800 == 1705


def test_b_rest_pure_red_arms():
    plan, sA, cb = _paths()
    P, tl = plan["tokens"], plan["token_log"]
    with pytest.raises(AssertionError):
        BR.b_rest(P, tl, sA["tokens"], cb["tokens"][:-1] + [sA["tokens"][0]])           # carve B overlaps Step A
    with pytest.raises(AssertionError):
        BR.b_rest(P, tl, sA["tokens"] + ["deadbeefdeadbeef"], cb["tokens"])             # a Step A token not in the plan
    with pytest.raises(AssertionError):
        BR.b_rest(P + [P[0]], tl, sA["tokens"], cb["tokens"])                           # plan tokens not unique
    rest, _ = BR.b_rest(P, tl, sA["tokens"], cb["tokens"])
    with pytest.raises(AssertionError):
        BR.b_rest(P, {k: v for k, v in tl.items() if k != rest[3]}, sA["tokens"], cb["tokens"])    # a B-REST token with no log


def test_data_green(tmp_path, good_text):
    ch = chain_for(tmp_path, good_text)
    pins = ch.check_seal()
    ch.check_data(pins)
    assert "ZZSPC_DATA_OK 905 tokens 24 logs" in ch.out.getvalue()


def _data_red(tmp_path, good_text, mutate):
    ch = chain_for(tmp_path, good_text)
    pins = ch.check_seal()
    mutate(ch.cfg)
    e = stops(ch.check_data, pins)
    assert e.code == 5 and e.reason == "DATA_REFUSED"


def test_data_red_token_replaced_dropped_swapped_appended(tmp_path, good_text):
    def edit(fn):
        def m(cfg):
            r = json.load(open(cfg.b_rest_json, encoding="utf-8"))
            fn(r)
            json.dump(r, open(cfg.b_rest_json, "w", encoding="utf-8"))
        return m
    def replaced(r): r["tokens"][5] = "0123456789abcdef"
    def dropped(r): r["tokens"].pop(); r["logs"].pop()
    def swapped(r): r["tokens"][0], r["tokens"][1] = r["tokens"][1], r["tokens"][0]
    def appended(r): r["tokens"].append("0123456789abcdef"); r["logs"].append(r["logs"][0])
    def empty(r): r["tokens"], r["logs"] = [], []
    for i, fn in enumerate((replaced, dropped, swapped, appended, empty)):
        d = os.path.join(str(tmp_path), str(i))
        os.makedirs(d)
        _data_red(d, good_text, edit(fn))


def test_data_red_overlap_with_step_a(tmp_path, good_text):
    def m(cfg):
        plan, sA, cb = _paths()
        rest = json.load(open(cfg.b_rest_json, encoding="utf-8"))["tokens"]
        sA2 = dict(sA); sA2["tokens"] = sA["tokens"][:-1] + [rest[0]]                      # a B-REST token now 'is' in Step A's sample: the recomputed B-REST loses it
        p = os.path.join(os.path.dirname(cfg.b_rest_json), "sampleA_planted.json")
        json.dump(sA2, open(p, "w"))
        cfg.stepa_sample = p
    _data_red(tmp_path, good_text, m)


def test_data_red_cid_sample_not_carve_b(tmp_path, good_text):
    def m(cfg):
        c = json.load(open(cfg.cid_json, encoding="utf-8"))
        rest = json.load(open(cfg.b_rest_json, encoding="utf-8"))["tokens"]
        c["tokens"][0] = rest[0]                                                          # a B-REST token smuggled into the C-ID control
        json.dump(c, open(cfg.cid_json, "w", encoding="utf-8"))
    _data_red(tmp_path, good_text, m)


def test_disk_red(tmp_path, good_text):
    ch = chain_for(tmp_path, good_text)
    ch.free_fn = lambda p: 3.0
    e = stops(ch.check_disk)
    assert e.code == 7 and e.reason == "DISK_LOW"


def test_launcher_constants_equal_the_files():
    r = json.load(open(os.path.join(RAW, "b_rest.json"), encoding="utf-8"))
    c = json.load(open(os.path.join(RAW, "cid_sample.json"), encoding="utf-8"))
    assert SL.B_REST_SHA256 == r["tokens_sha256"] == BR.sha_tokens(r["tokens"]) and r["n"] == 905
    assert SL.CID_SHA256 == c["tokens_sha256"] == BR.sha_tokens(c["tokens"]) and c["n"] == 16 and c["n_moved"] == 8


# ------------------------------------------------------------------------------------------------------------------------------ the data clause at the launch (Step B's own line, via carve_sha256=)
def test_data_clause_launcher_green_and_red():
    import stepB
    r = json.load(open(os.path.join(RAW, "b_rest.json"), encoding="utf-8"))["tokens"]
    stepB.check_data_clause({"token": np.array(r)}, SL.READS["C1"])                             # GREEN: exactly the sealed B-REST list
    for bad in (r[:-1], r[::-1], r[:5] + ["0123456789abcdef"] + r[6:], r + ["0123456789abcdef"], []):
        with pytest.raises(SystemExit):
            stepB.check_data_clause({"token": np.array(bad)}, SL.READS["C1"])
    with pytest.raises(SystemExit):
        stepB.check_data_clause({"token": np.array(r)}, SL.READS["CID"])                        # the B-REST list is not the C-ID list


@real
def test_data_clause_refuses_carve_b_for_c1():
    import stepB
    b = np.load(BANKED_BUNDLE, allow_pickle=False)
    with pytest.raises(SystemExit):
        stepB.check_data_clause({"token": b["token"]}, SL.READS["C1"])                          # carve B is NOT B-REST (and not the C-ID list)
    stepB.check_data_clause({"token": b["token"]}, None)                                         # GREEN control: Step B's own default accepts carve B (so the clause can see a difference)


def test_launcher_refuses_head_and_non_p2head_arm():
    with pytest.raises(SystemExit) as e1:
        SL.cmd_search(["--read", "C1", "--arm", "P2HEAD", "--head", "x.npz", "--head-sha256", "0" * 64])
    assert "NO head substitution" in str(e1.value)
    with pytest.raises(SystemExit) as e2:
        SL.cmd_search(["--read", "C1", "--arm", "P2HEAD", "--head-sha256=abc"])
    assert "NO head substitution" in str(e2.value)
    full = ["--bundle", "b", "--ctx", "c", "--ckpt", "k", "--ckpt-sha256", "s", "--refe-dir", "r", "--config", "c", "--config-sha256", "s", "--gate", "g", "--out-dir", "o"]
    with pytest.raises(SystemExit) as e3:
        SL.cmd_search(["--read", "C1", "--arm", "L2", *full])
    assert "P2HEAD only" in str(e3.value)


def test_launcher_passes_the_pinned_data_clause_to_step_b(monkeypatch):
    import stepB
    seen = {}
    monkeypatch.setattr(stepB, "run_search_cli", lambda ns, enforce_stamp=True, carve_sha256=None, **k: seen.update(carve=carve_sha256, stamp=enforce_stamp, arm=ns.arm) or 0)
    full = ["--arm", "P2HEAD", "--bundle", "b", "--ctx", "c", "--ckpt", "k", "--ckpt-sha256", "s", "--refe-dir", "r", "--config", "c", "--config-sha256", "s", "--gate", "g", "--out-dir", "o"]
    assert SL.cmd_search(["--read", "C1", *full]) == 0 and seen == {"carve": SL.B_REST_SHA256, "stamp": True, "arm": "P2HEAD"}
    assert SL.cmd_search(["--read", "CID", *full]) == 0 and seen["carve"] == SL.CID_SHA256
    with pytest.raises(SystemExit):
        SL.cmd_search(["--read", "OTHER", *full])                      # no free-form read: the carve sha is never a CLI value


# ------------------------------------------------------------------------------------------------------------------------------ G-BRIDGE / G-BUNDLE (the GPU stage)
def _tokens(which):
    return json.load(open(os.path.join(RAW, "b_rest.json" if which == "C1" else "cid_sample.json"), encoding="utf-8"))["tokens"]


def plant_bundle(cfgd, which, tokens=None, tamper_ctx=False, ctx_json_sha=None):
    toks = tokens if tokens is not None else _tokens(which)
    d = os.path.join(cfgd, "B_REST" if which == "C1" else "cid", "bundle")
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "bundle.npz"), token=np.array(toks))
    np.save(os.path.join(d, "ctx.npy"), np.zeros((len(toks), 2, 2), np.uint16))
    csha = CH.sha256_file(os.path.join(d, "ctx.npy"))
    json.dump({"n": len(toks), "ctx_file": "ctx.npy", "tokens_sha256": ctx_json_sha or BR.sha_tokens(toks), "ctx_sha256": csha}, open(os.path.join(d, "ctx.json"), "w"))
    if tamper_ctx:
        np.save(os.path.join(d, "ctx.npy"), np.ones((len(toks), 2, 2), np.uint16))


def stage_log(cfg, lines):
    os.makedirs(cfg.d("logs"), exist_ok=True)
    open(cfg.d("logs", "gpu_stage.log"), "w").write("\n".join(lines) + "\n")


GOOD_LOG = ["ZZGPULOCKED", "ZZBRIDGE_IDENTITY_OK 3", "ZZBRIDGE_RED_OK 3", "ZZBRIDGE_PASS", "ZZSTAGE_BRIDGE_OK", "ZZBUNDLE_OK 16 30 s", "ZZSTAGE_CID_OK", "  ZZBUNDLE_OK 905 1400 s", "ZZSTAGE_C1_OK", "ZZSTAGE_DONE"]


def test_gpu_stage_green_skips_the_child_when_outputs_validate(tmp_path, good_text):
    ch = chain_for(tmp_path, good_text)
    plant_bundle(ch.cfg.work, "CID"); plant_bundle(ch.cfg.work, "C1"); stage_log(ch.cfg, GOOD_LOG)
    ch.gpu_stage()
    assert "ZZSPC_GPU_STAGE_OK" in ch.out.getvalue()


@pytest.mark.parametrize("drop,why", [("ZZBRIDGE_PASS", "G-BRIDGE"), ("ZZBUNDLE_OK 16", "G-BUNDLE(16)"), ("ZZBUNDLE_OK 905", "G-BUNDLE(905)")])
def test_gpu_stage_red_missing_marker(tmp_path, good_text, drop, why):
    ch = chain_for(tmp_path, good_text)
    plant_bundle(ch.cfg.work, "CID"); plant_bundle(ch.cfg.work, "C1")
    stage_log(ch.cfg, [l for l in GOOD_LOG if drop not in l])
    e = stops(ch.gpu_stage)
    assert e.code == 3 and e.reason == "G_BUNDLE_NOT_PASSED" and why in e.detail


def test_gpu_stage_red_stage_fail_marker(tmp_path, good_text):
    ch = chain_for(tmp_path, good_text)
    plant_bundle(ch.cfg.work, "CID"); plant_bundle(ch.cfg.work, "C1"); stage_log(ch.cfg, GOOD_LOG + ["ZZSTAGE_FAIL BRIDGE rc=3"])
    assert stops(ch.gpu_stage).reason == "G_GPU_STAGE_FAILED"


@pytest.mark.parametrize("kind", ["token_replaced", "token_dropped", "ctx_tampered", "ctx_json_sha_wrong", "bundle_tokens_replaced_ctx_json_claims_the_pin"])
def test_gpu_stage_red_invalid_bundle(tmp_path, good_text, kind):
    ch = chain_for(tmp_path, good_text)
    plant_bundle(ch.cfg.work, "CID")
    r = _tokens("C1")
    if kind == "token_replaced":
        plant_bundle(ch.cfg.work, "C1", tokens=r[:7] + ["0123456789abcdef"] + r[8:])
    elif kind == "token_dropped":
        plant_bundle(ch.cfg.work, "C1", tokens=r[:-1])
    elif kind == "ctx_tampered":
        plant_bundle(ch.cfg.work, "C1", tamper_ctx=True)
    elif kind == "bundle_tokens_replaced_ctx_json_claims_the_pin":
        plant_bundle(ch.cfg.work, "C1", tokens=r[:7] + ["0123456789abcdef"] + r[8:], ctx_json_sha=SL.B_REST_SHA256)       # a consistent lie in ctx.json: only the bundle's own token list can catch it
    else:
        plant_bundle(ch.cfg.work, "C1", ctx_json_sha="f" * 64)
    stage_log(ch.cfg, GOOD_LOG)
    ch.run_child_fn = Recorder(rc=0)              # the child 'succeeds' and changes nothing: the output must STILL be refused
    e = stops(ch.gpu_stage)
    assert e.code == 3 and e.reason in ("G_BUNDLE_INVALID", "G_BUNDLE_NOT_PASSED")


def test_gpu_stage_runs_only_through_the_lock_launcher(tmp_path, good_text):
    ch = chain_for(tmp_path, good_text)
    rec = Recorder(rc=1)
    ch.run_child_fn = rec
    e = stops(ch.gpu_stage)
    assert e.reason == "G_GPU_STAGE_FAILED" and len(rec.calls) == 1
    argv, env, _ = rec.calls[0]
    assert argv[2].endswith("launch_stepA_ctype.py") and argv[3] == "run" and "--" in argv and any(a.endswith("spc_gpu_stage.py") for a in argv)
    assert env.get("CUDA_VISIBLE_DEVICES") != "-1"            # the GPU phase must see the GPU; every CPU child sees -1 (tested below)
    assert "devbox_gpu.lock" not in " ".join(argv)             # the chain never names, takes or removes the lock itself


def test_gpu_stage_script_plan_and_stops_at_first_failure(tmp_path):
    steps = GS.plan(str(tmp_path), "cid.json", "brest.json", py="py", bundle_script="mk.py", bridge_script="bridge.py")
    assert [s[0] for s in steps] == ["BRIDGE", "CID", "C1"]
    assert steps[2][1][steps[2][1].index("--sample") + 1] == "brest.json" and steps[1][1][steps[1][1].index("--sample") + 1] == "cid.json"

    class R:
        def __init__(self, rc, out): self.returncode, self.stdout, self.stderr = rc, out, ""
    seen = []
    def runner_for(results):
        def run(argv):
            seen.append(argv[2]); return results[len(seen) - 1]
        return run
    out = io.StringIO()
    assert GS.run(steps, runner_for([R(0, "ZZBRIDGE_PASS"), R(0, "ZZBUNDLE_OK 16 1 s"), R(0, "ZZBUNDLE_OK 905 9 s")]), out) == 0 and "ZZSTAGE_DONE" in out.getvalue()
    seen.clear(); out = io.StringIO()
    assert GS.run(steps, runner_for([R(0, "ZZBRIDGE_FAIL x"), R(0, "ZZBUNDLE_OK 16"), R(0, "ZZBUNDLE_OK 905")]), out) == 3 and len(seen) == 1       # a bridge without ZZBRIDGE_PASS stops everything after it
    seen.clear(); out = io.StringIO()
    assert GS.run(steps, runner_for([R(0, "ZZBRIDGE_PASS"), R(1, "ZZBUNDLE_OK 16"), R(0, "")]), out) == 3 and len(seen) == 2                          # rc != 0 even with the marker
    seen.clear(); out = io.StringIO()
    assert GS.run(steps, runner_for([R(0, "ZZBRIDGE_PASS"), R(0, "ZZBUNDLE_OK 16"), R(0, "ZZBUNDLE_OK 904 9 s")]), out) == 3                          # 904 is not 905


# ------------------------------------------------------------------------------------------------------------------------------ C-ID (spc_cid + the chain's gate)
KEYS_SHAPES = {"exec": (16, 3, 20, 3), "scorer_in": (16, 3, 20, 3), "theta": (16, 3, 2), "moved": (16, 3), "hist_best_agg": (16, 3, 5), "hist_mu": (16, 3, 5, 2), "pred_logit_out": (16, 3, 6),
               "pred_logit_pick": (16, 6), "arm_argmax": (16,), "pick_exec": (16, 20, 3), "pick": (16,)}
BK_SHAPES = {"traj_raw": (16, 4, 20, 3), "logits": (16, 4, 6), "pick": (16,), "sub": (16, 4, 6), "pdms": (16, 4), "valid": (16, 4)}


def planted(seed=0):
    rng = np.random.default_rng(seed)
    toks = np.array(_tokens("CID"))
    s = {"token": toks}
    for k, sh in KEYS_SHAPES.items():
        s[k] = (rng.random(sh) > 0.5) if k == "moved" else (rng.integers(0, 4, sh) if k in ("arm_argmax", "pick") else rng.standard_normal(sh).astype(np.float32))
    s["moved"][:8, 0] = True
    s["moved"][8:] = False
    b = {"token": toks}
    for k, sh in BK_SHAPES.items():
        b[k] = (rng.random(sh) > 0.5) if k == "valid" else (rng.integers(0, 4, sh) if k == "pick" else rng.standard_normal(sh).astype(np.float32))
    ctx = rng.integers(0, 65535, (16, 3, 4)).astype(np.uint16)
    return s, b, ctx


def shuffled(d, ctx, perm):
    return {k: np.asarray(v)[perm] for k, v in d.items()}, ctx[perm]


def test_cid_green_and_token_mapping():
    s, b, ctx = planted()
    perm = np.random.default_rng(1).permutation(16)
    s2, _ = shuffled(s, ctx, perm)
    b2, ctx2 = shuffled(b, ctx, perm)
    r = CID.check(s, s2, b, b2, ctx, ctx2)
    assert r["pass"] and r["non_vacuity"]["tokens_banked_search_moved"] == 8 and r["non_vacuity"]["tokens_never_moved"] == 8


@pytest.mark.parametrize("key", list(KEYS_SHAPES))
def test_cid_red_one_bit_in_each_search_array(key):
    s, b, ctx = planted()
    s2 = {k: np.array(v, copy=True) for k, v in s.items()}
    a = s2[key]
    flat = a.reshape(-1).view(np.uint8) if a.dtype != np.bool_ else a.reshape(-1)
    if a.dtype == np.bool_:
        flat[0] = ~flat[0]
    else:
        flat[0] ^= 1
    r = CID.check(s, s2, b, {k: np.array(v, copy=True) for k, v in b.items()}, ctx, ctx.copy())
    assert not r["pass"] and any(key in f for f in r["failures"])


@pytest.mark.parametrize("key", list(BK_SHAPES))
def test_cid_red_one_bit_in_each_bundle_array(key):
    s, b, ctx = planted()
    b2 = {k: np.array(v, copy=True) for k, v in b.items()}
    a = b2[key]
    flat = a.reshape(-1).view(np.uint8) if a.dtype != np.bool_ else a.reshape(-1)
    if a.dtype == np.bool_:
        flat[0] = ~flat[0]
    else:
        flat[0] ^= 1
    r = CID.check(s, {k: np.array(v, copy=True) for k, v in s.items()}, b, b2, ctx, ctx.copy())
    assert not r["pass"] and any(key in f for f in r["failures"])


def test_cid_red_ctx_bit_and_negative_zero_and_vacuous_and_unknown_token():
    s, b, ctx = planted()
    cp = lambda d: {k: np.array(v, copy=True) for k, v in d.items()}   # noqa: E731
    ctx2 = ctx.copy(); ctx2[3, 0, 0] ^= 1
    assert not CID.check(s, cp(s), b, cp(b), ctx, ctx2)["pass"]
    s2 = cp(s); s2["theta"][0, 0, 0] = 0.0; s["theta"][0, 0, 0] = -0.0                  # -0.0 vs +0.0 is a difference (bitwise)
    assert not CID.check(s, s2, b, cp(b), ctx, ctx.copy())["pass"]
    s, b, ctx = planted()
    s["moved"][:] = False                                                                # a sample the banked search never moved: the identity would be a pick with itself
    r = CID.check(s, cp(s), b, cp(b), ctx, ctx.copy())
    assert not r["pass"] and any("vacuous" in f for f in r["failures"])
    s, b, ctx = planted()
    s3 = cp(s); s3["token"] = s3["token"].copy(); s3["token"][0] = "0123456789abcdef"
    assert not CID.check(s, s3, b, cp(b), ctx, ctx.copy())["pass"]
    b3 = cp(b); b3["token"] = b3["token"][::-1].copy()                                    # search and bundle on different token sets/orders is mapped by token, so same set passes the mapping step ...
    b3["token"][0] = "0123456789abcdef"                                                   # ... but a different SET is refused
    assert not CID.check(s, cp(s), b, b3, ctx, ctx.copy())["pass"]


def test_cid_red_search_and_bundle_on_different_token_sets():
    s, b, ctx = planted()
    cp = lambda d: {k: np.array(v, copy=True) for k, v in d.items()}   # noqa: E731
    b2 = {k: np.asarray(v)[:15].copy() for k, v in b.items()}          # the re-run bundle lost one token the search has; every other comparison is by token and would pass
    r = CID.check(s, cp(s), b, b2, ctx, ctx[:15].copy())
    assert not r["pass"] and any("same tokens" in f for f in r["failures"])


def test_cid_banked_rows_through_real_schema():
    s, b, ctx = planted()
    r = CID.check(s, {k: np.array(v, copy=True) for k, v in s.items()}, b, {k: np.array(v, copy=True) for k, v in b.items()}, ctx, ctx.copy())
    assert r["pass"]


@real
def test_cid_green_on_the_real_banked_rows():
    """the real banked search rows of the 16 C-ID tokens, re-presented as a 'rerun' in another order, pass; a one-bit flip of a real moved row fails (the comparator sees real data)"""
    bs = {k: v for k, v in np.load(BANKED_SEARCH, allow_pickle=False).items() if k != "meta"}
    bb = {k: v for k, v in np.load(BANKED_BUNDLE, allow_pickle=False).items()}
    ctx = np.load(os.path.join(CH.BANKED_BUNDLE_DIR, "ctx.npy"), mmap_mode="r")
    cid = _tokens("CID")[::-1]
    ix = {str(t): i for i, t in enumerate(bs["token"])}
    rows = np.array([ix[t] for t in cid])
    s2 = {k: np.asarray(v)[rows] for k, v in bs.items() if k not in ("bundle_index", "seeds")}
    ib = {str(t): i for i, t in enumerate(bb["token"])}
    brow = np.array([ib[t] for t in cid])
    b2 = {k: np.asarray(v)[brow] for k, v in bb.items()}
    ctx2 = np.stack([np.asarray(ctx[i]) for i in brow])
    r = CID.check(bs, s2, bb, b2, ctx, ctx2)
    assert r["pass"] and r["non_vacuity"]["tokens_banked_search_moved"] == 8
    s2["pred_logit_out"] = s2["pred_logit_out"].copy()
    m = np.argwhere(np.asarray(s2["moved"]))[0]
    s2["pred_logit_out"][m[0], m[1], 0] = np.nextafter(s2["pred_logit_out"][m[0], m[1], 0], np.inf)
    assert not CID.check(bs, s2, bb, b2, ctx, ctx2)["pass"]


def _meta(**kw):
    m = {"arm": "P2HEAD", "head_sha256": None, "ckpt_sha256": "c" * 64, "config_sha256": "d" * 64, "data_clause": {"carve_sha256": SL.READS["CID"]}}
    m.update(kw)
    return json.dumps(m)


def plant_cid_run(ch, s, b, ctx, banked_dir, meta=None, rerun_mut=None):
    c = ch.cfg
    os.makedirs(banked_dir, exist_ok=True)
    os.makedirs(c.work, exist_ok=True)
    np.savez(os.path.join(banked_dir, "bundle.npz"), **b); np.save(os.path.join(banked_dir, "ctx.npy"), ctx)
    np.savez(os.path.join(str(c.work), "banked_search.npz"), **s)
    c.banked_search, c.banked_bundle_dir = os.path.join(str(c.work), "banked_search.npz"), banked_dir
    s2 = {k: np.array(v, copy=True) for k, v in s.items()}
    if rerun_mut:
        rerun_mut(s2)
    os.makedirs(c.work, exist_ok=True)
    np.savez(c.d("search_CID_P2HEAD.npz"), meta=np.array(meta or _meta()), **s2)
    plant_bundle(c.work, "CID")
    # the re-run bundle must carry the same arrays as the banked one: write them over the planted token-only bundle
    np.savez(os.path.join(c.work, "cid", "bundle", "bundle.npz"), **b); np.save(os.path.join(c.work, "cid", "bundle", "ctx.npy"), ctx)
    cj = os.path.join(c.work, "cid", "bundle", "ctx.json")
    j = json.load(open(cj)); j["ctx_sha256"] = CH.sha256_file(os.path.join(c.work, "cid", "bundle", "ctx.npy")); json.dump(j, open(cj, "w"))


PINS0 = {"ckpt_sha256": "c" * 64, "config_sha256": "d" * 64}


def test_chain_cid_gate_green_and_red(tmp_path, good_text):
    ch = chain_for(tmp_path, good_text)
    s, b, ctx = planted()
    plant_cid_run(ch, s, b, ctx, os.path.join(str(tmp_path), "banked"))
    ch.cid_gate(PINS0)
    assert "ZZSPC_CID_PASS" in ch.out.getvalue()
    # RED: one bit of one banked-vs-rerun search value -> the chain stops with exit 3
    def flip(s2):
        s2["theta"][0, 0, 0] = np.nextafter(s2["theta"][0, 0, 0], np.float32(9))
    d2 = os.path.join(str(tmp_path), "b")
    os.makedirs(d2)
    ch2 = chain_for(d2, good_text)
    plant_cid_run(ch2, s, b, ctx, os.path.join(d2, "banked2"), rerun_mut=flip)
    e = stops(ch2.cid_gate, PINS0)
    assert e.code == 3 and e.reason == "G_CID_NOT_PASSED"


@pytest.mark.parametrize("kind", ["head_set", "arm_L2", "wrong_carve", "wrong_ckpt", "wrong_config"])
def test_chain_nohead_and_meta_red(tmp_path, good_text, kind):
    ch = chain_for(tmp_path, good_text)
    s, b, ctx = planted()
    meta = {"head_set": _meta(head_sha256="a" * 64), "arm_L2": _meta(arm="L2"), "wrong_carve": _meta(data_clause={"carve_sha256": SL.READS["C1"]}), "wrong_ckpt": _meta(ckpt_sha256="0" * 64),
            "wrong_config": _meta(config_sha256="0" * 64)}[kind]
    plant_cid_run(ch, s, b, ctx, os.path.join(str(tmp_path), "banked"), meta=meta)
    ch.run_child_fn = Recorder(rc=0)
    e = stops(ch.cid_gate, PINS0)
    assert e.code == 6 and e.reason == "SEARCH_MERGE_INVALID"


# ------------------------------------------------------------------------------------------------------------------------------ G-REACH, G-LABEL, G-ORDER
def test_reach_red_no_stamp_and_wrong_bundle_stamp(tmp_path, good_text):
    ch = chain_for(tmp_path, good_text)
    plant_bundle(ch.cfg.work, "CID"); plant_bundle(ch.cfg.work, "C1")
    pins = {"ckpt_sha256": "c" * 64, "config_sha256": "d" * 64, "head_s0_sha256": "e" * 64}
    rec = Recorder(rc=0)                                       # the gate child 'exits 0' but writes no stamp
    ch.run_child_fn = rec
    e = stops(ch.mint_gates, pins)
    assert e.code == 3 and e.reason == "G_REACH_NOT_PASSED"
    argv, env, _ = rec.calls[0]
    assert env["CUDA_VISIBLE_DEVICES"] == "-1" and argv[3] == "gate" and "--read" in argv and argv[argv.index("--read") + 1] == "CID"
    gd = ch.cfg.d("gate_CID"); os.makedirs(gd, exist_ok=True)
    json.dump({"schema": "tanitad.g_reach_stepb/1", "verdict": "PASS", "bundle_sha256": "0" * 64, "ckpt_sha256": "c" * 64, "config_sha256": "d" * 64}, open(os.path.join(gd, "G-REACH-STEPB_PASS_x.json"), "w"))
    assert not ch.stamp_ok("CID", pins)                         # a stamp minted on ANOTHER bundle certifies nothing about this one
    json.dump({"schema": "tanitad.g_reach_stepb/1", "verdict": "FAIL", "bundle_sha256": CH.sha256_file(os.path.join(ch.bundle_dir("CID"), "bundle.npz")), "ckpt_sha256": "c" * 64, "config_sha256": "d" * 64},
              open(os.path.join(gd, "G-REACH-STEPB_PASS_x.json"), "w"))
    assert not ch.stamp_ok("CID", pins)
    json.dump({"schema": "tanitad.g_reach_stepb/1", "verdict": "PASS", "bundle_sha256": CH.sha256_file(os.path.join(ch.bundle_dir("CID"), "bundle.npz")), "ckpt_sha256": "c" * 64, "config_sha256": "d" * 64},
              open(os.path.join(gd, "G-REACH-STEPB_PASS_x.json"), "w"))
    assert ch.stamp_ok("CID", pins)                             # GREEN control


def plant_labels(path, tokens, ident=None, nan=False, tokens_lab=None):
    n = len(tokens)
    pd = np.full((n, 3), 0.5)
    if nan:
        pd[0, 0] = np.nan
    ident = ident if ident is not None else 'ZZIDENTITY ' + json.dumps({"n_checked": n, "max_abs_diff": 0.0, "passed": True, "control_ok": True})
    np.savez(path, token=np.array(tokens_lab if tokens_lab is not None else tokens), pdms=pd, sub=np.zeros((n, 3, 6)), identity=np.array(ident))


@pytest.mark.parametrize("kind,ok", [("green", True), ("identity_failed", False), ("control_not_ok", False), ("nan", False), ("tokens_differ", False), ("no_identity", False)])
def test_g_label_red_arms(tmp_path, good_text, kind, ok):
    ch = chain_for(tmp_path, good_text)
    toks = ["a%015d" % i for i in range(5)]
    sp = os.path.join(str(tmp_path), "s.npz"); lp = os.path.join(str(tmp_path), "l.npz")
    np.savez(sp, token=np.array(toks))
    ident = {"green": None, "identity_failed": 'ZZ ' + json.dumps({"passed": False, "control_ok": True}), "control_not_ok": 'ZZ ' + json.dumps({"passed": True, "control_ok": False}),
             "nan": None, "tokens_differ": None, "no_identity": "no json here"}[kind]
    plant_labels(lp, toks, ident=ident, nan=(kind == "nan"), tokens_lab=(toks[::-1] if kind == "tokens_differ" else None))
    assert ch.labels_valid(lp, sp)[0] is ok


@real
def test_g_label_green_on_the_real_banked_labels(tmp_path, good_text):
    ch = chain_for(tmp_path, good_text)
    assert ch.labels_valid(BANKED_LABELS, BANKED_SEARCH) == (True, "")


def test_g_label_run_labels_refuses_low_ram_and_failed_identity(tmp_path, good_text):
    ch = chain_for(tmp_path, good_text)
    toks = ["a%015d" % i for i in range(5)]
    sp = os.path.join(str(tmp_path), "s.npz"); np.savez(sp, token=np.array(toks))
    ch.ram_fn = lambda: 2.0
    e = stops(ch.run_labels, sp)
    assert e.code == 7 and e.reason == "RAM_LOW"
    ch.ram_fn = lambda: 64.0
    ch.run_child_fn = Recorder(rc=3)                                      # the labeller exits 3 = the planted identity failed: no labels are accepted
    e = stops(ch.run_labels, sp)
    assert e.code == 7 and e.reason == "G_LABEL_NOT_PASSED"


def test_g_order_analysis_refused_unless_every_stage_is_done(tmp_path, good_text):
    ch = chain_for(tmp_path, good_text)
    ch.st["done"] = ["SEAL_OK", "DATA_OK", "GPU_STAGE_OK", "REACH_PASS", "LABEL_DONE"]           # CID_PASS missing
    ch.run_child_fn = NoChild()
    e = stops(ch.run_analysis, "m", "l", PINS0)
    assert e.code == 8 and e.reason == "G_ORDER" and "CID_PASS" in e.detail


def test_chain_never_runs_the_admission_gate():
    src = open(os.path.join(HERE, "spc_chain.py"), encoding="utf-8").read()
    assert "ep_admission" not in src


# ------------------------------------------------------------------------------------------------------------------------------ the C1 classifier (a branch table; every term has a RED arm)
def DP(point, lo, hi): return {"point": point, "lo": lo, "hi": hi}
SAFE = {"nc": 0.0, "dac": 0.17, "ttc": 0.25, "c": 0.0}
NOT_INERT = {"inert": False}


@pytest.mark.parametrize("dp,rng,safety,inert,want", [
    (DP(0.33, 0.06, 0.72), 0.216, SAFE, NOT_INERT, "O1_CONFIRMED"),
    (DP(0.30, 0.01, 0.6), 0.1, SAFE, NOT_INERT, "O1_CONFIRMED"),                         # the bar is >= 0.30
    (DP(0.2999, 0.01, 0.6), 0.1, SAFE, NOT_INERT, "O2_DIRECTIONAL"),                     # point just below the bar
    (DP(0.33, 0.0, 0.7), 0.1, SAFE, NOT_INERT, "O3_NULL"),                                # lower CI exactly 0 is not > 0
    (DP(0.33, -0.05, 0.7), 0.1, SAFE, NOT_INERT, "O3_NULL"),
    (DP(0.33, 0.06, 0.72), 0.33, SAFE, NOT_INERT, "O2_DIRECTIONAL"),                      # point not above the seed range
    (DP(0.33, 0.06, 0.72), 0.5, SAFE, NOT_INERT, "O2_DIRECTIONAL"),
    (DP(0.33, 0.06, 0.72), 0.1, {**SAFE, "ttc": -0.2501}, NOT_INERT, "TRADE"),
    (DP(0.33, 0.06, 0.72), 0.1, {**SAFE, "nc": -0.25}, NOT_INERT, "O1_CONFIRMED"),        # exactly -0.25 passes (>=)
    (DP(0.33, 0.06, 0.72), 0.1, {**SAFE, "dac": -0.26}, NOT_INERT, "TRADE"),
    (DP(0.33, 0.06, 0.72), 0.1, {**SAFE, "c": -0.3}, NOT_INERT, "TRADE"),
    (DP(-0.4, -0.7, -0.1), 0.1, SAFE, NOT_INERT, "NEGATIVE"),
    (DP(-0.4, -0.7, 0.0), 0.1, SAFE, NOT_INERT, "O3_NULL"),                               # upper CI exactly 0 is not < 0
    (DP(0.1, 0.02, 0.2), 0.05, {**SAFE, "dac": -0.5}, NOT_INERT, "O2_DIRECTIONAL"),       # directional with a safety miss is still O2, flagged
    (DP(0.33, 0.06, 0.72), 0.1, SAFE, {"inert": True}, "O4_INERT"),
])
def test_classify_c1_branch_table(dp, rng, safety, inert, want):
    r = AC.classify_c1(dp, rng, safety, inert)
    assert r["outcome"] == want
    assert want in AC.NEXT_ARM and AC.NEXT_ARM[want]                                       # every outcome has a NAMED next arm (RULE ZERO)


def test_classify_c1_flags_a_safety_miss_beside_a_directional_read():
    assert AC.classify_c1(DP(0.1, 0.02, 0.2), 0.05, {**SAFE, "dac": -0.5}, NOT_INERT)["safety_fail_beside_directional"] is True


def test_classify_c1_constants_are_the_sealed_step_b_constants():
    import analyse_stepB as AB
    assert AC.BAR_DPDMS == AB.BAR_DPDMS == 0.30 and AC.BAR_SAFETY == AB.BAR_SAFETY == -0.25 and AC.SAFETY_TERMS == AB.SAFETY_TERMS == ("nc", "dac", "ttc", "c")
    assert set(AC.OUTCOMES) == set(AC.NEXT_ARM) and "PAUSED" in AC.STRATEGIC


@real
def test_analyse_c1_reproduces_the_banked_step_b_p2head_read_and_its_concordance():
    """IDENTITY on real data: the C1 analysis applied to carve B's banked P2HEAD files returns Step B's published numbers (+0.3273 [+0.0565, +0.7216]); under the C1 rule that read is O1_CONFIRMED (the MM ruling:
    'meets every O1 condition'), and the reported-only A3 equals Amendment 1's banked P2HEAD value (1.000 on 14 pairs)"""
    import stepA as A
    b = A.load_npz(BANKED_BUNDLE)
    so = {k: v for k, v in np.load(BANKED_SEARCH, allow_pickle=False).items()}
    lab = {k: v for k, v in np.load(BANKED_LABELS, allow_pickle=False).items()}
    r = AC.analyse_c1(b, so, lab)
    e = r["endpoint_dPDMS_points"]
    assert abs(e["point"] - 0.3272945259675774) < 1e-12 and abs(e["lo"] - 0.0564630321976771) < 1e-12 and abs(e["hi"] - 0.7216309267075762) < 1e-12
    assert r["outcome"] == "O1_CONFIRMED" and abs(r["per_seed_points"]["range"] - 0.21604789242179448) < 1e-12
    c = r["concordance_reported_only"]
    assert c["pooled"]["A3"]["point"] == 1.0 and c["pooled"]["A3"]["den"] == 14.0 and c["pooled"]["A4"]["den"] == 50.0
    # RED arm: a planted offset of -1 point on every search label turns the same code's verdict into something else
    lab2 = dict(lab); lab2["pdms"] = np.asarray(lab["pdms"]) - 0.01
    assert AC.analyse_c1(b, so, lab2, with_concordance=False)["outcome"] != "O1_CONFIRMED"


@real
def test_analysis_refuses_labels_for_other_tokens():
    import stepA as A
    b = A.load_npz(BANKED_BUNDLE)
    so = {k: v for k, v in np.load(BANKED_SEARCH, allow_pickle=False).items()}
    lab = {k: v for k, v in np.load(BANKED_LABELS, allow_pickle=False).items()}
    lab["token"] = lab["token"][::-1].copy()
    with pytest.raises(ValueError):
        AC.analyse_c1(b, so, lab, with_concordance=False)


# ------------------------------------------------------------------------------------------------------------------------------ the design calc and the cost are what the prereg quotes
def test_design_calc_identity_and_structures():
    d = json.load(open(os.path.join(RAW, "design_calc.json")))
    assert d["identity"]["reproduces_RESULT_STEPB_P2HEAD"] is True
    assert d["structures"]["B_REST"]["n_tokens"] == 905 and d["structures"]["B_REST"]["n_logs"] == 24
    assert d["structures"]["navtest"]["n_tokens"] == 12146 and d["structures"]["navtest"]["n_logs"] == 136
    assert "U_s1b_actual" not in d["structures"], "READ C3 (U) is dropped"
    p = d["conditions"]["B_REST|R|mu=0.0"]["prob"]
    assert p["O1_verbatim"] < 0.02 and p["sign_pos"] < 0.04                                 # the false-pass rates at a true 0 are small


def test_cost_has_no_u_and_the_b_rest_gpu_minutes_are_about_25():
    c = json.load(open(os.path.join(RAW, "cost.json")))
    assert "e_unseen_stratum_U_READ_C3" not in c and 22.0 <= c["a_bundle_forward_B_REST"]["low_min"] <= 26.0
    assert c["sizes"]["B_REST_tokens"] == 905 and c["d_navtest"]["tokens"] == 12146


def test_gpu_stage_dry_run_on_the_real_inputs(tmp_path, capsys):
    """CPU, no model: every input the GPU stage needs exists, the bundle script is the certified one, and the plan prints (a RED arm: a wrong certified sha is refused)"""
    if not (os.path.exists(GS.TABLE) and os.path.exists(GS.CKPT) and os.path.exists(os.path.join(GS.L1_CODE, "make_stepA_bundle.py"))):
        pytest.skip("real inputs absent")
    args = ["--work", str(tmp_path), "--cid-sample", os.path.join(RAW, "cid_sample.json"), "--brest-sample", os.path.join(RAW, "b_rest.json"), "--dry-run"]
    assert GS.main(args) == 0 and "ZZSTAGE_DRYRUN_OK" in capsys.readouterr().out
    old = GS.BUNDLE_SCRIPT_SHA256_LF
    try:
        GS.BUNDLE_SCRIPT_SHA256_LF = "0" * 64
        assert GS.main(args) == 3 and "not the certified one" in capsys.readouterr().out
    finally:
        GS.BUNDLE_SCRIPT_SHA256_LF = old
