"""Tests for the refcv8 queue (queue_after_5000.sh + queue_helpers_r8.py + slim_rolling_ckpt_r8.py + wrapper_clause_r8fix_gpu.amend).

Run:  python -m pytest -q -p no:cacheprovider test_queue_after_5000.py      (needs Git-for-Windows `sh`, the tanitad venv, the launch tree)

Every guard carries a REGRESSION arm (a deliberately wrong implementation / a failing stage that MUST be seen):
  * `derive_ema_model` is checked against the REAL `WeightEMA.model_state_dict` on a tiny model with a TIED weight; the naive derivation
    that skips aliases, and the one that returns the live weights, are shown to differ (so the test could see those defects).
  * the amended wrapper clause: a term over its bar flips the clause FAIL -> PASS only when it is exactly `r8_spd_crc`; W1 undetected -> VOID.
  * the queue control flow runs in a sandbox with stub scripts (no GPU, no real lock): ORDER of events, flags of the EMA pass, continue-on-
    failure (EMA gate fail / wrapper-fix crash must not stop the milestone stage and must release the lock), resume, single instance.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "C:/lgs/Lfb47a58376/stack")
BATTERY_CODE = "C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code"
sys.path.insert(0, BATTERY_CODE)
import queue_helpers_r8 as H  # noqa: E402
import slim_rolling_ckpt_r8 as S  # noqa: E402

SH = shutil.which("sh") or "C:/Program Files/Git/bin/sh.exe"
needs_sh = pytest.mark.skipif(not Path(SH).exists() and not shutil.which("sh"), reason="no sh")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


def msys(p: Path) -> str:
    s = str(p).replace("\\", "/")
    return "/" + s[0].lower() + s[2:] if len(s) > 1 and s[1] == ":" else s


# ------------------------------------------------------------------------------------------------ battstate
def put(p: Path, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_battstate_cases(tmp_path):
    log = tmp_path / "b.log"
    assert H.battstate(str(log), 5000) == "nostart:-:-"
    put(log, ["ZZR8BATTSTARTZZ 2026-10-16T06:51:46+0200 pid 64306 step 5000", "ZZR8BATTG0ENDEDZZ x", "ZZR8BATTLOCKEDZZ"])
    assert H.battstate(str(log), 5000) == "running:BATTLOCKED:64306"
    put(log, ["ZZR8BATTSTARTZZ t pid 64306 step 5000", "ZZR8BATTDONEZZ primary", "ZZR8BATTCORRDONEZZ"])
    assert H.battstate(str(log), 5000) == "running:BATTCORRDONE:64306"                      # CORRDONE is not the end: CHAINEND is
    put(log, ["ZZR8BATTSTARTZZ t pid 64306 step 5000", "ZZR8BATTCORRDONEZZ", "ZZR8BATTCHAINENDZZ"])
    assert H.battstate(str(log), 5000) == "ended:BATTCHAINEND:64306"
    for term in ("BATTSKIP", "BATTNOCOMMIT", "BATTNOLOCK", "BATTFAIL"):
        put(log, ["ZZR8BATTSTARTZZ t pid 7 step 5000", f"ZZR8{term}ZZ reason"])
        assert H.battstate(str(log), 5000) == f"ended:{term}:7"
    # CORRFAIL is NOT terminal (the chain still reaches its end after it)
    put(log, ["ZZR8BATTSTARTZZ t pid 7 step 5000", "ZZR8BATTCORRFAILZZ x"])
    assert H.battstate(str(log), 5000) == "running:BATTCORRFAIL:7"
    # the REAL history of this log: an old segment ended in FAIL, a RELAUNCH is running -> the old end must not end the new one
    put(log, ["ZZR8BATTSTARTZZ t pid 45849 step 5000", "ZZR8BATTFAILZZ old", "ZZR8BATTSTARTZZ t pid 64306 step 5000", "ZZR8BATTLOCKEDZZ"])
    assert H.battstate(str(log), 5000) == "running:BATTLOCKED:64306"
    put(log, ["ZZR8BATTSTARTZZ t pid 5 step 10500", "ZZR8BATTCHAINENDZZ"])
    assert H.battstate(str(log), 5000) == "nostart:-:-"                                      # another step's log never answers


# ------------------------------------------------------------------------------------------------ metrics_clean
def rows(*steps_eval, extra_train=2):
    out = [json.dumps({"step": i, "loss": 1.0}) for i in range(extra_train)]
    out += [json.dumps({"step": s, "eval_loss": 50.0 + s, "eval_traj": 0.5}) for s in steps_eval]
    return out


def test_metrics_clean(tmp_path):
    src, dst = tmp_path / "m.jsonl", tmp_path / "out" / "metrics_x.jsonl"
    src.write_text("\n".join(rows(5000, 10500)) + "\n" + '{"step": 10501, "loss": 1.', encoding="utf-8")        # torn tail
    assert H.metrics_clean(str(src), 10500, str(dst)) == "ok:4:dropped_torn=1"
    assert [json.loads(x)["step"] for x in dst.read_text().splitlines()] == [0, 1, 5000, 10500]            # tail gone, rest intact
    # REGRESSION arms: each must be REFUSED, and must not write the destination
    for name, text, want in (
        ("torn middle", "\n".join(rows(10500)[:1] + ['{"bad": '] + rows(10500)[1:]) + "\n", "fail:line 2 of"),
        ("no eval row at the step", "\n".join(rows(5000)) + "\n", "fail:0 eval rows at step 10500"),
        ("two eval rows", "\n".join(rows(10500, 10500)) + "\n", "fail:2 eval rows at step 10500"),
        ("empty", "", "fail:empty"),
    ):
        d2 = tmp_path / f"{name.replace(' ', '_')}.jsonl"
        src.write_text(text, encoding="utf-8")
        assert H.metrics_clean(str(src), 10500, str(d2)).startswith(want), name
        assert not d2.exists(), name


# ------------------------------------------------------------------------------------------------ the EMA derivation vs the REAL WeightEMA
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(4, 4)
        self.b = torch.nn.Linear(4, 4)
        self.b.weight = self.a.weight                      # TIED: one tensor under two state_dict names
        self.bn = torch.nn.BatchNorm1d(4)                  # float buffers (covered) + an int buffer (live)
        self.frozen = torch.nn.Parameter(torch.arange(4.0), requires_grad=False)


def make_rolling(tmp_path):
    from tanitad.train.weight_ema import WeightEMA
    torch.manual_seed(0)
    m = Tiny()
    ema = WeightEMA(m, 0.5)
    for _ in range(3):
        with torch.no_grad():
            for p in m.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p))
            m.bn.running_mean.add_(1.0)
        ema.update(m)
    p = tmp_path / "ckpt.pt"
    torch.save({"model": m.state_dict(), "opt": {"x": 1}, "step": 3, "ema": ema.state_dict()}, p)
    return m, ema, p


def test_derive_ema_model_equals_the_trainer_function(tmp_path):
    m, ema, p = make_rolling(tmp_path)
    ck = torch.load(p, map_location="cpu", weights_only=False)                      # storage sharing survives the round trip
    want = ema.model_state_dict(m)
    got, st = S.derive_ema_model(ck["model"], ck["ema"])
    assert set(got) == set(want) and st["n_alias"] >= 1                              # the tied name was filled from the one shadow
    for k in want:
        assert torch.equal(got[k], want[k]), k
    assert not torch.equal(got["a.weight"], ck["model"]["a.weight"])                 # the EMA really differs from the live weight
    assert torch.equal(got["b.weight"], got["a.weight"])                             # tied names agree
    assert torch.equal(got["frozen"], ck["model"]["frozen"])                         # a frozen param is not shadowed: stays live
    assert got["bn.num_batches_tracked"].dtype == torch.long and torch.equal(got["bn.num_batches_tracked"], ck["model"]["bn.num_batches_tracked"])
    # REGRESSION arms: the two plausible wrong derivations differ from the trainer's, so this test can see them
    naive = {k: (ck["ema"]["shadow"][k].clone() if k in ck["ema"]["shadow"] else ck["model"][k].clone()) for k in ck["model"]}
    assert not torch.equal(naive["b.weight"], want["b.weight"])                      # skipping the alias leaves the LIVE tied weight
    assert not torch.equal(ck["model"]["a.weight"], want["a.weight"])                # returning the live weights is not the EMA
    bad = copy.deepcopy(ck["ema"])
    bad["shadow"]["ghost.weight"] = torch.zeros(2)
    with pytest.raises(ValueError, match="absent from the model"):
        S.derive_ema_model(ck["model"], bad)


def test_slim_cli_end_to_end_and_refusals(tmp_path):
    m, ema, p = make_rolling(tmp_path)
    out = tmp_path / "ckpt_out"
    out.mkdir()
    run = lambda *extra: subprocess.run([PY, "-I", str(HERE / "slim_rolling_ckpt_r8.py"), "--src", str(p), "--out-dir", str(out), *extra],
                                        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "C:/lgs/Lfb47a58376/stack"})
    r = run("--expect-step", "99")                                                    # wrong step: refused, nothing written
    j = json.loads(r.stdout.strip().splitlines()[-1])
    assert j["ok"] is False and "expected 99" in j["reason"] and not list(out.iterdir())
    r = run("--expect-step", "3", "--provenance", "thor_run=tanitad-thor:/x", "note=hello")
    j = json.loads(r.stdout.strip().splitlines()[-1])
    assert j["ok"] is True and j["step"] == 3 and j["provenance"]["note"] == "hello"
    raw, em = torch.load(out / "ckpt_3.pt", weights_only=False), torch.load(out / "ckpt_ema_3.pt", weights_only=False)
    assert set(raw) == {"model", "step"} and raw["step"] == 3                        # exactly the trainer's milestone layout
    assert set(em) == {"model", "step", "ema_decay", "num_updates", "kind"} and em["kind"] == "ema_model_only" and em["num_updates"] == 3
    assert torch.equal(em["model"]["a.weight"], ema.model_state_dict(m)["a.weight"])
    assert (out / "ckpt_3.provenance.json").exists() and "DERIVED" in json.loads((out / "ckpt_3.provenance.json").read_text())["note"]
    j2 = json.loads(run("--expect-step", "3").stdout.strip().splitlines()[-1])         # a second run must NOT silently overwrite
    assert j2["ok"] is False and "exists" in j2["reason"]


# ------------------------------------------------------------------------------------------------ the amended wrapper clause
def raw_arms(spd_crc_wrapper=0.1, w1_detected=True):
    base = {"loss": 10.0, "goal_score_absmean": 1.0, "r8_spd_crc": 5.0}
    arms = {a: dict(base) for a in ("plain", "repeat", "permuted", "micro13", "micro31", "W1", "W2")}
    for a in ("micro13", "micro31"):
        arms[a]["loss"] = 10.0 * (1 + 1e-8)
        arms[a]["r8_spd_crc"] = 5.0 * (1 + spd_crc_wrapper)
    arms["permuted"]["r8_spd_crc"] = 5.0                         # floor 0 (a CRC that changes only under REORDERING is a different story on GPU)
    arms["W2"]["loss"] = 11.0
    if w1_detected:
        arms["W1"]["goal_score_absmean"] = 2.0
    return {"raw": arms}


def test_amended_wrapper_clause():
    import wrapper_clause_r8fix_gpu as WF
    import wrapper_probe_r7 as W
    cond = raw_arms()
    unamended = W._analyse(cond["raw"], include_mutations=True)
    assert WF.clause_of(unamended) == "FAIL" and unamended["terms_over_bar"] == ["r8_spd_crc"]            # the CRC alone breaks it
    am = WF.amend(cond)
    assert WF.clause_of(am) == "PASS" and am["terms_over_bar"] == [] and am["dropped_terms"] == ["r8_spd_crc"]
    assert abs(am["max_wrapper_rel"] - 1e-8) < 1e-9 and "r8_spd_crc" not in am["terms"]                   # known value: the loss term's 1e-8
    assert am["W1_detected"] and am["W2_detected"]
    # REGRESSION arms: dropping the CRC must not hide a REAL merge defect, and a blind instrument is VOID, not PASS
    cond2 = raw_arms()
    cond2["raw"]["micro13"]["loss"] = 10.0 * 1.01                                                          # a real 1 % merge defect
    assert WF.clause_of(WF.amend(cond2)) == "FAIL" and WF.amend(cond2)["terms_over_bar"] == ["loss"]
    assert WF.clause_of(WF.amend(raw_arms(w1_detected=False))) == "VOID"
    s = WF.summarise(am)
    assert s["clause"] == "PASS" and "loss" in s["key_terms"] and s["key_terms"]["loss"]["bar"] == 1e-5


# ------------------------------------------------------------------------------------------------ the queue, in a sandbox
STUBS_PY = {
    "wrapper_clause_r8fix_gpu.py": '''
import json, os, sys
a = sys.argv[1:]; v = lambda f: a[a.index(f) + 1]
open(os.environ["STUB_CALLS"], "a").write("wrapfix ckpt=%s\\n" % os.path.basename(v("--ckpt")))
if os.environ.get("STUB_WRAPFIX_FAIL") == "1": raise SystemExit("boom")
import hashlib
json.dump({"ok": True, "CLAUSE": "PASS", "max_wrapper_rel": 3e-6, "ckpt_md5": hashlib.md5(open(v("--ckpt"), "rb").read()).hexdigest()}, open(v("--out"), "w"))
''',
    "ema_check_r8.py": '''
import json, os, sys
a = sys.argv[1:]; v = lambda f: a[a.index(f) + 1]
open(os.environ["STUB_CALLS"], "a").write("ema_check ema=%s\\n" % os.path.basename(v("--ema")))
ident = os.environ.get("STUB_EMA_IDENTICAL") == "1"
os.makedirs(os.path.dirname(v("--out")), exist_ok=True)
import hashlib
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
json.dump({"ok": not ident, "md5": {"raw": md5(v("--raw")), "ema": md5(v("--ema"))}, "tensors": {"n_float_differing": 0 if ident else 821},
           "strict_load": {"missing": 0, "unexpected": 0}}, open(v("--out"), "w"))
''',
    "run_battery_r8.py": '''
import json, os, sys
a = sys.argv[1:]; v = lambda f: a[a.index(f) + 1]
open(os.environ["STUB_CALLS"], "a").write("run_battery ckpt=%s tag=%s nog0=%s nobase=%s nopose=%s seam=%s skiproll=%s lockheld=%s seeds=%s\\n" % (
    os.path.basename(v("--ckpt")), v("--tag"), "--no-g0" in a, "--no-baselines" in a, os.environ.get("R8_NO_POSE_SYNC"),
    os.path.basename(os.environ.get("R8_SEAM_TELE_DIR", "-")), "--skip-roll" in a, "--lock-held" in a, v("--infer-seeds")))
d = os.path.join(v("--out-root"), v("--tag")); os.makedirs(d, exist_ok=True)
json.dump({"bars": {"x": 1}, "finished": "now", "r8": {}}, open(os.path.join(d, "battery_summary.json"), "w"))
''',
    "pair_read_r8.py": '''
import json, os, sys
a = sys.argv[1:]; v = lambda f: a[a.index(f) + 1]
open(os.environ["STUB_CALLS"], "a").write("pair_read clock=%s g0json=%s\\n" % (v("--clock"), "--g0-json" in a))
json.dump({"clock": v("--clock")}, open(v("--out-json"), "w")); open(v("--out-md"), "w").write("x")
''',
    "bank_battery_r8.py": '''
import json, os, sys
a = sys.argv[1:]; v = lambda f: a[a.index(f) + 1]
open(os.environ["STUB_CALLS"], "a").write("bank tag=%s passes=%s\\n" % (v("--tag"), v("--passes")))
os.makedirs(a[1], exist_ok=True)
json.dump({"passes": {"unc": {"status": "COMPLETE"}}, "chain_status": v("--status")}, open(os.path.join(a[1], "BATTERY_SUMMARY_%s.json" % v("--tag")), "w"))
''',
}
STUBS_BAT = {
    "gpu_lock.py": '''
import json, os, sys
a = sys.argv[1:]; job = a[a.index("--job") + 1]
open(os.environ["STUB_CALLS"], "a").write("gpu_lock %s %s\\n" % (a[0], job))
if a[0] == "acquire": json.dump({"ok": os.environ.get("STUB_NOLOCK") != "1"}, open(a[a.index("--out") + 1], "w"))
''',
    "wait_commit.py": '''
import json, sys
a = sys.argv[1:]; json.dump({"met": True}, open(a[a.index("--out") + 1], "w"))
''',
    "chain_marker.py": "import sys\n",
    "gpu_watch.py": "import sys\n",
}
STUBS_SH = {
    "thor_pull_r8.sh": '''#!/bin/sh
echo "thor_pull $1" >> "$STUB_CALLS"
if [ "$STUB_PULL_FAIL" = 1 ]; then echo "pull:fail:no ckpt"; exit 1; fi
echo "raw$1" > "$R8_KIT/ckpt/ckpt_$1.pt"; echo "ema$1" > "$R8_KIT/ckpt/ckpt_ema_$1.pt"
echo "pull:ok:rolling"
''',
    "chain_g0_r8.sh": '''#!/bin/sh
echo "chain_g0 $1" >> "$STUB_CALLS"
mkdir -p "$R8_KIT/battery/step$1"
echo '{"ckpt_md5":"abc","verdict":{"G0":"FAIL","mutation_detection":{"m1":{"detected":true}}}}' > "$R8_KIT/battery/step$1/g0.json"
''',
    "chain_battery_r8.sh": '''#!/bin/sh
echo "chain_battery $1 corrected=$R8_CORRECTED seam=$(basename "$R8_SEAM_TELE_DIR")" >> "$STUB_CALLS"
mkdir -p "$R8_PKGL/raw/step$1"
echo '{"passes":{"unc":{"status":"COMPLETE"}}}' > "$R8_PKGL/raw/step$1/BATTERY_SUMMARY_step$1.json"
''',
}


def sandbox(tmp_path):
    pkg, kit, bat = tmp_path / "pkg", tmp_path / "kit", tmp_path / "bat"
    (pkg / "code").mkdir(parents=True)
    (pkg / "raw" / "step5000").mkdir(parents=True)
    (kit / "chain").mkdir(parents=True)
    (kit / "ckpt").mkdir()
    (kit / "battery" / "step5000").mkdir(parents=True)
    bat.mkdir()
    shutil.copy(HERE / "queue_helpers_r8.py", pkg / "code")
    shutil.copy(HERE / "queue_after_5000.sh", pkg / "code")
    for n, t in STUBS_PY.items():
        (pkg / "code" / n).write_text(t, encoding="utf-8")
    for n, t in STUBS_SH.items():
        (pkg / "code" / n).write_text(t, encoding="utf-8", newline="\n")
    for n, t in STUBS_BAT.items():
        (bat / n).write_text(t, encoding="utf-8")
    for n in ("ckpt_5000.pt", "ckpt_ema_5000.pt", "config.json"):
        (kit / "ckpt" / n).write_bytes(b"x")
    (kit / "battery" / "step5000" / "g0.json").write_text('{"ckpt_md5":"abc","verdict":{"G0":"FAIL"}}', encoding="utf-8")
    env = {**os.environ, "R8_PKGW": str(pkg).replace("\\", "/"), "R8_PKGL": msys(pkg), "R8_BAT": str(bat).replace("\\", "/"),
           "R8_KITW": str(kit).replace("\\", "/"), "R8_KIT": msys(kit), "R8_PY": PY, "STUB_CALLS": str(tmp_path / "calls.log"),
           "Q_NOWAIT": "1", "R8_POLL_S": "0.2"}
    return pkg, kit, env


def run_queue(pkg, env, timeout=120, **extra):
    e = {**env, **{k: str(v) for k, v in extra.items()}}
    r = subprocess.run([SH, msys(pkg / "code" / "queue_after_5000.sh")], env=e, capture_output=True, text=True, timeout=timeout)
    return r


def calls(env):
    p = Path(env["STUB_CALLS"])
    return p.read_text().splitlines() if p.exists() else []


def markers(kit):
    p = kit / "chain" / "queue_after_5000.log"
    return [ln.split()[0] for ln in p.read_text(encoding="utf-8").splitlines() if ln.startswith("ZZQ")] if p.exists() else []


@needs_sh
def test_queue_full_order_and_flags(tmp_path):
    pkg, kit, env = sandbox(tmp_path)
    r = run_queue(pkg, env)
    c, m = calls(env), markers(kit)
    assert m[-1] == "ZZQQUEUEENDZZ", (r.stderr[-500:], m)
    want = ["gpu_lock acquire refcv8-wrapfix-step5000", "wrapfix ckpt=ckpt_5000.pt", "gpu_lock release refcv8-wrapfix-step5000",
            "ema_check ema=ckpt_ema_5000.pt",
            "gpu_lock acquire refcv8-battery-step5000ema",
            "run_battery ckpt=ckpt_ema_5000.pt tag=step5000ema_unc nog0=True nobase=False nopose=1 seam=seam_tele_step5000ema skiproll=True lockheld=True seeds=0,1",
            "pair_read clock=standalone g0json=False", "bank tag=step5000ema passes=unc", "gpu_lock release refcv8-battery-step5000ema",
            "thor_pull 10500", "ema_check ema=ckpt_ema_10500.pt", "chain_g0 10500",
            "gpu_lock acquire refcv8-wrapfix-step10500", "wrapfix ckpt=ckpt_10500.pt", "gpu_lock release refcv8-wrapfix-step10500",
            "chain_battery 10500 corrected=0 seam=seam_tele_step10500"]
    assert c == want, "\n".join(c)                                           # lock BEFORE the work, release AFTER, EMA flags exact, pass 1 only
    assert (pkg / "raw/step5000/wrapper_clause_r8fix_gpu.json").exists()
    nog0 = json.loads((pkg / "raw/step5000ema/NO_G0_REASON_step5000ema.json").read_text())
    assert nog0["status"] == "PROVISIONAL" and "G0-checked on the raw weights" in nog0["no_g0_reason"]
    for tok in ("ZZQWRAPFIXOKZZ", "ZZQEMAGATEOKZZ", "ZZQEMADONEZZ", "ZZQMSPULLOKZZ", "ZZQMSG0OKZZ", "ZZQMSDONEZZ"):
        assert tok in m, (tok, m)
    # RESUME: a second run finds every artifact and does no work
    Path(env["STUB_CALLS"]).unlink()
    run_queue(pkg, env)
    assert calls(env) == [], calls(env)
    m2 = markers(kit)
    assert "ZZQWRAPFIXHAVEZZ" in m2 and "ZZQEMAHAVEZZ" in m2 and "ZZQMSHAVEZZ" in m2


@needs_sh
def test_queue_regression_arms_continue_on_failure(tmp_path):
    # (a) the EMA arm is IDENTICAL to raw: the stage STOPS before the lock, and the milestone stage still runs
    pkg, kit, env = sandbox(tmp_path)
    run_queue(pkg, env, STUB_EMA_IDENTICAL=1)
    c, m = calls(env), markers(kit)
    assert "ZZQEMAGATEFAILZZ" in m and "ZZQEMADONEZZ" not in m and "ZZQSTAGEFAILEDZZ" in m
    assert not any("battery-step5000ema" in x for x in c) and not any(x.startswith("run_battery") for x in c)
    assert "chain_battery 10500 corrected=0 seam=seam_tele_step10500" in c and m[-1] == "ZZQQUEUEENDZZ"
    # (b) the wrapper-fix crashes: the lock is still RELEASED and the later stages still run
    pkg, kit, env = sandbox(tmp_path / "b")
    run_queue(pkg, env, STUB_WRAPFIX_FAIL=1)
    c, m = calls(env), markers(kit)
    assert c.index("gpu_lock release refcv8-wrapfix-step5000") > c.index("gpu_lock acquire refcv8-wrapfix-step5000")
    assert "ZZQWRAPFIXFAILZZ" in m and "ZZQEMADONEZZ" in m and "ZZQMSDONEZZ" in m
    # (c) the pull fails: no G0, no battery for the milestone, the queue still ends
    pkg, kit, env = sandbox(tmp_path / "c")
    run_queue(pkg, env, STUB_PULL_FAIL=1)
    c, m = calls(env), markers(kit)
    assert "ZZQMSPULLFAILZZ" in m and not any(x.startswith("chain_g0") or x.startswith("chain_battery") for x in c) and m[-1] == "ZZQQUEUEENDZZ"
    # (d) the card never frees (no lock): the stage logs NOLOCK and runs NOTHING that needs the card
    pkg, kit, env = sandbox(tmp_path / "d")
    run_queue(pkg, env, STUB_NOLOCK=1, Q_STAGES="wrapfix")
    c, m = calls(env), markers(kit)
    assert "ZZQNOLOCKZZ" in m and not any(x.startswith("wrapfix") for x in c)


@needs_sh
def test_queue_single_instance_and_wait(tmp_path):
    pkg, kit, env = sandbox(tmp_path)
    pidfile = tmp_path / "sl.pid"
    sl = subprocess.Popen([SH, "-c", f"echo $$ > {msys(pidfile)}; exec sleep 60"])        # a LIVE pid that `kill -0` can see (MSYS pid)
    try:
        for _ in range(50):
            if pidfile.exists() and pidfile.read_text().strip():
                break
            time.sleep(0.1)
        (kit / "chain" / "queue_after_5000.pid").write_text(pidfile.read_text().strip())
        run_queue(pkg, env, Q_STAGES="none")
        m = markers(kit)
        assert "ZZQDUPZZ" in m and "ZZQQUEUESTARTZZ" not in m                              # a second queue refuses to start
    finally:
        sl.kill()
    # REGRESSION arm: a STALE pid file (dead pid) must not block the queue, and the queue removes its own pid file at the end
    pkg, kit, env = sandbox(tmp_path / "stale")
    (kit / "chain" / "queue_after_5000.pid").write_text("999999")
    run_queue(pkg, env, Q_STAGES="none")
    assert "ZZQQUEUESTARTZZ" in markers(kit) and not (kit / "chain" / "queue_after_5000.pid").exists()
    # the WAIT: a battery log whose newest segment is RUNNING under a dead shell, silent -> CHAINDEAD after 5 polls, then the queue goes on
    pkg, kit, env = sandbox(tmp_path / "w")
    env.pop("Q_NOWAIT")
    put(kit / "chain" / "battery_step5000.log", ["ZZR8BATTSTARTZZ t pid 999999 step 5000", "ZZR8BATTLOCKEDZZ"])
    run_queue(pkg, env, Q_STAGES="none")
    assert "ZZQCHAINDEADZZ" in markers(kit) and markers(kit)[-1] == "ZZQQUEUEENDZZ"
    # ... and a terminal marker in the newest segment ends the wait at once (an OLDER segment's marker would not have)
    pkg, kit, env = sandbox(tmp_path / "w2")
    env.pop("Q_NOWAIT")
    put(kit / "chain" / "battery_step5000.log", ["ZZR8BATTSTARTZZ t pid 1 step 5000", "ZZR8BATTFAILZZ old", "ZZR8BATTSTARTZZ t pid 2 step 5000",
                                                 "ZZR8BATTCHAINENDZZ"])
    run_queue(pkg, env, Q_STAGES="none")
    assert "ZZQCHAINENDEDZZ" in markers(kit) and "ZZQCHAINDEADZZ" not in markers(kit)


@needs_sh
def test_queue_stale_artifacts_are_rerun_not_trusted(tmp_path):
    pkg, kit, env = sandbox(tmp_path)
    (pkg / "raw" / "step5000ema").mkdir(parents=True)
    (pkg / "raw/step5000ema/EMA_CHECK_step5000.json").write_text(json.dumps(
        {"ok": True, "md5": {"raw": "0" * 32, "ema": "1" * 32}, "tensors": {"n_float_differing": 5}, "strict_load": {"missing": 0, "unexpected": 0}}))
    (pkg / "raw/step5000/wrapper_clause_r8fix_gpu.json").write_text(json.dumps({"ok": True, "CLAUSE": "PASS", "ckpt_md5": "f" * 32}))
    run_queue(pkg, env, Q_STAGES="wrapfix ema")
    c = calls(env)
    assert "wrapfix ckpt=ckpt_5000.pt" in c and "ema_check ema=ckpt_ema_5000.pt" in c, c                  # both stale -> both re-run
    assert "ZZQWRAPFIXHAVEZZ" not in markers(kit) and "ZZQEMADONEZZ" in markers(kit)
