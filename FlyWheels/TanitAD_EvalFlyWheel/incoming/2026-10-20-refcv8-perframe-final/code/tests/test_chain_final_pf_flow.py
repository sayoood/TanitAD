"""Dry-run of chain_final_pf_r8.sh + wait_final_pf_r8.sh with STUB tools (no GPU, no lock file, no Thor, no roll).

Every expectation is a LITERAL marker / stub-log line.  What is pinned:
  * PF-FIRST: the per-clip chain is restarted only AFTER the pf GPU job released the lock, and never before the pf battery ran;
  * the REGISTRATION gate: no lock is taken and no battery runs unless REGISTERED_PF_TABLE.sha256 == sha256(frozen table) (missing / mismatched / empty -> ABORT R);
  * one retry per GPU stage, then halt (and the per-clip chain still comes back);
  * a live per-clip chain at start -> ABORT a (no second puller);
  * a failed pull -> ABORT a, per-clip restarted;
  * resumability: a second run takes no lock and re-runs nothing;
  * the guardian: refuses while the per-clip chain is alive; relaunches a chain that ended without a terminal marker ONCE, then restarts the per-clip chain itself.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1]
CHAIN = Path(os.environ.get("PF_TEST_CHAIN", str(CODE / "chain_final_pf_r8.sh")))   # PF_TEST_CHAIN: run the SAME pins against a deliberate mutant
GUARD = Path(os.environ.get("PF_TEST_GUARD", str(CODE / "wait_final_pf_r8.sh")))   # PF_TEST_GUARD: the same pins against a deliberate mutant
SH = shutil.which("sh") or "C:/Program Files/Git/usr/bin/sh.exe"
PY = os.environ.get("PF_TEST_PY", "C:/Users/Admin/venvs/tanitad/Scripts/python.exe")
TAG = "step48852ema_pf"

ARGS = r'''
import sys
def kv(a):
    return {a[i]: a[i + 1] for i in range(len(a) - 1) if a[i].startswith("--")}
'''
STUB_PULL = r'''#!/bin/sh
echo "ZZPFNSTARTZZ x pid $$" >> "$FIN_LOG"
echo "pull" >> "$STUB_LOG"
if [ "$STUB_PULL_FAIL" = 1 ]; then echo "ZZPFNTIMEOUTZZ x" >> "$FIN_LOG"; exit 6; fi
F="$FIN_CKPT_DIR/ckpt_full_$FIN_STEP.pt"
if [ -s "$F" ]; then echo "ZZPFNALREADYZZ x" >> "$FIN_LOG"; exit 0; fi
printf 'fullweights' > "$F"
M=$(md5sum "$F" | cut -d' ' -f1)
echo "$M  ckpt_full_$FIN_STEP.pt" > "$F.md5"
printf '{"md5": "%s", "step": %s}' "$M" "$FIN_STEP" > "$F.provenance.json"
printf '{"done": true, "step": %s}' "$FIN_STEP" > "$FIN_CKPT_DIR/summary_thor_$FIN_STEP.json"
echo "ZZPFNDONEZZ x ckpt_md5 $M size 11" >> "$FIN_LOG"
'''
STUB_EXTRACT = ARGS + r'''
import json, os
a = sys.argv[2:]; d = kv(a)
open(os.environ["STUB_LOG"], "a").write("extract\n")
src_md5 = [x.split("=", 1)[1] for x in a if x.startswith("thor_ckpt_pt_md5=")][0]
od = d["--out-dir"]; st = d["--expect-step"]
open(os.path.join(od, f"ckpt_{st}.pt"), "w").write("raw")
open(os.path.join(od, f"ckpt_ema_{st}.pt"), "w").write("ema")
json.dump({"ok": True, "step": int(st), "src_md5": src_md5}, open(os.path.join(od, f"ckpt_{st}.provenance.extract_ema.json"), "w"))
'''
STUB_EMACHK = ARGS + r'''
import json, os, hashlib
d = kv(sys.argv[1:])
open(os.environ["STUB_LOG"], "a").write("ema_check\n")
m = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
json.dump({"ok": True, "md5": {"raw": m(d["--raw"]), "ema": m(d["--ema"])}, "tensors": {"n_float_differing": 7}, "strict_load": {"missing": 0, "unexpected": 0}}, open(d["--out"], "w"))
'''
STUB_LOCK = ARGS + r'''
import json, os
cmd = sys.argv[1]; d = kv(sys.argv[2:])
open(os.environ["STUB_LOG"], "a").write(f"gpu_lock {cmd} {d.get('--job')}\n")
if cmd == "acquire":
    json.dump({"ok": os.environ.get("STUB_LOCK_FAIL") != "1"}, open(d["--out"], "w"))
'''
STUB_WC = ARGS + r'''
import json
json.dump({"met": True}, open(kv(sys.argv[1:])["--out"], "w"))
'''
STUB_MARK = r'''
import sys, os
open(os.environ["STUB_LOG"], "a").write("chain_marker " + sys.argv[1] + "\n")
'''
STUB_BATT = ARGS + r'''
import json, os
d = kv(sys.argv[1:])
open(os.environ["STUB_LOG"], "a").write(f"battery {d['--tag']} pose={os.environ.get('R8_NO_POSE_SYNC')} lockheld={'--lock-held' in sys.argv}\n")
if os.environ.get("STUB_BATT_FAIL") == "1":
    sys.exit(1)
w = os.path.join(d["--out-root"], d["--tag"]); os.makedirs(w, exist_ok=True)
json.dump({"bars": [1], "finished": "now", "r8": {}, "pf": {"all_seeds_fed": True}}, open(os.path.join(w, "battery_summary.json"), "w"))
'''
STUB_PAIR = ARGS + r'''
import json, os
d = kv(sys.argv[1:])
open(os.environ["STUB_LOG"], "a").write("pair_read\n")
json.dump({"bars": {}}, open(d["--out-json"], "w"))
'''
STUB_BANK = ARGS + r'''
import json, os
a = sys.argv[1:]
open(os.environ["STUB_LOG"], "a").write("bank\n")
os.makedirs(a[1], exist_ok=True)
json.dump({"passes": {"pf": {"status": "COMPLETE"}}}, open(os.path.join(a[1], "BATTERY_SUMMARY_step48852ema.json"), "w"))
'''
STUB_LANE = ARGS + r'''
import json, os, hashlib
st = sys.argv[1]; d = kv(sys.argv[2:])
open(os.environ["STUB_LOG"], "a").write(f"lane {st}\n")
pkg = d["--pkg"]; t = d.get("--tree")
if st == "install":
    os.makedirs(pkg, exist_ok=True)
    json.dump({"installed_sha256": hashlib.sha256(open(d["--table"], "rb").read()).hexdigest(), "rs_table_loads_and_matches": True}, open(os.path.join(pkg, "PF_LANE_INSTALL.json"), "w"))
    sys.exit(0)
p = {"clamp": f"raw/clamp_posthoc/trees/{t}.json", "compliant": f"raw/compliant_reread/trees/{t}.json", "tf_forward": f"raw/retime_trainfit/FORWARD_TRAINFIT_{t}.json",
     "d8": f"raw/retime_deploy/FORWARD_DEPLOY8_{t}.json", "df": f"raw/retime_deploy/FORWARD_DEPLOY_{t}.json", "tac": f"raw/retime_deploy/TAC_{t}.json",
     "parity": "raw/retime_deploy/PRIOR_PARITY.json"}[st]
f = os.path.join(pkg, p); os.makedirs(os.path.dirname(f), exist_ok=True); json.dump({}, open(f, "w"))
'''
STUB_PAR = ARGS + r'''
import json, os
d = kv(sys.argv[1:])
open(os.environ["STUB_LOG"], "a").write("feed_parity\n")
json.dump({"status": "PASS"}, open(d["--out"], "w"))
'''
STUB_VERD = ARGS + r'''
import json, os
d = kv(sys.argv[1:])
open(os.environ["STUB_LOG"], "a").write("verdict\n")
od = d.get("--out-dir") or os.path.join(os.environ["STUB_HOLD"], "raw", "ship_r8")
os.makedirs(od, exist_ok=True)
json.dump({"overall": "PASS", "beside": {"per_clip_verdict_deliberate_regression_arm": {"overall": "FAIL"}}}, open(os.path.join(od, "SHIP_R8_step48852ema_pf.json"), "w"))
'''


def _w(p: Path, txt: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8", newline="\n")
    return p


def m(p) -> str:
    return str(p).replace("\\", "/")


@pytest.fixture()
def rig(tmp_path):
    st = tmp_path / "stubs"
    bat = tmp_path / "bat"
    _w(bat / "gpu_lock.py", STUB_LOCK)
    _w(bat / "wait_commit.py", STUB_WC)
    _w(bat / "chain_marker.py", STUB_MARK)
    files = {"PULL": _w(st / "pull.sh", STUB_PULL), "EXTRACT": _w(st / "extract.py", STUB_EXTRACT), "EMACHK": _w(st / "emachk.py", STUB_EMACHK),
             "BATT": _w(st / "batt.py", STUB_BATT), "PAIR": _w(st / "pair.py", STUB_PAIR), "BANK": _w(st / "bank.py", STUB_BANK), "LANE": _w(st / "lane.py", STUB_LANE),
             "PAR": _w(st / "par.py", STUB_PAR), "VERD": _w(st / "verd.py", STUB_VERD)}
    hold = tmp_path / "hold"
    _w(hold / "code" / "ship_r8_verdict_pf.py", STUB_VERD)                     # the guardian's beside refresh calls <hold>/code/ship_r8_verdict_pf.py
    fz = _w(hold / "raw" / "retime_trainfit_pf" / "FROZEN_GAINS_RETIME_train_pf.json", '{"pf": 1}\n')
    sha = hashlib.sha256(fz.read_bytes()).hexdigest()
    _w(hold / "raw" / "retime_trainfit_pf" / "FROZEN_GAINS_RETIME_train_pf.sha256", sha + "\n")
    _w(hold / "raw" / "retime_trainfit_pf" / "FROZEN_GAINS_RETIME_train_pf.provenance.json", json.dumps({"sha256": sha}))
    regf = _w(tmp_path / "reg" / "REGISTERED_PF_TABLE.sha256", sha + "\n")
    kit = tmp_path / "kit"
    (kit / "ckpt").mkdir(parents=True)
    (kit / "chain").mkdir(parents=True)
    _w(kit / "ckpt" / "config.json", "{}")
    arch = tmp_path / "arch"
    arch.mkdir()
    env = dict(os.environ)
    env.update({"PF_HOLDW": m(hold), "PF_HOLDL": m(hold), "PF_R8PKGW": m(tmp_path / "pkg"), "PF_R8PKGL": m(tmp_path / "pkg"), "PF_BAT": m(bat),
                "PF_KITW": m(kit), "PF_KIT": m(kit), "PF_BRW": m(arch), "PF_BRL": m(arch), "PF_PY": PY, "PF_REGF": m(regf),
                "PF_PULL_SH": m(files["PULL"]), "PF_EXTRACT_PY": m(files["EXTRACT"]), "PF_EMACHK_PY": m(files["EMACHK"]), "PF_XCHECK_PY": m(files["EMACHK"]),
                "PF_BATT_PY": m(files["BATT"]), "PF_PAIR_PY": m(files["PAIR"]), "PF_BANK_PY": m(files["BANK"]), "PF_LANE_PY": m(files["LANE"]),
                "PF_PAR_PY": m(files["PAR"]), "PF_VERD_PY": m(files["VERD"]), "PF_CFN_PIDF": m(kit / "chain" / "chain_final.pid"),
                "PF_CFN_RESTART": f"echo cfn_restart >> {m(tmp_path / 'stub.log')}", "PF_STATE": m(tmp_path / "state"), "PF_LOG": m(kit / "chain" / "chain_final_pf.log"),
                "PF_POLL_S": "1", "PF_FREEGB_CMD": "echo 50", "PF_GATE_GAP_S": "0", "PF_MIN_FREE_KB": "0", "PF_REG_MAXWAIT_S": "3",
                "PF_LOCKF": m(tmp_path / "devbox_gpu.lock"), "PF_PERCLIP_V": m(tmp_path / "perclip_verdict.json"), "PF_CHAIN": m(CHAIN),
                "PF_RESTART_CAP_S": "5", "PF_BESIDE_CAP_S": "2",
                "STUB_LOG": m(tmp_path / "stub.log"), "STUB_HOLD": m(hold)})
    return {"env": env, "tmp": tmp_path, "hold": hold, "kit": kit, "regf": regf, "sha": sha, "arch": arch}


def run(rig, script=CHAIN, args=("as=chain_final_r8",), **extra):
    env = dict(rig["env"])
    env.update(extra)
    subprocess.run([SH, m(script), *args], env=env, timeout=600, stdin=subprocess.DEVNULL, capture_output=True)
    lp = rig["kit"] / "chain" / "chain_final_pf.log"
    log = lp.read_text(encoding="utf-8") if lp.exists() else ""
    sp = rig["tmp"] / "stub.log"
    stub = sp.read_text(encoding="utf-8") if sp.exists() else ""
    return log, stub


def reset_logs(rig):
    for p in (rig["kit"] / "chain" / "chain_final_pf.log", rig["tmp"] / "stub.log"):
        if p.exists():
            p.unlink()


def test_happy_path_pf_first_then_resume(rig):
    log, stub = run(rig)
    for mk in ("ZZPFSTARTZZ", "ZZPFMD5OKZZ", "ZZPFEXTRACTOKZZ", "ZZPFEMAGATEOKZZ", "ZZPFXCHECKNAZZ", "ZZPFREGOKZZ", "ZZPFLOCKEDZZ", "ZZPFBATTDONEZZ", "ZZPFPAIREDZZ",
               "ZZPFRELEASEDZZ", "ZZPFCFNRESTARTZZ", "ZZPFBANKEDZZ", "ZZPFLANEINSTALLOKZZ", "ZZPFLANEOK_clampZZ", "ZZPFLANEOK_parityZZ", "ZZPFPARITY_s0ZZ",
               "ZZPFPARITY_s1ZZ", "ZZPFVERDICT_PASSZZ", "ZZPFENDZZ"):
        assert mk in log, mk
    assert stub.count("gpu_lock acquire refcv8-battery-step48852ema_pf") == 1 and stub.count("gpu_lock release refcv8-battery-step48852ema_pf") == 1
    assert "battery step48852ema_pf pose=1 lockheld=True" in stub
    # PF-FIRST: battery and pair read inside the hold, the per-clip chain only after the release
    assert stub.index("battery step48852ema_pf") < stub.index("pair_read") < stub.index("gpu_lock release") < stub.index("cfn_restart")
    assert log.index("ZZPFRELEASEDZZ") < log.index("ZZPFCFNRESTARTZZ") < log.index("ZZPFLANEINSTALLOKZZ")
    assert stub.count("cfn_restart") == 1
    gate = json.loads((rig["hold"] / "raw" / "REGISTRATION_GATE.json").read_text(encoding="utf-8"))
    assert gate["registered_sha256"] == rig["sha"]
    # resume: nothing re-run, no lock taken
    reset_logs(rig)
    log2, stub2 = run(rig)
    assert "ZZPFREGHAVEZZ" in log2 and "ZZPFGPUHAVEZZ" in log2 and "ZZPFLANEHAVE_clampZZ" in log2 and "ZZPFENDZZ" in log2
    assert "gpu_lock acquire" not in stub2 and "battery step48852ema_pf pose" not in stub2 and "extract" not in stub2


@pytest.mark.parametrize("mut", ["missing", "mismatch", "empty"])
def test_mutation_unregistered_table_never_rolls(rig, mut):
    if mut == "missing":
        rig["regf"].unlink()
    elif mut == "mismatch":
        rig["regf"].write_text("0" * 64 + "\n")
    else:
        rig["regf"].write_text("\n")
    log, stub = run(rig)
    assert "ZZPFREGWAITZZ" in log and "ZZPFABORTZZ" in log and "stage R" in log
    assert "gpu_lock acquire" not in stub and "battery step48852ema_pf pose" not in stub and not (rig["hold"] / "raw" / "REGISTRATION_GATE.json").exists()
    assert "cfn_restart" in stub                                                # the per-clip chain still comes back


def test_mutation_battery_fails_twice_one_retry_then_halt(rig):
    log, stub = run(rig, STUB_BATT_FAIL="1")
    assert stub.count("battery step48852ema_pf") == 2 and "ZZPFBATTRETRYZZ" in log and "ZZPFBATTFAILZZ" in log and "ZZPFABORTZZ" in log
    assert stub.index("gpu_lock release") < stub.index("cfn_restart")         # lock released before the per-clip chain returns
    assert "lane " not in stub and "ZZPFENDZZ" not in log


def test_mutation_lock_not_acquired_no_battery(rig):
    log, stub = run(rig, STUB_LOCK_FAIL="1")
    assert "ZZPFNOLOCKZZ" in log and "battery step48852ema_pf pose" not in stub and "cfn_restart" in stub


def test_mutation_live_perclip_chain_aborts_before_pull(rig):
    pidf = rig["kit"] / "chain" / "chain_final.pid"
    p = subprocess.Popen([SH, "-c", f"echo $$ > {m(pidf)}; sleep 60"], stdin=subprocess.DEVNULL)
    try:
        for _ in range(50):
            if pidf.exists() and pidf.read_text().strip():
                break
            time.sleep(0.1)
        log, stub = run(rig)
        assert "ZZPFABORTZZ" in log and "stage a" in log and "pull" not in stub and "gpu_lock" not in stub
        assert "ZZPFCFNALIVEZZ" in log and "cfn_restart" not in stub
        # the guardian refuses too
        log2, _ = run(rig, script=GUARD, args=())
        assert "ZZPFGREFUSEZZ" in log2
    finally:
        p.kill()


def test_mutation_pull_fails_abort_a_and_perclip_back(rig):
    log, stub = run(rig, STUB_PULL_FAIL="1")
    assert "ZZPFPULLFAILZZ" in log and "ZZPFABORTZZ" in log and "extract" not in stub and "gpu_lock" not in stub and "cfn_restart" in stub


def test_guardian_happy_path_and_beside_refresh(rig):
    (rig["tmp"] / "perclip_verdict.json").write_text('{"overall": "FAIL"}')
    log, stub = run(rig, script=GUARD, args=())
    assert "ZZPFGSTARTZZ" in log and "ZZPFENDZZ" in log and "ZZPFGCHAINTERMINALZZ" in log and "ZZPFGCFNOKZZ" in log
    assert "ZZPFGBESIDEREFRESHZZ" in log and "PASS FAIL" in log and "ZZPFGENDZZ" in log
    assert stub.count("cfn_restart") == 1


def test_guardian_relaunches_once_then_restarts_perclip(rig):
    bad = _w(rig["tmp"] / "bad_chain.sh", "#!/bin/sh\necho \"ZZPFSTARTZZ x\" >> \"$PF_LOG\"\necho chainrun >> \"$STUB_LOG\"\nexit 0\n")
    log, stub = run(rig, script=GUARD, args=(), PF_CHAIN=m(bad))
    assert stub.count("chainrun") == 2 and "ZZPFGCHAINRELAUNCHZZ" in log and "ZZPFGCHAINGIVEUPZZ" in log
    assert "ZZPFGCFNRESTARTZZ" in log and stub.count("cfn_restart") == 1


def test_guardian_never_restarts_while_the_pf_job_holds_the_lock(rig):
    bad = _w(rig["tmp"] / "bad_chain.sh", "#!/bin/sh\necho \"ZZPFSTARTZZ x\" >> \"$PF_LOG\"\nexit 0\n")
    (rig["tmp"] / "devbox_gpu.lock").write_text('{"job": "refcv8-battery-step48852ema_pf", "pid": 1}')
    log, stub = run(rig, script=GUARD, args=(), PF_CHAIN=m(bad))
    assert "ZZPFGCFNSTUCKZZ" in log and "cfn_restart" not in stub


@pytest.mark.parametrize("form", ["bom_crlf", "two_tokens", "upper"])
def test_registration_file_forms_a_writer_may_produce_are_accepted(rig, form):
    sha = rig["sha"]
    raw = {"bom_crlf": b"\xef\xbb\xbf" + sha.encode() + b"\r\n", "two_tokens": (sha + "  FROZEN_GAINS_RETIME_train_pf.json\n").encode(),
           "upper": (sha.upper() + "\n").encode()}[form]
    rig["regf"].write_bytes(raw)
    log, stub = run(rig)
    assert "ZZPFREGOKZZ" in log and "battery step48852ema_pf pose=1" in stub


def test_mutation_registration_one_hex_short_is_refused(rig):
    rig["regf"].write_text(rig["sha"][:63] + "\n")
    log, stub = run(rig)
    assert "ZZPFREGWAITZZ" in log and "ZZPFABORTZZ" in log and "gpu_lock acquire" not in stub


def test_guardian_probe_ignores_a_shell_that_quotes_the_name_but_counts_a_real_pf_python(rig):
    """The self-match trap: a shell whose command line QUOTES run_battery_r8_pf.py must not block the per-clip restart; a python running it for the tag must."""
    bad = _w(rig["tmp"] / "bad_chain.sh", "#!/bin/sh\necho \"ZZPFSTARTZZ x\" >> \"$PF_LOG\"\nexit 0\n")
    decoy = subprocess.Popen([SH, "-c", "sleep 40 # run_battery_r8_pf.py --tag step48852ema_pf"], stdin=subprocess.DEVNULL)
    try:
        log, stub = run(rig, script=GUARD, args=(), PF_CHAIN=m(bad))
        assert "ZZPFGCFNRESTARTZZ" in log and "ZZPFGCFNSTUCKZZ" not in log
    finally:
        decoy.kill()
    for p in (rig["kit"] / "chain" / "chain_final_pf.log", rig["tmp"] / "stub.log", rig["tmp"] / "state" / "cfn_restarted"):
        if p.exists():
            p.unlink()
    # launched FROM an msys shell, as the chain launches it (a process started natively by pytest is invisible to msys `ps -ef`)
    real = subprocess.Popen([SH, "-c", f"exec '{PY}' -c 'import time; time.sleep(40)' run_battery_r8_pf.py --tag step48852ema_pf"], stdin=subprocess.DEVNULL)
    try:
        time.sleep(2)
        log, stub = run(rig, script=GUARD, args=(), PF_CHAIN=m(bad))
        assert "ZZPFGCFNSTUCKZZ" in log and "cfn_restart" not in stub
    finally:
        real.kill()
