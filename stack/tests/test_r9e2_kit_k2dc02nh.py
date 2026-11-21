"""The records / restart kit of arm K2DC02nh (stack/ops/r9e2_kit/k2dc02nh/): CPU only, no ssh, no GPU, nothing is run on Thor.

  * every script has LF endings and valid bash syntax;
  * the records queue (r9_recqueue25.sh) WAITS for the K2DC02-TAIL read before its first GPU step: `wait_read` is defined, is called after the dry-run exit and before the first `gpu` call,
    reads stageB/ROLL_STATUS.txt for AWAIT-S6C, refuses on a stopped / aborted read, and its process probes cannot match the queue's own command line;
  * its arm guard (the python condition) is RUN on the arm's argv and on K2DC02's: the arm passes, K2DC02 is refused (deliberate regression: with the condition inverted the arm is refused);
  * K0's queue 19 is skipped when it already ended DONE at the commit (a sibling queue made the records);
  * the restart wrapper keeps every guard of restart_e2sg.sh (FDISK / THORDISK / THORSHM / NORECORDS / WRONGV / NOK0REC / RECINPUT / K0CLOSURE / NOAPPROVAL / NOZGENV / CHAINLIVE) and swaps the
    K2SG-only guards for the arm's (controller present, bounds 0.1,25, NOARM); no executable line names K2SG or queue 23;
  * the after-read waiter reads the TAIL read's ROLL_STATUS and requires no compute app.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "ops" / "r9e2_kit" / "k2dc02nh"
RUNS = ROOT / "ops" / "runs.d"
ARM = "K2DC02nh"
FILES = ("r9_recqueue25.sh", "prep_q25.sh", "q25_after_read.sh", "restart_e2dc02nh.sh")


def _bash():
    for c in ("C:/Program Files/Git/usr/bin/bash.exe", "C:/Program Files/Git/bin/bash.exe"):      # never System32's WSL bash
        if Path(c).exists():
            return c
    w = shutil.which("bash")
    if w and "system32" not in w.lower():
        return w
    pytest.skip("no usable bash")


def _t(name):
    return (KIT / name).read_bytes().decode("utf-8")


def _argv(name):
    return json.loads((RUNS / f"r9e2_{name}.argv.json").read_text(encoding="utf-8"))["argv"]


def _code(t):
    return [ln for ln in t.splitlines() if not ln.lstrip().startswith("#")]


@pytest.mark.parametrize("name", FILES)
def test_script_has_lf_endings_and_valid_syntax(name):
    assert b"\r" not in (KIT / name).read_bytes()
    r = subprocess.run([_bash(), "-n", str(KIT / name)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("name", FILES)
def test_no_executable_line_names_k2sg_or_queue_23(name):
    for ln in _code(_t(name)):
        assert "K2SG" not in ln and "recq23" not in ln and "RECQ23" not in ln and "QUEUE23" not in ln and "prep_q23" not in ln, ln[:140]


def test_queue_waits_for_the_tail_read_before_its_first_gpu_step():
    t = _t("r9_recqueue25.sh")
    code = _code(t)
    i_def = next(i for i, ln in enumerate(code) if ln.startswith("wait_read() {"))
    i_dry = next(i for i, ln in enumerate(code) if ln.startswith('if [ "${R9Q25_DRY:-}" = 1 ]'))
    i_call = next(i for i, ln in enumerate(code) if ln == "wait_read")
    i_gpu = next(i for i, ln in enumerate(code) if ln.startswith("gpu mapb_$K2 "))
    assert i_def < i_dry < i_call < i_gpu
    body = "\n".join(code[i_def:i_def + 16])
    assert "E2TAIL_K2DC02" in t and "stageB/ROLL_STATUS.txt" in body and "AWAIT-S6C" in body and "ROLL_ABORTED" in body and "ALLSTEP STOP" in body
    assert "nvidia-smi --query-compute-apps=pid" in body and "[r]9e2_roll[.]sh" in body and "[r]efc_v3_train[.]py" in body
    assert "r9_recqueue" not in body and "prep_q" not in body                         # the probe cannot match this queue's own command line
    assert "ok -ge 2" in body and "sleep 60" in body


def _queue_pycode():
    m = re.search(r"/usr/bin/python3 -c \"(import json,sys; a=json\.load.*?)\" \$LT/stack", _t("r9_recqueue25.sh"))
    assert m, "no arm guard python in the queue"
    return m.group(1)


def _run_guard(code, argv_tokens, tmp_path):
    f = tmp_path / "a.json"
    f.write_text(json.dumps({"argv": argv_tokens}), encoding="utf-8")
    return subprocess.run([sys.executable, "-c", code, str(f)], capture_output=True, text=True, encoding="utf-8").returncode


def test_queue_arm_guard_accepts_the_arm_and_refuses_k2dc02(tmp_path):
    code = _queue_pycode()
    assert _run_guard(code, _argv(ARM), tmp_path) == 0
    assert _run_guard(code, _argv("K2DC02"), tmp_path) == 1
    assert _run_guard(code, _argv("K2M1N20"), tmp_path) != 0
    inverted = code.replace("sys.exit(0 if ", "sys.exit(1 if ", 1)                       # DELIBERATE REGRESSION: the guard turned inside out refuses the arm itself
    assert inverted != code and _run_guard(inverted, _argv(ARM), tmp_path) == 1


def test_queue_other_guards_and_q19_skip_are_present():
    code = "\n".join(_code(_t("r9_recqueue25.sh")))
    assert 'case "$K2" in ' + ARM + ') ;;' in code
    assert "NODC" in code and "NOK2DC02" in code and "NOREDOSE" in code and "NOARM" in code
    assert code.count('"$(tail -n 1 $R1/logs/recq19_$C12.status 2>/dev/null)" = ZZRECQ19-DONEZZ') == 2   # dry branch + real branch
    assert "ZZRECQ25-DONEZZ" in code and "ZZRECQ25-FAIL-" in code and "READSTOP" in code


def test_restart_keeps_the_k2sg_guards_and_swaps_the_arm_specific_ones():
    t = _t("restart_e2dc02nh.sh")
    for g in ("FDISK", "THORDISK", "THORSHM", "NORECORDS", "WRONGV", "NOK0REC", "RECINPUT", "K0CLOSURE", "NOAPPROVAL", "NOREDOSE", "NOZGENV", "CHAINLIVE"):
        assert "bad " + g + " " in t, g
    assert "bad NODC " in t and "bad NOK2DC02 " in t and "bad NOARM " in t and "NSG" not in t and "NOGATEFIX" not in t and "NOK2SG" not in t
    assert 'V=' + ARM + ';' in t and 'ZZRECQ25-DONEZZ' in t and "[ \"$FG\" -ge 15 ]" in t and "[ \"$TG\" -ge 15 ]" in t
    assert 'AC in K0:10.5 $V:$CAP' in t and "map_dose_ctl.py" in t


def test_after_read_waiter_reads_the_tail_read_and_requires_no_compute_app():
    t = _t("q25_after_read.sh")
    assert "E2TAIL_K2DC02/stageB/ROLL_STATUS.txt" in t and "AWAIT-S6C" in t and "nvidia-smi --query-compute-apps=pid" in t and "prep_q25.sh" in t
    assert "E2_READ_K2DC02" not in "\n".join(_code(t))
