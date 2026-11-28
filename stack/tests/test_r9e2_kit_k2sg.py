"""The records / restart kit of arm K2SG (stack/ops/r9e2_kit/k2sg/): CPU only, no ssh, no GPU, nothing is run on Thor.

  * every script has LF endings and valid bash syntax;
  * the records queue (r9_recqueue26.sh) WAITS for the K2DC02s1-TAIL read before its first GPU step: `wait_read` is defined, is called after the dry-run exit and before the first `gpu` call,
    reads stageB/ROLL_STATUS.txt for AWAIT-S6C, refuses on a stopped / aborted read, and its process probes cannot match the queue's own command line;
  * its arm guards (the python conditions) are RUN on the arm's argv, on K2DC02's and K2DC02nh's: the arm passes, those are refused (deliberate regression: with a condition inverted the arm is refused);
  * K0's queue 19 is skipped when it already ended DONE at the commit (a sibling queue made the records);
  * the restart wrapper keeps every guard of the K2DC02nh restart (FDISK / THORDISK / THORSHM / NORECORDS / WRONGV / NOK0REC / RECINPUT / K0CLOSURE / NOAPPROVAL / NOREDOSE / NOZGENV / CHAINLIVE) and swaps the
    dose-controller guards (NODC / NOK2DC02) for K2SG's (DC = no controller, NOK2SG = exactly one --map-hires-trunk-grad-scale 0.0); no executable line names K2DC02nh or queue 25;
  * the after-read waiter reads the K2DC02s1 TAIL read's ROLL_STATUS (overridable by R9K_RS) and requires no compute app.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "ops" / "r9e2_kit" / "k2sg"
RUNS = ROOT / "ops" / "runs.d"
ARM = "K2SG"
FILES = ("r9_recqueue26.sh", "prep_q26.sh", "q26_after_read.sh", "restart_e2sg.sh")


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


def test_static_gate_check_is_importable_and_runs_nothing_on_import():
    src = (KIT / "static_gate_check.py").read_text(encoding="utf-8")
    compile(src, "static_gate_check.py", "exec")
    assert 'if __name__ == "__main__":' in src and "def main(argv):" in src and '("K2M1N20", arm)' in src


@pytest.mark.parametrize("name", FILES)
def test_no_executable_line_names_another_arm_or_queue_25(name):
    for ln in _code(_t(name)):
        for bad in ("K2DC02nh", "e2dc02nh", "recq25", "RECQ25", "QUEUE25", "prep_q25", "R9Q25", "E2NH"):
            assert bad not in ln, (bad, ln[:140])


def test_queue_waits_for_the_tail_read_before_its_first_gpu_step():
    t = _t("r9_recqueue26.sh")
    code = _code(t)
    i_def = next(i for i, ln in enumerate(code) if ln.startswith("wait_read() {"))
    i_dry = next(i for i, ln in enumerate(code) if ln.startswith('if [ "${R9Q26_DRY:-}" = 1 ]'))
    i_call = next(i for i, ln in enumerate(code) if ln == "wait_read")
    i_gpu = next(i for i, ln in enumerate(code) if ln.startswith("gpu mapb_$K2 "))
    assert i_def < i_dry < i_call < i_gpu
    body = "\n".join(code[i_def:i_def + 16])
    assert "runs/E2TAIL_K2DC02s1" in t and "stageB/ROLL_STATUS.txt" in body and "AWAIT-S6C" in body and "ROLL_ABORTED" in body and "ALLSTEP STOP" in body
    assert "nvidia-smi --query-compute-apps=pid" in body and "[r]9e2_roll[.]sh" in body and "[r]efc_v3_train[.]py" in body
    assert "r9_recqueue" not in body and "prep_q" not in body                         # the probe cannot match this queue's own command line
    assert "ok -ge 2" in body and "sleep 60" in body


def _queue_pycodes():
    out = {}
    for m in re.finditer(r"/usr/bin/python3 -c \"(import json,sys; a=json\.load.*?)\" \$LT/stack/ops/runs\.d/r9e2_\$K2\.argv\.json 2>/dev/null \|\| fail \"[^\"]*\" (\w+) 2", _t("r9_recqueue26.sh")):
        out[m.group(2)] = m.group(1)
    assert set(out) == {"NOK2SG", "NOARM"}, out.keys()
    return out


def _run_guard(code, argv_tokens, tmp_path):
    f = tmp_path / "a.json"
    f.write_text(json.dumps({"argv": argv_tokens}), encoding="utf-8")
    return subprocess.run([sys.executable, "-c", code, str(f)], capture_output=True, text=True, encoding="utf-8").returncode


def test_queue_arm_guards_accept_the_arm_and_refuse_the_dose_controller_arms(tmp_path):
    for tag, code in _queue_pycodes().items():
        assert _run_guard(code, _argv(ARM), tmp_path) == 0, tag
    stop_gradient = _queue_pycodes()["NOK2SG"]
    for other in ("K2DC02", "K2DC02nh", "K2M1N20"):                                    # none carries the stop-gradient flag
        assert _run_guard(stop_gradient, _argv(other), tmp_path) == 1, other
    assert _run_guard(_queue_pycodes()["NOARM"], _argv("K2DC02nh"), tmp_path) == 1      # no T1v
    twice = _argv(ARM) + ["--map-hires-trunk-grad-scale", "0.0"]
    assert _run_guard(stop_gradient, twice, tmp_path) == 1                              # the flag exactly once
    half = list(_argv(ARM)); half[half.index("--map-hires-trunk-grad-scale") + 1] = "0.5"
    assert _run_guard(stop_gradient, half, tmp_path) == 1                               # and exactly 0.0
    for tag, code in _queue_pycodes().items():                                          # DELIBERATE REGRESSION: a guard turned inside out refuses the arm itself
        inverted = code.replace("sys.exit(0 if ", "sys.exit(1 if ", 1)
        assert inverted != code and _run_guard(inverted, _argv(ARM), tmp_path) == 1, tag


def test_queue_other_guards_and_q19_skip_are_present():
    code = "\n".join(_code(_t("r9_recqueue26.sh")))
    assert 'case "$K2" in ' + ARM + ') ;;' in code
    assert "DC 2" in code and "--map-hires-share-target" in code and "NOK2SG" in code and "NOREDOSE" in code and "NOARM" in code
    assert "NODC" not in code and "NOK2DC02" not in code
    assert code.count('"$(tail -n 1 $R1/logs/recq19_$C12.status 2>/dev/null)" = ZZRECQ19-DONEZZ') == 2   # dry branch + real branch
    assert "ZZRECQ26-DONEZZ" in code and "ZZRECQ26-FAIL-" in code and "READSTOP" in code


def test_restart_keeps_the_guards_and_swaps_the_dose_controller_ones_for_the_stop_gradient_ones():
    t = _t("restart_e2sg.sh")
    for g in ("FDISK", "THORDISK", "THORSHM", "NORECORDS", "WRONGV", "NOK0REC", "RECINPUT", "K0CLOSURE", "NOAPPROVAL", "NOREDOSE", "NOZGENV", "CHAINLIVE", "DC", "NOK2SG", "NOARM"):
        assert "bad " + g + " " in t, g
    assert "bad NODC " not in t and "bad NOK2DC02 " not in t and "0.1,25" not in t
    assert "'--map-hires-trunk-grad-scale'" in t and "== '0.0'" in t and "'--r9-hist-supervise-geom') == 1" in t
    assert 'V=' + ARM + ';' in t and 'ZZRECQ26-DONEZZ' in t and "[ \"$FG\" -ge 15 ]" in t and "[ \"$TG\" -ge 15 ]" in t
    assert 'AC in K0:10.5 $V:$CAP' in t and "map_trunk_gs.py" in t and "AUTO_E2SG.txt" in t


def test_after_read_waiter_reads_the_s1_tail_read_and_requires_no_compute_app():
    t = _t("q26_after_read.sh")
    assert "R9K_RS:-/home/nvidia/r8ns/r9e2/runs/E2TAIL_K2DC02s1/stageB/ROLL_STATUS.txt" in t and "AWAIT-S6C" in t
    assert "nvidia-smi --query-compute-apps=pid" in t and "prep_q26.sh" in t and "[k]2dc02s1tail" in t and "[r]9e2_roll[.]sh" in t
    assert "E2TAIL_K2DC02/" not in "\n".join(_code(t))
