"""The records / restart kit of arm K2DC02cl (stack/ops/r9e2_kit/k2dc02cl/, queue 27): CPU only, no ssh, no GPU, nothing is run on Thor.

Mirror of test_r9e2_kit_k2dc02s1.py / _k2dc02nh.py, adapted to the clamp arm (K2DC02's flags, --w-map-hires 3.0, --map-hires-w-bounds 2.5,3.5, seed 0):

  * every script has LF endings and valid bash syntax;
  * the records queue (r9_recqueue27.sh) WAITS for the K2DC02s1 TAIL read before its first GPU step (`wait_read` defined, called after the dry-run exit and before the first `gpu` call);
  * ITS ARM GUARDS ARE RUN (a real bash, a fake launch tree) on the arm's argv and on K2DC02's: the arm passes; K2DC02's argv under the arm's name is refused (NOREDOSE: no 3.0), a
    stale 25.0 / 0.1,25 is refused (NOSTALE), seed 1 is refused (NOARM); DELIBERATE REGRESSION: with the grep guards deleted K2DC02's argv slips through to the python guard;
  * K0's queue 19 is skipped when it already ended DONE at the commit;
  * the restart wrapper keeps every guard of restart_e2dc02s1.sh and swaps in the clamp's (3.0 present, 25.0 absent, 2.5,3.5 present, 0.1,25 absent, seed 0): a real run of its NOARM python;
  * the after-read waiter reads the K2DC02s1 TAIL read's ROLL_STATUS and requires no compute app; it is never started by anything.
Records are bound to the argv sha256 (G-MAP-OVERFIT / G-BOX-OVERFIT), so K2DC02's records do NOT transfer: the queue produces this arm's own.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "ops" / "r9e2_kit" / "k2dc02cl"
RUNS = ROOT / "ops" / "runs.d"
ARM = "K2DC02cl"
FILES = ("r9_recqueue27.sh", "prep_q27.sh", "q27_after_read.sh", "restart_e2dc02cl.sh", "static_gate_check.py")
PY = sys.executable.replace("\\", "/")


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
def test_script_has_lf_endings(name):
    assert b"\r" not in (KIT / name).read_bytes()


@pytest.mark.parametrize("name", [f for f in FILES if f.endswith(".sh")])
def test_script_has_valid_bash_syntax(name):
    r = subprocess.run([_bash(), "-n", str(KIT / name)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("name", FILES)
def test_no_executable_line_names_another_arms_queue(name):
    for ln in _code(_t(name)):
        assert "K2SG" not in ln and "K2DC02s1 " not in ln.replace("K2DC02s1 TAIL", "") and "K2DC02nh" not in ln, ln[:140]
        for old in ("recq23", "RECQ23", "recq24", "RECQ24", "recq25", "RECQ25", "recq26", "RECQ26", "prep_q24", "prep_q25", "recqueue24", "recqueue25", "QUEUE24", "QUEUE25"):
            assert old not in ln, (old, ln[:140])


def test_queue_waits_for_the_s1_tail_read_before_its_first_gpu_step():
    t = _t("r9_recqueue27.sh")
    code = _code(t)
    i_def = next(i for i, ln in enumerate(code) if ln.startswith("wait_read() {"))
    i_dry = next(i for i, ln in enumerate(code) if ln.startswith('if [ "${R9Q27_DRY:-}" = 1 ]'))
    i_call = next(i for i, ln in enumerate(code) if ln == "wait_read")
    i_gpu = next(i for i, ln in enumerate(code) if ln.startswith("gpu mapb_$K2 "))
    assert i_def < i_dry < i_call < i_gpu
    body = "\n".join(code[i_def:i_def + 16])
    assert "TAIL=/home/nvidia/r8ns/r9e2/runs/E2TAIL_K2DC02s1\n" in t and "stageB/ROLL_STATUS.txt" in body and "AWAIT-S6C" in body and "ROLL_ABORTED" in body and "ALLSTEP STOP" in body
    assert "nvidia-smi --query-compute-apps=pid" in body and "[r]9e2_roll[.]sh" in body and "[r]efc_v3_train[.]py" in body
    assert "r9_recqueue" not in body and "prep_q" not in body                         # the probe cannot match this queue's own command line
    assert "ok -ge 2" in body and "sleep 60" in body


GUARD_FIRST = "grep -q '\"3.0\"'"                                                     # first arm guard line of the queue (NOREDOSE)


def _queue_guard_lines():
    code = _code(_t("r9_recqueue27.sh"))
    i0 = next(i for i, ln in enumerate(code) if ln.startswith(GUARD_FIRST))
    i1 = next(i for i, ln in enumerate(code) if ln.startswith("/usr/bin/python3 -c"))
    return code[i0:i1 + 1]


def _run_queue_guards(tmp_path, argv_tokens, lines=None):
    lt = tmp_path / "lt"
    (lt / "stack" / "ops" / "runs.d").mkdir(parents=True, exist_ok=True)
    (lt / "stack" / "ops" / "runs.d" / f"r9e2_{ARM}.argv.json").write_text(json.dumps({"argv": argv_tokens}, indent=1), encoding="utf-8")
    body = "\n".join(lines if lines is not None else _queue_guard_lines()).replace("/usr/bin/python3", PY)
    script = f'LT="{lt.as_posix()}"; K2={ARM}\nfail() {{ echo "FAILED-$2"; exit 1; }}\n{body}\necho GUARDS-PASS\n'
    f = tmp_path / "g.sh"
    f.write_bytes(script.encode("utf-8"))
    r = subprocess.run([_bash(), str(f)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "NO-OUTPUT " + r.stderr[-200:]


def test_queue_arm_guards_accept_the_arm_and_refuse_everything_else(tmp_path):
    assert len(_queue_guard_lines()) == 6, _queue_guard_lines()                       # 3.0, stale 25.0, share-target (NODC), target + 2.5,3.5, stale 0.1,25, the python seed + T1v guard
    a = list(_argv(ARM))
    assert _run_queue_guards(tmp_path, a) == "GUARDS-PASS"
    assert _run_queue_guards(tmp_path, _argv("K2DC02")) == "FAILED-NOREDOSE"          # K2DC02's own argv carries no 3.0 and its records must not be reused
    s = list(a)
    s[s.index("--seed") + 1] = "1"
    assert _run_queue_guards(tmp_path, s) == "FAILED-NOARM"                           # the replicate seed
    n = [t for t in a if t != "--r9-hist-supervise-geom"]
    assert _run_queue_guards(tmp_path, n) == "FAILED-NOARM"                           # K2DC02nh's argv (no T1v)
    st = list(a) + ["25.0"]
    assert _run_queue_guards(tmp_path, st) == "FAILED-NOSTALE"                        # a stale K2DC02 initial weight
    sb = list(a) + ["0.1,25"]
    assert _run_queue_guards(tmp_path, sb) == "FAILED-NOSTALE"                        # stale K2DC02 bounds
    wb = [("2.5,3.0" if t == "2.5,3.5" else t) for t in a]
    assert _run_queue_guards(tmp_path, wb) == "FAILED-NOK2DC02"                       # other bounds


def test_DELIBERATE_REGRESSION_without_the_grep_guards_K2DC02_slips_through(tmp_path):
    lines = [ln for ln in _queue_guard_lines() if ln.startswith("/usr/bin/python3")]  # only the python seed / T1v guard left
    assert _run_queue_guards(tmp_path, _argv("K2DC02"), lines) == "GUARDS-PASS"       # K2DC02 has seed 0 and T1v too: ONLY the grep guards tell the two apart
    assert _run_queue_guards(tmp_path, _argv(ARM), lines) == "GUARDS-PASS"


def test_queue_other_guards_and_q19_skip_are_present():
    code = "\n".join(_code(_t("r9_recqueue27.sh")))
    assert 'case "$K2" in ' + ARM + ') ;;' in code
    assert "NODC" in code and "NOK2DC02" in code and "NOREDOSE" in code and "NOARM" in code and "NOSTALE" in code
    assert code.count('"$(tail -n 1 $R1/logs/recq19_$C12.status 2>/dev/null)" = ZZRECQ19-DONEZZ') == 2   # dry branch + real branch
    assert "ZZRECQ27-DONEZZ" in code and "ZZRECQ27-FAIL-" in code and "READSTOP" in code
    assert "dry mapb_$K2 map-banded $K2" in code and "dry box_$K2 box $K2" in code and "gpu mapb_$K2 map-banded $K2" in code and "gpu box_$K2 box $K2" in code   # its OWN records (argv-sha bound)


def test_restart_keeps_the_guards_and_swaps_in_the_clamp_ones():
    t = _t("restart_e2dc02cl.sh")
    for g in ("FDISK", "THORDISK", "THORSHM", "NORECORDS", "WRONGV", "NOK0REC", "RECINPUT", "K0CLOSURE", "NOAPPROVAL", "NOREDOSE", "NOSTALE", "NOZGENV", "CHAINLIVE", "NODC", "NOK2DC02", "NOARM"):
        assert "bad " + g + " " in t, g
    assert 'V=' + ARM + ';' in t and 'ZZRECQ27-DONEZZ' in t and "[ \"$FG\" -ge 15 ]" in t and "[ \"$TG\" -ge 15 ]" in t
    assert 'AC in K0:10.5 $V:$CAP' in t and "map_dose_ctl.py" in t and 'QUEUE27-START $RC12' in t
    assert "grep -q '\"3.0\"'" in t and "grep -q '\"25.0\"' && bad NOSTALE" in t and "grep -q '\"2.5,3.5\"'" in t and "grep -q '\"0.1,25\"' && bad NOSTALE" in t


def test_restart_NOARM_python_accepts_the_arm_only(tmp_path):
    m = re.search(r"-I -c \"(import json,sys; a=json\.load\(sys\.stdin\).*?)\" \| tr", _t("restart_e2dc02cl.sh"))
    assert m, "no NOARM python in the restart"
    code = m.group(1)

    def run(tokens):
        return subprocess.run([sys.executable, "-I", "-c", code], input=json.dumps({"argv": tokens}), capture_output=True, text=True, encoding="utf-8").stdout.strip()
    assert run(_argv(ARM)) == "1"
    s = list(_argv(ARM))
    s[s.index("--seed") + 1] = "1"
    assert run(s) == "0"
    assert run([t for t in _argv(ARM) if t != "--r9-hist-supervise-geom"]) == "0"


def test_after_read_waiter_reads_the_s1_tail_read_and_requires_no_compute_app():
    t = _t("q27_after_read.sh")
    assert "E2TAIL_K2DC02s1/stageB/ROLL_STATUS.txt" in t and "AWAIT-S6C" in t and "nvidia-smi --query-compute-apps=pid" in t and "prep_q27.sh" in t
    assert "nothing starts it automatically" in t


def test_static_gate_check_is_import_safe_and_the_arm_is_registered_in_it():
    t = _t("static_gate_check.py")
    assert 'if __name__ == "__main__":' in t and "def main(argv):" in t and '("K2DC02", arm)' in t
