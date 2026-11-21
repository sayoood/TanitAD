"""test_q5_clipdone.py -- the pod run loop's done predicate (MM, 2026-11-15).
The defect it pins: 2 of the 141 eval clips have no eval window, so they write no rows file, and the old predicate (ledger OK AND rows file) never held
for them; the run loop rebuilt both 80 times. A zero-window clip is done on its OK ledger line alone; a windowed clip still needs its rows file.
Run: OMP_NUM_THREADS=2 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q test_q5_clipdone.py --basetemp <dir>
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TANITAD_STACK", "F:/TanitAD-scratch/stopbal_read/tree/stack")


def _mod():
    spec = importlib.util.spec_from_file_location("q5_frames_pod_t", os.path.join(HERE, "q5_frames_pod.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _old(led, c, root):
    return led.get(c, {}).get("status") == "OK" and os.path.exists(f"{root}/rows/{c}.npz")


def test_clip_done_cases(tmp_path):
    P = _mod()
    root = str(tmp_path)
    os.makedirs(f"{root}/rows")
    open(f"{root}/rows/win_rows.npz", "wb").write(b"x")
    need = {"nowin": ("eval", []), "win_rows": ("eval", [10, 20]), "win_norows": ("train", [5]), "failed": ("eval", [])}
    led = {"nowin": {"status": "OK"}, "win_rows": {"status": "OK"}, "win_norows": {"status": "OK"}, "failed": {"status": "FAIL"}}
    assert P.clip_done(led, "nowin", root, need) is True              # the fix: zero windows, OK ledger line, no rows file
    assert P.clip_done(led, "win_rows", root, need) is True
    assert P.clip_done(led, "win_norows", root, need) is False        # a windowed clip still needs its rows file
    assert P.clip_done(led, "failed", root, need) is False
    assert P.clip_done({}, "nowin", root, need) is False              # no ledger line at all
    # regression arm: the old predicate reproduces the loop (never done on a zero-window clip)
    assert _old(led, "nowin", root) is False


def test_run_loop_uses_the_predicate():
    src = open(os.path.join(HERE, "q5_frames_pod.py"), encoding="utf-8").read()
    assert src.count("not clip_done(led, c, root, need)") == 2, "both the todo list and the final check must use clip_done"
    assert 'os.path.exists(f"{root}/rows/{c}.npz")) and att.get' not in src, "the old todo predicate is back"
