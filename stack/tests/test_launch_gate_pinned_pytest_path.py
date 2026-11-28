"""The pinned check's child pytest finds the test runner on Thor (MM 4a3d05, 2026-11-28).

MEASURED 2026-11-28 on Thor: Thor's launch venv (/home/nvidia/venvs/tanitad-train) has NO pytest -- it lives in a --no-deps side
folder (/home/nvidia/refcv8_audit/pyt) that the suite runner puts on PYTHONPATH -- and `pinned_env` REPLACED PYTHONPATH with the
archive's tree only. Every child pytest the pinned check started then ended "No module named pytest": 9 ids of
test_launch_gate_pinned_split.py under the runner's environment, and a real Thor half (r9gate3_thor_pinned_half.sh, which set no
PYTHONPATH at all) would have lost every pinned file. The fix keeps a caller PYTHONPATH entry that holds pytest and no tree,
AFTER the archive's tree, and the Thor-half script passes the side folder explicitly.
"""
import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_gate as LG  # noqa: E402

OPS = Path(__file__).resolve().parents[1] / "ops" / "r9e1_gate3"


def _pkg(root: Path, name: str, body: str = "") -> Path:
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "__init__.py").write_text(body, encoding="utf-8")
    return root


def _pp(env: dict) -> list[str]:
    return env["PYTHONPATH"].split(os.pathsep)


def _dest(tmp_path: Path) -> Path:
    d = tmp_path / "archive"
    (d / "stack").mkdir(parents=True)
    (d / "taniteval").mkdir()
    return d


def test_a_caller_entry_holding_pytest_is_kept_AFTER_the_archive_tree_and_a_tree_entry_never_is(tmp_path):
    dest = _dest(tmp_path)
    side = _pkg(tmp_path / "side", "pytest")
    other = tmp_path / "other"
    other.mkdir()
    tree = _pkg(_pkg(tmp_path / "tree", "pytest"), "tanitad")       # holds pytest AND a tree: never kept
    base = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.pathsep.join([str(other), str(side), str(tree), str(side)])}
    env = LG.pinned_env(dest, {"omp": 4, "cpu_only": True}, base)
    assert _pp(env) == [str(dest / "stack"), str(dest / "taniteval"), str(side)]


def test_CONTROL_without_a_pytest_entry_the_path_is_the_archive_tree_only(tmp_path):
    dest = _dest(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    for base_pp in (None, "", str(other)):
        base = {"PATH": os.environ.get("PATH", "")}
        if base_pp is not None:
            base["PYTHONPATH"] = base_pp
        env = LG.pinned_env(dest, {"omp": 4}, base)
        assert _pp(env) == [str(dest / "stack"), str(dest / "taniteval")], base_pp


def _child_pytest_file(env: dict, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", "import pytest; print(pytest.__file__)"], cwd=str(cwd), env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)


def test_a_REAL_child_imports_the_side_folders_pytest(tmp_path):
    dest = _dest(tmp_path)
    side = _pkg(tmp_path / "side", "pytest", "SIDE_MARK = 1\n")
    base = dict(os.environ, PYTHONPATH=str(side))
    r = _child_pytest_file(LG.pinned_env(dest, {"omp": 4, "cpu_only": True}, base), tmp_path)
    assert r.returncode == 0, r.stderr[-400:]
    assert Path(r.stdout.strip()).resolve().is_relative_to(side.resolve()), r.stdout


def test_REGRESSION_ARM_without_the_keep_the_child_never_reaches_the_side_folder(tmp_path, monkeypatch):
    """The measured defect, re-installed: PYTHONPATH replaced by the tree only. On Thor the child then fails outright; on a
    host whose venv has pytest it imports the venv's copy -- either way it never reaches the side folder."""
    monkeypatch.setattr(LG, "pytest_side_entries", lambda pythonpath: [])
    dest = _dest(tmp_path)
    side = _pkg(tmp_path / "side", "pytest", "SIDE_MARK = 1\n")
    r = _child_pytest_file(LG.pinned_env(dest, {"omp": 4, "cpu_only": True}, dict(os.environ, PYTHONPATH=str(side))), tmp_path)
    assert r.returncode != 0 or not Path(r.stdout.strip()).resolve().is_relative_to(side.resolve()), r.stdout


def test_both_pinned_job_paths_build_their_child_environment_through_pinned_env():
    assert "pinned_env(dest, o, os.environ)" in inspect.getsource(LG.job_suite_pinned)
    assert "_pinned_split_env(" in inspect.getsource(LG.job_suite_pinned_split)
    assert "pinned_env(dest, ctx.options, os.environ)" in inspect.getsource(LG._pinned_split_env)
    assert "pytest_side_entries(" in inspect.getsource(LG.pinned_env)


def test_the_Thor_half_script_passes_the_side_folder_and_refuses_before_any_work_without_it():
    txt = (OPS / "r9gate3_thor_pinned_half.sh").read_text(encoding="utf-8")
    assert "\r" not in txt
    assert 'PYT="${R9_THOR_PYTEST_PATH:-/home/nvidia/refcv8_audit/pyt}"' in txt
    guard = '[ -f "$PYT/pytest/__init__.py" ] || refuse'
    assert guard in txt
    assert 'PYTHONPATH="$PYT" PYTHONDONTWRITEBYTECODE=1 timeout "${R9_PINNED_TIMEOUT_S:-14400}" "$PY"' in txt
    # the refusal comes before the previous archive is removed and before the gate starts
    assert txt.index(guard) < txt.index('rm -rf "${SW:?}"') < txt.index("--pinned-half thor")
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not on PATH")
    r = subprocess.run([bash, "-n", str(OPS / "r9gate3_thor_pinned_half.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
