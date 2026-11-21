"""The refcv9 chain must hand the PRE-REGISTERED zero-gradient allow-list to every arm (MM 4a3d05, 2026-11-10).

MEASURED 2026-11-10 (Thor, refcv9 E2 K0): the arm trained 6,000 steps and FAILED the supervisor's G5 on core.decoder.r8_rc_gate and r8_rc_to_cond.{weight,bias}
(1,921 values), pre-registered dead under --r8-no-rc. The tool that derives the exemption was right; the chain appended a controls env with a LITERAL
ZEROGRAD_ALLOW=() and the derived line never reached the manifest. ``r9gate3_zg_env.sh`` is the one hop between them; these tests pin it and its wiring.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "ops" / "r9e1_gate3" / "r9gate3_zg_env.sh"
CHAIN = ROOT / "ops" / "r9e1_gate3" / "r9gate3_chain.sh"
K0_ARGV = ROOT / "ops" / "runs.d" / "r9e2_K0.argv.json"
CONTROLS = ROOT / "ops" / "r9e1_gate3" / "zerograd_controls.env.derived"   # the committed shape: a CONTROL line and ZEROGRAD_ALLOW=()
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def _posix(p: Path) -> str:
    return p.as_posix()


def _run(helper: Path, profile: str, argv: Path, controls: Path, env_py: bool = True):
    import os
    env = dict(os.environ)
    env["PY"] = _posix(Path(sys.executable)) if env_py else env.get("PY", "")
    r = subprocess.run([BASH, _posix(helper), profile, _posix(argv), _posix(controls), _posix(ROOT.parent)],
                       capture_output=True, text=True, encoding="utf-8", timeout=300, env=env)
    return r.returncode, r.stdout, r.stderr


def _source(block: str, tmp: Path):
    f = tmp / "block.env"
    f.write_text(block, encoding="utf-8", newline="\n")
    probe = 'ZEROGRAD_ALLOW=(); ZEROGRAD_CONTROL=(); . "$1"; echo "ALLOW=${#ZEROGRAD_ALLOW[@]} CONTROL=${#ZEROGRAD_CONTROL[@]}"'
    r = subprocess.run([BASH, "-c", probe, "x", _posix(f)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    return r.stdout.strip()


@pytest.mark.parametrize("profile", ["refcv9-e2", "refcv9-full"])
def test_K0_block_carries_the_r8_rc_exemptions(profile):
    rc, out, err = _run(HELPER, profile, K0_ARGV, CONTROLS)
    assert rc == 0, err
    allow = [l for l in out.splitlines() if l.startswith("ZEROGRAD_ALLOW=")]
    assert len(allow) == 1
    assert r"core\.decoder\.r8_rc_gate" in allow[0] and r"core\.decoder\.r8_rc_to_cond" in allow[0]
    assert "ZEROGRAD_ALLOW=()" not in out


def test_sourced_block_has_12_allow_and_a_control(tmp_path):
    rc, out, err = _run(HELPER, "refcv9-e2", K0_ARGV, CONTROLS)
    assert rc == 0, err
    m = re.fullmatch(r"ALLOW=(\d+) CONTROL=(\d+)", _source(out, tmp_path))
    assert m and int(m.group(1)) == 12 and int(m.group(2)) >= 1


def _mutant(tmp_path: Path) -> Path:
    """the helper with the derived ALLOW replaced by the controls env's own (empty) line -- the Thor defect."""
    src = HELPER.read_text(encoding="utf-8")
    line = r"""printf '%s\n' "$DERIVED"  # EMIT-ALLOW"""
    assert src.count(line) == 1, "the emit line moved: update the mutation"
    out = tmp_path / "r9gate3_zg_env_mutant.sh"
    out.write_text(src.replace(line, "grep '^ZEROGRAD_ALLOW=' \"$CTRL\""), encoding="utf-8", newline="\n")
    return out


def test_MUTATION_RED_a_helper_passing_the_controls_empty_allow_fails_the_K0_check(tmp_path):
    mut = _mutant(tmp_path)
    rc, out, err = _run(mut, "refcv9-e2", K0_ARGV, CONTROLS)
    assert rc == 0 and "ZEROGRAD_ALLOW=()" in out                              # the mutant runs and emits the Thor defect ...
    with pytest.raises(AssertionError):                                        # ... and test (a)'s assertion goes RED on it
        assert r"core\.decoder\.r8_rc_gate" in out


def _controls(tmp_path: Path, text: str) -> Path:
    f = tmp_path / "controls.env"
    f.write_text(text, encoding="utf-8", newline="\n")
    return f


def test_refuses_without_a_control_line(tmp_path):
    rc, out, err = _run(HELPER, "refcv9-e2", K0_ARGV, _controls(tmp_path, "ZEROGRAD_ALLOW=()\n"))
    assert rc != 0 and out == "" and "ZEROGRAD_CONTROL" in err


def test_refuses_a_nonempty_allow_that_differs(tmp_path):
    rc, out, err = _run(HELPER, "refcv9-e2", K0_ARGV, _controls(tmp_path, CONTROLS.read_text(encoding="utf-8").replace("ZEROGRAD_ALLOW=()", "ZEROGRAD_ALLOW=('foo.*')")))
    assert rc != 0 and out == "" and "differs" in err


def test_accepts_a_nonempty_allow_equal_to_the_derived_one(tmp_path):
    rc, derived, err = _run(HELPER, "refcv9-e2", K0_ARGV, CONTROLS)
    assert rc == 0, err
    allow_line = [l for l in derived.splitlines() if l.startswith("ZEROGRAD_ALLOW=")][0]
    ctl = [l for l in CONTROLS.read_text(encoding="utf-8").splitlines() if l.startswith("ZEROGRAD_CONTROL=")][0]
    rc2, out2, err2 = _run(HELPER, "refcv9-e2", K0_ARGV, _controls(tmp_path, ctl + "\n" + allow_line + "\n"))
    assert rc2 == 0, err2 and out2.count("ZEROGRAD_ALLOW=") == 1


def test_refuses_when_the_tool_fails(tmp_path):
    rc, out, err = _run(HELPER, "no-such-profile", K0_ARGV, CONTROLS)
    assert rc != 0 and out == "" and "REFUSED" in err
    rc, out, err = _run(HELPER, "refcv9-e2", tmp_path / "missing.json", CONTROLS)
    assert rc != 0 and out == ""


def test_refuses_a_missing_controls_env(tmp_path):
    rc, out, err = _run(HELPER, "refcv9-e2", K0_ARGV, tmp_path / "nope.env")
    assert rc != 0 and out == ""


def test_the_chain_calls_the_helper_and_no_longer_appends_the_controls_env_directly():
    s = CHAIN.read_text(encoding="utf-8")
    assert not re.search(r"cat\s+\S*zerograd_controls\.env\s*>>", s), "the bare cat of the controls env is back"
    assert "r9gate3_zg_env.sh" in s and '"$R9_PROFILE" "$ARGV"' in s
    assert "ZGENV-REFUSED" in s
    # the refusal branch must not reach the launch line
    i = s.index("ZGENV-REFUSED"); j = s.index("-LAUNCH")
    assert i < j and "continue; fi" in s[i:j]
