"""Locks the 2026-07-24 TanitEval productionization guarantees.

  * ``data.list_val_episodes`` refuses the known-leaky physicalai val split by
    default (78 % train leak), passes the CLEAN split, honors ``allow_leaky``,
    and leaves other corpora (comma / cosmos / OOD) alone.
  * ``runner.py`` exposes closed-loop as a first-class subcommand, so the ONE
    canonical entrypoint covers open-loop + beyond-ADE + closed-loop.

Pure-logic + one subprocess ``--help`` probe; no GPU, no pod data. Runs green on
the dev box and the pod (paths derived from this file's location).
"""
import os
import subprocess
import sys

import pytest

from taniteval import data

LEAKY = "physicalai-val-f1b378f295ae"
CLEAN = "physicalai-val-0c5f7dac3b11"


def test_split_constants_are_canonical():
    assert data.CLEAN_VAL == CLEAN
    assert data.LEAKY_VAL == LEAKY


def test_leaky_split_refused_by_default():
    with pytest.raises(RuntimeError) as ei:
        data.list_val_episodes(f"/root/valdata/{LEAKY}", 40)
    msg = str(ei.value)
    assert "leak" in msg.lower()
    assert CLEAN in msg          # error must point at the clean replacement


def test_clean_split_not_refused():
    # dir absent on the dev box -> returns [], but must NOT raise the guard
    assert isinstance(data.list_val_episodes(f"/root/valdata/{CLEAN}", 40), list)


def test_allow_leaky_escape_hatch():
    assert isinstance(
        data.list_val_episodes(f"/root/valdata/{LEAKY}", 40, allow_leaky=True),
        list)


def test_other_corpora_not_refused():
    # generalization evaluates comma / cosmos / OOD dirs — the guard must trigger
    # ONLY on the specific leaky hash, never on a different corpus.
    for corpus in ("comma-val-abc123", "cosmos-gen-xyz", CLEAN):
        assert isinstance(data.list_val_episodes(f"/root/valdata/{corpus}"), list)


def _runner_help():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        os.path.join(repo, "stack"),
        os.path.join(repo, "stack", "scripts"),
        os.path.join(repo, "taniteval"),
        env.get("PYTHONPATH", ""),
    ])
    return subprocess.run([sys.executable, "-m", "taniteval.runner", "--help"],
                          capture_output=True, text=True, env=env, timeout=180)


def test_runner_is_one_entrypoint_with_closedloop():
    r = _runner_help()
    assert r.returncode == 0, r.stderr[-600:]
    # closed-loop is now wired in...
    for cmd in ("closedloop", "closedloop-all", "closedloop-report"):
        assert cmd in r.stdout, f"runner missing subcommand {cmd!r}"
    # ...and the pre-existing standard axes are still present.
    for cmd in ("run", "run-all", "driving", "hierarchy", "efficiency", "report"):
        assert cmd in r.stdout, f"runner lost subcommand {cmd!r}"
