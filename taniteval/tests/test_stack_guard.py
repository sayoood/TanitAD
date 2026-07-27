"""⛔ THE STALE-IMPORT GUARD — and its DEMONSTRATED FAILURE.

WHY THIS FILE IS SUBPROCESS-BASED
---------------------------------
The defect is about which ``tanitad`` a *fresh interpreter* resolves. Exercising
it in-process would mean mutating ``sys.modules``/``sys.path``/``sys.meta_path``
of the running test session and importing a fake ``tanitad`` on top of the real
one — which poisons every later test. Each scenario therefore runs in its own
``sys.executable`` with a hand-built ``sys.path``, and the assertion is on the
child's exit code and output. That is also what makes these a *demonstration*
rather than a mock.

⭐ THE ANTI-C13 CONTRACT. ``test_the_wrong_number_is_produced_when_the_guard_is_off``
is the RED half: the same stale tree, guard disabled, silently yields
**HFOV 120.0 instead of 117.0**. Every GREEN assertion below is only meaningful
because that RED one passes — a guard that cannot fail is worse than none.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_PKG_PARENT = _HERE.parent                  # <repo>/taniteval  (holds `taniteval/`)
_REPO = _PKG_PARENT.parent                  # <repo>
_REAL_STACK = _REPO / "stack"

# The two numbers the fake trees disagree on. 117.0 is v5's ACTUAL frame (the
# rig-clean 176x624 slice of the 120 deg parent cache) — so a stale tree that
# still answers 120.0 is exactly the plausible-wrong-number shape.
GOOD_HFOV, STALE_HFOV = 117.0, 120.0


def _write(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(s), encoding="utf-8")


def _make_tree(root: Path, *, tag: str, hfov: float, post_v5: bool) -> Path:
    """A miniature `stack` tree. ``post_v5=False`` reproduces pod2's
    ``/root/TanitAD/stack``: no ``heldout_gate``, no ``heldout_goal``, no
    ``register_v2_geometry_sibling``, no ``resolve_v2_frames``."""
    _write(root / "tanitad" / "__init__.py",
           f'STACK_TAG = "{tag}"\n__version__ = "{tag}"\n')
    _write(root / "tanitad" / "geometry.py",
           f'STACK_TAG = "{tag}"\nHFOV_DEG = {hfov}\n'
           f'def frame_from_args(*a, **k):\n    return HFOV_DEG\n')
    _write(root / "tanitad" / "train" / "__init__.py", "")
    _write(root / "tanitad" / "data" / "__init__.py", "")
    _write(root / "scripts" / "train_flagship_v4.py", f'STACK_TAG = "{tag}"\n')
    if post_v5:
        _write(root / "tanitad" / "train" / "heldout_gate.py",
               'PRIMARY_NAME = "pseudosim_composite_PSS_recovery_progress'
               '@twosided_v2"\n')
        _write(root / "tanitad" / "train" / "heldout_goal.py",
               "def make_goal_kwargs_fn(*a, **k):\n    return None\n")
        _write(root / "tanitad" / "data" / "parity.py",
               "def register_v2_geometry_sibling(*a, **k):\n    return None\n")
        with open(root / "scripts" / "train_flagship_v4.py", "a",
                  encoding="utf-8") as fh:
            fh.write("def resolve_v2_frames(*a, **k):\n    return None\n")
    else:
        _write(root / "tanitad" / "data" / "parity.py", "PRE_V5 = True\n")
    return root


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    base = tmp_path_factory.mktemp("stacks")
    good = _make_tree(base / "workspace_stack", tag="GOOD", hfov=GOOD_HFOV,
                      post_v5=True)
    stale = _make_tree(base / "root_stack", tag="STALE", hfov=STALE_HFOV,
                       post_v5=False)
    return {"good": good, "stale": stale}


_STANDALONE_LOADER = f"""
def _load_stack_guard_standalone():
    '''Load stack_guard WITHOUT running taniteval/__init__ (which pins a root),
    and with the dev box's editable-install finder for `tanitad` removed, so the
    child interpreter sees only the fake trees this test built.'''
    import importlib.util as _u
    sys.meta_path = [f for f in sys.meta_path
                     if "editable" not in getattr(type(f), "__module__", "").lower()]
    sys.modules.pop("tanitad", None)
    _s = _u.spec_from_file_location(
        "_sg_standalone",
        {str(_PKG_PARENT / "taniteval" / "stack_guard.py")!r})
    _m = _u.module_from_spec(_s); _s.loader.exec_module(_m)
    return _m
"""


def _run(body: str, *, env_extra=None, path_extra=()) -> subprocess.CompletedProcess:
    lines = ["import sys, json",
             f"sys.path.insert(0, {str(_PKG_PARENT)!r})   # the REAL taniteval pkg"]
    lines += [f"sys.path.insert(0, {str(p)!r})" for p in path_extra]
    prelude = "\n".join(lines) + "\n" + _STANDALONE_LOADER + "\n"
    env = dict(os.environ)
    env.pop("TANITEVAL_STACK_OVERRIDE", None)
    env.pop("TANITEVAL_STACK_GUARD", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = ""
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(body)],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env, timeout=180)


# ===========================================================================
# RED — the failure, demonstrated. Without these the GREENs prove nothing.
# ===========================================================================
def test_the_wrong_number_is_produced_when_the_guard_is_off(trees):
    """⛔ THE DEFECT ITSELF, reproduced end to end.

    A stale tree at ``sys.path[0]`` — exactly what
    ``sys.path.insert(0, "/root/TanitAD/stack")`` did — with the guard disabled.
    The process does **not** crash. It answers **120.0** where the pinned stack
    says **117.0**, and prints a number an artifact would carry."""
    r = _run(
        f"""
        import taniteval                       # guard OFF, so no pinning
        sys.path.insert(0, {str(trees['stale'])!r})   # the old hardcoded line
        import tanitad.geometry as g
        print(json.dumps({{"tag": g.STACK_TAG, "hfov": g.HFOV_DEG,
                           "file": g.__file__}}))
        """,
        env_extra={"TANITEVAL_STACK_GUARD": "off"},
        path_extra=[trees["good"]])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    # ⭐ No exception, no warning — just the wrong number.
    assert out["tag"] == "STALE"
    assert out["hfov"] == STALE_HFOV != GOOD_HFOV


def test_a_bare_import_tanitad_does_not_catch_it(trees):
    """⚠️ The instruction 'verify with a real `import tanitad`' is INSUFFICIENT
    — and this is the two-process shape that made it look sufficient on pod2.

    Process 1 is the sync proof (``python3 -c "import tanitad"``, no
    ``taniteval``): it resolves to the GOOD tree and PASSES.
    Process 2 is the eval: ``taniteval`` puts the stale tree on ``sys.path``
    first — the old hardcoded line, simulated here — and the identical
    ``import tanitad`` resolves to the STALE tree. Same command, same host,
    opposite answer."""
    proof = _run("""
        import tanitad
        print(json.dumps({"tag": tanitad.STACK_TAG, "file": tanitad.__file__}))
        """, path_extra=[trees["good"]])
    assert proof.returncode == 0, proof.stderr
    assert json.loads(proof.stdout.strip().splitlines()[-1])["tag"] == "GOOD"

    evalproc = _run(
        f"""
        sys.path.insert(0, {str(trees['stale'])!r})   # <- the old hardcoded line
        import tanitad
        print(json.dumps({{"tag": tanitad.STACK_TAG, "file": tanitad.__file__}}))
        """,
        env_extra={"TANITEVAL_STACK_GUARD": "off"},
        path_extra=[trees["good"]])
    assert evalproc.returncode == 0, evalproc.stderr
    assert json.loads(
        evalproc.stdout.strip().splitlines()[-1])["tag"] == "STALE"


# ===========================================================================
# GREEN — the guard refuses the same layouts.
# ===========================================================================
def test_sentinel_refuses_a_late_stale_insert(trees):
    """⭐ THE CASE THAT NEEDS NO ENV VAR — 'the next person copies a command
    from somewhere else'. ``taniteval`` pins from PYTHONPATH at import; a later
    stale insert is refused."""
    r = _run(
        f"""
        import taniteval
        sys.path.insert(0, {str(trees['stale'])!r})
        import tanitad.train.heldout_gate
        print("NO GUARD FIRED")
        """,
        path_extra=[trees["good"]])
    assert r.returncode != 0
    assert "STACK SHADOWING" in r.stderr
    assert "NO GUARD FIRED" not in r.stdout


def test_override_is_verified_not_merely_set(trees, tmp_path):
    """⭐ ``TANITEVAL_STACK_OVERRIDE`` used to be SET AND TRUSTED — and it prints
    a success line either way.

    Here it points at a directory that holds no ``tanitad`` (a typo, a moved
    checkout, a pod where the sync never landed), so the stale tree still wins.
    The old code printed ``[taniteval] tanitad OVERRIDE -> <the empty dir>``
    *followed by the stale file path* and carried on. An env var that is set but
    ineffective is the same 'reports success, ships different bytes' shape as
    the CRLF sync trap, so it is now checked against what actually resolved."""
    empty = tmp_path / "not-a-stack"
    empty.mkdir()
    r = _run(
        f"""
        sys.path.insert(0, {str(trees['stale'])!r})    # stale WINS on sys.path
        import taniteval
        import tanitad.geometry as g
        print("HFOV", g.HFOV_DEG)
        """,
        env_extra={"TANITEVAL_STACK_OVERRIDE": str(empty)})
    assert r.returncode != 0, r.stdout
    assert "STACK SHADOWING" in r.stderr or "STACK GUARD REFUSED" in r.stderr
    assert "HFOV" not in r.stdout          # no number ever reached stdout


def test_sentinel_catches_an_already_imported_stale_tanitad(trees):
    """The one import that matters most must not be the one never seen: if a
    stale ``tanitad`` is already in ``sys.modules`` when ``taniteval`` loads,
    the sentinel checks it by hand at install time."""
    r = _run(
        f"""
        sys.path.insert(0, {str(trees['stale'])!r})
        import tanitad                                  # stale, imported FIRST
        import taniteval
        print("NO GUARD FIRED")
        """,
        env_extra={"TANITEVAL_STACK_OVERRIDE": str(trees["good"])})
    assert r.returncode != 0
    assert "STACK SHADOWING" in r.stderr
    assert "NO GUARD FIRED" not in r.stdout


def test_capability_probe_refuses_a_pre_v5_tree(trees):
    """⭐ Identity is NOT sufficient. The stale tree can be the only tree and
    still be pre-v5 — a pod that was never `git pull`ed. ``require='v5'``
    catches that; the path check alone cannot."""
    r = _run(
        """
        import taniteval
        from taniteval import stack_guard as sg
        try:
            sg.assert_stack(require="v5", label="probe")
        except sg.StackShadowError as ex:
            print("REFUSED"); print(str(ex))
            sys.exit(3)
        print("ACCEPTED")
        """,
        env_extra={"TANITEVAL_STACK_OVERRIDE": str(trees["stale"])})
    assert r.returncode == 3, r.stdout + r.stderr
    assert "REFUSED" in r.stdout
    assert "heldout_gate" in r.stdout


def test_capability_probe_accepts_the_matching_tree(trees):
    """…and it is not a rubber stamp in the other direction either."""
    r = _run(
        """
        import taniteval
        from taniteval import stack_guard as sg
        rep = sg.assert_stack(require="v5", label="probe")
        print(json.dumps({"ok": rep["ok"], "problems": rep["problems"]}))
        """,
        env_extra={"TANITEVAL_STACK_OVERRIDE": str(trees["good"])})
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["ok"] is True and out["problems"] == []


@pytest.mark.parametrize("tree,rc", [("stale", 2), ("good", 0)])
def test_the_DOCUMENTED_module_command_is_the_one_that_works(trees, tree, rc):
    """⛔ Pin the exact string the v5 command set tells people to paste.

    `python3 -m taniteval.stack_check --require v5` — run as a real subprocess,
    both directions. A published command that has never been executed is how
    `--anchors-dense` shipped pointing at an empty directory."""
    env = dict(os.environ)
    env["TANITEVAL_STACK_OVERRIDE"] = str(trees[tree])
    env["PYTHONPATH"] = os.pathsep.join(
        [str(trees[tree]), str(trees[tree] / "scripts"), str(_PKG_PARENT)])
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("TANITEVAL_STACK_GUARD", None)
    r = subprocess.run([sys.executable, "-m", "taniteval.stack_check",
                        "--require", "v5"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=180)
    assert r.returncode == rc, r.stdout + r.stderr
    assert "runpy" not in r.stderr        # the entry point exists to avoid this


def test_a_GREEN_says_what_it_PROBED_and_a_no_require_GREEN_does_not(trees):
    """⛔ FOUND ON THE GUARD'S FIRST FIELD TEST (tanitad-pod3, 2026-07-27).

    RED, measured before the fix: the stdout of

        … stack_check --require v5       (5 capabilities verified)
        … stack_check                    (ZERO capabilities verified)

    was **byte-identical** — both printed ``"ok": true, "problems": []``. The
    operator pastes one of these in front of an eval and reads exactly that
    summary, so a dropped or misspelled ``--require`` reads as a pass. The JSON
    report always carried ``required``; only the human surface did not.

    ⚠️ This is the same failure shape the guard exists to close: a GREEN that
    looks identical whether or not the check happened."""
    env = dict(os.environ)
    env["TANITEVAL_STACK_OVERRIDE"] = str(trees["good"])
    env["PYTHONPATH"] = os.pathsep.join(
        [str(trees["good"]), str(trees["good"] / "scripts"), str(_PKG_PARENT)])
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("TANITEVAL_STACK_GUARD", None)

    def _cli(*args):
        r = subprocess.run([sys.executable, "-m", "taniteval.stack_check", *args],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env, timeout=180)
        assert r.returncode == 0, r.stdout + r.stderr
        return json.loads(r.stdout[r.stdout.index("{"):])

    with_req, without = _cli("--require", "v5"), _cli()
    # both are honest passes …
    assert with_req["ok"] is True and without["ok"] is True
    # … and they are now DISTINGUISHABLE on stdout, which is the whole point.
    assert with_req != without
    assert len(with_req["required"]) == 5
    assert any("heldout_gate" in c for c in with_req["required"])
    assert without["required"] == []


def test_a_capability_REFUSAL_also_prints_what_was_demanded(trees):
    """The refusal path builds its report separately, so it needs its own pin —
    otherwise a reader cannot tell a capability refusal from an identity one."""
    r = _run(
        f"""
        from taniteval import stack_guard as sg
        sys.exit(sg.main(["--stack", {str(trees['stale'])!r}, "--require", "v5"]))
        """)
    assert r.returncode == 2, r.stdout + r.stderr
    out = json.loads(r.stdout[r.stdout.index("{"):])
    assert out["ok"] is False
    assert len(out["required"]) == 5


def test_cli_exits_2_on_a_stale_tree_and_writes_its_json(trees, tmp_path):
    """``python3 -m taniteval.stack_guard`` in front of an eval command turns a
    wrong number into a one-second exit 2."""
    outp = tmp_path / "guard.json"
    r = _run(
        f"""
        from taniteval import stack_guard as sg
        sys.exit(sg.main(["--stack", {str(trees['stale'])!r}, "--require", "v5",
                          "--json", {str(outp)!r}]))
        """)
    assert r.returncode == 2, r.stdout + r.stderr
    rep = json.loads(outp.read_text(encoding="utf-8"))
    assert rep["ok"] is False and rep["problems"]


# ===========================================================================
# The deployed path — pinned, so this change cannot break it.
# ===========================================================================
def test_legacy_only_layout_is_unchanged(trees, monkeypatch):
    """⛔ THE PIN ON THE DEPLOYED EVAL PATH.

    Every published closed-loop number was produced on a host where the legacy
    tree is the ONLY tree. There, ``resolve_intended_stack`` returns it with
    provenance ``legacy-fallback``, ``ensure_stack_on_path`` puts it (and its
    ``scripts/``) at the FRONT exactly as the hardcoded lines did, and the guard
    has no teeth because nothing else was named."""
    r = _run(
        f"""
        sg = _load_stack_guard_standalone()   # no package __init__, nothing pre-pinned
        sg.LEGACY_STACK = {str(trees['stale'])!r}          # play the deployed tree
        root, prov = sg.resolve_intended_stack()
        info = sg.ensure_stack_on_path(legacy={str(trees['stale'])!r})
        import tanitad
        print(json.dumps({{"root": root, "prov": prov, "info_prov": info["provenance"],
                           "legacy_used": info["legacy_used"],
                           "head": sys.path[0], "tag": tanitad.STACK_TAG,
                           "front_is_stack": sys.path[0] == {str(trees['stale'])!r},
                           "scripts_present": {str(trees['stale'] / 'scripts')!r} in sys.path}}))
        """)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["prov"] == "legacy-fallback"
    assert out["legacy_used"] is True
    assert out["front_is_stack"] is True and out["scripts_present"] is True
    assert out["tag"] == "STALE"        # i.e. the deployed tree still wins here


def test_named_stack_is_never_shadowed_by_legacy(trees):
    """The inverse of the pin: once a stack IS named, the legacy tree is not put
    in front of it — and is not appended as a silent fallback either."""
    r = _run(
        f"""
        from taniteval import stack_guard as sg
        sg.LEGACY_STACK = {str(trees['stale'])!r}
        info = sg.ensure_stack_on_path(legacy={str(trees['stale'])!r})
        import tanitad
        print(json.dumps({{"prov": info["provenance"], "tag": tanitad.STACK_TAG,
                           "legacy_on_path": {str(trees['stale'])!r} in sys.path}}))
        """,
        env_extra={"TANITEVAL_STACK_OVERRIDE": str(trees["good"])})
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["prov"] == f"env:{'TANITEVAL_STACK_OVERRIDE'}"
    assert out["tag"] == "GOOD"
    assert out["legacy_on_path"] is False


# ===========================================================================
# Modes, and the regression pin on the cause.
# ===========================================================================
def test_two_stack_trees_and_neither_named_SHOUTS(trees, tmp_path):
    """⭐ pod2's real layout, and the one case the guard structurally cannot
    refuse: nothing was named, so the deployed tree IS the intent — but there is
    a second one on the host. MEASURED 2026-07-27: `tanitad-eval` has only
    `/root/TanitAD/stack`, pod3 only `/workspace/TanitAD/stack`, **pod2 has
    both** and they differ by a release. Resolution is deliberately unchanged
    (moving it would move the deployed eval host); the silence is not."""
    harness = tmp_path / "TanitAD" / "taniteval"          # a repo-shaped sibling
    (harness / "taniteval").mkdir(parents=True)
    make = _make_tree(tmp_path / "TanitAD" / "stack", tag="OTHER",
                      hfov=GOOD_HFOV, post_v5=True)
    shutil_src = _PKG_PARENT / "taniteval" / "stack_guard.py"
    (harness / "taniteval" / "stack_guard.py").write_text(
        shutil_src.read_text(encoding="utf-8"), encoding="utf-8")
    (harness / "taniteval" / "__init__.py").write_text("", encoding="utf-8")
    r = _run(
        f"""
        import importlib.util as u
        s = u.spec_from_file_location(
            "sg_sib", {str(harness / 'taniteval' / 'stack_guard.py')!r})
        sg = u.module_from_spec(s); s.loader.exec_module(sg)
        sg.LEGACY_STACK = {str(trees['stale'])!r}
        info = sg.ensure_stack_on_path(legacy={str(trees['stale'])!r})
        print(json.dumps({{"prov": info["provenance"],
                           "alts": info["ambiguous_alternatives"]}}))
        """)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["prov"] == "legacy-fallback"
    assert str(make) in " ".join(out["alts"]), out
    assert "AMBIGUOUS STACK" in r.stderr


def test_warn_mode_shouts_and_records_but_does_not_raise(trees):
    r = _run(
        f"""
        import taniteval
        sys.path.insert(0, {str(trees['stale'])!r})
        import tanitad                         # would raise in error mode
        from taniteval import stack_guard as sg
        rep = sg.report()
        print(json.dumps({{"mode": rep["mode"],
                           "n": len(rep["sentinel_violations"])}}))
        """,
        env_extra={"TANITEVAL_STACK_GUARD": "warn"},
        path_extra=[trees["good"]])
    assert r.returncode == 0, r.stderr
    assert "STACK SHADOWING" in r.stderr             # it still SHOUTS
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["mode"] == "warn" and out["n"] >= 1   # …and it is on the record


def test_off_mode_cannot_hide_itself(trees):
    """⚠️ A disabled guard that leaves no trace is how a wrong number gets
    published with a clean-looking artifact. ``report()`` always carries it."""
    r = _run(
        """
        import taniteval
        from taniteval import stack_guard as sg
        print(json.dumps({"mode": sg.report()["mode"],
                          "installed": sg.report()["sentinel_installed"]}))
        """,
        env_extra={"TANITEVAL_STACK_GUARD": "off"},
        path_extra=[trees["good"]])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["mode"] == "off" and out["installed"] is False


def test_no_live_hardcoded_stack_insert_survives_in_the_package():
    """⛔ THE REGRESSION PIN ON THE CAUSE. 28 modules carried the literal
    ``sys.path.insert(0, "/root/TanitAD/stack")``. If one comes back, this
    fails — the documentation fix alone would not have stopped it."""
    import re
    pat = re.compile(r'^\s*sys\.path\.insert\(\s*0\s*,\s*'
                     r'["\']/root/TanitAD/stack')
    offenders = []
    for f in sorted((_PKG_PARENT / "taniteval").glob("*.py")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if pat.match(line):
                offenders.append(f"{f.name}:{i}")
    assert offenders == [], offenders


@pytest.mark.skipif(not (_REAL_STACK / "tanitad" / "train" / "heldout_gate.py").is_file(),
                    reason="repo stack not present")
def test_this_repo_stack_satisfies_the_v5_capability_set():
    """The probe list must describe the REAL post-v5 tree, or ``--require v5``
    is a rubber stamp. Checked against this checkout's own ``stack/``."""
    r = _run(
        """
        import taniteval
        from taniteval import stack_guard as sg
        bad = [c for c in sg.V5_CAPABILITIES if sg._probe(c)]
        print(json.dumps({"bad": bad}))
        """,
        env_extra={"TANITEVAL_STACK_OVERRIDE": str(_REAL_STACK)},
        path_extra=[_REAL_STACK / "scripts"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["bad"] == [], out["bad"]
