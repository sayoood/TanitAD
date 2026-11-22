"""E2 read launcher for arm K2SG (stack/ops/r9e2_read/e2read_launch_k2sg.sh).  CPU only, no scipy, no GPU, no ssh.

  * the K2SG launcher differs from the K2DC02 one ONLY in the intended lines (compared after K2SG -> K2DC02, so a stray K2DC02 left in an executable line shows up as a missing change);
  * `bash -n` passes, LF endings;
  * the guards are RUN (their lines are extracted and executed in bash): a K2SG config (trunk-grad-scale 0.0 once, no share target) passes; a K2DC02 config, a K2M1N20 config (no flag),
    the flag twice, the value 0.25 and a missing K0 seed directory are refused;
  * DELIBERATE REGRESSION: a copy of the launcher without the trunk-grad-scale guard lets a K2M1N20 config through, so the refusal check goes RED on it;
  * the launcher's literals equal the tokens of stack/ops/runs.d/r9e2_K2SG.argv.json (the single source).
  * THE TRAINER'S REAL LAYOUT (MM 4a3d05, 2026-11-22): the 1f14f8d8 trainer writes the argv TWICE in config.json (`argv` and `argv_effective`); the same
    whole-file count refused the valid K2DC02s1 read that day (fixed there in 8b581767). The guard here counts inside each array: the two-key layout
    passes, a disagreeing / flagless / doubled `argv_effective` is refused, and the old whole-file line must refuse the real layout.
"""
import difflib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops" / "r9e2_read"
NEW = OPS / "e2read_launch_k2sg.sh"
OLD = OPS / "e2read_launch_k2dc02.sh"
ARGV = ROOT / "ops" / "runs.d" / "r9e2_K2SG.argv.json"
FLAG, VAL = "--map-hires-trunk-grad-scale", "0.0"


def _bash():
    for c in ("C:/Program Files/Git/usr/bin/bash.exe", "C:/Program Files/Git/bin/bash.exe"):      # never System32's WSL bash
        if Path(c).exists():
            return c
    w = shutil.which("bash")
    if w and "system32" not in w.lower():
        return w
    pytest.skip("no usable bash")


def _text(p):
    return p.read_bytes().decode("utf-8")


def _norm(t):
    return t.replace("K2SG", "K2DC02").replace("k2sg", "k2dc02")


# ---------------------------------------------------------------------------------------------------------------- #
def test_launcher_has_lf_endings_and_valid_syntax():
    assert b"\r" not in NEW.read_bytes()
    r = subprocess.run([_bash(), "-n", str(NEW)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr


def test_launcher_differs_from_k2dc02_only_in_the_intended_lines():
    a, b = _text(OLD).splitlines(), _norm(_text(NEW)).splitlines()
    removed, added = [], []
    for ln in difflib.unified_diff(a, b, lineterm="", n=0):
        if ln.startswith(("---", "+++", "@@")):
            continue
        (removed if ln.startswith("-") else added).append(ln[1:])
    rem = [r for r in removed if not r.startswith("#")]
    add = [x for x in added if not x.startswith("#")]
    assert len(rem) == 4, rem
    assert sum(r.startswith("grep -q -- '\"--map-hires-share-target\"'") for r in rem) == 1
    assert sum(r.startswith('case "$CFG1"') for r in rem) == 2
    assert sum(r.lstrip().startswith("grep -q 'controller_for'") for r in rem) == 1
    assert len(add) == 3, add
    assert sum(x.startswith("grep -q -- '\"--map-hires-share-target\"'") and "&& die" in x for x in add) == 1
    assert sum(x.startswith("NF=") for x in add) == 1
    assert sum(x.lstrip().startswith("grep -q '_mgs.attach'") and "map_trunk_gs.py" in x for x in add) == 1


def test_every_executable_k2dc02_mention_is_gone():
    for i, ln in enumerate(_text(NEW).splitlines(), 1):
        if ln.lstrip().startswith("#"):
            continue
        assert "K2DC02" not in ln and "k2dc02" not in ln, f"line {i} still names K2DC02: {ln[:120]}"
        if "K2DC" in ln:                                      # the one intended K2DC: the default K0 seed source
            assert ln.startswith("ARM=K2SG; OLD=${K0_SEED_SRC:-$R/runs/E2_READ_K2DC};"), f"line {i}: {ln[:120]}"
    t = _text(NEW)
    assert r"arms \[K2SG\]" in t and "' K2SG-DONE '" in t and "ARM=K2SG;" in t and "runs/E2_READ_K2DC}" in t


def test_launcher_literals_equal_the_argv_json_tokens():
    argv = json.loads(ARGV.read_text(encoding="utf-8"))["argv"]
    assert argv.count(FLAG) == 1 and argv[argv.index(FLAG) + 1] == VAL
    assert "--map-hires-share-target" not in argv
    t = _text(NEW)
    assert f'"{FLAG}","{VAL}"' in t and f'"{FLAG}"' in t


# ---------------------------------------------------------------------------------------------------------------- #
# the guards are RUN                                                                                                #
# ---------------------------------------------------------------------------------------------------------------- #
def _guard_lines(text):
    keep = [ln for ln in text.splitlines()
            if ln.startswith(("CFG1=", "NF=", "for d in $OLD ")) or ln.startswith("grep -q -- '\"--map-hires-share-target\"'")]
    assert len(keep) == 4, keep
    return keep


def _run_guards(text, tmp_path, cfg_tokens, make_seed=True, lines=None, effective=None):
    k2 = tmp_path / "k2"
    k2.mkdir(exist_ok=True)
    cfg = {"argv": cfg_tokens} if effective is None else {"arm": "K2SG", "seed": 0, "argv": cfg_tokens, "argv_effective": effective}
    (k2 / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    old = tmp_path / "seed_src"
    if make_seed:
        for d in ("E2_K0_pf", "E2_K0_mapsel", "read/dump_K0"):
            (old / d).mkdir(parents=True, exist_ok=True)
    script = 'die() { echo "ZZE2READ-REFUSED $*"; exit 3; }\nK2=%s\nOLD=%s\n%s\necho GUARDS-PASSED\n' % (
        k2.as_posix(), old.as_posix(), "\n".join(_guard_lines(text) if lines is None else lines))
    p = tmp_path / "guards.sh"
    p.write_bytes(script.encode("utf-8"))
    r = subprocess.run([_bash(), str(p)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout


K2SG_CFG = ["--map-hires", "on", "--w-map-hires", "25.0", FLAG, VAL]
K2M1N20_CFG = ["--map-hires", "on", "--w-map-hires", "25.0"]
K2DC02_CFG = ["--map-hires-share-target", "0.02", "--map-hires-share-every", "250", "--map-hires-w-bounds", "0.1,25"]


def _case(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return d


def test_guards_accept_k2sg_and_refuse_the_rest(tmp_path):
    t = _text(NEW)
    rc, out = _run_guards(t, _case(tmp_path, "a"), K2SG_CFG)
    assert rc == 0 and "GUARDS-PASSED" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "b"), K2DC02_CFG)
    assert rc == 3 and "share-target" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "c"), K2M1N20_CFG)
    assert rc == 3 and "exactly once" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "d"), K2SG_CFG + [FLAG, VAL])
    assert rc == 3 and "exactly once" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "e"), ["--map-hires", "on", FLAG, "0.25"])
    assert rc == 3 and "exactly once" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "f"), K2SG_CFG + [FLAG, "0.25"])                  # flag twice, one of them 0.0
    assert rc == 3 and "exactly once" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "g"), K2SG_CFG, make_seed=False)
    assert rc == 3 and "K0 seed source directory missing" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "h"), K2SG_CFG + ["--map-hires-share-target", "0.02"])   # K2SG plus a controller flag
    assert rc == 3 and "share-target" in out, out


def test_deliberate_regression_without_the_trunk_grad_scale_guard_is_caught(tmp_path):
    t = _text(NEW)
    mutant = "\n".join(ln for ln in t.splitlines() if not ln.startswith("NF="))
    assert mutant != t and "NF=" not in mutant
    with pytest.raises(AssertionError):                       # _guard_lines' own census goes red: the guard line is gone
        _guard_lines(mutant)
    # and the BEHAVIOUR: with only that guard line dropped from the executed set, a K2M1N20 config (no flag) passes the guards
    keep = [ln for ln in _guard_lines(t) if not ln.startswith("NF=")]
    rc, out = _run_guards(t, _case(tmp_path, "m"), K2M1N20_CFG, lines=keep)
    assert rc == 0 and "GUARDS-PASSED" in out, out            # the mutant is blind to a flagless config; the pristine guard refuses it (previous test)
    rc, out = _run_guards(t, _case(tmp_path, "p"), K2M1N20_CFG)
    assert rc == 3, out


# the guard line as landed before 2026-11-22 (whole-file count): kept verbatim so the regression class stays on record
OLD_FLAG_GUARD = ("NF=$(( $(printf '%s' \"$CFG1\" | grep -o '\"--map-hires-trunk-grad-scale\"' | wc -l) )); "
                  "NP=$(( $(printf '%s' \"$CFG1\" | grep -o '\"--map-hires-trunk-grad-scale\",\"0.0\"' | wc -l) )); "
                  "[ \"$NF\" = 1 ] && [ \"$NP\" = 1 ] || die \"K2SG config.json must carry --map-hires-trunk-grad-scale 0.0 exactly once (flag x$NF, pair x$NP)\"")


def test_guards_read_the_trainers_real_layout_argv_and_argv_effective(tmp_path):
    t = _text(NEW)
    rc, out = _run_guards(t, _case(tmp_path, "r1"), K2SG_CFG, effective=list(K2SG_CFG))              # the real 1f14f8d8 layout
    assert rc == 0 and "GUARDS-PASSED" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "r2"), K2SG_CFG, effective=K2SG_CFG[:-1] + ["0.25"])    # argv_effective disagrees on the value
    assert rc == 3 and "exactly once" in out and "argv_effective" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "r3"), K2SG_CFG, effective=list(K2M1N20_CFG))           # argv_effective without the flag
    assert rc == 3 and "exactly once" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "r4"), K2SG_CFG, effective=K2SG_CFG + [FLAG, VAL])      # argv_effective with the flag twice
    assert rc == 3 and "exactly once" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "r5"), K2M1N20_CFG, effective=list(K2SG_CFG))           # argv without the flag, argv_effective with it
    assert rc == 3 and "exactly once" in out, out


def test_the_old_whole_file_flag_guard_refuses_the_real_layout(tmp_path):
    t = _text(NEW)
    assert OLD_FLAG_GUARD not in t
    keep = [OLD_FLAG_GUARD if ln.startswith("NF=") else ln for ln in _guard_lines(t)]
    assert keep != _guard_lines(t)
    rc, out = _run_guards(t, _case(tmp_path, "o1"), K2SG_CFG, lines=keep)                            # one array: the old line passed (why the fixture missed it)
    assert rc == 0 and "GUARDS-PASSED" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "o2"), K2SG_CFG, lines=keep, effective=list(K2SG_CFG))  # the real layout: the old line REFUSES a valid read
    assert rc == 3 and "flag x2, pair x2" in out, out
