"""E2 read launcher for arm K2DC02s1 (stack/ops/r9e2_read/e2read_launch_k2dc02s1.sh).  CPU only, no scipy, no GPU, no ssh.

  * the launcher differs from e2read_launch_k2dc02.sh ONLY in two added guard lines (compared after K2DC02s1 -> K2DC02: nothing may be REMOVED from an executable line);
  * `bash -n` passes, LF endings; every blob pin equals the CURRENT tree's file (recomputed: a re-pin that was copied, not re-checked, goes red);
  * the R9E2_CKPT_NAME guard (model_ema.pt only) is present and byte-equal to K2DC02's;
  * the guards are RUN (their lines are extracted and executed in bash): the K2DC02s1 argv passes; the K2DC02 argv (seed 0), the seed twice, the seed 2, the K2DC02s1 argv without
    --r9-hist-supervise-geom, a K2SG-like config (no controller) and a missing K0 seed directory are refused;
  * DELIBERATE REGRESSION: a copy without the seed guard lets the K2DC02 (seed 0) config through, so the refusal check goes RED on it;
  * the launcher's literals equal the tokens of stack/ops/runs.d/r9e2_K2DC02s1.argv.json (the single source).
  * THE TRAINER'S REAL LAYOUT (MM 4a3d05, 2026-11-22): the 1f14f8d8 trainer writes the argv TWICE in config.json (`argv` and `argv_effective`). The first
    seed guard counted over the whole file, read flag x2 / pair x2 on the real runs/r9e2-K2DC02s1/config.json and REFUSED a valid read (14:18Z); every
    fixture here had written `{"argv": ...}` only, so no test could see it. The guard now counts inside each array: the two-key layout passes, a
    disagreeing / seedless / doubled `argv_effective` is refused, and the old whole-file line is kept as a literal that must refuse the real layout.
"""
import difflib
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops" / "r9e2_read"
RUNS = ROOT / "ops" / "runs.d"
ARM = "K2DC02s1"
NEW = OPS / "e2read_launch_k2dc02s1.sh"
OLD = OPS / "e2read_launch_k2dc02.sh"
ARGV = RUNS / "r9e2_K2DC02s1.argv.json"
HSG = "--r9-hist-supervise-geom"


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
    return t.replace(ARM, "K2DC02").replace(ARM.lower(), "k2dc02")


def _argv(name):
    return json.loads((RUNS / f"r9e2_{name}.argv.json").read_text(encoding="utf-8"))["argv"]


def _blob(p):
    d = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest()


# ---------------------------------------------------------------------------------------------------------------- #
def test_launcher_has_lf_endings_and_valid_syntax():
    assert b"\r" not in NEW.read_bytes()
    r = subprocess.run([_bash(), "-n", str(NEW)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr


def test_launcher_differs_from_k2dc02_only_by_the_two_added_guards():
    a, b = _text(OLD).splitlines(), _norm(_text(NEW)).splitlines()
    removed, added = [], []
    for ln in difflib.unified_diff(a, b, lineterm="", n=0):
        if ln.startswith(("---", "+++", "@@")):
            continue
        (removed if ln.startswith("-") else added).append(ln[1:])
    rem = [r for r in removed if not r.startswith("#")]
    add = [x for x in added if not x.startswith("#")]
    assert rem == [], rem
    assert len(add) == 2, add
    assert add[0].startswith("NS=$(") and '"--seed","1"' in add[0] and add[1].startswith('case "$CFG1" in *\'"--r9-hist-supervise-geom"\'*) ;;')


def test_every_executable_k2dc02_mention_is_the_arm_or_one_of_the_two_guard_messages():
    for i, ln in enumerate(_text(NEW).splitlines(), 1):
        if ln.lstrip().startswith("#"):
            continue
        rest = ln.replace(ARM, "").replace(ARM.lower(), "")
        if "K2DC02" in rest or "k2dc02" in rest:
            assert ln.startswith(("NS=$(", 'case "$CFG1" in *\'"--r9-hist-supervise-geom"\'*)')), f"line {i} still names K2DC02: {ln[:120]}"
        if "K2DC" in rest and "K2DC02" not in rest:                   # the one intended K2DC: the default K0 seed source
            assert ln.startswith(f"ARM={ARM}; OLD=${{K0_SEED_SRC:-$R/runs/E2_READ_K2DC}};"), f"line {i}: {ln[:120]}"
    t = _text(NEW)
    assert r"arms \[K2DC02s1\]" in t and "' K2DC02s1-DONE '" in t and f"ARM={ARM};" in t and "runs/E2_READ_K2DC}" in t


def test_blob_pins_equal_the_current_trees_files_and_the_ckpt_guard_is_carried():
    t = _text(NEW)
    wrap, roll = OPS / "run_battery_r9_e2.py", OPS / "r9e2_roll.sh"
    if b"\r" in wrap.read_bytes() or b"\r" in roll.read_bytes():
        pytest.skip("CRLF checkout: the blob is not the git blob")
    assert f"WRAP_BLOB={_blob(wrap)}; ROLL_BLOB={_blob(roll)}" in t
    ck = [ln for ln in _text(OLD).splitlines() if ln.startswith('case "${R9E2_CKPT_NAME:-model_ema.pt}"')]
    assert len(ck) == 1 and ck[0] in _text(NEW).splitlines()


def test_launcher_literals_equal_the_argv_json_tokens():
    argv = _argv(ARM)
    assert argv.count("--seed") == 1 and argv[argv.index("--seed") + 1] == "1" and argv.count(HSG) == 1
    t = _text(NEW)
    assert '"--seed","1"' in t and f'"{HSG}"' in t


# ---------------------------------------------------------------------------------------------------------------- #
# the guards are RUN                                                                                                #
# ---------------------------------------------------------------------------------------------------------------- #
def _guard_lines(text):
    keep = [ln for ln in text.splitlines()
            if ln.startswith(("CFG1=", "NS=", 'case "$CFG1"', "for d in $OLD ")) or ln.startswith("grep -q -- '\"--map-hires-share-target\"'")]
    assert len(keep) == 7, keep                                   # share-target present, CFG1, target, bounds, seed (NS), HSG present, K0 seed dirs
    return keep


def _run_guards(text, tmp_path, cfg_tokens, make_seed=True, lines=None, effective=None):
    k2 = tmp_path / "k2"
    k2.mkdir(exist_ok=True)
    cfg = {"argv": cfg_tokens} if effective is None else {"arm": ARM, "seed": 1, "argv": cfg_tokens, "argv_effective": effective}
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


def _case(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return d


def _with_seed(argv, seed):
    a = list(argv)
    a[a.index("--seed") + 1] = seed
    return a


def test_guards_accept_the_replicate_and_refuse_the_rest(tmp_path):
    t = _text(NEW)
    s1 = _argv(ARM)
    rc, out = _run_guards(t, _case(tmp_path, "a"), s1)
    assert rc == 0 and "GUARDS-PASSED" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "b"), _argv("K2DC02"))                              # K2DC02 itself: seed 0
    assert rc == 3 and "--seed 1 exactly once" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "c"), s1 + ["--seed", "1"])                         # seed twice
    assert rc == 3 and "--seed 1 exactly once" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "d"), _with_seed(s1, "2"))                          # seed 2
    assert rc == 3 and "--seed 1 exactly once" in out, out
    nohsg = [x for x in s1 if x != HSG]                                                          # seed 1, but without --r9-hist-supervise-geom: that is K2DC02nh's lever set, not the replicate's
    rc, out = _run_guards(t, _case(tmp_path, "f"), nohsg)
    assert rc == 3 and "lacks --r9-hist-supervise-geom" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "g"), _argv("K2SG"))                                 # no controller at all
    assert rc == 3 and "share-target" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "h"), s1, make_seed=False)
    assert rc == 3 and "K0 seed source directory missing" in out, out


def test_deliberate_regression_without_the_seed_guard_is_caught(tmp_path):
    t = _text(NEW)
    mutant = "\n".join(ln for ln in t.splitlines() if not ln.startswith("NS=$("))
    assert mutant != t
    with pytest.raises(AssertionError):                       # _guard_lines' own census goes red: the guard line is gone
        _guard_lines(mutant)
    keep = [ln for ln in _guard_lines(t) if not ln.startswith("NS=$(")]
    rc, out = _run_guards(t, _case(tmp_path, "m"), _argv("K2DC02"), lines=keep)
    assert rc == 0 and "GUARDS-PASSED" in out, out            # the mutant is blind to a seed-0 config; the pristine guard refuses it (previous test)
    rc, out = _run_guards(t, _case(tmp_path, "p"), _argv("K2DC02"))
    assert rc == 3, out


# the guard line as landed in 34bb48d2 (whole-file count): kept verbatim so the regression it caused stays on record
OLD_SEED_GUARD = ("NS=$(printf '%s' \"$CFG1\" | grep -o '\"--seed\"' | wc -l); NV=$(printf '%s' \"$CFG1\" | grep -o '\"--seed\",\"1\"' | wc -l); "
                  "[ \"$NS\" = 1 ] && [ \"$NV\" = 1 ] || die \"K2DC02s1 config.json must carry --seed 1 exactly once (flag x$NS, pair x$NV): not the replicate argv\"")


def test_guards_read_the_trainers_real_layout_argv_and_argv_effective(tmp_path):
    t = _text(NEW)
    s1, s0 = _argv(ARM), _argv("K2DC02")
    rc, out = _run_guards(t, _case(tmp_path, "r1"), s1, effective=list(s1))                      # the real 1f14f8d8 layout: both arrays, seed 1 once each
    assert rc == 0 and "GUARDS-PASSED" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "r2"), s1, effective=_with_seed(s1, "0"))           # argv_effective disagrees
    assert rc == 3 and "--seed 1 exactly once" in out and "argv_effective" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "r3"), s1, effective=[x for x in s1 if x != "--seed"])   # argv_effective without a --seed flag
    assert rc == 3 and "--seed 1 exactly once" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "r4"), s1, effective=s1 + ["--seed", "1"])         # argv_effective with the seed twice
    assert rc == 3 and "--seed 1 exactly once" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "r5"), s0, effective=list(s1))                      # argv seed 0 (K2DC02), argv_effective seed 1
    assert rc == 3 and "--seed 1 exactly once" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "r6"), s0, effective=list(s0))                      # K2DC02's own real layout: seed 0 twice over
    assert rc == 3 and "--seed 1 exactly once" in out, out


def test_the_old_whole_file_seed_guard_refuses_the_real_layout(tmp_path):
    t = _text(NEW)
    assert OLD_SEED_GUARD not in t
    keep = [OLD_SEED_GUARD if ln.startswith("NS=$(") else ln for ln in _guard_lines(t)]
    assert keep != _guard_lines(t)
    s1 = _argv(ARM)
    rc, out = _run_guards(t, _case(tmp_path, "o1"), s1, lines=keep)                              # one array: the old line passed (why the fixture missed it)
    assert rc == 0 and "GUARDS-PASSED" in out, out
    rc, out = _run_guards(t, _case(tmp_path, "o2"), s1, lines=keep, effective=list(s1))          # the real layout: the old line REFUSES a valid read
    assert rc == 3 and "flag x2, pair x2" in out, out
