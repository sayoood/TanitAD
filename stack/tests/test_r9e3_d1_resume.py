"""E3 resumes on ITS OWN launch tree after the early-D1 restart at a newer tip (stack/ops/r9e3_kit/k2dc02/d1_restart.sh + the TREEDRIFT branch of resume_e3.sh).

CPU only, no ssh, no GPU, nothing is run on Thor. Pins, from the sources:
  * LF endings, valid bash syntax;
  * every guard exists BY NAME in the code (not a comment), DRY changes nothing (its exit comes before the first write), REAL-only steps come after it;
  * ORDER: the fingerprint and sidecars are written, then E3's tree is renamed, then the D1 restart runs; at the resume D1's tree moves aside BEFORE E3's comes back, and the
    re-check comes before the supervisor starts; a failed re-check swaps the trees back;
  * a tree is never deleted (no rm / rmdir in either script, only mv);
  * the tree fingerprint is the SAME string in both scripts, is stable, and moves when a file's size, a path or the supervisor / trainer bytes change;
  * the closure path list equals prep_q28.sh's RECINPUT list (and restart_e3.sh's).
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "ops" / "r9e3_kit" / "k2dc02"
RUNS = ROOT / "ops" / "runs.d"
FILES = ("d1_restart.sh", "resume_e3.sh")


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


def _code(name):
    return "\n".join(ln for ln in _t(name).splitlines() if not ln.lstrip().startswith("#"))


def _fpsh(name):
    m = re.search(r'^FPSH="(.*)"$', _t(name), re.M)
    assert m, f"{name} defines no FPSH"
    return m.group(1)


@pytest.mark.parametrize("name", FILES)
def test_lf_endings_and_bash_syntax(name):
    assert b"\r" not in (KIT / name).read_bytes()
    r = subprocess.run([_bash(), "-n", str(KIT / name)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr


def test_d1_restart_guards_present_by_name():
    code = _code("d1_restart.sh")
    for g in ("NOPAUSED", "NOCKPT", "NOMANIFEST", "CHAINLIVE", "SUPLIVE", "GPULOCK", "NOTIP", "TREEMISMATCH", "PRESEXISTS", "NOAPPROVAL", "NOARM", "NOZGENV",
              "CLOSUREMOVED", "NOD1RECORDS"):
        assert f"bad {g} " in code, g
    assert "K2DC02_d1" in _t("d1_restart.sh") and "R9_STUDY=e3" in code and 'R9_PRECHECK_ARMS="$D1"' in code
    # the D1 record marker queue 28 writes (r9_recqueue28.sh) and the D1-only rebind named in the refusal
    assert "${D1}-RECORDS-READY at $RC12" in code and "R9Q28_D1_ONLY=1" in code
    assert "pgrep" not in code and "pkill" not in code


def test_resume_guards_present_by_name():
    code = _code("resume_e3.sh")
    for g in ("NOPAUSED", "ALREADYDONE", "NOMANIFEST", "TREEDRIFT", "CHAINLIVE", "D1NOTDONE", "D1NOSUMMARY", "RESTORE", "RESTORECHECK"):
        assert f"bad {g} " in code, g
    assert "pgrep" not in code and "pkill" not in code


def test_dry_changes_nothing_and_flags_exist():
    d = _code("d1_restart.sh")
    assert 'DRY|REAL' in d and 'if [ "$MODE" = DRY ]; then say "ZZD1RESTART-DRYENDZZ' in d
    dry = d.index('if [ "$MODE" = DRY ]; then say "ZZD1RESTART-DRYENDZZ')
    for later in ('mkdir -p "$D"', "mv $R/launch_tree $PRES", "echo $FP > $PRES.fp", 'bash "$D/stack/ops/r9e1_gate3/r9gate3_restart.sh"'):
        assert d.index(later) > dry, later
    assert d.index('[ "$MODE" = DRY ] || [ -n "$(th "flock -n $LOCKF') < dry          # the lock is only demanded in REAL
    r = _code("resume_e3.sh")
    dry = r.index('if [ "$MODE" = DRY ]; then say "ZZRESUMEE3-DRYOKZZ')
    for later in ('if [ "$RESTORE" = 1 ]; then', "mv $PRES launch_tree", "setsid nohup"):
        assert r.index(later) > dry, later
    # DRY failing guards are counted, not fatal
    assert 'if [ "$MODE" = DRY ]; then FAILS=$((FAILS+1)); else exit' in d and 'if [ "$MODE" = DRY ]; then FAILS=$((FAILS+1)); else exit' in r


def test_preserve_then_restart_order_in_d1_restart():
    d = _code("d1_restart.sh")
    i_fp = d.index('FP=$(th "cd $R/launch_tree && $FPSH")')
    i_side = d.index("echo $FP > $PRES.fp && echo $MC > $PRES.commit")
    i_mv = d.index("mv $R/launch_tree $PRES")
    i_run = d.index('bash "$D/stack/ops/r9e1_gate3/r9gate3_restart.sh"')
    assert i_fp < i_side < i_mv < i_run
    assert 'PRES=$R/launch_tree_e3_${MC:0:12}' in d
    # the D1 restart, not the E3 arm, is what is restarted, with the record commit the caller named
    assert '"$TIP" "$D1" $RC12' in d
    # a refusal before the ship puts E3's tree back by rename
    assert d.index('mv $PRES $R/launch_tree && echo back') > i_run


def test_restore_order_in_resume():
    r = _code("resume_e3.sh")
    i_aside = r.index("mv launch_tree $ASIDE")
    i_back = r.index("mv $PRES launch_tree")
    i_commit = r.index("echo $MC > launch_tree.commit")
    i_verify = r.index("launch_gate.py verify")
    i_check = r.index('if [ "$OKR" != 1 ]; then')
    i_start = r.index("setsid nohup")
    assert i_aside < i_back < i_commit < i_verify < i_check < i_start
    # the re-check reads the commit file, the fingerprint AND the gate's own verdict
    assert 'cat $R/launch_tree.commit' in r and "cat $PRES.fp" in r and "MATCH" in r
    # D1's commit is kept next to its aside tree; a failed re-check swaps back
    assert "cp launch_tree.commit $ASIDE.commit" in r and "mv launch_tree $PRES && mv $ASIDE launch_tree" in r
    # the restore needs the manifest's commit to equal the preserved .commit
    assert 'test \\"\\$(cat $PRES.commit)\\" = $MC' in r


@pytest.mark.parametrize("name", FILES)
def test_never_deletes_a_tree(name):
    code = _code(name)
    assert not re.search(r"\brm\b|\brmdir\b|\bshred\b|find [^|]*-delete", code), "a tree (or anything) must be moved aside, never deleted"
    assert re.search(r"\bmv\b", code)


def test_fingerprint_is_one_string_and_behaves(tmp_path):
    a, b = _fpsh("d1_restart.sh"), _fpsh("resume_e3.sh")
    assert a == b, "d1_restart.sh and resume_e3.sh must compute the identical fingerprint"
    tree = tmp_path / "t"
    for rel, data in (("stack/ops/sup_refcv7.sh", b"sup\n"), ("stack/scripts/refc_v3_train.py", b"train\n"), ("stack/tanitad/m.py", b"x=1\n"), ("taniteval/e.py", b"e\n")):
        p = tree / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def fp():
        r = subprocess.run([_bash(), "-c", f"cd '{tree.as_posix()}' && {a}"], capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 0, r.stderr
        assert re.fullmatch(r"[0-9a-f]{64}", r.stdout.strip()), r.stdout
        return r.stdout.strip()

    base = fp()
    assert fp() == base                                                                    # stable
    (tree / "stack/tanitad/__pycache__").mkdir()
    (tree / "stack/tanitad/__pycache__/m.cpython-312.pyc").write_bytes(b"bytecode")
    (tree / "stack/tanitad/m.pyc").write_bytes(b"x")
    assert fp() == base                                                                    # bytecode is ignored
    (tree / "stack/tanitad/m.py").write_bytes(b"x=12\n")
    changed = fp()
    assert changed != base                                                                 # a size change moves it
    (tree / "stack/tanitad/m.py").write_bytes(b"x=1\n")
    assert fp() == base
    (tree / "stack/tanitad/new.py").write_bytes(b"")
    assert fp() != base                                                                    # a new file moves it
    (tree / "stack/tanitad/new.py").unlink()
    (tree / "stack/ops/sup_refcv7.sh").write_bytes(b"sux\n")                               # same size, other bytes: the supervisor hash moves it
    assert fp() != base


def _closure_tokens(text, anchor):
    i = text.index(anchor)
    seg = text[i:].split("||")[0]
    seg = seg.replace("\\\n", " ")
    seg = seg.split(" -- ", 1)[1]
    seg = seg.replace("${V}", "K2DC02").replace("$V", "K2DC02").replace("$RD", "stack/ops/runs.d").replace("\n", " ")
    return sorted(seg.split())


def test_closure_list_equals_prep_q28_recinput_list():
    prep = _closure_tokens(_t("prep_q28.sh"), "diff --quiet $REC $TIP")
    d1 = _closure_tokens(_t("d1_restart.sh"), "diff --quiet $REC $TIP")
    rst = _closure_tokens(_t("restart_e3.sh"), "diff --quiet $REC $T ")
    assert d1 == prep, (d1, prep)
    # restart_e3.sh's list has no cost-approval files (it checks them per arm below); everything else is the same
    assert set(rst) <= set(prep) and set(prep) - set(rst) == {
        "stack/ops/runs.d/r9e3_K2DC02.pi_cost_approval.json", "stack/ops/runs.d/r9e3_K2DC02_d1.pi_cost_approval.json"}
    for must in ("stack/tanitad", "taniteval", "stack/scripts/refc_v3_train.py", "stack/scripts/map_hires_overfit.py", "stack/scripts/g_box_overfit.py"):
        assert must in d1, must
    # the r9e1_bind.sh non-comment-line check is the same expression in both
    nc = "grep -E '^[+-][^+-]' | grep -v -E '^[+-][[:space:]]*#' | wc -l"
    assert nc in _t("d1_restart.sh") and nc in _t("prep_q28.sh")


def test_d1_argv_facts_the_script_checks_hold_for_the_landed_argv():
    import json
    a = json.loads((RUNS / "r9e3_K2DC02_d1.argv.json").read_text(encoding="utf-8"))["argv"]
    v = lambda k: a[a.index(k) + 1]
    assert a.count("--steps") == 1 and v("--steps") == "36000" and a.count("--wsd-branch-from") == 1
    assert v("--wsd-branch-from").endswith("/r9e3/runs/r9e3-K2DC02/ckpt_full_30000.pt") and v("--wsd-branch-decay") == "6000"
    assert "--save-at" not in a and a.count("--r9-map-banded") == 1
    d = _t("d1_restart.sh")
    for lit in ("'36000'", "'6000'", "ckpt_full_30000.pt", "'--save-at' not in a"):
        assert lit in d, lit
    assert '\\"max_s_per_step\\": $CAP,' in d and "CAP=10.5" in d


def test_d1_out_dir_and_study_constants_match_the_argv_and_resume():
    import json
    a = json.loads((RUNS / "r9e3_K2DC02_d1.argv.json").read_text(encoding="utf-8"))["argv"]
    assert a[a.index("--out") + 1] == "/home/nvidia/r8ns/r9e3/runs/r9e3-K2DC02-d1"
    assert "$R/runs/r9e3-K2DC02-d1/summary.json" in _t("resume_e3.sh")
    assert "R=/home/nvidia/r8ns/r9e3" in _t("d1_restart.sh") and "R=/home/nvidia/r8ns/r9e3" in _t("resume_e3.sh")
