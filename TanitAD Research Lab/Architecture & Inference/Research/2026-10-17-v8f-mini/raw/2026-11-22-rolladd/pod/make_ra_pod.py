"""make_ra_pod.py -- ROLLADD build. Writes the RA (arm `--dagger-mode add`) pod scripts into this folder from the RL cf27dbb7 forms in F:/TanitAD-scratch/reclab/pod/{,cf27/} (sources sha256-pinned below).
RA = the RH argv + exactly `--dagger-mode add` and its own `--out` (same bank, same dose 0.1, same declarations). The RL chain is the template, file for file; every edit is an exact anchor that must match
the stated number of times or this script fails. The RA commit is NOT known at build time: every pin that depends on it is a `__RA_*__` placeholder that fill_ra_commit.sh (MM, after landing) replaces and verifies.
Usage: python make_ra_pod.py"""
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RL = "F:/TanitAD-scratch/reclab/pod"
SRC = {
    "rh_rl_chain.sh": (f"{RL}/cf27/rh_rl_chain.sh", "d8a660055627f77c"), "rl_arm.sh": (f"{RL}/cf27/rl_arm.sh", "0ee7c8a08e254125"), "chk_arm_rl.sh": (f"{RL}/cf27/chk_arm_rl.sh", "784d0162eac05642"),
    "rl_gsuite.sh": (f"{RL}/cf27/rl_gsuite.sh", "2a2c2d0177b252ae"), "rl_gate_record.py": (f"{RL}/cf27/rl_gate_record.py", "a9a0a5fa76642870"), "rl_argv.py": (f"{RL}/rl_argv.py", "877349052dac9b88"),
    "rh_rl_marker_watch.sh": (f"{RL}/rh_rl_marker_watch.sh", "53a9ddd03a6df0f5"), "launch_rl.sh": (f"{RL}/launch_rl.sh", "8d3ee4bcf7ee6425"),
    "test_rl_gate_record.py": (f"{RL}/cf27/test_rl_gate_record.py", "3f6b78e950a579bb"),
}
NAMES = [("GSUITE_RL", "GSUITE_RA"), ("[r]h_rl_chain", "[r]h_ra_chain"), ("rh_rl_chain[.]sh", "rh_ra_chain[.]sh"), ("tree_cf27", "tree_ra"), ("three label checks", "three add-mode checks"), ("three label", "three add-mode"), ("rh_rl_marker_watch", "rh_ra_marker_watch"), ("rh_rl_chain", "rh_ra_chain"), ("chk_arm_rl", "chk_arm_ra"), ("dscale10k_check_rl", "dscale10k_check_ra"), ("gsuite_result_rl", "gsuite_result_ra"),
         ("test_rl_gate_record", "test_ra_gate_record"), ("rl_gate_record", "ra_gate_record"), ("rl_gsuite", "ra_gsuite"), ("rl_argv", "ra_argv"), ("rl_arm", "ra_arm"), ("rl_env", "ra_env"), ("launch_rl", "launch_ra"),
         ("done_rl", "done_ra"), ("junit_rl", "junit_ra"), ("tmp_gsuite_rl", "tmp_gsuite_ra"), ("suite_compare_rl", "suite_compare_ra"), ("blobcheck_rl", "blobcheck_ra"), ("inputs_verify_rl", "inputs_verify_ra"),
         ("rl_render", "ra_render"), ("rl_marker_watch", "ra_marker_watch"), ("ROLLHIST-RL", "ROLLHIST-RA"), ("RL-RENDER", "RA-RENDER"), ("RL GSUITE", "RA GSUITE"), ("RL TREE", "RA TREE"),
         ("RL LAUNCHED", "RA LAUNCHED"), ("RL-GATE-RECORD", "RA-GATE-RECORD"), ("rl-chain-gate-record", "ra-chain-gate-record"), ("argv_nolabel", "argv_noadd"), ("rl_label_rows_changed", "rows_appended")]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read(name):
    p, want = SRC[name]
    raw = open(p, "rb").read()
    assert sha(raw).startswith(want), f"{name}: sha256 {sha(raw)[:16]} is not the RL build's {want}"
    return raw.decode("utf-8")


def edit(s, edits, name):
    for old, new, n in edits:
        c = s.count(old)
        assert c == n, f"{name}: anchor matched {c} times, want {n}: {old[:80]!r}"
        s = s.replace(old, new)
    return s


def rename(s):
    for a, b in NAMES:
        s = s.replace(a, b)
    s = re.sub(r"\bRL_s", "RA_s", s)
    s = re.sub(r"\bRL\b", "RA", s)
    s = s.replace("--dagger-label speed_v1", "--dagger-mode add").replace("--dagger-label", "--dagger-mode").replace("dagger_label", "dagger_mode").replace("speed_v1", "add")
    s = s.replace("cf27dbb7", "the RA commit")
    return s


def write(name, s, mode_exec=False):
    assert "speed_v1" not in s and "--dagger-label" not in s, f"{name}: a label token is left"
    open(os.path.join(HERE, name), "w", encoding="utf-8", newline="\n").write(s)
    print("wrote", name, sha(s.encode())[:16])


def main():
    # ---- chain, arm, checker wrapper, gsuite, marker watcher, launcher: the RL forms with the renames, plus the anchored semantic edits
    for src, dst, extra in [
        ("rh_rl_chain.sh", "rh_ra_chain.sh", [("(the label lever on own states; the card's HARMS-row next lever; Amendment 9 DRAFT)", "(the add-mode lever: own states APPENDED, logged rows kept; PREREG_V8F_ROLLADD)", 1),
                                              ("RL_s0 RL_s1 (--dagger-p 0.1 --dagger-label speed_v1, same banks)", "RA_s0 RA_s1 (--dagger-p 0.1 --dagger-mode add, same banks)", 1)]),
        ("rl_arm.sh", "ra_arm.sh", [("assert t[i + 1] == \"speed_v1\" and i + 2 == len(t)          # exactly the last two tokens", "assert t[i + 1] == \"add\" and i + 2 == len(t)          # exactly the last two tokens", 1)]),
        ("chk_arm_rl.sh", "chk_arm_ra.sh", []), ("rl_gsuite.sh", "ra_gsuite.sh", []), ("rh_rl_marker_watch.sh", "rh_ra_marker_watch.sh", []), ("launch_rl.sh", "launch_ra.sh", []),
    ]:
        s = read(src)
        s = edit(s, [e for e in extra if e[0] in s], src) if extra else s
        # the two label-specific anchors above are applied BEFORE the generic rename, on the RL text
        s = rename(s)
        write(dst, s)
    # the launcher also refuses while an RL chain runs (one chain at a time)
    lp = os.path.join(HERE, "launch_ra.sh")
    t = open(lp, encoding="utf-8").read()
    anchor = 'ps -eo args | grep -q "[r]h_ra_chain.sh" && { echo "an rh_ra_chain.sh is running; refusing"; exit 1; }' + chr(10)
    assert t.count(anchor) == 1
    t = t.replace(anchor, anchor + 'ps -eo args | grep -q "[r]h_rl_chain.sh" && { echo "an rh_rl_chain.sh is running; refusing"; exit 1; }' + chr(10))
    open(lp, "w", encoding="utf-8", newline=chr(10)).write(t)
    # the ra_arm.sh label strip: after rename the python block asserts the exact last two tokens `--dagger-mode add`
    a = open(os.path.join(HERE, "ra_arm.sh"), encoding="utf-8").read()
    assert 'i = t.index("--dagger-mode")' in a and 't[i + 1] == "add" and i + 2 == len(t)' in a and "rollhist_pair_check.py --rh $D/argv_noadd.json" in a
    # ---- the argv renderer
    s = rename(read("rl_argv.py"))
    s = edit(s, [("LABEL = [\"add\"]", "MODE = [\"add\"]", 1), ("a.get(\"--dagger-label\")", "a.get(\"--dagger-mode\")", 0)], "ra_argv.py") if False else s
    s = s.replace("LABEL = [\"add\"]", "MODE = [\"add\"]").replace("a.get(\"--dagger-mode\") != LABEL", "a.get(\"--dagger-mode\") != MODE").replace("!= LABEL", "!= MODE").replace("== LABEL", "== MODE")
    s = s.replace("the label lever", "the add-mode lever")
    assert "LABEL" not in s, "ra_argv.py: LABEL is left"
    s = s.replace("RH_s{seed} already names --dagger-mode {f['--dagger-mode']}: RH is the human-label arm", "RH_s{seed} already names --dagger-mode {f['--dagger-mode']}: RH is the default-mode (swap) arm")
    write("ra_argv.py", s)
    # ---- the gate record: the RA branch is the RH branch plus three add-mode checks; any other tag is refused
    s = read("rl_gate_record.py")
    s = edit(s, [
        ('COMMIT = "cf27dbb79f2491a6db1197860bc50ca7c8cdeff4"', 'COMMIT = "__RA_COMMIT__"                                      # filled by fill_ra_commit.sh (MM, after landing): 40 hex or this file refuses to run', 1),
        ('        if "speed_v1" not in set(map(str, find_key(sc.get("levers", {}), "dagger_label"))):\n            bad.append("the built levers do not carry dagger_label speed_v1")\n'
         '        rows = [v for v in find_key(sc.get("dagger_probe") or {}, "rl_label_rows_changed") if isinstance(v, (int, float))]\n        if not rows or max(rows) <= 0:\n'
         '            bad.append(f"the dagger probe read rl_label_rows_changed {rows!r}: the speed-consistent label did not reach the batch")\n',
         '        if "add" not in set(map(str, find_key(sc.get("levers", {}), "dagger_label"))):\n            bad.append("the built levers do not carry dagger_mode add")\n'
         '        pr = sc.get("dagger_probe") or {}\n'
         '        rows = [v for v in find_key(pr, "rows_appended") if isinstance(v, (int, float))]\n        if not rows or max(rows) <= 0:\n'
         '            bad.append(f"the dagger probe read rows_appended {rows!r}: no own row was appended to the batch")\n'
         '        gd = [v for v in find_key(pr, "grad_diff_all") if isinstance(v, (int, float))]\n        if not gd or max(gd) <= 0:\n'
         '            bad.append(f"the dagger probe read grad_diff_all {gd!r}: the appended rows do not move the gradient")\n'
         '        z = [v for v in find_key(pr, "control_p0_grad_diff") if isinstance(v, (int, float))]\n        if not z or max(z) != 0.0:\n'
         '            bad.append(f"the dagger probe control_p0_grad_diff reads {z!r}, not exactly 0.0")\n', 1),
        ('        if m.get("--dagger-label") != "speed_v1":\n            bad.append(f"RL argv --dagger-label is {m.get(\'--dagger-label\')!r}, not speed_v1")\n',
         '        if m.get("--dagger-mode") != "add":\n            bad.append(f"RA argv --dagger-mode is {m.get(\'--dagger-mode\')!r}, not add")\n', 1)], "rl_gate_record.py")
    s = rename(s).replace('find_key(sc.get("levers", {}), "dagger_mode")', 'find_key(sc.get("levers", {}), "dagger_mode")')
    s = s.replace('"add" not in set(map(str, find_key(sc.get("levers", {}), "dagger_label")))', '"add" not in set(map(str, find_key(sc.get("levers", {}), "dagger_mode")))')
    s = s.replace('COMMIT = "__RA_COMMIT__"                                      # filled', 'COMMIT = "__RA_COMMIT__"                                      # filled')
    s = s.replace("if tag not in (\"RA_s0\", \"RA_s1\"):", "if tag not in (\"RA_s0\", \"RA_s1\") or not re.fullmatch(r\"[0-9a-f]{40}\", COMMIT):").replace("import argparse\n", "import argparse\nimport re\n", 1)
    s = s.replace('print("PROBLEM ra_gate_record.py serves RA_s0 / RA_s1 only")', 'print("PROBLEM ra_gate_record.py serves RA_s0 / RA_s1 only, with a filled commit (fill_ra_commit.sh)")')
    assert "dagger_label" not in s and "speed" not in s.replace("speed-consistent", "")
    write("ra_gate_record.py", s)
    # ---- its test (planted files)
    s = read("test_rl_gate_record.py")
    s = rename(s)
    s = s.replace('CF27 = "cf27dbb79f2491a6db1197860bc50ca7c8cdeff4"', 'CF27 = "0123456789abcdef0123456789abcdef01234567"      # a planted RA commit; the record under test has it filled in by the test')
    s = s.replace('"human" if mut == "M2" else "add"', '"swap" if mut == "M2" else "add"').replace('"human" if mut == "M3" else "add"', '"swap" if mut == "M3" else "add"')
    s = s.replace('probe = {"pass": True, "n_selected": 12}', 'probe = {"pass": True, "n_selected": 12, "grad_diff_all": 0.0 if mut == "M11" else 3.5, "control_p0_grad_diff": 0.25 if mut == "M12" else 0.0}')
    s = s.replace('probe["rows_appended"] = 0 if mut == "M4" else 7', 'probe["rows_appended"] = 0 if mut == "M4" else 12')
    s = s.replace('"97fe0a56767320dbc5f4b823d67fb07a3209a745" if mut == "M6"', '"cf27dbb79f2491a6db1197860bc50ca7c8cdeff4" if mut == "M6"')
    s = s.replace('for m in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"):', 'for m in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M11", "M12"):')
    s = s.replace("GR, ROOT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])", "GR0, ROOT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])\nGR = os.path.join(ROOT, 'ra_gate_record_filled.py')")
    s = s.replace("ok = True\nR = os.path.join(ROOT, \"R\")", "os.makedirs(ROOT, exist_ok=True)\nopen(GR, 'w', encoding='utf-8', newline='\\n').write(open(GR0, encoding='utf-8').read().replace('__RA_COMMIT__', CF27))\nok = True\nR = os.path.join(ROOT, \"R\")")
    s = s.replace("M1 argv without --dagger-mode; M2 --dagger-mode human;", "M1 argv without --dagger-mode; M2 --dagger-mode swap; M11 probe grad_diff_all 0; M12 control_p0_grad_diff != 0; M13 the commit placeholder left unfilled;")
    s = s.replace("M3 built levers without add; M4 dagger probe rows changed 0;", "M3 built levers say swap; M4 rows_appended 0;")
    s = s.replace("shutil.rmtree(R)\nprint(", "unf = subprocess.run([sys.executable, '-I', GR0, 'mint', '--tag', 'RA_s0', '--readout', 'causal_g1x'], env=dict(os.environ, RH_ROOT=R), capture_output=True, text=True, encoding='utf-8')\nred = unf.returncode != 0 and 'REFUSED' in unf.stdout\nprint('M13', 'RED' if red else 'NOT-RED')\nok &= red\nshutil.rmtree(R)\nprint(")
    s = s.replace("Usage: python -I test_ra_gate_record.py <ra_gate_record.py>", "Usage: python -I test_ra_gate_record.py <ra_gate_record.py (unfilled)>")
    write("test_ra_gate_record.py", s)
    print("ZZ MAKE-RA-POD DONE")


if __name__ == "__main__":
    sys.exit(main())
