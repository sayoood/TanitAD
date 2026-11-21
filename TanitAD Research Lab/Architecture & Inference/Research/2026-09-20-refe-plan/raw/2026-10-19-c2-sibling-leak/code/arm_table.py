"""Print the arm x gate table of C2-GATE-SIBLING-LEAK: each arm (poison, deleted-dir, failed-load) on the gate in <tree>, and on each mutant of that gate that restores one process-global lookup.
    python -B -I arm_table.py <tree>      (the tree holds srttc_c2_gate.py and the NEW test file; a tree with the OLD gate shows the arms on the unfixed gate and no mutants)
CPU only; no world, no R1 run (the arms do not need them).  Output: one markdown table + the first assertion text of every RED cell."""
import os
import sys

tree = os.path.abspath(sys.argv[1])
os.chdir(tree)
sys.path.insert(0, tree)
import test_srttc_c2_gate as T   # noqa: E402

CG = T.CG
tc = T.TestEachLoadUsesExactlyTheVerifiedFiles("test_a_second_code_dir_runs_its_own_srttc_r1_poison_arm")
T.TestEachLoadUsesExactlyTheVerifiedFiles.setUpClass()
ARMS = (("poison (A, B, A2 + a global decoy)", tc.arm_poison), ("deleted dir (A gone, B identical)", tc.arm_gone), ("failed load leaves nothing", tc.arm_failed_load))
MUTANTS = (("sha-keyed sys.modules lookup (old loaders)", "MUTATION-TARGET-SIBLING-NAME", "name = stem\nif name in sys.modules:\n    return sys.modules[name]"),
           ("sibling import via sys.modules / sys.path (old global binding)", "MUTATION-TARGET-SIBLING-BIND", "if False:"),
           ("failed load stays registered (no poison guard)", "MUTATION-TARGET-POISON-GUARD", "pass"))


def cell(arm, gm):
    try:
        arm(gm)
        return "GREEN", ""
    except AssertionError as e:
        return "RED", str(e).splitlines()[0][:160]
    except BaseException as e:                                                       # noqa: BLE001
        return f"RED ({type(e).__name__})", str(e).splitlines()[0][:160] if str(e) else ""


rows = [("gate in " + os.path.basename(tree), CG)]
for i, (label, marker, new) in enumerate(MUTANTS):
    try:
        rows.append((label, tc.mutant(marker, new, f"t{i}")))
    except AssertionError:
        pass                                                                         # the old gate has no such marker
print("| gate | " + " | ".join(a for a, _ in ARMS) + " |")
print("|---|" + "---|" * len(ARMS))
notes = []
for label, gm in rows:
    cells = []
    for aname, arm in ARMS:
        st, why = cell(arm, gm)
        cells.append(st)
        if st != "GREEN":
            notes.append(f"- {label} / {aname}: {why}")
    print(f"| {label} | " + " | ".join(cells) + " |")
print()
print("\n".join(notes))
T.TestEachLoadUsesExactlyTheVerifiedFiles.tearDownClass()
