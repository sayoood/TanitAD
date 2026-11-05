"""Run each REINDEXED mutation against its check and print the failure it raises (the log of 'mutations go RED'). Usage (from the tree, CPU): python -I code/run_mutations.py"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
TREE = Path(sys.argv[1] if len(sys.argv) > 1 else "F:/TanitAD-scratch/motion_c/ri/tree_ri")
for p in (TREE / "stack", TREE / "taniteval", TREE / "stack" / "scripts", TREE / "stack" / "tests"):
    sys.path.insert(0, str(p))
import test_v8f_t1_ri as T  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402

root = Path(tempfile.mkdtemp(prefix="syn_ri_mut_", dir="F:/TanitAD-scratch/motion_c/ri"))
world = {"root": root, **D.make_synthetic_world(root, n_train=2, n_eval=3, T=90, seed=5)}
print("MUT unmutated tool:", end=" ")
for chk in (T.check_kri2, T.check_kri3, T.check_kri5):
    chk(T.T1, world)
    print(chk.__name__, "PASS", end="; ")
print()
bad = 0
for old, new, why, check in ((T.APPLY, "        pass\n", "m1: the apply line deleted", T.check_kri2),
                             (T.CLAMP, "tau_star", "m2: the min(i, .) clamp dropped", T.check_kri3),
                             (T.PASTMASK, "past = i_rows < -(10 ** 9)", "m3: the past rows reindexed too", T.check_kri5)):
    mt = T.mutated(old, new)
    try:
        check(mt, world)
        print(f"MUT {why} -> {check.__name__}: SURVIVED (NOT RED)")
        bad += 1
    except AssertionError as e:
        print(f"MUT {why} -> {check.__name__}: RED  AssertionError: {str(e)[:200]}")
# m3 through the receipt's own past-row check
mt = T.mutated(T.PASTMASK, "past = i_rows < -(10 ** 9)")
sw = T.make_straight(world, mt, 6.0, 20, perturb_past=3.0)
mt.ri_reset()
T.drive(sw, 3.0)
ps = mt.RI_STATS["past"]
red = ps["rows_checked"] > 0 and ps["rows_changed"] > 0
print(f"MUT m3 -> receipt past-row check: {'RED' if red else 'NOT RED'} (rows_checked {ps['rows_checked']}, rows_changed {ps['rows_changed']})")
bad += int(not red)
print("MUTATIONS ALL RED" if bad == 0 else f"MUTATIONS NOT ALL RED ({bad})")
sys.exit(1 if bad else 0)
