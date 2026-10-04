"""MEASURED probe (gate agent, 2026-09-27): the trainer's own `--dry-run` on a nav launch of the merge.
Arm: the tiny S-W rehearsal argv WITH --nav-cond (as every v7F launch must be). Control: the identical argv with
the pre-nav acknowledgement instead (nav not built). Both call train_v6_staged.main() directly -- no gate seam."""
import json, sys, tempfile, traceback
from pathlib import Path
sys.path.insert(0, "C:/Users/Admin/v7f_gate/tree_m/stack")
sys.path.insert(0, "C:/Users/Admin/v7f_gate/tree_m/stack/tests")
sys.path.insert(0, "C:/Users/Admin/v7f_gate/tree_m/stack/scripts")
import test_launch_gate_v7f as TT
LG = TT.LG
root = Path(tempfile.mkdtemp(prefix="dryrun_nav_", dir="C:/Users/Admin/v7f_gate/gate_runs"))
ctx = TT._ctx(root, TT.SW_ARGV)
T = LG.load_trainer_v6(ctx)
tiny, _ = LG.v6_rehearsal_argv(ctx.prof, TT.SW_ARGV, root)
res = {}
for label, argv in (("nav_cond", tiny),
                    ("control_pre_nav", LG.set_flag(LG.set_flag(LG.set_flag(tiny, "--nav-cond", None),
                                                                "--nav-labels", None),
                                                    "--i-know-this-arm-predates-nav", []))):
    argv = LG.set_flag(LG.set_flag(argv, "--out", [str(root / label)]), "--dry-run", [])
    try:
        rc = T.main(argv)
        res[label] = {"rc": rc, "raised": None}
    except SystemExit as e:
        res[label] = {"rc": f"SystemExit {e}", "raised": None}
    except Exception as e:  # noqa: BLE001
        res[label] = {"rc": None, "raised": f"{type(e).__name__}: {str(e)[:220]}",
                      "where": traceback.format_exc().splitlines()[-8:]}
    res[label]["argv_has_nav_cond"] = "--nav-cond" in argv
print("RESULT", json.dumps(res, indent=1))
