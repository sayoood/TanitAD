"""Did a K2 knockout (argv: tag [seed]) move the planner output at all?  max |P_knockout - P_K2| per window, same inference seed, DEV+HOLD all windows."""
import sys
from pathlib import Path
import numpy as np
sys.path[:0] = ["F:/refcv9_e3/famfix_mm/tree/stack/scripts", "F:/refcv9_e3/famfix_mm/tree/stack", "F:/refcv9_e3/famfix_mm/tree/taniteval"]
import r9_e2_families as F
B = Path("C:/Users/Admin/r9e2_read_K2M1N20")
TAG = sys.argv[1] if len(sys.argv) > 1 else "mapzero"; SD = sys.argv[2] if len(sys.argv) > 2 else "0"
A = F.load_dump(B / f"abl/E2_K2abl_{TAG}_pf/dump_s{SD}"); K = F.load_dump(B / f"reread_dtfix/src/E2_K2_pf/dump_s{SD}")
pa, pk, eid, rep = F.align(A, K)
d = np.abs(pa["P"] - pk["P"]).reshape(len(eid), -1).max(1)
print("windows", len(d), "gt_max_abs_diff", rep["gt_max_abs_diff_m"])
print("bit-identical windows", int((d == 0).sum()), "| max |dP| m: median %.2e p90 %.2e max %.2e" % (np.median(d), np.quantile(d, .9), d.max()))
print("windows moved > 1 mm", int((d > 1e-3).sum()), "> 1 cm", int((d > 1e-2).sum()))
