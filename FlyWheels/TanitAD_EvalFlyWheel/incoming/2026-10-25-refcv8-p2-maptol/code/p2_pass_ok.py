"""p2_pass_ok.py <p2_<arm>.json> <min_map_windows> -> prints 1 iff the pass artifact exists, covers >= min windows and every in-pass control passed, else 0.
Assert on the ARTIFACT (CLAUDE.md): never an exit code, never `$?` through a pipe."""
import json
import sys

try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    c = d["controls"]
    ok = (d["n_map_windows"] >= int(sys.argv[2]) and c["law_frame"]["pass"]
          and c["C4_cheb_k1_vs_EDT_0p2m"]["mismatching_cells"] == 0
          and c["C0_counts_vs_MH_window_stats"]["mismatching_arrays"] == 0
          and c["C1p_gt_as_pred"]["all_pass"] and c["C2p_all_one_class"]["all_pass"])
    print(1 if ok else 0)
except Exception:                                # missing / partial artifact == not done
    print(0)
