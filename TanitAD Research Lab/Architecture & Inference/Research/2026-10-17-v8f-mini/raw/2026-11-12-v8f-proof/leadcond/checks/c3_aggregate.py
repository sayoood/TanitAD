"""c3 aggregate: reads c3_reach_patch.json (the lever as built: every read true), c3_reach_mutO.json and c3_reach_mutT.json (copies of the tree with the line adding the token in run_o / run_t replaced by `pass`:
the reach read must be RED in each). PASS = patch green and both mutants red on exactly the layer whose line was removed."""
import json
L = lambda n: json.load(open(f"F:/TanitAD-scratch/v8proof/leadcond/checks/c3_reach_{n}.json", encoding="utf-8"))
p, o, t = L("patch"), L("mutO"), L("mutT")
res = {"patch": p, "mutO_operative_line_removed": o, "mutT_tactical_line_removed": t}
res["PASS"] = bool(p["PASS"] and not o["reach_ok"] and o["a_reach"]["operative_traj60_max_abs_change_code_fixed"] == 0.0 and o["a_reach"]["tactical_logits_max_abs_change"] > 0.0
                   and not t["reach_ok"] and t["a_reach"]["tactical_logits_max_abs_change"] == 0.0 and t["a_reach"]["operative_traj60_max_abs_change_code_fixed"] > 0.0)
json.dump(res, open("F:/TanitAD-scratch/v8proof/leadcond/checks/c3_reach.json", "w", encoding="utf-8", newline="\n"), indent=1)
print("C3", res["PASS"])
