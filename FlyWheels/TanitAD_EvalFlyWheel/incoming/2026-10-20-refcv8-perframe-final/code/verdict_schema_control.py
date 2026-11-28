"""verdict_schema_control.py -- KNOWN-VALUE control of ship_r8_verdict_pf.py on REAL artifact schemas (EvalFlyWheel 2026-10-20).

Runs the Amendment-3 verdict, orientation-only, on the per-clip tree step40000ema_unc using the files pf_lane.py re-produced in the 40k lane control
(clip-fed table installed as the 'registered' one).  Expected, as LITERALS: clause (1) FAIL (the registered table is the clip-fed one -- Amendment 3 (ii)
refuses it), clause (1b) INCOMPLETE (no per-frame parity exists for a clip-fed tree), overall INCOMPLETE; and clauses (2)-(7) EXACTLY the statuses of the
banked per-clip verdict SHIP_R8_step40000ema_unc.json (the same instruments' files, read by the byte-copied base clauses).
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ship_r8_verdict_pf as V

R8 = Path("C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery")
LC = Path("C:/Users/Admin/refcv8_pf/lanectl40k")
T = "step40000ema_unc"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else LC / "verdict_control"
v = V.run(T, LC / "pkg", LC / "REGISTERED_CONTROL.sha256", R8 / "raw" / "retime_trainfit" / "FROZEN_GAINS_RETIME_train.json", LC / "no_parity",
          LC / "no_gate.json", out, perclip_file=R8 / "raw" / "ship_r8" / f"SHIP_R8_{T}.json")
banked = json.load(open(R8 / "raw" / "ship_r8" / f"SHIP_R8_{T}.json", encoding="utf-8"))
b = {str(c["id"]): c["status"] for c in banked["clauses"]}
got = v["per_clause"]
rows = {k: {"pf_verdict": got.get(k), "banked_per_clip": b.get(k)} for k in ("2", "3", "4", "5", "6", "7")}
ok = (got["1"] == "FAIL" and got["1b"] == "INCOMPLETE" and v["overall"] == "INCOMPLETE" and all(r["pf_verdict"] == r["banked_per_clip"] for r in rows.values()))
rec = {"tool": "verdict_schema_control.py", "tree": T, "status": "PASS" if ok else "FAIL", "clause1": got["1"], "clause1b": got["1b"], "overall": v["overall"],
       "clauses_2_7_vs_banked": rows, "banked_overall": banked.get("overall"), "evidence_class": "MEASURED (ours): real lane-control artifacts vs the banked per-clip verdict"}
(out / "VERDICT_SCHEMA_CONTROL.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
print(json.dumps(rec), f"ZZVSC_{rec['status']}ZZ")
