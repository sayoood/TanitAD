"""lh_verdict_void.py -- PREREG_R9PF_LH s2 precondition: R9-PF's own K-EQ / K-OFF must have PASSED in its final verdict, or the LH arm is VOID.
Reads result/R9PF_PA_VERDICT.json + R9PF_KEQ.json; writes result/R9PF_LH_VERDICT.{json,md}.  No LH arm is composed when VOID.  NEW (EvalFlyWheel 2026-10-26)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r9pf_common as RC
LH = RC.WORK / "PREREG_R9PF_LH.md"; LH_SHA = "6de4e4bb0e8abfe0ada03e3e0e401faea9996857ed80744547b923c4f9f2bd59"
got = RC.sha256_file(LH)
if len(got) != 64 or got != LH_SHA:
    raise SystemExit(f"[lh] PREREG_R9PF_LH seal mismatch / INCONCLUSIVE {got}")
v = json.loads((RC.RESULT / "R9PF_PA_VERDICT.json").read_text(encoding="utf-8"))
k = json.loads((RC.RESULT / "R9PF_KEQ.json").read_text(encoding="utf-8"))
keq, koff = bool(k["K-EQ"]["PASS"]), bool(k["K-OFF"]["PASS"])
if keq and koff:
    raise SystemExit("[lh] precondition HOLDS -- this script writes only the VOID record; run the LH arms instead")
TIER = "T1 self-action OPEN LOOP on logged eval139 frames, not driving performance"
EST = "paired episode-cluster bootstrap over EPISODES, B 2000 seed 0 -- blind to training and inference variance; fit seeds 0/1/2 = training-variance read"
reason = (f"PREREG_R9PF_LH s2: 'R9-PF's own K-EQ / K-OFF must have PASSED in its final verdict, or this arm is VOID.' R9PF_PA_VERDICT = {v['verdict']}; "
          f"K-OFF {'PASS' if koff else 'FAIL'}, K-EQ {'PASS' if keq else 'FAIL'} (seed 0: {k['K-EQ']['per_seed']['s0']['n_windows_over_tol']} / 515 windows "
          f"over 2 ulp, seed 1: {k['K-EQ']['per_seed']['s1']['n_windows_over_tol']} / 515; decision agreement 100 %).")
rec = {"verdict": "VOID", "prereg": {"file": str(LH), "sha256": got}, "tier": TIER, "estimator": EST, "reason": reason,
       "arms_composed": [], "note": "No R9LH-ARG / R9LH-SOFT / R9DER-ARG composition was run: the precondition is checked first, and a VOID arm is not scored.",
       "unblock": "R9-PF K-EQ resolved (CPU roll discriminating CUDA-vs-CPU head arithmetic + a Master-Mind ruling on the tolerance), then rerun this prereg.",
       "r9pf_verdict": {"file": str(RC.RESULT / "R9PF_PA_VERDICT.json"), "sha256": RC.sha256_file(RC.RESULT / "R9PF_PA_VERDICT.json")},
       "keq_record": {"file": str(RC.RESULT / "R9PF_KEQ.json"), "sha256": RC.sha256_file(RC.RESULT / "R9PF_KEQ.json")}, "generated": time.strftime("%FT%T%z")}
RC.jdump(rec, RC.RESULT / "R9PF_LH_VERDICT.json")
(RC.RESULT / "R9PF_LH_VERDICT.md").write_text(
    f"# R9PF_LH_VERDICT -- VOID\n\n**tier:** {TIER}  \n**estimator:** {EST}  \n**prereg:** PREREG_R9PF_LH.md sha256 {got}\n\n"
    f"**committed outcome:** VOID -- {reason}\n\n{rec['note']}\n\n**Unblock:** {rec['unblock']}\n", encoding="utf-8")
print("[lh] VOID  ZZLHVOIDZZ")
