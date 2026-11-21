"""Bank the A16 arm's PARTIAL record from its launch log (the harness writes its record only at
the end of all arms; the Master Mind stopped the run after MAIN's FAIL -- SPEC_REFCV7 §22.1 --
so no harness record exists). Every number here is PARSED from the harness's own per-100-step
log lines (declared rule only: the raw rule, CE ratios, C1-C3 and the verdict function run only
at the end and never ran). Must-fail readings are evaluated here on the logged step-1,000
declared IoU against the spec's bars -- a DERIVED reading, labelled so, not the harness verdict.
Usage: a16_partial_record.py <a16_launch.log> <A16_DONE json> <spec json> <out json>"""
import hashlib
import json
import sys
from pathlib import Path

log_p, done_p, spec_p, out_p = map(Path, sys.argv[1:5])
spec = json.loads(spec_p.read_text(encoding="utf-8"))
bars = spec["thresholds"]["iou"]
arms, diag, header = {}, None, []
for line in log_p.read_text(encoding="utf-8").splitlines():
    s = line.strip()
    if s.startswith("{"):
        j = json.loads(s)
        if "arm" in j and "step" in j:
            arms.setdefault(j["arm"], []).append({"step": j["step"], "train_loss": j["train_loss"],
                                                   "iou": j["iou"]})
        elif j.get("diag") == "MAIN_final":
            diag = {k: v for k, v in j.items() if k != "diag"}
    else:
        header.append(s)
out = {"schema": "tanitad.g_map_overfit_record/1+PARTIAL",
       "binding": False, "partial": True,
       "stopped": {"by": "the Master Mind (SPEC_REFCV7 §22.1: A16's remaining must-fail arms "
                         "cannot change MAIN's FAIL)", "signal": "SIGTERM to the arm's python by "
                         "explicit PID 3801215", "done_marker": json.loads(done_p.read_text())},
       "spec": spec_p.name, "spec_md5": hashlib.md5(spec_p.read_bytes()).hexdigest(),
       "launch_log_md5": hashlib.md5(log_p.read_bytes()).hexdigest(),
       "launch_log_header": [h for h in header if h][:8],
       "evidence_class": "MEASURED (parsed from the harness's own log lines; declared rule only)",
       "arms": {}, "MAIN_final_diag": diag}
for arm, curve in arms.items():
    last = curve[-1]
    rec = {"curve": curve, "last_step": last["step"], "complete": last["step"] == int(spec["steps"])}
    if rec["complete"]:
        iou = last["iou"]
        rec["final_declared_iou"] = iou
        rec["bars_pass_declared"] = {k: (iou[k] is not None and iou[k] >= bars[k]) for k in bars}
        mf = (spec.get("must_fail") or {}).get(arm)
        if mf:
            failed = [c for c in mf if not rec["bars_pass_declared"][c]]
            rule_all = bool((spec.get("must_fail_all") or {}).get(arm))
            rec["must_fail_DERIVED"] = {
                "must_fail": mf, "failed": failed, "rule": "all" if rule_all else "any",
                "failed_as_required": (len(failed) == len(mf)) if rule_all else bool(failed)}
    out["arms"][arm] = rec
m = out["arms"]["healthy"]
out["MAIN_DERIVED"] = {"fails_declared": [k for k, ok in m["bars_pass_declared"].items() if not ok],
                       "verdict_on_bars": "FAIL" if not all(m["bars_pass_declared"].values()) else "PASS"}
out_p.write_bytes((json.dumps(out, indent=1) + "\n").encode("utf-8"))
print(json.dumps({"arms": {a: (r["last_step"], r["complete"]) for a, r in out["arms"].items()},
                  "MAIN": out["MAIN_DERIVED"],
                  "must_fail": {a: r.get("must_fail_DERIVED", {}).get("failed_as_required")
                                for a, r in out["arms"].items() if "must_fail_DERIVED" in r}}))
print(out_p.name, hashlib.md5(out_p.read_bytes()).hexdigest())
