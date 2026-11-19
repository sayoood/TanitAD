# check_k8_n2.py : the train-log part of control k8 of card PREREG_V8F_PROOF_ADD46 in the form of its note N2. Pure Python (json only); runs on the pod.
# Inputs: the reader run folder (<run-dir>/<arm>/train_log.jsonl of the two reader arms) and DIAG_TAIL3.json (diag_tail3.py: the exact count of gap-labelled
# frames for every training step of the flagged log intervals and of five known-value intervals).
# Parts (all must hold for PASS):
#   a  the lead term of the last 1,000 steps below that of the first 1,000 steps, and the log complete at 19,164 steps (both arms; unchanged from the tool check)
#   b  gap-labelled frames above 0 at every log record (the wording of card ADD27)
#   c  the replay is the trained batch order (hash of the first 200 batches) and its exact count reproduces the logged interval minimum on the known-value intervals
#   d  the log intervals with a zero minimum are those of the replay, each holds a step with an exact count of 0, and every step with an exact count of 0 is a
#      batch with fewer rows than the batch size and at most MAX_ROWS rows
#   e  the steps with no gap-labelled frame are at most MAX_SHARE of the training steps
# The tool check on every training step (the interval minimum above 0) is computed and written beside; it is reported, it does not enter PASS (note N2).
# Usage: python check_k8_n2.py --run-dir <lr16> --diag <DIAG_TAIL3.json> --out <K8_N2.json>      |      python check_k8_n2.py --selftest
import argparse
import json
import math
import os
import sys
import tempfile

TOTAL, WINDOW, ARMS, MAX_ROWS, MAX_SHARE, N_KNOWN = 19164, 1000, ("TK16_s0", "TK16_s1"), 8, 0.01, 5


def read_arm(run_dir, arm, total):
    recs = [json.loads(ln) for ln in open(os.path.join(run_dir, arm, "train_log.jsonl"), encoding="utf-8") if ln.strip()]
    hdr, recs = recs[0], recs[1:]
    body = [r for r in recs if not r["first"]]
    first = [r["lead_sup_mean"] for r in body if r["step"] <= WINDOW]
    last = [r["lead_sup_mean"] for r in body if r["step"] > total - WINDOW]
    fm = math.fsum(first) / len(first) if first else None
    lm = math.fsum(last) / len(last) if last else None
    flagged = sorted(int(r["step"]) for r in recs if r["n_gap_min_interval"] == 0)
    return {"first_1000_mean": fm, "last_1000_mean": lm, "n_records": len(recs),
            "a_lead_term_falls": bool(fm is not None and lm is not None and lm < fm),
            "a_complete": bool(recs and recs[-1]["step"] == total and int(hdr["total"]) == total),
            "min_n_gap_at_log_records": min(r["n_gap"] for r in recs),
            "b_gap_labels_at_every_log_record": bool(min(r["n_gap"] for r in recs) > 0),
            "tool_check_gap_labels_at_every_training_step": bool(min(min(r["n_gap"], r["n_gap_min_interval"]) for r in recs) > 0),
            "flagged_intervals": flagged}


def check(run_dir, diag, total=TOTAL, arms=ARMS):
    d = json.load(open(diag, encoding="utf-8"))
    A = {arm: read_arm(run_dir, arm, total) for arm in arms}
    ctl = d.get("control_intervals") or []
    parts = {
        "a": all(A[x]["a_lead_term_falls"] and A[x]["a_complete"] for x in arms),
        "b": all(A[x]["b_gap_labels_at_every_log_record"] for x in arms),
        "c": bool(d.get("same_order_first_200_batches") is True and d.get("control_intervals_all_equal") is True and len(ctl) >= N_KNOWN
                  and all(c["logged_interval_min"] > 0 and int(c["logged_interval_min"]) == int(c["exact_interval_min"]) for c in ctl) and int(d.get("total_steps", -1)) == total),
        "d": bool(all(A[x]["flagged_intervals"] == d.get("flagged_log_intervals") for x in arms) and d.get("every_flagged_interval_has_a_zero_step") is True
                  and d.get("n_steps_counted_exactly", 0) >= d.get("n_steps_in_flagged_intervals", 1)
                  and (d.get("n_zero_steps") == 0 or (d.get("zero_all_short") is True and d.get("zero_max_rows") is not None and int(d["zero_max_rows"]) <= MAX_ROWS))),
        "e": bool(d.get("n_zero_steps") is not None and int(d["n_zero_steps"]) / total <= MAX_SHARE),
    }
    return {"what": "train-log part of control k8 of card PREREG_V8F_PROOF_ADD46 in the form of note N2", "total_steps": total, "max_rows": MAX_ROWS, "max_share": MAX_SHARE,
            "arms": A, "parts": parts, "n_zero_steps": d.get("n_zero_steps"), "zero_rows": d.get("zero_rows"), "share_of_steps_zero": (int(d["n_zero_steps"]) / total) if d.get("n_zero_steps") is not None else None,
            "share_of_window_rows_in_zero_steps": d.get("share_of_window_rows_in_zero_steps"),
            "tool_check_every_training_step_all_arms": all(A[x]["tool_check_gap_labels_at_every_training_step"] for x in arms),
            "PASS": bool(all(parts.values()))}


def _write_case(root, total, flagged, zero_rec=False, rising=False):
    for arm in ARMS:
        os.makedirs(os.path.join(root, arm))
        with open(os.path.join(root, arm, "train_log.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"arm": arm, "total": total}) + "\n")
            steps = [1] + list(range(50, total + 1, 50)) + ([total] if total % 50 else [])
            for s in steps:
                m = (0.4 - 0.2 * s / total) if not rising else (0.2 + 0.2 * s / total)
                fh.write(json.dumps({"step": s, "first": s == 1, "lead_sup_mean": m, "n_gap": 0.0 if (zero_rec and s == 100) else 150.0,
                                     "n_gap_min_interval": 0.0 if s in flagged else 40.0}) + "\n")


def _diag(total, flagged, **kw):
    d = {"same_order_first_200_batches": True, "total_steps": total, "flagged_log_intervals": flagged, "n_steps_counted_exactly": 50 * (len(flagged) + 5),
         "n_steps_in_flagged_intervals": 50 * len(flagged), "n_zero_steps": len(flagged), "zero_rows": [1] * len(flagged), "zero_all_short": True, "zero_max_rows": 1,
         "every_flagged_interval_has_a_zero_step": True, "share_of_window_rows_in_zero_steps": 1e-5,
         "control_intervals": [{"step": 50 * (i + 30), "logged_interval_min": 40.0, "exact_interval_min": 40} for i in range(5)], "control_intervals_all_equal": True}
    d.update(kw)
    return d


def selftest():
    total, flagged = 3000, [150, 800]
    cases = [("good", {}, {}, True), ("order differs", {}, {"same_order_first_200_batches": False}, False), ("known-value interval differs", {}, {"control_intervals_all_equal": False}, False),
             ("a zero step with 20 rows", {}, {"zero_max_rows": 20}, False), ("a zero step in a full batch", {}, {"zero_all_short": False}, False),
             ("a flagged interval without a zero step", {}, {"every_flagged_interval_has_a_zero_step": False}, False), ("flagged intervals differ", {}, {"flagged_log_intervals": [150]}, False),
             ("too many zero steps", {}, {"n_zero_steps": 40}, False), ("a log record without gap labels", {"zero_rec": True}, {}, False), ("lead term rises", {"rising": True}, {}, False),
             ("flagged steps not all counted", {}, {"n_steps_counted_exactly": 60}, False)]
    bad = 0
    for name, kw, dk, want in cases:
        with tempfile.TemporaryDirectory() as td:
            _write_case(td, total, flagged, **kw)
            dp = os.path.join(td, "diag.json")
            json.dump(_diag(total, flagged, **dk), open(dp, "w", encoding="utf-8"))
            got = check(td, dp, total=total)["PASS"]
        ok = got is want
        bad += 0 if ok else 1
        print("selftest %-42s want %-5s got %-5s %s" % (name, want, got, "ok" if ok else "MISMATCH"))
    print("ZZ SELFTEST %s (%d cases)" % ("PASS" if bad == 0 else "FAIL", len(cases)))
    return 0 if bad == 0 else 1


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = check(a.run_dir, a.diag)
    json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("K8-N2 %s parts %s zero steps %s tool check on every training step %s" % ("PASS" if res["PASS"] else "FAIL", res["parts"], res["n_zero_steps"], res["tool_check_every_training_step_all_arms"]))
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
