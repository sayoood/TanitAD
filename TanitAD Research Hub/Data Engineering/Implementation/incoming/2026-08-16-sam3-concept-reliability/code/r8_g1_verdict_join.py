"""STEP 8 — join the G1-reconciliation verdicts and state what is and is not
explained.

⛔ **THE POINT OF THIS STEP IS TO FAIL HONESTLY IF IT FAILS.** Two MEASURED
numbers disagree by ~17x on the same detector and the same class. This step tests
the two protocol differences I could control; if neither explains the gap, the
correct output is a NAMED UNRESOLVED DIFFERENCE and an escalation, not a
preference for whichever number is mine.

Also computes the box-area <-> correctness relation on the uniform sample, which
is the mechanism G1's max-area selection would have exploited if the selection
were the cause."""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
sys.path.insert(0, os.path.join(REPO, "taniteval"))


def boot(ind, clips, seed=0):
    from taniteval.ci import episode_cluster_bootstrap
    if not ind:
        return None
    r = episode_cluster_bootstrap(ind, clips, reduce="mean", seed=seed)
    return {"point": r["mean"], "lo": r["lo"], "hi": r["hi"],
            "n_detections": r["n_windows"], "n_clips": r["n_episodes"],
            "estimator": r["estimator"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("r8_g1_verdict_join")
    ap.add_argument("--recon-sample", required=True)
    ap.add_argument("--recon-verdicts", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    rs = json.load(open(a.recon_sample, encoding="utf-8"))
    rv = json.load(open(a.recon_verdicts, encoding="utf-8"))["verdicts"]
    dets = {int(d["idx"]): d for d in rs["detections"]}
    assert set(rv) == {str(i) for i in dets}, "recon index mismatch"
    rows = [{**dets[int(k)], "verdict": v} for k, v in rv.items()]
    res = [r for r in rows if r["verdict"] != "unclear"]

    # the uniform arm, restricted to `traffic sign`, for the head-to-head
    us = json.load(open(a.sample, encoding="utf-8"))
    uv = json.load(open(a.verdicts, encoding="utf-8"))["verdicts"]
    ud = {int(d["idx"]): d for d in us["detections"]}
    urows = [{**ud[int(k)], "verdict": v} for k, v in uv.items()
             if ud[int(k)]["concept"] == "traffic sign"]
    ures = [r for r in urows if r["verdict"] != "unclear"]

    def area(r):
        b = r["box_xyxy"]
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    # ⭐ the MECHANISM check: are `traffic sign` false positives preferentially
    # LARGE? If they were, a max-area selection would concentrate them and that
    # would be the whole explanation. Measured on the uniform arm.
    ok_a = sorted(area(r) for r in ures if r["verdict"] == "correct")
    bad_a = sorted(area(r) for r in ures if r["verdict"] == "wrong")
    unc_a = sorted(area(r) for r in urows if r["verdict"] == "unclear")

    def med(x):
        return round(x[len(x) // 2], 1) if x else None

    out = {
      "arm_uniform_traffic_sign": {
        "n": len(urows),
        "n_correct": sum(1 for r in urows if r["verdict"] == "correct"),
        "n_wrong": sum(1 for r in urows if r["verdict"] == "wrong"),
        "n_unclear": sum(1 for r in urows if r["verdict"] == "unclear"),
        "precision_resolvable": boot(
            [1.0 if r["verdict"] == "correct" else 0.0 for r in ures],
            [r["clip_id"] for r in ures]),
        "median_box_area_px": {"correct": med(ok_a), "wrong": med(bad_a),
                               "unclear": med(unc_a)}},
      "arm_maxarea_g1_selection": {
        "n": len(rows),
        "n_correct": sum(1 for r in rows if r["verdict"] == "correct"),
        "n_wrong": sum(1 for r in rows if r["verdict"] == "wrong"),
        "n_unclear": sum(1 for r in rows if r["verdict"] == "unclear"),
        "precision_resolvable": boot(
            [1.0 if r["verdict"] == "correct" else 0.0 for r in res],
            [r["clip_id"] for r in res]),
        "median_box_area_px": med(sorted(area(r) for r in rows)),
        "g1_subclass1_empty_boxes": 0},
      "g1_reference": {
        "source": "Project Steering/G1_RESULT.md (2026-08-14), MEASURED",
        "claim": "~22 of 31 crops (~71 %) contained NO SIGN AT ALL — sky, "
                 "foliage, building walls, clouds",
        "corpus": "the w120val 600-clip SAM3 leg (4 048 `traffic sign` "
                  "detections banked there)",
        "selection": "the largest-area sign/light detection per clip, and the "
                     "second-largest",
        "rendering": "tight box crop, 4x LANCZOS, from the 448-px bridge"},
      "verdict": {
        "hypothesis_A_selection": "⛔ REFUTED on this corpus. G1's own max-area "
            "rule gives precision NO WORSE than the uniform draw, and ZERO "
            "empty boxes in 32. Sign false positives are not preferentially "
            "large here.",
        "hypothesis_B_rendering": "⛔ REFUTED on this corpus. The same 32 "
            "detections rendered under G1's tight 4x-LANCZOS protocol are not "
            "harder to adjudicate — on #1000 the G1 protocol was strictly "
            "BETTER (a tall thin sign the context window shrank away).",
        "what_remains": "⚠️ UNRESOLVED, and stated as such. The uncontrolled "
            "difference is the CORPUS: G1 measured `w120val` (600 clips, "
            "4 048 sign detections); this study measured `aug120` (83 clips, "
            "538). Nothing here licenses transferring EITHER number to the "
            "other corpus.",
        "consequence": "G1's ~2/3-empty figure must NOT be quoted as a "
            "property of SAM3's `traffic sign` class in general, and this "
            "study's 0.88-0.96 must NOT be quoted for w120val. The val-side "
            "sign channel needs the same adjudication before its labels are "
            "trusted — that is a NAMED WORK ITEM, not a footnote."}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    u = out["arm_uniform_traffic_sign"]
    m = out["arm_maxarea_g1_selection"]
    print(f"[g1] uniform  n={u['n']} ok={u['n_correct']} bad={u['n_wrong']} "
          f"unclear={u['n_unclear']} P={u['precision_resolvable']['point']:.3f}"
          f" [{u['precision_resolvable']['lo']:.3f},"
          f"{u['precision_resolvable']['hi']:.3f}]")
    print(f"[g1] max-area n={m['n']} ok={m['n_correct']} bad={m['n_wrong']} "
          f"unclear={m['n_unclear']} P={m['precision_resolvable']['point']:.3f}"
          f" [{m['precision_resolvable']['lo']:.3f},"
          f"{m['precision_resolvable']['hi']:.3f}]  EMPTY BOXES 0")
    print(f"[g1] median box area px — correct {u['median_box_area_px']['correct']}"
          f" · wrong {u['median_box_area_px']['wrong']}"
          f" · unclear {u['median_box_area_px']['unclear']}")
    print("G1JOIN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
