"""Print the P-BOX result tables (markdown) from raw/pbox_result.json: per arm the registered metrics, the paired
deltas vs PB0 with CIs and the replicate floor, the DESCRIPTIVE deltas vs refcv7's step-0 decoder, the bars and the
must-fails. Run: python pbox_table.py [path/to/pbox_result.json]"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")       # the dev box console is cp1252
p = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "raw" / "pbox_result.json")
r = json.loads(p.read_text(encoding="utf-8"))
A, PR, FL = r["arms"], r["paired_vs_PB0"], r["replicate_floor"] or {}
R2 = (r.get("paired_vs_ref2_DESCRIPTIVE") or {}).get("pairs") or {}
ORDER = ["STEP0", "PB0", "PB0r", "PB1", "PB2", "PB3", "PB4", "PB4h", "PB6", "PB7", "PB8",
         "R-pres0z", "R-pres0", "R-memshuf", "R-qshuf"]
arms = [a for a in ORDER if a in A] + [a for a in A if a not in ORDER]


def f(x, n=4):
    return "—" if x is None else f"{x:.{n}f}"


def d(v):
    if not v:
        return "—"
    lo, hi = v["ci95"]
    return f"{v['point']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


print(f"estimator: {r['estimator']}; EVAL {r['n_eval_windows']} windows / {r['n_eval_episodes']} episodes\n")
print("| arm | AP@2 m NMS | AP@2 m raw | AP@1 m raw | AP@0.5 m raw | AP@4 m raw | NMS 0–20 / 20–40 / 40–60 | "
      "boxes / obj | conf_ratio | ECE | car / person AP@2 m |")
print("|---|" + "---|" * 10)
for a in arms:
    x = A[a]
    print(f"| {a} | {f(x['ap2_nms2_all'])} | {f(x['ap2_raw_all'])} | {f(x['ap1_raw_all'])} | {f(x['ap0.5_raw_all'])} | "
          f"{f(x['ap4_raw_all'])} | {f(x['ap2_nms2_0_20'], 3)} / {f(x['ap2_nms2_20_40'], 3)} / {f(x['ap2_nms2_40_60'], 3)} | "
          f"{f(x['boxes_per_detected_object'], 2)} | {f(x['conf_ratio'], 2)} | {f(x['ece_all'], 3)} | "
          f"{f(x['ap2_cls_automobile_all'], 3)} / {f(x['ap2_cls_person_all'], 3)} |")
print("\nreplicate floor |PB0r − PB0|:", {k: round(v, 4) for k, v in FL.items()})
print("\n| arm − PB0 (paired) | AP@2 m NMS | AP@1 m raw | AP@2 m raw | NMS 0–20 | NMS 20–40 | NMS 40–60 | boxes / obj |")
print("|---|" + "---|" * 7)
K = ["ap2_nms2_all", "ap1_raw_all", "ap2_raw_all", "ap2_nms2_0_20", "ap2_nms2_20_40", "ap2_nms2_40_60",
     "boxes_per_object"]
for a in arms:
    if a in PR:
        print(f"| {a} | " + " | ".join(d(PR[a].get(k)) for k in K) + " |")
if R2:
    print("\nDESCRIPTIVE (not registered): arm − refcv7 step 0")
    print("| arm − STEP0 | " + " | ".join(K) + " |")
    print("|---|" + "---|" * len(K))
    for a in arms:
        if a in R2:
            print(f"| {a} | " + " | ".join(d(R2[a].get(k)) for k in K) + " |")
print("\nverdict:", json.dumps(r["verdict"], indent=1))
print("must_fail:", json.dumps(r["must_fail"], indent=1, default=float))
print("VOID:", r["VOID"], r["void_reasons"])
print("C-step:", {k: (round(v, 4) if v else v) for k, v in r.get("C_step_rel_param_change", {}).items()})
for a in arms:
    lo = A[a].get("loc", {})
    print(a, "TP centre |dlon| / |dlat| median by band:",
          {b: (v["n"], f(v["med_dlon"], 2), f(v["med_dlat"], 2)) for b, v in lo.items()})
