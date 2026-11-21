"""perc8_map_read.py -- 10 cm map IoU per class x distance band, arm A (refcv8) vs arm B (refcv7 banked), SAME windows.
Uses the programme's own `taniteval.map_hires_metrics` (summarize = pooled IoU/P/R/F1 + clip-cluster bootstrap; paired_delta = the
paired episode-cluster bootstrap).  CPU only.

    python perc8_map_read.py --a-table <perc_map_table.npz> --b-table <banked refcv7 perc_map_table.npz> --b-arm refcv7 --out <json> [--n-boot 1000]

Estimator: episode-cluster bootstrap over clips; the CI answers "would another draw of EPISODES say this?" (ONE training seed per arm).
"""
import argparse, json, sys, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-table", required=True)
    ap.add_argument("--a-arm", default="refcv7")           # the arm label inside the table (perc_pass_r8 keeps the r7 name for refcv8 weights)
    ap.add_argument("--b-table", required=True)
    ap.add_argument("--b-arm", default="refcv7")
    ap.add_argument("--a-name", default="refcv8_ema48852")
    ap.add_argument("--b-name", default="refcv7_50400")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-paired", action="store_true")
    a = ap.parse_args()
    from taniteval import map_hires_metrics as MH
    t0 = time.time()
    A = MH.WindowTable.load(a.a_table)
    B = MH.WindowTable.load(a.b_table)
    eq = list(A.eid) == list(B.eid)
    print(f"[map] A windows {A.n_windows} clips {A.n_clips}; B windows {B.n_windows} clips {B.n_clips}; eid sequences equal: {eq}", flush=True)
    if not eq:
        raise SystemExit("[map] the two tables are not on the same window sequence -- refusing to pair")
    T = MH.WindowTable(arms=(a.a_name, a.b_name), eid=list(A.eid), n_rows=A.n_rows)
    T.stats[a.a_name] = A.stats[a.a_arm]
    T.stats[a.b_name] = B.stats[a.b_arm]
    out = {"tool": "perc8_map_read.py", "arms": {"a": a.a_name, "b": a.b_name}, "n_windows": T.n_windows, "n_clips": T.n_clips,
           "bands": list(T.band_names), "tol_m": MH.TOL_M,
           "estimator": f"episode-cluster bootstrap over clips (taniteval.ci), n_boot {a.n_boot}, seed 0; paired = MH.paired_delta (the same draws for both arms)",
           "what_the_ci_answers": "would another draw of EPISODES say this? ONE training seed per arm: nothing about training-run or inference variance",
           "decision_rule_note": "table built by perc_pass with the run's declared map decision rule (prior_corrected, the run's frozen class weights)"}
    out["summary"] = MH.summarize(T, n_boot=a.n_boot, seed=0)
    print(f"[map] summarize {time.time() - t0:.0f}s", flush=True)
    if not a.no_paired:
        names = T.band_names
        paired = {}
        for k in range(MH.N_FINE_CLASSES):
            for b in range(len(names)):
                metrics = ("iou", "F1") if k in MH.THIN_CLASSES else ("iou",)
                for m in metrics:
                    r = MH.paired_delta(T, a.a_name, a.b_name, m, k, b, n_boot=a.n_boot, seed=0)
                    paired[f"{MH.CLASS_NAMES[k]}|{names[b]}|{m}"] = r
        out["paired_a_minus_b"] = paired
        print(f"[map] paired {time.time() - t0:.0f}s", flush=True)
    out["wall_s"] = round(time.time() - t0, 1)
    open(a.out, "w", encoding="utf-8").write(MH.to_json(out))
    print("[map] DONE", a.out, flush=True)


if __name__ == "__main__":
    sys.exit(main())
