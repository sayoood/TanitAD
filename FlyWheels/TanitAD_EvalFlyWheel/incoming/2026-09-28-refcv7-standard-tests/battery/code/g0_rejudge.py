"""Re-judge an EXISTING G0 artifact (zero GPU): the verdict as registered AND under SPEC AMENDMENT A2.

    python g0_rejudge.py <g0.json> <metrics.jsonl copy> [--out <g0.json>]

The reproduction's per-seed rows, mutations and wrapper probe are read from the artifact; the in-run
row is re-read from the metrics copy (the same one G0 used, matched by step). Writes
`verdict_as_registered` + `verdict` (A2) back (or to --out), keeping every other field.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g0_refcv7 as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("g0_json")
    ap.add_argument("metrics")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rec = json.load(open(a.g0_json, encoding="utf-8"))
    rows = [json.loads(ln) for ln in open(a.metrics, encoding="utf-8") if ln.strip()]
    ev = [r for r in rows if r.get("step") == rec["step"] and "eval_loss" in r]
    if len(ev) != 1:
        raise SystemExit(f"{len(ev)} eval rows at step {rec['step']}")
    by_seed = {int(k): v for k, v in rec["by_seed"].items()}
    old = rec.get("verdict")
    rec["verdict_as_registered"] = G.judge(ev[0], by_seed, rec)
    rec["verdict"] = G.judge(ev[0], by_seed, rec, amend="A2")
    rec["rejudged"] = {"tool": "g0_rejudge.py", "previous_G0": (old or {}).get("G0"),
                       "note": "zero-GPU re-judge of the banked reproduction (SPEC AMENDMENT A2)"}
    out = a.out or a.g0_json
    json.dump(rec, open(out, "w", encoding="utf-8"), indent=1, default=str)
    v, r = rec["verdict"], rec["verdict_as_registered"]
    print(json.dumps({"G0_as_registered": r["G0"], "reasons_as_registered": r["reasons"][:6],
                      "G0_A2": v["G0"], "reasons_A2": v["reasons"][:10],
                      "by_class_counts_A2": v["by_class_counts"], "medians_A2": v["medians"],
                      "mutation_detection_A2": {m: {k: d.get(k) for k in ("detected", "n_terms_out")}
                                                for m, d in v["mutation_detection"].items()}},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
