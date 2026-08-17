"""Measure O2's UNIQUE content from banked trainer logs. ZERO GPU.

THE POINT (2026-08-17-O234-DESIGN-RESEARCH.md 2.1 / 2.1a). `cells()` is a pure
reshape, so O2 and O5 score the identical tensor and the algebra closes exactly:

    O2 = (O5's step-j term) + Cov_c(w, err)          w mean-1 over cells

and `o2_near_field_loss` HAPPENS TO LOG BOTH HALVES (train_v6_staged.py:640-643):

    o2_loss        = (w * err).mean()      <- O2
    o2_unweighted  =      err.mean()       <- O5's step-j term, exactly

=> Cov = o2_loss - o2_unweighted, and O2's entire distinct content is that
difference. No forward pass, no checkpoint, no GPU -- just arithmetic on log rows.

WARNING. The banked rows this finds are DRY-LADDER steps 1-2, i.e. at
initialisation, where the per-cell error profile is near-uniform and Cov is small
almost by construction. Pointing this at the LIVE 30k log is the measurement that
settles it (experiment E-O2-A). Usage:

    python o2_cov_from_logs.py <file-or-dir> [...]     # defaults to the banked dry ladder
"""
import json
import pathlib
import re
import statistics
import sys

# .../Architecture & Inference/Research/raw/2026-08-17-O234/this.py -> parents[3]
DEFAULT = (pathlib.Path(__file__).resolve().parents[3] / "Implementation" / "incoming"
           / "2026-08-16-v6-stage-chain" / "raw")

# log rows are embedded as JSON objects, sometimes inside a JSON string (escaped)
ROW = re.compile(r'\{[^{}]*"o2_unweighted"[^{}]*\}')


def rows_from(path: pathlib.Path):
    txt = path.read_text(encoding="utf-8", errors="replace").replace('\\"', '"')
    for m in ROW.finditer(txt):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "o2_loss" in d and "o2_unweighted" in d and d["o2_unweighted"]:
            yield d


def main(argv):
    targets = [pathlib.Path(a) for a in argv[1:]] or [DEFAULT]
    files = []
    for t in targets:
        files.extend(sorted(t.rglob("*")) if t.is_dir() else [t])
    files = [f for f in files if f.is_file() and f.suffix in (".json", ".log", ".jsonl")]

    seen, out = set(), []
    for f in files:
        for d in rows_from(f):
            key = (d["o2_loss"], d["o2_unweighted"])
            if key in seen:
                continue           # the chain writes the same row to .json and .log
            seen.add(key)
            cov = d["o2_loss"] - d["o2_unweighted"]
            out.append({"file": f.name, "stage": d.get("stage"), "step": d.get("step"),
                        "o2_loss": d["o2_loss"], "o2_unweighted": d["o2_unweighted"],
                        "cov": cov, "rel_pct": abs(cov) / d["o2_unweighted"] * 100.0,
                        "o5_loss": d.get("o5_loss"), "o5_k": d.get("o5_k"),
                        "o2_at_step": d.get("o2_at_step")})

    if not out:
        print("no rows with both o2_loss and o2_unweighted found")
        return 1

    print(f"{'stage':6} {'step':>4} {'o2_loss':>9} {'o2_unwt(=O5 step-j)':>20} "
          f"{'Cov':>9} {'|Cov|/unwt':>11}")
    for r in out:
        print(f"{str(r['stage']):6} {str(r['step']):>4} {r['o2_loss']:9.4f} "
              f"{r['o2_unweighted']:20.4f} {r['cov']:+9.4f} {r['rel_pct']:10.2f}%")

    rel = [r["rel_pct"] for r in out]
    signs = "".join("+" if r["cov"] > 0 else "-" for r in out)
    print(f"\nn={len(rel)}  |Cov|/unweighted: min {min(rel):.2f}%  "
          f"median {statistics.median(rel):.2f}%  max {max(rel):.2f}%")
    print(f"sign of Cov: {signs}   ({signs.count('+')} positive, "
          f"{signs.count('-')} negative)")
    print("\n=> O2's UNIQUE content is that percentage. The rest is O5's step-j term.")

    p = pathlib.Path(__file__).with_suffix(".json")
    p.write_text(json.dumps({"rows": out, "n": len(rel), "rel_pct_min": min(rel),
                             "rel_pct_median": statistics.median(rel),
                             "rel_pct_max": max(rel), "signs": signs},
                            indent=2), encoding="utf-8")
    print(f"wrote {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
