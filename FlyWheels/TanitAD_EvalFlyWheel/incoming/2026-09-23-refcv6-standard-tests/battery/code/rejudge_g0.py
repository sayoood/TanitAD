"""Re-apply `reproduce_inrun_eval.judge` (the CURRENT file) to a banked G0 JSON: zero GPU.
usage: python rejudge_g0.py <g0.json>   (rewrites verdict, keeps the old one as verdict_as_run)"""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reproduce_inrun_eval as R  # noqa: E402
p = sys.argv[1]
rec = json.load(open(p, encoding="utf-8"))
by_seed = {int(k): v for k, v in rec["by_seed"].items()}
new = R.judge(rec["inrun_row"], by_seed, rec.get("mutation_m1"), rec)
rec.setdefault("verdict_as_run", rec.get("verdict"))
rec["verdict"] = new
rec["rejudged_by"] = "rejudge_g0.py (term_class token rule, SPEC §2 examples)"
json.dump(rec, open(p, "w", encoding="utf-8"), indent=1, default=str)
print(json.dumps({k: new[k] for k in ("G0", "reasons", "n_terms", "by_class_counts", "medians")}, indent=1))
