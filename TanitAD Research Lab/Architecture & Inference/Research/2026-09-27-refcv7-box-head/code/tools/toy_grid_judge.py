"""toy_grid_judge.py <tree with the addendum test> <out.json> <grid json> ... -- applies the addendum test's OWN checks
(``_memorised_failures`` on MAIN, ``_blinded_arm_failures`` on memory_zeros; imported from the tree's test module, never
re-derived) to every run of every ``toy_repro.py`` grid file, and tabulates platform x threads x seed x arm.
Exit 0 only when every MAIN run memorised and every memory_zeros run failed as required."""
import importlib.util
import json
import sys
from pathlib import Path

tree, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("tgbo_judge", str(tree / "stack" / "tests" / "test_g_box_overfit.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
rows, bad = [], 0
for gf in sys.argv[3:]:
    g = json.loads(Path(gf).read_text(encoding="utf-8"))
    plat = g["platform"]
    for r in g["runs"]:
        res = {"rows": r["rows"], "verdict": r["verdict"]}
        reasons = (M._memorised_failures(res) if r["arm"] == "main" else M._blinded_arm_failures(res))
        f = r["rows"][-1]
        ok = not reasons
        bad += not ok
        rows.append({"grid": Path(gf).name, "machine": plat["machine"], "torch": plat["torch"],
                     "threads": plat["threads"], "has_A17": g["has_A17"], "seed": r["seed"], "arm": r["arm"],
                     "step": f["step"], "ap2m": f["ap2m"], "prec": f["prec"], "rec": f["rec"],
                     "centre_p50_m": f.get("centre_p50_m"), "lr_factor": f.get("lr_factor"),
                     "verdict": r["verdict"], "check": "PASS" if ok else "FAIL", "reasons": reasons})
for x in rows:
    print(f"{x['machine']:8s} t{x['threads']:<2d} seed {x['seed']} {x['arm']:13s} A17={str(x['has_A17']):5s} "
          f"ap2m {x['ap2m']:.4f} rec {x['rec']:.4f} centre {x['centre_p50_m']:.3f} fac {x['lr_factor']} "
          f"-> {x['check']} {'; '.join(x['reasons'])}")
n_main = sum(x["arm"] == "main" for x in rows)
n_zero = sum(x["arm"] == "memory_zeros" for x in rows)
print(f"{len(rows)} runs ({n_main} main, {n_zero} memory_zeros); check FAILS: {bad}")
out.write_text(json.dumps({"runs": rows, "n_fail": bad}, indent=1), encoding="utf-8")
sys.exit(1 if bad else 0)
