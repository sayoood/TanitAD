"""POST-HOC DIAGNOSTIC read (not a ladder verdict): any two eval passes on the SAME windows, scored with ladder_score's
own window_metrics + paired episode-cluster bootstrap (shared seed-0 draws, METRICS orientation), both sampler seeds.
usage: python diag_describe.py <treat_dir> <treat_row> <base_dir> <base_row> [label]
   e.g. diag_describe.py E:/lgdiag/eval/V-R8 emit_off E:/lgdiag/thor/V0 base "V-R8 emit_off vs V0"
Prints JSON: per seed, per metric {family, base, treat, n, delta, ci, sep_better, sep_worse, ...}."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder_score as LS  # noqa: E402

td, trow, bd, brow = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
label = sys.argv[5] if len(sys.argv) > 5 else f"{td.name}:{trow} vs {bd.name}:{brow}"
out = {"what": "POST-HOC diagnostic, not a ladder verdict", "label": label, "seeds": {}}
for s in LS.SEEDS:
    zb = LS.load_pass(bd / f"{brow}_s{s}.npz")
    zt = LS.load_pass(td / f"{trow}_s{s}.npz")
    if zb["_json"]["window_sha12_t_sha256"] != zt["_json"]["window_sha12_t_sha256"]:
        raise SystemExit(f"windows differ on seed {s}")
    eps, inv = LS.episodes(zb["win_sha12"])
    cnt = LS.counts_of(LS.make_draws(len(eps)), len(eps))
    rb, rt = LS.score_pass(zb, inv, cnt, None), LS.score_pass(zt, inv, cnt, None)
    rows = {}
    for m, (fam, better) in LS.METRICS.items():
        if m in rb and m in rt:
            p = LS.paired(rt[m], rb[m], better)
            rows[m] = {"family": fam, "base": rb[m]["pt"], "treat": rt[m]["pt"], "n": rt[m]["n"], **(p or {})}
    out["seeds"][f"s{s}"] = {"pair": rows}
out["treatment"], out["base"] = f"{td.name}:{trow}", f"{bd.name}:{brow}"
print(json.dumps(out, default=float))
