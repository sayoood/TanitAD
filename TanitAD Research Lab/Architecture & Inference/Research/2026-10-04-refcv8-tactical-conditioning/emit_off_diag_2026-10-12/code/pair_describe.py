"""DESCRIPTIVE, NOT A VERDICT (SPEC_WPB_LADDER sec. 5 scores nothing before I-0): treatment arm vs base arm on the
SAME 1,112 windows with ladder_score's own window_metrics, paired episode-cluster bootstrap (shared seed-0 draws) and
METRICS orientation; plus the treatment's own intervention rows vs its base row. Writes nothing inside <W>.
usage: python pair_describe.py <W> <treatment> <base>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/nvidia/refcv8_ladder/code")
import numpy as np  # noqa: E402

import ladder_score as LS  # noqa: E402

w, tr, ba = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
out = {"what": "DESCRIPTIVE pre-I-0 read; not a ladder verdict", "treatment": tr, "base": ba, "seeds": {}}


def scored(arm, row, s, inv, cnt):
    z = LS.load_pass(w / "eval" / arm / f"{row}_s{s}.npz")
    return z, LS.score_pass(z, inv, cnt, None)


for s in LS.SEEDS:
    zb = LS.load_pass(w / "eval" / ba / f"base_s{s}.npz")
    zt = LS.load_pass(w / "eval" / tr / f"base_s{s}.npz")
    assert zb["_json"]["window_sha12_t_sha256"] == zt["_json"]["window_sha12_t_sha256"], "windows differ"
    eps, inv = LS.episodes(zb["win_sha12"])
    cnt = LS.counts_of(LS.make_draws(len(eps)), len(eps))
    rb, rt = LS.score_pass(zb, inv, cnt, None), LS.score_pass(zt, inv, cnt, None)
    rows = {}
    for m, (fam, better) in LS.METRICS.items():
        if m in rb and m in rt:
            p = LS.paired(rt[m], rb[m], better)
            rows[m] = {"family": fam, "base": rb[m]["pt"], "treat": rt[m]["pt"], "n": rt[m]["n"], **(p or {})}
    rows["lat3_majority_v9"] = {"base": rb.get("lat3_majority_v9", {}).get("pt"),
                                "treat": rt.get("lat3_majority_v9", {}).get("pt")}
    inter = {}
    for row in ("legal", "rc_off", "rc_shuf", "vmax_off", "vmax_shuf"):
        f = w / "eval" / tr / f"{row}_s{s}.npz"
        if not f.exists():
            continue
        _, rr = scored(tr, row, s, inv, cnt)
        inter[row] = {m: {**LS.paired(rr[m], rt[m], LS.METRICS[m][1]), "pt": rr[m]["pt"]}
                      for m in ("ade_all", "speed_mae_2_6s", "dir_correct_turn", "lat3_f1_v9", "ceil_violation")
                      if m in rr and m in rt and LS.paired(rr[m], rt[m], LS.METRICS[m][1])}
    out["seeds"][f"s{s}"] = {"pair": rows, "rows_vs_treat_base": inter}
print(json.dumps(out, default=float))
