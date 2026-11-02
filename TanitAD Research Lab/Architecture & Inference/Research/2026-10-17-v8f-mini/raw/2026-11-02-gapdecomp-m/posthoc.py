#!/usr/bin/env python3
"""posthoc.py -- POST-HOC descriptive probes after the frozen verdict (NOT in PREREG_FROZEN.md; decides nothing; every number labelled POST-HOC in RESULTS.md).

    python posthoc.py --signed F:/TanitAD-scratch/gapdecomp_m/signed --out raw/POSTHOC.json

(1) the pooled low-speed view of M (v0 < 8 m/s = the 0.5-3 and 3-8 buckets together) for the five rows: share of Delta_M, interval, per arm and seed;
(2) D2'' = the trigger windows (L + S) AND the 3-8 m/s windows of M both set to 0, and D2''' = every window with v0 < 8 set to 0 (ESTIMATED decomposition arithmetic);
(3) mechanism descriptors of the 3-8 m/s windows, DAG-OACC and P10: distribution of the per-window mean |kappa| (median, p90, share carried by the top 10 % of windows), the share of
    windows whose plan has a step below 0.5 m/s, and the GT speed mass near the 5.2 m/s threshold of M1 / L2.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
spec = importlib.util.spec_from_file_location("gapdecomp_m", f"{HERE}/gapdecomp_m.py")
GM = importlib.util.module_from_spec(spec)
sys.modules["gapdecomp_m"] = GM
spec.loader.exec_module(GM)
GD = GM.GD


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--signed", required=True)
    ap.add_argument("--out", required=True)
    a_ = ap.parse_args(argv)
    Z = {(arm, s): np.load(f"{GM.LAND}/raw/arrays/{arm}_s{s}.npz") for arm in GM.ARMS for s in GM.SEEDS}
    z0 = Z[("CTRL", 0)]
    eid, v0, g = z0["eid"], z0["v0"], z0["g"]
    cls, masks = GD.classify(v0, g)
    mmask, trig = masks["M"], masks["L"] | masks["S"]
    sp = np.full(len(v0), -1)
    sp[mmask] = GM.speed_index(v0[mmask])
    s38 = mmask & (sp == 1)
    low = mmask & (v0 < 8.0)
    out = {"label": "POST-HOC descriptive; not in PREREG_FROZEN.md; decides nothing", "pooled_low_speed": {}, "D2_double_zero": {}, "mechanism_3to8": {}}
    # (1) pooled v0 < 8 view of M
    lab = np.where(mmask, np.where(low, 0, 1), -1)
    for arm in GM.ARMS:
        out["pooled_low_speed"][arm] = {}
        for s in GM.SEEDS:
            out["pooled_low_speed"][arm][f"s{s}"] = {}
            for row in GM.VROWS:
                i = GD.LABELS.index(row)
                r, _ = GM.decompose_groups(Z[(arm, s)][f"a_{i}"], Z[(arm, s)][f"b_{i}"], eid, lab, ["v0<8", "v0>=8"])
                gg = r["groups"]["v0<8"]
                out["pooled_low_speed"][arm][f"s{s}"][row] = {"M_delta": r["M"]["delta"], "M_separated": r["M"]["separated"], "n": gg["n"], "n_episodes": gg["n_episodes"], "w": gg["w"],
                                                              "share": gg["share"], "share_lo": gg["share_lo"], "share_hi": gg["share_hi"], "enrichment": gg["enrichment"]}
    # (2) trigger windows and 3-8 m/s windows both zeroed
    for arm in GM.ARMS:
        out["D2_double_zero"][arm] = {}
        for s in GM.SEEDS:
            zr = GM.zeroed_rows(GD, Z[(arm, s)], trig | s38, GD.LABELS)
            worse = [l for l in GD.LABELS if zr[l]["separated_worse"]]
            out["D2_double_zero"][arm][f"s{s}"] = {"worse_rows": worse, "D2": not worse, "rows_worse_detail": {l: {k: zr[l][k] for k in ("delta_zeroed", "lo", "hi")} for l in worse}}
    out["D2_all_windows_v0_lt_8_zero"] = {}
    for arm in GM.ARMS:
        out["D2_all_windows_v0_lt_8_zero"][arm] = {}
        for s in GM.SEEDS:
            zr = GM.zeroed_rows(GD, Z[(arm, s)], v0 < 8.0, GD.LABELS)
            worse = [l for l in GD.LABELS if zr[l]["separated_worse"]]
            out["D2_all_windows_v0_lt_8_zero"][arm][f"s{s}"] = {"n_windows_zeroed": int((v0 < 8.0).sum()), "worse_rows": worse, "D2": not worse,
                                                              "rows_worse_detail": {l: {k: zr[l][k] for k in ("delta_zeroed", "lo", "hi")} for l in worse}}
    # (3) mechanism descriptors of the 3-8 m/s windows
    import torch
    sys.path.insert(0, f"{GM.TREE}/taniteval")
    os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", os.path.join(GM.TREE, "stack").replace("/", os.sep))
    from taniteval import four_families as FF
    sys.path.insert(0, HERE)
    import extract_signed as ES
    import json as _j
    def q(x, m):
        v = np.asarray(x, np.float64)[m]
        v = v[np.isfinite(v)]
        top = np.sort(v)[::-1][: max(1, int(0.1 * len(v)))]
        return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)), "share_of_sum_in_top10pct_windows": float(top.sum() / v.sum())}
    Qg = FF._seq_geometry(torch.as_tensor(g.astype(np.float64)), GM.DT)
    gt_sp = Qg["speed"].numpy()
    out["mechanism_3to8"]["gt_step_speed_share_in_[5.2,6.2)"] = float(((gt_sp[s38] >= 5.2) & (gt_sp[s38] < 6.2)).mean())
    out["mechanism_3to8"]["gt_step_speed_share_below_5.2"] = float((gt_sp[s38] < 5.2).mean())
    for arm in GM.ARMS:
        for s in GM.SEEDS:
            meta = _j.load(open(f"{GM.LAND}/raw/arrays/{arm}_s{s}.npz.meta.json", encoding="utf-8"))
            d = ES.load_dir(meta["dump_dir"])
            sg = np.load(f"{a_.signed}/signed_{arm}_s{s}.npz")
            P = FF._seq_geometry(torch.as_tensor(d["cl"].astype(np.float64)), GM.DT)
            F = FF._seq_geometry(torch.as_tensor(d["fl"].astype(np.float64)), GM.DT)
            rec = {}
            for who, G_, key in (("x", P, "kabs_x"), ("p10", F, "kabs_p10")):
                stall = (~G_["valid"]).any(1).numpy()
                rec[who] = {"plan_abs_curv_in_3to8": q(sg[key], s38), "plan_abs_curv_in_rest_of_M": q(sg[key], mmask & ~s38), "share_windows_with_a_plan_step_below_0.5mps_in_3to8": float(stall[s38].mean()),
                            "share_windows_with_a_plan_step_below_0.5mps_in_rest_of_M": float(stall[mmask & ~s38].mean()),
                            "plan_min_step_speed_median_in_3to8": float(np.median(G_["speed"].numpy()[s38].min(1)))}
            out["mechanism_3to8"][f"{arm}_s{s}"] = rec
    os.makedirs(os.path.dirname(os.path.abspath(a_.out)), exist_ok=True)
    with open(a_.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("POSTHOC OK ->", a_.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
