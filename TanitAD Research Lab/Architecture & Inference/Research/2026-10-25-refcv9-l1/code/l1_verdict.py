"""l1_verdict.py -- the MECHANICAL verdict of PREREG_L1.md section 7-8 from L1_RESULT.json (no fitting, no data access).

    python -X utf8 l1_verdict.py <result_dir>      -> <result_dir>/VERDICT.json + VERDICT_TABLES.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SEEDS = ("0", "1")


def sep_below(c):
    return c is not None and c.get("hi") is not None and c["hi"] < 0


def sep_above(c):
    return c is not None and c.get("lo") is not None and c["lo"] > 0


def fmt(c):
    if not c or c.get("delta") is None:
        return "n/a"
    s = " SEP" if (c.get("lo", 0) > 0 or c.get("hi", 0) < 0) else ""
    return f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]{s}"


def main() -> int:
    d = Path(sys.argv[1])
    R = json.loads((d / "L1_RESULT.json").read_text(encoding="utf-8"))
    S2 = R["surfaces"]["S2"]
    sd = S2["seeds"]
    floor = abs(S2["inference_seed_floor_os"]["delta"])
    K = {}
    if not R.get("dryrun_synthetic"):
        K["K1"] = all(v["PASS"] for k, v in R["controls_K1"].items() if k.startswith("S2"))
        K["K4_K5_eval"] = all(R["bank_controls"][f"s{s}"]["PASS"] for s in SEEDS)
        K["K6"] = bool(R["bank_controls"]["K6_ztac_seed_dependence"].get("PASS"))   # AMENDMENT 1: binding
    K["K2"] = all(S2["controls"][f"s{s}"]["K2_PASS"] for s in SEEDS)
    K["K3"] = all(S2["controls"][f"s{s}"]["K3_PASS"] for s in SEEDS)
    K["K5_train"] = bool(S2["controls"]["K5_train_join"]["PASS"])
    K["K7"] = bool(R.get("controls_K7", {}).get("PASS"))
    K["V_m"] = not any(sep_below(sd[s]["paired"]["H2m-H2s"]) for s in SEEDS)
    valid = all(K.values())
    P1 = all(sep_below(sd[s]["E1_minus_ha0_ext_pose"]["H2"]) for s in SEEDS)
    P2 = all(sep_below(sd[s]["paired"]["H2-H0"]) for s in SEEDS)
    P3 = all(sep_below(sd[s]["paired"]["H2-H2s"]) and abs(sd[s]["paired"]["H2-H2s"]["delta"]) > floor for s in SEEDS)
    P4 = all(abs(sd[s]["E1_minus_ha0_ext_pose"]["H2"]["delta"] - sd[s]["E1_minus_ha0_ext_pose"]["H2r"]["delta"])
             < abs(sd[s]["E1_minus_ha0_ext_pose"]["H2"]["delta"] - sd[s]["E1_minus_ha0_ext_pose"]["H0"]["delta"]) for s in SEEDS)
    worse_both = []
    for k in sd["0"]["four_families"]["H2-H0"]:
        cells = [sd[s]["four_families"]["H2-H0"].get(k) for s in SEEDS]
        lower_better = not (k.startswith("TAC_") or k == "LON_speed_within_0p5")
        if all(c is not None and (sep_above(c) if lower_better else sep_below(c)) for c in cells):
            worse_both.append(k)
    G = not worse_both
    H2g_P2 = all(sep_below(sd[s]["paired"]["H2g-H0"]) for s in SEEDS)
    H2n_P3 = all(sep_below(sd[s]["paired"]["H2n-H2s"]) and abs(sd[s]["paired"]["H2n-H2s"]["delta"]) > floor for s in SEEDS)
    if not valid:
        O = "O5 INVALID"
    elif P1 and P2 and P3 and P4 and G:
        O = "O1 L1 = (c4) with z_tac"
    elif not P3:
        O = "O2 the frozen latent adds nothing over the scalars at this head class and fit size"
    elif P3 and not P2:
        O = "O3 compose (H2g " + ("passes P2" if H2g_P2 else "fails P2") + ")"
    elif not G:
        O = "O4 guard failed: " + ",".join(worse_both)
    else:
        O = "UNCLASSIFIED (P1/P4 failed with P2, P3, G passing)"
    V = {"controls": K, "valid": valid, "bars": {"P1": P1, "P2": P2, "P3": P3, "P4": P4, "G": G}, "guard_worse_at_both_seeds": worse_both,
         "secondary": {"H2n_P3": H2n_P3, "H2g_P2": H2g_P2}, "inference_seed_floor_os_m": floor, "outcome": O,
         "dryrun_synthetic": R.get("dryrun_synthetic")}
    (d / "VERDICT.json").write_text(json.dumps(V, indent=1), encoding="utf-8")
    L = [f"# L1 verdict tables ({'DRY-RUN SYNTHETIC' if R.get('dryrun_synthetic') else R['tree']})", "",
         f"**Outcome: {O}**", "", f"controls {K}", "", f"bars {V['bars']}; secondary {V['secondary']}", "",
         f"inference-seed floor (OS s0 vs s1, S2 ADE): {fmt(S2['inference_seed_floor_os'])}", ""]
    for surf, SS in R["surfaces"].items():
        L += [f"## {surf}", "", "| arm | E1 s0 | E1 s1 |", "|---|---|---|"]
        for a in SS["seeds"]["0"]["E1_minus_ha0_ext_pose"]:
            L.append(f"| {a} | {fmt(SS['seeds']['0']['E1_minus_ha0_ext_pose'][a])} | {fmt(SS['seeds']['1']['E1_minus_ha0_ext_pose'][a])} |")
        L += ["", "| pair (ADE) | s0 | s1 |", "|---|---|---|"]
        for p in SS["seeds"]["0"]["paired"]:
            L.append(f"| {p} | {fmt(SS['seeds']['0']['paired'][p])} | {fmt(SS['seeds']['1']['paired'][p])} |")
        for p in SS["seeds"]["0"]["four_families"]:
            L += ["", f"four families {p}", "", "| metric | s0 | s1 |", "|---|---|---|"]
            for k in SS["seeds"]["0"]["four_families"][p]:
                L.append(f"| {k} | {fmt(SS['seeds']['0']['four_families'][p][k])} | {fmt(SS['seeds']['1']['four_families'][p][k])} |")
        L += ["", "fit: " + json.dumps({a: {k: v for k, v in f.items() if k in ('k', 'lambda', 'train_R2', 'oof_R2', 'wd', 'epochs', 'n', 'n_params', 'H2g_oof_gain')}
                                         for a, f in SS["fit"].items() if a != "H1"}, default=float), "",
              "eval R2 s0: " + json.dumps({a: [round(x, 3) for x in v] for a, v in SS["seeds"]["0"]["eval_R2"].items()}), ""]
    (d / "VERDICT_TABLES.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[l1-verdict] {O}  ZZL1VERDICTZZ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
