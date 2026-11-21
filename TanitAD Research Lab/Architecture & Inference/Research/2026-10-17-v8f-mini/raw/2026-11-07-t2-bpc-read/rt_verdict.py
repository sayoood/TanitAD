"""The route term RT = BPC - MC, read by PREREG_T2_ATTRIB.md section "Outcomes" 2, as written:
RT-WITHIN-FLOOR if, on M1 crawl share, ADE6 and along-track MAE, RT is not separated at either seed, or |RT| is below that row's seed floor of MC
at both seeds; RT-MATTERS otherwise. The seed floor of MC per row is |X_s1 - X_s0| of MC against P10 (`seed_gap_beside[row].gap` of MC's pinned read).
usage: python -I rt_verdict.py <PAIR_T2.json of BPC - MC> <MC DOMINANCE_T2.json> <out.json>"""
import json
import sys

ROWS = ("M1 crawl share", "ADE6", "along-track MAE")
pair, mc, out = json.load(open(sys.argv[1], encoding="utf-8")), json.load(open(sys.argv[2], encoding="utf-8")), sys.argv[3]
assert pair["arm_a"] in ("BPC", {"name": "BPC"}) or "BPC" in json.dumps(pair["arm_a"]), pair["arm_a"]
assert "MC" in json.dumps(pair["arm_b"]), pair["arm_b"]
res, ok_all = {}, True
for r in ROWS:
    floor = mc["seed_gap_beside"][r]["gap"]
    d = {s: pair["by_seed"][s]["rows"][r] for s in ("s0", "s1")}
    not_sep = not d["s0"]["separated"] and not d["s1"]["separated"]
    below = abs(d["s0"]["delta"]) < floor and abs(d["s1"]["delta"]) < floor
    ok = not_sep or below
    ok_all &= ok
    res[r] = {"rt_s0": d["s0"], "rt_s1": d["s1"], "mc_seed_floor": floor, "not_separated_at_either_seed": not_sep, "abs_below_floor_both_seeds": below, "within": ok}
verdict = "RT-WITHIN-FLOOR" if ok_all else "RT-MATTERS"
json.dump({"verdict": verdict, "rule": __doc__.split("\n")[1:3], "rows": res, "pair": sys.argv[1], "mc_read": sys.argv[2]}, open(out, "w", encoding="utf-8"), indent=1)
print("verdict:", verdict)
for r, v in res.items():
    print(f"  {r}: s0 {v['rt_s0']['delta']:+.4f} [{v['rt_s0']['lo']:+.4f}, {v['rt_s0']['hi']:+.4f}] sep={v['rt_s0']['separated']} | s1 {v['rt_s1']['delta']:+.4f} [{v['rt_s1']['lo']:+.4f}, {v['rt_s1']['hi']:+.4f}] sep={v['rt_s1']['separated']} | MC floor {v['mc_seed_floor']:.4f} -> within={v['within']}")
